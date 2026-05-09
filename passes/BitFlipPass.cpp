#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using namespace llvm;

static cl::opt<long long> FITargetInst(
    "fi-target-inst",
    cl::desc("Eligible instruction index to inject. -1 means choose randomly."),
    cl::init(-1), cl::Hidden);

static cl::opt<int> FITargetBit(
    "fi-bit", cl::desc("Bit index to flip. -1 means choose randomly."),
    cl::init(-1), cl::Hidden);

static cl::opt<unsigned> FISeed(
    "fi-seed", cl::desc("Random seed used when target instruction or bit is -1."),
    cl::init(1), cl::Hidden);

static cl::opt<std::string> FIKernelFilter(
    "fi-kernel",
    cl::desc("Only inject into functions whose names contain this string."),
    cl::init(""), cl::Hidden);

static cl::opt<bool> FIProtectHighRisk(
    "fi-protect-high-risk",
    cl::desc("Skip high-risk instructions as a simple selective protection policy."),
    cl::init(false), cl::Hidden);

static cl::opt<bool> FIListOnly(
    "fi-list-only",
    cl::desc("List eligible instructions without mutating IR."),
    cl::init(false), cl::Hidden);

namespace {

struct Candidate {
  Instruction *Inst;
  unsigned BitWidth;
  bool HighRisk;
  std::string ValueType;
};

static bool isSupportedType(Type *Ty, unsigned &BitWidth,
                            std::string &ValueType) {
  if (Ty->isIntegerTy()) {
    BitWidth = Ty->getIntegerBitWidth();
    ValueType = "int";
    return BitWidth > 0 && BitWidth <= 64;
  }

  if (Ty->isFloatTy()) {
    BitWidth = 32;
    ValueType = "float";
    return true;
  }

  if (Ty->isDoubleTy()) {
    BitWidth = 64;
    ValueType = "double";
    return true;
  }

  return false;
}

static bool isHighRiskInstruction(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::FDiv:
  case Instruction::SDiv:
  case Instruction::UDiv:
  case Instruction::SRem:
  case Instruction::URem:
    return true;
  case Instruction::FMul:
  case Instruction::FAdd:
  case Instruction::FSub:
    return true;
  default:
    return isa<CallBase>(I);
  }
}

static bool isInjectableInstruction(Instruction &I, unsigned &BitWidth,
                                    bool &HighRisk,
                                    std::string &ValueType) {
  if (I.getType()->isVoidTy() || I.isTerminator() || I.isEHPad())
    return false;

  if (isa<PHINode>(I) || isa<AllocaInst>(I) || isa<LandingPadInst>(I))
    return false;

  if (!I.getNextNode())
    return false;

  if (!isSupportedType(I.getType(), BitWidth, ValueType))
    return false;

  HighRisk = isHighRiskInstruction(I);
  if (FIProtectHighRisk && HighRisk)
    return false;

  return true;
}

static std::vector<Candidate> collectCandidates(Module &M) {
  std::vector<Candidate> Candidates;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    if (!FIKernelFilter.empty() && !F.getName().contains(FIKernelFilter))
      continue;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        unsigned BitWidth = 0;
        bool HighRisk = false;
        std::string ValueType;
        if (isInjectableInstruction(I, BitWidth, HighRisk, ValueType))
          Candidates.push_back({&I, BitWidth, HighRisk, ValueType});
      }
    }
  }

  return Candidates;
}

static Value *insertBitFlip(Instruction &I, unsigned Bit) {
  IRBuilder<> Builder(I.getNextNode());
  Type *Ty = I.getType();

  if (Ty->isIntegerTy()) {
    unsigned Width = Ty->getIntegerBitWidth();
    APInt Mask(Width, 0);
    Mask.setBit(Bit);
    Value *MaskValue = ConstantInt::get(Ty, Mask);
    return Builder.CreateXor(&I, MaskValue, Twine(I.getName()) + ".fi");
  }

  unsigned Width = Ty->isFloatTy() ? 32 : 64;
  IntegerType *IntTy = Builder.getIntNTy(Width);
  APInt Mask(Width, 0);
  Mask.setBit(Bit);

  Value *AsInt =
      Builder.CreateBitCast(&I, IntTy, Twine(I.getName()) + ".fi.asint");
  Value *Flipped =
      Builder.CreateXor(AsInt, ConstantInt::get(IntTy, Mask),
                        Twine(I.getName()) + ".fi.xor");
  return Builder.CreateBitCast(Flipped, Ty, Twine(I.getName()) + ".fi");
}

static bool useCanBeReplaced(Instruction &Source, Use &U,
                             ArrayRef<Value *> NewValues) {
  User *Usr = U.getUser();
  if (std::find(NewValues.begin(), NewValues.end(), static_cast<Value *>(Usr)) !=
      NewValues.end())
    return false;

  auto *UserInst = dyn_cast<Instruction>(Usr);
  if (!UserInst)
    return true;

  if (auto *PN = dyn_cast<PHINode>(UserInst)) {
    unsigned OperandNo = U.getOperandNo();
    if (OperandNo >= PN->getNumIncomingValues())
      return false;
    return PN->getIncomingBlock(OperandNo) == Source.getParent();
  }

  if (UserInst->getParent() == Source.getParent())
    return Source.comesBefore(UserInst);

  return true;
}

static StringRef instructionCategory(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
    return "float_arith";
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::SDiv:
  case Instruction::UDiv:
  case Instruction::SRem:
  case Instruction::URem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    return "int_arith";
  case Instruction::Load:
  case Instruction::Store:
  case Instruction::GetElementPtr:
    return "memory";
  case Instruction::ICmp:
  case Instruction::FCmp:
    return "compare";
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
    return "conversion";
  case Instruction::Call:
  case Instruction::Invoke:
  case Instruction::CallBr:
    return "call";
  case Instruction::Select:
    return "select";
  default:
    return "other";
  }
}

static std::string logicalLayerName(StringRef FunctionName) {
  if (FunctionName.contains("gemm"))
    return "gemm";
  if (FunctionName.contains("softmax"))
    return "softmax";
  if (FunctionName.contains("layernorm"))
    return "layernorm";
  if (FunctionName.contains("attention"))
    return "attention";
  return FunctionName.str();
}

class BitFlipModulePass : public PassInfoMixin<BitFlipModulePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    std::vector<Candidate> Candidates = collectCandidates(M);
    errs() << "FI_CANDIDATES count=" << Candidates.size()
           << " protect_high_risk=" << (FIProtectHighRisk ? "1" : "0")
           << " kernel_filter=" << FIKernelFilter << "\n";

    if (Candidates.empty() || FIListOnly)
      return PreservedAnalyses::all();

    std::mt19937 Rng(FISeed);
    size_t TargetIndex = 0;
    if (FITargetInst >= 0) {
      TargetIndex = static_cast<size_t>(FITargetInst);
      if (TargetIndex >= Candidates.size()) {
        errs() << "FI_ERROR target instruction " << TargetIndex
               << " is outside candidate range 0.." << Candidates.size() - 1
               << "\n";
        return PreservedAnalyses::all();
      }
    } else {
      std::uniform_int_distribution<size_t> PickInst(0, Candidates.size() - 1);
      TargetIndex = PickInst(Rng);
    }

    Candidate &Selected = Candidates[TargetIndex];
    unsigned Bit = 0;
    if (FITargetBit >= 0) {
      Bit = static_cast<unsigned>(FITargetBit) % Selected.BitWidth;
    } else {
      std::uniform_int_distribution<unsigned> PickBit(0,
                                                      Selected.BitWidth - 1);
      Bit = PickBit(Rng);
    }

    Instruction &I = *Selected.Inst;
    Value *Faulty = insertBitFlip(I, Bit);

    SmallVector<Value *, 4> NewValues;
    if (auto *FaultyInst = dyn_cast<Instruction>(Faulty)) {
      for (Instruction &Inserted : make_range(I.getIterator(),
                                              FaultyInst->getIterator())) {
        if (&Inserted != &I)
          NewValues.push_back(&Inserted);
      }
      NewValues.push_back(FaultyInst);
    } else {
      NewValues.push_back(Faulty);
    }

    I.replaceUsesWithIf(Faulty, [&](Use &U) {
      return useCanBeReplaced(I, U, NewValues);
    });

    errs() << "FI_SELECTED index=" << TargetIndex << " bit=" << Bit
           << " bitwidth=" << Selected.BitWidth
           << " high_risk=" << (Selected.HighRisk ? "1" : "0")
           << " function=" << I.getFunction()->getName()
           << " layer=" << logicalLayerName(I.getFunction()->getName())
           << " opcode=" << I.getOpcodeName()
           << " inst_type=" << instructionCategory(I)
           << " value_type=" << Selected.ValueType << "\n";

    return PreservedAnalyses::none();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "BitFlipPass", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "bitflip-fi") {
                    MPM.addPass(BitFlipModulePass());
                    return true;
                  }
                  return false;
                });
          }};
}
