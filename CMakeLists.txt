cmake_minimum_required(VERSION 3.16)
project(FaultInjectionLLVM LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_llvm_pass_plugin(BitFlipPass
  passes/BitFlipPass.cpp
  PARTIAL_SOURCES_INTENDED
)

target_compile_features(BitFlipPass PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(BitFlipPass PRIVATE /EHsc)
else()
  target_compile_options(BitFlipPass PRIVATE -fno-rtti)
endif()
