#!/usr/bin/env python3
import argparse
import csv
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


KERNELS = ("gemm", "softmax", "layernorm", "attention")
OUTCOMES = ("Masked", "Crash", "Timeout", "SDC")
SELECTED_RE = re.compile(r"FI_SELECTED (?P<fields>.*)")


def exe_name(name: str) -> str:
    return f"{name}.exe" if platform.system() == "Windows" else name


def run_cmd(cmd, *, timeout=None, cwd=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def require_tool(path_or_name: str) -> str:
    if Path(path_or_name).exists():
        return path_or_name
    found = shutil.which(path_or_name)
    if found:
        return found
    raise SystemExit(f"required tool not found: {path_or_name}")


def guess_plugin(build_dir: Path, explicit: str | None) -> Path:
    if explicit:
        plugin = Path(explicit)
        if plugin.exists():
            return plugin
        raise SystemExit(f"pass plugin not found: {plugin}")

    suffixes = [".dll", ".so", ".dylib"]
    names = [f"BitFlipPass{suffix}" for suffix in suffixes]
    candidates = []
    for name in names:
        candidates.extend(
            [
                build_dir / name,
                build_dir / "Debug" / name,
                build_dir / "Release" / name,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise SystemExit(
        "BitFlipPass plugin was not found. Build it first with: "
        "cmake -S . -B build -DLLVM_DIR=<llvm-cmake-dir>; cmake --build build"
    )


def parse_values(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for token in handle.read().split():
            values.append(float(token))
    return values


def outputs_match(golden: list[float], faulty: list[float], atol: float, rtol: float) -> bool:
    if len(golden) != len(faulty):
        return False
    for expected, actual in zip(golden, faulty):
        if math.isnan(actual) or math.isinf(actual):
            return False
        if abs(expected - actual) > atol + rtol * abs(expected):
            return False
    return True


def parse_selected(stderr: str) -> dict[str, str]:
    match = SELECTED_RE.search(stderr)
    if not match:
        return {}

    selected: dict[str, str] = {}
    for token in match.group("fields").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        selected[key] = value
    return selected


def bit_region(value_type: str, bit_text: str, bitwidth_text: str) -> str:
    try:
        bit = int(bit_text)
        bitwidth = int(bitwidth_text)
    except (TypeError, ValueError):
        return "unknown"

    if value_type == "float" and bitwidth == 32:
        if bit == 31:
            return "sign"
        if 23 <= bit <= 30:
            return "exponent"
        return "mantissa"

    if value_type == "double" and bitwidth == 64:
        if bit == 63:
            return "sign"
        if 52 <= bit <= 62:
            return "exponent"
        return "mantissa"

    if value_type == "int":
        if bit == bitwidth - 1:
            return "msb"
        if bit >= max(0, bitwidth * 3 // 4):
            return "high"
        if bit >= max(0, bitwidth // 4):
            return "middle"
        return "low"

    return "unknown"


def compile_ir(args, source: Path, ir_path: Path):
    cmd = [
        args.clangxx,
        "-std=c++17",
        "-O1",
        "-S",
        "-emit-llvm",
        str(source),
        "-o",
        str(ir_path),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise SystemExit(f"failed to emit LLVM IR:\n{result.stderr}")


def compile_exe(args, input_path: Path, exe_path: Path):
    cmd = [
        args.clangxx,
        "-std=c++17",
        "-O2",
        str(input_path),
        "-o",
        str(exe_path),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise SystemExit(f"failed to compile {input_path}:\n{result.stderr}")


def run_kernel(exe_path: Path, kernel: str, output_path: Path, timeout: float):
    return run_cmd([str(exe_path), kernel, str(output_path)], timeout=timeout)


def inject_ir(args, plugin: Path, base_ir: Path, faulty_ir: Path, seed: int, kernel: str):
    cmd = [
        args.opt,
        f"-load-pass-plugin={plugin}",
        "-passes=bitflip-fi",
        f"-fi-seed={seed}",
        f"-fi-kernel={kernel}",
        str(base_ir),
        "-S",
        "-o",
        str(faulty_ir),
    ]
    if args.protect_high_risk:
        cmd.insert(3, "-fi-protect-high-risk")

    result = run_cmd(cmd)
    selected = parse_selected(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"opt failed:\n{result.stderr}")
    return selected, result.stderr


def classify_trial(args, plugin: Path, base_ir: Path, kernel: str, golden: list[float], trial_dir: Path, trial: int):
    seed = args.seed + trial
    faulty_ir = trial_dir / f"{kernel}_{trial:04d}.ll"
    faulty_exe = trial_dir / exe_name(f"{kernel}_{trial:04d}")
    faulty_out = trial_dir / f"{kernel}_{trial:04d}.out"

    selected, _ = inject_ir(args, plugin, base_ir, faulty_ir, seed, kernel)
    compile_exe(args, faulty_ir, faulty_exe)

    try:
        result = run_kernel(faulty_exe, kernel, faulty_out, args.timeout)
    except subprocess.TimeoutExpired:
        return "Timeout", selected, "execution timed out"

    if result.returncode != 0:
        return "Crash", selected, result.stderr.strip()

    try:
        faulty = parse_values(faulty_out)
    except Exception as exc:
        return "Crash", selected, f"failed to parse output: {exc}"

    if outputs_match(golden, faulty, args.atol, args.rtol):
        return "Masked", selected, ""

    return "SDC", selected, ""


def outcome_counts(rows: list[dict]) -> dict[str, int]:
    counts = {label: 0 for label in OUTCOMES}
    for row in rows:
        counts[row["classification"]] += 1
    return counts


def summarize_rows(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in rows:
        key = tuple(row.get(field) or "unknown" for field in group_fields)
        grouped.setdefault(key, []).append(row)

    summary: list[dict] = []
    for key, group in sorted(grouped.items()):
        counts = outcome_counts(group)
        total = len(group)
        out = {field: value for field, value in zip(group_fields, key)}
        out["trials"] = total
        for label in OUTCOMES:
            out[label.lower()] = counts[label]
            out[f"{label.lower()}_rate"] = counts[label] / total if total else 0.0
        summary.append(out)
    return summary


def write_csv(rows: list[dict], path: Path, fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], path: Path):
    counts = outcome_counts(rows)

    with path.open("w", encoding="utf-8") as handle:
        total = len(rows)
        handle.write("Overall\n")
        for label in OUTCOMES:
            pct = 100.0 * counts[label] / total if total else 0.0
            handle.write(f"{label}: {counts[label]} ({pct:.2f}%)\n")

        handle.write("\nSDC rate by kernel\n")
        for row in summarize_rows(rows, ("kernel",)):
            handle.write(
                f"{row['kernel']}: {row['sdc']}/{row['trials']} "
                f"({100.0 * row['sdc_rate']:.2f}%)\n"
            )

        handle.write("\nSDC rate by instruction type\n")
        for row in summarize_rows(rows, ("inst_type",)):
            handle.write(
                f"{row['inst_type']}: {row['sdc']}/{row['trials']} "
                f"({100.0 * row['sdc_rate']:.2f}%)\n"
            )

        handle.write("\nSDC rate by bit region\n")
        for row in summarize_rows(rows, ("bit_region",)):
            handle.write(
                f"{row['bit_region']}: {row['sdc']}/{row['trials']} "
                f"({100.0 * row['sdc_rate']:.2f}%)\n"
            )

        handle.write("\nSDC rate by layer\n")
        for row in summarize_rows(rows, ("layer",)):
            handle.write(
                f"{row['layer']}: {row['sdc']}/{row['trials']} "
                f"({100.0 * row['sdc_rate']:.2f}%)\n"
            )


def write_grouped_summaries(rows: list[dict], results_dir: Path):
    metrics = ["trials"]
    for label in OUTCOMES:
        name = label.lower()
        metrics.extend([name, f"{name}_rate"])

    summaries = {
        "summary_by_kernel.csv": ("kernel",),
        "summary_by_instruction_type.csv": ("inst_type",),
        "summary_by_opcode.csv": ("opcode",),
        "summary_by_bit.csv": ("value_type", "bit"),
        "summary_by_bit_region.csv": ("value_type", "bit_region"),
        "summary_by_layer.csv": ("layer",),
    }

    for filename, group_fields in summaries.items():
        summary_rows = summarize_rows(rows, group_fields)
        write_csv(summary_rows, results_dir / filename, [*group_fields, *metrics])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLVM IR bit-flip fault injection campaign.")
    parser.add_argument("--kernel", choices=KERNELS + ("all",), default="all")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--clangxx", default="clang++")
    parser.add_argument("--opt", default="opt")
    parser.add_argument("--pass-plugin")
    parser.add_argument("--protect-high-risk", action="store_true")
    args = parser.parse_args()

    args.clangxx = require_tool(args.clangxx)
    args.opt = require_tool(args.opt)

    root = Path(__file__).resolve().parents[1]
    build_dir = (root / args.build_dir).resolve()
    results_dir = (root / args.results_dir).resolve()
    work_dir = build_dir / "campaign"
    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    plugin = guess_plugin(build_dir, args.pass_plugin)
    source = root / "benchmarks" / "kernels.cpp"
    base_ir = work_dir / "kernels.ll"
    golden_exe = work_dir / exe_name("kernels_golden")

    compile_ir(args, source, base_ir)
    compile_exe(args, source, golden_exe)

    kernels = KERNELS if args.kernel == "all" else (args.kernel,)
    rows: list[dict] = []

    for kernel in kernels:
        kernel_dir = work_dir / kernel
        kernel_dir.mkdir(parents=True, exist_ok=True)
        golden_out = kernel_dir / "golden.out"
        result = run_kernel(golden_exe, kernel, golden_out, args.timeout)
        if result.returncode != 0:
            raise SystemExit(f"golden run failed for {kernel}:\n{result.stderr}")
        golden = parse_values(golden_out)

        for trial in range(args.trials):
            classification, selected, message = classify_trial(
                args, plugin, base_ir, kernel, golden, kernel_dir, trial
            )
            row = {
                "kernel": kernel,
                "trial": trial,
                "seed": args.seed + trial,
                "classification": classification,
                "message": message,
                **selected,
            }
            row["bit_region"] = bit_region(
                row.get("value_type", ""),
                row.get("bit", ""),
                row.get("bitwidth", ""),
            )
            rows.append(row)
            print(
                f"{kernel} trial={trial:04d} class={classification} "
                f"inst={selected.get('index', '?')} bit={selected.get('bit', '?')} "
                f"type={selected.get('inst_type', '?')} layer={selected.get('layer', '?')}"
            )

    csv_path = results_dir / "campaign.csv"
    fields = [
        "kernel",
        "trial",
        "seed",
        "classification",
        "index",
        "bit",
        "bitwidth",
        "bit_region",
        "high_risk",
        "function",
        "layer",
        "opcode",
        "inst_type",
        "value_type",
        "message",
    ]
    write_csv(rows, csv_path, fields)

    summary_path = results_dir / "summary.txt"
    write_summary(rows, summary_path)
    write_grouped_summaries(rows, results_dir)
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
