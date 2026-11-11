#!/usr/bin/env python3
"""Compare original/recipe vectorization of one saved Triton fused checkpoint.

The installed Triton-Ascend launcher owns the hidden-argument ABI. The worker
uses its launcher and metadata for both replay binaries, never the three-pointer
ACL demo runner. All device work runs in a disposable, timeout-bound process.
"""
import argparse
import difflib
import json
import math
import os
from pathlib import Path
import re
import sys

from experiment import create, digest, replay, run_process, write_json
from evaluate import summarize_samples


BLOCK_SIZE = 64
EXPECTED_ARGUMENTS = ["ptr addrspace(1)"] * 5 + ["i32"] * 3


def entry_arguments(llvm_ir, name):
    # Reject an unexpected ABI before letting a launcher interpret the binary.
    line = next((s for s in llvm_ir.splitlines()
                 if s.startswith("define ") and f"@{name}(" in s), None)
    if line is None:
        raise ValueError("missing LLVM kernel entry")
    arguments = line.split(f"@{name}(", 1)[1].rsplit(")", 1)[0]
    return [re.sub(r"\s+%[A-Za-z0-9_.]+$", "", arg.strip())
            for arg in arguments.split(",")]


def validate_options(args):
    if any(n < BLOCK_SIZE or n % BLOCK_SIZE or n > 10000000 for n in args.elements):
        raise ValueError("element counts must be positive multiples of 64, at most 10000000")
    if len(set(args.elements)) != len(args.elements):
        raise ValueError("element counts must be distinct")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if args.device < 0 or args.warmups < 0 or args.rounds < 2 or args.iterations < 1:
        raise ValueError("invalid device or measurement configuration")


def worker(args):
    import torch
    import torch_npu
    import triton
    import triton.backends.ascend.compiler as ascend_compiler
    from triton.runtime import driver
    from examples.fused_elementwise import fused_elementwise

    output = args.output
    build = args.build_dir
    compiler = build / "bin/bishengir-compile"
    examples = Path(__file__).with_name("examples")
    report = {"status": "error", "performance_eligible": False,
              "runner_sha256": digest(Path(__file__).read_bytes()),
              "expression": "a * b + a", "block_size": BLOCK_SIZE,
              "elements": args.elements, "seeds": [7, 19, 42], "correctness": {},
              "versions": {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                           "triton": triton.__version__}}
    try:
        torch.npu.set_device(args.device)
        report["device"] = {"index": args.device, "name": torch.npu.get_device_name(args.device)}
        report["frontend_adapter"] = {"path": ascend_compiler.__file__,
            "sha256": digest(Path(ascend_compiler.__file__).read_bytes())}
        (output / "kernel.py").write_bytes((examples / "fused_elementwise.py").read_bytes())
        a = torch.ones(args.elements[0], device="npu", dtype=torch.float32)
        b, result = torch.ones_like(a), torch.empty_like(a)
        # Explicitly select this checkout even if Triton bundles another driver.
        # This adapter targets the installed Triton-Ascend 3.6 compiler API and
        # runs only in this isolated, single-threaded worker.
        previous = ascend_compiler._get_npucompiler_path
        ascend_compiler._get_npucompiler_path = lambda: (str(compiler), dict(os.environ))
        try:
            kernel = fused_elementwise.warmup(a, b, result, BLOCK_SIZE,
                grid=(args.elements[0] // BLOCK_SIZE,), enable_fp_fusion=False,
                disable_fma=True, enable_auto_vectorize_v2=False,
                enable_auto_bind_sub_block=False,
                enable_preload=False, multibuffer=False)
        finally:
            ascend_compiler._get_npucompiler_path = previous
        report["metadata"] = dict(kernel.metadata._asdict())
        # Serialize compiler metadata without silently dropping backend fields.
        write_json(output / "metadata.json", json.loads(json.dumps(report.pop("metadata"), default=str)))
        if kernel.metadata.mix_mode != "aiv":
            raise ValueError("only standalone AIV kernels are supported")
        for kind in ("ttir", "ttadapter"):
            (output / ("kernel." + kind)).write_text(kernel.asm[kind])
        report["frontend_binary_sha256"] = digest(kernel.asm["npubin"])
        kernel[(args.elements[0] // BLOCK_SIZE, 1, 1)](a, b, result, BLOCK_SIZE)
        torch.npu.synchronize()
        if not torch.equal(result, a * b + a):
            raise ValueError("frontend control failed")

        capture = output / "capture"
        capture.mkdir()
        command = [str(compiler), str(output / "kernel.ttadapter"),
            "--target=Ascend950PR_9579", "--enable-triton-kernel-compile=true", "--disable-ffts",
            "--enable-auto-blockify-loop",
            "--enable-hfusion-compile=true", "--disable-hfusion-vectorize=true",
            "--use-pipeline-version=2", "--hivm-pre-vectorize-checkpoint=checkpoint.bc",
            "--hivm-replace-vectorization",
            "--hivm-pre-vectorize-recipe=" + str(examples / "original-vectorizer.mlir"),
            "--enable-preload=false", "--enable-auto-multi-buffer=false",
            "--enable-lir-compile=false", "-o", "capture"]
        report["capture"] = dict(run_process(command, capture, dict(os.environ), args.timeout),
                                 command=command)
        if report["capture"]["status"] != "success":
            raise ValueError("checkpoint capture failed; see capture/stderr.log")
        checkpoint = capture / "checkpoint.bc"
        report["checkpoint_sha256"] = digest(checkpoint.read_bytes())
        dependencies = list((build / "lib").glob("*.bc"))
        if not dependencies:
            raise ValueError("missing template bitcode")
        loaded, compilations = {}, {}
        for label, recipe in (("baseline", examples / "original-vectorizer.mlir"),
                              ("candidate", args.recipe)):
            manifest = create(output / (label + "-bundle"), checkpoint, recipe, compiler,
                Path(__file__).resolve().parents[2], "Ascend950PR_9579", "__transform_main",
                adapter="driver-binary", toolchain_bin=args.toolchain_bin,
                dependency=dependencies, replace_vectorization=True, triton_kernel=True,
                backend_options="-mllvm -cce-simt-fpmath-combine=false")
            if manifest["inputs"]["checkpoint.bc"] != report["checkpoint_sha256"]:
                raise ValueError("baseline and candidate checkpoint mismatch")
            run_dir = output / (label + "-compile")
            compilation = replay(output / (label + "-bundle"), run_dir, timeout=args.timeout)
            compilations[label] = compilation
            report["compilations"] = compilations
            if compilation["status"] != "success":
                raise ValueError(label + " compilation failed")
            arguments = entry_arguments((run_dir / "intermediates/candidate.ll").read_text(),
                                        kernel.metadata.kernel_name)
            if arguments != EXPECTED_ARGUMENTS:
                raise ValueError("replay ABI does not match the supported Triton launcher: " + str(arguments))
            loaded[label] = driver.active.utils.load_binary(kernel.metadata.kernel_name,
                (run_dir / "candidate.o").read_bytes(), kernel.metadata.shared,
                args.device, kernel.metadata.mix_mode)

        for name in ("candidate.ll", "module.hivm.opt.mlir"):
            texts = [(output / (label + "-compile/intermediates") / name).read_text().splitlines(True)
                     for label in ("baseline", "candidate")]
            (output / (name + ".diff")).write_text("".join(difflib.unified_diff(
                *texts, fromfile="original-vectorizer/" + name, tofile="recipe/" + name)))

        def launch(label, a, b, out):
            kernel.module, kernel.function = loaded[label][:2]
            kernel[(a.numel() // BLOCK_SIZE, 1, 1)](a, b, out, BLOCK_SIZE)

        buffers = {}
        for n in args.elements:
            cases = []
            for seed in report["seeds"]:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                lhs, rhs = (torch.randn(n, generator=generator, dtype=torch.float32) for _ in range(2))
                expected = lhs * rhs + lhs  # Independent PyTorch CPU reference, f32 steps.
                a, b = lhs.to("npu"), rhs.to("npu")
                outs = {label: torch.full_like(a, float("nan")) for label in loaded}
                case = {"seed": seed, "variants": {}}
                for label, out in outs.items():
                    launch(label, a, b, out)
                    actual = out.cpu()
                    close = torch.isclose(actual, expected, atol=1e-6, rtol=1e-5) & torch.isfinite(actual)
                    case["variants"][label] = {"passed": bool(close.all()),
                        "mismatch_count": int((~close).sum()),
                        "max_abs_error": float((actual - expected).abs().max()) if torch.isfinite(actual).all() else None}
                    torch.save(actual, output / f"{n}-{seed}-{label}.pt")
                torch.save({"lhs": lhs, "rhs": rhs, "reference": expected}, output / f"{n}-{seed}-reference.pt")
                cases.append(case)
                buffers[n] = (a, b, outs)
            report["correctness"][str(n)] = cases
        report["performance_eligible"] = all(v["passed"] for cases in report["correctness"].values()
            for case in cases for v in case["variants"].values())
        if not report["performance_eligible"]:
            report["status"] = "incorrect"
            return report
        report["benchmarks"] = {}
        for n, (a, b, outs) in buffers.items():
            for _ in range(args.warmups):
                for label in loaded:
                    launch(label, a, b, outs[label])
            torch.npu.synchronize()
            samples = []
            for i in range(args.rounds):
                for label in (("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")):
                    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    for _ in range(args.iterations):
                        launch(label, a, b, outs[label])
                    end.record()
                    end.synchronize()
                    samples.append({"round": i, "variant": label,
                                    "milliseconds": start.elapsed_time(end) / args.iterations})
                    write_json(output / f"samples-{n}.json", samples)
            report["benchmarks"][str(n)] = dict(summarize_samples(samples, args.rounds),
                warmups=args.warmups, rounds=args.rounds, iterations=args.iterations,
                clock="torch.npu.Event", unit="milliseconds per launch including scheduling gaps")
        report["status"] = "success"
        return report
    except Exception as error:
        report["diagnostic"] = str(error)
        raise
    finally:
        report["artifacts"] = {str(p.relative_to(output)): digest(p.read_bytes())
            for p in sorted(output.rglob("*")) if p.is_file() and p.name not in ("report.json", "stdout.log", "stderr.log")}
        write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-dir", type=Path, default=Path("build_a5"))
    parser.add_argument("--toolchain-bin", required=True, type=Path)
    parser.add_argument("--recipe", type=Path, default=Path(__file__).with_name("examples") / "elementwise.mlir")
    parser.add_argument("--elements", type=int, nargs="+", default=[4096, 65536])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    validate_options(args)
    for key in ("output", "build_dir", "toolchain_bin", "recipe"):
        setattr(args, key, getattr(args, key).resolve())
    if args.worker:
        return 0 if worker(args)["status"] == "success" else 1
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
        "--output", str(args.output), "--build-dir", str(args.build_dir),
        "--toolchain-bin", str(args.toolchain_bin), "--recipe", str(args.recipe),
        "--device", str(args.device), "--warmups", str(args.warmups), "--rounds", str(args.rounds),
        "--iterations", str(args.iterations), "--timeout", str(args.timeout),
        "--elements", *(str(n) for n in args.elements)]
    environment = dict(os.environ, BISHENG_INSTALL_PATH=str(args.toolchain_bin))
    # A private cache guarantees the frontend control uses this checkout's
    # compiler even when the installed Triton cache predates local changes.
    environment["TRITON_CACHE_DIR"] = str(args.output / "frontend-cache")
    environment["PATH"] = str(args.toolchain_bin) + os.pathsep + environment.get("PATH", "")
    execution = run_process(command, args.output, environment, args.timeout)
    write_json(args.output / "execution.json", dict(execution, command=command))
    print(json.dumps({"execution": execution, "report": str(args.output / "report.json")}))
    return 0 if execution["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
