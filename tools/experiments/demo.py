#!/usr/bin/env python3
"""Prepare and evaluate a saved tensor-add/no-op experiment in one command."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

from evaluate import evaluate
from experiment import create, digest, replay, run_process, write_json


def demo(build_dir, toolchain_bin, output, acl_library=None, device=0,
         compile_only=False, timeout=120, elementwise_recipe=False):
    if not math.isfinite(timeout) or timeout <= 0 or type(device) is not int or device < 0:
        raise ValueError("invalid timeout or device")
    build_dir = Path(build_dir).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "error", "optimization_claim": False}
    try:
        examples = Path(__file__).with_name("examples")
        source = output / "add.mlir"
        source.write_bytes((examples / "add.mlir").read_bytes())
        recipe = output / "recipe.mlir"
        recipe.write_bytes((examples / ("elementwise.mlir" if elementwise_recipe else "noop.mlir")).read_bytes())
        specification = json.loads((examples / "add.json").read_text())
        specification["device"] = device
        write_json(output / "evaluation.json", specification)
        preparation = output / "prepare"
        preparation.mkdir()
        command = [str(build_dir / "bin/bishengir-opt"), str(source),
                   "--hfusion-auto-vectorize", "--outline-vector-function", "--canonicalize",
                   "--hivm-v2-pre-vectorize-pipeline", "--emit-bytecode", "-o", "checkpoint.bc"]
        report["preparation"] = run_process(command, preparation, dict(os.environ), timeout)
        report["preparation"]["command"] = command
        if report["preparation"]["status"] != "success":
            report["status"] = "preparation_failed"
            return report
        candidate_checkpoint = preparation / "checkpoint.bc"
        if elementwise_recipe:
            research_source = output / "add-hivm.mlir"
            research_source.write_bytes((examples / "add-hivm.mlir").read_bytes())
            research_preparation = output / "prepare-candidate"
            research_preparation.mkdir()
            command = [str(build_dir / "bin/bishengir-opt"), str(research_source),
                       "--hivm-v2-pre-vectorize-pipeline", "--emit-bytecode", "-o", "checkpoint.bc"]
            report["candidate_preparation"] = run_process(
                command, research_preparation, dict(os.environ), timeout)
            report["candidate_preparation"]["command"] = command
            if report["candidate_preparation"]["status"] != "success":
                report["status"] = "preparation_failed"
                return report
            candidate_checkpoint = research_preparation / "checkpoint.bc"
        dependencies = list((build_dir / "lib").glob("*.bc"))
        if not dependencies:
            raise ValueError("no template bitcode found in build/lib; build templates first")
        for label in ("baseline", "candidate"):
            create(output / (label + "-bundle"),
                   candidate_checkpoint if label == "candidate" else preparation / "checkpoint.bc",
                   recipe if label == "candidate" else None,
                   build_dir / "bin/bishengir-compile", Path(__file__).resolve().parents[2],
                   "Ascend950PR_9579", "__transform_main", adapter="driver-binary",
                   toolchain_bin=toolchain_bin, dependency=dependencies,
                   replace_vectorization=elementwise_recipe and label == "candidate")
        if compile_only:
            report["compilations"] = {
                label: replay(output / (label + "-bundle"), output / (label + "-compile"), timeout=timeout)
                for label in ("baseline", "candidate")}
            report["status"] = "compiled_not_executed" if all(
                r["status"] == "success" for r in report["compilations"].values()) else "compile_failed"
        else:
            if not acl_library:
                raise ValueError("--acl-library is required for hardware evaluation")
            report["evaluation"] = evaluate(output / "baseline-bundle", output / "candidate-bundle",
                output / "evaluation.json", output / "evaluation", acl_library, timeout=timeout)
            report["status"] = report["evaluation"]["status"]
    except (ValueError, OSError) as error:
        report["diagnostic"] = str(error)
    finally:
        report["artifacts"] = {str(p.relative_to(output)): digest(p.read_bytes())
                               for p in sorted(output.rglob("*")) if p.is_file()}
        write_json(output / "demo.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build_a5"))
    parser.add_argument("--toolchain-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acl-library", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--elementwise-recipe", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    try:
        result = demo(**vars(parser.parse_args()))
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] in ("success", "compiled_not_executed") else 1
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
