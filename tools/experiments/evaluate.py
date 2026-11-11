#!/usr/bin/env python3
"""Evaluate a saved baseline/candidate pair against a seeded f32 add reference."""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import struct
import sys
import time

from experiment import digest, encoded, replay, run_process, write_json


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    fields = {"schema_version", "reference", "abi", "shape", "dtype", "seeds",
              "atol", "rtol", "entry", "blocks", "device"}
    if not isinstance(spec, dict) or set(spec) != fields:
        raise ValueError("unsupported evaluation specification fields")
    if (spec["schema_version"] != 1 or spec["reference"] != "add"
            or spec["abi"] != "three-pointers" or spec["dtype"] != "f32"):
        raise ValueError("only the explicit three-pointer f32 add ABI is supported")
    if (not isinstance(spec["shape"], list) or not spec["shape"]
            or any(type(n) is not int or n <= 0 for n in spec["shape"])
            or math.prod(spec["shape"]) > 10000000):
        raise ValueError("invalid shape")
    if (not isinstance(spec["seeds"], list) or not spec["seeds"]
            or any(type(s) is not int or s < 0 for s in spec["seeds"])
            or len(spec["seeds"]) > 100):
        raise ValueError("invalid input seeds")
    for key in ("atol", "rtol"):
        value = spec[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid tolerance")
    if (type(spec["device"]) is not int or spec["device"] < 0
            or type(spec["blocks"]) is not int or not 0 < spec["blocks"] < 2**32
            or not isinstance(spec["entry"], str) or not spec["entry"]):
        raise ValueError("invalid launch specification")
    return spec


def floats(data):
    return [x[0] for x in struct.iter_unpack("<f", data)]


def pack(values):
    return b"".join(struct.pack("<f", x) for x in values)


def make_cases(spec, output):
    cases = []
    count = math.prod(spec["shape"])
    for index, seed in enumerate(spec["seeds"]):
        generator = random.Random(seed)
        arrays = [pack(generator.uniform(-1, 1) for _ in range(count)) for _ in range(2)]
        # Each input is rounded to f32 before the independent scalar reference.
        expected = pack(a + b for a, b in zip(*(floats(data) for data in arrays)))
        paths = [output / f"input-{index}-{i}.bin" for i in range(2)]
        for path, data in zip(paths, arrays):
            path.write_bytes(data)
        reference = output / f"reference-{index}.bin"
        reference.write_bytes(expected)
        cases.append({"seed": seed, "inputs": [str(p) for p in paths],
                      "reference": str(reference)})
    return cases


def compare(expected, actual, atol, rtol):
    if len(expected) != len(actual):
        return {"passed": False, "diagnostic": "output byte count mismatch",
                "expected_bytes": len(expected), "actual_bytes": len(actual)}
    mismatches = []
    total = 0
    for i, (reference, value) in enumerate(zip(floats(expected), floats(actual))):
        good = math.isfinite(value) and abs(reference - value) <= atol + rtol * abs(reference)
        if not good:
            total += 1
            if len(mismatches) < 16:
                mismatches.append({"index": i, "expected": reference,
                                   "actual": value if math.isfinite(value) else str(value)})
    return {"passed": total == 0, "mismatch_count": total, "mismatches": mismatches}


def evaluate(baseline, candidate, specification, output, acl_library,
             timeout=120, runner=None):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    output = Path(output).resolve()
    for bundle in (baseline, candidate):
        bundle = Path(bundle).resolve()
        if output == bundle or bundle in output.parents:
            raise ValueError("evaluation output must be outside saved bundles")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "error", "correctness": {},
              "performance_eligible": False, "compilations": {}}
    start = time.monotonic()
    try:
        spec = load_spec(specification)
        write_json(output / "evaluation.json", spec)
        report["specification_sha256"] = digest(encoded(spec))
        report["host"] = {"python": sys.version, "platform": platform.platform()}
        library = Path(acl_library).resolve(strict=True)
        report["acl_library"] = {"path": str(library), "sha256": digest(library.read_bytes())}
        runner = Path(runner or Path(__file__).with_name("acl_runner.py")).resolve(strict=True)
        report["runner_sha256"] = digest(runner.read_bytes())
        cases = make_cases(spec, output)
        for label, bundle in (("baseline", baseline), ("candidate", candidate)):
            compilation = replay(bundle, output / (label + "-compile"), timeout=timeout)
            report["compilations"][label] = compilation
            if compilation["status"] != "success" or compilation.get("candidate") != "candidate.o":
                report["status"] = "compile_failed"
                return report
        for label in ("baseline", "candidate"):
            run_dir = output / (label + "-execute")
            run_dir.mkdir()
            request = {"acl_library": str(library), "device": spec["device"],
                       "entry": spec["entry"], "blocks": spec["blocks"],
                       "bytes": math.prod(spec["shape"]) * 4,
                       "binary": str(output / (label + "-compile/candidate.o")),
                       "cases": [dict(c, output=str(run_dir / f"output-{i}.bin"))
                                 for i, c in enumerate(cases)]}
            write_json(run_dir / "request.json", request)
            execution = run_process([sys.executable, str(runner), "--request", "request.json"],
                                    run_dir, dict(os.environ), timeout)
            verdict = {"execution": execution, "cases": [], "passed": False}
            report["correctness"][label] = verdict
            if execution["status"] != "success":
                continue
            for case in request["cases"]:
                path = Path(case["output"])
                comparison = compare(Path(case["reference"]).read_bytes(), path.read_bytes(),
                                     spec["atol"], spec["rtol"]) if path.is_file() else {
                                         "passed": False, "diagnostic": "missing output"}
                verdict["cases"].append(dict(comparison, seed=case["seed"]))
            verdict["passed"] = all(c["passed"] for c in verdict["cases"])
        passed = all(v["passed"] for v in report["correctness"].values())
        report.update(status="correct" if passed else "incorrect_or_execution_failed",
                      performance_eligible=passed)
    except (ValueError, KeyError, TypeError, OSError) as error:
        report["diagnostic"] = str(error)
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        report["artifacts"] = {str(p.relative_to(output)): digest(p.read_bytes())
                               for p in sorted(output.rglob("*")) if p.is_file()}
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--specification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--acl-library", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    args = vars(parser.parse_args())
    try:
        report = evaluate(**args)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "correct" else 1
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
