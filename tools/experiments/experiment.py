#!/usr/bin/env python3
"""Content-addressed experiment manifests and isolated HIVM checkpoint replay."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SCHEMA = 1
BOUNDARY = "hivm.pre-vectorize"
# Capture only compiler-related environment, never the whole user environment.
ENV_KEYS = (
    "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "TOOLCHAIN_HOME",
    "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH",
    "ASCEND_HOME_PATH",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def compiler_identity(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("compiler must be an executable file")
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def source_identity(repo):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], stderr=subprocess.PIPE
        ).decode().strip()
    return {
        "revision": git("rev-parse", "HEAD"),
        "submodules": git("submodule", "status", "--recursive"),
        "tracked_diff_sha256": digest(git("diff", "HEAD", "--binary").encode()),
        "status": git("status", "--porcelain"),
    }


def create(bundle, checkpoint, recipe, compiler, repo, target, entry,
           preload=False, multi_buffer=False, dependency=()):
    # Materialize inputs before publishing the manifest. No source paths are
    # needed for replay and subsequent edits to originals cannot change a run.
    compiler = compiler_identity(compiler)
    if not re.fullmatch(r"[A-Za-z0-9_]+", target):
        raise ValueError("invalid target")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry):
        raise ValueError("invalid recipe entry")
    payload = Path(checkpoint).read_bytes()
    library = Path(recipe).read_bytes() if recipe else None
    source = source_identity(repo)
    environment = {key: os.environ[key] for key in ENV_KEYS if key in os.environ}
    environment.update({"LC_ALL": "C", "LANG": "C", "TZ": "UTC"})
    manifest = {
        "schema_version": SCHEMA, "boundary": BOUNDARY,
        "compiler": compiler, "source": source, "environment": environment,
        "dependencies": {str(Path(p).resolve(strict=True)): digest(Path(p).read_bytes())
                         for p in dependency},
        "pipeline": {"target": target, "entry": entry,
                     "enable_preload": preload, "enable_multi_buffer": multi_buffer},
        "inputs": {"checkpoint.bc": digest(payload)},
    }
    if library is not None:
        manifest["inputs"]["recipe.mlir"] = digest(library)
    manifest["experiment_id"] = digest(encoded(manifest))
    bundle = Path(bundle)
    bundle.mkdir(parents=True, exist_ok=False)
    (bundle / "checkpoint.bc").write_bytes(payload)
    if library is not None:
        (bundle / "recipe.mlir").write_bytes(library)
    write_json(bundle / "manifest.json", manifest)
    return manifest


def validate(bundle, compiler):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    identity = manifest.pop("experiment_id", None)
    if identity != digest(encoded(manifest)):
        raise ValueError("manifest integrity mismatch")
    manifest["experiment_id"] = identity
    if manifest.get("schema_version") != SCHEMA or manifest.get("boundary") != BOUNDARY:
        raise ValueError("incompatible manifest schema or checkpoint boundary")
    if set(manifest) != {"schema_version", "boundary", "compiler", "source",
                         "environment", "pipeline", "inputs", "experiment_id", "dependencies"}:
        raise ValueError("unsupported manifest fields")
    if not isinstance(manifest["compiler"], dict):
        raise ValueError("invalid compiler identity")
    dependencies = manifest["dependencies"]
    if not isinstance(dependencies, dict):
        raise ValueError("invalid dependency identities")
    for path, expected in dependencies.items():
        if not Path(path).is_absolute() or digest(Path(path).read_bytes()) != expected:
            raise ValueError("incompatible dependency: " + path)
    actual = compiler_identity(compiler or manifest["compiler"]["path"])
    if actual["sha256"] != manifest["compiler"]["sha256"]:
        raise ValueError("incompatible compiler digest")
    inputs = manifest["inputs"]
    if not isinstance(inputs, dict) or set(inputs) not in (
            {"checkpoint.bc"}, {"checkpoint.bc", "recipe.mlir"}):
        raise ValueError("unsupported manifest inputs")
    for name, expected in inputs.items():
        if digest((bundle / name).read_bytes()) != expected:
            raise ValueError("input digest mismatch: " + name)
    pipeline = manifest["pipeline"]
    if not isinstance(pipeline, dict) or set(pipeline) != {
            "target", "entry", "enable_preload", "enable_multi_buffer"}:
        raise ValueError("incompatible pipeline options")
    if not isinstance(pipeline["target"], str) or not re.fullmatch(
            r"[A-Za-z0-9_]+", pipeline["target"]):
        raise ValueError("invalid target")
    if not isinstance(pipeline["entry"], str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", pipeline["entry"]):
        raise ValueError("invalid recipe entry")
    if any(type(pipeline[k]) is not bool for k in ("enable_preload", "enable_multi_buffer")):
        raise ValueError("pipeline switches must be boolean")
    env = manifest["environment"]
    if not isinstance(env, dict) or any(
            k not in (*ENV_KEYS, "LC_ALL", "LANG", "TZ") or not isinstance(v, str)
            or "\0" in v for k, v in env.items()):
        raise ValueError("unsupported compiler environment")
    return manifest, actual


def replay(bundle, output, compiler=None, timeout=60):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    bundle = Path(bundle).resolve()
    output = Path(output).resolve()
    if output == bundle or bundle in output.parents:
        raise ValueError("run directory must be outside the immutable bundle")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": SCHEMA, "status": "incompatible", "artifacts": {}}
    start = time.monotonic()
    try:
        manifest, actual = validate(bundle, compiler)
        # Read back and check the copied bytes as well: a concurrent modification
        # of a bundle must not silently change the candidate being compiled.
        for name, expected in manifest["inputs"].items():
            data = (bundle / name).read_bytes()
            if digest(data) != expected:
                raise ValueError("input changed during replay: " + name)
            (output / name).write_bytes(data)
        write_json(output / "manifest.json", manifest)
        p = manifest["pipeline"]
        options = ["target=" + p["target"],
                   "enable-preload=" + str(p["enable_preload"]).lower(),
                   "enable-multi-buffer=" + str(p["enable_multi_buffer"]).lower()]
        if "recipe.mlir" in manifest["inputs"]:
            options += ["recipe=recipe.mlir", "recipe-entry=" + p["entry"]]
        command = [actual["path"], "checkpoint.bc",
                   "--hivm-v2-replay-pre-vectorize=" + " ".join(options),
                   "-o", "candidate.mlir"]
        result.update(experiment_id=manifest["experiment_id"], command=command,
                      compiler=actual, status="error")
        with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=output, env=manifest["environment"],
                                       stdin=subprocess.DEVNULL, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
                result["status"] = "success" if code == 0 else "crash" if code < 0 else "failed"
            except subprocess.TimeoutExpired:
                result["status"] = "timeout"
            finally:
                # Also terminate descendants retaining files after the leader exits.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            result["returncode"] = process.returncode
        candidate = output / "candidate.mlir"
        if result["status"] == "success" and not candidate.is_file():
            result.update(status="error", diagnostic="compiler produced no candidate")
        for name in ("stdout.log", "stderr.log", "candidate.mlir"):
            path = output / name
            if path.is_file():
                result["artifacts"][name] = digest(path.read_bytes())
    except (ValueError, KeyError, TypeError, OSError) as error:
        result["diagnostic"] = str(error)
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        result["timeout_seconds"] = timeout
        write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create_parser = commands.add_parser("create")
    create_parser.add_argument("--bundle", required=True, type=Path)
    create_parser.add_argument("--checkpoint", required=True, type=Path)
    create_parser.add_argument("--recipe", type=Path)
    create_parser.add_argument("--dependency", action="append", type=Path, default=[],
                               help="Runtime library or toolchain file to fingerprint; repeatable")
    create_parser.add_argument("--compiler", required=True, type=Path)
    create_parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    create_parser.add_argument("--target", default="Ascend950PR_9579")
    create_parser.add_argument("--entry", default="__transform_main")
    create_parser.add_argument("--preload", action="store_true")
    create_parser.add_argument("--multi-buffer", action="store_true")
    run_parser = commands.add_parser("replay")
    run_parser.add_argument("--bundle", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    run_parser.add_argument("--compiler", type=Path)
    run_parser.add_argument("--timeout", type=float, default=60)
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        result = create(**args) if command == "create" else replay(**args)
        print(json.dumps(result, sort_keys=True))
        return 0 if command == "create" or result["status"] == "success" else 1
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
