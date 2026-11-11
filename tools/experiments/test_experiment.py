#!/usr/bin/env python3
"""Coordinator tests; optionally exercise the current checkout's compiler."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("experiment", Path(__file__).with_name("experiment.py"))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
REAL_COMPILER = os.environ.get("BISHENGIR_EXPERIMENT_COMPILER")


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="experiment test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "source.bc"
        self.checkpoint.write_bytes(b"checkpoint")
        self.recipe = self.root / "source.mlir"
        self.recipe.write_text("recipe")
        self.compiler = self.root / "compiler"
        self.compiler.write_text(
            "#!/usr/bin/python3\nfrom pathlib import Path\n"
            "Path('candidate.mlir').write_bytes(Path('checkpoint.bc').read_bytes())\n"
        )
        self.compiler.chmod(0o755)
        self.bundle = self.root / "bundle"
        self.count = 0

    def create(self):
        with patch.object(experiment, "source_identity", return_value={"revision": "test"}):
            return experiment.create(self.bundle, self.checkpoint, self.recipe,
                                     self.compiler, self.root, "Ascend950PR_9579",
                                     "__transform_main")

    def run_replay(self, **kwargs):
        self.count += 1
        self.output = self.root / ("run " + str(self.count))
        result = experiment.replay(self.bundle, self.output, **kwargs)
        self.assertEqual(result, json.loads((self.output / "result.json").read_text()))
        return result

    def reseal(self, change):
        path = self.bundle / "manifest.json"
        m = json.loads(path.read_text())
        change(m)
        m.pop("experiment_id")
        m["experiment_id"] = experiment.digest(experiment.encoded(m))
        path.write_text(json.dumps(m))

    def test_reproducibility_and_immutable_inputs(self):
        self.create()
        before = {p.name: p.read_bytes() for p in self.bundle.iterdir()}
        self.checkpoint.write_text("changed original")
        self.recipe.unlink()
        first = self.run_replay()
        second = self.run_replay()
        self.assertEqual(first["status"], "success")
        self.assertEqual(first["experiment_id"], second["experiment_id"])
        self.assertEqual(first["artifacts"], second["artifacts"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.bundle.iterdir()})
        with self.assertRaises(FileExistsError):
            experiment.replay(self.bundle, self.output)
        with self.assertRaises(ValueError):
            experiment.replay(self.bundle, self.bundle / "run")

    def test_crash(self):
        self.compiler.write_text("#!/usr/bin/python3\nimport os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")
        self.create()
        result = self.run_replay()
        self.assertEqual(result["status"], "crash")
        self.assertEqual(result["returncode"], -9)
        self.assertNotIn("candidate.mlir", result["artifacts"])

    def test_timeout_kills_descendants(self):
        self.compiler.write_text(
            "#!/usr/bin/python3\nimport subprocess, time\n"
            "subprocess.Popen(['/usr/bin/python3', '-c', "
            "\"import time; from pathlib import Path; Path('started').touch(); time.sleep(0.7); Path('leaked').touch()\"])\n"
            "time.sleep(30)\n"
        )
        self.create()
        result = self.run_replay(timeout=0.2)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue((self.output / "started").exists())
        time.sleep(0.8)
        self.assertFalse((self.output / "leaked").exists())

    def test_compiler_failure_and_diagnostics(self):
        self.compiler.write_text("#!/usr/bin/python3\nimport sys\nprint('bad recipe', file=sys.stderr)\nsys.exit(2)\n")
        self.create()
        result = self.run_replay()
        self.assertEqual(result["status"], "failed")
        self.assertIn("bad recipe", (self.output / "stderr.log").read_text())

    def test_incompatible_schema_and_boundary(self):
        for key, value in (("schema_version", 999), ("boundary", "hivm.post-vectorize")):
            with self.subTest(key=key):
                if self.bundle.exists():
                    shutil.rmtree(self.bundle)
                self.create()
                self.reseal(lambda m: m.update({key: value}))
                result = self.run_replay()
                self.assertEqual(result["status"], "incompatible")
                self.assertFalse((self.output / "stdout.log").exists())

    def test_changed_compiler(self):
        self.create()
        self.compiler.write_text("#!/bin/sh\nexit 0\n")
        result = self.run_replay()
        self.assertEqual(result["status"], "incompatible")
        self.assertIn("compiler digest", result["diagnostic"])

    def test_tampered_inputs(self):
        self.create()
        (self.bundle / "recipe.mlir").write_text("changed")
        self.assertEqual(self.run_replay()["status"], "incompatible")

    def test_changed_options_without_new_identity(self):
        self.create()
        path = self.bundle / "manifest.json"
        m = json.loads(path.read_text())
        m["pipeline"]["enable_preload"] = True
        path.write_text(json.dumps(m))
        self.assertEqual(self.run_replay()["status"], "incompatible")

    def test_unknown_options(self):
        self.create()
        self.reseal(lambda m: m["pipeline"].update({"unknown": True}))
        self.assertEqual(self.run_replay()["status"], "incompatible")

    def test_changed_dependency(self):
        dependency = self.root / "runtime.so"
        dependency.write_bytes(b"runtime")
        with patch.object(experiment, "source_identity", return_value={}):
            experiment.create(self.bundle, self.checkpoint, None, self.compiler,
                              self.root, "Ascend950PR_9579", "__transform_main",
                              dependency=[dependency])
        dependency.write_bytes(b"new runtime")
        result = self.run_replay()
        self.assertEqual(result["status"], "incompatible")
        self.assertIn("dependency", result["diagnostic"])

    def test_invalid_environment(self):
        self.create()
        self.reseal(lambda m: m["environment"].update({"LD_PRELOAD": "unknown.so"}))
        self.assertEqual(self.run_replay()["status"], "incompatible")

    def test_cli_create_and_replay(self):
        repo = self.root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test",
                        "-c", "user.email=test@example.invalid", "commit",
                        "--allow-empty", "-qm", "fixture"], check=True)
        cli = Path(experiment.__file__)
        subprocess.run([sys.executable, str(cli), "create", "--repo", str(repo),
                        "--bundle", str(self.bundle), "--checkpoint", str(self.checkpoint),
                        "--compiler", str(self.compiler)], check=True, capture_output=True)
        subprocess.run([sys.executable, str(cli), "replay", "--bundle", str(self.bundle),
                        "--output", str(self.root / "cli-run")], check=True, capture_output=True)
        result = json.loads((self.root / "cli-run/result.json").read_text())
        self.assertEqual(result["status"], "success")

    def test_driver_adapter(self):
        self.compiler.write_text(
            "#!/usr/bin/python3\nimport sys\nfrom pathlib import Path\n"
            "assert '--hivm-replay-pre-vectorize' in sys.argv\n"
            "assert '--enable-lir-compile=false' in sys.argv\n"
            "assert '--enable-triton-kernel-compile=true' in sys.argv\n"
            "Path('intermediates').mkdir()\n"
            "Path('intermediates/candidate.ll').write_text('LLVM output')\n"
            "Path('intermediates/payload.mlir').write_text('IR')\n"
        )
        with patch.object(experiment, "source_identity", return_value={}):
            experiment.create(self.bundle, self.checkpoint, self.recipe, self.compiler,
                              self.root, "Ascend950PR_9579", "__transform_main",
                              adapter="driver-llvm", backend_options="-mllvm -example=false",
                              triton_kernel=True)
        result = self.run_replay()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["candidate"], "intermediates/candidate.ll")
        self.assertIn("intermediates/payload.mlir", result["artifacts"])
        self.assertGreater(result["compilation_seconds"], 0)
        self.assertGreater(result["preflight_seconds"], 0)
        self.assertIn("--append-bisheng-options=-mllvm -example=false", result["command"])

    def test_opt_rejects_triton_kernel_mode(self):
        with self.assertRaisesRegex(ValueError, "requires a driver adapter"):
            experiment.create(self.bundle, self.checkpoint, self.recipe, self.compiler,
                              self.root, "Ascend950PR_9579", "__transform_main",
                              triton_kernel=True)

    def test_trace_intervals(self):
        trace = self.root / "trace.jsonl"
        event = {"run_id": "r", "tid": 1, "lane": "AIV", "phase": "compile"}
        trace.write_text("\n".join(json.dumps(dict(event, event=kind, time_ns=t))
                                   for kind, t in (("begin", 10), ("end", 1000000010)))
                         + '\n{"partial":')
        self.assertEqual(experiment.trace_timings(trace),
                         [{"lane": "AIV", "phase": "compile", "seconds": 1.0}])

    def test_missing_output(self):
        self.compiler.write_text("#!/usr/bin/python3\n")
        self.create()
        self.assertEqual(self.run_replay()["status"], "error")

    def test_malformed_manifest(self):
        self.create()
        (self.bundle / "manifest.json").write_text("not JSON")
        self.assertEqual(self.run_replay()["status"], "incompatible")
        self.assertFalse((self.output / "stdout.log").exists())

    def test_invalid_timeout(self):
        self.create()
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.run_replay(timeout=timeout)

    @unittest.skipUnless(REAL_COMPILER, "set BISHENGIR_EXPERIMENT_COMPILER for integration")
    def test_real_compiler_replay(self):
        self.compiler = Path(shutil.which(REAL_COMPILER) or REAL_COMPILER).resolve()
        repo = Path(__file__).resolve().parents[2]
        fixture = repo / "bishengir/test/Dialect/HIVM/TransformOps/external-recipe.mlir"
        sections = {}
        for section in fixture.read_text().split("//--- ")[1:]:
            name, content = section.split("\n", 1)
            sections[name.strip()] = content
        payload = self.root / "tensor.mlir"
        payload.write_text(sections["tensor.mlir"])
        self.recipe.write_text(sections["recipe.mlir"])
        subprocess.run([str(self.compiler), str(payload), "--hivm-v2-pre-vectorize-pipeline",
                        "--emit-bytecode", "-o", str(self.checkpoint)], check=True)
        self.create()
        first = self.run_replay(timeout=60)
        second = self.run_replay(timeout=60)
        self.assertEqual(first["status"], "success", first)
        self.assertEqual(first["artifacts"]["candidate.mlir"],
                         second["artifacts"]["candidate.mlir"])
        direct = self.root / "direct.mlir"
        # Stable relative recipe paths also exercise directories containing spaces.
        subprocess.run([str(self.compiler), str(payload),
                        "--hivm-v2-pipeline=recipe=recipe.mlir", "-o", str(direct)],
                       cwd=self.bundle, check=True)
        self.assertEqual(experiment.digest(direct.read_bytes()),
                         first["artifacts"]["candidate.mlir"])
        driver = self.compiler.with_name("bishengir-compile")
        driver_bundle = self.root / "driver-bundle"
        with patch.object(experiment, "source_identity", return_value={}):
            experiment.create(driver_bundle, self.checkpoint, self.recipe, driver,
                              self.root, "Ascend950PR_9579", "__transform_main",
                              adapter="driver-llvm")
        driver_result = experiment.replay(driver_bundle, self.root / "driver-run")
        self.assertEqual(driver_result["status"], "success", driver_result)
        self.assertIn("define", (self.root / "driver-run/intermediates/candidate.ll").read_text())
        # A well-formed manifest cannot override the compiler's checkpoint marker.
        m = json.loads((self.bundle / "manifest.json").read_text())
        m["pipeline"]["target"] = "Ascend960"
        m.pop("experiment_id")
        m["experiment_id"] = experiment.digest(experiment.encoded(m))
        (self.bundle / "manifest.json").write_text(json.dumps(m))
        mismatch = self.run_replay()
        self.assertEqual(mismatch["status"], "failed")
        self.assertIn("incompatible", (self.output / "stderr.log").read_text())


if __name__ == "__main__":
    unittest.main()
