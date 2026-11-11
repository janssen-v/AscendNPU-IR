#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = self.root / "spec.json"
        self.spec.write_text(json.dumps({"schema_version": 1, "reference": "add",
            "abi": "three-pointers", "shape": [16], "dtype": "f32", "seeds": [7, 19],
            "atol": 1e-6, "rtol": 1e-5, "entry": "add", "blocks": 1, "device": 0}))
        self.library = self.root / "libacl.so"
        self.library.write_bytes(b"fixture-library")
        self.runner = self.root / "runner.py"
        self.runner.write_text('''import json, struct
from pathlib import Path
r = json.loads(Path('request.json').read_text())
for c in r['cases']:
 a, b = [list(struct.iter_unpack('<f', Path(p).read_bytes())) for p in c['inputs']]
 Path(c['output']).write_bytes(b''.join(struct.pack('<f', x[0]+y[0]) for x,y in zip(a,b)))
''')

    def fake_compile(self, bundle, output, **kwargs):
        output.mkdir()
        (output / "candidate.o").write_bytes(b"fixture-device-ELF")
        return {"status": "success", "candidate": "candidate.o", "compilation_seconds": 0.01}

    def run_evaluation(self, timeout=5):
        with patch.object(evaluate, "replay", side_effect=self.fake_compile):
            result = evaluate.evaluate(self.root / "baseline", self.root / "candidate",
                self.spec, self.root / "report", self.library, timeout, self.runner)
        self.assertEqual(result, json.loads((self.root / "report/report.json").read_text()))
        return result

    def test_correct_baseline_candidate(self):
        result = self.run_evaluation()
        self.assertEqual(result["status"], "correct")
        self.assertTrue(result["performance_eligible"])
        for label in ("baseline", "candidate"):
            self.assertEqual(len(result["correctness"][label]["cases"]), 2)
        self.assertIn("reference-0.bin", result["artifacts"])

    def test_mismatch_retained(self):
        self.runner.write_text(self.runner.read_text().replace('x[0]+y[0]', 'x[0]-y[0]'))
        result = self.run_evaluation()
        self.assertFalse(result["performance_eligible"])
        self.assertGreater(result["correctness"]["candidate"]["cases"][0]["mismatch_count"], 0)
        self.assertTrue((self.root / "report/candidate-execute/output-0.bin").is_file())

    def test_crash_retained(self):
        self.runner.write_text('import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n')
        result = self.run_evaluation()
        self.assertFalse(result["performance_eligible"])
        self.assertEqual(result["correctness"]["candidate"]["execution"]["status"], "crash")

    def test_timeout_retained(self):
        self.runner.write_text('import time\ntime.sleep(30)\n')
        result = self.run_evaluation(timeout=0.1)
        self.assertFalse(result["performance_eligible"])
        self.assertEqual(result["correctness"]["baseline"]["execution"]["status"], "timeout")

    def test_compile_failure(self):
        with patch.object(evaluate, "replay", return_value={"status": "failed"}):
            result = evaluate.evaluate(self.root / "baseline", self.root / "candidate",
                self.spec, self.root / "report", self.library)
        self.assertEqual(result["status"], "compile_failed")
        self.assertFalse(result["performance_eligible"])
        self.assertTrue((self.root / "report/report.json").is_file())

    def test_nonfinite_and_wrong_size(self):
        self.assertFalse(evaluate.compare(evaluate.pack([1]), evaluate.pack([float('nan')]), 0, 0)["passed"])
        self.assertFalse(evaluate.compare(evaluate.pack([1]), b"", 0, 0)["passed"])

    def test_invalid_spec(self):
        spec = json.loads(self.spec.read_text())
        spec["atol"] = -1
        self.spec.write_text(json.dumps(spec))
        self.assertEqual(self.run_evaluation()["status"], "error")

    def test_inputs_reproducible(self):
        spec = evaluate.load_spec(self.spec)
        first, second = self.root / "a", self.root / "b"
        first.mkdir()
        second.mkdir()
        evaluate.make_cases(spec, first)
        evaluate.make_cases(spec, second)
        self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()},
                         {p.name: p.read_bytes() for p in second.iterdir()})


if __name__ == "__main__":
    unittest.main()
