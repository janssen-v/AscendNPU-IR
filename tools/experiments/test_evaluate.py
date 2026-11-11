#!/usr/bin/env python3
import ctypes as C
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import evaluate
import acl_runner


class ACLLauncherTests(unittest.TestCase):
    def setUp(self):
        self.library = Mock()
        with patch.object(acl_runner.C, "CDLL", return_value=self.library):
            # The mock still receives the actual ctypes argtypes/restype setup.
            self.library.configure_mock(**{
                name + ".return_value": 0 for name in
                ("aclInit", "aclrtSetDevice", "aclrtCreateStream")})
            self.runtime = acl_runner.ACL("unused.so", 0)

    def test_load_vector_binary_eagerly(self):
        def load(data, size, configuration, handle):
            self.assertEqual(C.string_at(data, size), b"fixture ELF")
            options = C.cast(configuration, C.POINTER(
                acl_runner.BinaryLoadOptions)).contents
            self.assertEqual(options.num_options, 2)
            self.assertEqual([(options.options[i].type, options.options[i].value.magic)
                              for i in range(2)], [(1, 0), (2, 0x41415246)])
            C.cast(handle, C.POINTER(C.c_void_p))[0] = 123
            return 0

        self.library.aclrtBinaryLoadFromData.side_effect = load
        self.library.aclrtBinaryGetFunction.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "kernel.o"
            binary.write_bytes(b"fixture ELF")
            self.runtime.load(binary, "add")
        self.library.aclrtBinaryLoadFromFile.assert_not_called()
        self.assertEqual(self.runtime.binaries[0].value, 123)

    def test_launch_copies_host_pointer_arguments(self):
        self.runtime.stream = C.c_void_p(99)

        def launch(function, blocks, stream, config, arguments, size, placeholders, count):
            self.assertEqual((function.value, blocks, stream.value), (17, 1, 99))
            self.assertIsNone(config)
            self.assertIsNone(placeholders)
            self.assertEqual(count, 0)
            self.assertEqual(size, 3 * C.sizeof(C.c_void_p))
            self.assertEqual(list(arguments), [0x120000017000, 0x120000018000, 0x120000019000])
            return 0

        self.library.aclrtLaunchKernelWithHostArgs.side_effect = launch
        self.library.aclrtSynchronizeStream.return_value = 0
        pointers = [C.c_void_p(x) for x in (0x120000017000, 0x120000018000, 0x120000019000)]
        self.runtime.launch(C.c_void_p(17), pointers, 1)
        self.library.aclrtLaunchKernel.assert_not_called()
        self.library.aclrtSynchronizeStream.assert_called_once_with(self.runtime.stream)
        self.library.aclrtSynchronizeStream.reset_mock()
        self.runtime.launch(C.c_void_p(17), pointers, 1, synchronize=False)
        self.library.aclrtSynchronizeStream.assert_not_called()


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
if r.get('mode') == 'benchmark':
 samples = [{'round': i, 'variant': label, 'milliseconds': 1.0}
            for i in range(r['rounds']) for label in
            (('baseline', 'candidate') if i % 2 == 0 else ('candidate', 'baseline'))]
 Path(r['samples']).write_text(json.dumps(samples))
 raise SystemExit(0)
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
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["performance_eligible"])
        for label in ("baseline", "candidate"):
            self.assertEqual(len(result["correctness"][label]["cases"]), 2)
        self.assertIn("reference-0.bin", result["artifacts"])

    def test_mismatch_retained(self):
        self.runner.write_text(self.runner.read_text().replace('x[0]+y[0]', 'x[0]-y[0]'))
        result = self.run_evaluation()
        self.assertFalse(result["performance_eligible"])
        self.assertFalse((self.root / "report/benchmark").exists())
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

    def test_benchmark_failure(self):
        self.runner.write_text(self.runner.read_text().replace("raise SystemExit(0)", "raise SystemExit(2)"))
        result = self.run_evaluation()
        self.assertTrue(result["performance_eligible"])
        self.assertEqual(result["status"], "benchmark_failed")
        self.assertNotIn("baseline_over_candidate", result["benchmark"])

    def test_invalid_samples(self):
        with self.assertRaises(ValueError):
            evaluate.summarize_samples([], 2)
        with self.assertRaises(ValueError):
            evaluate.summarize_samples([{"round": 0, "variant": "baseline", "milliseconds": float('nan')}] * 4, 2)

    def test_benchmark_warmup_and_order(self):
        inputs = [self.root / "lhs", self.root / "rhs"]
        for p in inputs:
            p.write_bytes(evaluate.pack([1]))
        from unittest.mock import Mock
        runtime = Mock()
        runtime.load.side_effect = lambda path, entry: str(path)
        runtime.measure.return_value = 1.0
        request = {"inputs": inputs, "bytes": 4,
                   "binaries": {"baseline": "baseline", "candidate": "candidate"},
                   "entry": "add", "blocks": 1, "warmups": 3,
                   "rounds": 2, "iterations": 10, "samples": self.root / "samples.json"}
        samples = acl_runner.benchmark(runtime, request)
        self.assertEqual(runtime.launch.call_count, 6)
        self.assertEqual([s["variant"] for s in samples],
                         ["baseline", "candidate", "candidate", "baseline"])
        self.assertEqual([c.args[0] for c in runtime.measure.call_args_list],
                         ["baseline", "candidate", "candidate", "baseline"])

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
