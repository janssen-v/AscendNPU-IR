#!/usr/bin/env python3
"""Inventory/coverage contracts; NPU validation is performed by run.py."""
import ast
from pathlib import Path
import unittest
import json
import tempfile
from unittest.mock import patch

import run
from catalog import FAMILIES, KERNELS, PROFILES, specification


class SuiteTests(unittest.TestCase):
    def test_twelve_kernels_three_per_family(self):
        self.assertEqual(len(FAMILIES), 4)
        self.assertEqual([len(v) for v in FAMILIES.values()], [3, 3, 3, 3])
        self.assertEqual(len(set(KERNELS)), 12)
        tree = ast.parse(Path(__file__).with_name('kernels.py').read_text())
        kernels = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(kernels, set(KERNELS))

    def test_explicit_coverage_and_tail_profiles(self):
        for name in KERNELS:
            aligned, tail = (specification(name, p) for p in ('aligned', 'tail'))
            self.assertEqual(aligned['n'] % 64, 0)
            self.assertNotEqual(tail['n'] % 64, 0)
            self.assertEqual(aligned['seeds'], [7, 19, 42])
            self.assertIsNone(aligned['candidate_recipe'])
            self.assertEqual(aligned['coverage'], 'recipe_required')
            self.assertIsNone(tail['candidate_recipe'])
            if aligned['family'] == 'layout':
                self.assertEqual((aligned['atol'], aligned['rtol']), (0, 0))

    def test_reject_unknown_case(self):
        for name, profile in [('missing', 'aligned'), ('mul_add', 'missing')]:
            with self.assertRaises(ValueError):
                specification(name, profile)

    def test_supervisor_preserves_missing_recipe_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'results'
            def fail_worker(command, cwd, env, timeout):
                (cwd / 'report.json').write_text(json.dumps({
                    'status': 'recipe_required',
                    'candidate_status': 'recipe_required',
                    'performance_eligible': False}))
                return {'status': 'failed', 'returncode': 1}
            argv = ['run.py', '--output', str(output), '--toolchain-bin', temp,
                    '--mode', 'compare', '--kernels', 'silu', '--profiles', 'aligned']
            with patch.object(run.sys, 'argv', argv), patch.object(
                    run, 'run_process', side_effect=fail_worker):
                self.assertEqual(run.main(), 1)
            result = json.loads((output / 'suite.json').read_text())
            self.assertEqual(result['results']['silu-aligned']['status'], 'recipe_required')
            self.assertFalse(list(output.rglob('samples.json')))

    def test_supervisor_reports_worker_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'results'
            argv = ['run.py', '--output', str(output), '--toolchain-bin', temp,
                    '--kernels', 'silu', '--profiles', 'aligned']
            with patch.object(run.sys, 'argv', argv), patch.object(
                    run, 'run_process', return_value={'status': 'timeout', 'returncode': -9}):
                self.assertEqual(run.main(), 1)
            result = json.loads((output / 'suite.json').read_text())
            self.assertEqual(result['results']['silu-aligned']['status'], 'worker_failed')


if __name__ == '__main__':
    unittest.main()
