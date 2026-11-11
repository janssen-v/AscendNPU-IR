#!/usr/bin/env python3
"""Host-side contract tests; actual kernels are validated on the NPU host."""
import argparse
import unittest

import triton_fused


class TritonFusedContractTests(unittest.TestCase):
    def test_triton_pointer_abi(self):
        arguments = ', '.join(f'{kind} %{i}' for i, kind in enumerate(triton_fused.EXPECTED_ARGUMENTS))
        ir = f'define dso_local void @fused({arguments}) #2 {{\nret void\n}}'
        self.assertEqual(triton_fused.entry_arguments(ir, 'fused'), triton_fused.EXPECTED_ARGUMENTS)
        # The old replay path inserted an FFTS argument; do not launch it using
        # Triton's differently packed workspace/pointer arguments.
        wrong = ir.replace('@fused(', '@fused(i64 %ffts, ')
        self.assertNotEqual(triton_fused.entry_arguments(wrong, 'fused'), triton_fused.EXPECTED_ARGUMENTS)
        with self.assertRaisesRegex(ValueError, 'missing LLVM kernel entry'):
            triton_fused.entry_arguments(ir, 'different_kernel')

    def test_only_supported_shapes_and_measurements(self):
        args = argparse.Namespace(elements=[4096, 65536], timeout=180,
                                  device=0, warmups=5, rounds=20, iterations=10)
        triton_fused.validate_options(args)
        for elements in ([63], [65], [0], [64, 64], [10000064]):
            with self.subTest(elements=elements), self.assertRaises(ValueError):
                triton_fused.validate_options(argparse.Namespace(**dict(vars(args), elements=elements)))
        for key, value in [('timeout', float('nan')), ('device', -1), ('rounds', 1), ('iterations', 0)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                triton_fused.validate_options(argparse.Namespace(**dict(vars(args), **{key:value})))


if __name__ == '__main__':
    unittest.main()
