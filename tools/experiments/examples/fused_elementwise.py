"""Aligned f32 multiply/add chain; BLOCK_SIZE is fixed at 64 by the experiment."""
import triton
import triton.language as tl


@triton.jit
def fused_elementwise(a, b, output, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(a + offsets)
    y = tl.load(b + offsets)
    tl.store(output + offsets, x * y + x)
