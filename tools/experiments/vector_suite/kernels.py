"""Twelve f32 vector kernels. Dimensions are constexpr; tails are masked."""
import triton
import triton.language as tl


@triton.jit
def mul_add(A, B, Y, N: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    a = tl.load(A + i, i < N, 0)
    b = tl.load(B + i, i < N, 0)
    tl.store(Y + i, a * b + a, i < N)


@triton.jit
def silu(X, Y, N: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    x = tl.load(X + i, i < N, 0)
    tl.store(Y + i, x / (1.0 + tl.exp(-x)), i < N)


@triton.jit
def gelu(X, Y, N: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    x = tl.load(X + i, i < N, 0)
    z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_z = 2.0 / (1.0 + tl.exp(-2.0 * z)) - 1.0
    tl.store(Y + i, 0.5 * x * (1.0 + tanh_z), i < N)


@triton.jit
def bias_add(X, B, Y, M: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * TILE + tl.arange(0, TILE)
    x = tl.load(X + row * N + c, c < N, 0)
    bias = tl.load(B + c, c < N, 0)
    tl.store(Y + row * N + c, x + bias, c < N)


@triton.jit
def row_scale(X, S, Y, M: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * TILE + tl.arange(0, TILE)
    x = tl.load(X + row * N + c, c < N, 0)
    scale = tl.load(S + row)
    tl.store(Y + row * N + c, x * scale, c < N)


@triton.jit
def affine(X, S, B, Y, M: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * TILE + tl.arange(0, TILE)
    x = tl.load(X + row * N + c, c < N, 0)
    scale = tl.load(S + c, c < N, 0)
    bias = tl.load(B + c, c < N, 0)
    tl.store(Y + row * N + c, x * scale + bias, c < N)


@triton.jit
def row_sum(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + c, c < N, 0)
    tl.store(Y + row, tl.sum(x, axis=0))


@triton.jit
def softmax(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + c, c < N, float('-inf'))
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(Y + row * N + c, e / tl.sum(e, axis=0), c < N)


@triton.jit
def rmsnorm(X, W, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + c, c < N, 0)
    w = tl.load(W + c, c < N, 0)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / N + 1.0e-5)
    tl.store(Y + row * N + c, x * inv * w, c < N)


@triton.jit
def transpose(X, Y, M: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    r = tl.program_id(0) * TILE + tl.arange(0, TILE)
    c = tl.program_id(1) * TILE + tl.arange(0, TILE)
    mask = (r[:, None] < M) & (c[None, :] < N)
    x = tl.load(X + r[:, None] * N + c[None, :], mask, 0)
    tl.store(Y + c[None, :] * M + r[:, None], x, mask)


@triton.jit
def strided_copy(X, Y, M: tl.constexpr, N: tl.constexpr, STRIDE: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    x = tl.load(X + i * STRIDE, i < M * N, 0)
    tl.store(Y + i, x, i < M * N)


@triton.jit
def channels_last(X, Y, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, TILE: tl.constexpr):
    # One NCHW image -> contiguous NHWC; flat output indexing exposes strided reads.
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    channel = i % C
    spatial = i // C
    x = tl.load(X + channel * H * W + spatial, i < C * H * W, 0)
    tl.store(Y + i, x, i < C * H * W)
