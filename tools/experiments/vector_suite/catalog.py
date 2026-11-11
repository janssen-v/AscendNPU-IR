"""Dependency-free benchmark inventory and explicit transform coverage."""
FAMILIES = {
    'elementwise': ('mul_add', 'silu', 'gelu'),
    'broadcast': ('bias_add', 'row_scale', 'affine'),
    'reduction': ('row_sum', 'softmax', 'rmsnorm'),
    'layout': ('transpose', 'strided_copy', 'channels_last'),
}
KERNELS = tuple(name for names in FAMILIES.values() for name in names)
PROFILES = {'aligned': (16, 128), 'tail': (17, 65)}


def specification(name, profile):
    if name not in KERNELS or profile not in PROFILES:
        raise ValueError('unknown kernel/profile')
    m, n = PROFILES[profile]
    family = next(f for f, names in FAMILIES.items() if name in names)
    return {'kernel': name, 'family': family, 'profile': profile, 'm': m, 'n': n,
            'dtype': 'float32', 'seeds': [7, 19, 42],
            'atol': 0.0 if family == 'layout' else (2e-5 if family == 'reduction' or name in ('silu', 'gelu') else 1e-6),
            'rtol': 0.0 if family == 'layout' else (2e-4 if family == 'reduction' or name in ('silu', 'gelu') else 1e-5),
            'candidate_recipe': None,
            'coverage': 'recipe_required'}


def make_case(torch, spec, seed):
    """CPU inputs and independent PyTorch reference; no device or Triton imports."""
    m, n, name = spec['m'], spec['n'], spec['kernel']
    g = torch.Generator().manual_seed(seed)
    def rand(*shape):
        return torch.randn(shape, generator=g, dtype=torch.float32)
    x = rand(m, n)
    if name == 'mul_add':
        b = rand(m, n)
        inputs, expected, constants, grid = [x, b], x * b + x, (m*n, 64), ((m*n+63)//64,)
    elif name in ('silu', 'gelu'):
        expected = torch.nn.functional.silu(x) if name == 'silu' else torch.nn.functional.gelu(x, approximate='tanh')
        inputs, constants, grid = [x], (m*n, 64), ((m*n+63)//64,)
    elif name == 'bias_add':
        b = rand(n)
        inputs, expected, constants, grid = [x, b], x + b, (m, n, 64), (m, (n+63)//64)
    elif name == 'row_scale':
        s = rand(m)
        inputs, expected, constants, grid = [x, s], x * s[:, None], (m, n, 64), (m, (n+63)//64)
    elif name == 'affine':
        s, b = rand(n), rand(n)
        inputs, expected, constants, grid = [x, s, b], x * s + b, (m, n, 64), (m, (n+63)//64)
    elif name in ('row_sum', 'softmax', 'rmsnorm'):
        inputs = [x]
        if name == 'row_sum':
            expected = x.sum(dim=1)
        elif name == 'softmax':
            expected = torch.softmax(x, dim=1)
        else:
            w = rand(n)
            inputs.append(w)
            expected = x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + 1e-5) * w
        constants, grid = (m, n, 1 << (n-1).bit_length()), (m,)
    elif name == 'transpose':
        inputs, expected, constants, grid = [x], x.t().contiguous(), (m, n, 16), ((m+15)//16, (n+15)//16)
    elif name == 'strided_copy':
        x = rand(m*n*3)
        inputs, expected, constants, grid = [x], x[::3].reshape(m, n).contiguous(), (m, n, 3, 64), ((m*n+63)//64,)
    else:
        x = rand(3, m, n)
        inputs, expected, constants, grid = [x], x.permute(1, 2, 0).contiguous(), (3, m, n, 64), ((3*m*n+63)//64,)
    return inputs, expected, constants, grid + (1,) * (3-len(grid))
