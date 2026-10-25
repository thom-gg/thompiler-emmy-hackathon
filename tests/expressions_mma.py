"""Tensor-core (mma.sync) test problems
"""

import numpy as np


def _matmul_ref(dtype):
    """a @ b computed in float64, then rounded to the output dtype."""
    return lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(dtype)]


OPEN = [
    # f16 inputs, f16 output.
    dict(id="t01_matmul_f16",
         code="torch.randn(128, 256, dtype=torch.float16) @ torch.randn(256, 64, dtype=torch.float16)",
         ref=_matmul_ref(np.float16), rtol=1e-2, atol=1e-2),
    # f16 inputs, result widened to f32.
    dict(id="t02_matmul_f16_to_f32",
         code="(torch.randn(128, 256, dtype=torch.float16) @ torch.randn(256, 64, dtype=torch.float16)).float()",
         ref=_matmul_ref(np.float32), rtol=1e-2, atol=1e-2),
]
