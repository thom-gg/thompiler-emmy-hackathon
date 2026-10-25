"""f64 test problems


Submit.py casts every output to float32 before comparing, so we need to add some tolerance
because even getting perfectly accurate results will have rounding errors induced by submit.py
"""

import numpy as np

D = "dtype=torch.float64"

OPEN = [
    # add, multiply, maximum (relu) in f64.
    dict(id="x01_f64_relu_mul_add",
         code=f"x = torch.randn(4096, {D}); y = torch.randn(4096, {D}); torch.relu(x) * y + x",
         ref=lambda x, y: [np.maximum(x, 0) * y + x], rtol=1e-7, atol=0),
    # multiply, add, sqrt, divide in f64, and an f64 constant passed as a kernel arg.
    dict(id="x02_f64_sqrt_div",
         code=f"x = torch.randn(2048, {D}); y = torch.randn(2048, {D}); torch.sqrt(x * x + 1.0) / y",
         ref=lambda x, y: [np.sqrt(x * x + 1.0) / y], rtol=1e-7, atol=0),
]
