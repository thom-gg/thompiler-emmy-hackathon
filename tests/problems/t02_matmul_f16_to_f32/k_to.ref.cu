#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_to(const float* to__partial, float* to) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 64;
        int a1 = (_gid) % 64;
        float acc0 = 0.0f;
        for (int _a2_ks = 0; _a2_ks < 2; _a2_ks++) {
            float acc0__p = to__partial[_a2_ks * 8192 + a0 * 64 + a1];
            acc0 += acc0__p;
        }
        __half v1 = __float2half(acc0);
        float v2 = __half2float(v1);
        to[a0 * 64 + a1] = v2;
}
