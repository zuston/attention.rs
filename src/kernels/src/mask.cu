#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>

template <typename T>
__device__ __forceinline__ T neg_inf_value();

// FP32 specialization
template <>
__device__ __forceinline__ float neg_inf_value<float>() {
    return -FLT_MAX;
}

// FP16 specialization
template <>
__device__ __forceinline__ __half neg_inf_value<__half>() {
#if __CUDA_ARCH__ >= 530
    return __half_raw{0xFC00};  // bit pattern for -inf in IEEE-754 half
#else
    return __float2half(-FLT_MAX);
#endif
}

// BF16 specialization
template <>
__device__ __forceinline__ __nv_bfloat16 neg_inf_value<__nv_bfloat16>() {
    return __nv_bfloat16_raw{0xFF80};  // bit pattern for -inf in bfloat16
}

constexpr int BLOCK_X = 256;  // threads per block
constexpr int EPT     = 4;    // elements per thread

template <typename T>
__global__ void causal_mask_upper_inf(T* __restrict__ out,
                                      int tgt_len,
                                      int sliding_window)
{
    int i = blockIdx.y;  // row index
    if (i >= tgt_len) return;

    int cols_per_block = blockDim.x * EPT;
    int block_col_base = blockIdx.x * cols_per_block;
    int thread_col_base = block_col_base + threadIdx.x * EPT;

    size_t row_offset = static_cast<size_t>(i) * static_cast<size_t>(tgt_len);
    const T neg_inf = neg_inf_value<T>();

    #pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int j = thread_col_base + e;
        if (j >= tgt_len) break;

        // Rule:
        //   -inf if j > i  (above diagonal)
        //   optionally also mask far-past if j + sliding_window < i
        if (j > i || (sliding_window > 0 && j + sliding_window < i)) {
            out[row_offset + j] = neg_inf;
        }
    }
}

extern "C" void causal_mask_f32(void* d_out,
                            int tgt_len,
                            int sliding_window,
                            cudaStream_t stream)
{
    int cols_per_block = BLOCK_X * EPT;
    int grid_x = (tgt_len + cols_per_block - 1) / cols_per_block;
    dim3 grid(grid_x, tgt_len, 1);
    dim3 block(BLOCK_X, 1, 1);

    causal_mask_upper_inf<float><<<grid, block, 0, stream>>>(reinterpret_cast<float*>(d_out), tgt_len, sliding_window);
}

extern "C" void causal_mask_bf16(nv_bfloat16* d_out,
                            int tgt_len,
                            int sliding_window,
                            cudaStream_t stream)
{
    int cols_per_block = BLOCK_X * EPT;
    int grid_x = (tgt_len + cols_per_block - 1) / cols_per_block;
    dim3 grid(grid_x, tgt_len, 1);
    dim3 block(BLOCK_X, 1, 1);

    causal_mask_upper_inf<nv_bfloat16><<<grid, block, 0, stream>>>(reinterpret_cast<nv_bfloat16*>(d_out), tgt_len, sliding_window);
}

extern "C" void causal_mask_f16(half* d_out,
                            int tgt_len,
                            int sliding_window,
                            cudaStream_t stream) {
    int cols_per_block = BLOCK_X * EPT;
    int grid_x = (tgt_len + cols_per_block - 1) / cols_per_block;
    dim3 grid(grid_x, tgt_len, 1);
    dim3 block(BLOCK_X, 1, 1);

    causal_mask_upper_inf<half><<<grid, block, 0, stream>>>(reinterpret_cast<half*>(d_out), tgt_len, sliding_window);
}
