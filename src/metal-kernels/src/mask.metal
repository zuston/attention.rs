#include "metal_dtype.metal"
#include <metal_stdlib>
using namespace metal;

template <typename T>
T neg_inf_value();

// FP32 specialization
template <>
float neg_inf_value<float>() {
    return -FLT_MAX;
}

// FP16 specialization
template <>
half neg_inf_value<half>() {
    return half(-FLT_MAX);  // Use half(-FLT_MAX) for FP16 in Metal
}

// BF16 specialization
template <>
bfloat16_t neg_inf_value<bfloat16_t>() {
    return bfloat16_t(0xFF80);  // -inf in bfloat16
}

template <typename T>
kernel void causal_mask_upper_inf(
    device T* out [[buffer(0)]],
    constant int& tgt_len [[buffer(1)]],
    constant int& sliding_window [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    const int BLOCK_X = 256;  // threads per block
    const int EPT = 4;       // elements per thread
    int i = gid;  // row index
    if (i >= tgt_len) return;

    int cols_per_block = BLOCK_X * EPT;
    int block_col_base = gid * cols_per_block;
    int thread_col_base = block_col_base + tid * EPT;

    size_t row_offset = i * tgt_len;
    const T neg_inf = neg_inf_value<T>();

    for (int e = 0; e < EPT; ++e) {
        int j = thread_col_base + e;
        if (j >= tgt_len) break;

        // Rule:
        //   -inf if j > i (above diagonal)
        //   optionally also mask far-past if j + sliding_window < i
        if (j > i || (sliding_window > 0 && j + sliding_window < i)) {
            out[row_offset + j] = neg_inf;
        }
    }
}

#define instantiate_causal_mask(type)                         \
  template [[host_name("causal_mask_" #type)]]                \
  kernel void causal_mask_upper_inf<type>(                   \
      device type* out [[buffer(0)]],                         \
      constant int& tgt_len [[buffer(1)]],                    \
      constant int& sliding_window [[buffer(2)]],             \
      uint gid [[thread_position_in_grid]],                   \
      uint tid [[thread_position_in_threadgroup]],            \
      uint threads_per_threadgroup [[threads_per_threadgroup]]);

instantiate_causal_mask(float)
instantiate_causal_mask(bfloat16_t)
instantiate_causal_mask(half)
