/**
 * @brief WMMA-based CUDA kernel for Mixture-of-Experts (MoE) GEMM with tiling and output masking.
 * Copyright (c) 2025, Guoqing Bao.  All rights reserved.
 *
 * This kernel performs a matrix multiplication for a segment of tokens routed to the same expert
 * using NVIDIA's Warp Matrix Multiply-Accumulate (WMMA) API. It operates on one contiguous
 * segment of tokens (as grouped by the host) and supports optional top-k gating weights.
 *
 * This CUDA kernel is developed for vLLM.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/moe_gemm_wmma.cu
 *
 * Each CUDA block computes one WMMA tile (16x16) of the output matrix, corresponding to one
 * token segment for a single expert. The kernel efficiently handles partial tiles along the
 * M dimension (i.e., when the segment has fewer than 16 rows) using output masking.
 *
 * @details
 * - Each CUDA block computes a single WMMA tile of size (16 x 16).
 * - BlockDim.x = 32 (one warp per block).
 * - The host launches one kernel per expert segment (i.e., per contiguous group of tokens assigned to an expert).
 * - Shared memory layout (in bytes):
 *      [A_sh (WMMA_M * WMMA_K * sizeof(T))] 
 *      [B_sh_col (WMMA_K * WMMA_N * sizeof(T))] 
 *      [C_sh (WMMA_M * WMMA_N * sizeof(float))]
 * - Shared memory pointers are properly aligned for mixed-precision computation.
 * - Optimized for non-quantized MoE GEMM with K-tiling only.
 *
 * @note: this wwma MoE kernel is only used for prefill not compatible with cuda graph (decoding) 
 * since we requires dynamic host ptr to build expert segments (each segment responsible for a kernel launch)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cassert>
#include <cstring>
#include "attention/attention_dtypes.h"
#include "attention/attention_utils.cuh"
using namespace nvcuda::wmma;

namespace vllm {

inline __device__ void from_float(half& dst, float src) {
  dst = static_cast<half>(float_to_half(src));
}

inline __device__ float to_float(half u) {
  return half_to_float(static_cast<uint16_t>(u));
}

}

#define CEILDIV(x,y) (((x) + (y) - 1) / (y))

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

/*
 * @tparam T                   Data type (e.g., half, float) used for computation.
 *
 * @param input                [size_m, size_k] - Input activations for all tokens.
 * @param weights              [num_experts, size_n, size_k] - Expert weight matrices (expert-major layout).
 * @param sorted_token_ids     [size_m] - Indices of tokens sorted by expert assignment.
 * @param topk_weights         [size_m] (optional) - Per-token top-k gating weights (can be nullptr).
 * @param segment_start        Starting index within `sorted_token_ids` for this expert segment.
 * @param num_rows             Number of valid rows (tokens) in this segment (M_i).
 * @param expert_id            Expert ID corresponding to this kernel segment.
 * @param topk                 Number of experts selected per token (top-k routing).
 * @param output               [size_m, size_n] - Output activations for all tokens.
 * @param size_m               Total number of tokens.
 * @param size_n               Output dimension (per expert).
 * @param size_k               Input dimension (per expert).
*/
template<typename T>
__global__ void moe_gemm_wmma_kernel(
    const T* __restrict__ input,             // [size_m, size_k]
    const T* __restrict__ weights,           // [num_experts, size_n, size_k]
    const int32_t* __restrict__ sorted_token_ids, // [size_m]
    const float* __restrict__ topk_weights,
    const int32_t segment_start,                // start index (in sorted_token_ids) of this segment
    const int32_t num_rows,                     // number of valid rows in this segment (M_i)
    const int32_t expert_id,
    const int32_t topk,
    T* __restrict__ output,                 // [size_m, size_n]
    const int32_t size_m,
    const int32_t size_n,
    const int32_t size_k
) {
    // tile coordinates
    const int tile_m_idx = blockIdx.x; // tile index along M within the segment
    const int tile_n_idx = blockIdx.y; // tile index along N
    const int laneId = threadIdx.x & 31; // lane within warp

    const int row_start_in_segment = tile_m_idx * WMMA_M; // 0..num_rows
    const int col_start = tile_n_idx * WMMA_N;           // 0..size_n

    if (row_start_in_segment >= num_rows) return;
    if (col_start >= size_n) return;

    // expert weight base pointer (expert_w[n * size_k + k])
    const T* expert_w = weights + (size_t)expert_id * (size_t)size_n * (size_t)size_k;

    // Shared memory pointer (bytes). We'll reinterpret as halves & floats appropriately.
    extern __shared__ uint8_t smem_bytes[]; // allocated by launcher
    // Layout offsets (in bytes)
    const size_t A_sh_elems = WMMA_M * WMMA_K; // 256 halves
    const size_t B_sh_elems = WMMA_K * WMMA_N; // 256 halves
    const size_t C_sh_elems = WMMA_M * WMMA_N; // 256 floats

    uint8_t* ptr = smem_bytes;
    T* A_sh = reinterpret_cast<T*>(ptr); // size A_sh_elems
    ptr += A_sh_elems * 2;
    T* B_sh_col = reinterpret_cast<T*>(ptr); // size B_sh_elems
    ptr += B_sh_elems * 2;
    // align next pointer to float alignment
    size_t offset = reinterpret_cast<uintptr_t>(ptr) % alignof(float);
    if (offset != 0) {
        ptr += (alignof(float) - offset);
    }
    float* C_sh = reinterpret_cast<float*>(ptr); // size C_sh_elems

    // Determine valid rows in this tile (1..16)
    int rows_in_tile = num_rows - row_start_in_segment;
    if (rows_in_tile > WMMA_M) rows_in_tile = WMMA_M;

    // Number of K-chunks
    const int K_chunks = CEILDIV(size_k, WMMA_K);

    // accumulator fragment
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    T zero_value;
    vllm::zero(zero_value);
    // Loop over K chunks
    for (int kc = 0; kc < K_chunks; ++kc) {
        const int k_base = kc * WMMA_K;

        // Cooperative load into shared memory (A_sh and B_sh_col). Each warp thread loads multiple elements.
        // TODO (guoqingbao): optimize with vectorized load
        const int total_elems = (int)A_sh_elems + (int)B_sh_elems; // 512
        for (int idx = laneId; idx < total_elems; idx += 32) {
            if (idx < (int)A_sh_elems) {
                // A_sh: row-major [m_local * WMMA_K + k_local]
                int m_local = idx / WMMA_K;
                int k_local = idx % WMMA_K;
                int global_row = row_start_in_segment + m_local;
                T v = zero_value;
                if (global_row < num_rows) {
                    int token_index = sorted_token_ids[segment_start + global_row]; // original token index
                    int k_idx = k_base + k_local;
                    if (k_idx < size_k && token_index < size_m) {
                        v = input[(size_t)(token_index / (topk_weights? 1: topk)) * size_k + k_idx];
                    }
                }
                A_sh[m_local * WMMA_K + k_local] = v;
            } else {
                int idxB = idx - (int)A_sh_elems;
                // B_sh_col arranged column-major: (k_local + n_local*WMMA_K)
                int k_local = idxB / WMMA_N;
                int n_local = idxB % WMMA_N;
                int n_global = col_start + n_local;
                T wb = zero_value;
                if (n_global < size_n) {
                    int k_idx = k_base + k_local;
                    if (k_idx < size_k) {
                        wb = expert_w[(size_t)n_global * size_k + k_idx];
                    }
                }
                B_sh_col[k_local + n_local * WMMA_K] = wb;
            }
        }

        __syncthreads();

        // Load fragments from shared memory
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, T, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, T, col_major> b_frag;

        // pointers & leading dims
        const T* A_sh_ptr = A_sh;       // row-major with ld = WMMA_K
        const T* B_sh_ptr = B_sh_col;   // col-major with ld = WMMA_K

        // load fragments (warp-level)
        load_matrix_sync(a_frag, A_sh_ptr, WMMA_K);
        load_matrix_sync(b_frag, B_sh_ptr, WMMA_K);

        // mma
        mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    } // kc

    // Store c_frag to shared float buffer (C_sh) using store_matrix_sync (row-major)
    // store_matrix_sync arguments: (ptr, frag, ld, mem_row_major)
    store_matrix_sync(C_sh, c_frag, WMMA_N, mem_row_major);

    __syncthreads();

    // Write only valid rows (masking) from C_sh to global output.
    // Each lane writes multiple elements from C_sh cooperatively
    const int total_c_elems = WMMA_M * WMMA_N; // 256
    // TODO (guoqingbao): optimize with vectorized store
    for (int idx = laneId; idx < total_c_elems; idx += 32) {
        int m_local = idx / WMMA_N; // 0..15
        int n_local = idx % WMMA_N; // 0..15
        if (m_local >= rows_in_tile) continue; // mask out padded rows
        int row_in_segment = row_start_in_segment + m_local;
        int token_index = sorted_token_ids[segment_start + row_in_segment];
        int col = col_start + n_local;
        if (col >= size_n) continue;
        // Write float to output[token_index, col]
        float val = C_sh[m_local * WMMA_N + n_local];
        if (topk_weights) val *= topk_weights[token_index];
        vllm::from_float(output[(size_t)token_index * size_n + col], val);
    }
}


// Segment description
struct ExpertSegment { int start; int len; int expert; };

// Make segments from expert_ids_host (per-sorted-token expert ids)
static std::vector<ExpertSegment> make_segments_from_expert_ids_host(const int32_t* expert_ids_host, int size_m) {
    std::vector<ExpertSegment> out;
    int cur = 0;
    while (cur < size_m) {
        int e = expert_ids_host[cur];
        int s = cur;
        ++cur;
        while (cur < size_m && expert_ids_host[cur] == e) ++cur;
        out.push_back({s, cur - s, e});
    }
    return out;
}

extern "C" void moe_gemm_wmma(
    const void* input,                 // [size_m, size_k]
    const void* weights,               // [num_experts, size_n, size_k]
    const int32_t* sorted_token_ids,   // [size_m]
    const int32_t* expert_ids_host,      // host array [size_m] (expert id per sorted token)
    const float* topk_weights,
    void* output,                     // [size_m, size_n]
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int data_type,
    cudaStream_t stream
) {
    // Build segments
    std::vector<ExpertSegment> segments = make_segments_from_expert_ids_host(expert_ids_host, size_m);

    // Shared memory bytes: A_sh + B_sh_col (halves) + C_sh floats (ensure float alignment)
    const size_t A_sh_elems = WMMA_M * WMMA_K;
    const size_t B_sh_elems = WMMA_K * WMMA_N;
    const size_t C_sh_elems = WMMA_M * WMMA_N;

    // Compute shared size that the kernel expects:
    // bytes = (A_sh_elems + B_sh_elems)*half_bytes + padding + C_sh_elems*float_bytes
    size_t smem_bytes_min = (A_sh_elems + B_sh_elems) * 2;
    // ensure alignment for float buffer
    size_t pad = (alignof(float) - (smem_bytes_min % alignof(float))) % alignof(float);
    size_t smem_bytes = smem_bytes_min + pad + C_sh_elems * 4;

    // Launch per-segment
    for (const auto &seg : segments) {
        if (seg.len <= 0) continue;
        int seg_start = seg.start;
        int seg_len = seg.len;
        int expert = seg.expert;
        assert(expert >= 0 && expert < num_experts);

        int grid_x = CEILDIV(seg_len, WMMA_M); // M tiles inside segment
        int grid_y = CEILDIV(size_n, WMMA_N);  // N tiles
        dim3 grid(grid_x, grid_y, 1);
        dim3 block(32, 1, 1); // one warp per block

        if (data_type == 0) {
            moe_gemm_wmma_kernel<<<grid, block, smem_bytes, stream>>>(
                reinterpret_cast<const half*>(input),
                reinterpret_cast<const half*>(weights),
                sorted_token_ids,
                topk_weights,
                seg_start,
                seg_len,
                expert,
                topk,
                reinterpret_cast<half*>(output),
                size_m,
                size_n,
                size_k
            );
        } else if (data_type == 1) {
            moe_gemm_wmma_kernel<<<grid, block, smem_bytes, stream>>>(
                reinterpret_cast<const nv_bfloat16*>(input),
                reinterpret_cast<const nv_bfloat16*>(weights),
                sorted_token_ids,
                topk_weights,
                seg_start,
                seg_len,
                expert,
                topk,
                reinterpret_cast<nv_bfloat16*>(output),
                size_m,
                size_n,
                size_k
            );
        }
        
    }
}
