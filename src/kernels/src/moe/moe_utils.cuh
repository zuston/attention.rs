#undef __CUDA_FP8_TYPES_EXIST__
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

/**
 * @brief Counts the number of tokens assigned to each expert.
 *
 * @param expert_ids     Device pointer to the sorted expert IDs [size_m].
 * @param expert_counts  Device pointer to the output counts [num_experts]
 * (must be pre-initialized to zero).
 * @param size_m         Total number of tokens.
 */
static __global__ void count_tokens_per_expert_kernel(
    const int32_t* expert_ids, 
    int32_t* expert_counts, 
    int size_m) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size_m) {
        int32_t expert_id = expert_ids[i];
        // expert_id is from a sorted list, so we assume it's valid
        // (i.e., 0 <= expert_id < num_experts)
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

/**
 * @brief Calculates expert offsets array on the GPU.
 *
 * @param d_expert_ids     Device pointer to sorted expert IDs [size_m].
 * @param size_m           Total number of tokens.
 * @param d_expert_offsets Device pointer for output offsets [num_experts + 1].
 * @param num_experts      Number of experts.
 * @param stream           CUDA stream.
 */
static void calculate_expert_offsets(
    const int32_t* d_expert_ids,
    int size_m,
    int32_t* d_expert_offsets,
    int num_experts,
    cudaStream_t stream
) {
    // We need a temporary buffer for counts
    int32_t* d_expert_counts;
    cudaMallocAsync(&d_expert_counts, num_experts * sizeof(int32_t), stream);

    // 1. Zero-initialize the counts buffer
    cudaMemsetAsync(d_expert_counts, 0, num_experts * sizeof(int32_t), stream);

    // 2. Launch kernel to count tokens per expert
    int threads = 256;
    int blocks = (size_m + threads - 1) / threads;
    count_tokens_per_expert_kernel<<<blocks, threads, 0, stream>>>(
        d_expert_ids, d_expert_counts, size_m
    );

    // 3. Perform prefix sum (scan)
    // We will use inclusive_scan on [counts] and store results in [offsets + 1]
    // This is a common and efficient pattern.

    // Wrap raw pointers for Thrust
    thrust::device_ptr<const int32_t> d_counts_ptr(d_expert_counts);
    thrust::device_ptr<int32_t> d_offsets_ptr(d_expert_offsets);

    // Run inclusive scan.
    // Input:  [c0, c1, c2, ...] (size num_experts)
    // Output: [c0, c0+c1, c0+c1+c2, ...] (stored at offsets[1])
    thrust::inclusive_scan(
        thrust::cuda::par.on(stream), // Execute on the specified stream
        d_counts_ptr,                 // Input start
        d_counts_ptr + num_experts,   // Input end
        d_offsets_ptr + 1             // Output start (shifted by 1)
    );

    // 4. Set the first offset (offsets[0]) to 0
    // This completes the exclusive scan.
    cudaMemsetAsync(d_expert_offsets, 0, sizeof(int32_t), stream);

    // 5. Clean up temporary buffer
    cudaFreeAsync(d_expert_counts, stream);
}
