#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <type_traits>
#include <limits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Custom type trait for floating point types including __half and __nv_bfloat16
// This helps in selecting the correct padding value for sorting.
template<typename T> struct is_custom_fp {
    static const bool value = std::is_floating_point_v<T>;
};
template<> struct is_custom_fp<__half> { static const bool value = true; };
template<> struct is_custom_fp<__nv_bfloat16> { static const bool value = true; };

template<typename T>
inline __device__ void swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

// Parallel Bitonic Sort Kernel (modified for 2D processing)
// Each block in the Y-dimension processes a separate row.
template <typename T, bool ascending>
__global__ void bitonic_sort_kernel(T* arr, uint32_t* dst, int ncols_pad, int j, int k) {
    // blockIdx.y identifies the row this block is responsible for.
    const unsigned int row_idx = blockIdx.y;

    // Calculate the base pointers for the current row.
    T* row_arr = arr + (size_t)row_idx * ncols_pad;
    uint32_t* row_dst = dst + (size_t)row_idx * ncols_pad;

    // Calculate the global thread ID within the row (X-dimension).
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    // Ensure thread is within the padded row bounds.
    if (i >= ncols_pad) return;

    // Determine the index of the element to compare with.
    unsigned int ij = i ^ j;

    // Ensure the comparison is only done once by the thread with the smaller index.
    if (ij > i) {
        // Ensure the other index is also within bounds.
        if (ij < ncols_pad) {
            // Determine the sort direction for this stage.
            bool sort_direction = ((i & k) == 0);

            // Compare and swap elements if they are in the wrong order.
            if ((row_arr[i] > row_arr[ij]) == (sort_direction == ascending)) {
                swap(row_arr[i], row_arr[ij]);
                swap(row_dst[i], row_dst[ij]);
            }
        }
    }
}

// Kernel to copy original data to padded buffer, set padding values, and initialize indices.
template <typename T>
__global__ void prepare_data_kernel(
    const T* original_data,
    uint32_t* dst_padded,
    T* data_padded,
    int nrows,
    int ncols,
    int ncols_pad,
    T pad_value)
{
    // 2D grid-stride loop not strictly necessary if grid is sized perfectly,
    // but this is more robust.
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < nrows && col < ncols_pad) {
        size_t padded_idx = (size_t)row * ncols_pad + col;
        if (col < ncols) {
            // Copy original data.
            size_t original_idx = (size_t)row * ncols + col;
            data_padded[padded_idx] = original_data[original_idx];
            // Initialize indices for original data.
            dst_padded[padded_idx] = col;
        } else {
            // Add padding values.
            data_padded[padded_idx] = pad_value;
            // Initialize indices for padded data (e.g., with 0 or a sentinel).
            dst_padded[padded_idx] = 0;
        }
    }
}


// Calculates the next power of 2 for a given integer.
int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n <<= 1;
    }
    return n;
}

#define ASORT_OP(T, RUST_NAME, ASC) \
extern "C" void RUST_NAME( \
    void * x1, void * dst1, const int nrows, const int ncols, bool inplace, int64_t stream \
) { \
    T* x = reinterpret_cast<T*>(x1); \
    uint32_t* dst = reinterpret_cast<uint32_t*>(dst1); \
    const cudaStream_t custream = (cudaStream_t)stream; \
    \
    /* If there's nothing to sort, return early. */ \
    if (nrows == 0 || ncols == 0) return; \
    \
    /* Calculate padding and total padded size. */ \
    int ncols_pad = next_power_of_2(ncols); \
    size_t padded_size_elements = (size_t)nrows * ncols_pad; \
    \
    /* Allocate padded device buffers for sorting. */ \
    T* x_padded; \
    uint32_t* dst_padded; \
    cudaMallocAsync((void**)&x_padded, padded_size_elements * sizeof(T), custream); \
    cudaMallocAsync((void**)&dst_padded, padded_size_elements * sizeof(uint32_t), custream); \
    \
    /* Determine padding value based on sort order */ \
    T pad_value; \
    if constexpr (ASC) { \
        pad_value = std::numeric_limits<T>::max(); \
    } else { \
        if constexpr (is_custom_fp<T>::value) { \
            pad_value = std::numeric_limits<T>::lowest(); \
        } else { \
            pad_value = std::numeric_limits<T>::min(); \
        } \
    } \
    \
    /* Launch kernel to prepare data (copy, pad, initialize indices) */ \
    dim3 threads_per_block_2d(16, 16); \
    dim3 blocks_per_grid_2d( \
        (ncols_pad + threads_per_block_2d.x - 1) / threads_per_block_2d.x, \
        (nrows + threads_per_block_2d.y - 1) / threads_per_block_2d.y \
    ); \
    prepare_data_kernel<T><<<blocks_per_grid_2d, threads_per_block_2d, 0, custream>>>( \
        x, dst_padded, x_padded, nrows, ncols, ncols_pad, pad_value); \
    \
    /* Bitonic Sort Execution (on all rows in parallel) */ \
    int threads_per_block_1d = 256; \
    dim3 blocks_per_grid_sort( \
        (ncols_pad + threads_per_block_1d - 1) / threads_per_block_1d, \
        nrows \
    ); \
    \
    for (int k = 2; k <= ncols_pad; k <<= 1) { \
        for (int j = k >> 1; j > 0; j >>= 1) { \
            bitonic_sort_kernel<T, ASC><<<blocks_per_grid_sort, threads_per_block_1d, 0, custream>>>( \
                x_padded, dst_padded, ncols_pad, j, k); \
        } \
    } \
    \
    /* If in-place, copy the sorted data back to the original array. */ \
    if (inplace) { \
        cudaMemcpy2DAsync(x, ncols * sizeof(T), x_padded, ncols_pad * sizeof(T), \
                          ncols * sizeof(T), nrows, cudaMemcpyDeviceToDevice, custream); \
    } \
    \
    /* Copy the sorted indices back. */ \
    cudaMemcpy2DAsync(dst, ncols * sizeof(uint32_t), dst_padded, ncols_pad * sizeof(uint32_t), \
                      ncols * sizeof(uint32_t), nrows, cudaMemcpyDeviceToDevice, custream); \
    \
    cudaFreeAsync(x_padded, custream); \
    cudaFreeAsync(dst_padded, custream); \
}

// Instantiate templates for various types and sort orders
ASORT_OP(__nv_bfloat16, asort_asc_bf16, true)
ASORT_OP(__nv_bfloat16, asort_desc_bf16, false)

ASORT_OP(__half, asort_asc_f16, true)
ASORT_OP(__half, asort_desc_f16, false)

ASORT_OP(float, asort_asc_f32, true)
ASORT_OP(float, asort_desc_f32, false)

ASORT_OP(double, asort_asc_f64, true)
ASORT_OP(double, asort_desc_f64, false)

ASORT_OP(uint8_t, asort_asc_u8, true)
ASORT_OP(uint8_t, asort_desc_u8, false)

ASORT_OP(uint32_t, asort_asc_u32, true)
ASORT_OP(uint32_t, asort_desc_u32, false)

ASORT_OP(int64_t, asort_asc_i64, true)
ASORT_OP(int64_t, asort_desc_i64, false)