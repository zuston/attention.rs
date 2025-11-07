#include <stdint.h>

// Grid: (num_layers, num_pairs)
template<typename scalar_t>
__global__ void copy_blocks_kernel(
  int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int64_t* __restrict__ block_mapping,
  const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

extern "C" void copy_blocks_f16(int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int64_t* __restrict__ block_mapping,
  const int num_layers,
  const int num_pairs,
  int numel_per_block,
  cudaStream_t stream
) {
  if (numel_per_block > 1024) {
    numel_per_block = 1024;
  }
  copy_blocks_kernel<int16_t><<<dim3(num_layers, num_pairs, 1), numel_per_block, 0, stream>>>(key_cache_ptrs, value_cache_ptrs, block_mapping, numel_per_block);
}

extern "C" void copy_blocks_bf16(int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int64_t* __restrict__ block_mapping,
  const int num_layers,
  const int num_pairs,
  int numel_per_block,
  cudaStream_t stream
) {
  if (numel_per_block > 1024) {
    numel_per_block = 1024;
  }
  copy_blocks_kernel<int16_t><<<dim3(num_layers, num_pairs, 1), numel_per_block, 0, stream>>>(key_cache_ptrs, value_cache_ptrs, block_mapping, numel_per_block);
}

extern "C" void copy_blocks_f32(int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int64_t* __restrict__ block_mapping,
  const int num_layers,
  const int num_pairs,
  int numel_per_block,
  cudaStream_t stream
) {
  if (numel_per_block > 1024) {
    numel_per_block = 1024;
  }
  copy_blocks_kernel<float><<<dim3(num_layers, num_pairs, 1), numel_per_block, 0, stream>>>(key_cache_ptrs, value_cache_ptrs, block_mapping, numel_per_block);
}


extern "C" void copy_blocks_u8(int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int64_t* __restrict__ block_mapping,
  const int num_layers,
  const int num_pairs,
  int numel_per_block,
  cudaStream_t stream
) {
  if (numel_per_block > 1024) {
    numel_per_block = 1024;
  }
  copy_blocks_kernel<uint8_t><<<dim3(num_layers, num_pairs, 1), numel_per_block, 0, stream>>>(key_cache_ptrs, value_cache_ptrs, block_mapping, numel_per_block);
}