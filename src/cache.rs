use candle_core::{backend::BackendDevice, Device, Result, Storage, Tensor};
use std::collections::HashMap;

#[cfg(feature = "cuda")]
pub fn swap_blocks(
    src: &Tensor,
    dst: &Tensor,
    block_mapping: &HashMap<usize, usize>,
) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::{result, CudaSlice, DevicePtr};
    use std::slice;
    let block_size_elements = src.elem_count() / src.dim(0)?;
    let (src_storage, _) = src.storage_and_layout();
    let (dst_storage, _) = dst.storage_and_layout();

    match (src.device(), dst.device()) {
        (Device::Cpu, Device::Cuda(dst_dev)) => {
            let Storage::Cpu(src_storage) = &*src_storage else {
                candle_core::bail!("Invalid source kvcache storage!")
            };
            let Storage::Cuda(dst_storage) = &*dst_storage else {
                candle_core::bail!("Invalid dst kvcache storage!")
            };
            let cpu_num_blocks = src.dim(0)?;
            let gpu_num_blocks = dst.dim(0)?;

            let dst_ptr = dst_storage.as_cuda_slice::<half::bf16>()?.device_ptr();
            let src_slice: &[half::bf16] = src_storage.as_slice()?;

            for (src_block_number, dst_block_number) in block_mapping {
                let src_offset: usize = src_block_number * block_size_elements;
                assert!(
                    *src_block_number < cpu_num_blocks,
                    "Invalid cpu block {} / {}",
                    src_block_number,
                    cpu_num_blocks
                );
                assert!(
                    *dst_block_number < gpu_num_blocks,
                    "Invalid gpu block {} / {}",
                    dst_block_number,
                    gpu_num_blocks
                );

                assert!(
                    src_offset + block_size_elements <= src_slice.len(),
                    "Invalid cpu kvcache block {} for offload",
                    src_block_number
                );

                let dst_offset: u64 = (dst_block_number * block_size_elements).try_into().unwrap();
                let dst_slice: std::mem::ManuallyDrop<CudaSlice<half::bf16>> = unsafe {
                    let slice =
                        dst_dev.upgrade_device_ptr(dst_ptr + dst_offset, block_size_elements);
                    std::mem::ManuallyDrop::new(slice)
                };

                unsafe {
                    result::memcpy_htod_async(
                        *dst_slice.device_ptr(),
                        &src_slice[src_offset..src_offset + block_size_elements],
                        *dst_dev.cu_stream(),
                    )
                    .map_err(candle_core::Error::wrap)?
                }
            }
            let _ = dst_dev.synchronize();
        }
        (Device::Cuda(src_dev), Device::Cpu) => {
            let Storage::Cuda(src_storage) = &*src_storage else {
                candle_core::bail!("Invalid source kvcache storage!")
            };
            let Storage::Cpu(dst_storage) = &*dst_storage else {
                candle_core::bail!("Invalid dst kvcache storage!")
            };
            let gpu_num_blocks = src.dim(0)?;
            let cpu_num_blocks = dst.dim(0)?;

            let src_ptr = src_storage
                .as_cuda_slice::<half::bf16>()
                .map_err(candle_core::Error::wrap)?
                .device_ptr();
            let dst_slice: &[half::bf16] =
                dst_storage.as_slice().map_err(candle_core::Error::wrap)?;
            let ptr = dst_slice.as_ptr() as *mut half::bf16;

            for (src_block_number, dst_block_number) in block_mapping {
                assert!(
                    *src_block_number < gpu_num_blocks,
                    "Invalid gpu block {} / {}",
                    src_block_number,
                    gpu_num_blocks
                );
                assert!(
                    *dst_block_number < cpu_num_blocks,
                    "Invalid cpu block {} / {}",
                    dst_block_number,
                    cpu_num_blocks
                );

                let src_offset: u64 = (src_block_number * block_size_elements).try_into().unwrap();
                let dst_offset: usize =
                    (dst_block_number * block_size_elements).try_into().unwrap();
                let dst_slice = unsafe {
                    slice::from_raw_parts_mut(ptr.wrapping_add(dst_offset), block_size_elements)
                };

                let src_slice = unsafe {
                    let slice: CudaSlice<half::bf16> =
                        src_dev.upgrade_device_ptr(src_ptr + src_offset, block_size_elements);
                    std::mem::ManuallyDrop::new(slice)
                };

                unsafe {
                    result::memcpy_dtoh_async(
                        dst_slice,
                        *src_slice.device_ptr(),
                        *src_dev.cu_stream(),
                    )
                    .map_err(candle_core::Error::wrap)?;
                }
            }
            let _ = src_dev.synchronize();
        }
        (src, dst) => {
            candle_core::bail!("Tensors must be on either the GPU or CPU to swap,, got {src:?} (src) and {dst:?} (dst).")
        }
    }

    Ok(())
}

#[cfg(feature = "metal")]
pub fn swap_blocks(
    src: &Tensor,
    dst: &Tensor,
    block_mapping: &HashMap<usize, usize>,
) -> Result<()> {
    candle_core::bail!("Unified memory does not support block swap!")
}
