use candle_core as candle;
use candle_core::{DType, Result, Tensor};
#[cfg(feature = "cuda")]
use kernels::ffi;

#[derive(Debug, Clone)]
struct CausalMask {
    sliding_window: i32,
}

impl candle::InplaceOp1 for CausalMask {
    fn name(&self) -> &'static str {
        "causal_mask"
    }

    fn cpu_fwd(&self, _: &mut candle::CpuStorage, _: &candle::Layout) -> Result<()> {
        panic!("causal_mask is not implemented on CPU!")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        input: &mut candle::CudaStorage,
        input_layout: &candle::Layout,
    ) -> Result<()> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::CudaStorageSlice;
        let dev = input.device();
        let (tgt_len, tgt_len1) = input_layout.shape().dims2()?;
        assert!(
            tgt_len == tgt_len1,
            "Casual mask tensor should has same dim0 and dim1!"
        );
        use std::ffi::c_void;
        let src_ptr = match &input.slice {
            CudaStorageSlice::F32(inp) => *inp.device_ptr() as *mut c_void,
            CudaStorageSlice::F16(inp) => *inp.device_ptr() as *mut c_void,
            CudaStorageSlice::BF16(inp) => *inp.device_ptr() as *mut c_void,
            _ => {
                candle_core::bail!("Casual mask tensor should has dtype of f16, bf16 or f32!")
            }
        };
        let stream = *dev.cu_stream() as i64;

        unsafe {
            match input.dtype() {
                DType::F32 => ffi::causal_mask_f32(
                    src_ptr,
                    tgt_len as i32,
                    self.sliding_window as i32,
                    stream,
                ),
                DType::F16 => ffi::causal_mask_f16(
                    src_ptr,
                    tgt_len as i32,
                    self.sliding_window as i32,
                    stream,
                ),
                DType::BF16 => ffi::causal_mask_bf16(
                    src_ptr,
                    tgt_len as i32,
                    self.sliding_window as i32,
                    stream,
                ),
                _ => {
                    candle_core::bail!("Casual mask tensor should has dtype of f16, bf16 or f32!")
                }
            }
        }
        Ok(())
    }
}

pub fn causal_mask(mask: &Tensor, sliding_window: Option<usize>) -> Result<()> {
    let op = CausalMask {
        sliding_window: sliding_window.unwrap_or(0) as i32,
    };
    mask.inplace_op1(&op)
}
