use candle_core::{DType, Device, Result, Tensor};

#[derive(Clone)]
pub enum KvScaleCalculator {
    InProgress {
        k_scale: Tensor,
        v_scale: Tensor,
        n: usize,
    },
    Done {
        k_scale: Tensor,
        v_scale: Tensor,
    },
}

impl KvScaleCalculator {
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self::InProgress {
            k_scale: Tensor::new(1f32, device)?,
            v_scale: Tensor::new(1f32, device)?,
            n: 0,
        })
    }

    pub fn collect(&mut self, k_scale_new: &Tensor, v_scale_new: &Tensor) -> Result<usize> {
        match self {
            Self::InProgress {
                k_scale,
                v_scale,
                n,
            } => {
                *k_scale = k_scale.clone().maximum(k_scale_new)?;
                *v_scale = v_scale.clone().maximum(v_scale_new)?;
                *n += 1;
                Ok(*n)
            }
            Self::Done { .. } => {
                candle_core::bail!("KvScaleCalculator::collect requires InProgress scales");
            }
        }
    }

    pub fn finish(&mut self) -> Result<()> {
        match self {
            Self::InProgress {
                k_scale,
                v_scale,
                n: _,
            } => {
                *self = Self::Done {
                    k_scale: k_scale.clone(),
                    v_scale: v_scale.clone(),
                }
            }
            Self::Done { .. } => {
                candle_core::bail!("KvScaleCalculator::finalize requires InProgress scales");
            }
        }

        Ok(())
    }

    pub fn compute_scale(x: &Tensor) -> Result<Tensor> {
        let mut absmax = x.abs()?.to_dtype(DType::F32)?;
        while !absmax.dims().is_empty() {
            absmax = absmax.max(0)?;
        }
        (absmax / 240.)?.to_dtype(DType::F32)
    }
}
