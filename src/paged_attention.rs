use candle::backend::BackendStorage;
#[cfg(feature = "cuda")]
use candle::CudaStorage;
#[cfg(feature = "metal")]
use candle::MetalStorage;
use candle::{CpuStorage, DType, Layout, Result, Shape, Storage, Tensor};
use candle_core as candle;
#[allow(unused_imports)]
use half::{bf16, f16};
#[allow(dead_code)]
struct PagedAttention {
    softmax_scale: f32,
    softcapping: f32,
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    alibi_slopes: Option<Tensor>,
    cu_query_lens: Option<Tensor>,
    num_query_tokens: usize,
    max_context_len: usize,
    sliding_window: i32,
}

impl PagedAttention {
    #[cfg(feature = "cuda")]
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &CudaStorage,
        q_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::WrapErr;
        use std::ffi::c_int;
        let dtype = q.dtype();
        let internal_type = match dtype {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            dtype => candle::bail!("dtype {dtype:?} is not supported"),
        };

        let dev = q.device();
        let out_shape = q_l.shape().clone();

        let (kc, kc_l) = self.key_cache.storage_and_layout();
        let kc = match &*kc {
            Storage::Cuda(kc) => kc,
            _ => candle::bail!("key_cache must be a cuda tensor"),
        };

        let (vc, vc_l) = self.value_cache.storage_and_layout();
        let vc = match &*vc {
            Storage::Cuda(vc) => vc,
            _ => candle::bail!("value_cache must be a cuda tensor"),
        };

        let (bt, bt_l) = self.block_tables.storage_and_layout();
        let bt = match &*bt {
            Storage::Cuda(bt) => bt,
            _ => candle::bail!("block_tables must be a cuda tensor"),
        };

        let (cl, cl_l) = self.context_lens.storage_and_layout();
        let cl = match &*cl {
            Storage::Cuda(cl) => cl,
            _ => candle::bail!("context_lens must be a cuda tensor"),
        };

        let q_rank = q_l.stride().len();
        let kc_rank = kc_l.stride().len();
        let vc_rank = vc_l.stride().len();

        if q_rank != 3 {
            candle::bail!(
                "paged-attention expects `q` tensor to be of rank 3 \
                (q: {q_l:?})"
            )
        }

        if kc_rank != 5 {
            candle::bail!(
                "paged-attention expects `key_cache` tensor to be of rank 5 \
                (key_cache: {kc_l:?})"
            )
        }

        if vc_rank != 4 {
            candle::bail!(
                "paged-attention expects `value_cache` tensor to be of rank 4 \
                (value_cache: {vc_l:?})"
            )
        }

        // Get cuda slices for all tensors
        let q = q.as_cuda_slice::<T>()?;
        let kc = kc.as_cuda_slice::<T>()?;
        let vc = vc.as_cuda_slice::<T>()?;
        let cl = cl.as_cuda_slice::<u32>()?; // Should be i32!
        let bt = bt.as_cuda_slice::<u32>()?; // Should be i32!

        // Get cuda views for all tensors
        let q = q.slice(q_l.start_offset()..);
        let kc = kc.slice(kc_l.start_offset()..);
        let vc = vc.slice(vc_l.start_offset()..);
        let cl = cl.slice(cl_l.start_offset()..);
        let bt = bt.slice(bt_l.start_offset()..);

        let (num_seqs, num_heads, head_size) = q_l.shape().dims3()?;
        if !(head_size == 64
            || head_size == 80
            || head_size == 96
            || head_size == 112
            || head_size == 128
            || head_size == 192
            || head_size == 256)
        {
            candle::bail!("`head_size` must be one of 64, 80, 96, 112, 128, 192 or 256");
        }

        let (num_seqs_bt, max_num_blocks_per_seq) = bt_l.shape().dims2()?;

        if num_seqs_bt != num_seqs && self.cu_query_lens.as_ref().is_none() {
            candle::bail!(
                "shape mismatch block_tables {:?}, expected {:?}",
                bt_l.shape(),
                (num_seqs, max_num_blocks_per_seq)
            )
        }

        let (num_blocks, num_kv_heads, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
        if head_size_kc != head_size / x {
            candle::bail!(
                "shape mismatch value_cache {:?}, expected {:?}",
                vc_l.shape(),
                (num_blocks, num_heads, head_size / x, block_size, x)
            )
        }

        if (num_blocks, num_kv_heads, head_size, block_size) != vc_l.shape().dims4()? {
            candle::bail!(
                "shape mismatch key_cache {:?} and value_cache {:?}",
                kc_l.shape(),
                vc_l.shape()
            )
        }

        if (num_seqs) != cl_l.shape().dims1()? && self.cu_query_lens.as_ref().is_none() {
            candle::bail!(
                "shape mismatch context_lens {:?}, expected {:?}",
                cl_l.shape(),
                (num_seqs)
            )
        }

        let q_stride = q_l.stride()[0];
        let kv_block_stride = kc_l.stride()[0];
        let kv_head_stride = kc_l.stride()[1];

        let partition_size = 512;
        let max_num_partitions = (self.max_context_len + partition_size - 1) / partition_size;
        let use_v1 = (max_num_partitions == 1 || num_seqs * num_heads > 512)
            && partition_size % block_size == 0;

        let elem_count = out_shape.elem_count();
        let out = unsafe { dev.alloc::<T>(elem_count) }.w()?;

        let out_ptr = *out.device_ptr() as *const core::ffi::c_void;
        let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
        let kc_ptr = *kc.device_ptr() as *const core::ffi::c_void;
        let vc_ptr = *vc.device_ptr() as *const core::ffi::c_void;
        let bt_ptr = *bt.device_ptr() as *const core::ffi::c_int;
        let cl_ptr = *cl.device_ptr() as *const core::ffi::c_int;

        if let Some(cu_query_lens) = &self.cu_query_lens {
            let o_stride_tokens = q_l.stride()[0] as i32;
            let sinks_ptr = std::ptr::null();

            let query_start_len_ptr = {
                let (cu_query_lens, cu_query_lens_layout) = cu_query_lens.storage_and_layout();
                let cu_query_lens = match &*cu_query_lens {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                    _ => candle::bail!("cu_query_lens must be a cuda tensor"),
                };
                let cu_query_lens = cu_query_lens.slice(cu_query_lens_layout.start_offset()..);
                *cu_query_lens.device_ptr()
            };

            unsafe {
                kernels::ffi::paged_attention_prefill(
                    out_ptr,
                    q_ptr,
                    kc_ptr,
                    vc_ptr,
                    num_kv_heads as c_int,
                    self.softmax_scale,
                    bt_ptr,
                    cl_ptr,
                    block_size as c_int,
                    self.max_context_len as c_int,
                    num_seqs_bt as c_int,
                    num_heads as c_int,
                    self.num_query_tokens as c_int,
                    head_size as c_int,
                    max_num_blocks_per_seq as c_int,
                    q_stride as c_int,
                    num_blocks as c_int,
                    kv_block_stride as c_int,
                    kv_head_stride as c_int,
                    internal_type,
                    self.softcapping,
                    o_stride_tokens as c_int,
                    query_start_len_ptr as *const u32,
                    sinks_ptr as *const f32,
                    self.sliding_window as c_int,
                    *dev.cu_stream() as i64,
                )
            }
        } else if use_v1 {
            unsafe {
                kernels::ffi::paged_attention_v1(
                    out_ptr,
                    q_ptr,
                    kc_ptr,
                    vc_ptr,
                    num_kv_heads as c_int,
                    self.softmax_scale,
                    bt_ptr,
                    cl_ptr,
                    block_size as c_int,
                    self.max_context_len as c_int,
                    num_seqs as c_int,
                    num_heads as c_int,
                    head_size as c_int,
                    max_num_blocks_per_seq as c_int,
                    q_stride as c_int,
                    kv_block_stride as c_int,
                    kv_head_stride as c_int,
                    internal_type,
                    self.softcapping,
                    *dev.cu_stream() as i64,
                )
            }
        } else {
            let tmp_out_shape = Shape::from((num_seqs, num_heads, max_num_partitions, head_size));
            let exp_sums_shape = Shape::from((num_seqs, num_heads, max_num_partitions));
            let tmp_out = unsafe { dev.alloc::<T>(tmp_out_shape.elem_count()) }.w()?;
            let exp_sums = unsafe { dev.alloc::<f32>(exp_sums_shape.elem_count()) }.w()?;
            let max_logits = unsafe { dev.alloc::<f32>(exp_sums_shape.elem_count()) }.w()?;

            let tmp_out_ptr = *tmp_out.device_ptr() as *const core::ffi::c_void;
            let exp_sums_ptr = *exp_sums.device_ptr() as *const f32;
            let max_logits_ptr = *max_logits.device_ptr() as *const f32;

            unsafe {
                kernels::ffi::paged_attention_v2(
                    out_ptr,
                    exp_sums_ptr,
                    max_logits_ptr,
                    tmp_out_ptr,
                    q_ptr,
                    kc_ptr,
                    vc_ptr,
                    num_kv_heads as c_int,
                    self.softmax_scale,
                    bt_ptr,
                    cl_ptr,
                    block_size as c_int,
                    self.max_context_len as c_int,
                    num_seqs as c_int,
                    num_heads as c_int,
                    head_size as c_int,
                    max_num_blocks_per_seq as c_int,
                    q_stride as c_int,
                    kv_block_stride as c_int,
                    kv_head_stride as c_int,
                    internal_type,
                    self.softcapping,
                    *dev.cu_stream() as i64,
                )
            }
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, out_shape))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd_t(&self, q: &MetalStorage, q_l: &Layout) -> Result<(MetalStorage, Shape)> {
        let dtype = q.dtype();
        let internal_type = match dtype {
            DType::F16 => metal_kernels::PagedAttentionDType::F16,
            DType::BF16 => metal_kernels::PagedAttentionDType::BF16,
            DType::F32 => metal_kernels::PagedAttentionDType::F32,
            dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
        };

        let dev = q.device();
        let out_shape = q_l.shape().clone();

        let (kc, kc_l) = self.key_cache.storage_and_layout();
        let kc = match &*kc {
            Storage::Metal(kc) => kc,
            _ => candle_core::bail!("key_cache must be a metal tensor"),
        };

        let (vc, vc_l) = self.value_cache.storage_and_layout();
        let vc = match &*vc {
            Storage::Metal(vc) => vc,
            _ => candle_core::bail!("value_cache must be a metal tensor"),
        };

        let (bt, bt_l) = self.block_tables.storage_and_layout();
        let bt = match &*bt {
            Storage::Metal(bt) => bt,
            _ => candle_core::bail!("block_tables must be a metal tensor"),
        };

        let (cl, cl_l) = self.context_lens.storage_and_layout();
        let cl = match &*cl {
            Storage::Metal(cl) => cl,
            _ => candle_core::bail!("context_lens must be a metal tensor"),
        };

        let q_rank = q_l.stride().len();
        let kc_rank = kc_l.stride().len();
        let vc_rank = vc_l.stride().len();

        if q_rank != 3 {
            candle_core::bail!(
                "paged-attention expects `q` tensor to be of rank 3 \
                (q: {q_l:?})"
            )
        }

        if kc_rank != 5 {
            candle_core::bail!(
                "paged-attention expects `key_cache` tensor to be of rank 5 \
                (key_cache: {kc_l:?})"
            )
        }

        if vc_rank != 4 {
            candle_core::bail!(
                "paged-attention expects `value_cache` tensor to be of rank 4 \
                (value_cache: {vc_l:?})"
            )
        }

        let alibi_storage_and_offset = if let Some(alibi_slopes) = self.alibi_slopes.as_ref() {
            let (alibi_s, alibi_s_l) = alibi_slopes.storage_and_layout();
            let alibi_s = match &*alibi_s {
                Storage::Metal(alibi_s) => alibi_s,
                _ => candle_core::bail!("alibi_slopes must be a metal tensor"),
            };
            Some((
                alibi_s.clone(),
                alibi_s_l.start_offset() * alibi_s.dtype().size_in_bytes(),
            ))
        } else {
            None
        };

        let (num_seqs, num_heads, head_size) = q_l.shape().dims3()?;
        if !(head_size == 64
            || head_size == 80
            || head_size == 96
            || head_size == 112
            || head_size == 128
            || head_size == 256)
        {
            candle_core::bail!("`head_size` must be one of 64, 80, 96, 112, 128 or 256");
        }

        let (num_seqs_bt, max_num_blocks_per_seq) = bt_l.shape().dims2()?;

        if num_seqs_bt != num_seqs && self.cu_query_lens.as_ref().is_none() {
            candle_core::bail!(
                "shape mismatch block_tables {:?}, expected {:?}",
                bt_l.shape(),
                (num_seqs, max_num_blocks_per_seq)
            )
        }

        let (num_blocks, num_kv_heads, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
        if head_size_kc != head_size / x {
            candle_core::bail!(
                "shape mismatch value_cache {:?}, expected {:?}",
                vc_l.shape(),
                (num_blocks, num_heads, head_size / x, block_size, x)
            )
        }

        if (num_blocks, num_kv_heads, head_size, block_size) != vc_l.shape().dims4()? {
            candle_core::bail!(
                "shape mismatch key_cache {:?} and value_cache {:?}",
                kc_l.shape(),
                vc_l.shape()
            )
        }

        if (num_seqs) != cl_l.shape().dims1()? && self.cu_query_lens.as_ref().is_none() {
            candle_core::bail!(
                "shape mismatch context_lens {:?}, expected {:?}",
                cl_l.shape(),
                (num_seqs)
            )
        }

        let q_stride = q_l.stride()[0];
        let kv_block_stride = kc_l.stride()[0];
        let kv_head_stride = kc_l.stride()[1];

        let partition_size = 512;
        let max_num_partitions = self.max_context_len.div_ceil(partition_size);
        let use_v1 = (max_num_partitions == 1 || num_seqs * num_heads > 512)
            && partition_size % block_size == 0;

        let elem_count = out_shape.elem_count();

        let command_buffer = dev.command_buffer()?;
        command_buffer.set_label("paged-attention");

        let out = dev.new_buffer(elem_count, dtype, "paged-attention-out")?;

        if let Some(cu_query_lens) = &self.cu_query_lens {
            let o_stride_tokens = q_l.stride()[0] as i32;

            let (qs, qs_l) = {
                let (cu_query_lens, cu_query_lens_layout) = cu_query_lens.storage_and_layout();
                let qs = match &*cu_query_lens {
                    candle::Storage::Metal(c) => c,
                    _ => candle::bail!("cu_query_lens must be a metal tensor"),
                };
                (qs.clone(), cu_query_lens_layout)
            };

            metal_kernels::paged_attention_prefill(
                dev.device(),
                &command_buffer,
                metal_kernels::Kernels::default(),
                internal_type,
                &out,
                q.buffer(),
                q_l.start_offset() * q.dtype().size_in_bytes(),
                kc.buffer(),
                kc_l.start_offset() * kc.dtype().size_in_bytes(),
                vc.buffer(),
                vc_l.start_offset() * vc.dtype().size_in_bytes(),
                bt.buffer(),
                bt_l.start_offset() * bt.dtype().size_in_bytes(),
                cl.buffer(),
                cl_l.start_offset() * cl.dtype().size_in_bytes(),
                qs.buffer(),
                qs_l.start_offset() * qs.dtype().size_in_bytes(),
                alibi_storage_and_offset,
                None,
                num_kv_heads as i32,
                self.softmax_scale,
                max_num_blocks_per_seq as i32,
                num_seqs_bt as i32,
                num_heads as i32,
                self.num_query_tokens as i32,
                head_size as i32,
                block_size as i32,
                self.softcapping,
                o_stride_tokens as i32,
                self.sliding_window as i32,
                num_blocks as i32,
                kv_block_stride as i32,
                kv_head_stride as i32,
            )
            .map_err(candle_core::Error::wrap)?;
        } else if use_v1 {
            metal_kernels::paged_attention_v1(
                dev.device(),
                &command_buffer,
                metal_kernels::Kernels::default(),
                internal_type,
                q.buffer(),
                q_l.start_offset() * q.dtype().size_in_bytes(),
                kc.buffer(),
                kc_l.start_offset() * kc.dtype().size_in_bytes(),
                vc.buffer(),
                vc_l.start_offset() * vc.dtype().size_in_bytes(),
                bt.buffer(),
                bt_l.start_offset() * bt.dtype().size_in_bytes(),
                cl.buffer(),
                cl_l.start_offset() * cl.dtype().size_in_bytes(),
                alibi_storage_and_offset,
                &out,
                num_kv_heads as i32,
                self.softmax_scale,
                self.softcapping,
                block_size as i32,
                self.max_context_len as i32,
                num_seqs as i32,
                num_heads as i32,
                head_size as i32,
                max_num_blocks_per_seq as i32,
                q_stride as i32,
                kv_block_stride as i32,
                kv_head_stride as i32,
            )
            .map_err(candle_core::Error::wrap)?;
        } else {
            let tmp_out_shape = Shape::from((num_seqs, num_heads, max_num_partitions, head_size));
            let exp_sums_shape = Shape::from((num_seqs, num_heads, max_num_partitions));
            let tmp_out =
                dev.new_buffer(tmp_out_shape.elem_count(), dtype, "paged-attention-tmpout")?;
            let exp_sums = dev.new_buffer(
                exp_sums_shape.elem_count(),
                DType::F32,
                "paged-attention-expsums",
            )?;
            let max_logits = dev.new_buffer(
                exp_sums_shape.elem_count(),
                DType::F32,
                "paged-attention-maxlogits",
            )?;

            metal_kernels::paged_attention_v2(
                dev.device(),
                &command_buffer,
                metal_kernels::Kernels::default(),
                internal_type,
                &exp_sums,
                &max_logits,
                q.buffer(),
                q_l.start_offset() * q.dtype().size_in_bytes(),
                kc.buffer(),
                kc_l.start_offset() * kc.dtype().size_in_bytes(),
                vc.buffer(),
                vc_l.start_offset() * vc.dtype().size_in_bytes(),
                bt.buffer(),
                bt_l.start_offset() * bt.dtype().size_in_bytes(),
                cl.buffer(),
                cl_l.start_offset() * cl.dtype().size_in_bytes(),
                alibi_storage_and_offset,
                &tmp_out,
                &out,
                num_kv_heads as i32,
                self.softmax_scale,
                self.softcapping,
                block_size as i32,
                self.max_context_len as i32,
                num_seqs as i32,
                num_heads as i32,
                head_size as i32,
                max_num_blocks_per_seq as i32,
                q_stride as i32,
                kv_block_stride as i32,
                kv_head_stride as i32,
            )
            .map_err(candle_core::Error::wrap)?;
        }

        let newstorage =
            candle_core::MetalStorage::new(out, q.device().clone(), elem_count, q.dtype());
        Ok((newstorage, out_shape))
    }
}

impl candle::CustomOp1 for PagedAttention {
    fn name(&self) -> &'static str {
        "paged-attention"
    }

    fn cpu_fwd(&self, q: &CpuStorage, q_l: &Layout) -> Result<(CpuStorage, Shape)> {
        use candle::Storage;

        let q_slice = q.as_slice::<f32>()?;
        let (kc_s, kc_l) = self.key_cache.storage_and_layout();
        let kc_slice = match &*kc_s {
            Storage::Cpu(cpu) => cpu.as_slice::<f32>()?,
            _ => candle::bail!("key_cache must be a cpu tensor"),
        };
        let (vc_s, vc_l) = self.value_cache.storage_and_layout();
        let vc_slice = match &*vc_s {
            Storage::Cpu(cpu) => cpu.as_slice::<f32>()?,
            _ => candle::bail!("value_cache must be a cpu tensor"),
        };
        let (bt_s, bt_l) = self.block_tables.storage_and_layout();
        let bt_slice = match &*bt_s {
            Storage::Cpu(cpu) => cpu.as_slice::<u32>()?,
            _ => candle::bail!("block_tables must be a cpu tensor"),
        };
        let (cl_s, cl_l) = self.context_lens.storage_and_layout();
        let cl_slice = match &*cl_s {
            Storage::Cpu(cpu) => cpu.as_slice::<u32>()?,
            _ => candle::bail!("context_lens must be a cpu tensor"),
        };

        // Validate shapes
        if q_l.stride().len() != 3 {
            candle::bail!("q must be rank 3");
        }
        if kc_l.stride().len() != 5 {
            candle::bail!("key_cache must be rank 5");
        }
        if vc_l.stride().len() != 4 {
            candle::bail!("value_cache must be rank 4");
        }

        let (num_seqs, num_heads, head_size) = q_l.shape().dims3()?;
        let (num_blocks, num_kv_heads, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
        if head_size_kc * x != head_size {
            candle::bail!("key_cache head_size mismatch");
        }
        let (_, _, _, vc_block_size) = vc_l.shape().dims4()?;
        if vc_block_size != block_size {
            candle::bail!("value_cache block_size mismatch");
        }

        let mut out = vec![0f32; q_slice.len()];

        for s in 0..num_seqs {
            let context_len = cl_slice[s] as usize;
            let bt_row_start = s * (bt_l.shape().dims2()?.1);
            let bt_row = &bt_slice[bt_row_start..bt_row_start + context_len];

            for h in 0..num_heads {
                let q_offset = s * num_heads * head_size + h * head_size;
                let q_vec = &q_slice[q_offset..q_offset + head_size];

                // Collect logits and corresponding value vectors
                let mut logits = Vec::with_capacity(context_len);
                let mut v_values = Vec::with_capacity(context_len * head_size);

                for t in 0..context_len {
                    let block_id = bt_row[t / block_size] as usize;
                    let offset_in_block = t % block_size;

                    let k_offset = ((((block_id * num_heads + h) * head_size_kc)
                        * block_size + offset_in_block) * x) as usize;
                    let v_offset = (((block_id * num_heads + h) * head_size * block_size)
                        + offset_in_block * head_size) as usize;

                    let mut dot = 0f32;
                    for d in 0..head_size {
                        dot += q_vec[d] * kc_slice[k_offset + d];
                    }
                    logits.push(dot * self.softmax_scale);

                    v_values.extend_from_slice(&vc_slice[v_offset..v_offset + head_size]);
                }

                // Softmax
                let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
                let weights: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp() / exp_sum).collect();

                // Weighted sum
                for d in 0..head_size {
                    let mut acc = 0f32;
                    for t in 0..context_len {
                        acc += weights[t] * v_values[t * head_size + d];
                    }
                    out[q_offset + d] = acc;
                }
            }
        }

        Ok((CpuStorage::F32(out), q_l.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, q: &CudaStorage, q_l: &Layout) -> Result<(CudaStorage, Shape)> {
        match q.dtype() {
            DType::F32 => self.cuda_fwd_t::<f32>(q, q_l),
            DType::F16 => self.cuda_fwd_t::<f16>(q, q_l),
            DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l),
            dt => candle::bail!("paged-attention is only supported for f32/f16/bf16 ({dt:?})"),
        }
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(&self, q: &MetalStorage, q_l: &Layout) -> Result<(MetalStorage, Shape)> {
        match q.dtype() {
            DType::F32 | DType::F16 | DType::BF16 => self.metal_fwd_t(q, q_l),
            dt => candle::bail!("paged-attention is only supported for f32/f16/bf16 ({dt:?})"),
        }
    }
}

/// Paged Attention layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors key_cache and value_cache
/// with fewer heads than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(num_sequences, num_heads_q, head_size)`.
/// * `key_cache` - Key cache paged tensor of shape `(num_blocks, num_heads_kv, head_size / x, block_size, x)`
///   with `x` being the size of an element in bytes.
/// * `value_cache` - Value cache paged tensor of shape `(num_blocks, num_heads_kv, head_size, block_size)`.
/// * `block_tables` - Padded table associating blocks to each sequence of shape `(num_sequences, max_context_len // block_size)`
/// * `context_lens` - Tensor associating lengths to each sequence of shape `(num_sequences)`
/// * `max_context_len` - Max of `context_len`
/// * `softmax_scale` - scaling factor
///
/// The resulting tensor has dimensions `(num_sequences, num_heads_q, head_size)`.
pub fn paged_attention(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
    max_context_len: usize,
    softmax_scale: f32,
    softcapping: f32,
    cu_query_lens: Option<Tensor>,
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    let sliding_window = if let Some(sliding_window) = sliding_window {
        sliding_window as i32
    } else {
        -1
    };
    let num_query_tokens = q.dim(0)?;
    let op = PagedAttention {
        softmax_scale,
        key_cache: key_cache.to_owned(),
        value_cache: value_cache.to_owned(),
        block_tables: block_tables.to_owned(),
        context_lens: context_lens.to_owned(),
        alibi_slopes: alibi_slopes.to_owned().cloned(),
        max_context_len,
        softcapping,
        cu_query_lens: cu_query_lens.to_owned(),
        num_query_tokens,
        sliding_window,
    };
    q.apply_op1(op)
}

struct ReshapeCache {
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
}

impl ReshapeCache {
    #[cfg(feature = "cuda")]
    pub fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        k: &CudaStorage,
        k_l: &Layout,
        value: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        use std::ffi::c_int;
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        let dtype = k.dtype();
        let dev = k.device();
        let internal_type = match dtype {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            dtype => candle::bail!("dtype {dtype:?} is not supported"),
        };

        let (v, v_l) = value.storage_and_layout();
        let v = match &*v {
            Storage::Cuda(v) => v,
            _ => candle::bail!("value must be a cuda tensor"),
        };

        let (kc, kc_l) = key_cache.storage_and_layout();
        let kc = match &*kc {
            Storage::Cuda(kc) => kc,
            _ => candle::bail!("key_cache must be a cuda tensor"),
        };

        let (vc, vc_l) = value_cache.storage_and_layout();
        let vc = match &*vc {
            Storage::Cuda(vc) => vc,
            _ => candle::bail!("value_cache must be a cuda tensor"),
        };

        let (s, s_l) = slot_mapping.storage_and_layout();
        let s = match &*s {
            Storage::Cuda(s) => s,
            _ => candle::bail!("slot_mapping must be a cuda tensor"),
        };

        let k_rank = k_l.stride().len();
        let v_rank = v_l.stride().len();
        let kc_rank = kc_l.stride().len();
        let vc_rank = vc_l.stride().len();

        if k_rank != 3 || v_rank != 3 {
            candle::bail!(
                "paged-attention expects input tensors of rank 3 (k: {k_l:?}, v: {v_l:?})"
            )
        }

        #[cfg(feature = "flash-decoding")]
        if kc_rank != 4 {
            candle::bail!(
                "flash-attention expects `key_cache` tensor to be of rank 4 \
                    (key_cache: {kc_l:?})"
            )
        }

        #[cfg(not(feature = "flash-decoding"))]
        if kc_rank != 5 {
            candle::bail!(
                "paged-attention expects `key_cache` tensor to be of rank 5 \
                    (key_cache: {kc_l:?})"
            )
        }

        if vc_rank != 4 {
            candle::bail!(
                "paged-attention expects `value_cache` tensor to be of rank 4 \
                    (value_cache: {vc_l:?})"
            )
        }

        // Get cuda slices for all tensors
        let k = k.as_cuda_slice::<T>()?;
        let v = v.as_cuda_slice::<T>()?;
        let kc = kc.as_cuda_slice::<T>()?;
        let vc = vc.as_cuda_slice::<T>()?;
        let s = s.as_cuda_slice::<i64>()?;

        // Get cuda views for all tensors
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);
        let kc = kc.slice(kc_l.start_offset()..);
        let vc = vc.slice(vc_l.start_offset()..);
        let s = s.slice(s_l.start_offset()..);

        let (num_tokens, num_heads, head_size) = k_l.shape().dims3()?;
        if (num_tokens, num_heads, head_size) != v_l.shape().dims3()? {
            candle::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
        }

        #[cfg(feature = "flash-decoding")]
        let (block_size, _x) = {
            // [num_blocks, block_size, num_heads, head_size]
            let (_, block_size, _, _) = kc_l.shape().dims4()?;
            (block_size, 1)
        };

        #[cfg(not(feature = "flash-decoding"))]
        let (block_size, x) = {
            let (num_blocks, num_heads_kc, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
            if num_heads_kc != num_heads || head_size_kc != head_size / x {
                candle::bail!(
                    "shape mismatch value_cache {:?}, expected {:?}",
                    vc_l.shape(),
                    (num_blocks, num_heads, head_size / x, block_size, x)
                )
            }

            if (num_blocks, num_heads, head_size, block_size) != vc_l.shape().dims4()? {
                candle::bail!(
                    "shape mismatch key_cache {:?} and value_cache {:?}",
                    kc_l.shape(),
                    vc_l.shape()
                )
            }
            (block_size, x)
        };

        if (num_tokens) != s_l.shape().dims1()? {
            candle::bail!(
                "shape mismatch slot_mapping {:?}, expected {:?}",
                s_l.shape(),
                (num_tokens)
            )
        }

        let key_stride = k_l.stride()[0] as c_int;
        let value_stride = v_l.stride()[0] as c_int;

        let k_ptr = *k.device_ptr() as *const core::ffi::c_void;
        let v_ptr = *v.device_ptr() as *const core::ffi::c_void;
        let kc_ptr = *kc.device_ptr() as *const core::ffi::c_void;
        let vc_ptr = *vc.device_ptr() as *const core::ffi::c_void;
        let s_ptr = *s.device_ptr() as *const core::ffi::c_long;
        unsafe {
            #[cfg(feature = "flash-decoding")]
            {
                let block_stride = kc_l.stride()[0];
                let page_stride = kc_l.stride()[1];
                let head_stride = kc_l.stride()[2];

                kernels::ffi::call_reshape_and_cache_flash(
                    k_ptr,  // [num_tokens, num_heads, head_size]
                    v_ptr,  // [num_tokens, num_heads, head_size]
                    kc_ptr, // [num_blocks, block_size, num_heads, head_size]
                    vc_ptr, // [num_blocks, block_size, num_heads, head_size]
                    s_ptr,  // [num_tokens]
                    num_tokens as c_int,
                    num_heads as c_int,
                    head_size as c_int,
                    block_size as c_int,
                    key_stride,
                    value_stride,
                    block_stride as c_int,
                    page_stride as c_int,
                    head_stride as c_int,
                    internal_type,
                    *dev.cu_stream() as i64,
                );
            }
            #[cfg(not(feature = "flash-decoding"))]
            {
                kernels::ffi::call_reshape_and_cache(
                    k_ptr,
                    v_ptr,
                    kc_ptr,
                    vc_ptr,
                    s_ptr,
                    num_tokens as c_int,
                    num_heads as c_int,
                    head_size as c_int,
                    block_size as c_int,
                    x as c_int,
                    key_stride,
                    value_stride,
                    internal_type,
                    *dev.cu_stream() as i64,
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    pub fn metal_fwd_t(
        &self,
        k: &MetalStorage,
        k_l: &Layout,
        value: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        let dtype = k.dtype();
        let internal_type = match dtype {
            DType::F16 => metal_kernels::PagedAttentionDType::F16,
            DType::BF16 => metal_kernels::PagedAttentionDType::BF16,
            DType::F32 => metal_kernels::PagedAttentionDType::F32,
            dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
        };

        let (v, v_l) = value.storage_and_layout();
        let v = match &*v {
            Storage::Metal(v) => v,
            _ => candle_core::bail!("value must be a metal tensor"),
        };

        let (kc, kc_l) = key_cache.storage_and_layout();
        let kc = match &*kc {
            Storage::Metal(kc) => kc,
            _ => candle_core::bail!("key_cache must be a metal tensor"),
        };

        let (vc, vc_l) = value_cache.storage_and_layout();
        let vc = match &*vc {
            Storage::Metal(vc) => vc,
            _ => candle_core::bail!("value_cache must be a metal tensor"),
        };

        let (s, s_l) = slot_mapping.storage_and_layout();
        let s = match &*s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("slot_mapping must be a metal tensor"),
        };

        let k_rank = k_l.stride().len();
        let v_rank = v_l.stride().len();
        let kc_rank = kc_l.stride().len();
        let vc_rank = vc_l.stride().len();

        if k_rank != 3 || v_rank != 3 {
            candle_core::bail!(
                "paged-attention expects input tensors of rank 3 (k: {k_l:?}, v: {v_l:?})"
            )
        }

        if kc_rank != 5 {
            candle_core::bail!(
                "paged-attention expects `key_cache` tensor to be of rank 5 \
                    (key_cache: {kc_l:?})"
            )
        }

        if vc_rank != 4 {
            candle_core::bail!(
                "paged-attention expects `value_cache` tensor to be of rank 4 \
                    (value_cache: {vc_l:?})"
            )
        }

        let (num_tokens, num_heads, head_size) = k_l.shape().dims3()?;
        if (num_tokens, num_heads, head_size) != v_l.shape().dims3()? {
            candle_core::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
        }

        let (num_blocks, num_heads_kc, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
        if num_heads_kc != num_heads || head_size_kc != head_size / x {
            candle_core::bail!(
                "shape mismatch value_cache {:?}, expected {:?}",
                vc_l.shape(),
                (num_blocks, num_heads, head_size / x, block_size, x)
            )
        }

        if (num_blocks, num_heads, head_size, block_size) != vc_l.shape().dims4()? {
            candle_core::bail!(
                "shape mismatch key_cache {:?} and value_cache {:?}",
                kc_l.shape(),
                vc_l.shape()
            )
        }

        if (num_tokens) != s_l.shape().dims1()? {
            candle_core::bail!(
                "shape mismatch slot_mapping {:?}, expected {:?}",
                s_l.shape(),
                (num_tokens)
            )
        }

        let key_stride = k_l.stride()[0] as i32;
        let value_stride = v_l.stride()[0] as i32;

        let dev = k.device();

        let command_buffer = dev.command_buffer()?;
        command_buffer.set_label("reshape-and-cache");

        metal_kernels::call_reshape_and_cache(
            dev.device(),
            &command_buffer,
            metal_kernels::Kernels::default(),
            internal_type,
            k.buffer(),
            k_l.start_offset() * k.dtype().size_in_bytes(),
            v.buffer(),
            v_l.start_offset() * value.dtype().size_in_bytes(),
            kc.buffer(),
            kc_l.start_offset() * key_cache.dtype().size_in_bytes(),
            vc.buffer(),
            vc_l.start_offset() * value_cache.dtype().size_in_bytes(),
            s.buffer(),
            s_l.start_offset() * slot_mapping.dtype().size_in_bytes(),
            num_tokens as i32,
            num_heads as i32,
            head_size as i32,
            block_size as i32,
            x as i32,
            key_stride,
            value_stride,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok(())
    }
}

impl candle::InplaceOp1 for ReshapeCache {
    fn name(&self) -> &'static str {
        "reshape-cache"
    }

    fn cpu_fwd(&self, k: &mut CpuStorage, k_l: &Layout) -> Result<()> {
        use candle::{Storage, Layout, Result};
        // Extract slices for input and caches
        // Only f32 is supported for now (could add f16/bf16 if needed)
        let k_slice = k.as_slice::<f32>()?;
        let (v_s, v_l) = self.value.storage_and_layout();
        let v_slice = match &*v_s {
            Storage::Cpu(vcpu) => vcpu.as_slice::<f32>()?,
            _ => candle::bail!("value must be a cpu tensor"),
        };
        let (kc_s, kc_l) = self.key_cache.storage_and_layout();
        let kc_slice = match &*kc_s {
            Storage::Cpu(kccpu) => kccpu.as_slice::<f32>()?,
            _ => candle::bail!("key_cache must be a cpu tensor"),
        };
        let (vc_s, vc_l) = self.value_cache.storage_and_layout();
        let vc_slice = match &*vc_s {
            Storage::Cpu(vccpu) => vccpu.as_slice::<f32>()?,
            _ => candle::bail!("value_cache must be a cpu tensor"),
        };
        let (s_s, s_l) = self.slot_mapping.storage_and_layout();
        let slots = match &*s_s {
            Storage::Cpu(scpu) => scpu.as_slice::<i64>()?,
            _ => candle::bail!("slot_mapping must be a cpu tensor"),
        };

        let (num_tokens, num_heads, head_size) = k_l.shape().dims3()?;
        if (num_tokens, num_heads, head_size) != v_l.shape().dims3()? {
            candle::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
        }
        let (num_blocks, num_heads_kc, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
        if num_heads_kc != num_heads || head_size_kc * x != head_size {
            candle::bail!("key_cache shape mismatch");
        }
        if (num_blocks, num_heads, head_size, block_size) != vc_l.shape().dims4()? {
            candle::bail!("value_cache shape mismatch");
        }
        if slots.len() != num_tokens {
            candle::bail!("slot_mapping length mismatch");
        }

        // Copy data token by token
        for t in 0..num_tokens {
            let slot = slots[t] as usize;
            if slot >= num_blocks * block_size {
                candle::bail!("slot_mapping index {} out of bounds (max {})", slot, num_blocks * block_size - 1);
            }
            let block_idx = slot / block_size;
            let offset_in_block = slot % block_size;
            for h in 0..num_heads {
                let src_offset = t * num_heads * head_size + h * head_size;
                // Key cache: [num_blocks, num_heads, head_size/x, block_size, x]
                let dst_k_offset = ((((block_idx * num_heads + h) * head_size_kc)
                                     * block_size + offset_in_block) * x) as usize;
                // Copy key from input to key_cache
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        k_slice[src_offset..src_offset + head_size].as_ptr(),
                        kc_slice.as_ptr().add(dst_k_offset) as *mut f32,
                        head_size,
                    );
                }
                // Value cache: [num_blocks, num_heads, head_size, block_size]
                let dst_v_offset = (((block_idx * num_heads + h) * head_size * block_size)
                                    + offset_in_block * head_size) as usize;
                // Copy value from input to value_cache
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        v_slice[src_offset..src_offset + head_size].as_ptr(),
                        vc_slice.as_ptr().add(dst_v_offset) as *mut f32,
                        head_size,
                    );
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, k: &mut CudaStorage, k_l: &Layout) -> Result<()> {
        match k.dtype() {
            DType::F32 => self.cuda_fwd_t::<f32>(
                k,
                k_l,
                &self.value,
                &self.key_cache,
                &self.value_cache,
                &self.slot_mapping,
            ),
            DType::F16 => self.cuda_fwd_t::<f16>(
                k,
                k_l,
                &self.value,
                &self.key_cache,
                &self.value_cache,
                &self.slot_mapping,
            ),
            DType::BF16 => self.cuda_fwd_t::<bf16>(
                k,
                k_l,
                &self.value,
                &self.key_cache,
                &self.value_cache,
                &self.slot_mapping,
            ),
            dt => candle::bail!("reshape-cache is only supported for f32/f16/bf16 ({dt:?})"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, k: &mut MetalStorage, k_l: &Layout) -> Result<()> {
        match k.dtype() {
            DType::F32 | DType::F16 | DType::BF16 => self.metal_fwd_t(
                k,
                k_l,
                &self.value,
                &self.key_cache,
                &self.value_cache,
                &self.slot_mapping,
            ),
            dt => candle::bail!("reshape-cache is only supported for f32/f16/bf16 ({dt:?})"),
        }
    }
}
/// Insert key and values at the provided slot mapping inside the key value paged cache
///
/// # Arguments
///
/// * `key` - Key tensor of shape `(num_tokens, num_heads, head_size)`.
/// * `value` - Value tensor of shape `(num_tokens, num_heads, head_size)`.
/// * `key_cache` - Key cache paged tensor of shape `(num_blocks, num_heads, head_size / x, block_size, x)`
///   with `x` being the size of an element in bytes.
/// * `value_cache` - Value cache paged tensor of shape `(num_blocks, num_heads, head_size, block_size)`.
/// * `slot_mapping` - Mapping associating a slot to each token of shape `(num_tokens)`.
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let op = ReshapeCache {
        value: value.to_owned(),
        key_cache: key_cache.to_owned(),
        value_cache: value_cache.to_owned(),
        slot_mapping: slot_mapping.to_owned(),
    };
    key.inplace_op1(&op)
}
