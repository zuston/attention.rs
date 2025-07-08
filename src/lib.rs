pub mod paged_attention;
use candle_core::{Device, Result, Tensor};
use paged_attention::{paged_attention, reshape_and_cache};

pub struct InputMetadata {
    pub is_prefill: bool,
    pub slot_mapping: Tensor,
    pub block_tables: Option<Tensor>,
    pub context_lens: Option<Tensor>,
    pub cu_seqlens_q: Option<Tensor>,
    pub cu_seqlens_k: Option<Tensor>,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub max_context_len: usize,
}

#[allow(dead_code)]
pub struct PagedAttention {
    num_attention_heads: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    scale: f32,
    sliding_window: Option<usize>,
    num_queries_per_kv: usize,
    alibi_slopes: Option<Tensor>,
}

impl PagedAttention {
    pub fn new(
        num_attention_heads: usize,
        head_dim: usize,
        scale: f32,
        num_key_value_heads: Option<usize>,
        sliding_window: Option<usize>,
        device: Device,
        alibi_slopes: Option<Vec<f64>>,
    ) -> Result<Self> {
        let num_key_value_heads = num_key_value_heads.unwrap_or(num_attention_heads);
        let num_queries_per_kv = num_attention_heads / num_key_value_heads;
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            Some(Tensor::new(alibi_slopes, &device)?)
        } else {
            None
        };
        Ok(Self {
            num_attention_heads,
            head_dim,
            num_key_value_heads,
            scale,
            sliding_window,
            num_queries_per_kv,
            alibi_slopes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]
    /// query: shape = [batch_size, seq_len, num_heads * head_size]
    /// key: shape = [batch_size, seq_len, num_kv_heads * head_size]
    /// value: shape = [batch_size, num_kv_heads * head_size]
    /// key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
    ///     block_size, x]
    /// value_cache: shape = [num_blocks, num_kv_heads, head_size,
    ///     block_size]
    /// input_metadata: metadata for paged attention.
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        key_cache: Option<Tensor>,
        value_cache: Option<Tensor>,
        input_metadata: &InputMetadata,
        softcapping: Option<f64>,
    ) -> Result<Tensor> {
        let dims = input_metadata.slot_mapping.dims();
        let slot_mapping = if dims.len() > 1 {
            input_metadata.slot_mapping.flatten_all()?
        } else {
            input_metadata.slot_mapping.clone()
        };

        let (batch_size, attention_heads, seq_len, head_size) = query.shape().dims4()?;
        let (_, key_value_heads, _, _) = key.shape().dims4()?;

        #[cfg(feature = "flash-attn")]
        let att = if input_metadata.is_prefill {
            let q = query
                .transpose(1, 2)?
                .reshape(((), attention_heads, head_size))?;
            let k = key
                .transpose(1, 2)?
                .reshape(((), key_value_heads, head_size))?;
            let v = value
                .transpose(1, 2)?
                .reshape(((), key_value_heads, head_size))?;

            let attn = if self.sliding_window.is_some() {
                candle_flash_attn::flash_attn_varlen_windowed_softcap(
                    &q,
                    &k,
                    &v,
                    input_metadata.cu_seqlens_q.as_ref().unwrap(),
                    input_metadata.cu_seqlens_k.as_ref().unwrap(),
                    input_metadata.max_seqlen_q,
                    input_metadata.max_seqlen_k,
                    self.scale as f32,
                    Some(softcapping.unwrap_or(0.0f64) as f32),
                    self.sliding_window,
                    Some(0),
                )?
            } else {
                candle_flash_attn::flash_attn_varlen_softcap(
                    &q,
                    &k,
                    &v,
                    input_metadata.cu_seqlens_q.as_ref().unwrap(),
                    input_metadata.cu_seqlens_k.as_ref().unwrap(),
                    input_metadata.max_seqlen_q,
                    input_metadata.max_seqlen_k,
                    self.scale as f32,
                    Some(softcapping.unwrap_or(0.0f64) as f32),
                    true,
                )?
            };
            Some(attn.transpose(1, 2)?)
        } else {
            None
        };
        fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
            if n_rep == 1 {
                Ok(x)
            } else {
                let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
                Tensor::cat(&vec![&x; n_rep], 2)?.reshape((
                    b_sz,
                    n_kv_head * n_rep,
                    seq_len,
                    head_dim,
                ))
            }
        }

        #[cfg(not(feature = "flash-attn"))]
        let att = if input_metadata.is_prefill {
            let indices = &input_metadata
                .cu_seqlens_q
                .as_ref()
                .unwrap()
                .to_vec1::<u32>()?[1..];
            let seqlens: Vec<_> = indices.iter().map(|x| x).collect();

            let mut vec_attn = Vec::new();
            let mut start = 0usize;
            //chunked attention for each sequence
            for (i, seqlen) in seqlens.iter().enumerate() {
                let seq_len = (**seqlen as usize - start) as usize;
                let chunk_size = 1024;
                let mut attn_chunks = vec![];

                let query_seq = query.narrow(2, start, seq_len)?.contiguous()?;
                let key_seq = key.narrow(2, start, seq_len)?.contiguous()?;
                let value_seq = value.narrow(2, start, seq_len)?.contiguous()?;

                let key_seq = if key_value_heads != attention_heads {
                    repeat_kv(key_seq, attention_heads / key_value_heads)?
                } else {
                    key_seq
                };

                let value_seq = if key_value_heads != attention_heads {
                    repeat_kv(value_seq, attention_heads / key_value_heads)?
                } else {
                    value_seq
                };

                let num_chunks = (seq_len + chunk_size - 1) / chunk_size;

                for c in 0..num_chunks {
                    let offset = c * chunk_size;
                    let len = chunk_size.min(seq_len - offset);
                    //chunk at query is correct for the following
                    let q_chunk = query_seq.narrow(2, offset, len)?.contiguous()?;
                    let mut att = (q_chunk.matmul(&key_seq.t()?)? * f64::from(self.scale))?;

                    if let Some(sc) = softcapping {
                        att = ((att / sc)?.tanh()? * sc)?;
                    }

                    if let Some(mask) = &attention_mask {
                        //mask needs to be chunked
                        let q_chunk_mask = mask[i].narrow(2, offset, len)?; // shape: [1, 1, chunk_len, K_len]
                        att = att.broadcast_add(&q_chunk_mask)?;
                    }

                    att =
                        candle_nn::ops::softmax_last_dim(&att.to_dtype(candle_core::DType::F32)?)?
                            .to_dtype(att.dtype())?;

                    let att_chunk = att.matmul(&value_seq)?;
                    attn_chunks.push(att_chunk);
                }

                let att = Tensor::cat(&attn_chunks, 2)?.contiguous()?;
                vec_attn.push(att);

                start = **seqlen as usize;
            }
            Some(Tensor::cat(&vec_attn, 2)?.contiguous()?)
        } else {
            None
        };

        // // paged-attn expects [batch_size, num_tokens, num_heads, head_size]
        let query = query
            .transpose(1, 2)?
            .reshape(((), attention_heads, head_size))?;
        let key = key
            .transpose(1, 2)?
            .reshape(((), key_value_heads, head_size))?;
        let value = value
            .transpose(1, 2)?
            .reshape(((), key_value_heads, head_size))?;

        // key: Tensor,              // [num_tokens, num_heads, head_size]
        // value: Tensor,            // [num_tokens, num_heads, head_size]
        // key_cache: &mut Tensor,   // [num_blocks, num_heads, head_size/x, block_size, x] 48,32,16,16,8
        // value_cache: &mut Tensor, // [num_blocks, num_heads, head_size, block_size] 48,32,128,16
        // slot_mapping: Tensor,     // [num_tokens]
        if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            reshape_and_cache(
                &key,
                &value,
                key_cache.as_ref().unwrap(),
                value_cache.as_ref().unwrap(),
                &slot_mapping,
            )?;
        }

        if let Some(att) = att {
            //prefill result
            return Ok(att.transpose(1, 2)?);
        }
        //  Args:
        //  output: shape = [num_generation_tokens, num_heads, head_size]
        //
        //  query: shape = [num_generation_tokens, num_heads, head_size]
        //
        //  key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
        //      block_size, x]
        //
        //  value_cache: shape = [num_blocks, num_kv_heads, head_size,
        //      block_size]
        //
        //  input_metadata: metadata for paged attention.
        //
        //  alibi_slopes: shape = [num_heads]
        let max_context_len = if self.sliding_window.is_some() {
            self.sliding_window.unwrap()
        } else {
            input_metadata.max_context_len
        };
        let block_tables = input_metadata.block_tables.as_ref().unwrap();
        let context_lens = input_metadata.context_lens.as_ref().unwrap();
        // println!("slot_mapping {:?}, block_tables {:?} context_lens {:?}", slot_mapping, block_tables, context_lens);

        paged_attention(
            &query,
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
            block_tables,
            context_lens,
            None,
            max_context_len,
            self.scale,
            softcapping.unwrap_or(1.0f64) as f32,
        )
    }
}
