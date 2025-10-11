use core::ffi::{c_int, c_long, c_void};
#[allow(dead_code)]
extern "C" {
    pub fn call_reshape_and_cache(
        key: *const c_void,
        value: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        k_scale: *const c_void,
        v_scale: *const c_void,
        slot_mapping: *const c_long,

        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        block_size: c_int,
        x: c_int,
        key_stride: c_int,
        value_stride: c_int,
        dtype: u32,
        stream: i64,
    );

    pub fn call_reshape_and_cache_flash(
        key: *const c_void,         // [num_tokens, num_heads, head_size]
        value: *const c_void,       // [num_tokens, num_heads, head_size]
        key_cache: *const c_void,   // [num_blocks, block_size, num_heads, head_size]
        value_cache: *const c_void, // [num_blocks, block_size, num_heads, head_size]
        k_scale: *const c_void,
        v_scale: *const c_void,
        slot_mapping: *const c_long, // [num_tokens]

        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        block_size: c_int,
        key_stride: c_int,
        value_stride: c_int,
        block_stride: c_int,
        page_stride: c_int,
        head_stride: c_int,
        dtype: u32,
        stream: i64,
    );

    pub fn paged_attention_v1(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        k_scale: *const c_void,
        v_scale: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,

        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,

        dtype: u32,
        softscapping: f32,
        stream: i64,
    );

    pub fn paged_attention_v2(
        out: *const c_void,
        exp_sums: *const f32,
        max_logits: *const f32,
        tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        k_scale: *const c_void,
        v_scale: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,

        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,

        dtype: u32,
        softscapping: f32,
        stream: i64,
    );

    pub fn paged_attention_prefill(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        k_scale: *const c_void,
        v_scale: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,

        num_seqs: c_int,
        num_heads: c_int,
        num_query_tokens: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        num_blocks: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,

        dtype: u32,
        softscapping: f32,

        o_stride_tokens: c_int,
        query_start_len: *const u32,
        sinks: *const f32,
        sliding_window: c_int,
        stream: i64,
    );

    pub fn update_kv_scales_f32(
        k: *const c_void,
        v: *const c_void,
        elements: c_long,
        k_scales: *const f32,
        v_scales: *const f32,
        stream: i64,
    );

    pub fn update_kv_scales_f16(
        k: *const c_void,
        v: *const c_void,
        elements: c_long,
        k_scales: *const f32,
        v_scales: *const f32,
        stream: i64,
    );

    pub fn update_kv_scales_bf16(
        k: *const c_void,
        v: *const c_void,
        elements: c_long,
        k_scales: *const f32,
        v_scales: *const f32,
        stream: i64,
    );

    pub fn marlin_4bit_f16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );

    pub fn marlin_4bit_bf16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );

    pub fn marlin_awq_4bit_f16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );

    pub fn marlin_awq_4bit_bf16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );
    pub fn gptq_repack(
        weight: *const c_void,
        result: *const c_void,
        m: c_int,
        n: c_int,
        stream: i64,
    );

    pub fn awq_repack(
        weight: *const c_void,
        result: *const c_void,
        k: c_int,
        n: c_int,
        bits: c_int,
        stream: i64,
    );

    pub fn gemm_half_q_half_alt(
        a: *const c_void,
        weight: *const u32,
        qzeros: *const u32,
        scales: *const c_void,
        g_idx: *const i32,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        bit: i32,
        stream: i64,
    );

    pub fn copy_blocks_bf16(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block: i32,
        stream: i64,
    );

    pub fn copy_blocks_f16(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block: i32,
        stream: i64,
    );

    pub fn copy_blocks_f32(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block: i32,
        stream: i64,
    );

    pub fn asort_asc_f32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_f16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_bf16(
        x: *const c_void,
        dst: *const c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_f64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_u8(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_u32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_i64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_f32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_f16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_bf16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_f64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_u8(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_u32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_i64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
}
