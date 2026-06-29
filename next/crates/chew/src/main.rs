use std::error::Error;

use chew_cuda::{CudaBackend, DeviceInfo, ModuleCache, NvrtcCompileOptions};
use chew_model_gemma4::{Gemma4Model, Gemma4ModelConfig};
use chew_model_llama::{LlamaModel, LlamaModelConfig};
use chew_runtime::{
    KvArenaLayout, KvCacheState, KvLayerInputU16, KvScalarType, KvSwaType, KvUbatch, Runtime,
    RuntimeLimits,
};

fn main() -> Result<(), Box<dyn Error>> {
    let cuda = CudaBackend::new(DeviceInfo { ordinal: 0 })?;
    let mut module_cache = ModuleCache::default();
    let compile_options = NvrtcCompileOptions {
        use_fast_math: true,
        line_info: false,
        max_registers: Some(128),
    };
    module_cache.prewarm_all(&cuda, compile_options)?;
    let runtime = Runtime::new(
        cuda,
        RuntimeLimits {
            n_ctx: 8192,
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 4,
        },
    )?;
    let prefill = runtime.plan_prefill(4096)?;

    let llama = LlamaModel::new(LlamaModelConfig {
        n_layers: 32,
        n_heads: 32,
        n_kv_heads: 8,
        head_dim: 128,
    });
    let gemma4 = Gemma4Model::new(Gemma4ModelConfig {
        n_layers: 42,
        n_kv_heads: 8,
        head_dim: 256,
        sliding_window: 1024,
        full_attention_stride: 6,
        shared_kv_layers: 0,
    });
    assert!(llama.validate_limits(runtime.limits()));
    assert!(gemma4.validate_limits(runtime.limits()));
    let llama_layout = llama.kv_cache_layout();
    let gemma4_layout = gemma4.kv_cache_layout();
    let v_trans = true;
    let mut kv_state = KvCacheState::new_with_policy(512, runtime.limits().n_seq_max, true, 32, 0, KvSwaType::None);
    let mut kv_ctx = kv_state
        .prepare_context(vec![KvUbatch::single_seq(0, &[384, 385, 386, 387])], true)
        .expect("llama-style KV context");
    kv_ctx.apply(&mut kv_state).expect("llama-style KV apply");
    let kv_demo_ctx = kv_ctx.n_kv().max(512);
    let kv_arena = KvArenaLayout::for_layout(
        &llama_layout,
        kv_demo_ctx,
        kv_state.n_stream(),
        KvScalarType::F16,
    )?;
    let kv_batch = kv_ctx.batch_execution_plan(&kv_arena, &llama_layout, v_trans)?;
    let llama_masks = kv_ctx.batch_attention_plan(&kv_state, &llama_layout, true)?;
    let gemma_masks = kv_ctx.batch_attention_plan(&kv_state, &gemma4_layout, true)?;
    let kv_mask_finite = llama_masks.layers[0]
        .mask
        .values
        .iter()
        .filter(|v| v.is_finite())
        .count();
    let gemma_mask_finite = gemma_masks.layers[1]
        .mask
        .values
        .iter()
        .filter(|v| v.is_finite())
        .count();
    let kv_layer0 = &kv_batch.layers[0];
    let kv_input = kv_layer0.input_copy_plan(llama.n_embd_k_gqa(), llama.n_embd_v_gqa())?;
    let prepared_kv = runtime.prepare_kv_write_bundle(
        &mut module_cache,
        &kv_batch,
        &llama_layout,
        compile_options,
    )?;
    let kv_inputs: Vec<KvLayerInputU16> = kv_batch
        .layers
        .iter()
        .map(|layer| {
            let input = layer.input_copy_plan(llama.n_embd_k_gqa(), llama.n_embd_v_gqa())?;
            let key_len = (input.key.row_width_el as usize) * (input.key.row_count as usize);
            let value_len = (input.value.row_width_el as usize) * (input.value.row_count as usize);

            Ok::<_, Box<dyn Error>>(KvLayerInputU16 {
                layer_idx: layer.layer_idx,
                key_src: (0..key_len)
                    .map(|i| layer.layer_idx as u16 + 1 + i as u16)
                    .collect(),
                value_src: (0..value_len)
                    .map(|i| 10_000_u16 + layer.layer_idx as u16 + i as u16)
                    .collect(),
            })
        })
        .collect::<Result<_, _>>()?;
    let kv_jobs = kv_batch.jobs_u16(&llama_layout, &kv_inputs)?;
    let mut kv_arena_u16 = vec![0_u16; (kv_arena.total_bytes / 2) as usize];
    runtime.execute_prepared_kv_write_bundle_u16(&prepared_kv, &kv_jobs, &mut kv_arena_u16)?;
    let kv_written_nonzero = kv_arena_u16.iter().filter(|&&v| v != 0).count();

    let mut demo_runtime = Runtime::new(
        CudaBackend::new(DeviceInfo { ordinal: 0 })?,
        runtime.limits(),
    )?;
    let demo_slot = demo_runtime.open_session(2048)?;
    demo_runtime.ingest_tokens(demo_slot, 384)?;
    let llama_kv = demo_runtime.plan_session_kv_write(demo_slot, 16, &llama_layout)?;
    let demo_view = demo_runtime.commit_session_kv_write(&llama_kv)?;
    let scheduled_decode = demo_runtime.schedule_decode(1)?;

    let mut gemma_demo_runtime = Runtime::new(
        CudaBackend::new(DeviceInfo { ordinal: 0 })?,
        runtime.limits(),
    )?;
    let gemma_slot = gemma_demo_runtime.open_session(2048)?;
    gemma_demo_runtime.ingest_tokens(gemma_slot, 384)?;
    let gemma4_kv = gemma_demo_runtime.plan_session_kv_write(gemma_slot, 16, &gemma4_layout)?;

    println!(
        "chew-next bootstrap: n_ctx={}, n_batch={}, n_ubatch={}, n_seq_max={}, prefill_batches={}, cached_modules={}, cache_compiles={}, llama_layers={}, gemma4_swa_layers={}, llama_attn_span={}, gemma4_attn_span={}, committed_ctx={}, scheduled_decode={}, kv_ctx_batches={}, kv_batch_layers={}, kv_n={}, kv_mask_finite={}, gemma_mask_finite={}, kv_k_idxs={}, kv_v_idxs={}, kv_view_streams={}, kv_view_key_off={}, kv_copy_rows={}, kv_input_rows_k={}, kv_input_rows_v={}, kv_kernel_k={}, kv_kernel_v={}, kv_demo_ctx={}, kv_written_nonzero={}",
        runtime.limits().n_ctx,
        runtime.limits().n_batch,
        runtime.limits().n_ubatch,
        runtime.limits().n_seq_max,
        prefill.batches.len(),
        module_cache.entry_count(),
        module_cache.compile_count(),
        llama_layout.layer_count(),
        gemma4_layout.sliding_window_layer_count(),
        llama_kv.layers[0].attention_span.token_count,
        gemma4_kv.layers[1].attention_span.token_count,
        demo_view.used_ctx,
        scheduled_decode.slots.len(),
        kv_ctx.len(),
        kv_batch.layers.len(),
        kv_layer0.n_kv,
        kv_mask_finite,
        gemma_mask_finite,
        kv_layer0.k_indices.len(),
        kv_layer0.v_indices.len(),
        kv_layer0.cache_view.stream_count,
        kv_layer0.cache_view.key.byte_offset,
        kv_layer0.copy_plan.value.row_count,
        kv_input.key.row_count,
        kv_input.value.row_count,
        prepared_kv.launches[0].key.kernel_name,
        prepared_kv.launches[0].value.kernel_name,
        kv_demo_ctx,
        kv_written_nonzero
    );

    Ok(())
}
