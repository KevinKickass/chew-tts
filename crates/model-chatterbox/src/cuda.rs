use crate::{ATTENTION_HEADS, HEAD_DIM, HIDDEN_SIZE, INTERMEDIATE_SIZE};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_safetensors::MappedSafetensors;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

pub struct ChatterboxT3Layer {
    input_norm: CudaSlice<f16>,
    q_proj: CudaSlice<f16>,
    k_proj: CudaSlice<f16>,
    v_proj: CudaSlice<f16>,
    o_proj: CudaSlice<f16>,
    post_attention_norm: CudaSlice<f16>,
    gate_proj: CudaSlice<f16>,
    up_proj: CudaSlice<f16>,
    down_proj: CudaSlice<f16>,
}

pub struct ChatterboxT3Transformer {
    layers: Vec<ChatterboxT3Layer>,
    final_norm: CudaSlice<f16>,
    rope_factors: CudaSlice<f32>,
}

pub struct ChatterboxT3Session {
    caches: Vec<ChatterboxT3KvCache>,
    scratch: ChatterboxT3BatchScratch,
    max_seq_len: usize,
}

struct ChatterboxT3KvCache {
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    position: usize,
    max_seq_len: usize,
}

struct ChatterboxT3BatchScratch {
    max_tokens: usize,
    norm: CudaSlice<f16>,
    q: CudaSlice<f16>,
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    attention: CudaSlice<f16>,
    attention_out: CudaSlice<f16>,
    gate: CudaSlice<f16>,
    up: CudaSlice<f16>,
    activation: CudaSlice<f16>,
    mlp_out: CudaSlice<f16>,
}

struct ChatterboxT3Scratch {
    norm: CudaSlice<f16>,
    q: CudaSlice<f16>,
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    attention: CudaSlice<f16>,
    attention_out: CudaSlice<f16>,
    gate: CudaSlice<f16>,
    up: CudaSlice<f16>,
    activation: CudaSlice<f16>,
    mlp_out: CudaSlice<f16>,
}

impl ChatterboxT3Scratch {
    fn allocate(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        Ok(Self {
            norm: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
            q: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
            k: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
            v: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
            attention: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
            attention_out: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
            gate: stream.alloc_zeros::<f16>(INTERMEDIATE_SIZE)?,
            up: stream.alloc_zeros::<f16>(INTERMEDIATE_SIZE)?,
            activation: stream.alloc_zeros::<f16>(INTERMEDIATE_SIZE)?,
            mlp_out: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
        })
    }
}

impl ChatterboxT3BatchScratch {
    fn allocate(max_tokens: usize, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        ensure!(max_tokens > 0, "T3 scratch capacity must be non-zero");
        Ok(Self {
            max_tokens,
            norm: stream.alloc_zeros::<f16>(max_tokens * HIDDEN_SIZE)?,
            q: stream.alloc_zeros::<f16>(max_tokens * HIDDEN_SIZE)?,
            k: stream.alloc_zeros::<f16>(max_tokens * HIDDEN_SIZE)?,
            v: stream.alloc_zeros::<f16>(max_tokens * HIDDEN_SIZE)?,
            attention: stream.alloc_zeros::<f16>(max_tokens * HIDDEN_SIZE)?,
            attention_out: stream.alloc_zeros::<f16>(max_tokens * HIDDEN_SIZE)?,
            gate: stream.alloc_zeros::<f16>(max_tokens * INTERMEDIATE_SIZE)?,
            up: stream.alloc_zeros::<f16>(max_tokens * INTERMEDIATE_SIZE)?,
            activation: stream.alloc_zeros::<f16>(max_tokens * INTERMEDIATE_SIZE)?,
            mlp_out: stream.alloc_zeros::<f16>(max_tokens * HIDDEN_SIZE)?,
        })
    }
}

impl ChatterboxT3KvCache {
    fn allocate(max_seq_len: usize, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        ensure!(max_seq_len > 0, "T3 KV capacity must be non-zero");
        Ok(Self {
            k: stream.alloc_zeros::<f16>(max_seq_len * HIDDEN_SIZE)?,
            v: stream.alloc_zeros::<f16>(max_seq_len * HIDDEN_SIZE)?,
            position: 0,
            max_seq_len,
        })
    }
}

impl ChatterboxT3Layer {
    pub fn load(
        model_dir: &Path,
        layer_index: usize,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        ensure!(
            layer_index < crate::LAYERS,
            "Chatterbox T3 layer {layer_index} is outside 0..{}",
            crate::LAYERS
        );
        let path = model_dir.join("t3_mtl23ls_v3.safetensors");
        let weights = MappedSafetensors::open(&path)?;
        let prefix = format!("tfmr.layers.{layer_index}");
        let load = |suffix: &str, expected: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let name = format!("{prefix}.{suffix}");
            let (shape, values) = weights
                .tensor_f16(&name)
                .with_context(|| format!("could not load Chatterbox T3 {name}"))?;
            ensure!(
                shape == expected,
                "Chatterbox T3 {name} has shape {shape:?}, expected {expected:?}"
            );
            stream
                .clone_htod(&values)
                .with_context(|| format!("could not upload Chatterbox T3 {name}"))
        };
        Ok(Self {
            input_norm: load("input_layernorm.weight", &[HIDDEN_SIZE])?,
            q_proj: load("self_attn.q_proj.weight", &[HIDDEN_SIZE, HIDDEN_SIZE])?,
            k_proj: load("self_attn.k_proj.weight", &[HIDDEN_SIZE, HIDDEN_SIZE])?,
            v_proj: load("self_attn.v_proj.weight", &[HIDDEN_SIZE, HIDDEN_SIZE])?,
            o_proj: load("self_attn.o_proj.weight", &[HIDDEN_SIZE, HIDDEN_SIZE])?,
            post_attention_norm: load("post_attention_layernorm.weight", &[HIDDEN_SIZE])?,
            gate_proj: load("mlp.gate_proj.weight", &[INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
            up_proj: load("mlp.up_proj.weight", &[INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
            down_proj: load("mlp.down_proj.weight", &[HIDDEN_SIZE, INTERMEDIATE_SIZE])?,
        })
    }

    /// Validate one complete native T3 decoder layer. At position zero the
    /// Llama-3 scaled RoPE factors are all identity, making this an exact
    /// correctness target before the cached multi-token path is added.
    pub fn forward_first_token(
        &self,
        hidden_host: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            hidden_host.len() == HIDDEN_SIZE,
            "Chatterbox T3 hidden input has {} values, expected {HIDDEN_SIZE}",
            hidden_host.len()
        );
        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        let mut scratch = ChatterboxT3Scratch::allocate(&stream)?;
        self.forward_first_token_device(&mut hidden, &mut scratch, kernels)?;
        stream.synchronize()?;
        let mut output = vec![0.0; HIDDEN_SIZE];
        stream.memcpy_dtoh(&hidden, &mut output)?;
        Ok(output)
    }

    fn forward_first_token_device(
        &self,
        hidden: &mut CudaSlice<f32>,
        scratch: &mut ChatterboxT3Scratch,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<()> {
        kernels.ops.rms_norm_f32in(
            hidden,
            &self.input_norm,
            &mut scratch.norm,
            1,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        for (weight, output) in [
            (&self.q_proj, &mut scratch.q),
            (&self.k_proj, &mut scratch.k),
            (&self.v_proj, &mut scratch.v),
        ] {
            kernels.gemm.matmul_f16(
                &scratch.norm,
                weight,
                output,
                1,
                HIDDEN_SIZE as u32,
                HIDDEN_SIZE as u32,
            )?;
        }
        // RoPE at position zero is an identity transform.
        kernels.ops.mha_fused(
            &scratch.q,
            &scratch.k.slice(..),
            &scratch.v.slice(..),
            &mut scratch.attention,
            HEAD_DIM as u32,
            ATTENTION_HEADS as u32,
            ATTENTION_HEADS as u32,
            1,
            1,
            0,
        )?;
        kernels.gemm.matmul_f16(
            &scratch.attention,
            &self.o_proj,
            &mut scratch.attention_out,
            1,
            HIDDEN_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(hidden, &scratch.attention_out, HIDDEN_SIZE as u32)?;

        kernels.ops.rms_norm_f32in(
            hidden,
            &self.post_attention_norm,
            &mut scratch.norm,
            1,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        kernels.gemm.matmul_f16(
            &scratch.norm,
            &self.gate_proj,
            &mut scratch.gate,
            1,
            INTERMEDIATE_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels.gemm.matmul_f16(
            &scratch.norm,
            &self.up_proj,
            &mut scratch.up,
            1,
            INTERMEDIATE_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels.ops.silu(
            &scratch.gate,
            &scratch.up,
            &mut scratch.activation,
            INTERMEDIATE_SIZE as u32,
        )?;
        kernels.gemm.matmul_f16(
            &scratch.activation,
            &self.down_proj,
            &mut scratch.mlp_out,
            1,
            HIDDEN_SIZE as u32,
            INTERMEDIATE_SIZE as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(hidden, &scratch.mlp_out, HIDDEN_SIZE as u32)?;
        Ok(())
    }

    fn forward_cached_device(
        &self,
        hidden: &mut CudaSlice<f32>,
        seq_len: usize,
        rope_factors: &CudaSlice<f32>,
        cache: &mut ChatterboxT3KvCache,
        scratch: &mut ChatterboxT3BatchScratch,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<()> {
        ensure!(seq_len > 0, "T3 sequence length must be non-zero");
        ensure!(
            seq_len <= scratch.max_tokens,
            "T3 sequence has {seq_len} tokens, scratch holds {}",
            scratch.max_tokens
        );
        ensure!(
            hidden.len() >= seq_len * HIDDEN_SIZE,
            "T3 device hidden state is too small"
        );
        ensure!(
            cache.position + seq_len <= cache.max_seq_len,
            "T3 KV capacity {} exceeded by position {} + {seq_len}",
            cache.max_seq_len,
            cache.position
        );
        let rows = u32::try_from(seq_len).context("T3 sequence exceeds CUDA limits")?;
        let position = u32::try_from(cache.position).context("T3 position exceeds CUDA limits")?;
        let total_kv_len =
            u32::try_from(cache.position + seq_len).context("T3 KV length exceeds CUDA limits")?;

        kernels.ops.rms_norm_f32in(
            hidden,
            &self.input_norm,
            &mut scratch.norm,
            rows,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        if seq_len == 1 {
            for (weight, output) in [
                (&self.q_proj, &mut scratch.q),
                (&self.k_proj, &mut scratch.k),
                (&self.v_proj, &mut scratch.v),
            ] {
                kernels.gemv.gemv_f16(
                    &scratch.norm,
                    weight,
                    output,
                    HIDDEN_SIZE as u32,
                    HIDDEN_SIZE as u32,
                )?;
            }
        } else {
            for (weight, output) in [
                (&self.q_proj, &mut scratch.q),
                (&self.k_proj, &mut scratch.k),
                (&self.v_proj, &mut scratch.v),
            ] {
                kernels.gemm.matmul_f16(
                    &scratch.norm,
                    weight,
                    output,
                    rows,
                    HIDDEN_SIZE as u32,
                    HIDDEN_SIZE as u32,
                )?;
            }
        }
        kernels.ops.rope_neox_freqs(
            &mut scratch.q,
            rope_factors,
            rows,
            ATTENTION_HEADS as u32,
            HEAD_DIM as u32,
            position,
            500_000.0,
        )?;
        kernels.ops.rope_neox_freqs(
            &mut scratch.k,
            rope_factors,
            rows,
            ATTENTION_HEADS as u32,
            HEAD_DIM as u32,
            position,
            500_000.0,
        )?;

        let cache_offset = cache.position * HIDDEN_SIZE;
        let cache_end = cache_offset + seq_len * HIDDEN_SIZE;
        {
            let mut destination = cache.k.slice_mut(cache_offset..cache_end);
            kernels
                .ops
                .copy_f16(&scratch.k, &mut destination, (seq_len * HIDDEN_SIZE) as u32)?;
        }
        {
            let mut destination = cache.v.slice_mut(cache_offset..cache_end);
            kernels
                .ops
                .copy_f16(&scratch.v, &mut destination, (seq_len * HIDDEN_SIZE) as u32)?;
        }
        kernels.ops.mha_fused(
            &scratch.q,
            &cache.k.slice(..cache_end),
            &cache.v.slice(..cache_end),
            &mut scratch.attention,
            HEAD_DIM as u32,
            ATTENTION_HEADS as u32,
            ATTENTION_HEADS as u32,
            rows,
            total_kv_len,
            position,
        )?;
        if seq_len == 1 {
            kernels.gemv.gemv_f16(
                &scratch.attention,
                &self.o_proj,
                &mut scratch.attention_out,
                HIDDEN_SIZE as u32,
                HIDDEN_SIZE as u32,
            )?;
        } else {
            kernels.gemm.matmul_f16(
                &scratch.attention,
                &self.o_proj,
                &mut scratch.attention_out,
                rows,
                HIDDEN_SIZE as u32,
                HIDDEN_SIZE as u32,
            )?;
        }
        kernels.ops.add_inplace_f32_f16(
            hidden,
            &scratch.attention_out,
            (seq_len * HIDDEN_SIZE) as u32,
        )?;
        kernels.ops.rms_norm_f32in(
            hidden,
            &self.post_attention_norm,
            &mut scratch.norm,
            rows,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        if seq_len == 1 {
            kernels.gemv.gemv_dual_f16(
                &scratch.norm,
                &self.gate_proj,
                &self.up_proj,
                &mut scratch.gate,
                &mut scratch.up,
                INTERMEDIATE_SIZE as u32,
                HIDDEN_SIZE as u32,
            )?;
        } else {
            kernels.gemm.matmul_f16(
                &scratch.norm,
                &self.gate_proj,
                &mut scratch.gate,
                rows,
                INTERMEDIATE_SIZE as u32,
                HIDDEN_SIZE as u32,
            )?;
            kernels.gemm.matmul_f16(
                &scratch.norm,
                &self.up_proj,
                &mut scratch.up,
                rows,
                INTERMEDIATE_SIZE as u32,
                HIDDEN_SIZE as u32,
            )?;
        }
        kernels.ops.silu(
            &scratch.gate,
            &scratch.up,
            &mut scratch.activation,
            (seq_len * INTERMEDIATE_SIZE) as u32,
        )?;
        if seq_len == 1 {
            kernels.gemv.gemv_f16(
                &scratch.activation,
                &self.down_proj,
                &mut scratch.mlp_out,
                HIDDEN_SIZE as u32,
                INTERMEDIATE_SIZE as u32,
            )?;
        } else {
            kernels.gemm.matmul_f16(
                &scratch.activation,
                &self.down_proj,
                &mut scratch.mlp_out,
                rows,
                HIDDEN_SIZE as u32,
                INTERMEDIATE_SIZE as u32,
            )?;
        }
        kernels.ops.add_inplace_f32_f16(
            hidden,
            &scratch.mlp_out,
            (seq_len * HIDDEN_SIZE) as u32,
        )?;
        cache.position += seq_len;
        Ok(())
    }

    fn forward_cached_pair_device(
        &self,
        hidden: &mut CudaSlice<f32>,
        rope_factors: &CudaSlice<f32>,
        cache_a: &mut ChatterboxT3KvCache,
        cache_b: &mut ChatterboxT3KvCache,
        scratch_a: &mut ChatterboxT3BatchScratch,
        scratch_b: &mut ChatterboxT3BatchScratch,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<()> {
        ensure!(
            cache_a.position == cache_b.position,
            "paired T3 caches are out of sync"
        );
        ensure!(
            scratch_a.max_tokens >= 2 && scratch_b.max_tokens >= 1,
            "paired T3 decode requires two scratch rows"
        );
        ensure!(
            cache_a.position < cache_a.max_seq_len && cache_b.position < cache_b.max_seq_len,
            "paired T3 KV capacity exceeded"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let position = cache_a.position;
        let position_u32 = u32::try_from(position).context("T3 position exceeds CUDA limits")?;
        let kv_len = u32::try_from(position + 1).context("T3 KV length exceeds CUDA limits")?;

        kernels.ops.rms_norm_f32in(
            hidden,
            &self.input_norm,
            &mut scratch_a.norm,
            2,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        for (weight, output) in [
            (&self.q_proj, &mut scratch_a.q),
            (&self.k_proj, &mut scratch_a.k),
            (&self.v_proj, &mut scratch_a.v),
        ] {
            kernels.gemm.matmul_f16(
                &scratch_a.norm,
                weight,
                output,
                2,
                HIDDEN_SIZE as u32,
                HIDDEN_SIZE as u32,
            )?;
        }
        kernels.ops.rope_neox_freqs_batched(
            &mut scratch_a.q,
            rope_factors,
            2,
            1,
            ATTENTION_HEADS as u32,
            HEAD_DIM as u32,
            position_u32,
            500_000.0,
        )?;
        kernels.ops.rope_neox_freqs_batched(
            &mut scratch_a.k,
            rope_factors,
            2,
            1,
            ATTENTION_HEADS as u32,
            HEAD_DIM as u32,
            position_u32,
            500_000.0,
        )?;

        stream.memcpy_dtod(
            &scratch_a.q.slice(HIDDEN_SIZE..2 * HIDDEN_SIZE),
            &mut scratch_b.q.slice_mut(..HIDDEN_SIZE),
        )?;
        stream.memcpy_dtod(
            &scratch_a.k.slice(HIDDEN_SIZE..2 * HIDDEN_SIZE),
            &mut scratch_b.k.slice_mut(..HIDDEN_SIZE),
        )?;
        stream.memcpy_dtod(
            &scratch_a.v.slice(HIDDEN_SIZE..2 * HIDDEN_SIZE),
            &mut scratch_b.v.slice_mut(..HIDDEN_SIZE),
        )?;

        let cache_offset = position * HIDDEN_SIZE;
        let cache_end = cache_offset + HIDDEN_SIZE;
        for (source, destination) in [
            (&scratch_a.k, &mut cache_a.k),
            (&scratch_a.v, &mut cache_a.v),
            (&scratch_b.k, &mut cache_b.k),
            (&scratch_b.v, &mut cache_b.v),
        ] {
            stream.memcpy_dtod(
                &source.slice(..HIDDEN_SIZE),
                &mut destination.slice_mut(cache_offset..cache_end),
            )?;
        }
        kernels.ops.mha_fused(
            &scratch_a.q,
            &cache_a.k.slice(..cache_end),
            &cache_a.v.slice(..cache_end),
            &mut scratch_a.attention,
            HEAD_DIM as u32,
            ATTENTION_HEADS as u32,
            ATTENTION_HEADS as u32,
            1,
            kv_len,
            position_u32,
        )?;
        kernels.ops.mha_fused(
            &scratch_b.q,
            &cache_b.k.slice(..cache_end),
            &cache_b.v.slice(..cache_end),
            &mut scratch_b.attention,
            HEAD_DIM as u32,
            ATTENTION_HEADS as u32,
            ATTENTION_HEADS as u32,
            1,
            kv_len,
            position_u32,
        )?;
        stream.memcpy_dtod(
            &scratch_b.attention.slice(..HIDDEN_SIZE),
            &mut scratch_a.attention.slice_mut(HIDDEN_SIZE..2 * HIDDEN_SIZE),
        )?;
        kernels.gemm.matmul_f16(
            &scratch_a.attention,
            &self.o_proj,
            &mut scratch_a.attention_out,
            2,
            HIDDEN_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels.ops.add_inplace_f32_f16(
            hidden,
            &scratch_a.attention_out,
            (2 * HIDDEN_SIZE) as u32,
        )?;
        kernels.ops.rms_norm_f32in(
            hidden,
            &self.post_attention_norm,
            &mut scratch_a.norm,
            2,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        kernels.gemm.matmul_f16(
            &scratch_a.norm,
            &self.gate_proj,
            &mut scratch_a.gate,
            2,
            INTERMEDIATE_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels.gemm.matmul_f16(
            &scratch_a.norm,
            &self.up_proj,
            &mut scratch_a.up,
            2,
            INTERMEDIATE_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels.ops.silu(
            &scratch_a.gate,
            &scratch_a.up,
            &mut scratch_a.activation,
            (2 * INTERMEDIATE_SIZE) as u32,
        )?;
        kernels.gemm.matmul_f16(
            &scratch_a.activation,
            &self.down_proj,
            &mut scratch_a.mlp_out,
            2,
            HIDDEN_SIZE as u32,
            INTERMEDIATE_SIZE as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(hidden, &scratch_a.mlp_out, (2 * HIDDEN_SIZE) as u32)?;
        cache_a.position += 1;
        cache_b.position += 1;
        Ok(())
    }
}

impl ChatterboxT3Transformer {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let mut layers = Vec::with_capacity(crate::LAYERS);
        for layer_index in 0..crate::LAYERS {
            layers.push(
                ChatterboxT3Layer::load(model_dir, layer_index, stream)
                    .with_context(|| format!("could not load Chatterbox T3 layer {layer_index}"))?,
            );
        }
        let path = model_dir.join("t3_mtl23ls_v3.safetensors");
        let weights = MappedSafetensors::open(path)?;
        let (shape, values) = weights.tensor_f16("tfmr.norm.weight")?;
        ensure!(
            shape == [HIDDEN_SIZE],
            "Chatterbox T3 final norm has shape {shape:?}, expected [{HIDDEN_SIZE}]"
        );
        let final_norm = stream.clone_htod(&values)?;
        let rope_factors = stream.clone_htod(&llama3_rope_factors())?;
        Ok(Self {
            layers,
            final_norm,
            rope_factors,
        })
    }

    pub fn forward_first_token(
        &self,
        hidden_host: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            hidden_host.len() == HIDDEN_SIZE,
            "Chatterbox T3 hidden input has {} values, expected {HIDDEN_SIZE}",
            hidden_host.len()
        );
        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        let mut scratch = ChatterboxT3Scratch::allocate(&stream)?;
        for layer in &self.layers {
            layer.forward_first_token_device(&mut hidden, &mut scratch, kernels)?;
        }
        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.final_norm,
            &mut scratch.norm,
            1,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        stream.synchronize()?;
        let mut output = vec![f16::ZERO; HIDDEN_SIZE];
        stream.memcpy_dtoh(&scratch.norm, &mut output)?;
        Ok(output.into_iter().map(f16::to_f32).collect())
    }

    pub fn start_session(
        &self,
        max_seq_len: usize,
        max_batch_tokens: usize,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<ChatterboxT3Session> {
        let caches = (0..self.layers.len())
            .map(|_| ChatterboxT3KvCache::allocate(max_seq_len, stream))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(ChatterboxT3Session {
            caches,
            scratch: ChatterboxT3BatchScratch::allocate(max_batch_tokens, stream)?,
            max_seq_len,
        })
    }

    pub fn forward_session(
        &self,
        session: &mut ChatterboxT3Session,
        hidden_host: &[f32],
        seq_len: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(seq_len > 0, "T3 sequence length must be non-zero");
        ensure!(
            hidden_host.len() == seq_len * HIDDEN_SIZE,
            "T3 hidden input has {} values, expected {}",
            hidden_host.len(),
            seq_len * HIDDEN_SIZE
        );
        ensure!(
            session.position() + seq_len <= session.max_seq_len,
            "T3 session length {} exceeds maximum {}",
            session.position() + seq_len,
            session.max_seq_len
        );
        let expected_position = session.position();
        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        for (layer, cache) in self.layers.iter().zip(&mut session.caches) {
            ensure!(
                cache.position == expected_position,
                "T3 layer caches are out of sync"
            );
            layer.forward_cached_device(
                &mut hidden,
                seq_len,
                &self.rope_factors,
                cache,
                &mut session.scratch,
                kernels,
            )?;
        }
        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.final_norm,
            &mut session.scratch.norm,
            seq_len as u32,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        stream.synchronize()?;
        let mut output = vec![f16::ZERO; seq_len * HIDDEN_SIZE];
        stream.memcpy_dtoh(
            &session.scratch.norm.slice(..seq_len * HIDDEN_SIZE),
            &mut output,
        )?;
        Ok(output.into_iter().map(f16::to_f32).collect())
    }

    pub fn forward_session_pair(
        &self,
        session_a: &mut ChatterboxT3Session,
        session_b: &mut ChatterboxT3Session,
        hidden_a: &[f32],
        hidden_b: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        ensure!(
            hidden_a.len() == HIDDEN_SIZE && hidden_b.len() == HIDDEN_SIZE,
            "paired T3 decode expects two hidden rows"
        );
        ensure!(
            session_a.position() == session_b.position(),
            "paired T3 sessions are out of sync"
        );
        ensure!(
            session_a.position() < session_a.max_seq_len
                && session_b.position() < session_b.max_seq_len,
            "paired T3 session capacity exceeded"
        );
        let mut host = Vec::with_capacity(2 * HIDDEN_SIZE);
        host.extend_from_slice(hidden_a);
        host.extend_from_slice(hidden_b);
        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(&host)?;
        for index in 0..self.layers.len() {
            self.layers[index].forward_cached_pair_device(
                &mut hidden,
                &self.rope_factors,
                &mut session_a.caches[index],
                &mut session_b.caches[index],
                &mut session_a.scratch,
                &mut session_b.scratch,
                kernels,
            )?;
        }
        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.final_norm,
            &mut session_a.scratch.norm,
            2,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        stream.synchronize()?;
        let mut output = vec![f16::ZERO; 2 * HIDDEN_SIZE];
        stream.memcpy_dtoh(
            &session_a.scratch.norm.slice(..2 * HIDDEN_SIZE),
            &mut output,
        )?;
        let output = output.into_iter().map(f16::to_f32).collect::<Vec<_>>();
        Ok((
            output[..HIDDEN_SIZE].to_vec(),
            output[HIDDEN_SIZE..].to_vec(),
        ))
    }
}

impl ChatterboxT3Session {
    pub fn position(&self) -> usize {
        self.caches.first().map_or(0, |cache| cache.position)
    }

    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.position = 0;
        }
    }
}

pub fn llama3_rope_factors() -> Vec<f32> {
    let factor = 8.0f32;
    let low_frequency_wavelength = 8_192.0f32;
    let high_frequency_wavelength = 2_048.0f32;
    (0..HEAD_DIM / 2)
        .map(|index| {
            let inv_frequency = 1.0 / 500_000.0f32.powf((2 * index) as f32 / HEAD_DIM as f32);
            let wavelength = std::f32::consts::TAU / inv_frequency;
            let scaled = if wavelength < high_frequency_wavelength {
                inv_frequency
            } else if wavelength > low_frequency_wavelength {
                inv_frequency / factor
            } else {
                let smooth = (8_192.0 / wavelength - 1.0) / 3.0;
                (1.0 - smooth) * inv_frequency / factor + smooth * inv_frequency
            };
            inv_frequency / scaled
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama3_rope_factors_preserve_and_scale_expected_bands() {
        let factors = llama3_rope_factors();
        assert_eq!(factors.len(), HEAD_DIM / 2);
        assert_eq!(factors[0], 1.0);
        assert!((factors[factors.len() - 1] - 8.0).abs() < 1e-5);
        assert!(factors.windows(2).all(|pair| pair[0] <= pair[1]));
    }
}
