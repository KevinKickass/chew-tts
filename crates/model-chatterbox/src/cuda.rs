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
