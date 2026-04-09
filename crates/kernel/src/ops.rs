use crate::fast_launch::{FastStream, slice_ptr, slice_ptr_mut, view_ptr, view_mut_ptr, scalar_ptr};
use crate::loader::{self, KernelError};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use std::ffi::c_void;
use std::sync::Arc;

const OPS_CU: &str = include_str!("cuda/ops.cu");

/// Transformer operation kernels — one set per GPU.
pub struct OpsKernels {
    stream: Arc<CudaStream>,
    fast: FastStream,
    _module: Arc<CudaModule>,
    rms_norm: CudaFunction,
    rms_norm_f32in: CudaFunction,
    rope: CudaFunction,
    silu: CudaFunction,
    silu_q8: CudaFunction,
    softmax: CudaFunction,
    embed_tokens_f32: CudaFunction,
    add_f16: CudaFunction,
    add_f32_f16: CudaFunction,
    copy_f32_to_f16: CudaFunction,
    copy_f16: CudaFunction,
    fused_add_rmsnorm: CudaFunction,
    rms_norm_f32in_q8: CudaFunction,
    fused_add_rmsnorm_q8: CudaFunction,
    add_inplace_f32_f16: CudaFunction,
    fused_rope_kv: CudaFunction,
    argmax_f16: CudaFunction,
    sample_top_k: CudaFunction,
    mha_fused: CudaFunction,
    // CUDA Graph-compatible variants
    rope_graph: CudaFunction,
    copy_f16_with_offset: CudaFunction,
    mha_fused_graph: CudaFunction,
    // Gemma 4 kernels
    gelu: CudaFunction,
    rms_norm_no_weight: CudaFunction,
    scale_f16: CudaFunction,
    scale_f32_inplace: CudaFunction,
    logit_softcap: CudaFunction,
    logit_softcap_inplace: CudaFunction,
    rope_neox: CudaFunction,
    rope_neox_freqs: CudaFunction,
    rope_neox_graph: CudaFunction,
    post_norm_add: CudaFunction,
    mul_f16: CudaFunction,
    gelu_act: CudaFunction,
    gather_rows_quant: CudaFunction,
    pe_strided_mul: CudaFunction,
    // Fused kernels for launch reduction
    rope_kv_write: CudaFunction,
}

impl OpsKernels {
    pub fn load(stream: &Arc<CudaStream>) -> Result<Self, KernelError> {
        let module = loader::load_module_from_source(stream, OPS_CU, "ops")?;

        Ok(Self {
            stream: Arc::clone(stream),
            fast: FastStream::new(stream),
            rms_norm: loader::get_fn(&module, "rms_norm")?,
            rms_norm_f32in: loader::get_fn(&module, "rms_norm_f32in")?,
            rope: loader::get_fn(&module, "rope")?,
            silu: loader::get_fn(&module, "silu")?,
            silu_q8: loader::get_fn(&module, "silu_q8")?,
            softmax: loader::get_fn(&module, "softmax")?,
            embed_tokens_f32: loader::get_fn(&module, "embed_tokens_f32")?,
            add_f16: loader::get_fn(&module, "add_f16")?,
            add_f32_f16: loader::get_fn(&module, "add_f32_f16")?,
            copy_f32_to_f16: loader::get_fn(&module, "copy_f32_to_f16")?,
            copy_f16: loader::get_fn(&module, "copy_f16")?,
            fused_add_rmsnorm: loader::get_fn(&module, "fused_add_rmsnorm")?,
            rms_norm_f32in_q8: loader::get_fn(&module, "rms_norm_f32in_q8")?,
            fused_add_rmsnorm_q8: loader::get_fn(&module, "fused_add_rmsnorm_q8")?,
            add_inplace_f32_f16: loader::get_fn(&module, "add_inplace_f32_f16")?,
            fused_rope_kv: loader::get_fn(&module, "fused_rope_kv")?,
            argmax_f16: loader::get_fn(&module, "argmax_f16")?,
            sample_top_k: loader::get_fn(&module, "sample_top_k")?,
            mha_fused: loader::get_fn(&module, "mha_fused")?,
            rope_graph: loader::get_fn(&module, "rope_graph")?,
            copy_f16_with_offset: loader::get_fn(&module, "copy_f16_with_offset")?,
            mha_fused_graph: loader::get_fn(&module, "mha_fused_graph")?,
            // Gemma 4 kernels
            gelu: loader::get_fn(&module, "gelu")?,
            rms_norm_no_weight: loader::get_fn(&module, "rms_norm_no_weight")?,
            scale_f16: loader::get_fn(&module, "scale_f16")?,
            scale_f32_inplace: loader::get_fn(&module, "scale_f32_inplace")?,
            logit_softcap: loader::get_fn(&module, "logit_softcap")?,
            logit_softcap_inplace: loader::get_fn(&module, "logit_softcap_inplace")?,
            rope_neox: loader::get_fn(&module, "rope_neox")?,
            rope_neox_freqs: loader::get_fn(&module, "rope_neox_freqs")?,
            rope_neox_graph: loader::get_fn(&module, "rope_neox_graph")?,
            post_norm_add: loader::get_fn(&module, "post_norm_add")?,
            mul_f16: loader::get_fn(&module, "mul_f16")?,
            gelu_act: loader::get_fn(&module, "gelu_act")?,
            gather_rows_quant: loader::get_fn(&module, "gather_rows_quant")?,
            pe_strided_mul: loader::get_fn(&module, "pe_strided_mul")?,
            rope_kv_write: loader::get_fn(&module, "rope_kv_write")?,
            _module: module,
        })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// RMSNorm: f16 input → f16 output.
    /// x shape: [n_rows, dim], weight shape: [dim] (f16)
    pub fn rms_norm(
        &self,
        x: &CudaSlice<half::f16>,
        weight: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        n_rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<(), KernelError> {
        let threads = 256u32.min(dim);
        let cfg = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * 4,
        };

        let dim_i = dim as i32;

        unsafe {
            self.stream
                .launch_builder(&self.rms_norm)
                .arg(x)
                .arg(weight)
                .arg(out)
                .arg(&dim_i)
                .arg(&eps)
                .launch(cfg)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
        }

        Ok(())
    }

    /// RMSNorm: f32 input → f16 output.
    /// Bridge from f32 hidden state to f16 GEMM input.
    pub fn rms_norm_f32in(
        &self,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        n_rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<(), KernelError> {
        let threads = 256u32.min(dim);
        let cfg = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * 4,
        };

        let dim_i = dim as i32;

        let mut args: [*mut c_void; 5] = [
            slice_ptr(x), slice_ptr(weight), slice_ptr_mut(out),
            scalar_ptr(&dim_i), scalar_ptr(&eps),
        ];
        unsafe { self.fast.fire(&self.rms_norm_f32in, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }

        Ok(())
    }

    /// RoPE: apply rotary position embeddings in-place on f16 data.
    /// x shape: [seq_len, n_heads, head_dim]
    pub fn rope(
        &self,
        x: &mut CudaSlice<half::f16>,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        pos: u32,
        theta_base: f32,
    ) -> Result<(), KernelError> {
        let cfg = LaunchConfig {
            grid_dim: (seq_len, n_heads, 1),
            block_dim: (head_dim / 2, 1, 1),
            shared_mem_bytes: 0,
        };

        let hd = head_dim as i32;
        let nh = n_heads as i32;
        let p = pos as i32;

        let mut args: [*mut c_void; 5] = [
            slice_ptr_mut(x), scalar_ptr(&hd), scalar_ptr(&nh),
            scalar_ptr(&p), scalar_ptr(&theta_base),
        ];
        unsafe { self.fast.fire(&self.rope, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }

        Ok(())
    }

    /// SiLU: out = SiLU(gate) * up, element-wise, all f16.
    pub fn silu(
        &self,
        gate: &CudaSlice<half::f16>,
        up: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        n: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i = n as i32;

        let mut args: [*mut c_void; 4] = [
            slice_ptr(gate), slice_ptr(up), slice_ptr_mut(out),
            scalar_ptr(&n_i),
        ];
        unsafe { self.fast.fire(&self.silu, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Fused SiLU + Q8_1 quantize: computes SiLU(gate)*up AND quantizes result to Q8_1.
    /// Saves 1 kernel launch by combining silu + quantize_input.
    pub fn silu_q8(
        &self,
        gate: &CudaSlice<half::f16>,
        up: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        x_q8: &mut CudaSlice<u8>,
        n: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut args: [*mut c_void; 5] = [
            slice_ptr(gate), slice_ptr(up), slice_ptr_mut(out),
            slice_ptr_mut(x_q8), scalar_ptr(&n_i),
        ];
        unsafe { self.fast.fire(&self.silu_q8, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }

        Ok(())
    }

    /// Softmax in-place over `dim` elements per row, f16.
    /// x shape: [n_rows, dim]
    pub fn softmax(
        &self,
        x: &mut CudaSlice<half::f16>,
        n_rows: u32,
        dim: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32.min(dim);
        let cfg = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * 4,
        };

        let dim_i = dim as i32;

        unsafe {
            self.stream
                .launch_builder(&self.softmax)
                .arg(x)
                .arg(&dim_i)
                .launch(cfg)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
        }

        Ok(())
    }

    /// Lookup token embeddings to f32 output: out[i] = embd[token_ids[i]]
    pub fn embed_tokens_f32(
        &self,
        embd: &CudaSlice<half::f16>,
        token_ids: &CudaSlice<i32>,
        out: &mut CudaSlice<f32>,
        n_tokens: u32,
        dim: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32.min(dim);
        let cfg = LaunchConfig {
            grid_dim: (n_tokens, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let dim_i = dim as i32;

        unsafe {
            self.stream
                .launch_builder(&self.embed_tokens_f32)
                .arg(embd)
                .arg(token_ids)
                .arg(out)
                .arg(&dim_i)
                .launch(cfg)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
        }

        Ok(())
    }

    /// Element-wise add: f16 + f16 -> f16
    pub fn add_f16(
        &self,
        a: &CudaSlice<half::f16>,
        b: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        n: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i = n as i32;

        unsafe {
            self.stream
                .launch_builder(&self.add_f16)
                .arg(a)
                .arg(b)
                .arg(out)
                .arg(&n_i)
                .launch(cfg)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
        }

        Ok(())
    }

    /// Element-wise add: f32 + f16 -> f32 (residual connection with f16 input)
    pub fn add_f32_f16(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<half::f16>,
        out: &mut CudaSlice<f32>,
        n: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i = n as i32;

        unsafe {
            self.stream
                .launch_builder(&self.add_f32_f16)
                .arg(a)
                .arg(b)
                .arg(out)
                .arg(&n_i)
                .launch(cfg)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
        }

        Ok(())
    }

    /// Fused add + RMSNorm: hidden += delta, norm_out = rmsnorm(hidden) * weight
    /// Saves one kernel launch vs separate add + norm.
    pub fn fused_add_rmsnorm(
        &self,
        hidden: &mut CudaSlice<f32>,
        delta: &CudaSlice<half::f16>,
        weight: &CudaSlice<half::f16>,
        norm_out: &mut CudaSlice<half::f16>,
        n_rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<(), KernelError> {
        let threads = 256u32.min(dim);
        let cfg = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * 4,
        };
        let dim_i = dim as i32;
        let mut args: [*mut c_void; 6] = [
            slice_ptr_mut(hidden), slice_ptr(delta), slice_ptr(weight),
            slice_ptr_mut(norm_out), scalar_ptr(&dim_i), scalar_ptr(&eps),
        ];
        unsafe { self.fast.fire(&self.fused_add_rmsnorm, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// RMSNorm (f32→f16) + Q8_1 quantize in one kernel.
    /// Writes both norm_out (f16) and x_q8 (Q8_1 format) simultaneously.
    pub fn rms_norm_f32in_q8(
        &self,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        x_q8: &mut CudaSlice<u8>,
        n_rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<(), KernelError> {
        let threads = 256u32.min(dim);
        let cfg = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * 4,
        };
        let dim_i = dim as i32;
        let mut args: [*mut c_void; 6] = [
            slice_ptr(x), slice_ptr(weight), slice_ptr_mut(out),
            slice_ptr_mut(x_q8), scalar_ptr(&dim_i), scalar_ptr(&eps),
        ];
        unsafe { self.fast.fire(&self.rms_norm_f32in_q8, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Fused add + RMSNorm + Q8_1 quantize in one kernel.
    /// hidden += delta, norm_out = rmsnorm(hidden) * weight, x_q8 = quantize(norm_out).
    pub fn fused_add_rmsnorm_q8(
        &self,
        hidden: &mut CudaSlice<f32>,
        delta: &CudaSlice<half::f16>,
        weight: &CudaSlice<half::f16>,
        norm_out: &mut CudaSlice<half::f16>,
        x_q8: &mut CudaSlice<u8>,
        n_rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<(), KernelError> {
        let threads = 256u32.min(dim);
        let cfg = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * 4,
        };
        let dim_i = dim as i32;
        let mut args: [*mut c_void; 7] = [
            slice_ptr_mut(hidden), slice_ptr(delta), slice_ptr(weight),
            slice_ptr_mut(norm_out), slice_ptr_mut(x_q8),
            scalar_ptr(&dim_i), scalar_ptr(&eps),
        ];
        unsafe { self.fast.fire(&self.fused_add_rmsnorm_q8, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// In-place add: hidden[i] += delta[i] (f32 + f16)
    pub fn add_inplace_f32_f16(
        &self,
        hidden: &mut CudaSlice<f32>,
        delta: &CudaSlice<half::f16>,
        n: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut args: [*mut c_void; 3] = [
            slice_ptr_mut(hidden), slice_ptr(delta), scalar_ptr(&n_i),
        ];
        unsafe { self.fast.fire(&self.add_inplace_f32_f16, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Copy f32 → f16.
    pub fn copy_f32_to_f16(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaViewMut<'_, half::f16>,
        n: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i = n as i32;

        unsafe {
            self.stream
                .launch_builder(&self.copy_f32_to_f16)
                .arg(src)
                .arg(dst)
                .arg(&n_i)
                .launch(cfg)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
        }

        Ok(())
    }

    /// Copy f16 → f16 (for KV cache writes from f16 projections).
    pub fn copy_f16(
        &self,
        src: &CudaSlice<half::f16>,
        dst: &mut CudaViewMut<'_, half::f16>,
        n: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i = n as i32;

        let mut args: [*mut c_void; 3] = [
            slice_ptr(src), view_mut_ptr(dst), scalar_ptr(&n_i),
        ];
        unsafe { self.fast.fire(&self.copy_f16, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }

        Ok(())
    }

    /// Fused Multi-Head Attention with GQA support.
    ///
    /// Q: [seq_len, n_heads, head_dim]     f16
    /// K: [kv_len, n_kv_heads, head_dim]   f16 (from KV cache)
    /// V: [kv_len, n_kv_heads, head_dim]   f16 (from KV cache)
    /// out: [seq_len, n_heads, head_dim]    f16
    ///
    /// Fused RoPE(Q) + RoPE(K) + KV cache write. One launch replaces 4.
    pub fn fused_rope_kv(
        &self,
        q: &mut CudaSlice<half::f16>,
        k: &mut CudaSlice<half::f16>,
        v: &CudaSlice<half::f16>,
        k_cache: &mut CudaViewMut<'_, half::f16>,
        v_cache: &mut CudaViewMut<'_, half::f16>,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        seq_len: u32,
        pos: u32,
        theta_base: f32,
    ) -> Result<(), KernelError> {
        let cfg = LaunchConfig {
            grid_dim: (seq_len, n_heads + n_kv_heads, 1),
            block_dim: (head_dim / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        let hd = head_dim as i32;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let p = pos as i32;
        unsafe {
            self.stream
                .launch_builder(&self.fused_rope_kv)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(k_cache)
                .arg(v_cache)
                .arg(&hd)
                .arg(&nh)
                .arg(&nkv)
                .arg(&p)
                .arg(&theta_base)
                .launch(cfg)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Argmax over f16 vector on GPU. Returns index of max element.
    pub fn argmax_f16(
        &self,
        x: &CudaSlice<half::f16>,
        out: &mut CudaSlice<i32>,
        n: u32,
    ) -> Result<(), KernelError> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut args: [*mut c_void; 3] = [
            slice_ptr(x), slice_ptr_mut(out), scalar_ptr(&n_i),
        ];
        unsafe { self.fast.fire(&self.argmax_f16, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// GPU Top-K + softmax + sampling. Returns sampled token index.
    pub fn sample_top_k(
        &self,
        logits: &CudaSlice<half::f16>,
        out: &mut CudaSlice<i32>,
        vocab_size: u32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        random_seed: u32,
    ) -> Result<(), KernelError> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let vs = vocab_size as i32;
        let tk = top_k as i32;
        unsafe {
            self.stream
                .launch_builder(&self.sample_top_k)
                .arg(logits)
                .arg(out)
                .arg(&vs)
                .arg(&temperature)
                .arg(&tk)
                .arg(&top_p)
                .arg(&random_seed)
                .launch(cfg)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    // === CUDA Graph-compatible kernel wrappers ===

    /// RoPE reading pos from device memory (decode_params[0]).
    /// For CUDA Graph capture — pos changes each step but pointer is stable.
    pub fn rope_graph(
        &self,
        x: &mut CudaSlice<half::f16>,
        decode_params: &CudaSlice<i32>,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        theta_base: f32,
    ) -> Result<(), KernelError> {
        let cfg = LaunchConfig {
            grid_dim: (seq_len, n_heads, 1),
            block_dim: (head_dim / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        let hd = head_dim as i32;
        let nh = n_heads as i32;
        let mut args: [*mut c_void; 5] = [
            slice_ptr_mut(x), slice_ptr(decode_params),
            scalar_ptr(&hd), scalar_ptr(&nh), scalar_ptr(&theta_base),
        ];
        unsafe { self.fast.fire(&self.rope_graph, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Copy f16 with offset read from decode_params[2].
    /// Writes src[0..n] to dst_base[offset..offset+n].
    /// For KV cache writes in CUDA Graph mode.
    pub fn copy_f16_with_offset(
        &self,
        src: &CudaSlice<half::f16>,
        dst_base: &mut CudaSlice<half::f16>,
        decode_params: &CudaSlice<i32>,
        n: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut args: [*mut c_void; 4] = [
            slice_ptr(src), slice_ptr_mut(dst_base), slice_ptr(decode_params),
            scalar_ptr(&n_i),
        ];
        unsafe { self.fast.fire(&self.copy_f16_with_offset, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Tiled MHA reading kv_len/pos from device memory. Uses base KV cache pointers.
    /// Fixed shared memory: (MHA_TILE_KV + threads) * 4 bytes — independent of kv_len.
    /// This allows CUDA Graph capture without occupancy degradation.
    pub fn mha_fused_graph(
        &self,
        q: &CudaSlice<half::f16>,
        k_base: &CudaSlice<half::f16>,
        v_base: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        decode_params: &CudaSlice<i32>,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        seq_len: u32,
        _max_kv_len: u32,  // unused now — smem is fixed
    ) -> Result<(), KernelError> {
        let threads = 128u32.min(head_dim);
        // Fixed smem: TILE_KV(128) + threads for scratch
        let tile_kv = 128u32;
        let smem = (tile_kv + threads) * 4;
        let hd = head_dim as i32;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let sl = seq_len as i32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut args: [*mut c_void; 10] = [
            slice_ptr(q), slice_ptr(k_base), slice_ptr(v_base), slice_ptr_mut(out),
            slice_ptr(decode_params),
            scalar_ptr(&hd), scalar_ptr(&nh), scalar_ptr(&nkv),
            scalar_ptr(&sl), scalar_ptr(&scale),
        ];
        unsafe { self.fast.fire(&self.mha_fused_graph, (n_heads, seq_len, 1), (threads, 1, 1), smem, &mut args); }
        Ok(())
    }

    // =============================================================
    // Gemma 4 kernel wrappers
    // =============================================================

    /// GELU activation: out = GELU(gate) * up, element-wise, all f16.
    pub fn gelu(
        &self,
        gate: &CudaSlice<half::f16>,
        up: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        n: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut args: [*mut c_void; 4] = [
            slice_ptr(gate), slice_ptr(up), slice_ptr_mut(out),
            scalar_ptr(&n_i),
        ];
        unsafe { self.fast.fire(&self.gelu, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// RMSNorm without weight: just normalize by RMS. f16 in/out.
    pub fn rms_norm_no_weight(
        &self,
        x: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        n_rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<(), KernelError> {
        let threads = 256u32.min(dim);
        let cfg = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * 4,
        };
        let dim_i = dim as i32;
        let mut args: [*mut c_void; 4] = [
            slice_ptr(x), slice_ptr_mut(out),
            scalar_ptr(&dim_i), scalar_ptr(&eps),
        ];
        unsafe { self.fast.fire(&self.rms_norm_no_weight, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Scale f16 tensor by scalar: out[i] = x[i] * scale.
    pub fn scale_f16(
        &self,
        x: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        n: u32,
        scale: f32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut args: [*mut c_void; 4] = [
            slice_ptr(x), slice_ptr_mut(out),
            scalar_ptr(&n_i), scalar_ptr(&scale),
        ];
        unsafe { self.fast.fire(&self.scale_f16, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Scale f32 tensor in-place: x[i] *= scale.
    pub fn scale_f32_inplace(
        &self,
        x: &mut CudaSlice<f32>,
        n: u32,
        scale: f32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut args: [*mut c_void; 3] = [
            slice_ptr_mut(x), scalar_ptr(&n_i), scalar_ptr(&scale),
        ];
        unsafe { self.fast.fire(&self.scale_f32_inplace, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Logit softcapping: out = tanh(x / cap) * cap. In-place on f16.
    pub fn logit_softcap(
        &self,
        x: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        n: u32,
        cap: f32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut args: [*mut c_void; 4] = [
            slice_ptr(x), slice_ptr_mut(out),
            scalar_ptr(&n_i), scalar_ptr(&cap),
        ];
        unsafe { self.fast.fire(&self.logit_softcap, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Logit softcap in-place: x = tanh(x/cap) * cap
    pub fn logit_softcap_inplace(
        &self,
        x: &mut CudaSlice<half::f16>,
        n: u32,
        cap: f32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut args: [*mut c_void; 3] = [
            slice_ptr_mut(x),
            scalar_ptr(&n_i), scalar_ptr(&cap),
        ];
        unsafe { self.fast.fire(&self.logit_softcap_inplace, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// RoPE NeoX-style: pairs are (x[i], x[i+d/2]) instead of (x[2i], x[2i+1]).
    /// x shape: [seq_len, n_heads, head_dim]
    pub fn rope_neox(
        &self,
        x: &mut CudaSlice<half::f16>,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        pos: u32,
        theta_base: f32,
    ) -> Result<(), KernelError> {
        let cfg = LaunchConfig {
            grid_dim: (seq_len, n_heads, 1),
            block_dim: (head_dim / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        let hd = head_dim as i32;
        let nh = n_heads as i32;
        let p = pos as i32;
        let mut args: [*mut c_void; 5] = [
            slice_ptr_mut(x), scalar_ptr(&hd), scalar_ptr(&nh),
            scalar_ptr(&p), scalar_ptr(&theta_base),
        ];
        unsafe { self.fast.fire(&self.rope_neox, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// RoPE NeoX with proportional frequency factors (for Gemma 4 full-attention layers).
    /// freq_factors shape: [head_dim/2], values 1.0 (rotate) or 1e30 (identity).
    pub fn rope_neox_freqs(
        &self,
        x: &mut CudaSlice<half::f16>,
        freq_factors: &CudaSlice<f32>,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        pos: u32,
        theta_base: f32,
    ) -> Result<(), KernelError> {
        let cfg = LaunchConfig {
            grid_dim: (seq_len, n_heads, 1),
            block_dim: (head_dim / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        let hd = head_dim as i32;
        let nh = n_heads as i32;
        let p = pos as i32;
        let mut args: [*mut c_void; 6] = [
            slice_ptr_mut(x), slice_ptr(freq_factors),
            scalar_ptr(&hd), scalar_ptr(&nh),
            scalar_ptr(&p), scalar_ptr(&theta_base),
        ];
        unsafe { self.fast.fire(&self.rope_neox_freqs, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// RoPE NeoX graph-compatible (reads pos from device memory).
    pub fn rope_neox_graph(
        &self,
        x: &mut CudaSlice<half::f16>,
        decode_params: &CudaSlice<i32>,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        theta_base: f32,
    ) -> Result<(), KernelError> {
        let cfg = LaunchConfig {
            grid_dim: (seq_len, n_heads, 1),
            block_dim: (head_dim / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        let hd = head_dim as i32;
        let nh = n_heads as i32;
        let mut args: [*mut c_void; 5] = [
            slice_ptr_mut(x), slice_ptr(decode_params),
            scalar_ptr(&hd), scalar_ptr(&nh), scalar_ptr(&theta_base),
        ];
        unsafe { self.fast.fire(&self.rope_neox_graph, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Post-norm fused add: norm_out = rmsnorm(delta) * weight, hidden += norm_out.
    /// For Gemma 4 post-attention/post-FFN norms.
    pub fn post_norm_add(
        &self,
        hidden: &mut CudaSlice<f32>,
        delta: &CudaSlice<half::f16>,
        weight: &CudaSlice<half::f16>,
        norm_out: &mut CudaSlice<half::f16>,
        n_rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<(), KernelError> {
        let threads = 256u32.min(dim);
        let cfg = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * 4,
        };
        let dim_i = dim as i32;
        let mut args: [*mut c_void; 6] = [
            slice_ptr_mut(hidden), slice_ptr(delta), slice_ptr(weight),
            slice_ptr_mut(norm_out), scalar_ptr(&dim_i), scalar_ptr(&eps),
        ];
        unsafe { self.fast.fire(&self.post_norm_add, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Element-wise multiply: out = a * b, all f16.
    pub fn mul_f16(
        &self,
        a: &CudaSlice<half::f16>,
        b: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        n: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut args: [*mut c_void; 4] = [
            slice_ptr(a), slice_ptr(b), slice_ptr_mut(out),
            scalar_ptr(&n_i),
        ];
        unsafe { self.fast.fire(&self.mul_f16, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Standalone GELU activation: out = GELU(x), all f16.
    /// Unlike `gelu()` which computes GELU(gate)*up, this just applies GELU.
    pub fn gelu_act(
        &self,
        x: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        n: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut args: [*mut c_void; 3] = [
            slice_ptr(x), slice_ptr_mut(out),
            scalar_ptr(&n_i),
        ];
        unsafe { self.fast.fire(&self.gelu_act, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Gather rows from a quantized tensor by token ID.
    /// Copies selected rows into a contiguous output buffer for subsequent dequantization.
    ///
    /// src: quantized tensor data on GPU
    /// token_ids: [n_tokens] i32 on GPU
    /// dst: output buffer, must be at least n_tokens * row_bytes
    /// row_bytes: bytes per row in the quantized format
    /// n_tokens: number of tokens to gather
    pub fn gather_rows_quant(
        &self,
        src: &CudaSlice<u8>,
        token_ids: &CudaSlice<i32>,
        dst: &mut CudaSlice<u8>,
        row_bytes: u32,
        n_tokens: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32;
        let y_blocks = (row_bytes + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (n_tokens, y_blocks, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let rb = row_bytes as i32;
        let nt = n_tokens as i32;
        let mut args: [*mut c_void; 5] = [
            slice_ptr(src), slice_ptr(token_ids), slice_ptr_mut(dst),
            scalar_ptr(&rb), scalar_ptr(&nt),
        ];
        unsafe { self.fast.fire(&self.gather_rows_quant, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Per-layer embedding strided multiply.
    /// For each token t and dim j:
    ///   out[t*epl + j] = a[t*epl + j] * embd[t*row_width + layer_off + j]
    ///
    /// a, out: [n_tokens, epl] contiguous f16
    /// embd: [n_tokens, row_width] contiguous f16
    pub fn pe_strided_mul(
        &self,
        a: &CudaSlice<half::f16>,
        embd: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        epl: u32,
        row_width: u32,
        layer_off: u32,
        n_tokens: u32,
    ) -> Result<(), KernelError> {
        let total = n_tokens * epl;
        let threads = 256u32;
        let blocks = (total + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let epl_i = epl as i32;
        let rw_i = row_width as i32;
        let lo_i = layer_off as i32;
        let nt_i = n_tokens as i32;
        let mut args: [*mut c_void; 7] = [
            slice_ptr(a), slice_ptr(embd), slice_ptr_mut(out),
            scalar_ptr(&epl_i), scalar_ptr(&rw_i),
            scalar_ptr(&lo_i), scalar_ptr(&nt_i),
        ];
        unsafe { self.fast.fire(&self.pe_strided_mul, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Fused RoPE Q+K + KV cache write (4 ops → 1 launch).
    /// Takes BASE pointers for K/V cache to avoid Rust double-borrow issues.
    pub fn rope_kv_write(
        &self,
        q: &mut CudaSlice<half::f16>,
        k: &mut CudaSlice<half::f16>,
        v: &CudaSlice<half::f16>,
        k_cache_base: &mut CudaSlice<half::f16>,
        v_cache_base: &mut CudaSlice<half::f16>,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        seq_len: u32,
        pos: u32,
        theta_base: f32,
        kv_stride: u32,
        kv_offset: u32,
    ) -> Result<(), KernelError> {
        let threads = 256u32.min(head_dim / 2);
        let cfg = LaunchConfig {
            grid_dim: (seq_len, n_heads + n_kv_heads, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let hd = head_dim as i32;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let p = pos as i32;
        let kvs = kv_stride as i32;
        let kvo = kv_offset as i32;
        let mut args: [*mut c_void; 12] = [
            slice_ptr_mut(q), slice_ptr_mut(k), slice_ptr(v),
            slice_ptr_mut(k_cache_base), slice_ptr_mut(v_cache_base),
            scalar_ptr(&hd), scalar_ptr(&nh), scalar_ptr(&nkv),
            scalar_ptr(&p), scalar_ptr(&theta_base),
            scalar_ptr(&kvs), scalar_ptr(&kvo),
        ];
        unsafe { self.fast.fire(&self.rope_kv_write, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }
        Ok(())
    }

    /// Computes Q@K^T/sqrt(head_dim), causal mask, softmax, @V — all fused.
    pub fn mha_fused(
        &self,
        q: &CudaSlice<half::f16>,
        k: &CudaView<'_, half::f16>,
        v: &CudaView<'_, half::f16>,
        out: &mut CudaSlice<half::f16>,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        seq_len: u32,
        kv_len: u32,
        pos_offset: u32,
    ) -> Result<(), KernelError> {
        self.mha_fused_scaled(q, k, v, out, head_dim, n_heads, n_kv_heads,
            seq_len, kv_len, pos_offset, 1.0 / (head_dim as f32).sqrt())
    }

    /// MHA with custom attention scale (Gemma 4 uses scale=1.0).
    pub fn mha_fused_scaled(
        &self,
        q: &CudaSlice<half::f16>,
        k: &CudaView<'_, half::f16>,
        v: &CudaView<'_, half::f16>,
        out: &mut CudaSlice<half::f16>,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        seq_len: u32,
        kv_len: u32,
        pos_offset: u32,
        scale: f32,
    ) -> Result<(), KernelError> {
        // Flash Attention: 2D block (32 lanes, 4 warps)
        let smem = (8 + 4 * head_dim) * 4; // max/sum slots + VKQ combine buffer
        let cfg = LaunchConfig {
            grid_dim: (n_heads, seq_len, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: smem,
        };

        let hd = head_dim as i32;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let sl = seq_len as i32;
        let kvl = kv_len as i32;
        let po = pos_offset as i32;

        let mut args: [*mut c_void; 11] = [
            slice_ptr(q), view_ptr(k), view_ptr(v), slice_ptr_mut(out),
            scalar_ptr(&hd), scalar_ptr(&nh), scalar_ptr(&nkv),
            scalar_ptr(&sl), scalar_ptr(&kvl), scalar_ptr(&po),
            scalar_ptr(&scale),
        ];
        unsafe { self.fast.fire(&self.mha_fused, (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2), (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2), cfg.shared_mem_bytes, &mut args); }

        Ok(())
    }

    // --- C dispatch accessors: expose CudaFunction handles for the C layer ---

    pub fn rms_norm_f32in_q8_fn(&self) -> &CudaFunction { &self.rms_norm_f32in_q8 }
    pub fn fused_add_rmsnorm_q8_fn(&self) -> &CudaFunction { &self.fused_add_rmsnorm_q8 }
    pub fn rope_fn(&self) -> &CudaFunction { &self.rope }
    pub fn copy_f16_fn(&self) -> &CudaFunction { &self.copy_f16 }
    pub fn mha_fused_fn(&self) -> &CudaFunction { &self.mha_fused }
    pub fn silu_fn(&self) -> &CudaFunction { &self.silu }
    pub fn add_inplace_f32_f16_fn(&self) -> &CudaFunction { &self.add_inplace_f32_f16 }
    pub fn rms_norm_f32in_fn(&self) -> &CudaFunction { &self.rms_norm_f32in }
    pub fn argmax_f16_fn(&self) -> &CudaFunction { &self.argmax_f16 }
}
