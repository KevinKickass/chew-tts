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
        unsafe { self.fast.launch(&self.rms_norm_f32in, cfg, &mut args)? }

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
        unsafe { self.fast.launch(&self.rope, cfg, &mut args)? }

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
        unsafe { self.fast.launch(&self.silu, cfg, &mut args)? }

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
        unsafe { self.fast.launch(&self.fused_add_rmsnorm, cfg, &mut args)? }
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
        unsafe { self.fast.launch(&self.rms_norm_f32in_q8, cfg, &mut args)? }
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
        unsafe { self.fast.launch(&self.fused_add_rmsnorm_q8, cfg, &mut args)? }
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
        unsafe { self.fast.launch(&self.add_inplace_f32_f16, cfg, &mut args)? }
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
        unsafe { self.fast.launch(&self.copy_f16, cfg, &mut args)? }

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
        unsafe { self.fast.launch(&self.argmax_f16, cfg, &mut args)? }
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
        unsafe { self.fast.launch(&self.rope_graph, cfg, &mut args)? }
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
        unsafe { self.fast.launch(&self.copy_f16_with_offset, cfg, &mut args)? }
        Ok(())
    }

    /// MHA reading kv_len/pos from device memory. Uses base KV cache pointers.
    /// shared_mem_bytes must be pre-computed for max kv_len (allocated at capture time).
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
        max_kv_len: u32,
    ) -> Result<(), KernelError> {
        let threads = 128u32.min(head_dim);
        // Allocate shared memory for max_kv_len (graph-stable)
        let smem = (max_kv_len + threads) * 4;
        let cfg = LaunchConfig {
            grid_dim: (n_heads, seq_len, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: smem,
        };
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
        unsafe { self.fast.launch(&self.mha_fused_graph, cfg, &mut args)? }
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
        let threads = 128u32.min(head_dim);
        // Shared memory: kv_len floats (scores) + threads floats (reduction scratch)
        let smem = (kv_len + threads) * 4;
        let cfg = LaunchConfig {
            grid_dim: (n_heads, seq_len, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: smem,
        };

        let hd = head_dim as i32;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let sl = seq_len as i32;
        let kvl = kv_len as i32;
        let po = pos_offset as i32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut args: [*mut c_void; 11] = [
            slice_ptr(q), view_ptr(k), view_ptr(v), slice_ptr_mut(out),
            scalar_ptr(&hd), scalar_ptr(&nh), scalar_ptr(&nkv),
            scalar_ptr(&sl), scalar_ptr(&kvl), scalar_ptr(&po),
            scalar_ptr(&scale),
        ];
        unsafe { self.fast.launch(&self.mha_fused, cfg, &mut args)? }

        Ok(())
    }
}
