use crate::loader::{self, KernelError};
use crate::fast_launch::{FastStream, slice_ptr, slice_ptr_mut, scalar_ptr};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

const GEMV_CU: &str = include_str!("cuda/gemv.cu");

/// Fused quantized GEMV kernels for decode (M=1).
///
/// Two-phase: quantize x to Q8_1 once, then GEMV reads int8 input.
pub struct GemvKernels {
    fast: FastStream,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    quantize_x: CudaFunction,
    q4_k: CudaFunction,
    dual_q4_k: CudaFunction,
    q6_k: CudaFunction,
    q8_0: CudaFunction,
    /// Pre-allocated Q8_1 buffer for input vector (max_k / 32 * 36 bytes)
    /// Q8_1 format: half2 ds (4 bytes) + int8_t qs[32] = 36 bytes per block
    x_q8: CudaSlice<u8>,
}

impl GemvKernels {
    pub fn load(stream: &Arc<CudaStream>, max_k: usize) -> Result<Self, KernelError> {
        let module = loader::load_module_from_source(stream, GEMV_CU, "gemv")?;

        // Q8_1: 36 bytes per 32 elements (half2 ds + int8_t qs[32])
        let q8_bytes = (max_k / 32) * 36;
        let x_q8 = stream.alloc_zeros::<u8>(q8_bytes)
            .map_err(|e| KernelError::Launch(e.to_string()))?;

        Ok(Self {
            fast: FastStream::new(stream),
            stream: Arc::clone(stream),
            quantize_x: loader::get_fn(&module, "quantize_x_q8_1")?,
            q4_k: loader::get_fn(&module, "gemv_q4_k")?,
            dual_q4_k: loader::get_fn(&module, "gemv_dual_q4_k")?,
            q6_k: loader::get_fn(&module, "gemv_q6_k")?,
            q8_0: loader::get_fn(&module, "gemv_q8_0")?,
            _module: module,
            x_q8,
        })
    }

    /// Mutable reference to the pre-allocated Q8_1 input buffer.
    /// Used by fused norm+quantize kernels that write Q8_1 directly.
    pub fn x_q8_mut(&mut self) -> &mut CudaSlice<u8> {
        &mut self.x_q8
    }

    /// Quantize input vector x (f16) to Q8_1 format.
    /// Must be called once before any gemv calls with the same x.
    pub fn quantize_input(&mut self, x: &CudaSlice<half::f16>, k: u32) -> Result<(), KernelError> {
        let cfg = LaunchConfig {
            grid_dim: ((k + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let k_i32 = k as i32;
        let mut args = [
            slice_ptr(x), slice_ptr_mut(&mut self.x_q8), scalar_ptr(&k_i32),
        ];
        unsafe { self.fast.launch(&self.quantize_x, cfg, &mut args) }
    }

    /// Fused quantized GEMV: out[N] = W[N,K] @ x[K]
    /// Assumes quantize_input was called with the current x.
    pub fn gemv(
        &self,
        w: &CudaSlice<u8>,
        x: &CudaSlice<half::f16>,
        out: &mut CudaSlice<half::f16>,
        n: u32,
        k: u32,
        quant_type: chew_gguf::GgmlType,
    ) -> Result<bool, KernelError> {
        let kernel = match quant_type {
            chew_gguf::GgmlType::Q4_K => &self.q4_k,
            chew_gguf::GgmlType::Q6_K => &self.q6_k,
            chew_gguf::GgmlType::Q8_0 => &self.q8_0,
            _ => return Ok(false),
        };

        // 1 row per block, 128 threads (4 warps)
        let cfg = LaunchConfig {
            grid_dim: (n, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let k_i32 = k as i32;

        let mut args = [
            slice_ptr(w), slice_ptr(x), slice_ptr_mut(out),
            scalar_ptr(&n_i32), scalar_ptr(&k_i32), slice_ptr(&self.x_q8),
        ];
        unsafe { self.fast.launch(kernel, cfg, &mut args)?; }
        Ok(true)
    }

    /// Fused dual GEMV: compute gate[N,K] and up[N,K] in one kernel.
    /// Both share the same Q8_1 input, saving 1 launch + Q8_1 reads.
    /// Only for Q4_K. Assumes quantize_input was called.
    pub fn gemv_dual(
        &self,
        w_gate: &CudaSlice<u8>,
        w_up: &CudaSlice<u8>,
        out_gate: &mut CudaSlice<half::f16>,
        out_up: &mut CudaSlice<half::f16>,
        n: u32,
        k: u32,
        quant_type: chew_gguf::GgmlType,
    ) -> Result<bool, KernelError> {
        if quant_type != chew_gguf::GgmlType::Q4_K {
            return Ok(false);
        }

        let cfg = LaunchConfig {
            grid_dim: (n, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let k_i32 = k as i32;

        let mut args: [*mut std::ffi::c_void; 7] = [
            slice_ptr(w_gate), slice_ptr(w_up),
            slice_ptr_mut(out_gate), slice_ptr_mut(out_up),
            scalar_ptr(&n_i32), scalar_ptr(&k_i32), slice_ptr(&self.x_q8),
        ];
        unsafe { self.fast.launch(&self.dual_q4_k, cfg, &mut args)?; }
        Ok(true)
    }

}
