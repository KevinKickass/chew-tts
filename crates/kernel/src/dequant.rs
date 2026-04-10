use crate::loader::{self, KernelError};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use std::sync::Arc;

const DEQUANT_CU: &str = include_str!("cuda/dequant.cu");

/// Dequantization kernel handles — one set per GPU.
pub struct DequantKernels {
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    q8_0: CudaFunction,
    q4_0: CudaFunction,
    q5_1: CudaFunction,
    q4_k: CudaFunction,
    q5_k: CudaFunction,
    q6_k: CudaFunction,
    q2_k: CudaFunction,
    q3_k: CudaFunction,
    bf16_fn: CudaFunction,
    iq2_s: CudaFunction,
    iq3_xxs: CudaFunction,
    iq3_s: CudaFunction,
    iq4_xs: CudaFunction,
    f16_fn: CudaFunction,
    f32_fn: CudaFunction,
}

impl DequantKernels {
    pub fn load(stream: &Arc<CudaStream>) -> Result<Self, KernelError> {
        let module = loader::load_module_from_source(stream, DEQUANT_CU, "dequant")?;

        Ok(Self {
            stream: Arc::clone(stream),
            q8_0: loader::get_fn(&module, "dequant_q8_0")?,
            q4_0: loader::get_fn(&module, "dequant_q4_0")?,
            q5_1: loader::get_fn(&module, "dequant_q5_1")?,
            q4_k: loader::get_fn(&module, "dequant_q4_k")?,
            q5_k: loader::get_fn(&module, "dequant_q5_k")?,
            q6_k: loader::get_fn(&module, "dequant_q6_k")?,
            q2_k: loader::get_fn(&module, "dequant_q2_k")?,
            q3_k: loader::get_fn(&module, "dequant_q3_k")?,
            bf16_fn: loader::get_fn(&module, "dequant_bf16")?,
            iq2_s: loader::get_fn(&module, "dequant_iq2_s")?,
            iq3_xxs: loader::get_fn(&module, "dequant_iq3_xxs")?,
            iq3_s: loader::get_fn(&module, "dequant_iq3_s")?,
            iq4_xs: loader::get_fn(&module, "dequant_iq4_xs")?,
            f16_fn: loader::get_fn(&module, "dequant_f16")?,
            f32_fn: loader::get_fn(&module, "dequant_f32")?,
            _module: module,
        })
    }

    /// Dequantize quantized data on GPU to f16 output buffer.
    ///
    /// `src` contains the raw quantized blocks already on GPU.
    /// `dst` is pre-allocated f16 output (n_elements * 2 bytes).
    pub fn dequant(
        &self,
        src: &CudaSlice<u8>,
        dst: &mut CudaSlice<half::f16>,
        n_elements: u32,
        quant_type: chew_gguf::GgmlType,
    ) -> Result<(), KernelError> {
        let (kernel, cfg, n) = self.prepare_dequant(n_elements, quant_type)?;
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(src)
                .arg(dst)
                .arg(&n)
                .launch(cfg)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Dequantize a view (slice) of quantized data — used by chunked GEMM.
    pub fn dequant_view(
        &self,
        src: &CudaView<'_, u8>,
        dst: &mut CudaSlice<half::f16>,
        n_elements: u32,
        quant_type: chew_gguf::GgmlType,
    ) -> Result<(), KernelError> {
        let (kernel, cfg, n) = self.prepare_dequant(n_elements, quant_type)?;
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(src)
                .arg(dst)
                .arg(&n)
                .launch(cfg)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Dequantize into a mutable view of an output buffer.
    pub fn dequant_to_view(
        &self,
        src: &CudaView<'_, u8>,
        dst: &mut CudaViewMut<'_, half::f16>,
        n_elements: u32,
        quant_type: chew_gguf::GgmlType,
    ) -> Result<(), KernelError> {
        let (kernel, cfg, n) = self.prepare_dequant(n_elements, quant_type)?;
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(src)
                .arg(dst)
                .arg(&n)
                .launch(cfg)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    fn prepare_dequant(
        &self,
        n_elements: u32,
        quant_type: chew_gguf::GgmlType,
    ) -> Result<(&CudaFunction, LaunchConfig, i32), KernelError> {
        let threads = 256u32;
        let blocks = (n_elements + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = match quant_type {
            chew_gguf::GgmlType::Q8_0 => &self.q8_0,
            chew_gguf::GgmlType::Q4_0 => &self.q4_0,
            chew_gguf::GgmlType::Q5_1 => &self.q5_1,
            chew_gguf::GgmlType::Q4_K => &self.q4_k,
            chew_gguf::GgmlType::Q5_K => &self.q5_k,
            chew_gguf::GgmlType::Q6_K => &self.q6_k,
            chew_gguf::GgmlType::Q2_K => &self.q2_k,
            chew_gguf::GgmlType::Q3_K => &self.q3_k,
            chew_gguf::GgmlType::BF16 => &self.bf16_fn,
            chew_gguf::GgmlType::IQ2_S => &self.iq2_s,
            chew_gguf::GgmlType::IQ3_XXS => &self.iq3_xxs,
            chew_gguf::GgmlType::IQ3_S => &self.iq3_s,
            chew_gguf::GgmlType::IQ4_XS => &self.iq4_xs,
            chew_gguf::GgmlType::F16 => &self.f16_fn,
            chew_gguf::GgmlType::F32 => &self.f32_fn,
            other => {
                return Err(KernelError::Launch(format!(
                    "no GPU dequant kernel for {other}"
                )))
            }
        };

        Ok((kernel, cfg, n_elements as i32))
    }
}
