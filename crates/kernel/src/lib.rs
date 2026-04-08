mod loader;
mod dequant;
mod ops;
mod gemm;
mod gemv;

pub use dequant::DequantKernels;
pub use ops::OpsKernels;
pub use gemm::Gemm;
pub use gemv::GemvKernels;
pub use loader::KernelError;

use cudarc::driver::CudaStream;
use std::sync::Arc;

/// All GPU kernels for one device, ready to launch.
pub struct GpuKernels {
    pub dequant: DequantKernels,
    pub ops: OpsKernels,
    pub gemm: Gemm,
    pub gemv: GemvKernels,
}

impl GpuKernels {
    /// Load all kernels onto a GPU stream.
    ///
    /// `max_weight_elements`: largest single weight matrix element count.
    /// Used to size the dequant scratch buffer for on-the-fly GEMM.
    pub fn load(stream: &Arc<CudaStream>, max_weight_elements: usize, max_k: usize) -> Result<Self, KernelError> {
        Ok(Self {
            dequant: DequantKernels::load(stream)?,
            ops: OpsKernels::load(stream)?,
            gemm: Gemm::new(stream, max_weight_elements)?,
            gemv: GemvKernels::load(stream, max_k)?,
        })
    }
}
