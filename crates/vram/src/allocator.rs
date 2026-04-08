use cudarc::driver::{CudaContext, CudaStream, DevicePtr};
use std::sync::Arc;
use tracing::{debug, info};

/// A GPU memory allocation owned by us.
///
/// RAII: Drop frees the CUDA memory. Nobody else touches GPU memory.
pub struct GpuBuffer {
    ptr: cudarc::driver::CudaSlice<u8>,
    stream: Arc<CudaStream>,
    size: u64,
    gpu_idx: usize,
}

impl GpuBuffer {
    pub fn device_ptr(&self) -> u64 {
        let (ptr, _sync) = self.ptr.device_ptr(&self.stream);
        ptr
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn gpu_idx(&self) -> usize {
        self.gpu_idx
    }

    /// Get a reference to the underlying CudaSlice for kernel launches.
    pub fn as_cuda_slice(&self) -> &cudarc::driver::CudaSlice<u8> {
        &self.ptr
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

/// Low-level VRAM allocator — wraps cudarc for direct GPU memory control.
///
/// This is the ONLY code that calls cuMemAlloc/cuMemFree.
/// Everything else gets a GpuBuffer from here.
pub struct VramAllocator {
    streams: Vec<Arc<CudaStream>>,
}

impl VramAllocator {
    /// Initialize by probing all CUDA devices.
    pub fn init() -> Result<Self, VramError> {
        cudarc::driver::result::init().map_err(|e| VramError::CudaInit(e.to_string()))?;

        let count = cudarc::driver::result::device::get_count()
            .map_err(|e| VramError::CudaInit(e.to_string()))? as usize;

        if count == 0 {
            return Err(VramError::NoGpus);
        }

        let mut streams = Vec::with_capacity(count);
        for i in 0..count {
            let ctx = CudaContext::new(i).map_err(|e| VramError::CudaInit(e.to_string()))?;
            let stream = ctx.new_stream().map_err(|e| VramError::CudaInit(e.to_string()))?;

            let (free, total) = ctx
                .mem_get_info()
                .map_err(|e| VramError::CudaInit(e.to_string()))?;

            info!(
                gpu = i,
                total_mb = total / (1024 * 1024),
                free_mb = free / (1024 * 1024),
                "GPU detected"
            );

            streams.push(stream);
        }

        Ok(Self { streams })
    }

    /// Number of GPUs.
    pub fn gpu_count(&self) -> usize {
        self.streams.len()
    }

    /// Query free VRAM on a GPU (hardware truth).
    pub fn free_bytes(&self, gpu_idx: usize) -> Result<u64, VramError> {
        let ctx = self.streams[gpu_idx].context();
        let (free, _) = ctx
            .mem_get_info()
            .map_err(|e| VramError::Alloc(e.to_string()))?;
        Ok(free as u64)
    }

    /// Query total VRAM on a GPU.
    pub fn total_bytes(&self, gpu_idx: usize) -> Result<u64, VramError> {
        let ctx = self.streams[gpu_idx].context();
        let (_, total) = ctx
            .mem_get_info()
            .map_err(|e| VramError::Alloc(e.to_string()))?;
        Ok(total as u64)
    }

    /// Allocate GPU memory. Returns an owned GpuBuffer (freed on Drop).
    pub fn alloc(&self, gpu_idx: usize, size: u64) -> Result<GpuBuffer, VramError> {
        let stream = &self.streams[gpu_idx];

        let ptr = stream
            .alloc_zeros::<u8>(size as usize)
            .map_err(|e| VramError::Alloc(e.to_string()))?;

        debug!(gpu = gpu_idx, size_mb = size / (1024 * 1024), "VRAM allocated");

        Ok(GpuBuffer {
            ptr,
            stream: Arc::clone(stream),
            size,
            gpu_idx,
        })
    }

    /// Copy data from host (CPU) to a GPU buffer.
    pub fn upload(&self, buf: &mut GpuBuffer, data: &[u8]) -> Result<(), VramError> {
        let stream = &self.streams[buf.gpu_idx];
        stream
            .memcpy_htod(data, &mut buf.ptr)
            .map_err(|e| VramError::Alloc(e.to_string()))?;
        Ok(())
    }

    /// Get the CudaStream for a GPU.
    pub fn stream(&self, gpu_idx: usize) -> &Arc<CudaStream> {
        &self.streams[gpu_idx]
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VramError {
    #[error("CUDA init failed: {0}")]
    CudaInit(String),
    #[error("no GPUs found")]
    NoGpus,
    #[error("allocation failed: {0}")]
    Alloc(String),
    #[error("GPU index {0} out of range")]
    InvalidGpu(usize),
    #[error("not enough VRAM: need {need_mb} MB, have {free_mb} MB on GPU {gpu_idx}")]
    OutOfMemory {
        gpu_idx: usize,
        need_mb: u64,
        free_mb: u64,
    },
}
