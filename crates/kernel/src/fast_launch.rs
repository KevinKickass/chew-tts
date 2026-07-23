//! Zero-allocation kernel launching for the decode hot path.
//!
//! Bypasses cudarc's `launch_builder()` pattern which allocates 3 `Vec` per call.
//! Instead, builds a stack-allocated arg array and calls `cuLaunchKernel` directly.
//!
//! Requires the cudarc fork with public `cu_function`, `cu_stream`, `cu_device_ptr`, `ptr` fields.

use cudarc::driver::sys;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig};
use std::ffi::c_void;
use std::sync::Arc;

use crate::loader::KernelError;

/// Cached raw CUDA stream handle for fast kernel launches.
pub struct FastStream {
    raw: sys::CUstream,
}

// SAFETY: CUstream handles are thread-safe in the CUDA runtime.
// The parent CudaStream (which is Send+Sync) owns the lifetime;
// we only cache the raw handle for zero-alloc launches.
unsafe impl Send for FastStream {}
unsafe impl Sync for FastStream {}

impl FastStream {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            raw: stream.cu_stream,
        }
    }

    /// Launch a kernel with pre-built args. ZERO heap allocation.
    #[inline(always)]
    pub unsafe fn launch(
        &self,
        func: &CudaFunction,
        cfg: LaunchConfig,
        args: &mut [*mut c_void],
    ) -> Result<(), KernelError> {
        unsafe {
            sys::cuLaunchKernel(
                func.cu_function,
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes,
                self.raw,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .result()
            .map_err(|e| KernelError::Launch(e.to_string()))
        }
    }

    /// Fire-and-forget launch. No error check — caller must sync+check later.
    /// Use for hot loops where per-launch error checking costs too much.
    #[inline(always)]
    pub unsafe fn fire(
        &self,
        func: &CudaFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        smem: u32,
        args: &mut [*mut c_void],
    ) {
        unsafe {
            sys::cuLaunchKernel(
                func.cu_function,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                smem,
                self.raw,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
        }
        // Intentionally ignoring CUresult — check with sync_check() at end
    }

    /// Synchronize stream and check for any deferred errors.
    pub fn sync_check(&self) -> Result<(), KernelError> {
        unsafe {
            sys::cuStreamSynchronize(self.raw)
                .result()
                .map_err(|e| KernelError::Launch(format!("stream sync: {e}")))?;
        }
        Ok(())
    }
}

/// Helper to get a raw device pointer from a CudaSlice for kernel args.
#[inline(always)]
pub fn slice_ptr<T>(s: &CudaSlice<T>) -> *mut c_void {
    &s.cu_device_ptr as *const sys::CUdeviceptr as *mut c_void
}

/// Helper to get a raw device pointer from a mutable CudaSlice.
#[inline(always)]
pub fn slice_ptr_mut<T>(s: &mut CudaSlice<T>) -> *mut c_void {
    &s.cu_device_ptr as *const sys::CUdeviceptr as *mut c_void
}

/// Helper to get a raw device pointer from a CudaView.
#[inline(always)]
pub fn view_ptr<T>(v: &CudaView<'_, T>) -> *mut c_void {
    &v.ptr as *const sys::CUdeviceptr as *mut c_void
}

/// Helper to get a raw device pointer from a CudaViewMut.
#[inline(always)]
pub fn view_mut_ptr<T>(v: &mut CudaViewMut<'_, T>) -> *mut c_void {
    &v.ptr as *const sys::CUdeviceptr as *mut c_void
}

/// Helper to get a pointer to a scalar for kernel args.
#[inline(always)]
pub fn scalar_ptr<T>(v: &T) -> *mut c_void {
    v as *const T as *mut c_void
}
