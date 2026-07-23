use crate::allocator::{GpuBuffer, VramAllocator, VramError};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

const SAFETY_MARGIN: u64 = 512 * 1024 * 1024; // 512 MB

/// High-level VRAM pool manager.
///
/// Tracks all allocations by name (model_id, external registrations, etc.).
/// Source of truth = hardware query, not bookkeeping.
///
/// Supports external memory registration: other processes can register
/// GPU memory they own, and the pool accounts for it in placement decisions.
pub struct VramPool {
    allocator: Arc<VramAllocator>,
    /// Our allocations: name → GpuBuffer
    allocations: RwLock<HashMap<String, GpuBuffer>>,
    /// External registrations: name → (gpu_idx, size_bytes)
    /// These are memory owned by other processes. We don't free them,
    /// but we account for them when deciding if a model fits.
    external: RwLock<HashMap<String, (usize, u64)>>,
}

impl VramPool {
    pub fn new(allocator: Arc<VramAllocator>) -> Self {
        Self {
            allocator,
            allocations: RwLock::new(HashMap::new()),
            external: RwLock::new(HashMap::new()),
        }
    }

    pub fn allocator(&self) -> &Arc<VramAllocator> {
        &self.allocator
    }

    /// Allocate VRAM for a named resource (e.g., model weights).
    /// Fails if not enough free VRAM after safety margin.
    pub fn alloc(&self, name: &str, gpu_idx: usize, size: u64) -> Result<(), VramError> {
        let free = self.free(gpu_idx)?;
        if size > free {
            return Err(VramError::OutOfMemory {
                gpu_idx,
                need_mb: size / (1024 * 1024),
                free_mb: free / (1024 * 1024),
            });
        }

        let buf = self.allocator.alloc(gpu_idx, size)?;
        info!(
            name,
            gpu = gpu_idx,
            size_mb = size / (1024 * 1024),
            "VRAM allocated"
        );
        self.allocations.write().insert(name.to_string(), buf);
        Ok(())
    }

    /// Get a reference to an allocation's buffer for uploading data.
    pub fn get_buffer_mut(
        &self,
        name: &str,
    ) -> Option<parking_lot::MappedRwLockWriteGuard<'_, GpuBuffer>> {
        let guard = self.allocations.write();
        parking_lot::RwLockWriteGuard::try_map(guard, |map| map.get_mut(name)).ok()
    }

    /// Free a named allocation.
    pub fn free_alloc(&self, name: &str) {
        if let Some(buf) = self.allocations.write().remove(name) {
            info!(
                name,
                gpu = buf.gpu_idx(),
                freed_mb = buf.size() / (1024 * 1024),
                "VRAM freed"
            );
            // GpuBuffer::drop() frees the CUDA memory
        }
    }

    /// Register external GPU memory (owned by another process).
    /// We won't free it, but we account for it in placement.
    pub fn register_external(&self, name: &str, gpu_idx: usize, size: u64) {
        info!(
            name,
            gpu = gpu_idx,
            size_mb = size / (1024 * 1024),
            "external VRAM registered"
        );
        self.external
            .write()
            .insert(name.to_string(), (gpu_idx, size));
    }

    /// Unregister external GPU memory.
    pub fn unregister_external(&self, name: &str) {
        if self.external.write().remove(name).is_some() {
            info!(name, "external VRAM unregistered");
        }
    }

    /// Usable free VRAM on a GPU (hardware free minus safety margin).
    pub fn free(&self, gpu_idx: usize) -> Result<u64, VramError> {
        let hw_free = self.allocator.free_bytes(gpu_idx)?;
        Ok(hw_free.saturating_sub(SAFETY_MARGIN))
    }

    /// Find best GPU for a given size (best-fit: smallest free that still fits).
    pub fn find_placement(&self, size: u64) -> Result<Option<usize>, VramError> {
        let mut best: Option<(usize, u64)> = None;

        for i in 0..self.allocator.gpu_count() {
            let free = self.free(i)?;
            if free >= size {
                match best {
                    None => best = Some((i, free)),
                    Some((_, best_free)) if free < best_free => best = Some((i, free)),
                    _ => {}
                }
            }
        }

        Ok(best.map(|(idx, _)| idx))
    }

    /// Status for each GPU.
    pub fn status(&self) -> Vec<GpuStatus> {
        (0..self.allocator.gpu_count())
            .map(|i| {
                let total = self.allocator.total_bytes(i).unwrap_or(0);
                let hw_free = self.allocator.free_bytes(i).unwrap_or(0);
                let our_allocs: u64 = self
                    .allocations
                    .read()
                    .values()
                    .filter(|b| b.gpu_idx() == i)
                    .map(|b| b.size())
                    .sum();
                let ext_allocs: u64 = self
                    .external
                    .read()
                    .values()
                    .filter(|(idx, _)| *idx == i)
                    .map(|(_, size)| *size)
                    .sum();

                GpuStatus {
                    gpu_idx: i,
                    total_mb: total / (1024 * 1024),
                    hw_free_mb: hw_free / (1024 * 1024),
                    our_mb: our_allocs / (1024 * 1024),
                    external_mb: ext_allocs / (1024 * 1024),
                    usable_mb: hw_free.saturating_sub(SAFETY_MARGIN) / (1024 * 1024),
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuStatus {
    pub gpu_idx: usize,
    pub total_mb: u64,
    pub hw_free_mb: u64,
    pub our_mb: u64,
    pub external_mb: u64,
    pub usable_mb: u64,
}
