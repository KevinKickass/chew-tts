mod allocator;
mod pool;

pub use allocator::{GpuBuffer, VramAllocator, VramError};
pub use pool::VramPool;
