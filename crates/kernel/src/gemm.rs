use crate::dequant::DequantKernels;
use crate::loader::KernelError;
use cudarc::cublas::{CudaBlas, Gemm as GemmTrait, GemmConfig};
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

/// Max dequant scratch size: 64M f16 elements = 128 MB.
/// Weight matrices larger than this (e.g. output [128256, 4096]) are processed in chunks.
const MAX_SCRATCH_ELEMENTS: usize = 64 * 1024 * 1024; // 128 MB as f16

/// cuBLAS GEMM wrapper with on-the-fly dequantization support.
///
/// For quantized weights: dequant → temp f16 buffer → cuBLAS hgemm.
/// Large matrices are automatically chunked to fit the scratch buffer.
pub struct Gemm {
    blas: Arc<CudaBlas>,
    /// Reusable f16 scratch for dequantized weights (capped size)
    dequant_scratch: CudaSlice<half::f16>,
    scratch_elements: usize,
}

impl Gemm {
    /// Create GEMM handle with a capped dequant scratch buffer.
    pub fn new(stream: &Arc<CudaStream>, max_weight_elements: usize) -> Result<Self, KernelError> {
        let blas = CudaBlas::new(Arc::clone(stream))
            .map_err(|e| KernelError::Cublas(e.to_string()))?;

        let scratch_elements = max_weight_elements.min(MAX_SCRATCH_ELEMENTS);
        let dequant_scratch = stream
            .alloc_zeros::<half::f16>(scratch_elements)
            .map_err(|e| KernelError::Cublas(e.to_string()))?;

        Ok(Self {
            blas: Arc::new(blas),
            dequant_scratch,
            scratch_elements,
        })
    }

    /// C = A @ B^T where B is quantized.
    ///
    /// Dequantizes B into a temp f16 buffer, then runs cuBLAS hgemm.
    /// If B is too large for the scratch buffer, processes in chunks automatically.
    ///
    /// A: [m, k] f16 row-major
    /// B: quantized weight [n, k] (stored row-major, n output features)
    /// C: [m, n] f16 row-major
    pub fn matmul_dequant(
        &mut self,
        a: &CudaSlice<half::f16>,
        b_quant: &CudaSlice<u8>,
        b_type: chew_gguf::GgmlType,
        b_elements: u32,
        c: &mut CudaSlice<half::f16>,
        m: u32,
        n: u32,
        k: u32,
        dequant: &DequantKernels,
    ) -> Result<(), KernelError> {
        let total_elements = (n * k) as usize;

        if total_elements <= self.scratch_elements {
            // Fits in one shot
            dequant.dequant(b_quant, &mut self.dequant_scratch, b_elements, b_type)?;
            self.matmul_f16(a, &self.dequant_scratch, c, m, n, k)?;
        } else {
            // Chunked: split along the N (output) dimension
            let chunk_n_max = (self.scratch_elements / k as usize) as u32;
            assert!(chunk_n_max > 0, "scratch too small for even 1 row of weight matrix");

            let bs = b_type.block_size() as u32;
            let bb = b_type.block_bytes() as u64;

            // Align chunk_n to block_size for clean slicing
            let chunk_n_aligned = if bs > 1 {
                (chunk_n_max / bs) * bs
            } else {
                chunk_n_max
            };
            assert!(chunk_n_aligned > 0, "chunk_n too small after block alignment");

            let mut n_done: u32 = 0;
            while n_done < n {
                let remaining = n - n_done;
                let chunk = if remaining <= chunk_n_max {
                    remaining
                } else {
                    chunk_n_aligned
                };

                let chunk_elements = chunk * k;

                // Bytes per row: k elements / block_size blocks * block_bytes
                let blocks_per_row = (k as u64 + bs as u64 - 1) / bs as u64;
                let row_bytes = blocks_per_row * bb;
                let byte_offset = n_done as u64 * row_bytes;
                let chunk_byte_len = chunk as u64 * row_bytes;
                let b_chunk = b_quant.slice(byte_offset as usize..(byte_offset + chunk_byte_len) as usize);

                // Dequant this chunk into scratch (using view variant)
                dequant.dequant_view(&b_chunk, &mut self.dequant_scratch, chunk_elements, b_type)?;

                // GEMM: C[:, n_done..n_done+chunk] = A @ chunk^T
                self.gemm_chunked(
                    a,
                    &self.dequant_scratch,
                    c,
                    m, n, k,
                    chunk, n_done,
                )?;

                n_done += chunk;
            }
        }

        Ok(())
    }

    /// GEMM for a chunk: writes to C[:, n_offset..n_offset+chunk_n]
    fn gemm_chunked(
        &self,
        a: &CudaSlice<half::f16>,
        b_chunk: &CudaSlice<half::f16>,
        c: &mut CudaSlice<half::f16>,
        m: u32,
        n: u32,   // total N (for ldc stride)
        k: u32,
        chunk_n: u32,
        n_offset: u32,
    ) -> Result<(), KernelError> {
        let alpha = half::f16::from_f32(1.0);
        let beta = half::f16::from_f32(0.0);

        // cuBLAS col-major: C^T = B^T @ A^T
        // B_chunk is [chunk_n, k] row-major
        // A is [m, k] row-major
        // C is [m, n] row-major = [n, m] col-major
        // We write to C starting at column n_offset of C^T (= row-major column n_offset)
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: chunk_n as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: k as i32,
            ldb: k as i32,
            beta,
            ldc: n as i32,  // stride is full N, not chunk_n
        };

        // Offset into C: col-major C^T has offset = n_offset elements
        let mut c_view = c.slice_mut(n_offset as usize..);

        unsafe {
            self.blas
                .gemm(cfg, b_chunk, a, &mut c_view)
                .map_err(|e| KernelError::Cublas(e.to_string()))?;
        }

        Ok(())
    }

    /// C = A @ B (both f16, no chunking needed).
    ///
    /// A: [m, k] row-major
    /// B: [n, k] row-major (transposed in GEMM)
    /// C: [m, n] row-major
    pub fn matmul_f16(
        &self,
        a: &CudaSlice<half::f16>,
        b: &CudaSlice<half::f16>,
        c: &mut CudaSlice<half::f16>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), KernelError> {
        let alpha = half::f16::from_f32(1.0);
        let beta = half::f16::from_f32(0.0);

        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: k as i32,
            ldb: k as i32,
            beta,
            ldc: n as i32,
        };

        unsafe {
            self.blas
                .gemm(cfg, b, a, c)
                .map_err(|e| KernelError::Cublas(e.to_string()))?;
        }

        Ok(())
    }

    /// The actual scratch size in elements (for VRAM budget reporting).
    pub fn scratch_elements(&self) -> usize {
        self.scratch_elements
    }
}
