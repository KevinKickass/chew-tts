use crate::dequant::DequantKernels;
use crate::loader::KernelError;
use cudarc::cublas::{CudaBlas, Gemm as GemmTrait, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaSlice, CudaStream, CudaView};
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
        let blas =
            CudaBlas::new(Arc::clone(stream)).map_err(|e| KernelError::Cublas(e.to_string()))?;

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
        b_type: crate::GgmlType,
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
            assert!(
                chunk_n_max > 0,
                "scratch too small for even 1 row of weight matrix"
            );

            let bs = b_type.block_size() as u32;
            let bb = b_type.block_bytes() as u64;

            // Align chunk_n to block_size for clean slicing
            let chunk_n_aligned = if bs > 1 {
                (chunk_n_max / bs) * bs
            } else {
                chunk_n_max
            };
            assert!(
                chunk_n_aligned > 0,
                "chunk_n too small after block alignment"
            );

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
                let b_chunk =
                    b_quant.slice(byte_offset as usize..(byte_offset + chunk_byte_len) as usize);

                // Dequant this chunk into scratch (using view variant)
                dequant.dequant_view(
                    &b_chunk,
                    &mut self.dequant_scratch,
                    chunk_elements,
                    b_type,
                )?;

                // GEMM: C[:, n_done..n_done+chunk] = A @ chunk^T
                self.gemm_chunked(a, &self.dequant_scratch, c, m, n, k, chunk, n_done)?;

                n_done += chunk;
            }
        }

        Ok(())
    }

    pub fn matmul_dequant_view(
        &mut self,
        a: &CudaSlice<half::f16>,
        b_quant: &CudaView<'_, u8>,
        b_type: crate::GgmlType,
        b_elements: u32,
        c: &mut CudaSlice<half::f16>,
        m: u32,
        n: u32,
        k: u32,
        dequant: &DequantKernels,
    ) -> Result<(), KernelError> {
        let total_elements = (n * k) as usize;

        if total_elements <= self.scratch_elements {
            dequant.dequant_view(b_quant, &mut self.dequant_scratch, b_elements, b_type)?;
            self.matmul_f16(a, &self.dequant_scratch, c, m, n, k)?;
        } else {
            let chunk_n_max = (self.scratch_elements / k as usize) as u32;
            assert!(
                chunk_n_max > 0,
                "scratch too small for even 1 row of weight matrix"
            );

            let bs = b_type.block_size() as u32;
            let bb = b_type.block_bytes() as u64;
            let chunk_n_aligned = if bs > 1 {
                (chunk_n_max / bs) * bs
            } else {
                chunk_n_max
            };
            assert!(
                chunk_n_aligned > 0,
                "chunk_n too small after block alignment"
            );

            let mut n_done: u32 = 0;
            while n_done < n {
                let remaining = n - n_done;
                let chunk = if remaining <= chunk_n_max {
                    remaining
                } else {
                    chunk_n_aligned
                };

                let chunk_elements = chunk * k;
                let blocks_per_row = (k as u64 + bs as u64 - 1) / bs as u64;
                let row_bytes = blocks_per_row * bb;
                let byte_offset = n_done as u64 * row_bytes;
                let chunk_byte_len = chunk as u64 * row_bytes;
                let b_chunk =
                    b_quant.slice(byte_offset as usize..(byte_offset + chunk_byte_len) as usize);

                dequant.dequant_view(
                    &b_chunk,
                    &mut self.dequant_scratch,
                    chunk_elements,
                    b_type,
                )?;
                self.gemm_chunked(a, &self.dequant_scratch, c, m, n, k, chunk, n_done)?;

                n_done += chunk;
            }
        }

        Ok(())
    }

    /// Batched version for several quantized B matrices with identical shape/type.
    pub fn matmul_dequant_strided_batched(
        &mut self,
        a: &CudaSlice<half::f16>,
        b_quants: &[CudaView<'_, u8>],
        b_type: crate::GgmlType,
        b_elements_per_problem: u32,
        c: &mut CudaSlice<half::f16>,
        m: u32,
        n: u32,
        k: u32,
        dequant: &DequantKernels,
    ) -> Result<(), KernelError> {
        let batch_size = b_quants.len() as u32;
        if batch_size == 0 {
            return Ok(());
        }

        let total_elements = (batch_size * n * k) as usize;
        if total_elements > self.scratch_elements {
            return Err(KernelError::Cublas(format!(
                "batched dequant scratch too small: need {} elements, have {}",
                total_elements, self.scratch_elements
            )));
        }

        let per_problem = (n * k) as usize;
        for (i, bq) in b_quants.iter().enumerate() {
            let start = i * per_problem;
            let end = start + per_problem;
            let mut dst = self.dequant_scratch.slice_mut(start..end);
            dequant.dequant_to_view(bq, &mut dst, b_elements_per_problem, b_type)?;
        }

        self.matmul_f16_strided_batched_with_strides(
            a,
            &self.dequant_scratch,
            c,
            m,
            n,
            k,
            batch_size,
            0,
            (n * k) as i64,
            (m * n) as i64,
        )
    }

    /// Batched version for several quantized B matrices with identical shape/type,
    /// but with configurable A stride between batch items.
    ///
    /// This enables MoE-style batched down projections where each expert has its own
    /// activation row in A (instead of reusing the same A for every batch item).
    pub fn matmul_dequant_strided_batched_a_strided(
        &mut self,
        a: &CudaSlice<half::f16>,
        b_quants: &[CudaView<'_, u8>],
        b_type: crate::GgmlType,
        b_elements_per_problem: u32,
        c: &mut CudaSlice<half::f16>,
        m: u32,
        n: u32,
        k: u32,
        stride_a: i64,
        dequant: &DequantKernels,
    ) -> Result<(), KernelError> {
        let batch_size = b_quants.len() as u32;
        if batch_size == 0 {
            return Ok(());
        }

        let total_elements = (batch_size * n * k) as usize;
        if total_elements > self.scratch_elements {
            return Err(KernelError::Cublas(format!(
                "batched dequant scratch too small: need {} elements, have {}",
                total_elements, self.scratch_elements
            )));
        }

        let per_problem = (n * k) as usize;
        for (i, bq) in b_quants.iter().enumerate() {
            let start = i * per_problem;
            let end = start + per_problem;
            let mut dst = self.dequant_scratch.slice_mut(start..end);
            dequant.dequant_to_view(bq, &mut dst, b_elements_per_problem, b_type)?;
        }

        self.matmul_f16_strided_batched_with_strides(
            a,
            &self.dequant_scratch,
            c,
            m,
            n,
            k,
            batch_size,
            stride_a,
            (n * k) as i64,
            (m * n) as i64,
        )
    }

    /// GEMM for a chunk: writes to C[:, n_offset..n_offset+chunk_n]
    fn gemm_chunked(
        &self,
        a: &CudaSlice<half::f16>,
        b_chunk: &CudaSlice<half::f16>,
        c: &mut CudaSlice<half::f16>,
        m: u32,
        n: u32, // total N (for ldc stride)
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
            ldc: n as i32, // stride is full N, not chunk_n
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

    /// C = A @ B^T for native BF16 model weights and activations.
    pub fn matmul_bf16(
        &self,
        a: &CudaSlice<half::bf16>,
        b: &CudaSlice<half::bf16>,
        c: &mut CudaSlice<half::bf16>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), KernelError> {
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: half::bf16::from_f32(1.0),
            lda: k as i32,
            ldb: k as i32,
            beta: half::bf16::from_f32(0.0),
            ldc: n as i32,
        };
        unsafe {
            self.blas
                .gemm(cfg, b, a, c)
                .map_err(|e| KernelError::Cublas(e.to_string()))?;
        }
        Ok(())
    }

    /// C = A @ B (both f16, B NOT transposed).
    ///
    /// A: [m, k] row-major
    /// B: [k, n] row-major
    /// C: [m, n] row-major
    ///
    /// Unlike `matmul_f16` (which computes A @ B^T), this contracts A's columns
    /// with B's rows directly — used for the DiffusionGemma SC re-embedding
    /// `soft[C, n_embd] = probs[C, vocab] @ token_embd[vocab, n_embd]`, reusing
    /// the already-resident token_embd with no transpose buffer.
    pub fn matmul_f16_nt(
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

        // row-major C[m,n] = A[m,k] @ B[k,n]. In cuBLAS col-major terms the result
        // is C^T[n,m] = (B as col-major [n,k], op=N) @ (A as col-major [k,m], op=N).
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: n as i32,
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

    /// Strided batched f16 GEMM.
    ///
    /// A batch: [batch, m, k] row-major
    /// B batch: [batch, n, k] row-major
    /// C batch: [batch, m, n] row-major
    pub fn matmul_f16_strided_batched(
        &self,
        a: &CudaSlice<half::f16>,
        b: &CudaSlice<half::f16>,
        c: &mut CudaSlice<half::f16>,
        m: u32,
        n: u32,
        k: u32,
        batch_size: u32,
    ) -> Result<(), KernelError> {
        self.matmul_f16_strided_batched_with_strides(
            a,
            b,
            c,
            m,
            n,
            k,
            batch_size,
            (m * k) as i64,
            (n * k) as i64,
            (m * n) as i64,
        )
    }

    pub fn matmul_f16_strided_batched_with_strides(
        &self,
        a: &CudaSlice<half::f16>,
        b: &CudaSlice<half::f16>,
        c: &mut CudaSlice<half::f16>,
        m: u32,
        n: u32,
        k: u32,
        batch_size: u32,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
    ) -> Result<(), KernelError> {
        let alpha = half::f16::from_f32(1.0);
        let beta = half::f16::from_f32(0.0);

        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
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
            },
            batch_size: batch_size as i32,
            // gemm_strided_batched(cfg, b, a, c) passes b (weights/scratch) as the
            // 2nd arg (-> cfg.stride_a) and a (input) as the 3rd (-> cfg.stride_b).
            // The params name stride_a for the input and stride_b for the weights,
            // so they must be swapped here. (Previously not swapped -> wrong stride
            // on each buffer = the long-standing CHEW_MOE_BATCHED_EXPERTS garbage.)
            stride_a: stride_b,
            stride_b: stride_a,
            stride_c,
        };

        unsafe {
            self.blas
                .gemm_strided_batched(cfg, b, a, c)
                .map_err(|e| KernelError::Cublas(e.to_string()))?;
        }

        Ok(())
    }

    /// The actual scratch size in elements (for VRAM budget reporting).
    pub fn scratch_elements(&self) -> usize {
        self.scratch_elements
    }
}
