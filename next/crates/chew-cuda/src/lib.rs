//! NVIDIA-first CUDA backend.
//! The long-term goal is one binary with embedded CUDA sources compiled by NVRTC at startup.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceInfo {
    pub ordinal: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvrtcModuleSpec {
    pub name: &'static str,
    pub source: &'static str,
    pub kernels: &'static [&'static str],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledModule {
    pub name: &'static str,
    pub source_len: usize,
    pub kernels: &'static [&'static str],
    pub options: NvrtcCompileOptions,
}

impl CompiledModule {
    pub fn has_kernel(&self, kernel_name: &str) -> bool {
        self.kernels.contains(&kernel_name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NvrtcCompileOptions {
    pub use_fast_math: bool,
    pub line_info: bool,
    pub max_registers: Option<u32>,
}

impl Default for NvrtcCompileOptions {
    fn default() -> Self {
        Self {
            use_fast_math: false,
            line_info: false,
            max_registers: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModuleCacheKey {
    pub name: String,
    pub options: NvrtcCompileOptions,
}

#[derive(Debug, Default)]
pub struct ModuleCache {
    entries: BTreeMap<ModuleCacheKey, CompiledModule>,
    compile_count: usize,
}

impl ModuleCache {
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn compile_count(&self) -> usize {
        self.compile_count
    }

    pub fn get(&self, name: &str, options: NvrtcCompileOptions) -> Option<&CompiledModule> {
        self.entries.get(&ModuleCacheKey {
            name: name.to_string(),
            options,
        })
    }

    pub fn get_or_compile_named(
        &mut self,
        backend: &CudaBackend,
        name: &str,
        options: NvrtcCompileOptions,
    ) -> Result<&CompiledModule, CudaError> {
        let key = ModuleCacheKey {
            name: name.to_string(),
            options,
        };

        if !self.entries.contains_key(&key) {
            let compiled = backend.compile_embedded_module_named_with_options(name, options)?;
            self.entries.insert(key.clone(), compiled);
            self.compile_count += 1;
        }

        Ok(self.entries.get(&key).expect("cache entry inserted"))
    }

    pub fn prewarm_all(
        &mut self,
        backend: &CudaBackend,
        options: NvrtcCompileOptions,
    ) -> Result<(), CudaError> {
        for spec in backend.embedded_modules().all() {
            self.get_or_compile_named(backend, spec.name, options)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddedModuleRegistry {
    modules: &'static [NvrtcModuleSpec],
}

impl EmbeddedModuleRegistry {
    pub fn all(self) -> &'static [NvrtcModuleSpec] {
        self.modules
    }

    pub fn get(self, name: &str) -> Option<&'static NvrtcModuleSpec> {
        self.modules.iter().find(|spec| spec.name == name)
    }
}

#[derive(Debug)]
pub struct CudaBackend {
    device: DeviceInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    F16,
    BF16,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvWriteKernelKind {
    KeyRows,
    ValueRows,
    ValueRowsTransposed,
}

impl KvWriteKernelKind {
    pub fn kernel_name(self, scalar_type: ScalarType) -> &'static str {
        match (self, scalar_type) {
            (Self::KeyRows, ScalarType::F16) => "kv_write_k_rows_f16",
            (Self::KeyRows, ScalarType::BF16) => "kv_write_k_rows_bf16",
            (Self::KeyRows, ScalarType::F32) => "kv_write_k_rows_f32",
            (Self::ValueRows, ScalarType::F16) => "kv_write_v_rows_f16",
            (Self::ValueRows, ScalarType::BF16) => "kv_write_v_rows_bf16",
            (Self::ValueRows, ScalarType::F32) => "kv_write_v_rows_f32",
            (Self::ValueRowsTransposed, ScalarType::F16) => "kv_write_v_rows_trans_f16",
            (Self::ValueRowsTransposed, ScalarType::BF16) => "kv_write_v_rows_trans_bf16",
            (Self::ValueRowsTransposed, ScalarType::F32) => "kv_write_v_rows_trans_f32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelLaunchConfig {
    pub grid: Dim3,
    pub block: Dim3,
    pub shared_mem_bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvWriteKernelLaunch {
    pub module_name: &'static str,
    pub kind: KvWriteKernelKind,
    pub scalar_type: ScalarType,
    pub dst_byte_offset: u64,
    pub src_row_width_el: u32,
    pub row_width_el: u32,
    pub row_stride_bytes: u64,
    pub dst_row_count: u32,
    pub src_row_count: u32,
    pub index_count: u32,
    pub launch: KernelLaunchConfig,
}

impl KvWriteKernelLaunch {
    pub fn kernel_name(self) -> &'static str {
        self.kind.kernel_name(self.scalar_type)
    }

    pub fn validate(self) -> Result<Self, CudaError> {
        if self.module_name.is_empty() {
            return Err(CudaError::InvalidKvWriteLaunch {
                reason: "kv write launch requires a module name",
            });
        }
        if self.row_width_el == 0 {
            return Err(CudaError::InvalidKvWriteLaunch {
                reason: "kv write launch requires a non-zero row width",
            });
        }
        if self.src_row_width_el == 0 {
            return Err(CudaError::InvalidKvWriteLaunch {
                reason: "kv write launch requires a non-zero source row width",
            });
        }
        if self.row_stride_bytes == 0 {
            return Err(CudaError::InvalidKvWriteLaunch {
                reason: "kv write launch requires a non-zero row stride",
            });
        }
        if self.dst_row_count == 0 || self.src_row_count == 0 || self.index_count == 0 {
            return Err(CudaError::InvalidKvWriteLaunch {
                reason: "kv write launch requires non-zero row and index counts",
            });
        }
        if self.launch.grid.x == 0 || self.launch.block.x == 0 {
            return Err(CudaError::InvalidKvWriteLaunch {
                reason: "kv write launch requires non-zero grid and block dimensions",
            });
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvWriteLaunchPlan {
    pub key: KvWriteKernelLaunch,
    pub value: KvWriteKernelLaunch,
}

impl KvWriteLaunchPlan {
    pub fn validate(self) -> Result<Self, CudaError> {
        Ok(Self {
            key: self.key.validate()?,
            value: self.value.validate()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedKernelLaunch {
    pub module: CompiledModule,
    pub kernel_name: &'static str,
    pub spec: KvWriteKernelLaunch,
    pub launch: KernelLaunchConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedKvWriteLaunch {
    pub key: PreparedKernelLaunch,
    pub value: PreparedKernelLaunch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedKvWriteBundle {
    pub launches: Vec<PreparedKvWriteLaunch>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvWriteJobU16 {
    pub key_src: Vec<u16>,
    pub key_indices: Vec<i64>,
    pub value_src: Vec<u16>,
    pub value_indices: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvWriteJobF32 {
    pub key_src: Vec<f32>,
    pub key_indices: Vec<i64>,
    pub value_src: Vec<f32>,
    pub value_indices: Vec<i64>,
}

fn checked_row_stride_el_u16(launch: &KvWriteKernelLaunch) -> Result<usize, CudaError> {
    if !launch.row_stride_bytes.is_multiple_of(2) {
        return Err(CudaError::InvalidKvWriteLaunch {
            reason: "kv write launch row stride is not aligned to 16-bit elements",
        });
    }
    usize::try_from(launch.row_stride_bytes / 2).map_err(|_| CudaError::InvalidKvWriteLaunch {
        reason: "kv write launch row stride does not fit host usize",
    })
}

fn checked_row_stride_el_f32(launch: &KvWriteKernelLaunch) -> Result<usize, CudaError> {
    if !launch.row_stride_bytes.is_multiple_of(4) {
        return Err(CudaError::InvalidKvWriteLaunch {
            reason: "kv write launch row stride is not aligned to 32-bit elements",
        });
    }
    usize::try_from(launch.row_stride_bytes / 4).map_err(|_| CudaError::InvalidKvWriteLaunch {
        reason: "kv write launch row stride does not fit host usize",
    })
}

pub fn emulate_kv_write_u16(
    launch: &KvWriteKernelLaunch,
    src: &[u16],
    dst: &mut [u16],
    idxs: &[i64],
) -> Result<(), CudaError> {
    let launch = launch.validate()?;
    let src_row_width = usize::try_from(launch.src_row_width_el).map_err(|_| {
        CudaError::InvalidKvWriteLaunch {
            reason: "kv write launch source row width does not fit host usize",
        }
    })?;
    let dst_row_width = usize::try_from(launch.row_width_el).map_err(|_| {
        CudaError::InvalidKvWriteLaunch {
            reason: "kv write launch destination row width does not fit host usize",
        }
    })?;
    let src_row_count = usize::try_from(launch.src_row_count).map_err(|_| {
        CudaError::InvalidKvWriteLaunch {
            reason: "kv write launch source row count does not fit host usize",
        }
    })?;
    let index_count = usize::try_from(launch.index_count).map_err(|_| {
        CudaError::InvalidKvWriteLaunch {
            reason: "kv write launch index count does not fit host usize",
        }
    })?;
    let dst_row_stride = checked_row_stride_el_u16(&launch)?;
    let dst_base = usize::try_from(launch.dst_byte_offset / 2).map_err(|_| {
        CudaError::InvalidKvWriteLaunch {
            reason: "kv write launch destination offset does not fit host usize",
        }
    })?;

    if src.len() < src_row_width.saturating_mul(src_row_count) {
        return Err(CudaError::InvalidKvWriteLaunch {
            reason: "kv write source buffer is too small",
        });
    }
    if idxs.len() < index_count {
        return Err(CudaError::InvalidKvWriteLaunch {
            reason: "kv write index buffer is too small",
        });
    }

    for row in 0..index_count {
        let dst_row = usize::try_from(idxs[row]).map_err(|_| CudaError::InvalidKvWriteLaunch {
            reason: "kv write index is negative or does not fit host usize",
        })?;
        for col in 0..src_row_width.min(dst_row_width) {
            let dst_idx = dst_base + dst_row.saturating_mul(dst_row_stride) + col;
            if dst_idx >= dst.len() {
                return Err(CudaError::InvalidKvWriteLaunch {
                    reason: "kv write destination buffer is too small",
                });
            }
            dst[dst_idx] = src[row * src_row_width + col];
        }
    }

    Ok(())
}

pub fn emulate_kv_write_f32(
    launch: &KvWriteKernelLaunch,
    src: &[f32],
    dst: &mut [f32],
    idxs: &[i64],
) -> Result<(), CudaError> {
    let launch = launch.validate()?;
    let src_row_width = usize::try_from(launch.src_row_width_el).map_err(|_| {
        CudaError::InvalidKvWriteLaunch {
            reason: "kv write launch source row width does not fit host usize",
        }
    })?;
    let dst_row_width = usize::try_from(launch.row_width_el).map_err(|_| {
        CudaError::InvalidKvWriteLaunch {
            reason: "kv write launch destination row width does not fit host usize",
        }
    })?;
    let src_row_count = usize::try_from(launch.src_row_count).map_err(|_| {
        CudaError::InvalidKvWriteLaunch {
            reason: "kv write launch source row count does not fit host usize",
        }
    })?;
    let index_count = usize::try_from(launch.index_count).map_err(|_| {
        CudaError::InvalidKvWriteLaunch {
            reason: "kv write launch index count does not fit host usize",
        }
    })?;
    let dst_row_stride = checked_row_stride_el_f32(&launch)?;
    let dst_base = usize::try_from(launch.dst_byte_offset / 4).map_err(|_| {
        CudaError::InvalidKvWriteLaunch {
            reason: "kv write launch destination offset does not fit host usize",
        }
    })?;

    if src.len() < src_row_width.saturating_mul(src_row_count) {
        return Err(CudaError::InvalidKvWriteLaunch {
            reason: "kv write source buffer is too small",
        });
    }
    if idxs.len() < index_count {
        return Err(CudaError::InvalidKvWriteLaunch {
            reason: "kv write index buffer is too small",
        });
    }

    for row in 0..index_count {
        let dst_row = usize::try_from(idxs[row]).map_err(|_| CudaError::InvalidKvWriteLaunch {
            reason: "kv write index is negative or does not fit host usize",
        })?;
        for col in 0..src_row_width.min(dst_row_width) {
            let dst_idx = dst_base + dst_row.saturating_mul(dst_row_stride) + col;
            if dst_idx >= dst.len() {
                return Err(CudaError::InvalidKvWriteLaunch {
                    reason: "kv write destination buffer is too small",
                });
            }
            dst[dst_idx] = src[row * src_row_width + col];
        }
    }

    Ok(())
}

pub fn execute_prepared_kv_write_bundle_u16(
    bundle: &PreparedKvWriteBundle,
    jobs: &[KvWriteJobU16],
    arena: &mut [u16],
) -> Result<(), CudaError> {
    if bundle.launches.len() != jobs.len() {
        return Err(CudaError::InvalidKvWriteLaunch {
            reason: "kv write bundle/job count mismatch",
        });
    }

    for (prepared, job) in bundle.launches.iter().zip(jobs) {
        emulate_kv_write_u16(&prepared.key.spec, &job.key_src, arena, &job.key_indices)?;
        emulate_kv_write_u16(&prepared.value.spec, &job.value_src, arena, &job.value_indices)?;
    }

    Ok(())
}

pub fn execute_prepared_kv_write_bundle_f32(
    bundle: &PreparedKvWriteBundle,
    jobs: &[KvWriteJobF32],
    arena: &mut [f32],
) -> Result<(), CudaError> {
    if bundle.launches.len() != jobs.len() {
        return Err(CudaError::InvalidKvWriteLaunch {
            reason: "kv write bundle/job count mismatch",
        });
    }

    for (prepared, job) in bundle.launches.iter().zip(jobs) {
        emulate_kv_write_f32(&prepared.key.spec, &job.key_src, arena, &job.key_indices)?;
        emulate_kv_write_f32(&prepared.value.spec, &job.value_src, arena, &job.value_indices)?;
    }

    Ok(())
}

impl CudaBackend {
    pub fn new(device: DeviceInfo) -> Result<Self, CudaError> {
        Ok(Self { device })
    }

    pub fn device(&self) -> DeviceInfo {
        self.device
    }

    pub fn embedded_modules(&self) -> EmbeddedModuleRegistry {
        embedded_module_registry()
    }

    pub fn compile_embedded_module(
        &self,
        spec: &NvrtcModuleSpec,
    ) -> Result<CompiledModule, CudaError> {
        self.compile_embedded_module_with_options(spec, NvrtcCompileOptions::default())
    }

    pub fn compile_embedded_module_with_options(
        &self,
        spec: &NvrtcModuleSpec,
        options: NvrtcCompileOptions,
    ) -> Result<CompiledModule, CudaError> {
        if spec.name.is_empty() || spec.source.is_empty() {
            return Err(CudaError::InvalidModuleSpec);
        }
        Ok(CompiledModule {
            name: spec.name,
            source_len: spec.source.len(),
            kernels: spec.kernels,
            options,
        })
    }

    pub fn compile_embedded_module_named(&self, name: &str) -> Result<CompiledModule, CudaError> {
        self.compile_embedded_module_named_with_options(name, NvrtcCompileOptions::default())
    }

    pub fn compile_embedded_module_named_with_options(
        &self,
        name: &str,
        options: NvrtcCompileOptions,
    ) -> Result<CompiledModule, CudaError> {
        let spec = self
            .embedded_modules()
            .get(name)
            .ok_or_else(|| CudaError::UnknownEmbeddedModule {
                name: name.to_string(),
            })?;
        self.compile_embedded_module_with_options(spec, options)
    }

    pub fn compile_all_embedded_modules(&self) -> Result<Vec<CompiledModule>, CudaError> {
        self.compile_all_embedded_modules_with_options(NvrtcCompileOptions::default())
    }

    pub fn compile_all_embedded_modules_with_options(
        &self,
        options: NvrtcCompileOptions,
    ) -> Result<Vec<CompiledModule>, CudaError> {
        self.embedded_modules()
            .all()
            .iter()
            .map(|spec| self.compile_embedded_module_with_options(spec, options))
            .collect()
    }

    pub fn prepare_kv_write_launch(
        &self,
        cache: &mut ModuleCache,
        plan: &KvWriteLaunchPlan,
        options: NvrtcCompileOptions,
    ) -> Result<PreparedKvWriteLaunch, CudaError> {
        let plan = plan.clone().validate()?;
        let key_module = cache
            .get_or_compile_named(self, plan.key.module_name, options)?
            .clone();
        let value_module = if plan.value.module_name == plan.key.module_name {
            key_module.clone()
        } else {
            cache
                .get_or_compile_named(self, plan.value.module_name, options)?
                .clone()
        };

        Ok(PreparedKvWriteLaunch {
            key: PreparedKernelLaunch {
                module: key_module,
                kernel_name: plan.key.kernel_name(),
                spec: plan.key,
                launch: plan.key.launch,
            },
            value: PreparedKernelLaunch {
                module: value_module,
                kernel_name: plan.value.kernel_name(),
                spec: plan.value,
                launch: plan.value.launch,
            },
        })
        .and_then(|prepared| {
            if !prepared.key.module.has_kernel(prepared.key.kernel_name) {
                return Err(CudaError::UnknownModuleKernel {
                    module: prepared.key.module.name.to_string(),
                    kernel: prepared.key.kernel_name.to_string(),
                });
            }
            if !prepared.value.module.has_kernel(prepared.value.kernel_name) {
                return Err(CudaError::UnknownModuleKernel {
                    module: prepared.value.module.name.to_string(),
                    kernel: prepared.value.kernel_name.to_string(),
                });
            }
            Ok(prepared)
        })
    }

    pub fn prepare_kv_write_launches(
        &self,
        cache: &mut ModuleCache,
        plans: &[KvWriteLaunchPlan],
        options: NvrtcCompileOptions,
    ) -> Result<PreparedKvWriteBundle, CudaError> {
        let mut launches = Vec::with_capacity(plans.len());
        for plan in plans {
            launches.push(self.prepare_kv_write_launch(cache, plan, options)?);
        }
        Ok(PreparedKvWriteBundle { launches })
    }
}

pub fn embedded_module_registry() -> EmbeddedModuleRegistry {
    EmbeddedModuleRegistry {
        modules: EMBEDDED_MODULE_SPECS,
    }
}

const EMBEDDED_MODULE_SPECS: &[NvrtcModuleSpec] = &[
    NvrtcModuleSpec {
        name: "bootstrap",
        source: r#"extern "C" __global__ void bootstrap(void) {}"#,
        kernels: &["bootstrap"],
    },
    NvrtcModuleSpec {
        name: "rmsnorm",
        source: r#"extern "C" __global__ void rmsnorm_f32(const float * x, float * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i];
    }
}"#,
        kernels: &["rmsnorm_f32"],
    },
    NvrtcModuleSpec {
        name: "rope",
        source: r#"extern "C" __global__ void rope_f32(float * q, float * k, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        q[i] = q[i];
        k[i] = k[i];
    }
}"#,
        kernels: &["rope_f32"],
    },
    NvrtcModuleSpec {
        name: "kv_cache",
        source: include_str!("kernels/kv_cache.cu"),
        kernels: &[
            "kv_write_k_rows_f16",
            "kv_write_k_rows_bf16",
            "kv_write_k_rows_f32",
            "kv_write_v_rows_f16",
            "kv_write_v_rows_bf16",
            "kv_write_v_rows_f32",
            "kv_write_v_rows_trans_f16",
            "kv_write_v_rows_trans_bf16",
            "kv_write_v_rows_trans_f32",
        ],
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaError {
    InvalidModuleSpec,
    UnknownEmbeddedModule { name: String },
    UnknownModuleKernel { module: String, kernel: String },
    InvalidKvWriteLaunch { reason: &'static str },
}

impl Display for CudaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidModuleSpec => f.write_str("invalid embedded NVRTC module spec"),
            Self::UnknownEmbeddedModule { name } => {
                write!(f, "unknown embedded NVRTC module `{name}`")
            }
            Self::UnknownModuleKernel { module, kernel } => {
                write!(f, "module `{module}` does not export kernel `{kernel}`")
            }
            Self::InvalidKvWriteLaunch { reason } => {
                write!(f, "invalid KV write launch: {reason}")
            }
        }
    }
}

impl Error for CudaError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> CudaBackend {
        CudaBackend::new(DeviceInfo { ordinal: 0 }).unwrap()
    }

    #[test]
    fn embedded_registry_exposes_named_modules() {
        let registry = embedded_module_registry();
        assert_eq!(registry.all().len(), 4);
        assert_eq!(registry.get("bootstrap").unwrap().name, "bootstrap");
        assert_eq!(registry.get("kv_cache").unwrap().name, "kv_cache");
        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn compiles_embedded_modules_by_name() {
        let module = backend().compile_embedded_module_named("rmsnorm").unwrap();
        assert_eq!(module.name, "rmsnorm");
        assert!(module.source_len > 0);
        assert_eq!(module.options, NvrtcCompileOptions::default());
    }

    #[test]
    fn rejects_unknown_embedded_module_names() {
        let err = backend()
            .compile_embedded_module_named("not-real")
            .unwrap_err();
        assert_eq!(
            err,
            CudaError::UnknownEmbeddedModule {
                name: "not-real".to_string(),
            }
        );
    }

    #[test]
    fn compiles_all_embedded_modules() {
        let modules = backend().compile_all_embedded_modules().unwrap();
        assert_eq!(modules.len(), 4);
        assert_eq!(modules[0].name, "bootstrap");
        assert_eq!(modules[1].name, "rmsnorm");
        assert_eq!(modules[2].name, "rope");
        assert_eq!(modules[3].name, "kv_cache");
    }

    #[test]
    fn module_cache_reuses_compilation_for_same_options() {
        let backend = backend();
        let mut cache = ModuleCache::default();
        let options = NvrtcCompileOptions {
            use_fast_math: true,
            line_info: false,
            max_registers: Some(96),
        };

        cache
            .get_or_compile_named(&backend, "bootstrap", options)
            .unwrap();
        cache
            .get_or_compile_named(&backend, "bootstrap", options)
            .unwrap();

        assert_eq!(cache.entry_count(), 1);
        assert_eq!(cache.compile_count(), 1);
    }

    #[test]
    fn module_cache_distinguishes_compile_options() {
        let backend = backend();
        let mut cache = ModuleCache::default();

        cache
            .get_or_compile_named(&backend, "bootstrap", NvrtcCompileOptions::default())
            .unwrap();
        cache
            .get_or_compile_named(
                &backend,
                "bootstrap",
                NvrtcCompileOptions {
                    use_fast_math: true,
                    line_info: false,
                    max_registers: None,
                },
            )
            .unwrap();

        assert_eq!(cache.entry_count(), 2);
        assert_eq!(cache.compile_count(), 2);
    }

    #[test]
    fn prepares_kv_write_launch_and_reuses_module_cache() {
        let backend = backend();
        let mut cache = ModuleCache::default();
        let plan = KvWriteLaunchPlan {
            key: KvWriteKernelLaunch {
                module_name: "kv_cache",
                kind: KvWriteKernelKind::KeyRows,
                scalar_type: ScalarType::F16,
                dst_byte_offset: 0,
                src_row_width_el: 1024,
                row_width_el: 1024,
                row_stride_bytes: 2048,
                dst_row_count: 8192,
                src_row_count: 4,
                index_count: 4,
                launch: KernelLaunchConfig {
                    grid: Dim3 { x: 4, y: 1, z: 1 },
                    block: Dim3 { x: 128, y: 1, z: 1 },
                    shared_mem_bytes: 0,
                },
            },
            value: KvWriteKernelLaunch {
                module_name: "kv_cache",
                kind: KvWriteKernelKind::ValueRowsTransposed,
                scalar_type: ScalarType::F16,
                dst_byte_offset: 4096,
                src_row_width_el: 1,
                row_width_el: 1,
                row_stride_bytes: 2,
                dst_row_count: 4096,
                src_row_count: 4096,
                index_count: 4096,
                launch: KernelLaunchConfig {
                    grid: Dim3 { x: 4096, y: 1, z: 1 },
                    block: Dim3 { x: 128, y: 1, z: 1 },
                    shared_mem_bytes: 0,
                },
            },
        };

        let prepared = backend
            .prepare_kv_write_launch(&mut cache, &plan, NvrtcCompileOptions::default())
            .unwrap();

        assert_eq!(prepared.key.module.name, "kv_cache");
        assert_eq!(prepared.key.kernel_name, "kv_write_k_rows_f16");
        assert_eq!(prepared.value.kernel_name, "kv_write_v_rows_trans_f16");
        assert_eq!(cache.compile_count(), 1);
    }

    #[test]
    fn kv_cache_module_exports_expected_kernel_names_in_source() {
        let spec = embedded_module_registry().get("kv_cache").unwrap();
        for kernel in spec.kernels {
            assert!(spec.source.contains(kernel));
        }

        let module = backend().compile_embedded_module_named("kv_cache").unwrap();
        assert!(module.has_kernel("kv_write_k_rows_f16"));
        assert!(module.has_kernel("kv_write_v_rows_trans_f32"));
    }

    #[test]
    fn prepares_kv_write_launch_bundle_with_shared_module_cache() {
        let backend = backend();
        let mut cache = ModuleCache::default();
        let plan = KvWriteLaunchPlan {
            key: KvWriteKernelLaunch {
                module_name: "kv_cache",
                kind: KvWriteKernelKind::KeyRows,
                scalar_type: ScalarType::F16,
                dst_byte_offset: 0,
                src_row_width_el: 1024,
                row_width_el: 1024,
                row_stride_bytes: 2048,
                dst_row_count: 8192,
                src_row_count: 4,
                index_count: 4,
                launch: KernelLaunchConfig {
                    grid: Dim3 { x: 4, y: 1, z: 1 },
                    block: Dim3 { x: 128, y: 1, z: 1 },
                    shared_mem_bytes: 0,
                },
            },
            value: KvWriteKernelLaunch {
                module_name: "kv_cache",
                kind: KvWriteKernelKind::ValueRowsTransposed,
                scalar_type: ScalarType::F16,
                dst_byte_offset: 4096,
                src_row_width_el: 1,
                row_width_el: 1,
                row_stride_bytes: 2,
                dst_row_count: 4096,
                src_row_count: 4096,
                index_count: 4096,
                launch: KernelLaunchConfig {
                    grid: Dim3 { x: 4096, y: 1, z: 1 },
                    block: Dim3 { x: 128, y: 1, z: 1 },
                    shared_mem_bytes: 0,
                },
            },
        };

        let bundle = backend
            .prepare_kv_write_launches(
                &mut cache,
                &[plan.clone(), plan],
                NvrtcCompileOptions::default(),
            )
            .unwrap();

        assert_eq!(bundle.launches.len(), 2);
        assert_eq!(bundle.launches[0].key.kernel_name, "kv_write_k_rows_f16");
        assert_eq!(cache.compile_count(), 1);
    }

    #[test]
    fn executes_prepared_kv_write_bundle_u16_into_host_arena() {
        let backend = backend();
        let mut cache = ModuleCache::default();
        let plan = KvWriteLaunchPlan {
            key: KvWriteKernelLaunch {
                module_name: "kv_cache",
                kind: KvWriteKernelKind::KeyRows,
                scalar_type: ScalarType::F16,
                dst_byte_offset: 0,
                src_row_width_el: 4,
                row_width_el: 4,
                row_stride_bytes: 8,
                dst_row_count: 8,
                src_row_count: 2,
                index_count: 2,
                launch: KernelLaunchConfig {
                    grid: Dim3 { x: 2, y: 1, z: 1 },
                    block: Dim3 { x: 128, y: 1, z: 1 },
                    shared_mem_bytes: 0,
                },
            },
            value: KvWriteKernelLaunch {
                module_name: "kv_cache",
                kind: KvWriteKernelKind::ValueRows,
                scalar_type: ScalarType::F16,
                dst_byte_offset: 64,
                src_row_width_el: 4,
                row_width_el: 4,
                row_stride_bytes: 8,
                dst_row_count: 8,
                src_row_count: 2,
                index_count: 2,
                launch: KernelLaunchConfig {
                    grid: Dim3 { x: 2, y: 1, z: 1 },
                    block: Dim3 { x: 128, y: 1, z: 1 },
                    shared_mem_bytes: 0,
                },
            },
        };

        let bundle = backend
            .prepare_kv_write_launches(&mut cache, &[plan], NvrtcCompileOptions::default())
            .unwrap();
        let jobs = vec![KvWriteJobU16 {
            key_src: vec![1, 2, 3, 4, 5, 6, 7, 8],
            key_indices: vec![0, 2],
            value_src: vec![11, 12, 13, 14, 15, 16, 17, 18],
            value_indices: vec![1, 3],
        }];
        let mut arena = vec![0_u16; 64];

        execute_prepared_kv_write_bundle_u16(&bundle, &jobs, &mut arena).unwrap();

        assert_eq!(&arena[0..4], &[1, 2, 3, 4]);
        assert_eq!(&arena[8..12], &[5, 6, 7, 8]);
        assert_eq!(&arena[36..40], &[11, 12, 13, 14]);
        assert_eq!(&arena[44..48], &[15, 16, 17, 18]);
    }

    #[test]
    fn emulates_u16_kv_row_write() {
        let launch = KvWriteKernelLaunch {
            module_name: "kv_cache",
            kind: KvWriteKernelKind::KeyRows,
            scalar_type: ScalarType::F16,
            dst_byte_offset: 0,
            src_row_width_el: 4,
            row_width_el: 4,
            row_stride_bytes: 8,
            dst_row_count: 8,
            src_row_count: 2,
            index_count: 2,
            launch: KernelLaunchConfig {
                grid: Dim3 { x: 2, y: 1, z: 1 },
                block: Dim3 { x: 128, y: 1, z: 1 },
                shared_mem_bytes: 0,
            },
        };
        let src = vec![10_u16, 11, 12, 13, 20, 21, 22, 23];
        let idxs = vec![1_i64, 3_i64];
        let mut dst = vec![0_u16; 8 * 4];

        emulate_kv_write_u16(&launch, &src, &mut dst, &idxs).unwrap();

        assert_eq!(&dst[4..8], &[10, 11, 12, 13]);
        assert_eq!(&dst[12..16], &[20, 21, 22, 23]);
    }

    #[test]
    fn emulates_u16_transposed_v_write_with_flat_indices() {
        let launch = KvWriteKernelLaunch {
            module_name: "kv_cache",
            kind: KvWriteKernelKind::ValueRowsTransposed,
            scalar_type: ScalarType::F16,
            dst_byte_offset: 0,
            src_row_width_el: 1,
            row_width_el: 1,
            row_stride_bytes: 2,
            dst_row_count: 32,
            src_row_count: 4,
            index_count: 4,
            launch: KernelLaunchConfig {
                grid: Dim3 { x: 4, y: 1, z: 1 },
                block: Dim3 { x: 128, y: 1, z: 1 },
                shared_mem_bytes: 0,
            },
        };
        let src = vec![101_u16, 102, 201, 202];
        let idxs = vec![1_i64, 5_i64, 2_i64, 6_i64];
        let mut dst = vec![0_u16; 32];

        emulate_kv_write_u16(&launch, &src, &mut dst, &idxs).unwrap();

        assert_eq!(dst[1], 101);
        assert_eq!(dst[5], 102);
        assert_eq!(dst[2], 201);
        assert_eq!(dst[6], 202);
    }

    #[test]
    fn rejects_invalid_kv_write_launch() {
        let err = KvWriteKernelLaunch {
            module_name: "kv_cache",
            kind: KvWriteKernelKind::KeyRows,
            scalar_type: ScalarType::F16,
            dst_byte_offset: 0,
            src_row_width_el: 1024,
            row_width_el: 0,
            row_stride_bytes: 2048,
            dst_row_count: 1,
            src_row_count: 1,
            index_count: 1,
            launch: KernelLaunchConfig {
                grid: Dim3 { x: 1, y: 1, z: 1 },
                block: Dim3 { x: 128, y: 1, z: 1 },
                shared_mem_bytes: 0,
            },
        }
        .validate()
        .unwrap_err();

        assert_eq!(
            err,
            CudaError::InvalidKvWriteLaunch {
                reason: "kv write launch requires a non-zero row width",
            }
        );
    }
}
