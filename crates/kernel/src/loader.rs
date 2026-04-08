use cudarc::driver::{CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use std::sync::Arc;
use tracing::{info, warn};

/// Find CUDA include directory at runtime.
/// Checks CUDA_PATH env, then common install locations.
fn find_cuda_include() -> Option<String> {
    // 1. CUDA_PATH env (e.g. /usr/local/cuda)
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let inc = format!("{cuda_path}/include");
        if std::path::Path::new(&inc).join("cuda_fp16.h").exists() {
            return Some(inc);
        }
        // Also check targets/ subdirectory
        let inc_targets = format!("{cuda_path}/targets/x86_64-linux/include");
        if std::path::Path::new(&inc_targets).join("cuda_fp16.h").exists() {
            return Some(inc_targets);
        }
    }

    // 2. /usr/local/cuda symlink (most common)
    let default = "/usr/local/cuda/include";
    if std::path::Path::new(default).join("cuda_fp16.h").exists() {
        return Some(default.to_string());
    }
    let default_targets = "/usr/local/cuda/targets/x86_64-linux/include";
    if std::path::Path::new(default_targets).join("cuda_fp16.h").exists() {
        return Some(default_targets.to_string());
    }

    // 3. Scan versioned installs, pick highest version
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("cuda-") {
                let inc = format!("/usr/local/{name_str}/targets/x86_64-linux/include");
                if std::path::Path::new(&inc).join("cuda_fp16.h").exists() {
                    candidates.push(inc);
                }
                let inc2 = format!("/usr/local/{name_str}/include");
                if std::path::Path::new(&inc2).join("cuda_fp16.h").exists() {
                    candidates.push(inc2);
                }
            }
        }
    }
    candidates.sort();
    candidates.pop()
}

/// Compile CUDA source to PTX via NVRTC at runtime, then load as module.
/// Targets the GPU's actual compute capability to avoid PTX version mismatches.
pub fn load_module_from_source(
    stream: &Arc<CudaStream>,
    cu_source: &str,
    name: &str,
) -> Result<Arc<CudaModule>, KernelError> {
    info!(kernel = name, "compiling CUDA kernel via NVRTC");

    let include_paths = find_cuda_include()
        .map(|p| {
            info!(path = %p, "found CUDA include directory");
            vec![p]
        })
        .unwrap_or_default();

    // Query actual GPU compute capability to generate compatible PTX.
    // This avoids PTX ISA version mismatches between toolkit and driver.
    let ctx = stream.context();
    let (major, minor) = ctx.compute_capability().unwrap_or((8, 6));

    let arch_str = format!("compute_{major}{minor}");
    info!(kernel = name, arch = %arch_str, "targeting GPU architecture");

    // arch needs 'static str — leak is fine, happens once per kernel module
    let arch_static: &'static str = Box::leak(arch_str.into_boxed_str());

    let opts = CompileOptions {
        ftz: Some(true),
        prec_div: Some(false),
        prec_sqrt: Some(false),
        fmad: Some(true),
        name: Some(name.to_string()),
        include_paths,
        arch: Some(arch_static),
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(cu_source, opts)
        .map_err(|e| KernelError::Compile(format!("{name}: {e}")))?;

    // NVRTC from a newer toolkit (e.g. CUDA 13.1) may stamp a PTX ISA version
    // higher than what the installed driver supports (e.g. driver 580 = CUDA 13.0
    // supports up to PTX 9.0, but NVRTC 13.1 outputs PTX 9.1).
    // Since we target the GPU's compute capability, the generated code doesn't
    // actually need the newer ISA — so we can safely patch the version down.
    let ptx = downgrade_ptx_version_if_needed(ptx);

    info!(kernel = name, "NVRTC compilation done, loading module");

    ctx.load_module(ptx)
        .map_err(|e| KernelError::Load(e.to_string()))
}

/// If the PTX declares a version newer than the driver can handle,
/// patch it down. This is safe when we target compute_XX (not using
/// newer ISA features).
fn downgrade_ptx_version_if_needed(ptx: Ptx) -> Ptx {
    // Get the driver version to determine max supported PTX ISA
    let driver_version = {
        let mut ver: core::ffi::c_int = 0;
        let res = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut ver) };
        if res == cudarc::driver::sys::CUresult::CUDA_SUCCESS { ver as u32 } else { 0 }
    };

    // Map driver version to max PTX ISA version:
    // CUDA 13.0 (driver version 13000) → PTX 9.0
    // CUDA 13.1 (13010) → PTX 9.1
    // CUDA 12.8 (12080) → PTX 8.8
    // We only need to patch if NVRTC version > driver version
    let max_ptx = match driver_version {
        v if v >= 13010 => (9, 1),
        v if v >= 13000 => (9, 0),
        v if v >= 12080 => (8, 8),
        v if v >= 12060 => (8, 6),
        _ => return ptx, // don't know, don't touch
    };

    let src = ptx.to_src();

    // PTX starts with ".version X.Y" — find and check it
    if let Some(ver_start) = src.find(".version ") {
        let after = &src[ver_start + 9..];
        if let Some(dot) = after.find('.') {
            let major_str = &after[..dot];
            let minor_end = after[dot + 1..].find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len() - dot - 1);
            let minor_str = &after[dot + 1..dot + 1 + minor_end];

            if let (Ok(ptx_major), Ok(ptx_minor)) = (major_str.parse::<u32>(), minor_str.parse::<u32>()) {
                if ptx_major > max_ptx.0 || (ptx_major == max_ptx.0 && ptx_minor > max_ptx.1) {
                    let old_ver = format!(".version {ptx_major}.{ptx_minor}");
                    let new_ver = format!(".version {}.{}", max_ptx.0, max_ptx.1);
                    warn!(
                        from = %old_ver,
                        to = %new_ver,
                        "downgrading PTX version to match driver"
                    );
                    let patched = src.replacen(&old_ver, &new_ver, 1);
                    return Ptx::from_src(patched);
                }
            }
        }
    }

    ptx
}

pub fn get_fn(module: &Arc<CudaModule>, name: &str) -> Result<CudaFunction, KernelError> {
    module
        .load_function(name)
        .map_err(|e| KernelError::Load(format!("{name}: {e}")))
}

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("kernel compile failed: {0}")]
    Compile(String),
    #[error("kernel load failed: {0}")]
    Load(String),
    #[error("kernel launch failed: {0}")]
    Launch(String),
    #[error("cuBLAS error: {0}")]
    Cublas(String),
}
