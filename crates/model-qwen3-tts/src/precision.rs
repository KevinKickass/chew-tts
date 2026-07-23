use crate::{load_bf16_tensor, load_f16_tensor};
use anyhow::Context;
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::{bf16, f16};
use std::path::Path;
use std::sync::Arc;

mod private {
    pub trait Sealed {}
}

pub trait QwenDType:
    private::Sealed + DeviceRepr + ValidAsZeroBits + Copy + Default + Send + Sync + 'static
{
    fn load(
        model_dir: &Path,
        name: &str,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<(Vec<usize>, CudaSlice<Self>)>;
    fn zero() -> Self;
    fn to_f32(value: Self) -> f32;
    fn from_f32(value: f32) -> Self;
    fn rms_norm_f32in(
        kernels: &mut GpuKernels,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<Self>,
        output: &mut CudaSlice<Self>,
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> anyhow::Result<()>;
    fn matmul(
        kernels: &mut GpuKernels,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        c: &mut CudaSlice<Self>,
        m: u32,
        n: u32,
        k: u32,
    ) -> anyhow::Result<()>;
    fn gemv(
        kernels: &mut GpuKernels,
        input: &CudaSlice<Self>,
        weight: &CudaSlice<Self>,
        output: &mut CudaSlice<Self>,
        rows: u32,
        cols: u32,
    ) -> anyhow::Result<()>;
    #[allow(clippy::too_many_arguments)]
    fn gemv_dual(
        kernels: &mut GpuKernels,
        input: &CudaSlice<Self>,
        weight_a: &CudaSlice<Self>,
        weight_b: &CudaSlice<Self>,
        output_a: &mut CudaSlice<Self>,
        output_b: &mut CudaSlice<Self>,
        rows: u32,
        cols: u32,
    ) -> anyhow::Result<()>;
    fn add_residual(
        kernels: &mut GpuKernels,
        hidden: &mut CudaSlice<f32>,
        residual: &CudaSlice<Self>,
        count: u32,
    ) -> anyhow::Result<()>;
    fn silu(
        kernels: &mut GpuKernels,
        gate: &CudaSlice<Self>,
        up: &CudaSlice<Self>,
        output: &mut CudaSlice<Self>,
        count: u32,
    ) -> anyhow::Result<()>;
    fn silu_act(
        kernels: &mut GpuKernels,
        input: &CudaSlice<Self>,
        output: &mut CudaSlice<Self>,
        count: u32,
    ) -> anyhow::Result<()>;
    fn gather(
        kernels: &mut GpuKernels,
        table: &CudaSlice<Self>,
        ids: &CudaSlice<i32>,
        output: &mut CudaSlice<Self>,
        rows: u32,
        cols: u32,
    ) -> anyhow::Result<()>;
    fn add_bias(
        kernels: &mut GpuKernels,
        values: &mut CudaSlice<Self>,
        bias: &CudaSlice<Self>,
        rows: u32,
        cols: u32,
    ) -> anyhow::Result<()>;
    fn to_f16(
        kernels: &mut GpuKernels,
        input: &CudaSlice<Self>,
        output: &mut CudaSlice<f16>,
        count: u32,
    ) -> anyhow::Result<()>;
    fn from_f16(
        kernels: &mut GpuKernels,
        input: &CudaSlice<f16>,
        output: &mut CudaSlice<Self>,
        count: u32,
    ) -> anyhow::Result<()>;
    fn to_f32_device(
        kernels: &mut GpuKernels,
        input: &CudaSlice<Self>,
        output: &mut CudaSlice<f32>,
        count: u32,
    ) -> anyhow::Result<()>;
}

pub type F16 = f16;
pub type Bf16 = bf16;

impl private::Sealed for f16 {}
impl QwenDType for f16 {
    fn load(
        model_dir: &Path,
        name: &str,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<(Vec<usize>, CudaSlice<Self>)> {
        let tensor =
            load_f16_tensor(model_dir, name).with_context(|| format!("could not load {name}"))?;
        Ok((tensor.shape, stream.clone_htod(&tensor.values)?))
    }
    fn zero() -> Self {
        f16::ZERO
    }
    fn to_f32(value: Self) -> f32 {
        value.to_f32()
    }
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }
    fn rms_norm_f32in(
        k: &mut GpuKernels,
        i: &CudaSlice<f32>,
        w: &CudaSlice<Self>,
        o: &mut CudaSlice<Self>,
        r: u32,
        c: u32,
        e: f32,
    ) -> anyhow::Result<()> {
        k.ops.rms_norm_f32in(i, w, o, r, c, e)?;
        Ok(())
    }
    fn matmul(
        k: &mut GpuKernels,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        c: &mut CudaSlice<Self>,
        m: u32,
        n: u32,
        d: u32,
    ) -> anyhow::Result<()> {
        k.gemm.matmul_f16(a, b, c, m, n, d)?;
        Ok(())
    }
    fn gemv(
        k: &mut GpuKernels,
        i: &CudaSlice<Self>,
        w: &CudaSlice<Self>,
        o: &mut CudaSlice<Self>,
        r: u32,
        c: u32,
    ) -> anyhow::Result<()> {
        k.gemv.gemv_f16(i, w, o, r, c)?;
        Ok(())
    }
    fn gemv_dual(
        k: &mut GpuKernels,
        i: &CudaSlice<Self>,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        oa: &mut CudaSlice<Self>,
        ob: &mut CudaSlice<Self>,
        r: u32,
        c: u32,
    ) -> anyhow::Result<()> {
        k.gemv.gemv_dual_f16(i, a, b, oa, ob, r, c)?;
        Ok(())
    }
    fn add_residual(
        k: &mut GpuKernels,
        h: &mut CudaSlice<f32>,
        r: &CudaSlice<Self>,
        n: u32,
    ) -> anyhow::Result<()> {
        k.ops.add_inplace_f32_f16(h, r, n)?;
        Ok(())
    }
    fn silu(
        k: &mut GpuKernels,
        g: &CudaSlice<Self>,
        u: &CudaSlice<Self>,
        o: &mut CudaSlice<Self>,
        n: u32,
    ) -> anyhow::Result<()> {
        k.ops.silu(g, u, o, n)?;
        Ok(())
    }
    fn silu_act(
        k: &mut GpuKernels,
        i: &CudaSlice<Self>,
        o: &mut CudaSlice<Self>,
        n: u32,
    ) -> anyhow::Result<()> {
        k.ops.silu_act_f16(i, o, n)?;
        Ok(())
    }
    fn gather(
        k: &mut GpuKernels,
        t: &CudaSlice<Self>,
        i: &CudaSlice<i32>,
        o: &mut CudaSlice<Self>,
        r: u32,
        c: u32,
    ) -> anyhow::Result<()> {
        k.ops.gather_rows_f16(t, i, o, r, c)?;
        Ok(())
    }
    fn add_bias(
        k: &mut GpuKernels,
        v: &mut CudaSlice<Self>,
        b: &CudaSlice<Self>,
        r: u32,
        c: u32,
    ) -> anyhow::Result<()> {
        k.ops.add_bias_f16_inplace(v, b, r, c)?;
        Ok(())
    }
    fn to_f16(
        k: &mut GpuKernels,
        input: &CudaSlice<Self>,
        output: &mut CudaSlice<f16>,
        count: u32,
    ) -> anyhow::Result<()> {
        k.ops
            .copy_f16(input, &mut output.slice_mut(..count as usize), count)?;
        Ok(())
    }
    fn from_f16(
        k: &mut GpuKernels,
        input: &CudaSlice<f16>,
        output: &mut CudaSlice<Self>,
        count: u32,
    ) -> anyhow::Result<()> {
        k.ops
            .copy_f16(input, &mut output.slice_mut(..count as usize), count)?;
        Ok(())
    }
    fn to_f32_device(
        k: &mut GpuKernels,
        i: &CudaSlice<Self>,
        o: &mut CudaSlice<f32>,
        n: u32,
    ) -> anyhow::Result<()> {
        k.ops.copy_f16_to_f32(i, o, n)?;
        Ok(())
    }
}

impl private::Sealed for bf16 {}
impl QwenDType for bf16 {
    fn load(
        model_dir: &Path,
        name: &str,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<(Vec<usize>, CudaSlice<Self>)> {
        let tensor =
            load_bf16_tensor(model_dir, name).with_context(|| format!("could not load {name}"))?;
        Ok((tensor.shape, stream.clone_htod(&tensor.values)?))
    }
    fn zero() -> Self {
        bf16::ZERO
    }
    fn to_f32(value: Self) -> f32 {
        value.to_f32()
    }
    fn from_f32(value: f32) -> Self {
        bf16::from_f32(value)
    }
    fn rms_norm_f32in(
        k: &mut GpuKernels,
        i: &CudaSlice<f32>,
        w: &CudaSlice<Self>,
        o: &mut CudaSlice<Self>,
        r: u32,
        c: u32,
        e: f32,
    ) -> anyhow::Result<()> {
        k.ops.rms_norm_f32in_bf16(i, w, o, r, c, e)?;
        Ok(())
    }
    fn matmul(
        k: &mut GpuKernels,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        c: &mut CudaSlice<Self>,
        m: u32,
        n: u32,
        d: u32,
    ) -> anyhow::Result<()> {
        k.gemm.matmul_bf16(a, b, c, m, n, d)?;
        Ok(())
    }
    fn gemv(
        k: &mut GpuKernels,
        i: &CudaSlice<Self>,
        w: &CudaSlice<Self>,
        o: &mut CudaSlice<Self>,
        r: u32,
        c: u32,
    ) -> anyhow::Result<()> {
        k.gemv.gemv_bf16(i, w, o, r, c)?;
        Ok(())
    }
    fn gemv_dual(
        k: &mut GpuKernels,
        i: &CudaSlice<Self>,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        oa: &mut CudaSlice<Self>,
        ob: &mut CudaSlice<Self>,
        r: u32,
        c: u32,
    ) -> anyhow::Result<()> {
        k.gemv.gemv_dual_bf16(i, a, b, oa, ob, r, c)?;
        Ok(())
    }
    fn add_residual(
        k: &mut GpuKernels,
        h: &mut CudaSlice<f32>,
        r: &CudaSlice<Self>,
        n: u32,
    ) -> anyhow::Result<()> {
        k.ops.add_inplace_f32_bf16(h, r, n)?;
        Ok(())
    }
    fn silu(
        k: &mut GpuKernels,
        g: &CudaSlice<Self>,
        u: &CudaSlice<Self>,
        o: &mut CudaSlice<Self>,
        n: u32,
    ) -> anyhow::Result<()> {
        k.ops.silu_bf16(g, u, o, n)?;
        Ok(())
    }
    fn silu_act(
        k: &mut GpuKernels,
        i: &CudaSlice<Self>,
        o: &mut CudaSlice<Self>,
        n: u32,
    ) -> anyhow::Result<()> {
        k.ops.silu_act_bf16(i, o, n)?;
        Ok(())
    }
    fn gather(
        k: &mut GpuKernels,
        t: &CudaSlice<Self>,
        i: &CudaSlice<i32>,
        o: &mut CudaSlice<Self>,
        r: u32,
        c: u32,
    ) -> anyhow::Result<()> {
        k.ops.gather_rows_bf16(t, i, o, r, c)?;
        Ok(())
    }
    fn add_bias(
        k: &mut GpuKernels,
        v: &mut CudaSlice<Self>,
        b: &CudaSlice<Self>,
        r: u32,
        c: u32,
    ) -> anyhow::Result<()> {
        k.ops.add_bias_bf16_inplace(v, b, r, c)?;
        Ok(())
    }
    fn to_f16(
        k: &mut GpuKernels,
        i: &CudaSlice<Self>,
        o: &mut CudaSlice<f16>,
        n: u32,
    ) -> anyhow::Result<()> {
        k.ops.copy_bf16_to_f16(i, o, n)?;
        Ok(())
    }
    fn from_f16(
        k: &mut GpuKernels,
        i: &CudaSlice<f16>,
        o: &mut CudaSlice<Self>,
        n: u32,
    ) -> anyhow::Result<()> {
        k.ops.copy_f16_to_bf16(i, o, n)?;
        Ok(())
    }
    fn to_f32_device(
        k: &mut GpuKernels,
        i: &CudaSlice<Self>,
        o: &mut CudaSlice<f32>,
        n: u32,
    ) -> anyhow::Result<()> {
        k.ops.copy_bf16_to_f32(i, o, n)?;
        Ok(())
    }
}
