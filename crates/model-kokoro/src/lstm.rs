use crate::KokoroCheckpoint;
use anyhow::ensure;
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::sync::Arc;

struct LstmDirection {
    weight_ih: CudaSlice<f16>,
    weight_hh: CudaSlice<f16>,
    bias_ih: CudaSlice<f16>,
    bias_hh: CudaSlice<f16>,
}

/// A single-layer PyTorch-compatible bidirectional LSTM.
pub struct KokoroBiLstm {
    forward: LstmDirection,
    reverse: LstmDirection,
    input_size: usize,
    hidden_size: usize,
}

impl KokoroBiLstm {
    pub fn load(
        checkpoint: &KokoroCheckpoint,
        group: &str,
        prefix: &str,
        input_size: usize,
        hidden_size: usize,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            forward: LstmDirection::load(
                checkpoint,
                group,
                prefix,
                input_size,
                hidden_size,
                false,
                stream,
            )?,
            reverse: LstmDirection::load(
                checkpoint,
                group,
                prefix,
                input_size,
                hidden_size,
                true,
                stream,
            )?,
            input_size,
            hidden_size,
        })
    }

    /// Run frame-major input and return frame-major `[T, 2*hidden]`.
    pub fn forward(
        &self,
        input: &[f32],
        frames: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            input.len() == frames * self.input_size && frames > 0,
            "invalid Kokoro BiLSTM input geometry"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let input =
            stream.clone_htod(&input.iter().copied().map(f16::from_f32).collect::<Vec<_>>())?;
        let forward = self.forward.run(
            &input,
            frames,
            self.input_size,
            self.hidden_size,
            false,
            kernels,
        )?;
        let reverse = self.reverse.run(
            &input,
            frames,
            self.input_size,
            self.hidden_size,
            true,
            kernels,
        )?;
        stream.synchronize()?;
        let mut forward_host = vec![f16::ZERO; frames * self.hidden_size];
        let mut reverse_host = vec![f16::ZERO; frames * self.hidden_size];
        stream.memcpy_dtoh(&forward, &mut forward_host)?;
        stream.memcpy_dtoh(&reverse, &mut reverse_host)?;
        let mut output = Vec::with_capacity(frames * self.hidden_size * 2);
        for frame in 0..frames {
            output.extend(
                forward_host[frame * self.hidden_size..(frame + 1) * self.hidden_size]
                    .iter()
                    .map(|value| value.to_f32()),
            );
            output.extend(
                reverse_host[frame * self.hidden_size..(frame + 1) * self.hidden_size]
                    .iter()
                    .map(|value| value.to_f32()),
            );
        }
        Ok(output)
    }
}

impl LstmDirection {
    #[allow(clippy::too_many_arguments)]
    fn load(
        checkpoint: &KokoroCheckpoint,
        group: &str,
        prefix: &str,
        input_size: usize,
        hidden_size: usize,
        reverse: bool,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let suffix = if reverse { "_reverse" } else { "" };
        let load = |name: &str, expected: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let (shape, values) =
                checkpoint.tensor_f16(group, &format!("{prefix}.{name}_l0{suffix}"))?;
            ensure!(
                shape == expected,
                "invalid {group}.{prefix}.{name}_l0{suffix} shape {shape:?}"
            );
            Ok(stream.clone_htod(&values)?)
        };
        Ok(Self {
            weight_ih: load("weight_ih", &[hidden_size * 4, input_size])?,
            weight_hh: load("weight_hh", &[hidden_size * 4, hidden_size])?,
            bias_ih: load("bias_ih", &[hidden_size * 4])?,
            bias_hh: load("bias_hh", &[hidden_size * 4])?,
        })
    }

    fn run(
        &self,
        input: &CudaSlice<f16>,
        frames: usize,
        input_size: usize,
        hidden_size: usize,
        reverse: bool,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let stream = Arc::clone(kernels.ops.stream());
        let gate_width = hidden_size * 4;
        let mut input_gates = stream.alloc_zeros::<f16>(frames * gate_width)?;
        kernels.gemm.matmul_f16(
            input,
            &self.weight_ih,
            &mut input_gates,
            frames as u32,
            gate_width as u32,
            input_size as u32,
        )?;
        let mut hidden = stream.alloc_zeros::<f16>(hidden_size)?;
        let mut cell = stream.alloc_zeros::<f32>(hidden_size)?;
        let mut output = stream.alloc_zeros::<f16>(frames * hidden_size)?;
        for step in 0..frames {
            let timestep = if reverse { frames - 1 - step } else { step };
            let mut hidden_gates = stream.alloc_zeros::<f16>(gate_width)?;
            kernels.gemm.matmul_f16(
                &hidden,
                &self.weight_hh,
                &mut hidden_gates,
                1,
                gate_width as u32,
                hidden_size as u32,
            )?;
            kernels.ops.lstm_cell_f16(
                &input_gates,
                &hidden_gates,
                &self.bias_ih,
                &self.bias_hh,
                &mut hidden,
                &mut cell,
                &mut output,
                hidden_size as u32,
                timestep as u32,
                timestep as u32,
            )?;
        }
        Ok(output)
    }
}
