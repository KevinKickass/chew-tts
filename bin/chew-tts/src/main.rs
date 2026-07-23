use anyhow::Context;
use chew_model_qwen3_tts::{TalkerDecoderLayer, inspect_model, load_f16_tensor};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate a Qwen3-TTS model and print its inference geometry.
    Inspect {
        /// Directory containing config.json and Safetensors weights.
        model_dir: PathBuf,
    },
    /// Compile and load Chew's CUDA kernels for the selected GPU.
    CudaSmoke {
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
    },
    /// Validate one real Qwen linear layer against a CPU reference.
    CudaLinearSmoke {
        /// Directory containing config.json and Safetensors weights.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Two-dimensional weight tensor to test.
        #[arg(long, default_value = "talker.model.layers.0.self_attn.q_proj.weight")]
        tensor: String,
    },
    /// Run one complete native Qwen talker decoder layer on CUDA.
    CudaLayerSmoke {
        /// Directory containing config.json and Safetensors weights.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Decoder layer to load.
        #[arg(long, default_value_t = 0)]
        layer: usize,
        /// Optional raw little-endian f32 reference output.
        #[arg(long)]
        reference: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    match Cli::parse().command {
        Command::Inspect { model_dir } => {
            let inspection = inspect_model(&model_dir)
                .with_context(|| format!("could not inspect {}", model_dir.display()))?;
            let talker = &inspection.config.talker_config;
            let predictor = &talker.code_predictor_config;
            println!(
                "Qwen3-TTS {} {:?}",
                inspection.config.tts_model_size, inspection.config.tts_model_type
            );
            println!(
                "talker: {} layers, hidden {}, {} Q heads / {} KV heads",
                talker.num_hidden_layers,
                talker.hidden_size,
                talker.num_attention_heads,
                talker.num_key_value_heads
            );
            println!(
                "code predictor: {} layers, hidden {}, {} acoustic steps/frame",
                predictor.num_hidden_layers,
                predictor.hidden_size,
                talker.num_code_groups - 1
            );
            println!(
                "weights: {} tensors in {} file(s), {:.2} GiB",
                inspection.tensors.len(),
                inspection.weight_files.len(),
                inspection.total_weight_bytes as f64 / 1024.0_f64.powi(3)
            );
        }
        Command::CudaSmoke { gpu } => {
            let allocator = chew_vram::VramAllocator::init()?;
            if gpu >= allocator.gpu_count() {
                anyhow::bail!(
                    "GPU index {gpu} is out of range; detected {} device(s)",
                    allocator.gpu_count()
                );
            }
            let free_before = allocator.free_bytes(gpu)?;
            let stream = allocator.stream(gpu);
            let _kernels = chew_kernel::GpuKernels::load(stream, 1024 * 1024, 6 * 1024)
                .context("could not compile and load CUDA kernels")?;
            stream.synchronize()?;
            let free_after = allocator.free_bytes(gpu)?;
            println!(
                "CUDA device {gpu} ready: {:.1} MiB free, {:.1} MiB kernel/runtime allocation",
                free_after as f64 / 1024.0_f64.powi(2),
                free_before.saturating_sub(free_after) as f64 / 1024.0_f64.powi(2),
            );
        }
        Command::CudaLinearSmoke {
            model_dir,
            gpu,
            tensor,
        } => cuda_linear_smoke(&model_dir, gpu, &tensor)?,
        Command::CudaLayerSmoke {
            model_dir,
            gpu,
            layer,
            reference,
        } => cuda_layer_smoke(&model_dir, gpu, layer, reference.as_deref())?,
    }
    Ok(())
}

fn cuda_linear_smoke(model_dir: &PathBuf, gpu: usize, tensor_name: &str) -> anyhow::Result<()> {
    let tensor = load_f16_tensor(model_dir, tensor_name)
        .with_context(|| format!("could not load tensor {tensor_name}"))?;
    let [n, k]: [usize; 2] = tensor
        .shape
        .clone()
        .try_into()
        .map_err(|shape: Vec<usize>| anyhow::anyhow!("expected matrix, got shape {shape:?}"))?;

    let allocator = chew_vram::VramAllocator::init()?;
    if gpu >= allocator.gpu_count() {
        anyhow::bail!(
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
    }
    let stream = allocator.stream(gpu);
    let kernels = chew_kernel::GpuKernels::load(stream, n * k, k)?;

    let input = (0..k)
        .map(|index| half::f16::from_f32(((index as f32 + 1.0) * 0.013).sin() * 0.125))
        .collect::<Vec<_>>();
    let input_gpu = stream.clone_htod(&input)?;
    let weights_gpu = stream.clone_htod(&tensor.values)?;
    let mut output_gpu = stream.alloc_zeros::<half::f16>(n)?;
    kernels.gemm.matmul_f16(
        &input_gpu,
        &weights_gpu,
        &mut output_gpu,
        1,
        n as u32,
        k as u32,
    )?;
    let mut output = vec![half::f16::ZERO; n];
    stream.memcpy_dtoh(&output_gpu, &mut output)?;

    let sample_rows = [0, n / 3, (2 * n) / 3, n - 1];
    let mut max_abs_error = 0.0f32;
    for row in sample_rows {
        let weights = &tensor.values[row * k..(row + 1) * k];
        let expected = weights
            .iter()
            .zip(&input)
            .map(|(weight, value)| weight.to_f32() * value.to_f32())
            .sum::<f32>();
        let actual = output[row].to_f32();
        let error = (expected - actual).abs();
        max_abs_error = max_abs_error.max(error);
        println!("row {row}: GPU={actual:.6}, CPU={expected:.6}, abs_error={error:.6}");
    }
    if max_abs_error > 0.08 {
        anyhow::bail!("linear parity failed: maximum absolute error {max_abs_error:.6}");
    }
    println!("linear parity passed for {tensor_name} [{n}, {k}], max abs error {max_abs_error:.6}");
    Ok(())
}

fn cuda_layer_smoke(
    model_dir: &PathBuf,
    gpu: usize,
    layer: usize,
    reference: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    let inspection = inspect_model(model_dir)?;
    let config = &inspection.config.talker_config;
    if layer >= config.num_hidden_layers {
        anyhow::bail!(
            "layer {layer} is out of range for {} talker layers",
            config.num_hidden_layers
        );
    }
    let allocator = chew_vram::VramAllocator::init()?;
    if gpu >= allocator.gpu_count() {
        anyhow::bail!(
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
    }
    let free_before = allocator.free_bytes(gpu)?;
    let stream = allocator.stream(gpu);
    let mut kernels = chew_kernel::GpuKernels::load(
        stream,
        config.intermediate_size * config.hidden_size,
        config.intermediate_size,
    )?;
    let decoder = TalkerDecoderLayer::load(model_dir, layer, config, stream)?;
    let hidden = (0..config.hidden_size)
        .map(|index| ((index as f32 + 1.0) * 0.013).sin() * 0.125)
        .collect::<Vec<_>>();
    let output = decoder.forward_first_token(&hidden, config, &mut kernels)?;
    let free_after = allocator.free_bytes(gpu)?;
    let checksum = output
        .iter()
        .enumerate()
        .map(|(index, value)| (index as f64 + 1.0) * f64::from(*value))
        .sum::<f64>();
    let first = output.iter().take(8).copied().collect::<Vec<_>>();
    println!("layer {layer} output[0..8]: {first:?}");
    println!("weighted checksum: {checksum:.9}");
    if let Some(reference) = reference {
        let bytes = std::fs::read(reference)?;
        if bytes.len() != output.len() * 4 {
            anyhow::bail!(
                "{} has {} bytes, expected {} raw f32 bytes",
                reference.display(),
                bytes.len(),
                output.len() * 4
            );
        }
        let expected = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        let mut max_abs_error = 0.0f32;
        let mut mean_abs_error = 0.0f64;
        for (actual, expected) in output.iter().zip(&expected) {
            let error = (actual - expected).abs();
            max_abs_error = max_abs_error.max(error);
            mean_abs_error += f64::from(error);
        }
        mean_abs_error /= output.len() as f64;
        if max_abs_error > 0.002 {
            anyhow::bail!(
                "layer parity failed: max abs error {max_abs_error:.7}, mean {mean_abs_error:.7}"
            );
        }
        println!("layer parity passed: max abs error {max_abs_error:.7}, mean {mean_abs_error:.7}");
    }
    println!(
        "CUDA allocation: {:.1} MiB",
        free_before.saturating_sub(free_after) as f64 / 1024.0_f64.powi(2)
    );
    Ok(())
}
