use anyhow::Context;
use chew_model_qwen3_tts::inspect_model;
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
}

fn main() -> anyhow::Result<()> {
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
    }
    Ok(())
}
