//! Show VRAM budget for a model without loading anything.
//! Usage: cargo run --example vram_plan -- /path/to/model.gguf

use chew_engine::{config::ModelConfig, vram_plan::VramPlan};
use chew_gguf::GgufFile;
use chew_vram::VramAllocator;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: vram_plan <model.gguf>");

    let gguf = GgufFile::open(&path).expect("failed to open GGUF");
    let config = ModelConfig::from_gguf(&gguf.header).expect("failed to parse config");

    println!("Model: {path}");
    println!(
        "  Arch: {}, Layers: {}, Dim: {}, Heads: {}/{}, FF: {}, Vocab: {}",
        config.arch,
        config.n_layers,
        config.dim,
        config.n_heads,
        config.n_kv_heads,
        config.ff_dim,
        config.vocab_size
    );
    if gguf.find_tensor("output.weight").is_none() {
        println!("  Note: No output.weight — using tied embeddings");
    }
    println!();

    // Show budget for various context lengths
    println!(
        "{:<10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Context", "Weights", "Dequant", "KV", "Scratch", "cuBLAS", "Total", "Peak"
    );
    println!("{}", "-".repeat(74));

    for ctx in [512, 1024, 2048, 4096, 8192, 16384, 32768] {
        let batch = (ctx as u32).min(2048);
        let plan = VramPlan::compute(&config, &gguf, ctx, batch);
        println!(
            "{:<10} {:>6} MB {:>6} MB {:>6} MB {:>6} MB {:>6} MB {:>6} MB {:>6} MB",
            ctx,
            plan.weights_mb(),
            plan.dequant_scratch_mb(),
            plan.kv_cache_mb(),
            plan.scratch_mb(),
            plan.cublas_bytes / (1024 * 1024),
            plan.total_mb(),
            plan.peak_mb()
        );
    }

    // Try to detect GPU and auto-fit
    println!();
    if let Ok(alloc) = VramAllocator::init() {
        if let Ok(free) = alloc.free_bytes(0) {
            let free_mb = free / (1024 * 1024);

            if let Some(plan) = VramPlan::fit(&config, &gguf, 32768, free as u64) {
                plan.print_report(Some(free_mb));
            } else {
                println!("Model does not fit in GPU! ({} MB free)", free_mb);
                // Show minimum requirements
                let plan = VramPlan::compute(&config, &gguf, 256, 256);
                println!("Minimum (256 ctx): {} MB peak", plan.peak_mb());
            }
        }
    } else {
        println!("No GPU detected — showing numbers only.");
    }
}
