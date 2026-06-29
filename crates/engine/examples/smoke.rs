//! Smoke test: load a GGUF model onto the GPU.
//! Usage: cargo run --release --example smoke -- /path/to/model.gguf

use chew_engine::ChewEngine;
use chew_vram::VramAllocator;

fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let path = std::env::args().nth(1).expect("usage: smoke <model.gguf>");

    println!("=== Chew Smoke Test ===");
    println!("Model: {path}");

    let alloc = VramAllocator::init().expect("CUDA init failed");

    println!(
        "\nGPU 0: {} MB free, {} MB total",
        alloc.free_bytes(0).unwrap() / (1024 * 1024),
        alloc.total_bytes(0).unwrap() / (1024 * 1024),
    );

    println!("\nLoading model...");
    match ChewEngine::load(&path, &alloc, 0, Some(512)) {
        Ok(engine) => {
            let cfg = engine.config();
            println!("\n=== Model loaded successfully! ===");
            println!("  Arch:       {}", cfg.arch);
            println!("  Layers:     {}", cfg.n_layers);
            println!("  Dim:        {}", cfg.dim);
            println!("  Heads:      {}", cfg.n_heads);
            println!("  KV Heads:   {}", cfg.n_kv_heads);
            println!("  Head Dim:   {}", cfg.head_dim);
            println!("  FF Dim:     {}", cfg.ff_dim);
            println!("  Vocab:      {}", cfg.vocab_size);
            println!("  Context:    {}", cfg.context_length);
            println!("  GQA:        {}", cfg.is_gqa());

            println!(
                "\nGPU 0 after load: {} MB free",
                alloc.free_bytes(0).unwrap() / (1024 * 1024)
            );
        }
        Err(e) => {
            eprintln!("\n=== Load FAILED ===");
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}
