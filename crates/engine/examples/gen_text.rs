//! Basic autoregressive generation sanity check (any non-diffusion model).
//! Usage: cargo run --release --example gen_text -- /path/to/model.gguf ["prompt"]
use chew_engine::sample::SampleParams;
use chew_engine::ChewEngine;
use chew_gguf::GgufFile;
use chew_vram::VramAllocator;

fn main() {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let path = std::env::args().nth(1).expect("usage: gen_text <model.gguf> [prompt]");
    let user = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Write a Python function is_prime(n).".to_string());

    let gguf = GgufFile::open(&path).expect("open");
    let tokenizer = chew_gguf::extract_tokenizer(&gguf.header).expect("tokenizer");
    let eos = gguf.header.eos_token_id().unwrap_or(1);

    let alloc = VramAllocator::init().expect("CUDA init");
    let mut engine = ChewEngine::load(&path, &alloc, 0, Some(1024)).expect("load");

    let prompt = format!("<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n");
    let tokens: Vec<u32> = tokenizer.encode(prompt.as_str(), true).expect("encode").get_ids().to_vec();
    println!("prompt tokens: {}", tokens.len());

    let params = SampleParams { temperature: 0.0, top_k: 1, top_p: 1.0, repeat_penalty: 1.0, repeat_window: 0 };
    let t0 = std::time::Instant::now();
    let out = engine.generate(&tokens, 16, &params, eos).expect("generate");
    let dt = t0.elapsed();
    let text = tokenizer.decode(&out, false).unwrap_or_default();
    println!("\n=== OUTPUT ({} tok, {:.1}s, {:.1} tok/s) ===", out.len(), dt.as_secs_f32(), out.len() as f32 / dt.as_secs_f32());
    println!("{text}");
}
