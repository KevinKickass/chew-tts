//! End-to-end DiffusionGemma generation in chew.
//! Usage: cargo run --release --example diffusion_generate -- /path/to/model.gguf ["prompt"]
use chew_engine::arch::diffusion_gemma::EbParams;
use chew_engine::ChewEngine;
use chew_gguf::GgufFile;
use chew_vram::VramAllocator;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".into()))
        .init();
    let path = std::env::args()
        .nth(1)
        .expect("usage: diffusion_generate <model.gguf> [prompt]");
    let user = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Write a Python function is_prime(n).".to_string());

    let gguf = GgufFile::open(&path).expect("open gguf");
    let tokenizer = chew_gguf::extract_tokenizer(&gguf.header).expect("tokenizer");
    let eos = gguf.header.eos_token_id().unwrap_or(1);

    let alloc = VramAllocator::init().expect("CUDA init");
    println!("loading {path} ...");
    let mut engine = ChewEngine::load(&path, &alloc, 0, Some(1024)).expect("load");

    // Gemma chat template (or raw prompt for cross-impl comparison)
    let prompt = if std::env::var("CHEW_RAW").is_ok() {
        user.clone()
    } else {
        format!("<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n")
    };
    let enc = tokenizer.encode(prompt.as_str(), true).expect("encode");
    let mut tokens: Vec<u32> = enc.get_ids().to_vec();
    // gemma expects a leading BOS (token 2); the HF tokenizer here doesn't add it.
    if std::env::var("CHEW_NO_BOS").is_err() && tokens.first() != Some(&2) {
        tokens.insert(0, 2);
    }
    println!("prompt tokens: {} -> {:?}", tokens.len(), &tokens[..tokens.len().min(12)]);

    let eb = EbParams::default();
    let t0 = std::time::Instant::now();
    let out = engine
        .generate_diffusion(&tokens, eb, 1234, eos)
        .expect("generate");
    let dt = t0.elapsed();

    let text = tokenizer.decode(&out, false).unwrap_or_default();
    println!("\n=== OUTPUT ({} tokens, {:.1}s) ===", out.len(), dt.as_secs_f32());
    println!("{text}");
    println!("\nfirst 20 token ids: {:?}", &out[..out.len().min(20)]);
}
