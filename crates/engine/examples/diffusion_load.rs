//! Verify DiffusionGemma config parsing + required tensors are present.
//! Usage: cargo run --example diffusion_load -- /path/to/diffusiongemma.gguf

use chew_engine::config::ModelConfig;
use chew_gguf::GgufFile;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: diffusion_load <model.gguf>");

    let gguf = GgufFile::open(&path).expect("failed to open GGUF");
    let config = ModelConfig::from_gguf(&gguf.header).expect("failed to parse config");

    println!("Arch:              {}", config.arch);
    println!("is_gemma4:         {}", config.is_gemma4());
    println!("is_diffusion_gemma:{}", config.is_diffusion_gemma());
    println!("is_diffusion:      {}", config.is_diffusion());
    println!("canvas_length:     {:?}", config.canvas_length);
    println!("mask_token_id:     {:?}", config.mask_token_id);
    println!(
        "layers/dim/heads:  {} / {} / {} (kv max {})",
        config.n_layers, config.dim, config.n_heads, config.n_kv_heads
    );
    println!(
        "MoE:               {} experts, {} used, expert_ff {}",
        config.n_experts, config.n_experts_per_tok, config.expert_ff_dim
    );
    println!("attention_scale:   {}", config.attention_scale);
    println!("logit_softcap:     {:?}", config.logit_softcap);

    // Required diffusion tensors
    let mut missing = Vec::new();
    let sc_tensors = [
        "self_cond_pre_norm.weight",
        "self_cond_gate.weight",
        "self_cond_up.weight",
        "self_cond_down.weight",
    ];
    println!("\nSelf-conditioning tensors:");
    for t in sc_tensors {
        let present = gguf.find_tensor(t).is_some();
        println!("  {:<28} {}", t, if present { "OK" } else { "MISSING" });
        if !present {
            missing.push(t);
        }
    }

    // Per-layer scales (canvas + encoder/prompt)
    let mut canvas_scales = 0;
    let mut enc_scales = 0;
    for l in 0..config.n_layers {
        if gguf
            .find_tensor(&format!("blk.{l}.layer_output_scale.weight"))
            .is_some()
        {
            canvas_scales += 1;
        }
        if gguf
            .find_tensor(&format!("blk.{l}.enc_layer_output_scale.weight"))
            .is_some()
        {
            enc_scales += 1;
        }
    }
    println!(
        "\nPer-layer scales: {}/{} canvas, {}/{} encoder",
        canvas_scales, config.n_layers, enc_scales, config.n_layers
    );

    assert!(config.is_diffusion_gemma(), "arch not recognised as diffusion-gemma");
    assert!(config.is_diffusion(), "canvas_length missing");
    assert!(missing.is_empty(), "missing SC tensors: {missing:?}");
    assert_eq!(canvas_scales, config.n_layers, "missing canvas scales");
    assert_eq!(enc_scales, config.n_layers, "missing encoder scales");
    println!("\nALL CHECKS PASSED");
}
