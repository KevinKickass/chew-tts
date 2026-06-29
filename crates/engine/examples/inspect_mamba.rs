use chew_engine::{arch::mamba::MambaLayout, config::ModelConfig};
use chew_gguf::GgufFile;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: inspect_mamba <model.gguf>");
    let gguf = GgufFile::open(&path)?;
    let config = ModelConfig::from_gguf_unchecked(&gguf.header)?;
    if config.arch != "mamba" {
        return Err(format!("expected mamba GGUF, got arch {}", config.arch).into());
    }
    let layout = MambaLayout::inspect(&gguf, &config)?;
    println!("arch        : {}", config.arch);
    println!("layers      : {}", layout.n_layers);
    println!("model_dim   : {}", layout.model_dim);
    println!("inner_dim   : {}", layout.inner_dim);
    println!("state_dim   : {}", layout.state_dim);
    println!("conv_kernel : {}", layout.conv_kernel);
    println!("dt_rank     : {}", layout.dt_rank);
    println!("vocab_size  : {}", layout.vocab_size);
    println!("bos_token_id: {:?}", gguf.header.bos_token_id());
    println!("eos_token_id: {:?}", gguf.header.eos_token_id());
    println!(
        "chat_tmpl   : {}",
        gguf.header
            .chat_template()
            .map(|s| if s.is_empty() { "<empty>" } else { "<present>" })
            .unwrap_or("<none>")
    );
    Ok(())
}
