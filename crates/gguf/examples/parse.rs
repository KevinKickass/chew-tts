use chew_gguf::GgufFile;
use std::env;

fn main() {
    let path = env::args().nth(1).expect("usage: parse <file.gguf>");
    let gguf = GgufFile::open(&path).expect("failed to parse GGUF");

    println!("=== GGUF Header ===");
    println!("  Version:    {}", gguf.header.version);
    println!(
        "  Arch:       {}",
        gguf.header.architecture().unwrap_or("?")
    );
    println!("  Model:      {}", gguf.header.model_name().unwrap_or("?"));
    println!("  Vocab:      {:?}", gguf.header.vocab_size());
    println!("  Layers:     {:?}", gguf.header.block_count());
    println!("  Embed dim:  {:?}", gguf.header.embedding_length());
    println!("  Heads:      {:?}", gguf.header.head_count());
    println!("  KV heads:   {:?}", gguf.header.head_count_kv());
    println!("  RoPE base:  {:?}", gguf.header.rope_freq_base());
    println!("  RMS eps:    {:?}", gguf.header.rms_norm_eps());
    println!("  FF length:  {:?}", gguf.header.feed_forward_length());
    println!("  Context:    {:?}", gguf.header.context_length());
    println!("  Tensors:    {}", gguf.header.n_tensors);
    println!(
        "  Data size:  {:.1} MB",
        gguf.total_data_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!();

    // Count quant types
    let mut type_counts: std::collections::HashMap<String, (usize, u64)> =
        std::collections::HashMap::new();
    for t in &gguf.tensors {
        let entry = type_counts.entry(format!("{}", t.ggml_type)).or_default();
        entry.0 += 1;
        entry.1 += t.data_size();
    }
    println!("=== Quant Type Distribution ===");
    let mut types: Vec<_> = type_counts.iter().collect();
    types.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    for (name, (count, bytes)) in &types {
        println!(
            "  {:10} {:4} tensors  {:8.1} MB",
            name,
            count,
            *bytes as f64 / (1024.0 * 1024.0)
        );
    }
    println!();

    println!("=== First 20 Tensors ===");
    for t in gguf.tensors.iter().take(20) {
        let shape: Vec<String> = t.shape.iter().map(|s| s.to_string()).collect();
        println!(
            "  {:55} {:8} [{:>20}] {:>8.2} MB",
            t.name,
            format!("{}", t.ggml_type),
            shape.join(" x "),
            t.data_size() as f64 / (1024.0 * 1024.0),
        );
    }
    if gguf.tensors.len() > 20 {
        println!("  ... ({} more)", gguf.tensors.len() - 20);
    }
}
