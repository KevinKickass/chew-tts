//! Dump tensor info for a GGUF model.
use chew_gguf::GgufFile;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: tensors <model.gguf>");
    let gguf = GgufFile::open(&path).expect("failed to open GGUF");

    for t in &gguf.tensors {
        if t.name.contains("blk.0.attn") || t.name.contains("blk.0.ffn") {
            println!(
                "{:<40} shape={:<20?} type={:<8?} bytes={:<10} elems={}",
                t.name,
                t.shape,
                t.ggml_type,
                t.data_size(),
                t.n_elements()
            );
        }
    }
}
