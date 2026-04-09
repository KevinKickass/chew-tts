use chew_gguf::{GgufFile, extract_tokenizer};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let path = std::env::args().nth(1).expect("usage: tokenizer_test <model.gguf>");
    let gguf = GgufFile::open(&path).expect("failed to open GGUF");

    let tokenizer = extract_tokenizer(&gguf.header).expect("failed to extract tokenizer");

    // Test encoding
    let test = "Hello, world! How are you?";
    let encoding = tokenizer.encode(test, false).expect("encode failed");
    let ids = encoding.get_ids();
    println!("Input:  {test:?}");
    println!("Tokens: {ids:?} ({} tokens)", ids.len());

    // Test decoding
    let decoded = tokenizer.decode(ids, true).expect("decode failed");
    println!("Decoded: {decoded:?}");

    // Test special tokens
    for name in ["<turn|>", "<end_of_turn>", "<|eot_id|>", "</s>", "<bos>", "<eos>", "<pad>"] {
        if let Some(id) = tokenizer.token_to_id(name) {
            println!("Special: {name:?} -> {id}");
        }
    }

    // Show vocab size
    println!("Vocab size: {}", tokenizer.get_vocab_size(true));
}
