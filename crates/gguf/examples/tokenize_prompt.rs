use chew_gguf::{GgufFile, extract_tokenizer};

fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("usage: tokenize_prompt <model.gguf> <prompt-file>");
    let prompt_file = std::env::args()
        .nth(2)
        .expect("usage: tokenize_prompt <model.gguf> <prompt-file>");

    let gguf = GgufFile::open(&model).expect("failed to open GGUF");
    let tokenizer = extract_tokenizer(&gguf.header).expect("failed to extract tokenizer");
    let prompt = std::fs::read_to_string(prompt_file).expect("failed to read prompt file");

    let encoding = tokenizer.encode(prompt, false).expect("encode failed");
    println!("{:?}", encoding.get_ids());
}
