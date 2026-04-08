/// Sampling parameters for token generation.
#[derive(Debug, Clone)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_window: 64,
        }
    }
}

/// Sample a token from f16 logits on the CPU.
///
/// Logits are downloaded from GPU, converted to f32, then sampled.
pub fn sample_token(
    logits_f32: &mut [f32],
    params: &SampleParams,
    recent_tokens: &[u32],
) -> u32 {
    let vocab_size = logits_f32.len();

    // Repetition penalty
    if params.repeat_penalty != 1.0 {
        let window_start = recent_tokens.len().saturating_sub(params.repeat_window);
        for &tok in &recent_tokens[window_start..] {
            if (tok as usize) < vocab_size {
                let l = &mut logits_f32[tok as usize];
                if *l > 0.0 {
                    *l /= params.repeat_penalty;
                } else {
                    *l *= params.repeat_penalty;
                }
            }
        }
    }

    // Temperature
    if params.temperature > 0.0 && params.temperature != 1.0 {
        for l in logits_f32.iter_mut() {
            *l /= params.temperature;
        }
    }

    // Top-K: keep only top_k highest logits using a min-heap (O(n log k) instead of O(n log n))
    let top_k = (params.top_k as usize).min(vocab_size);
    let mut heap: Vec<(f32, u32)> = Vec::with_capacity(top_k + 1);
    for i in 0..vocab_size {
        let v = logits_f32[i];
        if heap.len() < top_k {
            heap.push((v, i as u32));
            if heap.len() == top_k {
                // Build min-heap
                for j in (0..top_k / 2).rev() {
                    sift_down(&mut heap, j);
                }
            }
        } else if v > heap[0].0 {
            heap[0] = (v, i as u32);
            sift_down(&mut heap, 0);
        }
    }
    // Sort the top-K descending
    heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut indices: Vec<u32> = heap.iter().map(|&(_, i)| i).collect();

    // Softmax over top-K
    let max_logit = logits_f32[indices[0] as usize];
    let mut probs: Vec<f32> = indices
        .iter()
        .map(|&i| (logits_f32[i as usize] - max_logit).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Top-P (nucleus sampling)
    if params.top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= params.top_p {
                cutoff = i + 1;
                break;
            }
        }
        indices.truncate(cutoff);
        probs.truncate(cutoff);
        // Re-normalize
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }
    }

    // Weighted random selection
    let r: f32 = simple_random();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return indices[i];
        }
    }

    // Fallback: return the most likely token
    indices[0]
}

/// Min-heap sift-down for top-K selection.
fn sift_down(heap: &mut [(f32, u32)], mut pos: usize) {
    let len = heap.len();
    loop {
        let mut smallest = pos;
        let left = 2 * pos + 1;
        let right = 2 * pos + 2;
        if left < len && heap[left].0 < heap[smallest].0 { smallest = left; }
        if right < len && heap[right].0 < heap[smallest].0 { smallest = right; }
        if smallest == pos { break; }
        heap.swap(pos, smallest);
        pos = smallest;
    }
}

/// Simple pseudo-random f32 in [0, 1) using thread-local state.
fn simple_random() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(0xdeadbeef12345678);
    }
    STATE.with(|s| {
        // xorshift64
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 40) as f32 / (1u64 << 24) as f32
    })
}
