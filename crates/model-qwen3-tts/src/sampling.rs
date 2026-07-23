use half::f16;

pub(crate) fn sample_top_k(
    logits: &[f16],
    allowed: impl Fn(usize) -> bool,
    temperature: f32,
    top_k: usize,
    previous: &[i32],
    repetition_penalty: f32,
    seed: &mut u64,
) -> i32 {
    let temperature = temperature.max(1e-5);
    let mut candidates = logits
        .iter()
        .enumerate()
        .filter(|(token, _)| allowed(*token))
        .map(|(token, logit)| {
            let mut value = logit.to_f32();
            if previous.contains(&(token as i32)) && repetition_penalty != 1.0 {
                value = if value >= 0.0 {
                    value / repetition_penalty
                } else {
                    value * repetition_penalty
                };
            }
            (token, value / temperature)
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    candidates.truncate(top_k.max(1).min(candidates.len()));

    let max = candidates[0].1;
    let weights = candidates
        .iter()
        .map(|(_, logit)| (logit - max).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    let mut threshold = random_unit(seed) * total;
    for ((token, _), weight) in candidates.iter().zip(weights) {
        if threshold <= weight {
            return *token as i32;
        }
        threshold -= weight;
    }
    candidates.last().map_or(0, |(token, _)| *token as i32)
}

fn random_unit(seed: &mut u64) -> f32 {
    if *seed == 0 {
        *seed = 0x9e37_79b9_7f4a_7c15;
    }
    *seed ^= *seed >> 12;
    *seed ^= *seed << 25;
    *seed ^= *seed >> 27;
    let value = seed.wrapping_mul(0x2545_f491_4f6c_dd1d);
    ((value >> 40) as f32) / ((1u32 << 24) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_one_is_argmax() {
        let logits = [f16::from_f32(-1.0), f16::from_f32(4.0), f16::from_f32(2.0)];
        let mut seed = 42;
        assert_eq!(
            sample_top_k(&logits, |_| true, 0.9, 1, &[], 1.0, &mut seed),
            1
        );
    }

    #[test]
    fn filter_excludes_largest_logit() {
        let logits = [f16::from_f32(9.0), f16::from_f32(4.0), f16::from_f32(2.0)];
        let mut seed = 42;
        assert_eq!(
            sample_top_k(&logits, |token| token != 0, 0.9, 1, &[], 1.0, &mut seed),
            1
        );
    }
}
