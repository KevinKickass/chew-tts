//! Microbenchmark: can CPU expert-FFN matmul beat the ~13s/step DMA streaming?
//! Models one diffusion step's MoE compute: C tokens, top-k experts, all layers.
//! gate_up: [dim, 2*ff] @ [dim] -> [2*ff]; down: [ff, dim] @ [ff] -> [dim].
//! Weights as f16 (RAM-resident strategy), accumulate in f32. Threaded.
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let dim = 2816usize;
    let ff = 704usize;
    let c = 256usize; // canvas tokens
    let topk = 8usize;
    let layers = 28usize; // streamed layers
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(16);

    // f16 weights for ONE expert (gate_up fused [dim,2ff] + down [ff,dim]), shared.
    let gate_up: Arc<Vec<half::f16>> =
        Arc::new((0..dim * 2 * ff).map(|i| half::f16::from_f32((i % 7) as f32 * 0.01 - 0.03)).collect());
    let down: Arc<Vec<half::f16>> =
        Arc::new((0..ff * dim).map(|i| half::f16::from_f32((i % 5) as f32 * 0.01 - 0.02)).collect());
    let x: Arc<Vec<f32>> = Arc::new((0..dim).map(|i| (i % 11) as f32 * 0.01).collect());

    // One expert-FFN: gate_up @ x -> [2ff], gelu*up -> [ff], down @ [ff] -> [dim]
    let expert_ffn = |gu: &[half::f16], dn: &[half::f16], x: &[f32]| -> Vec<f32> {
        let mut gu_out = vec![0.0f32; 2 * ff];
        for o in 0..2 * ff {
            let row = &gu[o * dim..(o + 1) * dim];
            let mut acc = 0.0f32;
            for k in 0..dim {
                acc += row[k].to_f32() * x[k];
            }
            gu_out[o] = acc;
        }
        let mut act = vec![0.0f32; ff];
        for j in 0..ff {
            let g = gu_out[j];
            let u = gu_out[ff + j];
            let gelu = 0.5 * g * (1.0 + (0.7978845608 * (g + 0.044715 * g * g * g)).tanh());
            act[j] = gelu * u;
        }
        let mut out = vec![0.0f32; dim];
        for o in 0..dim {
            let row = &dn[o * ff..(o + 1) * ff];
            let mut acc = 0.0f32;
            for k in 0..ff {
                acc += row[k].to_f32() * act[k];
            }
            out[o] = acc;
        }
        out
    };

    let total_experts = c * topk * layers;
    println!("threads={threads}, expert-FFN calls={total_experts} (C={c} topk={topk} layers={layers})");

    let t0 = Instant::now();
    let per_thread = total_experts.div_ceil(threads);
    std::thread::scope(|s| {
        for t in 0..threads {
            let (gu, dn, x) = (gate_up.clone(), down.clone(), x.clone());
            s.spawn(move || {
                let start = t * per_thread;
                let end = (start + per_thread).min(total_experts);
                let mut sink = 0.0f32;
                for _ in start..end {
                    let o = expert_ffn(&gu, &dn, &x);
                    sink += o[0];
                }
                std::hint::black_box(sink);
            });
        }
    });
    let dt = t0.elapsed();
    let gflop = (total_experts as f64 * (dim * 2 * ff + ff * dim) as f64 * 2.0) / 1e9;
    println!(
        "CPU MoE compute / step: {:.2}s  ({:.0} GFLOP, {:.0} GFLOP/s)",
        dt.as_secs_f64(),
        gflop,
        gflop / dt.as_secs_f64()
    );
    println!("(vs ~13s/step current DMA-streamed GPU path)");
}
