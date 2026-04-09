#!/usr/bin/env python3
"""
Performance benchmark: chew vs llama.cpp
Measures prefill tok/s, decode tok/s, TTFT, total throughput.
"""
import time, json, subprocess, signal, os, sys
import numpy as np

MODELS = {
    "llama-8b": {
        "gguf": "/run/media/kevin/KioxiaNVMe/NVMeR0/AI/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "tokenizer": "/run/media/kevin/KioxiaNVMe/KI-kram/models/llama3.1-8b-exl2-4bpw/tokenizer.json",
        "port": 8080,
    },
    "gemma4-e4b": {
        "gguf": "/run/media/kevin/KioxiaNVMe/NVMeR0/AI/gemma4/gemma-4-E4B-it-Q4_K_M.gguf",
        "tokenizer": "/run/media/kevin/KioxiaNVMe/NVMeR0/AI/gemma4/tokenizer.json",
        "port": 8082,
    },
}
CHEW_BIN = "/run/media/kevin/KioxiaNVMe/KI-kram/chew/target/release/chew"
# Default model for backward compat
MODEL = MODELS["llama-8b"]["gguf"]
TOKENIZER = MODELS["llama-8b"]["tokenizer"]

# Test cases: (name, prompt_text, max_tokens)
TESTS = [
    ("short→short",   "Hi",  16),
    ("short→medium",  "Hi",  64),
    ("short→long",    "Hi", 200),
    ("medium→short",  "Explain quantum computing in simple terms. What are qubits and how do they differ from classical bits? Give examples.", 16),
    ("medium→medium", "Explain quantum computing in simple terms. What are qubits and how do they differ from classical bits? Give examples.", 64),
    ("medium→long",   "Explain quantum computing in simple terms. What are qubits and how do they differ from classical bits? Give examples.", 200),
    ("long→short",    "Write a detailed analysis of the following topics: 1) The impact of artificial intelligence on modern healthcare, including diagnostics, drug discovery, and patient care. 2) The ethical considerations surrounding AI in medicine, including bias, privacy, and accountability. 3) Future predictions for AI-assisted surgery and telemedicine. 4) The role of machine learning in genomics and personalized medicine. Please provide specific examples and cite relevant developments from the past five years.", 16),
    ("long→medium",   "Write a detailed analysis of the following topics: 1) The impact of artificial intelligence on modern healthcare, including diagnostics, drug discovery, and patient care. 2) The ethical considerations surrounding AI in medicine, including bias, privacy, and accountability. 3) Future predictions for AI-assisted surgery and telemedicine. 4) The role of machine learning in genomics and personalized medicine. Please provide specific examples and cite relevant developments from the past five years.", 64),
    ("long→long",     "Write a detailed analysis of the following topics: 1) The impact of artificial intelligence on modern healthcare, including diagnostics, drug discovery, and patient care. 2) The ethical considerations surrounding AI in medicine, including bias, privacy, and accountability. 3) Future predictions for AI-assisted surgery and telemedicine. 4) The role of machine learning in genomics and personalized medicine. Please provide specific examples and cite relevant developments from the past five years.", 200),
]

REPS = 5  # per test case, total = 9*5 = 45 per engine (enough to be meaningful, fast enough to finish)

def bench_chew(tests, reps):
    """Benchmark chew via HTTP API."""
    import urllib.request
    results = []
    for name, prompt, max_tok in tests:
        times = []
        for _ in range(reps):
            body = json.dumps({
                "model": "llama",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tok,
                "temperature": 0.8,  # non-zero for varied output
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                "http://localhost:8080/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            elapsed = time.perf_counter() - t0

            prompt_tokens = data["usage"]["prompt_tokens"]
            completion_tokens = data["usage"]["completion_tokens"]
            total_tokens = prompt_tokens + completion_tokens
            times.append({
                "elapsed": elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })

        results.append({
            "name": name,
            "prompt_tokens": times[0]["prompt_tokens"],
            "target_completion": max_tok,
            "runs": times,
        })
    return results

def bench_llamacpp(tests, reps):
    """Benchmark llama.cpp via llama-cpp-python."""
    from llama_cpp import Llama
    llm = Llama(
        model_path=MODEL,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
        flash_attn=True,
    )

    results = []
    for name, prompt, max_tok in tests:
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            resp = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tok,
                temperature=0.8,
            )
            elapsed = time.perf_counter() - t0

            prompt_tokens = resp["usage"]["prompt_tokens"]
            completion_tokens = resp["usage"]["completion_tokens"]
            times.append({
                "elapsed": elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })

        results.append({
            "name": name,
            "prompt_tokens": times[0]["prompt_tokens"],
            "target_completion": max_tok,
            "runs": times,
        })
    return results

def summarize(results, engine_name):
    print(f"\n{'='*70}")
    print(f"  {engine_name}")
    print(f"{'='*70}")
    print(f"{'Test':<18} {'Prompt':>6} {'Compl':>6} {'Total':>8} {'Prefill':>10} {'Decode':>10} {'Total':>10}")
    print(f"{'':18} {'toks':>6} {'toks':>6} {'ms':>8} {'tok/s':>10} {'tok/s':>10} {'tok/s':>10}")
    print(f"{'-'*70}")

    all_decode_tps = []
    all_prefill_tps = []

    for r in results:
        elapsed_list = [run["elapsed"] for run in r["runs"]]
        comp_list = [run["completion_tokens"] for run in r["runs"]]
        prompt_toks = r["prompt_tokens"]

        # Median elapsed and completion tokens
        med_elapsed = np.median(elapsed_list)
        med_comp = int(np.median(comp_list))
        total_toks = prompt_toks + med_comp

        # Rough split: prefill ~ prompt_toks / (prompt_toks + comp_toks) * time
        # Better: estimate from short vs long output tests
        # For now: assume prefill is fast, most time is decode
        # decode tok/s ≈ completion_tokens / elapsed (lower bound)
        # total tok/s = total_tokens / elapsed
        decode_tps = med_comp / med_elapsed if med_elapsed > 0 else 0
        total_tps = total_toks / med_elapsed if med_elapsed > 0 else 0
        # prefill tok/s estimated from overhead
        prefill_tps = prompt_toks / (med_elapsed - med_comp / decode_tps) if decode_tps > 0 and med_elapsed > med_comp / decode_tps else 0

        all_decode_tps.append(decode_tps)
        if prefill_tps > 0 and prefill_tps < 100000:
            all_prefill_tps.append(prefill_tps)

        print(f"{r['name']:<18} {prompt_toks:>6} {med_comp:>6} {med_elapsed*1000:>8.0f} {prefill_tps:>10.1f} {decode_tps:>10.1f} {total_tps:>10.1f}")

    print(f"{'-'*70}")
    if all_decode_tps:
        print(f"{'Average decode tok/s:':<40} {np.mean(all_decode_tps):>10.1f}")
    if all_prefill_tps:
        print(f"{'Average prefill tok/s:':<40} {np.mean(all_prefill_tps):>10.1f}")

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("chew", "both"):
        print("Benchmarking chew...")
        chew_results = bench_chew(TESTS, REPS)
        summarize(chew_results, "CHEW (f16, custom CUDA)")
        with open("/tmp/bench_chew.json", "w") as f:
            json.dump(chew_results, f)

    if mode in ("llama", "both"):
        print("\nBenchmarking llama.cpp...")
        llama_results = bench_llamacpp(TESTS, REPS)
        summarize(llama_results, "LLAMA.CPP (GPU, flash_attn)")
        with open("/tmp/bench_llama.json", "w") as f:
            json.dump(llama_results, f)

    if mode == "both" or (os.path.exists("/tmp/bench_chew.json") and os.path.exists("/tmp/bench_llama.json")):
        if mode != "both":
            with open("/tmp/bench_chew.json") as f: chew_results = json.load(f)
            with open("/tmp/bench_llama.json") as f: llama_results = json.load(f)

        print(f"\n{'='*70}")
        print(f"  COMPARISON")
        print(f"{'='*70}")
        print(f"{'Test':<18} {'Chew ms':>10} {'Llama ms':>10} {'Ratio':>8} {'Winner':>8}")
        print(f"{'-'*70}")
        for cr, lr in zip(chew_results, llama_results):
            c_med = np.median([r["elapsed"] for r in cr["runs"]]) * 1000
            l_med = np.median([r["elapsed"] for r in lr["runs"]]) * 1000
            ratio = c_med / l_med if l_med > 0 else 0
            winner = "CHEW" if ratio < 1 else "LLAMA"
            print(f"{cr['name']:<18} {c_med:>10.0f} {l_med:>10.0f} {ratio:>7.2f}x  {winner:>6}")

if __name__ == "__main__":
    main()
