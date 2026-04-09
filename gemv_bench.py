"""Quick standalone GEMV benchmark via the server."""
import urllib.request, json, time, sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

def gen(p, mt):
    d = json.dumps({"messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0.0}).encode()
    r = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",data=d,headers={"Content-Type":"application/json"})
    t0=time.perf_counter()
    resp=urllib.request.urlopen(r,timeout=300)
    elapsed=time.perf_counter()-t0
    rr=json.loads(resp.read())
    return rr["usage"]["completion_tokens"], elapsed

# Warmup
gen("Hi", 5)

# Benchmark
results = []
for i in range(5):
    toks, t = gen("Explain quantum computing.", 200)
    tps = toks / t
    results.append(tps)
    print(f"Run {i+1}: {tps:.1f} tok/s")

avg = sum(results) / len(results)
best = max(results)
print(f"\nAvg: {avg:.1f} tok/s | Best: {best:.1f} tok/s")
