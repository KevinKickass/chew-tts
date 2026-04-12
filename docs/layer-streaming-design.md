# Layer Streaming: GGUF-Driven Residency and Honest Limits

Layer streaming in `chew` is driven by the model as it actually exists in GGUF, not by an idealized splitter.

## Core principle

`chew` does **not** assume models can be arbitrarily chopped into tiny interchangeable pieces.
It reads the tensor and stack layout from GGUF and streams at the granularity the artifact actually provides.

That means:

- fill VRAM with as much permanently resident data as possible
- keep fixed state resident first: KV cache, scratch, embeddings, norms, workspaces
- keep as many layers / model blocks resident as the remaining VRAM allows
- stream only the non-resident remainder
- keep compute on GPU; only weight storage/transport spills to host RAM

This is the design goal:

> Use VRAM as aggressively as possible, then degrade honestly into streaming.

## Why this matters

Many inference stacks behave like a black box:

- try to load
- allocate a lot
- maybe OOM
- maybe suggest random flags
- give poor visibility into what actually fit and what did not

`chew` takes the opposite approach:

- inspect GGUF metadata first
- compute a concrete VRAM plan
- decide what can remain resident
- stream only what really must move
- expose the resulting constraint to the operator

The point is not magical success on every model.
The point is to know **before and during runtime** what the system is doing.

## Streaming is constrained by GGUF reality

Streaming granularity comes from the structure encoded in GGUF:

- tensor names
- tensor sizes
- layer/block grouping
- architecture-specific stack layout
- model-specific large tensor boundaries

So the useful split unit is often "whatever the model file naturally gives us", not "whatever would be convenient in theory".

This matters a lot for larger models.

### Example: large chunked stacks

Some models expose very large natural chunks.
For example, a stack may be effectively split into ~3 GB sections.
In that case:

- you cannot finely shuffle tiny subpieces in and out
- only a small number of chunks may fit at once
- throughput is then dominated by the non-resident remainder
- this is a hardware/artifact constraint, not just a software policy

In other words:

> If only two chunks fit, then only two chunks fit. After that, physics takes over.

## Residency strategy

`chew` uses an adaptive residency strategy instead of a naive fixed ping-pong design.

### Resident first

VRAM is allocated in this order:

1. fixed runtime state
   - KV cache
   - forward scratch
   - dequant scratch
   - cuBLAS workspace
   - embeddings / norms / small always-hot tensors
2. permanently resident model data
   - as many layers / blocks as will fit
3. streaming buffers
   - DMA slots for the remaining non-resident layers / blocks

The result is not "stream the whole model".
It is:

- keep as much as possible local
- stream the minimum unavoidable tail

This is especially important on smaller cards, where partial residency can be the difference between:

- useful throughput
- and complete collapse

## Conservative first-pass, aggressive post-boot packing

`chew` intentionally starts with a conservative fit calculation.
It reserves headroom during the first planning step so the engine can boot reliably instead of gambling on a paper-perfect allocation.

That means the first pass is deliberately cautious:

- leave safety margin for runtime overhead
- avoid loading right up against the cliff edge
- prefer a successful start over a theoretical maximum residency count

After boot, `chew` can look at the memory that is actually still free and opportunistically densify residency:

- check real post-start free VRAM
- see whether another resident layer / block fits safely
- keep adding residents until the useful capacity is actually full

So the runtime behavior is:

1. conservative admission
2. successful startup
3. observed-memory recheck
4. pack VRAM with as many residents as reality allows

This is intentional.
It is better to come up safely and then fill the card than to chase a fragile pre-boot optimum and crash during initialization.

## Compute stays on GPU

Streaming mode is **not** CPU inference.

Host RAM is used as overflow storage for non-resident weights.
The math still happens on the GPU:

- weights move host -> device as needed
- dequant / kernels / GEMM run on GPU
- KV stays on GPU
- forward compute stays on GPU

So the fallback is:

- storage spills to RAM
- compute does not

That keeps the degraded mode much more honest and performant than a full CPU fallback.

## What the real limit becomes

For dense models that fit, `chew` can run close to the speed of highly tuned baselines.
For oversized or MoE models in streaming mode, the bottleneck often stops being raw compute and becomes:

- host->device transfer bandwidth
- chunk size / layer granularity
- overlap quality between DMA and compute
- how much of the model can remain resident

On constrained hardware, this can produce a hard throughput ceiling even when the GPU appears fully busy.
That does **not** automatically mean there is a bug.
It may simply mean the pipeline is already saturating the available transport + compute path.

## Operator-facing philosophy

`chew` should tell the truth plainly:

- what is resident
- what is streamed
- which granularity is imposed by the model artifact
- what the steady-state and peak VRAM costs are
- whether the likely bottleneck is VRAM fit, transfer bandwidth, or compute

The goal is not to promise that every model will be fast on every card.
The goal is to make the limits legible.

## Architectural implication

Different model families expose different realities:

- dense vs MoE
- per-layer embeddings vs standard embeddings
- shared-KV vs explicit V projections
- different stack layouts and split granularity

Because of that, streaming logic should remain grounded in model-specific architecture modules where needed, while sharing generic infrastructure for:

- VRAM planning
- residency decisions
- host buffer management
- DMA slots
- reporting

In short:

- generic infrastructure for the mechanism
- model-specific logic for the truth of each architecture

## Summary

Layer streaming in `chew` is built on four rules:

1. **Trust the GGUF artifact** rather than an imaginary ideal split
2. **Max out useful residency** before streaming anything
3. **Keep compute on GPU** even when storage spills to RAM
4. **Report the real limit honestly** when throughput is bounded by chunk size, PCIe, or residency

That makes streaming mode a controlled degradation path instead of an opaque failure mode.
