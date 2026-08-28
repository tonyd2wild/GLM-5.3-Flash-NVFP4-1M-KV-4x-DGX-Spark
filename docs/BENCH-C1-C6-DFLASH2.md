# DFlash2 concurrency sweep, C1–C6 (2026-08-28)

GLM-5.3-Flash-NVFP4 + `incoai/GLM-5.3-Flash-DFlash2`, TP2 on 2× DGX Spark (GB10),
fp8 KV, 262,144-token context, `--max-num-seqs 6`, `--kv-cache-memory 3221225472`
(310,292-token pool), thinking off. Harness: [`probes/bench_c1c6.py`](../probes/bench_c1c6.py)
— 2 waves per level, 400-token generations, code/reasoning prompts, unique salt per
request to defeat prefix caching, temp 1.0 / top-p 0.95. Numbers are end-to-end
(prefill + decode) wall-clock, i.e. what a user actually sees.

| concurrency | aggregate tok/s | per-stream tok/s | mean wall s | failures | accepted ÷ drafted |
|---|---|---|---|---|---|
| **C1** | 35.1 | 35.1 | 11.4 | 0 | 0.525 |
| **C2** | 41.6 | 23.2 | 19.2 | 0 | 0.447 |
| **C3** | 40.6 | 17.3 | 29.1 | 0 | 0.422 |
| **C4** | 47.5 | 15.3 | 33.7 | 0 | 0.401 |
| **C5** | **56.2** | 17.5 | 35.6 | 0 | 0.510 |
| **C6** | 47.7 | 13.3 | 50.3 | 0 | 0.398 |

Best single-stream measured separately on a warm engine with a code prompt:
**46.9 tok/s at 74.1 % acceptance** (420 of 567 drafted tokens accepted). The C1
row above is lower because this sweep's mixed prompt set includes prose-shaped
work, where acceptance drops.

## Reading the numbers

- **Single-stream is the win.** 46.9 tok/s warm vs the MTP-4 flagship's 21.8 on
  identical hardware and context — **2.15×**. Speculative decoding buys latency,
  and DFlash2's 7-token block plus high acceptance converts directly into felt
  speed.
- **Acceptance tracks prompt type, not load.** Code and structured reasoning sit
  at 0.5–0.74; prose pulls the mixed-set average toward 0.40. The C5 bump (0.510)
  is prompt-mix luck in that wave, not a concurrency effect.
- **Aggregate throughput peaks mid-range (C5, 56.2 tok/s)** and dips at C6.
  Speculation trades compute for latency: every verification step also validates
  7 draft tokens, so once the batch is compute-saturated the extra draft work stops
  paying for itself. If you are optimizing pure aggregate throughput at high
  concurrency rather than time-to-first-token and stream smoothness, run without a
  drafter.
- **Zero failures at every level.** An earlier sweep at `--kv-cache-memory
  4445787956` died at C3 when three concurrent 20K-token prefills drove
  MemAvailable to 3.06 GB and the `dgx-anti-oom` watchdog (3 GB threshold)
  terminated the engine. The shipping pin trades ~118K pool tokens for headroom
  that survives concurrent load — on GB10 the binding constraint is free-memory
  headroom, not the pool.

## Cold-start caveat

The first requests after a boot JIT-compile `_prepare_dflash_inputs_kernel`,
`_topk_topp_kernel` and `mhc_pre_big_fuse_with_norm_tilelang`. A C1 measurement
taken cold reads ~10 tok/s low (36.9 vs 46.9 observed on the same config). Warm
the engine before benchmarking.
