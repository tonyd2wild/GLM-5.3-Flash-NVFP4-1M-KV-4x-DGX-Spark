# GLM-5.3-Flash NVFP4 · 1M-Token KV · 4x DGX Spark · 36 tok/s

[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B / A18B MoE, released 2026-08-26) serving across **all four NVIDIA DGX Spark (GB10) nodes** at tensor-parallel 4, using the [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) quant — deployed the same day the model dropped.

**As far as we can tell: the first TP4 `glm5_next` deployment outside NVIDIA B200 hardware, the first fp8 KV cache for a NoPE-MLA model on any consumer Blackwell part, and a 1.26-million-token KV pool on $16K of desk hardware.**

## Two KV-cache lanes + censored/uncensored (added 2026-08-27)

This TP4 deployment now ships as a **2×2** — pick your KV-cache format and your weights:

|  | **Lane A — fp8 KV** (our FlashInfer SM12x unlock) | **Lane B — NVFP4 KV** (b12x path, credit [keys/drowzeys](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated)) |
|---|---|---|
| **KV size** | ~656 B/token/layer | **368 B/token/layer** (~half) |
| **Censored** ([LibertAIDAI](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)) | ✅ TP4 flagship — 5.03M-token pool | compatible (not separately benched) |
| **Uncensored** ([keys ablit](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock)) | ✅ drop-in — same launcher, verified | ✅ **TP4: 6,652,112-token pool = 6.34× a full 1M context** (32 GiB/rank) |

### Lane B — NVFP4 KV at TP4 (verified serving, 2026-08-27)

We took the **NVFP4 KV** recipe from [drowzeys/keys](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated) — the **Zero-RoPE shim** (pad GLM's NoPE attention with a virtual 64-dim RoPE so the cache presents DeepSeek's 576-dim record to the NVFP4 kernels, bit-for-bit identical to NoPE) + Luke Alonso's **b12x** `B12X_MLA_SPARSE` backend + `KV_DTYPE=nvfp4_ds_mla` — and **extended it from his 2-Spark TP2 to our 4-Spark TP4** on the uncensored `keys` ablit weights.

| Metric | NVFP4-KV TP4 (uncensored) |
|---|---|
| **KV pool** | **6,652,112 tokens = 6.34× a full 1,048,576-token context** (≈6.3 full-1M conversations at once), at 32 GiB KV/rank |
| KV dtype | `nvfp4_ds_mla`, `KV_FP8_ROPE=1` — **368 B/token/layer** (vs 656 fp8, 1152 bf16) |
| Context | 1,048,576 (1M), served on `:8000` as `glm-5.3-flash` (drop-in name) |
| Decode | **~36 tok/s single-stream** warmed (temp 0, structured) — `--enforce-eager` is required by the b12x kernels and caps single-stream; concurrent aggregate is far higher. Prefill ~1,050–1,290 tok/s, TTFT ~0.2 s |
| Vision | ON (image input verified) · Thinking off by default · uncensored (no refusals) |
| Spec decode | native MTP head, k=2 · GMU 0.85 hard cap · 32 GiB KV/rank |

**fp8 vs NVFP4 KV at EQUAL 32 GiB/rank budget:** NVFP4 KV = **6,652,112 tokens** vs fp8 = 5,033,164 — **1.32× the pool at the same memory** (the 368-vs-656 B/token density showing through). As far as we can tell this is the **first NVFP4 KV cache at TP4 on consumer Blackwell**, and ~5.4× the reference 2-Spark TP2 pool (1.22M tokens). **Full credit to [drowzeys / keys](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated)** for the Zero-RoPE shim, the b12x NVFP4 kernels, and the ablit weights; our contribution is the TP4 port + the 4-node fabric/memory config.

### fp8 vs NVFP4 KV — speed head-to-head (measured 2026-08-27)

The density win above is **not free** — we ran both lanes back-to-back on the same `:8000` endpoint (same uncensored `keys` ablit weights, TP4, `--enforce-eager`, MTP, temp 0, warmed) to price it:

| Metric | Lane A — fp8 KV | Lane B — NVFP4 KV | Winner |
|---|---:|---:|:--|
| **Decode** (structured/agentic, warmed) | **~55 tok/s** (51 / 56 / 55) | **~37 tok/s** (37 / 37) | **fp8 ~1.5×** |
| Prefill (warmed, ~9K-token prompt) | **~3,530 tok/s · 2.55 s TTFT** (3 runs) | ~1,449 tok/s · 6.9 s TTFT¹ | fp8¹ |
| KV pool @ 32 GiB/rank | 5,033,164 | **6,652,112** | **NVFP4 1.32×** |
| KV density | 656 B/token/layer | **368 B/token/layer** | **NVFP4 1.8×** |

¹ The NVFP4 prefill/TTFT is a **single sample that may have been partly cold** (our fp8 first-prefill was 19 s / 467 tok/s cold, then settled to ~2.5 s / ~3,530 warmed — the b12x kernels JIT on the first large prefill too). We did not capture a clean warmed long-prompt NVFP4 prefill before teardown, so **treat decode as the definitive head-to-head and the prefill row as directional, not final.**

**What this means:** NVFP4 KV's ~33 % slower decode is the b12x `B12X_MLA_SPARSE` sparse-attention path (+ the per-token NVFP4 dequant) doing more compute per step than fp8's marlin path — and `--enforce-eager`, which the b12x kernels require, caps single-stream on both lanes. So the trade is clean: **NVFP4 = KV *capacity* (bigger context pool at equal VRAM), fp8 = *speed* (faster tokens for the same agent work).** For our production endpoint we run **fp8 as the daily driver** (faster, uncensored, vision, simplest to operate) and keep **NVFP4 as the flex** for the rare job that needs the giant pool over throughput.

## Numbers

| Metric | TP4 flagship |
|---|---|
| Decode | **35.7 tok/s** generic median · **up to 63.8 tok/s** warmed on structured/agentic output (MTP acceptance runs hot — [re-bench below](#warmed-streaming-re-bench--the-357-is-a-floor-not-the-ceiling-2026-08-27)) |
| TTFT | **0.204 s median** |
| Context | **1,048,576 (model-native 1M) — launcher default** · the 1.26M-token KV pool physically holds a full 1M-token request. Cap --max-model-len lower (e.g. 300000) for a snappier multi-user endpoint |
| KV pool | **5,033,164 tokens fp8** — 4.82 concurrent full-context requests (or one ~1M-token context) |
| Speculative decode | native MTP head, 4 draft tokens |
| KV dtype | fp8_e4m3 (our FlashInfer SM12x unlock — see below) |
| Boot | ~12 min (quarter weights per rank) |

Progression on the same hardware pair count: 14.3 tok/s (day-1 bf16 TP2) → 21.8 (fp8+MTP TP2) → **35.7 (TP4)**.

### Warmed streaming re-bench — the 35.7 is a floor, not the ceiling (2026-08-27)

The 35.7 median above is a **generic 200-token greedy** number. Re-benched **warmed + streaming** (decode = `(completion_tokens − 1) / (t_last − t_first)`, measured off-box over the tailnet, temp 0), throughput is strongly **content-regime dependent** because the MTP head's draft-acceptance rate swings with how predictable the output is:

| Prompt | Decode (warmed) |
|---|---:|
| 🔢 **count 1→100** (structured) | **63.8 tok/s peak · ~61 median** (6 runs) |
| 🔤 alphabet ×8 | ~60 |
| 💻 code continuation | ~53 |
| 📝 freeform 400-word essay | ~37 |

So the honest picture: **freeform prose ≈ 37 (that's where the "36" comes from), but structured / list / code / tool-argument output — what agents actually generate — runs 53–64 tok/s** as MTP acceptance approaches 100%. Real agentic workloads live in the high-acceptance zone, so **~55–64 tok/s is the number that matters in production**, roughly 1.7× the headline. TTFT stays ~0.2 s across all regimes.

## What's in here

- [`launch-glm53-vllm-tp4.sh`](launch-glm53-vllm-tp4.sh) — the 4-rank launcher (head serves `:8000`; run worker-first: rank 3 → 2 → 1, head 0 last). Full NCCL fabric env included.
- [`cache_flusher.sh`](cache_flusher.sh) — **required sidecar** on every node during boot. GB10's driver fails allocations against page-cache-full memory; this holds the cache down through the 182 GiB shard read. Mechanism + measurements: [docs/GB10-KV-MEMORY-LADDER.md](docs/GB10-KV-MEMORY-LADDER.md).
- [`docker/`](docker/) — the **sm121-v8 image patch stack** (8 Dockerfiles, applied v1→v8 on the day-0 `vllm/vllm-openai:glm53-flash-arm64-cu130`). The vendor image dies five different ways on GB10; these fix: the NoPE-MLA backend gap, a FlashInfer FA2 NaN kernel bug, two dependency downgrades the FlashInfer nightly sneaks in (NCCL, cutlass-dsl), a PDL race surface, uninitialized indexer top-k memory, and the fp8-KV shared-memory tile bug.
- [`docs/DEPLOY-REPORT.md`](docs/DEPLOY-REPORT.md) — every failure, root cause, and receipt from the deploy day (eight kernel-level bugs).
- [`docs/`](docs/) issue drafts — upstream-ready reports for the FlashInfer fp8-MLA SM12x gap and the vLLM NoPE `fp8_ds_mla` layout gap.
- [`probes/`](probes/) — the debugging kit: probe a FlashInfer kernel with your model's real geometry before trusting arch-gate patches, a NaN bisect harness, kernel-vs-torch A/B, and the benchmark script.

## Quickstart (4 nodes)

One node owns the weights on local NVMe and NFS-exports them; the other three mount at the same path.

```bash
# every node: image + weights visible + flusher
docker load < sm121-v8.tar          # or build docker/Dockerfile.glm53-sm121* v1->v8 in order
ls /var/tmp/glm-5.3-flash-nvfp4/config.json
nohup ./cache_flusher.sh > flusher.log 2>&1 &
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

# workers first, ~20 s apart, head LAST
./launch-glm53-vllm-tp4.sh 3   # worker
./launch-glm53-vllm-tp4.sh 2   # worker
./launch-glm53-vllm-tp4.sh 1   # worker
./launch-glm53-vllm-tp4.sh 0   # head — serves http://<head>:8000/v1
```

Edit the rank→IP map at the top of the launcher for your fabric. Thinking is off by default (`--default-chat-template-kwargs`); re-enable per request with `chat_template_kwargs: {"enable_thinking": true}`. Tool calling ships enabled (`glm47` parser).

## Fast loading: InstantTensor (added 2026-08-27)

**Status: experimental — 15x load speedup measured, but NOT stable in our multi-node TP2 topology** (a rank dies silently ~1 min post-load in every v9 boot, at any KV size, including budgets that are 100% stable on v8; cf. eugr/spark-vllm-docker#29 for the same multi-node class of problem). The shipped launchers do NOT enable it; the stable image remains v8. The v9 image adds the InstantTensor direct-I/O loader (`--load-format instanttensor`): loads drop from ~10 minutes to 40-100 seconds. Two things to know: its pip install silently downgrades NCCL to a fabric-fatal version (v9 re-pins 2.30.7 in the same layer), and because direct I/O never fills the page cache, it also defeats the first layer of the GB10 KV-allocation wall -- the full story and the remaining (unsolved) second wall are in [docs/GB10-KV-MEMORY-LADDER.md](docs/GB10-KV-MEMORY-LADDER.md). Credit: jack6464 (NVIDIA forum) for the pointer.

## Hard-won rules (each one cost us a boot)

1. Tear down **all** ranks before relaunching **any** — a fresh rank that rendezvouses with a dying one hangs.
2. Verify `grep '^IMAGE'` matches on every node before every launch; copy launcher files whole, never sed over ssh.
3. Run the cache flusher during every boot; on TP2-class per-rank weights, take vLLM's suggested `--kv-cache-memory` verbatim (the ladder study shows why bigger dies).
4. Reboot a node that has been through many boot cycles — GB10's driver accumulates allocation-pool degradation that eventually kills even proven configs.
5. Capture `docker logs` before `docker rm -f`.

## Why TP4 (beyond speed)

At TP2 each rank carries ~97 GiB of weights and the GB10 driver can only reliably grant ~4.5 GiB of KV afterward (measured across six controlled boots — see the ladder study). At TP4, weights drop to ~50 GiB per rank and the KV ceiling simply dissolves: the 9 GiB slab allocates with ~60 GiB of slack, giving every one of the six request slots full 262K context simultaneously.

## vLLM v0.28.0 status (checked 2026-08-27)

**Not viable for GLM-5.3 yet**: the `glm5_next` architecture is NOT in the v0.28.0 release
(PR vllm-project/vllm#53906 still open/unmerged at check time), and no rebased day-0 image
exists (all `vllm/vllm-openai:glm53-flash*` tags still date to the original 2026-08-26 push).
The day-0 image used here is itself a main-branch dev snapshot (`0.1.dev20051`) cut around
the 0.28 branch point -- i.e. this stack already runs 0.28-era engine code plus the GLM
support 0.28 lacks. Upgrade path when it opens: watch the PR and the Docker Hub tags; the
patch stack here is guarded string-patches that apply-or-refuse loudly, so porting to a new
base is mechanical (apply v1->v10 in order, fix whichever guards fire, ladder through the
experiment lane before production).

## Credits

Model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · Quant: [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) · barrydeen (gmu reference + quant table) · vLLM [PR #53906](https://github.com/vllm-project/vllm/pull/53906) authors for the day-0 image · FlashInfer 0.6.18 · Deployed and debugged by Knox (Claude) for [@tonyd2wild](https://github.com/tonyd2wild). Companion deep-dive repo: [GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark).

## 5M-token KV pool at 1M context (2026-08-27, stress-gated)

The shipped config is now **16 GiB KV per rank = 2,516,582 fp8 tokens** (see docs/SM121-CRASH-FORENSICS-2026-08-27.md for why bigger pools fail) at `--max-model-len 1048576` — 3.6 concurrent full-1M-context requests. Found via the **residual-headroom rule**: grow the KV slab until only ~8-10 GB stays available per node (nodes idle at ~37-42 GB available on the old 9 GiB config).

**The 38 GiB cautionary tale:** 38 GiB/rank (5,975,779 tokens) allocates cleanly, boots, and answers short prompts — then the first 20K-token prefill NVRM-OOMs a rank and the engine dies. On GB10, "serving" is not the bar; **gate every KV bump behind a real long prefill** with the engine verified alive afterward. 32 GiB passed a single-prefill gate, then died under three overlapping real-traffic requests (head rank carries API server + NFS duty). The bar is CONCURRENT prefills: 24 GiB survives 3x simultaneous 20K prefills with ~18 GB residual on the head.

Also required for vision requests: `--chat-template chat_template_mm.jinja` (the checkpoint ships a text-only template; image requests 500 without the mm variant).
