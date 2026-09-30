# GLM-5.3-Flash · DFlash2 · TP4 · 1M Context · 3.9M-Token KV

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="runs/2026-09-29-knapcio-stack/charts/suite-dark.svg">
  <img alt="Single-stream decode on the 500K lane: count to 100 181.7, count to 300 165.7, tool call 174.0, math 141.8, code 120.5, json 120.5, sql 109.1, summary 70.6, prose 61.0, narrative 60.8 tok/s." src="runs/2026-09-29-knapcio-stack/charts/suite-light.svg" width="880">
</picture>

**Single-stream decode, tok/s** (median of 3; mean accepted draft length in brackets). Count to 100 is the peak.

| prompt | 500K lane | 262K lane |
|---|---|---|
| count to 100 ¹ | **181.7** (7.26) | 183.2 (7.42) |
| count to 300 ¹ | **165.7** (6.94) | 166.0 (6.91) |
| tool call ² | **174.0** (6.80) | 173.1 (6.80) |
| math | **141.8** (4.55) | 144.1 (4.54) |
| code | **120.5** (5.08) | 120.3 (5.15) |
| json | **120.5** (4.93) | 121.2 (4.91) |
| sql | **109.1** (4.23) | 107.4 (4.25) |
| summary | **70.6** (2.38) | 74.0 (2.48) |
| prose | **61.0** (2.05) | 62.3 (2.10) |
| narrative | **60.8** (2.03) | 59.0 (1.98) |

¹ The counting prompts are the draft-acceptance ceiling, not a typical rate. ² 34 output tokens.

> 🔀 **Only have two Sparks?** The same images run at TP2 (262K context) — see the sibling repo:
> **[GLM-5.3-Flash NVFP4 + DFlash2 · 2x DGX Spark →](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)**

[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B / A18B MoE)
serving across **all four NVIDIA DGX Spark (GB10) nodes** at tensor-parallel 4, with the
[`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
block-diffusion drafter.

**Current default (2026-09-29): [knapcio's stack](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4), run
unmodified on this fleet, with a 500,000-token window** on a 3,453,703-token KV pool, and a 262K lane beside it. On the
same hardware, harness and day as our previous recipe: prose decode +36% to +57% at every concurrency, cold prefill +28%
to +37%, repeated-prompt TTFT about 3× faster. The stack is knapcio's; the lane configs and measurements are ours. The
uncensored Blackfrost lane stays on the previous recipe for now, which is documented below unchanged.

---

## ⭐ Current default (2026-09-29): knapcio's stack, 500K context

The speed stack is **[knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)**
by [@knapcio](https://github.com/knapcio) (MIT). We run it **unmodified**, pinned at commit `770d115`, on our four
Sparks. Two things are ours: the lane configs (500K default, 262K) and the measurements below. His stack is built on
this repo's v11 image, its RoCE all-reduce port and its DFlash2 prefix-cache repair, and it credits them. The 8-bit
dense layers that give it its largest single win start from this repo's finding that the nvidia checkpoint leaves
18 GiB of non-expert weights in BF16. What he added is the rest: 8-bit dense kernels, certified LM head, draft-length
truncation chosen on the GPU, FP8 drafter, prefill kernels, and about twenty more measured pieces.

**Runbook, both lane configs, harness and raw results: [`runs/2026-09-29-knapcio-stack/`](runs/2026-09-29-knapcio-stack/).**

| lane | context | KV pool | full-length requests that fit | config |
|---|---|---|---|---|
| **500K (default)** | 500,000 | **3,453,703 tokens** | 6.9 | `env.500k` |
| 262K | 262,144 | 2,945,172 tokens | 11.2 | `env.262k` (knapcio's own context) |

Both lanes serve `glm-5.3-flash` on `:8000`, the name and port this repo's earlier recipes used, so clients do
not change. Weights: `nvidia/GLM-5.3-Flash-NVFP4` through his CPU conversion (50 s per node; config hashes match
his published ones). Warm reboot: 3 to 4 minutes (the previous recipe: about 12).

### Previous recipe vs knapcio's stack (same fleet, same day, same harness)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="runs/2026-09-29-knapcio-stack/charts/decode-dark.svg">
  <img alt="Decode throughput by concurrency. Prose: previous recipe 44/60/77/115/158, knapcio stack 63/89/121/163/215 tok/s at c1/c2/c4/c8/c16. Code: 102/119/164/217/315 vs 117/134/182/232/287. JSON: 110/136/170/242/338 vs 130/161/223/278/336." src="runs/2026-09-29-knapcio-stack/charts/decode-light.svg" width="880">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="runs/2026-09-29-knapcio-stack/charts/prefill-dark.svg">
  <img alt="Cold prefill: previous recipe 2,035 / 2,039 / 2,028 tok/s at 8K / 32K / 64K prompts, knapcio stack 2,607 / 2,770 / 2,786 (+28% / +36% / +37%)." src="runs/2026-09-29-knapcio-stack/charts/prefill-light.svg" width="880">
</picture>

| | previous recipe | knapcio stack | change |
|---|---|---|---|
| prose, 1 → 16 streams | 44 → 158 tok/s | 63 → 215 tok/s | **+36% to +57%** at every level |
| code, 1 → 8 streams | 102 → 217 | 117 → 232 | +7% to +15% |
| code, 16 streams | 315 | 287 | **−9%** |
| json, 1 → 8 streams | 110 → 242 | 130 → 278 | +15% to +31% |
| json, 16 streams | 338 | 336 | even |
| cold prefill, 8K / 32K / 64K | 2,035 / 2,039 / 2,028 tok/s | 2,607 / 2,770 / 2,786 | **+28% / +36% / +37%** |
| repeated-prompt TTFT (prefix cache) | 1.8 to 2.0 s | 0.5 to 0.7 s | about 3× faster |

Harness: knapcio's own `bench/conc_bench.py` (32 distinct prompts per type, 768 tokens, temperature 0,
`reasoning_effort: low`) and `bench/prefill_bench.py` (salted prompts), best of two runs per cell, both stacks warm,
GPU clock cap 2200 MHz. The one regression is code at 16 streams: his scheduler caps draft depth at 3 for batches of
3 to 27. Deeper tables and a `batch-max` scheduler did not move it (runbook §7).

### Full speed-night suite on the lanes

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="runs/2026-09-29-knapcio-stack/charts/sweep-dark.svg">
  <img alt="Aggregate throughput with mixed real prompts on the 500K lane, peak of 3 rounds: C1 116.5, C2 152.6, C3 155.8, C4 162.1, C5 176.1, C6 146.5 tok/s." src="runs/2026-09-29-knapcio-stack/charts/sweep-light.svg" width="880">
</picture>

**C1 to C6, mixed real prompts** (8 types rotated across streams, no counting; aggregate tok/s, median / peak of 3 rounds)

| streams | 500K lane | 262K lane | per-stream decode (500K) | TTFT p90 (500K) |
|---|---|---|---|---|
| C1 | 115.4 / **116.5** | 115.2 / 117.1 | 119.8 | 0.26 s |
| C2 | 103.4 / **152.6** | 98.6 / 151.2 | 90.8 | 2.03 s |
| C3 | 104.5 / **155.8** | 105.5 / 160.9 | 59.5 | 2.43 s |
| C4 | 159.9 / **162.1** | 162.3 / 163.4 | 56.5 | 0.63 s |
| C5 | 134.5 / **176.1** | 111.3 / 180.2 | 42.2 | 2.96 s |
| C6 | 141.2 / **146.5** | 108.8 / 125.0 | 36.1 | 3.46 s |

Median and peak differ because each round sends a different mix of prompt types; zero failures and zero preemptions in every cell.

**Cold prefill** (salted, 1 output token) **and long context**

| | 500K lane | 262K lane |
|---|---|---|
| 5,943-token prompt | 2,378 tok/s (TTFT 2.5 s) | 2,404 tok/s (TTFT 2.5 s) |
| 29,867-token prompt | 2,798 tok/s (TTFT 10.7 s) | 2,811 tok/s (TTFT 10.6 s) |
| 113,911-token prompt | 2,794 tok/s (TTFT 40.8 s) | 2,782 tok/s (TTFT 41.0 s) |
| **498,636-token prompt** | **2,462 tok/s (TTFT 203 s)** | over the 262K limit |
| needle test (3 codes at 10/50/90% depth) | **3/3 at 480,203 tokens** (TTFT 225 s) | 3/3 at 243,917 tokens (TTFT 92 s) |
| 114K-token prompts, 1 at once: decode per stream | 71.8 tok/s | 54.4 tok/s |
| 114K-token prompts, 2 at once: decode per stream | 49.4 tok/s | 25.2 tok/s |

Long-context rows are single runs, so the gap between the lanes there is not a lane effect; the two lanes run the
same code and differ only in the context limit and pool size. Lowest free memory on any node during the 480K needle
prompt: 18 GiB.

Harness: this repo's speed-night suite ([`bench/bench_tp2_night.py`](runs/2026-09-29-knapcio-stack/bench/bench_tp2_night.py)), temperature 0,
`reasoning_effort: low`, GPU clock cap 2200 MHz, 2026-09-29. Raw JSON: [`results/`](runs/2026-09-29-knapcio-stack/results/).

### What to know before you switch

- **Censored checkpoint.** This stack runs `nvidia/GLM-5.3-Flash-NVFP4`. The uncensored Blackfrost lane stays on
  the previous recipe (below) until his 8-bit conversion is tested on those weights.
- **Thinking cannot be switched off** in this chat template; it takes `reasoning_effort` low, high or max. The fleet
  and both lane configs default to **low** (next section: same eval score, faster answers). The previous recipe had
  thinking off.
- **500K is our setting, not his.** He validates at 262K. Past that we have one needle test per length (below)
  and a 498,636-token cold prefill, not a long-context quality evaluation.
- **Single RoCE rail.** He runs two; our second port is addressed on two of four nodes. Prefill lands at the low
  end of what his published gains predict, consistent with that.
- **Keep every draft depth 1 to 7 in the spec table**, or device-side selection silently fails and single-stream
  speed drops by half (runbook §6).


### Reasoning effort: the fleet serves at low (2026-09-29)

*From the session that relaunched the fleet at `DEFAULT_EFFORT=low`; raw data in [`runs/2026-09-29-knapcio-effort-low/`](runs/2026-09-29-knapcio-effort-low/).*

**Effort, not thinking.** The GLM-5.3 chat template in that stack takes `reasoning_effort` `low`, `high` or `max`
(anything else becomes `max`). There is no thinking-off switch, and `enable_thinking: false` is ignored. We relaunched
with `DEFAULT_EFFORT=low` (a request can still ask for `high` or `max` with `chat_template_kwargs`).

**Tool-calling eval, 69 scenarios, same endpoint, same day** (our harness):

| Server default effort | Quality | Pass / partial / fail | Median turn | Tokens used | Decode |
|---|---:|---:|---:|---:|---:|
| high | 92.8 | 63 / 2 / 4 | 771 ms | 63.3K | 70.1 tok/s |
| **low** | **93.5** | 63 / 3 / 3 | **632 ms** | 60.4K | 69.5 tok/s |

**Answer time, low vs high on the same boot** (5 prompts: margin math, code, a route, a summary, one-line arithmetic;
request-level `reasoning_effort`, temperature 0):

| Effort | Median per answer | Wall time, all 5 | Output tokens | Thinking text |
|---|---:|---:|---:|---:|
| **low** | **1.3 s** | **7.9 s** | 631 | 227 chars |
| high | 1.8 s | 11.1 s | 978 | 602 chars |

Decode speed is about the same at either effort; low finishes sooner because it thinks less before it answers. The
largest gap was the route question, 1.4 s vs 3.7 s. Low scored the same or better on the eval, so it is our default.

**Relaunching knapcio's stack with a new setting.** `./start.sh stop` keeps the containers, and `serve` refuses a
container name or an overlay path that overlaps any existing container mount on any node. A new setting therefore
needs a fresh `CTN` and a fresh `OVERLAY_REMOTE` (ours: `glm53kl`, `/srv/glm53-knapcio-low`; `/var/tmp` was out
because an old container mounts all of it). Copying the old overlay's `cache/` first (as root; some of it is
root-owned) gave a warm boot: `/health` 200 in about 200 s.

Raw logs, the A/B script and our env file: [`runs/2026-09-29-knapcio-effort-low/`](runs/2026-09-29-knapcio-effort-low/).

---

## Previous default (2026-09-20 to 09-29): `nvidia/GLM-5.3-Flash-NVFP4` plus NVFP4 attention

The default lane starts from [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
(MIT, ungated) and applies our own NVFP4 quantization to the non-expert projections, 403 tensors that the
upstream pack leaves in BF16. Built weights directory: **`nvidia-glm53-attn`**. This repo hosts **no weights**,
only the recipe, the patches and per-file checksums.

Boot it (head serves `http://<head>:8000/v1`, served model name `glm-5.3-flash`):

```bash
MODEL_DIR=nvidia-glm53-attn NVFP4_PATCH=1 MNBT=8192 SPEC_K=7 MAXLEN=500000 \
  bash /root/glm_boot.sh <label>
```

**`NVFP4_PATCH=1` is required, not a tuning flag.** It bind-mounts two patched
`vllm/models/glm5next/nvidia` files (`kda.py`, `model.py`) that stop `quant_config` being forced to `None` on
the attention projections. Without them vLLM builds those layers BF16, a packed 4-bit weight has nowhere to
load, and the build does not come up at all. Both files, and the image digest they were cut from, are in
[`runs/2026-09-20-nvidia-lanes/patches/`](runs/2026-09-20-nvidia-lanes/patches/).

Note the flags: this lane was measured at `MNBT=8192` and a 500,000-token window. The
`--max-num-batched-tokens 16384` and `--max-num-seqs 64` findings further down were measured on the
1M-context fp8/marlin lane and have **not** been re-measured here, so they are that lane's settings rather
than a recommendation for this one.

Measured on this fleet at 500K context, fp8 KV, temperature 0, medians of three passes after a discarded
warm-up:

| | |
|---|---|
| KV pool | **3,532,196 fp8 tokens** |
| Weights | **43.76 GiB/rank** |
| Boot | ~12 min to `/health` 200 |
| Draft acceptance | 0.394 at DFlash2 `k=7` |
| Quality gate | **PASS** |
| Cold prefill | **1,997 tok/s** (40,659 tokens, TTFT 20.4 s) on an idle lane |
| Token corruption | **not seen on this lane or the Blackfrost lane** in operator use; it tracked the retired `o_proj` transplant. Mechanism still unmeasured - see the appendix |

Against the previous LibertAI-based build, on the same harness and prompts: **7 of 9 C1 categories within
measurement noise**, aggregate level at C1-C2 and ahead at C3-C6, and concurrency ahead at every level from
C12 (**+4.8% C12, +6.9% C16, +7.6% C24, +10.0% C32**). The two C1 outliers sit just past their own spreads in
opposite directions. Full tables, spreads and the discarded-data notes:
[`runs/2026-09-20-nvidia-lanes/`](runs/2026-09-20-nvidia-lanes/).

### The uncensored lane, still on this recipe: `blackfrost-glm53-derisked-attn` (2026-09-21)

**Uncensored default.** Start from
[`Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4`](https://huggingface.co/Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4)
(abliterated from `zai-org/GLM-5.3-Flash-BF16`, 1.3% refusal on their eval) and apply **the identical recipe**
as the default lane: our NVFP4 attention quantization, the proven 14-entry ignore list, `W4A16_NVFP4`. Swap
`MODEL_DIR=blackfrost-glm53-derisked-attn` into the command above. Same 43.76 GiB/rank, same 3,532,196-token
pool, same knobs — the weights are the only difference, which is what makes the comparison single-variable.

Uncensored behaviour and long-session coherence are both **operator-confirmed**, in the same harness that
garbled on the retired transplant lane. It measures at or above the censored lane on almost every cell:
code 95.9 vs 78.7 (medians; Lane A's code cell is bimodal at 78.7/92.0/78.7, so the fair statement is that
**every Blackfrost rep beats every Lane A rep**, spread 1.01x vs 1.17x), counting 138.1 vs 107.7, prose 50.8 vs
40.8, C3 aggregate 125.0 vs 117.0, and **acceptance 0.415 / 3.90 tok/step against 0.394 / 3.76**. Math is the one category below (83.3 vs 88.8, about −6%, at the
edge of its spread). Full tables, spreads and caveats:
[`runs/2026-09-21-blackfrost-derisked/`](runs/2026-09-21-blackfrost-derisked/).

⚠️ **Fix its config or it will garble.** Blackfrost's `config.json` declares `quant_algo: NVFP4` — the W4A4
path — while shipping **zero `input_scale` tensors**, so W4A4 kernels would multiply real weight scales by
uninitialized memory. Its own `config_groups` says `input_activations: null`, so weight-only is the intent and
the string is mislabeled. `fix_ignore.py` rewrites it to `W4A16_NVFP4`, and the build script does this for you.

### Retired: the dealignai `o_proj` transplant (`nvidia-glm53-ablit-attn`)

**Do not use this lane.** It substituted `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4`'s `o_proj` tensors on
layers 12-44 (33 tensors) before quantization. It did stop refusing, but it **garbled in live agentic use** in
two unrelated harnesses — emoji runs, injected foreign-script fragments, duplicated paragraphs and a degenerate
tail — and its draft acceptance had collapsed to **0.224 / 2.57 tok/step with position 7 at exactly zero**.

The reason is arithmetic. Refusal in this model lives in the routed-expert `down_proj` tensors: our attribution
measured `down_proj` moving refusal by **0.81** against **0.03** for attention plus shared/dense MLP. That pack
edits **46** `o_proj` tensors; the real intervention is **12,384** `down_proj` tensors. It bought non-refusal by
perturbing attention, and that is also what made it incoherent over long contexts — one cause, two symptoms.
Blackfrost edits `down_proj` (verified: ~10% relative difference on every sampled tensor, against a
quantization-noise floor of 0.000 on untouched classes) and does not have the problem.

### Read these two before deploying

- **[`runs/2026-09-20-nvidia-lanes/`](runs/2026-09-20-nvidia-lanes/)**: this build. Both lanes measured
  side by side, the checksums, the two porting traps, and
  **[`RUNBOOK.md`](runs/2026-09-20-nvidia-lanes/RUNBOOK.md)** with every command and value used, in order.
- **[`runs/2026-09-20-tp4-vs-deepseek/`](runs/2026-09-20-tp4-vs-deepseek/)**: where the NVFP4 attention work
  came from, measured head to head against DeepSeek-V4.1-Flash on DeepSeek's own prompt set, plus the per-step
  cost model that predicted the gain and the W4A4 correctness audit.

---

## The 1M-context fp8/marlin configuration (measured 2026-08-31 to 09-02)

**This is where the 1M-context and 3.8M-token pool numbers come from, not the lane the default
boots today.** The default lane above serves a 500,000-token window with a 3,532,196-token pool,
and it was measured on a different checkpoint and a different harness, so do not read the two
sets of numbers against each other. Everything in this section was measured on the earlier
`keys`/LibertAI fp8 marlin lane and is kept because the engine settings still apply: the
`--max-num-seqs`, `--max-num-batched-tokens` and CUDA-graph findings below carry over.

| | |
|---|---|
| **KV pool** | **3,834,498 fp8 tokens** — 3.66x a full 1M-token context (0.86 GiB goes to graph buffers) |
| **Context** | **1,048,576** (model-native 1M) |
| **KV pin** | `--kv-cache-memory 25769803776` (**24 GiB/rank**) |
| **Speculative decoding** | **DFlash2**, `num_speculative_tokens: 7` |
| **Concurrency** | `--max-num-seqs 64` · `--max-num-batched-tokens 16384` |
| **CUDA graphs** | `cudagraph_mode: FULL_AND_PIECEWISE` — **not** `--enforce-eager` on this lane |
| **Aggregate throughput** | **530.0 tok/s** @C48 (was 183.1 @C6 before the three changes below) |
| Single stream | **105.6** count-to-100 · **77.3** code · **31.5** prose (temperature 0, median of 3) |
| Prefill | **1,863 tok/s** warmed @114K prompt · TTFT **54.8 s** (was 1,194 / 95.4 s) |

### What changed, and what each part bought (measured 2026-08-31 → 09-02)

| change | effect |
|---|---|
| `--max-num-seqs` 6 → **64** | aggregate **183 → 503 tok/s**. The cap was the constraint, not the fabric — the baseline curve was still climbing when it hit C6. Single stream unaffected by design. |
| `--max-num-batched-tokens` 8192 → **16384** | prefill **+52–79%**, TTFT @114K **95.4 s → 54.8 s**. Costs ~3% aggregate. |
| `--enforce-eager` → **`FULL_AND_PIECEWISE`** ([#5](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark/pull/5)) | aggregate **503 → 530**, and the only change that lifted **single stream**: prose **+17%**, code +10%, count-to-100 +4%. |

> **On `--enforce-eager`:** an earlier revision of this README told you to keep it, citing a
> −19% measurement. That measurement tested plain `PIECEWISE`, which forces piecewise graphs
> onto the *decode* path. `FULL_AND_PIECEWISE` uses FULL for uniform decode batches and
> piecewise only for mixed/prefill, and it is **faster than eager on every prompt type**.
> `--enforce-eager` is a property of the **NVFP4-KV/b12x lane**, not of this model — the b12x
> kernels require it, marlin does not. Keep it on the b12x lane and on the
> [topkfix image](docs/TOPK-OVERSUSCRIPTION-FIX.md), which deadlocks under graphs.

> **Quote the prompt and the temperature with any tok/s number for this model.** Acceptance
> is content-driven, so the same engine measures 105.6 on count-to-100 and 31.5 on dense
> prose, minutes apart. The harness ([`probes/bench_glm53_tp4.py`](probes/bench_glm53_tp4.py))
> uses a fixed 8-prompt set at temperature 0 with median-of-N for exactly this reason.
| Weights | the `keys`/LibertAI NVFP4 packs this section was measured on (the current default is `nvidia-glm53-attn`, above) |
| Vision | on (`chat_template_mm.jinja`) |
| Thinking | off by default |
| Launcher | [`launch-glm53-tp4-24g.sh`](launch-glm53-tp4-24g.sh) |
| Flusher | [`flusher-unconditional.sh`](flusher-unconditional.sh) — **required, and must be unconditional** |

Gate-passed 2026-08-29: two deep decodes at ~41K context (392 and 399 decoded tokens),
3x concurrent prefills at 32,879 tokens each, vision, `/health` 200 throughout. Residual
after gates: head rank 15 GiB, workers 19-20 GiB.

### The flusher is the whole trick

For a week we believed GB10 had "phantom KV backing" above 16 GiB/rank — reservations that
succeed but fault when touched under load. **It was the page cache.**

We had been running a **threshold-triggered** flusher (drop caches only when `Cached > 40 GiB`).
A threshold flusher can sit below its threshold and *still* leave the NVRM allocator short,
which is exactly why the same command booted or OOM'd depending on the moment. Making it
unconditional took the same 24 GiB pin that died on 2026-08-27 straight through the gate
suite, for **+54.8 %** pool over the old 16 GiB default.

Run it on every node, started before the launcher, for the entire boot. Credit to
[tonyliu312](https://github.com/tonyliu312/GLM-5.3-Flash-DFlash2-TP4-1M-Context) for naming
the requirement precisely.

### Measure your own ceiling — do not paste ours

24 GiB/rank is where **our** fleet lands, with 15 GiB residual on the head rank. Other
operators run 28 and 32 GiB/rank on the same hardware. The head rank is always the binding
constraint — it carries the API server and engine core on top of its shard — and startup
free memory varies several GiB between otherwise identical nodes. Ladder up, and gate every
step. **A config that boots and answers a short prompt is not a config that works.**

---

## Quickstart (4 nodes)

> **This quickstart is the previous recipe** (and the uncensored Blackfrost lane). For the current default, knapcio's
> stack at 500K, follow [`runs/2026-09-29-knapcio-stack/RUNBOOK.md`](runs/2026-09-29-knapcio-stack/RUNBOOK.md).

One node owns the weights on local NVMe and NFS-exports them; the other three mount at the
same path.

**Before anything: what you must edit for your own hardware.** The launchers are written for our
fabric. Change `MODEL_HOST_PATH` (or `MODEL_DIR` on the newer
[`runs/2026-09-20-nvidia-lanes/tools/glm53_tp4.sh`](runs/2026-09-20-nvidia-lanes/tools/glm53_tp4.sh)), the
rank->IP map, and the NCCL block (`NCCL_IB_HCA`,
`NCCL_IB_ADDR_RANGE`, `NCCL_SOCKET_IFNAME`/`GLOO`/`TP`/`MN`). `--memory 112g` assumes 128 GB
nodes. Get the NCCL values wrong and you hang at rendezvous with very little in the log.

**For the current default lane, follow
[`runs/2026-09-20-nvidia-lanes/RUNBOOK.md`](runs/2026-09-20-nvidia-lanes/RUNBOOK.md) instead of this
section.** It has the build steps, the boot command, the settle step and the measurement order for
`nvidia-glm53-attn`. What follows is the 4-node ground work both lanes need.

### 1. Image — pull once, fan out

Public and anonymous-pullable; no build required.

```bash
# ONE node pulls. Four nodes pulling a 31 GB image concurrently hits GHCR rate limits.
docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
docker save ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 \
  | zstd -3 -T8 | ssh <peer> "zstd -d | docker load"
```

### 2. Weights and files — on every node

Five things must exist on the default lane (four on the older lanes), and **most of them fail in
non-obvious ways if missing**, because Docker silently creates an empty directory over a bind-mount
source that does not exist.

```bash
# (a) main weights - the built default lane, or any checkpoint from the table below.
#     nvidia/GLM-5.3-Flash-NVFP4 is 190.4 GiB in 33 shards; the build rewrites 31 of them and
#     adds one NVFP4 attention shard. Build steps: runs/2026-09-20-nvidia-lanes/RUNBOOK.md
ls /var/tmp/models/nvidia-glm53-attn/config.json   # default lane (W4A16_NVFP4, needs NVFP4_PATCH=1)

# (b) the DFlash2 drafter (~2.2 GB)
huggingface-cli download incoai/GLM-5.3-Flash-DFlash2 \
  --local-dir /var/tmp/models/GLM-5.3-Flash-DFlash2

# (c) the SM121 indexer patch -- NOTE the rename. Without this the engine boots, answers
#     short prompts, then dies on EVERY decode past ~24K context (crash forensics, disease 1).
mkdir -p ~/patches
cp docker/sparse_attn_indexer_kpool_sm121.py ~/patches/sparse_attn_indexer_kpool.py

# (d) the vision chat template, INSIDE the weights dir (that mount is read-only at runtime)
cp chat_template_mm.jinja /var/tmp/models/nvidia-glm53-attn/

# (e) default lane only: the two patched glm5next files NVFP4_PATCH=1 mounts. Without them the
#     attention projections are built BF16 and the packed 4-bit weights cannot load.
install -d ~/patches/nvfp4
cp runs/2026-09-20-nvidia-lanes/patches/kda.py runs/2026-09-20-nvidia-lanes/patches/model.py ~/patches/nvfp4/

chmod +x launch-glm53-tp4-24g.sh flusher-unconditional.sh fleet_watchdog.sh
```

The launcher hard-fails with a named error if any of them is missing, so you find out in
one second rather than twenty minutes. `glm53_tp4.sh` also refuses to boot if the image's own copies of
`kda.py` and `model.py` no longer hash to the values the patches were cut from, because mounting stale
copies over a rebuilt image would silently run old model code.

### 3. Memory ritual — on every node

```bash
# swap must exist but must not be used. With NO swap the worker is killed outright during
# MoE marlin repack; with swappiness>0 the UVM driver can livelock unrecoverably and take
# the node off the network entirely. This does NOT survive a reboot - put it in
# /etc/sysctl.d/ or you will lose boots to it.
sudo sysctl -w vm.swappiness=0
sudo swapoff -a && sudo swapon -a
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

# leave this running for the WHOLE boot. It refuses to start without passwordless sudo.
setsid nohup ./flusher-unconditional.sh > flusher.log 2>&1 &
grep -q "flusher: starting" flusher.log || { echo "FLUSHER DID NOT START"; tail flusher.log; }
```

### 4. Tear down everything, then launch workers first

```bash
# Rule 2 is not optional. A fresh rank that rendezvouses with a dying one hangs, and a
# retry after a failed boot is the common case - do this every time.
for n in <node1> <node2> <node3> <node4>; do ssh $n 'docker rm -f vllm_glm53' ; done

./launch-glm53-tp4-24g.sh 3   # worker
./launch-glm53-tp4-24g.sh 2   # worker
./launch-glm53-tp4-24g.sh 1   # worker
./launch-glm53-tp4-24g.sh 0   # head - serves http://<head>:8000/v1
```

On the current default lane this is one command, which does the teardown, the cache drop and the
worker-first launch itself:

```bash
MODEL_DIR=nvidia-glm53-attn NVFP4_PATCH=1 MNBT=8192 SPEC_K=7 MAXLEN=500000 \
  bash /root/glm_boot.sh <label>
```

Two things it expects, both from
[`runs/2026-09-20-nvidia-lanes/RUNBOOK.md`](runs/2026-09-20-nvidia-lanes/RUNBOOK.md): run
`tools/glm_settle.sh` first if you are swapping off another large lane, because booting straight after
stopping a ~100 GiB one races the memory reclaim and vLLM refuses; and leave `VLLM_EXTRA` at its default
`--limit-mm-per-prompt {"image":4,"video":0}`, which keeps image input and skips the dummy-video warmup.
If you set `VLLM_EXTRA`, add to it rather than replacing it.

Boot is ~12 min on the default lane, ~20 min on the older fp8/marlin one: weight load, drafter load,
KV allocation, warmup. Stop the flusher once
serving (`pkill -f flusher-unconditional.sh`). Thinking is off by default; re-enable per
request with `chat_template_kwargs: {"enable_thinking": true}` — no restart needed. Tool
calling ships enabled (`glm47` parser).

### Verify the boot

On the default lane, expect `quant_algo=W4A16_NVFP4`, `Using MarlinNvFp4LinearKernel`, 43.76 GiB/rank of
weights, and:

```
GPU KV cache size: 3,532,196 tokens
```

On the older 1M fp8/marlin lane:

```
GPU KV cache size: 3,895,606 tokens, Maximum concurrency for 1,048,576 tokens per request: 3.72x
```

**Your number will differ**, and that is expected — see *Measure your own ceiling* above.
What matters is that the line appears and the pool is the size you pinned for.

Then gate it before you trust it — the suite is in
[docs/SM121-CRASH-FORENSICS-2026-08-27.md](docs/SM121-CRASH-FORENSICS-2026-08-27.md).

---

## Which weights

Everything in this table is MIT and ungated. The two lanes we serve are **built locally from the nvidia
pack**, so the row says which upstream weights go in, not a directory you can download.

| | upstream weights | what it is |
|---|---|---|
| **⭐ Default lane** (`nvidia-glm53-attn`) | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) | official nvidia quant plus our NVFP4 attention, declared `W4A16_NVFP4`. Needs `NVFP4_PATCH=1`. The only pack here that ships calibrated `input_scale` tensors (36,297 of them), and it keeps one full expert layer in BF16 |
| **⭐ Uncensored lane** (`blackfrost-glm53-derisked-attn`) | [Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4](https://huggingface.co/Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4) plus our NVFP4 attention | abliterated from `zai-org/GLM-5.3-Flash-BF16` at the routed-expert `down_proj` tensors, where refusal actually lives. Operator-confirmed uncensored and coherent; acceptance 0.415 vs the censored lane's 0.394. **Its config declares `quant_algo: NVFP4` with zero `input_scale` tensors — rewrite to `W4A16_NVFP4` or it garbles.** Ships 120 shards and no `chat_template_mm.jinja` |
| ~~Abliteration transplant~~ (`nvidia-glm53-ablit-attn`) | nvidia, plus [dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4) `o_proj` on layers 12-44 | **RETIRED, do not use.** Edits 46 `o_proj` tensors (0.03 of the refusal signal) instead of 12,384 `down_proj` (0.81). Stopped refusing, but garbled in live use in two unrelated harnesses and acceptance collapsed to 0.224 / 2.57 tok/step |
| Prior base | [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) | what the default lane replaced, and the baseline the comparison above is against. Its 27-Aug build shipped **no calibrated `input_scale` tensors** while declaring `NVFP4` (which selects W4A4); they later added a separate 4.6 MB `model-input-scales.safetensors`. Harmless on a `W4A16_NVFP4` lane, which never reads activation scales |
| Alternative | [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) | `compressed-tensors` rather than ModelOpt, with real activation scales. Was the default while it was the corruption workaround; still a clean drop-in if you would rather not build anything, at W4A4 |
| Historical | [drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock) | the abliterated pack the NVFP4-KV lane below was measured on (layers 15-45, anchor-stock) |

Build steps for the two lanes, including why the `ignore` list must not be hand-derived:
[`runs/2026-09-20-nvidia-lanes/RUNBOOK.md`](runs/2026-09-20-nvidia-lanes/RUNBOOK.md).

Transplant credit: `dealignai` for the donor `o_proj` tensors, and
[drowzeys / Keys](https://github.com/drowzeys) for documenting the `o_proj` byte-copy method their
`METHOD.md` measured at 32/32 bypass where computed-direction methods reached 6-9/32.

---

## Performance

**The numbers in this section are from the 2026-08/09 fp8/marlin lane, not the current default.** For the
default lane, see the tables at the top and the full spreads in
[`runs/2026-09-20-nvidia-lanes/`](runs/2026-09-20-nvidia-lanes/) and
[`runs/2026-09-20-tp4-vs-deepseek/`](runs/2026-09-20-tp4-vs-deepseek/). The reasoning below about *why*
single-stream tok/s on this model is a statement about the prompt still applies to both lanes, and the
plus-or-minus 25% run-to-run band measured on 2026-09-20 is the reason the current lane publishes
3-rep medians with their spreads.

**54.5 tok/s single stream** — one run, 408 tokens in 7.5 s, "write a function then
explain it", temperature 0, thinking off, on the gate-passing boot. n=1; treat it as
indicative, not a benchmark. **4,141.8 tok/s prefill** (single sample, warmed, measured off-box during the gate suite — cold first-prefill on this stack is far slower, ~467 tok/s, because the kernels JIT).

**Read this before quoting any speed number, including ours.** Draft acceptance on this
model is *content-driven, not config-driven* — roughly 0.70+ on structured/code output and
~0.33 on freeform prose, measured minutes apart on the same engine. `ms/step` barely moves;
what changes is how many of the drafter's 7 positions survive. So a single-stream tok/s
figure is really a statement about the prompt. **Quote the prompt or the number is
meaningless.** Cumulative acceptance across our gate suite was 0.271, but that suite was
deliberately prose-heavy and understates normal traffic.

`--max-num-batched-tokens 8192` is set. Left unset, vLLM derives 2048 from the speculative
settings and warns that this is suboptimal. The ladder on a 32K prompt — 2048 -> 4096 ->
8192 for -29 % TTFT and +42 % prefill at about +1 GiB — is
[tonyliu312's measurement](https://github.com/tonyliu312/GLM-5.3-Flash-DFlash2-TP4-1M-Context);
we adopted the flag and confirmed the direction at TP4.

> **CUDA graphs on the fp8/marlin lane — 2026-08-31.** `--enforce-eager` is required by
> the b12x kernels *on the NVFP4-KV path only*. On the **fp8/marlin lane** it can be
> dropped: `FULL_AND_PIECEWISE` graphs capture cleanly with DFlash2 (0.86 GiB, no
> vLLM#53030 pin) for **+22% decode** (26.7 -> 32.5 tok/s). Keep the accept-ratio check.
> Full method + probe battery: [docs/CUDA-GRAPHS-DFLASH2-FP8-TP4-2026-08-31.md](docs/CUDA-GRAPHS-DFLASH2-FP8-TP4-2026-08-31.md).

### DFlash2

**First working DFlash2 deployment of GLM-5.3-Flash on GB10.** The drafter was published for
SGLang; this repo carries the vLLM route. It costs **zero KV pool** — its layers slot-share
the MLA tensors the way GLM's own mamba layers do — and its 7 positions come from a single
parallel pass, so the step does not get longer, it just carries more accepted tokens.

At TP2/262K it measured 46.9 tok/s vs 21.8 for MTP-4 (2.15x) at 74.1 % acceptance, with a
concurrency sweep of C1 35.1 · C2 41.6 · C3 40.6 · C4 47.5 · **C5 56.2** · C6 47.7 aggregate.
*Those are TP2 numbers, kept because they are the clean matched-settings comparison.*

**We have not run a matched DFlash2-vs-MTP-4 comparison at TP4.** Our TP4 DFlash2 figure
(54.5 tok/s, mixed code+prose) and the TP4 MTP figure in the lane table below (~55 tok/s,
structured/warmed) were measured on *different prompts* and are not comparable — do not read
them as DFlash2 and MTP being level here. Another operator's matched TP4 run found DFlash2
well ahead on code and structured output and roughly level on prose, which is what the
acceptance data predicts.

Method, the nine-boot failure ladder, and the KV-layout fix that keeps GLM on its custom fast
path: **[docs/DFLASH2-SPECULATIVE-DECODING.md](docs/DFLASH2-SPECULATIVE-DECODING.md)** ·
bench detail: [docs/BENCH-C1-C6-DFLASH2.md](docs/BENCH-C1-C6-DFLASH2.md) · reproducible
overlay: [`overlay-dflash2/`](overlay-dflash2/)

---

## KV-cache lanes: fp8 (default) vs NVFP4

> **Naming:** "Lane A" and "Lane B" in this section are **KV-cache formats**, a separate axis from the two
> weight lanes at the top of this README. The current default serves fp8 KV.

**fp8 is the daily driver** — faster, simpler to operate, and what the config above ships.
**NVFP4 KV is the flex lane** for the rare job that needs pool capacity over throughput.

| | **fp8 KV** (default) | **NVFP4 KV** (flex) |
|---|---|---|
| KV density | 512 B/token/layer (NoPE, unpacked) | **368 B/token/layer** |
| Decode (structured, warmed) | **~55 tok/s** | ~37 tok/s |
| Pool at equal 32 GiB/rank | 5,033,164 | **6,652,112** (1.32x) |

The trade is clean: **NVFP4 buys capacity, fp8 buys speed.** But be careful with the density
ratio: our fp8 route runs the **512 B/token NoPE** record, not the 656 B packed one, so the
real format win is about **1.36x**, not the 1.78x raw byte counts imply — and roughly **1.17x**
once the standalone drafter's pages are counted. See [docs/OPEN-PROBLEMS.md](docs/OPEN-PROBLEMS.md). Note the pool figures in that
row were measured at a **32 GiB/rank** pin during the 2026-08-27 lane comparison, not at the
24 GiB default above — they are a like-for-like comparison between formats, not this repo's
shipped numbers. **32 GiB/rank is also the pin that later failed our concurrent-prefill gate**
(see Superseded configurations); those pool figures are real allocations, not endorsed configs.
They also come from a different engine generation than the 24 GiB number, which is why
tokens-per-GiB does not divide out evenly across the three.

Both `nvfp4_ds_mla` KV and `cudagraph_mode: FULL` have since been confirmed working alongside
DFlash2 on this fleet, closing two entries in
[docs/OPEN-PROBLEMS.md](docs/OPEN-PROBLEMS.md).

### Lane B — NVFP4 KV at TP4 (verified serving, 2026-08-27)

We took the **NVFP4 KV** recipe from [drowzeys/keys](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated) — the **Zero-RoPE shim** (pad GLM's NoPE attention with a virtual 64-dim RoPE so the cache presents DeepSeek's 576-dim record to the NVFP4 kernels, bit-for-bit identical to NoPE) + Luke Alonso's **b12x** `B12X_MLA_SPARSE` backend + `KV_DTYPE=nvfp4_ds_mla` — and **extended it from his 2-Spark TP2 to our 4-Spark TP4** on the abliterated `keys` weights.

| Metric | NVFP4-KV TP4 (abliterated `keys` weights) |
|---|---|
| **KV pool** | **6,652,112 tokens = 6.34× a full 1,048,576-token context** (≈6.3 full-1M conversations at once), at 32 GiB KV/rank |
| KV dtype | `nvfp4_ds_mla`, `KV_FP8_ROPE=1` — **368 B/token/layer** (vs 656 fp8, 1152 bf16) |
| Context | 1,048,576 (1M), served on `:8000` as `glm-5.3-flash` (drop-in name) |
| Decode | **~36 tok/s single-stream** warmed (temp 0, structured) — `--enforce-eager` is required by the b12x kernels and caps single-stream; concurrent aggregate is far higher. Prefill ~1,050–1,290 tok/s, TTFT ~0.2 s |
| Vision | ON (image input verified) · Thinking off by default · abliterated `keys` weights (refusal behaviour not re-tested by us) |
| Spec decode | native MTP head, k=2 · GMU 0.85 hard cap · 32 GiB KV/rank |

**fp8 vs NVFP4 KV at EQUAL 32 GiB/rank budget:** NVFP4 KV = **6,652,112 tokens** vs fp8 = 5,033,164 — **1.32× the pool at the same memory** (the 368-vs-656 B/token density showing through). As far as we can tell this is the **first NVFP4 KV cache at TP4 on consumer Blackwell**, and ~5.4× the reference 2-Spark TP2 pool (1.22M tokens). **Full credit to [drowzeys / keys](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated)** for the Zero-RoPE shim, the b12x NVFP4 kernels, and the ablit weights; our contribution is the TP4 port + the 4-node fabric/memory config.

### fp8 vs NVFP4 KV — speed head-to-head (measured 2026-08-27)

The density win above is **not free** — we ran both lanes back-to-back on the same `:8000` endpoint (same abliterated `keys` weights, TP4, `--enforce-eager`, MTP, temp 0, warmed) to price it:

| Metric | Lane A — fp8 KV | Lane B — NVFP4 KV | Winner |
|---|---:|---:|:--|
| **Decode** (structured/agentic, warmed) | **~55 tok/s** (51 / 56 / 55) | **~37 tok/s** (37 / 37) | **fp8 ~1.5×** |
| Prefill (warmed, ~9K-token prompt) | **~3,530 tok/s · 2.55 s TTFT** (3 runs) | ~1,449 tok/s · 6.9 s TTFT¹ | fp8¹ |
| KV pool @ 32 GiB/rank | 5,033,164 | **6,652,112** | **NVFP4 1.32×** |
| KV density | 512 B/token/layer (NoPE, unpacked) | **368 B/token/layer** | **NVFP4 1.8×** |

¹ The NVFP4 prefill/TTFT is a **single sample that may have been partly cold** (our fp8 first-prefill was 19 s / 467 tok/s cold, then settled to ~2.5 s / ~3,530 warmed — the b12x kernels JIT on the first large prefill too). We did not capture a clean warmed long-prompt NVFP4 prefill before teardown, so **treat decode as the definitive head-to-head and the prefill row as directional, not final.**

**What this means:** NVFP4 KV's ~33 % slower decode is the b12x `B12X_MLA_SPARSE` sparse-attention path (+ the per-token NVFP4 dequant) doing more compute per step than fp8's marlin path — and `--enforce-eager`, which the b12x kernels require, caps single-stream on both lanes. So the trade is clean: **NVFP4 = KV *capacity* (bigger context pool at equal VRAM), fp8 = *speed* (faster tokens for the same agent work).** For our production endpoint we run **fp8 as the daily driver** (faster, vision, simplest to operate) and keep **NVFP4 as the flex** for the rare job that needs the giant pool over throughput.

---

## Hard-won rules (each one cost us a boot)

1. **The cache flusher must be unconditional**, on every node, for the whole boot. A
   threshold-triggered one silently starves the allocator.
2. Tear down **all** ranks before relaunching **any** — a fresh rank that rendezvouses with
   a dying one hangs.
3. **Verify the image ID, not the tag name**, on every node before every launch:
   `docker image inspect <tag> --format '{{.Id}}'` must return the *same* sha256 everywhere.
   Matching tags prove nothing — four nodes that each built the tag locally get four different
   images. Pull or `docker save | ssh | docker load` from ONE node so the IDs are identical.
   A rank silently on a divergent image is the hardest failure in this repo to diagnose
   (see `docs/DEPLOY-REPORT.md`, boot 8). Copy launcher files whole, never `sed` over ssh.
4. **Gate with a long prompt AND a long answer.** `persistent_topk` crashes on decode
   *steps* past ~24K context, so a 49K prompt with a one-line answer proves nothing — ours
   decoded 15 tokens and passed meaninglessly. Force >=100 completion tokens, and vary the
   prompt per run or the prefix cache turns a 30 s gate into a 2 s no-op.
5. The bar is **concurrent** prefills, not one. 32 GiB passed a single-prefill gate and then
   died under three overlapping requests.
6. Reboot a node that has been through many boot cycles — GB10's driver accumulates
   allocation-pool degradation that eventually kills even proven configs.
7. Capture `docker logs` before `docker rm -f`.

---

## What's in here

- [`runs/2026-09-20-nvidia-lanes/`](runs/2026-09-20-nvidia-lanes/): **the current default lane.** Both
  weight lanes measured side by side, per-file checksums, the patched `kda.py` and
  `model.py` that `NVFP4_PATCH=1` mounts, a self-contained `tools/` including the launcher and harness,
  and [`RUNBOOK.md`](runs/2026-09-20-nvidia-lanes/RUNBOOK.md) with every command and value in order.
- [`runs/2026-09-20-tp4-vs-deepseek/`](runs/2026-09-20-tp4-vs-deepseek/): where the NVFP4 attention work
  came from, with the head-to-head against DeepSeek-V4.1-Flash on DeepSeek's prompt set, the per-step cost
  model, the two boot traps, and the W4A4 source audit.
- [`launch-glm53-tp4-24g.sh`](launch-glm53-tp4-24g.sh): the fp8/marlin launcher this README's 1M section
  was measured with. Head serves `:8000`; run worker-first (rank 3 -> 2 -> 1, head 0 last). Full NCCL
  fabric env included. The default lane boots through
  [`runs/2026-09-20-nvidia-lanes/tools/glm53_tp4.sh`](runs/2026-09-20-nvidia-lanes/tools/glm53_tp4.sh)
  and `glm_boot.sh` instead, and its `MODEL_HOST_PATH` is derived from `MODEL_DIR`.
- [`flusher-unconditional.sh`](flusher-unconditional.sh) — **required sidecar** on every node
  during boot. Mechanism and measurements: [docs/GB10-KV-MEMORY-LADDER.md](docs/GB10-KV-MEMORY-LADDER.md).
- [`chat_template_mm.jinja`](chat_template_mm.jinja) — **required for vision.** The checkpoint
  ships a text-only template; image requests 500 without this.
- [`docker/`](docker/) — the image patch stack (v1 -> v9) applied to the day-0
  `vllm/vllm-openai:glm53-flash-arm64-cu130`. Fixes the NoPE-MLA backend gap, a FlashInfer FA2
  NaN kernel bug, two dependency downgrades the FlashInfer nightly sneaks in, a PDL race
  surface, uninitialized indexer top-k memory, and the fp8-KV shared-memory tile bug.
> **On image provenance:** the shipping tag `sm121-v11-dflash2` is `docker/` v1->v9 plus the
> DFlash2 overlay in [`overlay-dflash2/`](overlay-dflash2/); there is no standalone `v11`
> Dockerfile here. Pull the published image rather than rebuilding unless you are changing the
> patch stack.

- [`overlay-dflash2/`](overlay-dflash2/) — the DFlash2 vLLM overlay: patches plus a CPU
  simulator that validates the KV geometry before you boot a node.
- [`docker/topkfix/`](docker/topkfix/) — exact-`torch.topk` top-k fix for Disease 1
  (cleaner than the SM-count gate). See
  [`docs/TOPK-OVERSUSCRIPTION-FIX.md`](docs/TOPK-OVERSUSCRIPTION-FIX.md).
- [`docs/SM121-CRASH-FORENSICS-2026-08-27.md`](docs/SM121-CRASH-FORENSICS-2026-08-27.md) — the
  two diseases behind "random" deaths, and the gate suite.
- [`docs/DEPLOY-REPORT.md`](docs/DEPLOY-REPORT.md) — every failure and receipt from deploy day.
- [`probes/`](probes/) — the debugging kit: kernel probes with real model geometry, a NaN
  bisect harness, kernel-vs-torch A/B, and the benchmark script. **`gb10_alloc_probe.py`
  allocates all available memory to map the allocation wall — maintenance window only, never
  on a node that is serving.**
- [`fleet_watchdog.sh`](fleet_watchdog.sh) — systemd-friendly self-healing: checks every
  rank's container, probes `/health`, and sends a one-token canary; on 3 consecutive failures
  tears down all ranks, runs the memory ritual, starts the unconditional flusher, and
  relaunches workers-first. It gives up after 3 failed relaunches in a row
  (`~/.fleet_watchdog.gaveup`) and pauses while `~/.fleet_watchdog.pause` exists. Recovery is
  ~15 min, so tune the threshold before pointing it at a busy endpoint. `/health` alone misses
  a dead worker: the head keeps answering 200 while it waits in a collective.

---

## Appendix: token corruption on the retired transplant lane (vLLM #54150)

> **This does not affect either shipping lane.** Both the censored `nvidia` lane and the uncensored
> `Blackfrost` lane have been exercised by the operator in the same harness that garbled, and neither
> reproduced it. Corruption tracked the **retired** dealignai `o_proj` transplant, which is no longer a lane
> here. You do not need to read this to deploy.

Kept because the bug was real, the history matters, and this repo published two explanations that turned out to
be wrong. If you are building your own abliterated pack, the Resolution note below is the part worth your time.

### What still stands

NVIDIA's ModelOpt-quantized NVFP4 builds scored **4 / 9 / 8** corrupted token IDs across three passes where
RedHat's `compressed-tensors` build scored **0 / 0 / 0** ([vLLM #54150](https://github.com/vllm-project/vllm/issues/54150)).
That probe demonstrably detects the fault, because it found it. Nearly invisible in English prose, but a
corrupted token inside a tool-call block desyncs the parser and generation can spiral into a repetition lock.

### Retraction 1: `W4A16_NVFP4` is not a fix

An earlier revision claimed the fix was declaring `quant_algo: W4A16_NVFP4`, on the strength of
`0 in 28,617` suspect characters from `tools/corrupt_probe.py`. **That is a false negative and the claim is
withdrawn.** The probe counted only CJK, Cyrillic, Hangul, Arabic and replacement characters; a corrupted token
ID can map to *any* vocabulary entry, including Latin-script words and structural tokens like `</tool_call>`.
It also ran at temperature 0, single-turn, at 219-619 token contexts. It could not have found this bug.

### Retraction 2: the W4A4 / missing-`input_scale` mechanism is wrong

The stated mechanism was that a pack declaring `quant_algo: NVFP4` takes a W4A4 activation path and multiplies
real weight scales by absent or placeholder `input_scale` values. **Checked directly against
`nvidia/GLM-5.3-Flash-NVFP4` and it does not hold** - the activation scales are all present:

| tensor class in the stock nvidia checkpoint | count |
|---|---|
| `*input_scale` | **36,297** |
| `*weight_scale` | 36,297 |
| `*weight_scale_2` | 36,297 |
| total tensors | 147,661 |

One `input_scale` per quantized linear, exactly matching the weight-scale count. Nothing is missing, so
`W4A16_NVFP4` is not "closing a corruption route" - it is a weight-only compatibility choice that ignores
activation scales which genuinely exist. The mechanism is unknown.

### What is ruled out

Each of these was a hypothesis that failed a measurement, listed so nobody re-runs them:

| hypothesis | test | result |
|---|---|---|
| Client/harness rendering artifact | two unrelated harnesses (dsh, OMP) on the same lane | **ruled out** - both corrupt identically |
| Speculative-decode rejection sampler (temperature > 0 path) | same prompt at temperature 0.0 / 0.3 / 0.7 / 1.0 | **ruled out** - behaviour identical at 0 and 1.0 |
| Context length alone | 12 accumulating turns, 33 KB system prompt, to 15.3 K context, non-streaming | **not reproduced** - 48 K characters clean |
| Missing activation scales (above) | tensor census of the stock checkpoint | **ruled out** - 36,297 present |
| The dealignai `o_proj` transplant | operator side-by-side: the transplant lane garbles in two unrelated harnesses, the censored nvidia lane and the Blackfrost lane do not; acceptance 0.224 on the transplant vs 0.394 / 0.415 | **leading cause** (see the resolution note below). An earlier revision of this table said "not the cause" on the grounds that `4 / 9 / 8` was measured on unmodified packs — that reasoning is withdrawn |

Not yet tested, in priority order: the **full tool-call round trip** (assistant `tool_calls` -> `role: tool`
result -> next turn, streamed, which is what both harnesses do and what every probe above omitted); context far
beyond 15 K. **Both current lanes have since been exercised by the operator** in the harness that garbled on
the retired transplant lane, and neither reproduced it.

### Resolution note (2026-09-21): the transplant lane, not the quantizer

Two operator tests settled the practical question even though the mechanism is still unmeasured. The censored
`nvidia` lane and the `Blackfrost` lane are both **ModelOpt** builds and both behave correctly in the harness
that garbled; the retired dealignai `o_proj` transplant garbled in **two unrelated harnesses**. So ModelOpt as a
class is not the trigger, and the earlier plan to move back to RedHat `compressed-tensors` is unnecessary.

A second, independent signal agrees. DFlash2 was trained against stock GLM-5.3-Flash, so draft acceptance
measures how far a target has drifted from stock:

| lane | acceptance | tokens/step | position 7 |
|---|---|---|---|
| dealignai `o_proj` transplant (retired) | **0.224** | **2.57** | **0.000** |
| nvidia (censored default) | 0.394 | 3.76 | healthy |
| Blackfrost (uncensored default) | **0.415** | **3.90** | healthy |

**This remains inferential.** Seven probes failed to reproduce the garbling directly: two harnesses corrupt
identically (not a client artifact), temperature 0 behaves like temperature 1.0 (not the spec-decode rejection
sampler), 48K characters over 12 turns to 15.3K context stayed clean, full streamed tool-call round trips stayed
clean, and the `input_scale` explanation was wrong (nvidia ships all 36,297). What is established is which lanes
are safe to run, not the mechanism inside the broken one.

### Use an adequate detector

`tools/corrupt_probe.py` and `tools/corrupt2.py` are kept only as a record of what not to do. A detector for
this fault needs to score, on long multi-turn streamed sessions with real tool cycles: emoji runs, any
foreign-script character in an English response, duplicated lines and repeated SSE deltas, literal tool-call
markup arriving as content, reasoning text leaking into the content channel, and 8-gram diversity of the output
tail (degenerate tails fall below ~0.40). Validated against a real corrupted transcript and silent on clean
prose.

build, is **not needed on a W4A16 lane**. It supplies calibrated activation scales for a path these lanes do
not execute. Switching to W4A4 to use it would additionally break any NVFP4 attention tensors added by this
recipe, since those carry no activation scales of their own.

Corruption first flagged by [@ajclark](https://github.com/ajclark) (issue #10).

---

> **Second site, 100G switched fabric:** [docs/FIELD-NOTES-4NODE-100G.md](docs/FIELD-NOTES-4NODE-100G.md)
> reproduces this recipe on four GX10 through an Arista 7060CX-32S at 100G. Decode matched
> the 200G numbers, which suggests it is not fabric-bound. Also covers a GID-index lookup
> for the launcher, why AOC transceivers overheat in the GX10 cages, and moving 185 GB
> between nodes without encrypting it.

---

## Superseded configurations

Kept for the record. **Do not deploy these** — the current config is at the top.

| date | config | pool | why superseded |
|---|---|---|---|
| 2026-08-27 | 16 GiB/rank fp8, TP4 | 2,516,582 | 24 GiB now passes gates with the unconditional flusher (+54.8 %) |
| 2026-08-27 | 32 GiB/rank | 5,033,164 | passed a single-prefill gate, died under three concurrent requests |
| 2026-08-27 | 38 GiB/rank | 5,975,779 | allocates and boots, then the first 20K prefill NVRM-OOMs a rank |
| earlier | TP4 with MTP-4 (no DFlash2) | — | DFlash2 is faster at zero KV cost |
| 2026-08-29 | `RedHatAI/GLM-5.3-Flash-NVFP4` as the default checkpoint | n/a | adopted as a token-corruption workaround; the nvidia lane is level-or-ahead on throughput and has not shown corruption in operator use. (An earlier entry here cited "0 corruption in 28,617 characters" — that was a false negative from an inadequate probe and is withdrawn.) RedHat is still a fine alternative, it is just no longer the default |
| 2026-09-20 | `keys-glm53-nvfp4-attn3` (LibertAI-parented, NVFP4 attention) | 3,532,196 | same recipe on the nvidia pack, which is MIT, ungated, and ships calibrated `input_scale` tensors |

The 38 GiB case is the cautionary one: it allocates cleanly, boots, and answers short prompts
before dying. On GB10, "serving" is not the bar.

---

## Fast loading: InstantTensor (added 2026-08-27)

**Status: experimental — 15x load speedup measured, but NOT stable in our multi-node TP2 topology** (a rank dies silently ~1 min post-load in every v9 boot, at any KV size, including budgets that are 100% stable on v8; cf. eugr/spark-vllm-docker#29 for the same multi-node class of problem). The shipped launchers do NOT enable it; the stable image remains v8. The v9 image adds the InstantTensor direct-I/O loader (`--load-format instanttensor`): loads drop from ~10 minutes to 40-100 seconds. Two things to know: its pip install silently downgrades NCCL to a fabric-fatal version (v9 re-pins 2.30.7 in the same layer), and because direct I/O never fills the page cache, it also defeats the first layer of the GB10 KV-allocation wall -- the full story and the remaining (unsolved) second wall are in [docs/GB10-KV-MEMORY-LADDER.md](docs/GB10-KV-MEMORY-LADDER.md). Credit: jack6464 (NVIDIA forum) for the pointer.

---

## vLLM v0.28.0 status (checked 2026-08-27)

**Not viable for GLM-5.3 yet**: the `glm5_next` architecture is NOT in the v0.28.0 release
(PR vllm-project/vllm#53906 still open/unmerged at check time), and no rebased day-0 image
exists (all `vllm/vllm-openai:glm53-flash*` tags still date to the original 2026-08-26 push).
The day-0 image used here is itself a main-branch dev snapshot (`0.1.dev20051`) cut around
the 0.28 branch point -- i.e. this stack already runs 0.28-era engine code plus the GLM
support 0.28 lacks. Upgrade path when it opens: watch the PR and the Docker Hub tags; the
patch stack here is guarded string-patches that apply-or-refuse loudly, so porting to a new
base is mechanical (apply v1->v9 in order, fix whichever guards fire, ladder through the
experiment lane before production).

---

## Credits

Model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) ·
Serving stack our fleet runs as of 2026-09-29 (lossless 8-bit dense, RoCE all-reduce, LeversScheduler, DFlash2 fp8-block drafter):
[knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4) by [@knapcio](https://github.com/knapcio) ·
Base quant: [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) (the default lane
builds on it) · `o_proj` donor for the abliteration-transplant lane:
[dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4) ·
the `o_proj` byte-copy method, documented in their `METHOD.md`:
[drowzeys / Keys](https://github.com/drowzeys) ·
alternative quant: [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
(compressed-tensors) · prior base quant:
[LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) ·
DFlash2 drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) ·
NVFP4-KV lane, Zero-RoPE shim, b12x kernels and the ablit weights:
[drowzeys / keys](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated) ·
the `--max-num-batched-tokens` ladder and the unconditional-flusher requirement:
[tonyliu312](https://github.com/tonyliu312/GLM-5.3-Flash-DFlash2-TP4-1M-Context) ·
barrydeen (gmu reference + quant table) · vLLM [PR #53906](https://github.com/vllm-project/vllm/pull/53906)
authors for the day-0 image · FlashInfer 0.6.18 · Luke Alonso (b12x) ·
jack6464 (InstantTensor pointer).

Everything upstream is MIT and ungated. **This repo hosts no weights**, only the recipe, the patches and the
checksums, so credit and traffic stay with the original authors.

Deployed and debugged by Knox (Claude) for [@tonyd2wild](https://github.com/tonyd2wild).
Sibling repos: [TP2 / 262K](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) ·
[262K deep-dive](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark).
