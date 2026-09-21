# Blackfrost DERISKED as the uncensored lane — 2026-09-21

**This replaces the dealignai `o_proj` transplant as the uncensored default.** That lane was uncensored
*because it was damaged*, and it garbled in live agentic use. This one is a real abliteration on the BF16
upstream, and it measures at or above the censored nvidia lane on almost every cell.

Serving on `:8000` as `glm-5.3-flash`, 500K context, built 1:1 with Lane A.

| | |
|---|---|
| Base | [`Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4`](https://huggingface.co/Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4) (from `zai-org/GLM-5.3-Flash-BF16`) |
| Our build | + NVFP4 attention (`mknvfp4b.py`), proven 14-entry ignore list, `quant_algo: W4A16_NVFP4` |
| Weights | **43.76 GiB/rank** (identical to Lane A) |
| KV pool | **3,532,196 fp8 tokens** at 500K (identical to Lane A) |
| Boot | ~13 min to `/health` 200 |
| Quality gate | **PASS** — count, JSON, code, math, prose |
| Draft acceptance | **0.415, 3.90 tokens/step** (Lane A: 0.394, 3.76) |
| Uncensored | **confirmed by the operator** on his own refusal probe |
| Long-session coherence | **confirmed by the operator** in the harness that garbled on the old lane |

## Why the old uncensored lane failed, and why this one does not

Refusal in this model lives in the routed-expert `down_proj` tensors. Our attribution measured
`down_proj` moving refusal by **0.81** against **0.03** for attention plus shared/dense MLP.

| lane | what it edits | tensors | result |
|---|---|---|---|
| keys / dealignai (retired) | `self_attn.o_proj` | **46** | uncensored but garbles; acceptance collapsed to 0.224 / 2.57 tok/step |
| **Blackfrost DERISKED** | routed-expert `down_proj` | **12,384** | uncensored and coherent; acceptance **0.415 / 3.90** |

The keys lane bought non-refusal by perturbing attention, which is also what made it incoherent over long
contexts. Acceptance is the cheapest tell: DFlash2 was trained against stock GLM-5.3-Flash, so a target that
rejects most of its drafts has drifted far from stock. The keys lane sat at 22.4% with position 7 at exactly
zero. Blackfrost sits **above** the nvidia pack.

### Locating Blackfrost's edits (their method is undisclosed)

Both packs are NVFP4 quantizations of the same `zai-org` base, so an untouched tensor differs only by
quantization noise. `bf_where.py` dequantizes matched tensors and uses never-abliterated classes as the floor:

| class | samples | rel. difference vs nvidia | min | max |
|---|---|---|---|---|
| **expert `down_proj`** | 6 | **0.101** | 0.092 | 0.108 |
| expert `gate_proj` | 6 | 0.039 | **0.000** | 0.082 |
| expert `up_proj` | 6 | 0.039 | **0.000** | 0.082 |

`gate_proj`/`up_proj` are byte-identical in some layers, which proves the two producers quantize untouched
weights identically. Every sampled `down_proj` differs, tightly clustered at 9–11%. **Caveat:** n=6 per class,
and the non-zero `gate`/`up` samples could be edits or could come from the two ModelOpt versions (0.45 vs 0.47).
The `down_proj` finding is consistent across all six; the rest is not resolved at this sample size.

## The trap in Blackfrost's config — fix this or it will garble

Its `config.json` declares **`quant_algo: NVFP4`**, which selects the **W4A4** path, while the checkpoint ships
**zero `input_scale` tensors**. W4A4 kernels would then multiply real weight scales by uninitialized memory.
Its own `config_groups` says `input_activations: null`, so weight-only is the intent and the string is
mislabeled. `fix_ignore.py` rewrites it to `W4A16_NVFP4`. **Booting it unedited would likely produce garbling
that looks exactly like the corruption this repo spent a night chasing.**

## Numbers

3-rep medians, same harness and prompts as `nvidia-laneA`, one discarded warm-up first. Full per-rep JSON in
[`results/`](results/); the lane-vs-lane judgement with spreads is [`results/compare-lanes.md`](results/compare-lanes.md).

### C1 per-stream decode tok/s

| category | Lane A (censored) | **Blackfrost (uncensored)** |
|---|---|---|
| code | 78.7 | **95.8** |
| JSON | 75.9 | **80.8** |
| math | 88.8 | 83.3 |
| prose | 40.8 | **50.8** |
| structure | 106.3 | 106.4 |
| counting | 107.7 | **138.1** |
| reasoning | 71.5 | **76.9** |
| summary | 51.5 | **58.3** |
| narrative | 38.2 | **41.2** |

**Math is the one category below Lane A** (−6%, at the edge of the measured spread). Everything else ties or wins.

### Aggregate and TTFT

| level | Lane A | Blackfrost |
|---|---|---|
| C1 | 63.2 (ttft 0.200) | **66.0** (0.219) |
| C2 | 93.3 (0.254) | **96.5** (0.251) |
| C3 | 117.0 (0.288) | **125.0** (0.283) |
| C4 | 140.0 (0.408) | 137.4 (0.397) |
| C5 | 152.9 (0.439) | 148.8 (0.411) |
| C6 | 172.4 (0.413) | 165.5 (0.438) |
| C8 / C12 / C16 | 193.1 / 241.9 / 271.1 | **198.1** / 237.6 / 271.0 |
| C24 / C32 | 315.4 / 352.8 | **326.2** / 351.9 |

### Cold prefill, rep 1 only

| target tokens | Lane A | Blackfrost |
|---|---|---|
| 3,814 | 1,092 * | 1,958 (ttft 1.9 s) |
| 15,168 | 926 * | 2,016 (ttft 7.5 s) |
| 60,917 | 1,557 * | 1,993 (ttft 30.6 s) |
| 121,681 | 1,576 * | 1,761 (ttft 69.1 s) |

\* Lane A's prefill cells are the **contaminated** ones this repo already flags (taken 2 minutes after boot).
Blackfrost's 1,958–2,016 match the clean LibertAI baseline's 1,979–2,009, so read this as *normal*, not a win.

### Idle probe and acceptance

```
count after idle   147.6 tok/s  7.96 tok/step  TTFT 0.15s  step p50 54.1 ms
count back-to-back 141.0        7.65          TTFT 0.17s
code  back-to-back  98.2        5.54          TTFT 0.18s  step p50 56.3 ms
code  (2nd sample)  58.2        3.27  <- real spread, single cells swing on this fleet
```

Cumulative acceptance by draft position, 79,788 drafts at k=7:

| p1 | p2 | p3 | p4 | p5 | p6 |
|---|---|---|---|---|---|
| 0.578 | 0.442 | 0.347 | 0.285 | 0.249 | 0.217 |

No collapse in the tail — contrast the keys lane, which hit 0.000 by position 7.

## Reproducing this exactly

1. `bf_dl.py` — download the 191.0 GiB / 120-shard checkpoint (resumable).
2. `bf_build.sh` — verify the download against its index, copy `chat_template_mm.jinja` from the nvidia pack
   (Blackfrost ships none, and the launcher hard-requires it), quantize attention with `mknvfp4b.py` (196 fused
   groups, shared amax per group), assemble with `build_nv.py` (hardlinks unchanged shards), then apply
   `fix_ignore.py` for the 14-entry ignore list and `W4A16_NVFP4`.
3. `boot_blackfrost.sh` — settle, then boot TP4 at 500K with `NVFP4_PATCH=1 MNBT=8192 SPEC_K=7`, image limit 16.
4. `measure_bf.sh` — the measurement half of `lane_run.sh`, unchanged.

Build output: `convert rc=0, 196 groups` / `build rc=0, VERIFY dups 0 | indexed-missing 0 |
index-points-at-wrong-file 0 | names 112170`. Census matches Lane A: q_proj 34/34/34, o_proj 46/46,
`W4A16_NVFP4`, ignore 14.

**No weights are published here.** Blackfrost's pack is theirs; this is the recipe to run it.

## What this does not settle

The **corruption mechanism on the keys lane was never reproduced in a probe.** Seven attempts failed: two
unrelated harnesses corrupt identically (so not a client artifact), temperature 0 behaves like temperature 1.0
(so not the spec-decode rejection sampler), 48K characters over 12 turns to 15.3K context stayed clean, full
streamed tool-call round trips stayed clean, and the `input_scale` explanation this repo used to give is wrong
(nvidia ships all 36,297). The evidence that it was the `o_proj` transplant is the operator's side-by-side plus
the acceptance collapse — strong, but inferential, not a measured mechanism.

Also unresolved: on the keys lane the model entered a **tool-call loop**, nine consecutive rounds calling the
same two tools with valid results returning and zero content emitted. Not yet re-checked on this lane.

## Credits

- **Blackfrost-AI / Blackfrost-Research** — the DERISKED checkpoint and its NVFP4 conversion.
- **zai-org** — GLM-5.3-Flash.
- **NVIDIA** — the ModelOpt NVFP4 pack this lane's censored counterpart is built from.
