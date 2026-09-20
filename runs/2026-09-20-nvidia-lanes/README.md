# Two nvidia lanes: official weights, and an uncensored variant, both with NVFP4 attention

Both lanes start from **`nvidia/GLM-5.3-Flash-NVFP4`**, which is MIT and ungated, and apply the NVFP4 attention
work from [`../2026-09-20-tp4-vs-deepseek`](../2026-09-20-tp4-vs-deepseek). The whole chain is MIT: the zai-org
base, nvidia's quantization, and dealignai's abliteration.

| | Lane A | Lane B |
|---|---|---|
| base | `nvidia/GLM-5.3-Flash-NVFP4` | same |
| abliteration | none, refusals intact | `dealignai` `o_proj`, layers 12-44 |
| NVFP4 attention | 403 tensors, 13.04 GiB | same 403, with the transplanted `o_proj` quantized |
| role | **new default** | **uncensored default** |

## Results

Measured with the same harness, prompts and 500K window as the baseline in
[`../2026-09-20-tp4-vs-deepseek`](../2026-09-20-tp4-vs-deepseek), three measured passes after a discarded
warm-up. Differences are judged against **each cell's own measured spread**, so a delta smaller than the
noise is reported as unresolvable rather than dressed up as a win or a loss.

### Lane A vs the LibertAI baseline: it ties, and wins at concurrency

| C1 per-stream tok/s | baseline (LibertAI) | Lane A (nvidia) | verdict |
|---|---|---|---|
| code | 92.7 (1.12x) | 78.7 (1.17x) | within noise |
| JSON | 74.0 (1.10x) | 75.9 (1.04x) | within noise |
| math | 101.3 (1.08x) | 88.8 (1.09x) | -12%, marginally outside |
| prose | 43.7 (1.06x) | 40.8 (1.10x) | within noise |
| structure | 114.7 (1.05x) | 106.3 (1.15x) | within noise |
| counting | 120.0 (1.46x) | 107.7 (1.12x) | within noise |
| reasoning | 70.1 (1.37x) | 71.5 (1.15x) | within noise |
| summary | 53.9 (1.12x) | 51.5 (1.11x) | within noise |
| narrative | 33.3 (1.13x) | 38.2 (1.12x) | +15%, marginally outside |

Seven of nine within noise. The two outliers fall in **opposite directions** and only just past their spreads,
which is what nine cells produce by chance at this sample size. A two-rep interim had math, prose and structure
all near -10%; the third pass brought prose and structure back inside. That is the argument for the 3-rep
protocol in one observation.

| aggregate tok/s | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| baseline | 64.1 | 96.1 | 114.9 | 130.7 | 146.2 | 167.2 |
| Lane A | 63.2 | 93.3 | **117.0** | **140.0** | **152.9** | **172.4** |
| TTFT, Lane A | 0.200 | 0.254 | 0.288 | 0.408 | 0.439 | 0.413 |

| concurrency aggregate | C8 | C12 | C16 | C24 | C32 |
|---|---|---|---|---|---|
| baseline | 193.9 | 230.9 | 253.7 | 293.2 | 320.7 |
| Lane A | 193.1 | **241.9** | **271.1** | **315.4** | **352.8** |

Draft acceptance 0.394 against the baseline's 0.389, so the drafter is unaffected. Quality gate PASS. KV pool
3,532,196 tokens and 43.76 GiB/rank of weights, both identical to the baseline, which is the expected signature
of two builds quantizing the same module set.

**The predicted 3-5% penalty from nvidia's extra BF16 expert layer is not visible above the noise floor.** The
arithmetic is real (3.375 GiB/rank, 0.37 ms per verify token, about 2.9 ms on a 60 ms step) but too small for
this harness to resolve, and the concurrency gain runs the other way.

### Prefill: unresolved, and both measurements are contaminated

Rep 1 was taken 2 minutes after boot while the baseline's was taken at 34 minutes, and a follow-up
`probe_prefill.sh` was run during a 190 GiB shard rewrite. Neither is usable. Decode, TTFT, concurrency and
acceptance are clean. Prefill needs one quiet window on each lane and has not had it.

### Lane B vs Lane A: ties, as the arithmetic requires

Lane B differs from Lane A only in the **values** inside 33 `o_proj` tensors. Identical tensor count, identical
format, identical kernels. Weight values do not change throughput, so any measured difference is noise, and the
useful question is whether the measurement agrees.

**Concurrency is the cleanest comparison**, because aggregates over many concurrent requests average out the
per-cell scatter that dominates C1:

| aggregate tok/s | Lane A | Lane B | delta |
|---|---|---|---|
| C8 | 193.1 | 202.3 | +4.8% |
| C12 | 241.9 | 244.9 | +1.2% |
| C16 | 271.1 | 273.8 | +1.0% |
| C24 | 315.4 | 319.6 | +1.3% |
| C32 | 352.8 | 354.3 | +0.4% |
| draft acceptance | 0.394 | 0.396 | identical |

Within 1-5% across five levels. Both lanes also beat the LibertAI baseline at concurrency, Lane B by +4.3% at
C8 rising to +10.5% at C32.

**C1 per-stream, 3 clean reps each** (Lane B's rep 1 discarded, see below). Every cell within noise:

| category | Lane A | Lane B | delta | Lane B spread |
|---|---|---|---|---|
| code | 78.7 | 94.3 | +19.8% | 1.29x |
| structure | 106.3 | 89.4 | -15.9% | 1.46x |
| math | 88.8 | 78.0 | -12.2% | 1.41x |
| JSON | 75.9 | 82.8 | +9.1% | 1.26x |
| counting | 107.7 | 117.2 | +8.8% | 1.35x |
| prose | 40.8 | 43.7 | +7.1% | 1.11x |
| summary | 51.5 | 54.7 | +6.1% | 1.07x |
| reasoning | 71.5 | 67.2 | -6.0% | 1.10x |
| narrative | 38.2 | 38.0 | -0.5% | 1.04x |

**Nine of nine within noise.** Going from two clean reps to three barely moved the medians (code +23.2% to
+19.8%) but widened the spreads, which is the more useful change: it exposed scatter that was always present
and that a two-rep median had hidden.

### Cold prefill: at parity, once measured on a quiet lane

| measurement | tok/s | conditions |
|---|---|---|
| LibertAI baseline, bench cells | 1,977-2,009 | rep 1, about 34 min post-boot |
| **Lane B, quiet probe** | **1,997** | 40,659 tokens, TTFT 20.4 s, about 37 min post-boot, nothing else running |
| Lane B bench rep 1 | 926-1,575 | discarded, competing probe |
| earlier probe | 958 | discarded, taken during a 190 GiB shard rewrite |

At matched lane age on an idle machine, prefill lands **inside the baseline's range**. The alarming numbers from
rep 1 were lane age plus self-inflicted load, nothing else. The contaminated probe was off by 52%, which is a
fair measure of how much a concurrent build distorts this.


### A note on discarded data

Lane B's bench rep 1 was thrown away: a token-corruption probe was generating against the same endpoint while
it ran. An earlier `probe_prefill.sh` was also discarded for running during a 190 GiB shard rewrite. Both were
self-inflicted, and both are why `RUNBOOK.md` says to run measurements one at a time. The published Lane B
medians use only clean reps.

### Token corruption (vLLM #54150)

The repo's own earlier finding was that ModelOpt-quantized NVFP4 builds emit intermittent corrupted token IDs
(4 / 9 / 8 in a prior test) while RedHat's compressed-tensors build scored 0 / 0 / 0, and that is why RedHat had
been the default.

**Both lanes here are ModelOpt builds and score zero.** `tools/corrupt_probe.py` sends English-only prompts at
temperature 0, including a tool-call-shaped one, and counts CJK, Cyrillic, Hangul, Arabic and replacement
characters:

| prompt | chars | suspect |
|---|---|---|
| long prose | 2,901 | 0 |
| code | 4,172 | 0 |
| JSON | 2,333 | 0 |
| tool-call shaped | 997 | 0 |
| repetitive counting | 2,718 | 0 |

A second run on a quiet lane scored **0 suspect characters in 15,496 generated**, for 28,617 characters total
with zero corruption.

The likely mechanism: the corruption lives in the **W4A4 activation path**, reading `input_scale` values that
are absent or placeholders. LibertAI's own erratum describes the same symptom from the same cause. Both lanes
here declare **`W4A16_NVFP4`**, which is weight-only and never reads activation scales, so that path is not
executed. The single config line that makes the quantized attention weights loadable also closes the corruption
route.

Stated carefully: this is strong evidence, not proof. An intermittent fault needs volume, and a direct
contrast against a W4A4 lane would turn correlation into demonstrated mechanism. Neither has been done.




## Verdict

**Both lanes pass.** Lane A ties the LibertAI baseline, Lane B ties Lane A, and both beat the baseline at
concurrency. The repo's default can move to official nvidia weights at no cost, with an uncensored variant that
measures identically.

| | verdict |
|---|---|
| Lane A vs baseline, C1 categories | 7 of 9 within noise; math -12% and narrative +15% sit just past their spreads **in opposite directions** |
| Lane A vs baseline, aggregate | level at C1-C2, **ahead at C3-C6** (up to +7%) |
| Lane A vs baseline, concurrency | **ahead at every level from C12**, +4.8% to +10.0% |
| Lane B vs Lane A, C1 categories | **9 of 9 within noise** |
| Lane B vs Lane A, concurrency | within 1-5% across five levels |
| draft acceptance | 0.394 vs 0.396, identical |
| cold prefill | 1,997 tok/s, inside the baseline's 1,977-2,009 |
| quality gate | PASS on both |
| token corruption | 0 in 28,617 characters |

The predicted 3-5% penalty from nvidia's extra BF16 expert layer is real arithmetic but **not visible above the
noise floor**, and the concurrency gain runs the other way.

**What is not established here:** whether Lane B actually refuses less. That is a behavioural check, not a
performance one, and it was deliberately left to the operator. Worth knowing before you run it: NVFP4 perturbs
the weights about 9.4% in Frobenius norm while the dealignai edit is only 2-4%, so the quantization noise is
roughly three times the abliteration signal. That does not mean the edit is destroyed, since quantization error
is unstructured and the edit is directional, but it is a concrete reason to test rather than assume.

## Why nvidia rather than LibertAI or RedHat

nvidia is the only one of these packs that ships **36,297 calibrated `input_scale` tensors**. LibertAI's 27
August build carries none while declaring `NVFP4` (which means W4A4), the defect LibertAI later fixed by adding a
separate 4.6 MB scales file. That does not bite a `W4A16_NVFP4` lane, which never reads activation scales, but
nvidia needs no such argument.

nvidia also keeps **one full expert layer in BF16**: 864 tensors, 288 experts x gate/up/down at the first sparse
layer. Every other expert layer is NVFP4. That is worth roughly 3% of decode step time, since its expert pool is
43.24 GiB per rank against LibertAI's 40.82, so each verify token streams about 6% more weight. Whether that one
layer's precision is worth 3% is unmeasured; 3% is below what this harness resolves.

## Building Lane A

```bash
# 1. quantize the non-expert projections (about 70 s on one GPU)
SRC=<nvidia checkpoint> OUT=<scratch> python3 tools/mknvfp4b.py
# 2. assemble: hardlink clean shards, rewrite the ones holding superseded tensors, write index and config
SRC=<nvidia checkpoint> DST=<new dir> NEWDIR=<scratch> python3 tools/build_nv.py
```

`build_nv.py` differs from the LibertAI version in one way that matters. nvidia's `ignore` list uses **broad
per-layer wildcards**, `layers.N.self_attn*` x45 and `...shared_experts*` x42. Dropping those wholesale would
un-ignore the DSA indexer and the KDA `f_b`/`g_b` gates, which stay BF16 and are not in the new shard, and vLLM
would then look for quantized parameters that do not exist. So it drops the broad entries and adds back narrow
ones for exactly the modules that remain BF16. The ignore list goes 132 -> 56.

31 of nvidia's 33 shards hold BF16 attention, so the rewrite touches nearly the whole 190 GiB, about 230 s.

## Building Lane B

```bash
# 1. fetch only the tensors dealignai actually changed (about 2.6 GiB, not the 180 GiB pack)
NV=<nvidia checkpoint> OUT=<donor dir> python3 tools/fetch_dealign.py
# 2. quantize, substituting the donor o_proj BEFORE quantization
SRC=<nvidia> SUBST=<donor>/dealign_oproj.safetensors OUT=<scratch> python3 tools/mknvfp4b.py
# 3. assemble as above
```

**Order matters.** The `o_proj` that gets quantized must be the transplanted one, not stock. Reversed, you would
be copying BF16 tensors over NVFP4 ones, which cannot work.

`fetch_dealign.py` does not trust a documented layer range. For each `o_proj` it compares against nvidia's stock
weight and **downloads only the tensors that actually differ**, skipping the identical ones. On this run it took
layers 12-44 and skipped 0-11 and 45, which matched an independent diff run through a separate code path. If
dealignai changes their range, the script adapts rather than silently transplanting stock weights.

## Verifying your build matches

`tools/mkchecksums.py` emits a manifest of what the recipe creates: the new NVFP4 shard, the rewritten index,
the edited config, and the donor tensor file. The untouched shards come straight from nvidia and verify against
the upstream checkpoint. See `checksums-laneA.json` and `checksums-laneB.json`.

No weights are hosted here. Everything upstream is MIT and ungated, so the recipe plus checksums gets you the
same artifact while credit and traffic stay with the original authors.

## Credits

- `zai-org/GLM-5.3-Flash`, the base model
- `nvidia/GLM-5.3-Flash-NVFP4`, the quantization both lanes build on
- `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4`, the `o_proj` donor for Lane B
- `drowzeys` (Keys), whose `METHOD.md` documented the `o_proj` byte-copy approach and measured it at 32/32
  bypass where computed-direction methods reached 6-9/32

## Two traps porting this recipe to a different pack

Both cost a boot here. Neither is visible until weight loading is well underway.

**1. The fleet must reclaim memory before the next lane allocates.** Booting straight after stopping a ~100 GiB
lane races the reclaim, and vLLM refuses:

```
ValueError: Free memory on device cuda:0 (103.04/121.69 GiB) on startup is less than
desired GPU memory utilization
```

103.04 free against 0.85 x 121.69 = 103.4 wanted, short by 0.4 GiB. Workers sat at 59 GiB while the head had
113. `tools/glm_settle.sh` stops everything, drops caches, then **polls every node until the minimum
MemAvailable clears the threshold**. On the retry it reported 116 GiB and the boot went through. Needed in
either direction when swapping large lanes.

**2. Do not hand-derive the ignore list.** nvidia's uses broad per-layer wildcards
(`layers.N.self_attn*` x45, `...shared_experts*` x42), so the temptation is to write narrow replacements. Doing
that here produced an entry for `*.self_attn.in_proj_qkvbfg_a`, which is the **fused target** of the six
q/k/v/b/f_a/g_a tensors being quantized. vLLM then builds that layer BF16 and the packed NVFP4 weights have
nowhere to go:

```
glm5next/nvidia/kda.py:114  weight_loader_v2
model_executor/layers/parameter.py:175  assert param_data.shape == loaded_weight.shape
AssertionError
```

The fix is to use the list already proven to boot on this architecture, 14 entries, in `tools/fix_ignore.py`.
Same module names, same quantized set, so it transfers. 132 entries collapse to 14.

A related warning about verifying this yourself: calling `is_layer_skipped()` directly gives **misleading**
answers, because ModelOpt preprocesses the ignore list first (`modelopt.py:229-237` expands trailing-`*`
entries). A check written against the raw list reported four modules as wrongly-unskipped that were in fact
fine. Verify through the config class, or by booting.

Both fixes are config-only. The shards are unaffected, so neither needs a rebuild.
