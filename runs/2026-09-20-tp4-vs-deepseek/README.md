# GLM-5.3-Flash TP4 vs DeepSeek-V4.1-Flash TP4, 2026-09-20 (overnight, 01:50-09:30 ET)

Goal: serve GLM-5.3-Flash NVFP4 + DFlash2 at tensor-parallel 4 across all four DGX Sparks and try to beat
DeepSeek-V4.1-Flash's measured numbers **on DeepSeek's own prompt set** - code, JSON, prose, math, counting,
structure - then hand the fleet back to DeepSeek with the GLM launcher left ready for a one-line swap.

Everything here was measured with the DeepSeek harness (`v41bench.py`, `sr_quality.py`, `idletest.py`,
temperature 0, streaming, decode measured after the first token), so the two models are compared on identical
prompts rather than on each project's own benchmark.

## Headline

The recipe as it stood **loses** to DeepSeek-V4.1-Flash by 10 to 16% on all six named categories. Finding out
why produced a per-step cost model, the model pointed at 18 GiB of bf16 weights nobody had quantized, and
quantizing them turned the result around.

**Final build: `keys-glm53-nvfp4-attn3` + `NVFP4_PATCH=1`, 500K context, medians of 3 measured passes on the
serving lane.**

| C1 per-stream tok/s | DeepSeek | **GLM NVFP4** | delta | spread over 3 passes |
|---|---|---|---|---|
| summary | 41.1 | **53.9** | **+31.1%** | 1.12x |
| structure | 98.6 | **114.7** | **+16.3%** | 1.05x |
| math | 87.5 | **101.3** | **+15.8%** | 1.08x |
| prose | 39.0 | **43.7** | **+12.2%** | 1.06x |
| counting | 113.0 | **120.0** | **+6.2%** | 1.46x |
| narrative | 31.4 | **33.3** | **+6.0%** | 1.13x |
| code | 90.7 | **92.7** | **+2.1%** | 1.12x |
| JSON | 75.8 | 74.0 | -2.4% | 1.10x |
| reasoning | 75.1 | 70.1 | -6.6% | 1.37x |

**GLM wins 7 of 9, and 5 of the 6 named categories.** Only JSON loses among the named six, by 2.4%, which is
inside its own 1.10x spread. Counting and reasoning carry the widest spreads (1.46x, 1.37x) and should be read
as the least settled cells.

**Concurrency is still DeepSeek's**, and that is the honest limit of this result:

| aggregate tok/s | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| DeepSeek | 61.3 | 102.3 | 128.9 | 153.4 | 174.3 | 189.3 |
| GLM NVFP4 | **64.1** | 96.1 | 114.9 | 130.7 | 146.2 | 167.2 |
| delta | **+5%** | -6% | -11% | -15% | -16% | -12% |
| GLM TTFT | 0.199 s | 0.236 | 0.249 | 0.269 | 0.382 | 0.403 |

The crossover is immediately after C1: GLM takes single stream, DeepSeek takes every concurrency level by a
fairly stable 6 to 16%.

**Cold prefill is flat and at parity**, rep 1 only since later passes hit the prefix cache:

| target | prompt tokens | tok/s | TTFT |
|---|---|---|---|
| 2K | 3,814 | 1,979 | 1.9 s |
| 8K | 15,168 | 2,009 | 7.5 s |
| 32K | 60,917 | 1,998 | 30.5 s |
| 64K | 121,681 | 1,977 | 61.6 s |

Roughly 2,000 tok/s from 3.8K to 121.7K tokens, against DeepSeek's 1,939 to 2,048. Note GLM tokenizes the same
prompt text into about 30% more tokens, so wall-clock TTFT on identical input is still worse.

**Long context: needles 5 of 5 exact** at 65K, 131K and 262K (depth 0.3) and 131K and 450K (depth 0.6). Quality
gate PASS 5/5. KV pool 3,532,196 tokens, 7.06x at full context, fp8_e4m3 pinned 24 GiB.

Four results underneath that are worth more than the config:

1. **A validated per-step cost model.** `step_ms = 34.6 + 4.12 x verify_tokens`, fitted across four draft lengths
   with residuals under 2.2 ms. Its slope matches the checkpoint's routed-expert bytes to 8%; its intercept is
   explained to within 0.8 ms by **17.7 ms of bf16 non-expert weights read on every step**. It predicted x1.179
   from quantizing them before such a build existed.
2. **The blocker was two lines of the model, not the checkpoint.** `kda.py:172` and `model.py:331` force
   `quant_config=None` on the attention projections, so a packed NVFP4 weight cannot load into a bf16 parameter.
3. **A plus-or-minus 25% run-to-run band** on single passes, which invalidated early apparent wins and later
   corrected this build's own numbers in both directions.
4. **A W4A4 correctness landmine** in this checkpoint class, with a one-line fix the winning build needs anyway.

Earlier revisions of this document quoted a 1M-context lane at C1 and C3 only. Those numbers are superseded by
the table above, which is the lane that actually serves.

## What was already known, and what this run adds

The TP4 recipe in this repo (seqs 64, mnbt 16384, block 2304, marlin, DFlash2 k=7, fp8 KV pinned 24 GiB,
FULL_AND_PIECEWISE) came from 2026-08-31/09-02. The 2026-09-18 TP2 night then shipped two levers that had
never run at TP4: the **b12x RoCEnante one-shot all-reduce** and the **#18 prefix-cache repair**. Both are live
in every boot here (`RoCEnante all-reduce is live: first routed all-reduce is 2097152 bytes`).

New in this run:
1. Two boot traps that any 1M-context TP4 deployment of this recipe will hit (below).
2. A source-level audit of the NVFP4 MoE backend selection that found a **correctness landmine** in this
   checkpoint class, and corrected a plausible-but-wrong performance theory of my own.
3. The noise measurement, and a 3-rep median harness (`glm_screen3.sh`) that publishes the spread.

## Two boot traps (fixed in `glm53_tp4.sh`)

**1. `$HOME` is `/root` on the workers.** The boot script launches each worker rank as root, so the launcher's
`$HOME/patches/...` prereq checks looked in `/root` while `sparse_attn_indexer_kpool.py`,
`kv_cache_coordinator.py` and `glm-roce/` live in each node's `tonyspark` home. Fixed with an explicit
per-rank `PATCH_HOME`. The launcher also has to exist on every node, not just the head.

**2. The multimodal warmup builds a dummy VIDEO sized to `max_model_len`.**
`renderers/base.py:_warmup_mm_processor` calls
`get_dummy_processor_inputs(seq_len=model_config.max_model_len, ...)` for every modality with a limit above
zero. At 1,048,576 context that is an enormous synthetic video normalized on CPU through torchvision,
single-threaded because serving has already dropped torch to one thread. py-spy on the API server showed it
parked in `normalize_image -> rescale_and_normalize -> glm5next._preprocess -> video_processing_utils` for
eight minutes with `/health` still refusing connections and the engine fully up behind it (engine init 331 s,
graphs captured, RoCE live). **Fix: `--limit-mm-per-prompt {"image":4,"video":0}`** - image support kept, video
warmup skipped. It also matches DeepSeek's mm config, which makes the comparison fairer. This is now the
launcher's **default** value of `VLLM_EXTRA`, because a swap back to GLM that forgot it would look like a
fourteen-minute boot turning into a twenty-two-minute one for no visible reason. If you set `VLLM_EXTRA`, add
to it rather than replacing it.

## The correctness landmine: do not run a W4A4 MoE backend on this checkpoint

While chasing decode time I proposed switching off `--moe-backend marlin`, reasoning from this boot line:

```
nvfp4.py:244  Using 'MARLIN' NvFp4 MoE backend out of potential backends: [...]
marlin_utils_fp4.py:354  WARNING Your GPU does not have native support for FP4 computation ...
```

**That reasoning was wrong, and acting on it would have served wrong output.** A source audit established:

- `marlin_utils_fp4.py:354` is an **unconditional** `logger.warning_once`, not a capability check. It prints on
  any GPU whenever the Marlin NVFP4 MoE path is used.
- **Marlin moves no extra bytes.** `_repack_marlin_experts` keeps 4-bit packing and
  `nvfp4_marlin_process_scales` ends at `.view(torch.float8_e4m3fn)`, so group-16 scales stay one byte:
  4.5 bits/weight, the same as a native FP4 path. The dequant to bf16 happens in registers. Marlin costs MMA
  throughput, which only matters when compute-bound.
- **This checkpoint declares W4A4 but ships no activation scales.** `model.safetensors.index.json` contains
  **zero** `input_scale` tensors and `config.json` has `"input_activations": null`, yet `modelopt.py:1392`
  sets `activation_key = kNvfp4Dynamic` because `quant_algo` is `"NVFP4"` rather than `"W4A16_NVFP4"`. vLLM
  then creates `w13_input_scale`/`w2_input_scale` with `torch.empty` and the loader marks them loaded, so
  nothing raises. Every W4A4 backend multiplies real weight scales by that **uninitialized memory**
  (`flashinfer_cutlass_moe.py:67-69`, `cutlass_moe.py:699-700`, then `nvfp4.py:526` takes `1.0/a13_scale`).
  Marlin escapes only because `nvfp4.py:406-407` discards those scales.

Our `flashinfer_cutlass` boot died in engine init, which was the lucky outcome: had it survived, it would have
served *plausible* text from corrupted weight scales, and a quality gate checking counting, JSON, arithmetic,
code and prose can pass that. **`flashinfer_cutlass` and `vllm_cutlass` are excluded on correctness.**

Free hardening for anyone running a ModelOpt GLM-5.3 pack: set `"quant_algo": "W4A16_NVFP4"` in its
`config.json`. Zero speed effect; makes it impossible for any W4A4 kernel to be selected against weights that
have no activation scales.

Also established: `FLASHINFER_TRTLLM` and `FLASHINFER_CUTEDSL` are hard-excluded on SM121 by
`is_device_capability_family(100)`; `swiglu_limit=10.0` in the config filters the candidate list to the
clamp-capable backends; `flashinfer_b12x` is the only native-FP4 MoE path on SM121 and is blocked solely by
that clamp filter, and it silently drops the clamp, so it needs a logprob comparison rather than a pass/fail
gate. No env var in this build reaches a native FP4 path.

## How these two models can and cannot be compared on one harness

Same prompts, same harness, temperature 0. Two things still need stating before the tables.

**Prompt tokenization differs by about 30%.** The identical prefill text is 3,814 GLM tokens against 2,950
DeepSeek tokens at the 2K target, and 121,681 against 93,335 at the 64K target. Prefill measured in tok/s
therefore flatters GLM, because it is credited for tokens the other model never had to process. The tables
below give prefill tok/s *and* wall-clock TTFT on the same text, and TTFT is the number a user feels.

**Output tokenization is close**, so decode tok/s is roughly comparable: characters per generated token are
3.35 GLM against 3.15 DeepSeek on code, 4.24 against 4.83 on prose. A chars/s table is published anyway, since
it is the only throughput metric that does not depend on either vocabulary.

## Measured properly

Single-run cells on this lane are not resolvable: **g00 and g09 are the same configuration** and differ by
code -23%, JSON -24%, counting -28%. Two repeats of k=9 differ by code +17%, counting -23%. So both lanes were
measured with the same protocol - `glm_screen3.sh` and `ds_screen3.sh`: one discarded warm-up pass, three
measured passes, per-cell medians, and the max/min spread published next to them. `mktable.py` builds the
tables below straight from those JSONs.

Only **rep 1's prefill cells are cold**. This build has no prefix-cache reset endpoint, so reps 2 and 3 hit
the cache and report fewer prompt tokens than they asked for (8,000 -> 7,622). Prefill medians across reps are
meaningless; every prefill number published here is rep 1.

| C1 per-stream tok/s | DeepSeek-V4.1-Flash | GLM boot 1 | GLM boot 2 | GLM mean | GLM vs DeepSeek |
|---|---|---|---|---|---|
| **code** | **90.7** | 77.4 | 74.9 | 76.2 | **-16%** |
| **JSON** | **75.8** | 68.9 | 66.8 | 67.9 | **-10%** |
| **math** | **87.5** | 79.4 | 69.3 | 74.4 | **-15%** |
| **prose** | **39.0** | 32.8 | 33.8 | 33.3 | **-15%** |
| **counting** | **113.0** | 96.8 | 94.7 | 95.8 | **-15%** |
| **structure** | **98.6** | 91.1 | 86.8 | 89.0 | **-10%** |

| aggregate tok/s | DeepSeek | GLM boot 1 | GLM boot 2 | GLM vs DeepSeek |
|---|---|---|---|---|
| C1 | 61.3 | 54.5 | 52.6 | -13% |
| C3 | 128.9 | 96.3 | 101.6 | -23% |
| C6 | 189.3 | 143.3 | 144.8 | -24% |

| concurrency, coding aggregate | DeepSeek | GLM k=7 | GLM k=9 |
|---|---|---|---|
| C8 | **376** | 261 | 208 |
| C12 | **475** | 289 | 283 |
| C16 | cannot serve (max-num-seqs 16) | 332 | 269 |
| C24 | cannot serve | 418 | 294 |
| C32 | cannot serve | **473** | 311 |

DeepSeek's lane refuses more than 16 concurrent streams. GLM serves 32, and at C32 its coding aggregate of 473
finally matches DeepSeek's C12 peak of 475. That is the one axis where GLM has something DeepSeek does not, and
it is a statement about how many streams a lane can hold, not about speed.

| cold prefill (rep 1, the only uncached pass) | DeepSeek | GLM | note |
|---|---|---|---|
| ~60,900 GLM tok / 46,810 DS tok | 2,007 | 1,858-2,014 | near parity in tok/s |
| ~121,700 GLM tok / 93,335 DS tok | 2,048 | 1,748-1,803 | -12% to -15% |

Prefill is close in tok/s, but GLM tokenizes the same prompt text into about 30% more tokens, so wall-clock TTFT
is worse than the tok/s comparison suggests. Both are published because only one of them is what a user feels.

**Quality gate: PASS on both lanes, 7/7, vision and tools included.**



## Verdict

**GLM-5.3-Flash TP4 beats DeepSeek-V4.1-Flash on four of the six named categories once its bf16 attention and
MLP projections are quantized to NVFP4**, and is at parity on C1 aggregate. It loses code and JSON on tok/s by
about 6%, though code wins on chars/s. DeepSeek keeps concurrency: C3 aggregate is still its by 13%.

The honest shape of the night is that **no configuration knob did this**. Every knob tried landed at or below the
first baseline boot, inside a plus-or-minus 25% noise band. What did it was measuring where the step time goes,
finding 17.7 ms of it in weights nobody had quantized, and then discovering the obstacle was two lines of the
model implementation rather than anything about the checkpoint.

**An interaction worth knowing before trusting this build:** the abliteration in this pack lives entirely in
`layers.{15..45}.self_attn.o_proj.weight`, and `o_proj` is the largest single component of what the NVFP4 build
quantizes (3.625 GiB of 13.88). So g18 stacks roughly 2-3% per-tensor quantization error on top of the very
transplant that makes the checkpoint uncensored, across 31 of 45 layers. Two things follow. The alignment
behaviour has **not** been re-tested on g18 - the quality gate covers counting, JSON, code, math and prose, none
of which probe refusal - so it should not be assumed unchanged in either direction. And this is a more specific
suspect for the 131K needle miss than "attention was quantized": those same layers already carry a 12.6%
perturbation before quantization touches them. Excluding `o_proj` for layers 15-45 costs about 2.8 ms of the
10.4 ms saved and is the cleanest way to separate the two effects.

Serve-readiness: **not yet, for long-context work.** One of four needles dropped a digit at 131K depth 0.3 while
65K, 98K and 131K depth 0.6 were exact. A 5/5 short-prompt gate cannot clear a numerics change, and this run has
no bf16 needle at those lengths for comparison. The next session should run that baseline, and if the regression
is real, try excluding `q_proj`/`k_proj` (1.06 GiB of the 13.88, worth about 0.8 ms of the 10.4 ms saved) since
those feed attention scores directly.

## Where the step time goes

Geometry from `config.json`: hidden 4096, `moe_intermediate_size` 2048 (512/rank at TP4), 288 routed experts,
top-8, 42 of 45 layers sparse. Routed MoE is 163.27 GiB, or 40.8 GiB/rank.

Fitting the step times the harness records at four draft lengths, against verify tokens t:

| k | verify tokens | measured step ms p50 | fit |
|---|---|---|---|
| none | 1 | 37.7 | 38.7 |
| 5 | 6 | 61.5 | 59.3 |
| 7 | 8 | 67.5 | 67.5 |
| 9 | 10 | 74.5 | 75.7 |

**step_ms = 34.6 + 4.12 x verify_tokens**, residuals within 2.2 ms across a 2x range.

The slope is the routed experts, and it checks out independently: top-8 of 288 is 8/288 x 40.8 GiB/rank =
1.134 GiB = **4.46 ms** predicted at 273 GB/s against 4.12 measured, correctly a little under because tokens in
one verify batch share experts.

The intercept was the finding. Summing the checkpoint by module class and dtype:

| module class | GiB | dtype |
|---|---|---|
| routed experts | 163.27 | U8 145.12 + F8_E4M3 18.14 |
| **attention** | **11.70** | **BF16** |
| dense MLP | 3.52 | BF16 |
| embeddings + head | 2.37 | BF16 |
| other | 0.43 | BF16 |

**18.01 GiB of non-expert weights, all bf16, read on every step** regardless of how many tokens are in flight:
4.50 GiB/rank = **17.7 ms**, which accounts for the intercept and leaves 16.9 ms of genuine non-weight overhead
(drafter forward, aux hidden-state extraction from 5 target layers, 45 layers of all-reduce, KV, kernel launch,
host bubble). Quantizing 13.88 GiB of those is what the final build does, and the model's x1.179 prediction held.

Speculation is not the lever. With no drafter at all **every category pins at 26.3 tok/s**. Measured acceptance
at k=9, cumulative by draft position, is 74.9 / 53.7 / 40.1 / 31.5 / 26.0 / 21.6 / 18.2 / 14.9 / 12.5%, with
conditional survival flat near 0.8 after the first position, so tokens per step is `1 + sum(cumulative)` and the
tail is geometric: k=9 buys +7% tokens per step for 29% more verify work and 9% less KV. And **k=7 is not a tuned
value** - the drafter's config says `block_size: 8`, so k=7 fills exactly one diffusion block and k=9 spills into
a second.

## What is left

1. **Settle the long-context question.** Run the bf16 needle at 65K/98K/131K to establish whether the one dropped
   digit at 131K depth 0.3 is a regression from NVFP4 or a pre-existing property of the lane. If it is a
   regression, exclude `q_proj`/`k_proj` (1.06 GiB of the 13.88, about 0.8 ms of the 10.4 ms saved) since those
   feed attention scores directly, and re-test.
2. **The 16.9 ms of non-weight overhead is now the largest single unexplained term** in the step. Halving it would
   be worth about another x1.15. It needs a per-step profile, not arithmetic.
3. **Concurrency.** C3 aggregate is still DeepSeek's by 13%, and GLM flattens between C12 and C16. Whether the
   NVFP4 build changes the concurrency curve was not measured; it should be.
4. **Upstream the two lines.** `quant_config=None` in `glm5next/nvidia` is correct for a checkpoint whose
   projections are bf16 and wrong for one whose are not. It should key off the checkpoint's own `ignore` list
   rather than being hardcoded, which is what `is_layer_skipped()` already does correctly per layer.

## The NVFP4 build: five boots, and what each failure taught

Having found that 18.01 GiB of non-expert weights are bf16 and are read on every step, the obvious move was to
quantize them. Four boots, none of which served, and the failures are the finding:

| boot | quantized | failed at | cause |
|---|---|---|---|
| g14 | 13.98 GiB, 334 tensors | `parameter.py:175` | stale bf16 duplicates: the safetensors index filters by **file**, not by tensor name, so hardlinking the originals left the old copies readable. Also only 3 of 6 fused constituents |
| g15 | same, fused groups fixed | `parameter.py:175` | same duplicates, masking the fix |
| g16 | 13.88 GiB, 412 tensors, duplicates removed | `kda.py:114` | KDA loads `f_a`/`g_a` as **replicated** shards with `output_size *= tp_size` and `tp_rank` forced to 0; that does not line up with packed NVFP4 parameters |
| g17 | 7.42 GiB, KDA group excluded | `model.py:917` `KeyError: o_proj.weight_scale` | **the model builds those projections with `quant_config=None`** |

g17's `KeyError` finally pointed at the constructor rather than the loader, and the cause is in the
image's own source:

```
kda.py:172     vllm_config.quant_config = None
model.py:331   quant_config=None,  # MLA projections are BF16 in checkpoint
model.py:1090  quant_config=None,
```

GLM-5.3-Flash's implementation hardcodes its attention projections as bf16: KDA strips the quant config for its
whole submodule tree, and the MLA path passes `quant_config=None` with a comment asserting the checkpoint is bf16
there. So a packed NVFP4 `(out, in/2)` weight could never load into the resulting bf16 `(out, in)` parameter -
which is the single cause of both g16 and g17, and which I misread as a KDA loader quirk for three boots.

**g18 fixed it with two bind-mounted lines**, using the same patch mechanism this lane already uses for
`sparse_attn_indexer_kpool.py`:

```
kda.py:172    vllm_config.quant_config = None       -> removed
model.py:331  quant_config=None                     -> quant_config=vllm_config.quant_config
```

Left alone on purpose, because they really are bf16 in every pack here and are in the checkpoint's `ignore` list:
`model.py:1090` (vision tower - quantizing it yields NaN image features) and `attention.py:263` (indexer
`wk_weights_proj`). `is_layer_skipped()` honours the ignore list per layer, so `indexer.*`, `f_b`/`g_b`,
`q_a`/`kv_a` and `eh_proj` stayed bf16.

It booted in 841 s, reported `quant_algo=W4A16_NVFP4` and `MarlinNvFp4LinearKernel`, loaded **43.76 GiB/rank**,
kept the 3,895,606-token KV pool, and passed the quality gate 5/5. It is wired behind an opt-in `NVFP4_PATCH=1`
so the default lane is unchanged, and the whole configuration is one line:

```bash
MODEL_DIR=keys-glm53-nvfp4-attn3 NVFP4_PATCH=1 MNBT=8192 SPEC_K=7 bash /root/glm_boot.sh <label>
```

One free win came out of it regardless: selecting `W4A16_NVFP4` instead of `NVFP4` picks
`ModelOptNvFp4W4A16LinearMethod`, which needs no `input_scale` and therefore **disarms the W4A4 landmine**
described above at no cost.

## Tools

All in [`tools/`](tools/), all run as root on the head.

| script | what it does |
|---|---|
| `glm53_tp4.sh` | the launcher: per-rank NFS model path, every lever an env knob, RoCE and prefix fix on by default, RoCEv2 GID lookup, prereq checks that fail loudly instead of letting Docker create empty dirs over them |
| `glm_boot.sh` | worker-first 3->2->1->0 with 20 s gaps, head last, polls `/health` (never `/v1/models`, which 200s from config with a dead engine behind it) and saves the head log before the container is reaped |
| `glm_screen3.sh` / `ds_screen3.sh` | the same 3-rep median protocol on each lane |
| `glm_hiconc.sh` / `ds_hiconc.sh` | the C8/C12/C16 sweep on each lane |
| `mktable.py` | builds the comparison tables above from both lanes' JSONs, including chars/s and wall-clock TTFT |
| `glm_final.sh`, `ds_final.sh` | boot plus the full measurement in one step |
| `glm_run.sh`, `glm_dq.sh`, `dist_glm.sh` | single cycle, dynamic queue with a STOP_AT and a two-failure stop, launcher distribution |

[`SWAP.md`](SWAP.md) is the operator page: one command each way between the two models, the knob table, and the
two traps that break a swap.

Weights reach ranks 1-3 over NFS from the head, because Spark4 has 51 GB free and Asusi 42 GB against a 182 GB
checkpoint.

## Harness lesson worth repeating

Three boots were lost to my own concurrent operations: twice by a `pkill` pattern that also matched the ssh
command carrying it, and once by redistributing `glm_boot.sh` while it was executing - **bash reads a script
incrementally, so overwriting a running script corrupts it mid-run** (`line 25: syntax error near unexpected
token 'done'`). Harness edits belong between experiments, never during one.

## Credits

- GLM-5.3-Flash by [zai-org](https://huggingface.co/zai-org/GLM-5.3-Flash); DFlash2 drafter by
  [incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
- The checkpoint served here is `keys-glm-5.3-flash-nvfp4-ablit-l15-45-anchorstock`, whose own `ABLIT_META.json`
  records: method `dealign-oproj-transplant`, parent **LibertAIDAI/GLM-5.3-Flash-NVFP4**, donor
  **dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4**, 31 `layers.{15..45}.self_attn.o_proj.weight` tensors
  transplanted (mean relative Frobenius 0.126), layers 0-14 left stock as a safety anchor, experts NVFP4
  passthrough.
- The TP4 recipe, patched image and `sparse_attn_indexer_kpool` SM121 fix are this repo's prior work; the RoCE
  and prefix-cache levers came from the 2026-09-18 TP2 night in the sibling repo.
- b12x RoCEnante: @original-el8 and @lukealonso (local-inference-lab/b12x).
- DeepSeek-V4.1-Flash target numbers: `tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark`, run 2026-09-19.
