# GLM-5.3-Flash TP4 vs DeepSeek-V4.1-Flash TP4, 2026-09-20 (overnight, 01:50-09:30 ET)

Goal: serve GLM-5.3-Flash NVFP4 + DFlash2 at tensor-parallel 4 across all four DGX Sparks and try to beat
DeepSeek-V4.1-Flash's measured numbers **on DeepSeek's own prompt set** - code, JSON, prose, math, counting,
structure - then hand the fleet back to DeepSeek with the GLM launcher left ready for a one-line swap.

Everything here was measured with the DeepSeek harness (`v41bench.py`, `sr_quality.py`, `idletest.py`,
temperature 0, streaming, decode measured after the first token), so the two models are compared on identical
prompts rather than on each project's own benchmark.

## Headline

GLM-5.3-Flash TP4 **serves cleanly at 1M context with a 3.9M-token KV pool** and passes the quality gate, but it
**does not beat DeepSeek-V4.1-Flash on DeepSeek's prompt set**. Measured as medians of three passes on two
independent boots, it loses all six named categories by 10 to 16% and the C6 aggregate by 24%.

Four results from the night are worth more than the config that came out of it:

1. **A validated per-step cost model.** `step_ms = 34.6 + 4.12 x verify_tokens`, fitted across four draft
   lengths with residuals under 2.2 ms. Its slope matches the checkpoint's routed-expert bytes to within 8%, and
   its intercept is explained to within 0.8 ms by **17.7 ms of bf16 non-expert weights read on every step**.
2. **A quantified, actionable bound.** NVFP4 on 13.88 GiB of those weights is worth **x1.179 on every
   category**, which would put five of the six at or above DeepSeek. Four boots established that the blocker is
   three lines of `quant_config=None` in `glm5next/nvidia/`, not the checkpoint or the quantization.
3. **A plus-or-minus 25% run-to-run band on single passes**, which invalidated the counting and structure "wins"
   the first boots showed, and a 3-rep median protocol that reproduces across boots to within 5% on eight of
   nine categories.
4. **A correctness landmine** in this checkpoint class that would have served plausible-but-wrong text, with a
   one-line fix.

Also measured: k=7 is not a tuned value, it is the DFlash2 drafter's `block_size 8` minus one; draft acceptance
decays geometrically at about 0.8 per position, which caps speculation as a lever; and GLM serves 32 concurrent
streams where DeepSeek's lane refuses more than 16.

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

**GLM-5.3-Flash TP4 did not beat DeepSeek-V4.1-Flash on DeepSeek's prompts.** It loses all six named
categories by 10 to 16%, C6 aggregate by 24%, and the concurrency sweep by 40 to 68% up to C16. It wins on
context (1M against 500K, a 3.9M-token KV pool), on vision, and on how many streams one lane can hold.

The gap is not a tuning gap. It is 4.5 bits per weight against EXL3's 3.5, plus 18 GiB of bf16 non-expert
weights that this model's own implementation hardcodes as bf16. Both numbers are measured, and the second one is
worth x1.179 to whoever removes three lines of `quant_config=None` from `glm5next/nvidia/`.

## Why decode cannot be tuned past DeepSeek here

Geometry from `config.json`: hidden 4096, `moe_intermediate_size` 2048 (512/rank at TP4), 288 routed experts,
top-8, 42 of 45 layers sparse. Routed MoE is 304.4B params = 171.2 GB = **42.8 GB/rank**. At one sequence with
k=9 the union of experts touched is at most 80 of 288, so a step streams **~11.9 GB/rank**; at GB10's
273 GB/s that is **~43 ms**, which is the ~66 ms step measured end to end. Decode is bandwidth-bound by 7x at
64 sequences and by more than 100x at one.

Both ends of the speculation range confirm it. With no drafter at all, **every category pins at 26.3 tok/s** -
the bare single-token step (~38 ms), content-independent. Ten tokens per step costs only ~66 ms, so extra
draft tokens are cheap; what limits code and prose is **acceptance**, not step time. Code accepts 5.1 of 8 and
would need ~6.9 to reach DeepSeek's 90.7; prose accepts 1-2. No configuration knob raises acceptance, and no
MoE backend changes the bytes. A lower-bit GLM-5.3-Flash checkpoint would, and none exists on this fleet (the
378 GB `GLM-5.3-Int4-Int8Mix` on disk is a different model: `GlmMoeDsaForCausalLM`, hidden 6144, 78 layers).

## What would actually close the gap

Not a config. The three things that would, in the order they are worth trying:

1. **A lower-bit GLM-5.3-Flash checkpoint.** Decode here streams 42.8 GB/rank of routed MoE weights at 4.5
   bits each. DeepSeek's lane is EXL3 3.5 bpw and resident at 56.6 GiB/rank against NVFP4's ~81. On a
   273 GB/s part, bits per weight is the number that sets decode speed, and it is the one number no launcher
   flag can change.
2. **Higher draft acceptance on code and prose.** Ten draft tokens per step cost only ~66 ms against ~38 ms
   for one, so drafting is cheap here; what limits code is that 5.1 of 8 draft tokens survive. A drafter
   trained or tuned on code and prose would convert directly into tok/s, which no amount of `num_speculative_tokens`
   will.
3. **A native FP4 MoE path on SM121.** `flashinfer_b12x` is the only one, and it is excluded here solely
   because it silently drops the `swiglu_limit=10.0` clamp this checkpoint declares. Validating it against
   Marlin logprobs, rather than against a pass/fail quality gate, is the honest way in. It buys MMA
   throughput, which matters at high concurrency, not at one stream.

## The NVFP4 attempt: four boots, and a bound worth more than the tuning

Having found that 18.01 GiB of non-expert weights are bf16 and are read on every step, the obvious move was to
quantize them. Four boots, none of which served, and the failures are the finding:

| boot | quantized | failed at | cause |
|---|---|---|---|
| g14 | 13.98 GiB, 334 tensors | `parameter.py:175` | stale bf16 duplicates: the safetensors index filters by **file**, not by tensor name, so hardlinking the originals left the old copies readable. Also only 3 of 6 fused constituents |
| g15 | same, fused groups fixed | `parameter.py:175` | same duplicates, masking the fix |
| g16 | 13.88 GiB, 412 tensors, duplicates removed | `kda.py:114` | KDA loads `f_a`/`g_a` as **replicated** shards with `output_size *= tp_size` and `tp_rank` forced to 0; that does not line up with packed NVFP4 parameters |
| g17 | 7.42 GiB, KDA group excluded | `model.py:917` `KeyError: o_proj.weight_scale` | **the model builds those projections with `quant_config=None`** |

The last one is conclusive and is in the image's own source:

```
kda.py:172     vllm_config.quant_config = None
model.py:331   quant_config=None,  # MLA projections are BF16 in checkpoint
model.py:1090  quant_config=None,
```

GLM-5.3-Flash's implementation hardcodes its attention projections as bf16. The checkpoint's `ignore` list was
never the gate, so no repack can reach those weights. What it needs is a source change, and the value of making
it is quantified: **x1.179 on every category**, which is code 89.8, JSON 80.0, math 87.7, prose 39.2, counting
112.9 and structure 104.9 against DeepSeek's 90.7 / 75.8 / 87.5 / 39.0 / 113.0 / 98.6 - five of the six at or
above it.

The conversion tooling is verified and reusable. `tools/mknvfp4b.py` quantizes in fused groups with one shared
global scale per group and `weight_scale_2` as shape `(1,)`, which is what `PerTensorScaleParameter` and the
`torch.unique(weight_scale_2).numel() != 1` check in `ModelOptNvFp4W4A16LinearMethod` require. Its format was
validated **byte-for-byte against this checkpoint's own expert tensors** before any boot:
`scaled_fp4_quant(w, 448*6/amax, is_sf_swizzled_layout=False)` reproduced the stored packed weights exactly and
the stored fp8 block scales at 1.0000 identical, with `weight_scale_2 = amax/2688` matching to every digit. The
swizzled layout matched only 12% of scale bytes, so that flag is not optional. `tools/build4.py` assembles a dir
by hardlinking clean shards and rewriting only the ones that mix superseded tensors with tensors still needed,
then verifies zero duplicate names, zero missing indexed names and zero index entries pointing at a file that
lacks the tensor - the three checks whose absence cost the first two boots.

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
  [incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2); the NVFP4 abliterated pack used here is the
  keys build.
- The TP4 recipe, patched image and `sparse_attn_indexer_kpool` SM121 fix are this repo's prior work; the RoCE
  and prefix-cache levers came from the 2026-09-18 TP2 night in the sibling repo.
- b12x RoCEnante: @original-el8 and @lukealonso (local-inference-lab/b12x).
- DeepSeek-V4.1-Flash target numbers: `tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark`, run 2026-09-19.
