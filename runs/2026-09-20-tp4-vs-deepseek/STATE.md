# GLM-5.3-Flash TP4 vs DeepSeek-V4.1-Flash TP4, 2026-09-20. STATE (read first after compaction).

## Goal (Tony's /goal, 05:48 UTC)
Serve GLM-5.3-Flash NVFP4 + DFlash2 at TP4 on all four Sparks (DeepSeek V4.1 Flash torn down) and
**beat DeepSeek's numbers on DeepSeek's own prompts**: code, JSON, **prose (important)**, math, counting,
structure. Document here. **At 13:30 UTC (9:30 AM ET) bring DeepSeek back** and leave the GLM launcher
ready for an easy swap.

## The target: DeepSeek V4.1 Flash TP4, measured 2026-09-19 (same harness, same prompts, temp 0)
Final config `sr2-e12-final500k-s16-go.sh` (EXL3 3.5bpw, DSpark k=5, 500K, SEQS 16).

| metric | DeepSeek (to beat) |
|---|---|
| C1 per-stream: code / JSON / math / prose / counting / structure(format) | **90.7 / 75.8 / 87.5 / 39.0 / 113.0 / 98.6** |
| aggregate C1 / C2 / C3 / C4 / C5 / C6 | **61.3 / 102.3 / 128.9 / 153.4 / 174.3 / 189.3** |
| aggregate C8 / C12 (coding) | **376.4 / 474.7** |
| TTFT C1 / C6 | **0.205 / 0.381 s** |
| cold prefill 2.9K / 11.6K / 46.8K / 93K | **1,939 / 1,878 / 2,007 / 2,048 tok/s** |
| idle count / code | **117.3 / 95.3** |
| KV pool | 4,011,085 (reported) |

GLM's own last-measured numbers, different harness/prompts, for orientation only: TP4 2026-09-02 =
105.6 count-to-100, 77.3 code, 31.5 prose, 530 agg @C48, 3,834,498 fp8 KV pool, 1M context.
TP2 2026-09-18 healthy fleet = 44.7 code, 46.2 math, 42.8 json, 18.1 prose.

**Everything tonight is measured with the DeepSeek harness** (`/root/v41bench.py`, `sr_quality.py`,
`idletest.py`, `probe_prefill.sh`, all now honouring `BENCH_MODEL`), so the comparison is apples to apples.

## Fleet and topology
Head Reddie (rank 0, 192.168.192.2, serves :8000), Spark4 rank 1 (.4), Asusi rank 2 (.3), Bluey rank 3 (.1).
Weights `keys-glm-5.3-flash-nvfp4-ablit-l15-45-anchorstock` (182G, abliterated, ModelOpt NVFP4) live on
Reddie and Bluey; **ranks 1-3 read them over NFS at /mnt/reddie-models** because Spark4 has 51 GB free and
Asusi 42 GB. Drafter `GLM-5.3-Flash-DFlash2` (2.2G) is local on all four. Image
`ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` on all four. `~/patches/sparse_attn_indexer_kpool.py`,
`~/patches/kv_cache_coordinator.py` (#18 prefix fix) and `~/patches/glm-roce/` (b12x) staged on all four.
- Corruption caveat: this is a ModelOpt build (vLLM #54150, intermittent bad token IDs, mostly invisible in
  English, breaks tool-call parsing). RedHatAI compressed-tensors is the repo's clean default but Kai deleted
  it on 09-17 and Reddie has only 157 GB free. Quality gate will tell us if it matters for these prompts.

## Tooling written tonight (root on Reddie)
- `glm53_tp4.sh <rank>` - the launcher: per-rank NFS path resolution, every lever an env knob
  (IMAGE MODEL_DIR GMU MAXLEN SEQS MNBT BLOCK MOE_BACKEND SPEC_K KV_MEM KV_DTYPE CG ROCE PREFIX_FIX
  VLLM_EXTRA NCCL_EXTRA), RoCE + prefix-fix mounts on by default, GID lookup, prereq checks.
- `glm_boot.sh <label>` - stop all, drop caches, worker-first 3->2->1->0, poll **/health** (never
  /v1/models: it 200s from config with a dead engine behind it).
- `glm_screen.sh <label> [levels] [prefill]` - the DeepSeek SCREEN protocol against GLM.
- Restore DeepSeek: `bash /root/sr_boot.sh sr2-e12-final500k-s16-go.sh <label>`.

## Log
- 05:53 DeepSeek torn down (liveness stopped). GLM TP4 baseline g00 booting on the repo default:
  seqs 64, mnbt 16384, block 2304, moe marlin, dflash k=7, fp8 KV pinned 24 GiB, gmu 0.85, 1M context,
  FULL_AND_PIECEWISE, plus RoCE and the #18 prefix fix (neither was in the TP4 default).
- 05:57 g00 all four ranks launched (after two self-inflicted fixes: the launcher had to be distributed
  to every node's /root, and it runs as root there so `$HOME` was /root while the patches live in the
  node's tonyspark home; added a per-rank `PATCH_HOME`). Polling /health.
- Harness note: `/root/glm_run.sh <label> [KNOB=val ...]` = boot + screen + one row in
  `/var/tmp/boot-results/glm53/table.txt`; status in `status.txt`.

## Experiment plan (one change per boot, stacked on the best so far)
Ranked by what the two GLM repos and last night's DeepSeek run say should move these prompts:
1. **g01 `SPEC_K=9`** - DFlash2 is a block-diffusion drafter; TP2 measured k=7 > k=5 (k=5 cost 15-24%
   single-stream) and never tried higher. Decode on these prompts is acceptance-bound, and prose is the
   weakest category, so more draft tokens is the most direct lever on the metric Tony flagged as important.
2. **g02 `MNBT=8192`** - the TP4 default traded ~3% aggregate for prefill; DeepSeek's harness prefill
   targets are 8K/32K, so check which side of that trade wins on these prompts.
3. **g03 `SEQS=16`** - DeepSeek's final serves 16; matching it makes the aggregate columns comparable at
   equal batch, and 64 costs graph memory that the KV pin could use.
4. **g04 `KV_MEM` down to 12-16 GiB** - the pool is 3.8M tokens for a 1M context (3.66x). Freeing 8-12 GiB
   of pinned KV gives the allocator room; on DeepSeek the same trade was worth nothing, so this is a check.
5. **g05 `MOE_BACKEND`** - marlin is the repo default. If another NVFP4 MoE backend is accepted on SM121
   at TP4 it is the one true kernel lever available without a rebuild (DeepSeek's kernel win last night was
   exactly this class of change).
6. **g06 `BLOCK=128`** - block 2304 is unusual; DeepSeek runs 128. Affects paging granularity and the
   indexer path.
7. Reserve: `ROCE=0` as a control if anything looks off, and `CG=EAGER` only to confirm the repo's
   FULL_AND_PIECEWISE finding still holds at TP4 with these prompts.

## Two boot traps found bringing TP4 up (both now fixed in the launcher)
1. **`$HOME` is /root on the workers.** `glm_boot.sh` launches each worker rank with `sudo -n env ... bash
   /root/glm53_tp4.sh`, so the launcher's `$HOME/patches/...` prereq checks looked in /root while the files
   live in each node's tonyspark home. Fixed with a per-rank `PATCH_HOME` (rank 0 /home/tonyspark2,
   1 /home/tonyspark4, 2 /home/tonyspark3, 3 /home/tonyspark1). The launcher must also exist on every
   node's /root, not just the head (`/root/dist_glm.sh`).
2. **The multimodal warmup builds a dummy VIDEO sized to `max_model_len`.**
   `renderers/base.py:_warmup_mm_processor` calls
   `processor.dummy_inputs.get_dummy_processor_inputs(seq_len=model_config.max_model_len, ...)` over every
   modality whose limit is > 0. At 1,048,576 context that is an enormous synthetic video normalized on CPU
   through torchvision, single-threaded because serving has already dropped torch to 1 thread
   ("Reducing Torch threads from 20 to 1"). py-spy on the API server showed it parked in
   `normalize_image -> rescale_and_normalize -> glm5next._preprocess -> video_processing_utils` for 8
   minutes with `/health` still refusing connections, engine fully up behind it (engine init 331 s, graphs
   captured 67 s / 2.74 GiB, RoCEnante live). **Fix: `--limit-mm-per-prompt {"image":4,"video":0}`** - image
   support kept, video warmup skipped, and it matches DeepSeek's mm config exactly, so the comparison is
   fairer too. Anyone running this recipe at 1M context with video enabled pays this on every boot.
   (Quoting trap on the way: the JSON's double quotes have to survive `env`, `ssh` and `eval`, so the boot
   script single-quotes env values.)
- 06:18:53 g00 relaunched with the fix; `--limit-mm-per-prompt {"image":4,"video":0}` verified in the
  container args. KV pool on the first attempt was 3,834,498 tokens at 1M (3.66x), matching the repo.

## g00 baseline: GLM-5.3 TP4 (repo default + RoCE + prefix fix) on DeepSeek's prompts
Serving 06:33:10, 860 s boot, KV 3,834,498 at 1M, quality PASS (so the ModelOpt corruption does not bite
these prompts), RoCEnante live, graphs 46 s / 4.88 GiB.

| C1 per-stream tok/s | DeepSeek target | GLM g00 | delta |
|---|---|---|---|
| counting | 113.0 | **114.5** | **+1.3%** |
| structure (format) | 98.6 | **98.8** | **+0.2%** |
| summary | 41.1 | **46.1** | **+12.2%** |
| math | 87.5 | 86.8 | -0.8% |
| JSON | 75.8 | 74.2 | -2.1% |
| prose | 39.0 | 34.2 | -12.3% |
| code | 90.7 | 73.6 | -18.9% |
| reasoning | 75.1 | 61.6 | -18.0% |
| narrative | 31.4 | 28.1 | -10.5% |
| aggregate C1 / C3 / C6 | 61.3 / 128.9 / 189.3 | 56.2 / 98.6 / 143.6 | -8% / -24% / -24% |
| cold prefill 8K / 32K target | 1,878 / 2,007 | 1,385 / 1,043 | -26% / -48% |
| idle count / code | 117.3 / 95.3 | 116.1 / 67.5 | -1% / -29% |

**Diagnosis: the step is slower, acceptance is fine.** On counting GLM accepts **7.65 tok/step** (k=7, cap 8)
against DeepSeek's 5.85 and only ties at ~115 tok/s, so GLM's decode step is ~66 ms vs DeepSeek's ~50 ms.
On code GLM accepts 5.10/8 (acceptance-limited too) at 67 tok/s => step ~76 ms vs DeepSeek ~52 ms.

**Comparison nuance to keep honest:** GLM's tokenizer emits ~30% more tokens for the same prompt text
(8K target = 15,168 GLM tokens vs 11,592 DeepSeek; 32K target = 60,917 vs 46,810). tok/s is still tok/s, but
user-visible TTFT on identical text favours DeepSeek by more than the prefill tok/s gap suggests
(10.95 s vs 6.21 s at 8K; 58.4 s vs 23.9 s at 32K).

## The kernel finding (top lever)
The engine logs, at TP4 with this checkpoint:
```
nvfp4.py:244 Using 'MARLIN' NvFp4 MoE backend out of potential backends:
  ['FLASHINFER_TRTLLM','FLASHINFER_CUTEDSL','FLASHINFER_CUTLASS','VLLM_CUTLASS','MARLIN','HUMMING','EMULATION']
marlin_utils_fp4.py:354 WARNING Your GPU does not have native support for FP4 computation but FP4
  quantization is being used. Weight-only FP4 compression will be used
```
The repo's launcher pins `--moe-backend marlin`, and Marlin is weight-only: it dequantizes NVFP4 to bf16
instead of using GB10's FP4 tensor cores, which doubles the bytes moved per decode step. DeepSeek's own
NVFP4 lane selected FLASHINFER_CUTLASS on this same hardware last night. Community reports warn CUTLASS FP4
on SM121 can produce silently wrong output, so the quality gate is the guard on every one of these boots.
- 06:41 g01-cutlass booting: `MOE_BACKEND=flashinfer_cutlass`, everything else identical to g00.
- Queue after it: g02-k9 (SPEC_K=9), g03-mnbt8k (MNBT=8192, prefill), g04-seqs16 (match DeepSeek's batch).
- `/root/glm_dq.sh` runs the queue (STOP_AT 12:40) so the last GLM boot lands before the DeepSeek restore.
- 06:59 **g01-cutlass FAILED**: `--moe-backend flashinfer_cutlass` loads all 120 shards then dies in engine
  init (`RuntimeError: Engine core initialization failed`, 1126 s in). Root cause line was lost because the
  next boot reaped the container; `glm_boot.sh` now saves `docker logs --tail 400` to
  `headlog-<label>.txt` before exiting. Agent dispatched to read the oracle's `is_supported_config`
  conditions for every NVFP4 backend and the Marlin capability check, rather than spending boots guessing.
- 07:00 self-inflicted: `dist_glm.sh` still carried the initial `pkill -f "^bash /root/glm_boot.sh"`, so
  redistributing the launcher killed the in-flight g02 (rc=143) and the two-failure rule stopped the queue.
  Kill line removed; queue restarted with g02 re-queued. Lesson repeated from the DeepSeek nights: never put
  a pkill of the harness inside a script the harness's operator runs mid-flight.

## CORRECTION + a correctness landmine (source audit, 07:15)
My "Marlin doubles the bytes per step" reading of the boot log was **wrong**, and chasing it would have
served wrong output. What the source actually says:

1. **`marlin_utils_fp4.py:354` is an unconditional `logger.warning_once`**, not a capability check - it
   prints "your GPU does not have native support for FP4" on any GPU whenever the Marlin NVFP4 MoE path is
   used. The only capability test in the file, `is_fp4_marlin_supported()` (`:34-35`), tests
   `has_device_capability(75)` and is not what selects the path.
2. **Marlin moves no extra bytes.** `_repack_marlin_experts` keeps 4-bit packing (`:311-330`) and
   `nvfp4_marlin_process_scales` ends at `marlin_scales.view(torch.float8_e4m3fn)` (`:119-120`), so group-16
   scales stay 1 byte: **4.5 bits/weight, identical to a native FP4 path**. The dequant-to-bf16 is in
   registers. Marlin costs MMA *throughput*, which only matters when compute-bound.
3. **Decode here is bandwidth-bound by 7x (64 seqs) to >100x (1 seq).** Geometry from config.json:
   hidden 4096, moe_intermediate 2048 (512/rank at TP4), 288 routed experts, top-8, 42 of 45 layers sparse;
   routed MoE = 304.4B params = 171.2 GB = **42.8 GB/rank**. At 1 sequence with k=9 (10 tokens/step) the union
   of touched experts is <=80 of 288 => **~11.9 GB/rank/step => ~43.5 ms at GB10's 273 GB/s**, which is the
   ~66 ms step I measured. **No MoE backend change can move decode step time.** The native-FP4 prize is
   entirely in prefill (Marlin bf16 MMA ~554 ms vs a ~157 ms memory floor at mnbt 16384).
4. **The landmine: this checkpoint declares W4A4 but ships no activation scales.**
   `model.safetensors.index.json` has **0** `input_scale` tensors and `config.json` has
   `"input_activations": null`, yet `modelopt.py:1392-1397` sets `activation_key = kNvfp4Dynamic` because
   `quant_algo` is `"NVFP4"` rather than `"W4A16_NVFP4"`. vLLM then creates `w13_input_scale`/`w2_input_scale`
   with **`torch.empty`** (`modelopt.py:1509-1523`) and `default_loader.py:455-464` marks them loaded, so
   nothing raises. Every W4A4 backend multiplies real weight scales by that uninitialised memory
   (`flashinfer_cutlass_moe.py:67-69`, `cutlass_moe.py:699-700`; then `nvfp4.py:526` takes `1.0/a13_scale`).
   **That is what killed g01-cutlass, and had it survived it would have served silently wrong output** - if the
   garbage reads as small nonzero junk, counting/JSON/arithmetic/code checks can all still pass. Marlin
   escapes only because `nvfp4.py:406-407` sets those scales to None and throws them away.
   => **`flashinfer_cutlass` and `vllm_cutlass` are excluded on correctness, not speed.**
   Also of note: `swiglu_limit=10.0` in the config filters the backend list to
   `NVFP4_BACKENDS_WITH_CLAMP` (`nvfp4.py:201-204`), which is exactly the 7-entry list in our log, and
   FLASHINFER_TRTLLM / FLASHINFER_CUTEDSL are hard-excluded by `is_device_capability_family(100)` (SM121 is
   family 120). `flashinfer_b12x` is the only native-FP4 path on SM121 and is blocked solely by that clamp
   filter; it also silently drops the clamp, so it needs logprob comparison, not a pass/fail gate.
   **Free safety fix for Tony (zero speed effect): set `"quant_algo": "W4A16_NVFP4"` in this checkpoint's
   config.json** so no W4A4 kernel can ever be selected for it. Not applied - his checkpoint, his call.

Queue after this: g03-ep (`--enable-expert-parallel`, the top numerics-safe perf bet: 72 whole experts per
rank, N=2048 instead of 512, 4x fewer expert GEMMs), g04-mnbt8k, g05-humming (numerics-safe, discards the
garbage scales and applies the clamp), g06-seqs16.

## Experiment table (DeepSeek's harness; target = DeepSeek final 2026-09-19)
| label | change | KV | agg C1/C3/C6 | code | json | math | prose | count | fmt | pf8k/32k | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **target** | DeepSeek V4.1 Flash TP4 | 4,011,085 | 61.3/128.9/189.3 | 90.7 | 75.8 | 87.5 | 39.0 | 113.0 | 98.6 | 1878/2007 | to beat |
| g00-baseline | repo default + RoCE + prefix fix, k=7 | 3,834,498 | 56.2/98.6/143.6 | 73.6 | 74.2 | 86.8 | 34.2 | **114.5** | **98.8** | 1385/1043 | reference; wins count, fmt, summary |
| g01-cutlass | `--moe-backend flashinfer_cutlass` | - | - | - | - | - | - | - | - | - | **BOOT FAILED**, and correctly so: W4A4 on a checkpoint with no activation scales |
| g02-k9 | `SPEC_K=9` | 3,465,506 | 52.3/97.2/135.9 | 67.5 | 68.5 | 65.7 | 35.5 | **129.4** | **106.2** | 912/687 | **DROP as default**: counting +13% and fmt +7.5% but code/json/math and prefill all worse. More draft tokens only pay where acceptance is already near-max |
| g03-ep | `--enable-expert-parallel` (marlin) | 3,834,498 | 49.1/92.8/134.4 | 56.1 | 66.0 | 71.4 | 35.8 | 88.3 | 85.2 | 720/1036 | **DROP**: worse everywhere but prose. At 10 tokens/step the all-to-all and expert idling dominate; the better Marlin shape (N=2048) never pays at this batch size |
| g07-k5 (1st try) | `SPEC_K=5` | - | - | - | - | - | - | - | - | - | **VOID, my fault**: I ran `dist_glm.sh` while its ranks were launching. **bash reads a script incrementally as it executes**, so overwriting `/root/glm_boot.sh` mid-run corrupted the running copy (`line 25: syntax error near unexpected token 'done'`). Requeued. Rule: never edit or redistribute harness scripts while a boot is in flight |
| g08-nospec | `SPEC=none` (no DFlash2 at all) | **4,205,579** | 25.6/55.4/91.2 | 26.3 | 26.4 | 26.3 | 26.4 | 26.3 | 26.3 | 1074/**2053** | **DROP for decode, but two findings**: every category pins at ~26.3 tok/s = the bare single-token step (~38 ms), so the drafter earns its keep everywhere incl. prose (34.2 vs 26.4, +30%); and **prefill 32K hits 2,053 > DeepSeek's 2,007**, i.e. DFlash2 costs ~half of prefill throughput. KV pool also +10% (no draft graphs), above DeepSeek's |

## The step-time arithmetic, now measured from both ends
- 1 token/step (no spec): ~38 ms. 10 tokens/step (k=9): ~66 ms. So +9 tokens costs only +28 ms - expert
  overlap makes extra tokens cheap, which is why speculation wins big and why k=7 beats k=9 only because
  *acceptance* stops scaling, not because the step got dear.
- Ceiling at k=7 with perfect acceptance: 8 / 0.066 = **~121 tok/s**. Counting already measures 114.5-129.4,
  i.e. counting and structure are at the hardware ceiling for this model at TP4.
- Code needs ~6.9 accepted tokens per 66 ms step to reach DeepSeek's 90.7; it accepts 5.10. Prose needs ~2.6
  and accepts ~1-2. **Those two categories are acceptance-bound, and no config knob raises acceptance.**
| g07-k5 | `SPEC_K=5` | 3,895,606 | 45.8/96.9/136.1 | **76.7** | 42.1 | 65.8 | 35.1 | 81.9 | 68.9 | 634/961 | **DROP as default**: best code of the night (76.7) and prose 35.1, but counting, format, JSON and math all fall hard. Confirms the k trade is per-content, not global |

## Checkpoint options checked and ruled out
- `/var/tmp/models/GLM-5.3-Int4-Int8Mix` (378G on disk) is **not** a smaller GLM-5.3-Flash: arch
  `GlmMoeDsaForCausalLM`, hidden 6144, 256 experts, 78 layers, 405 GB total - that is the GLM-5.2 QuantTrio
  model. No lower-bit GLM-5.3-Flash exists on the fleet, so bytes/step is fixed by the 4.5 bpw NVFP4 pack.
- `RedHatAI/GLM-5.3-Flash-NVFP4` (compressed-tensors, real activation scales, would unlock a safe native-FP4
  path worth ~3.5x on MoE prefill) was deleted from Reddie on 09-17 and Reddie has 157 GB free against a
  185 GB download. Not attemptable inside tonight's window, and it would not move decode (bandwidth-bound).
| g04-mnbt8k | `MNBT=8192` (repo default is 16384) | 3,895,606 | 52.9/96.4/**151.7** | 67.1 | 73.8 | 67.1 | 32.1 | 94.9 | 92.5 | **1601/1960** | **KEEP for prefill**: 32K prefill 1,043 -> 1,960 (+88%), 8K +16%, C6 aggregate +6%. Costs single-stream decode. The repo's 8192 -> 16384 trade runs the other way on DeepSeek's prompt lengths |

## Suspected systematic drift (why g09-rep7 and g10-rep9 are queued)
Counting has measured 114.5, 129.4, 88.3, 81.9, 94.9 and math 86.8, 65.7, 71.4, 65.8, 67.1 across five boots,
and the **first** boot of the night holds the best of both. Swings that large (±25%) cannot all be knob
effects, and if there is drift with fleet time-in-service then every experiment-to-experiment delta tonight is
confounded. g09-rep7 repeats the g00 config exactly and g10-rep9 repeats g02; if they land near their
originals the deltas are real, and if they land far below then the k-comparisons must be reported as
unreliable rather than as findings. This is the same "lanes age" question the DeepSeek run opened on 09-19.
| g05-humming | `--moe-backend humming` (the one W4A4-safe alternative) | 3,834,498 | 48.8/95.2/131.8 | 70.5 | 58.7 | 50.1 | 33.6 | 89.5 | 87.6 | 830/625 | **DROP**: slower than marlin nearly everywhere, quality PASS. Confirms the audit's read that humming is W4A16 by default and a different grouped GEMM, not a faster one. **Marlin stays** |
| g09-rep7 | **exact repeat of g00** (k=7, mnbt 16384) | 3,834,498 | 49.8/96.2/137.8 | 56.5 | 56.6 | 79.7 | 37.4 | 82.5 | 91.8 | 1148/918 | **THE NOISE CONTROL, and it is damning**: identical config, code -23%, JSON -24%, counting -28%, prose +9% vs g00. KV pool identical, clocks healthy 2190 MHz |

## Method change at 09:20: single-run cells on this lane are not resolvable
g00 vs g09-rep7 are the same configuration and differ by up to **28%** per category. So this lane has a
**+/-25% run-to-run band on single-run cells**, and most of tonight's knob comparisons cannot be resolved at
that granularity. What survives the band:
- **g08-nospec**: every category at 26.3 tok/s (the bare 1-token step). Unambiguous.
- **g04-mnbt8k**: 32K prefill 1,043 -> 1,960 (+88%), far outside the prefill band (which rep7 puts at ~17%).
- Everything else, including k=9's counting "+13%" and k=5's code "+4%", is **inside the noise and cannot be
  claimed**. Recorded as unresolved rather than as findings.
DeepSeek's harness runs one batch per cell, which was adequate there (~4% observed spread) and is far too
coarse for this lane, whose DFlash2 acceptance varies with content and warm-up. GLM's own TP2 harness took
"median of 3" for exactly this reason.
**Remaining hours go to measuring the shipping candidate properly** with `/root/glm_screen3.sh` (warm-up pass
discarded, 3 measured passes, per-cell medians, plus the max/min spread per cell so the band is published
alongside the numbers), rather than sampling more knobs at a granularity the lane cannot support.
| g10-rep9 | **exact repeat of g02** (k=9) | 3,465,506 | 53.4/99.2/136.0 | **78.8** | 67.8 | 69.2 | 33.1 | 100.2 | 103.6 | 1662/1444 | Second noise pair: same config as g02 yet code +17%, counting -23%, prefill +82/+110%. Best code reading of the night (78.8) |

## What two reps per k actually say (and it reverses the single-run call)
Mean of the two reps of each configuration, which is the most that can be claimed at this noise level:
| category | k=7 (g00, g09) | k=9 (g02, g10) |
|---|---|---|
| code | 65.1 | **73.2** |
| counting | 98.5 | **114.8** |
| structure | 95.3 | **104.9** |
| JSON | 65.4 | **68.2** |
| math | **83.3** | 67.5 |
| prose | **35.8** | 34.3 |
| C6 aggregate | **140.7** | 136.0 |
k=9 takes four of Tony's six named categories, k=7 takes math and prose. My earlier "k=7 stays, DROP k=9" call
came from single runs and does not survive the repeats. Both candidates therefore get a proper 3-rep
measurement: **g11-ship = k=7 + MNBT 8192**, **g12-ship9 = k=9 + MNBT 8192**.

## 10:12 UTC - g11 measured with medians, and the honest verdict

g11-ship (MNBT=8192, SPEC_K=7), medians of 3 measured passes after one discarded warm-up:

C1 per-stream: code 77.4, json 68.9, math 79.4, prose 32.8, format 91.1, counting 96.8, reasoning 55.9,
summary 49.7, narrative 28.5. Aggregate C1 54.5 / C3 96.3 / C6 143.3. Quality PASS. Per-cell spread
1.05x-1.41x over the three passes.

Cold prefill, rep 1 only (reps 2 and 3 hit the prefix cache: 8000 -> 7622 and 32000 -> 31988 reported
prompt tokens, so those cells are invalid): 2000 target 1,025 tok/s (3,814 tok, ttft 3.7 s), 8000 target
1,486 (15,168 tok, ttft 10.2 s), 32000 target 1,858 (60,917 tok, ttft 32.8 s), 64000 target 1,748
(121,681 tok, ttft 69.6 s).

**g11 loses all six of Tony's named categories** against DeepSeek's 09-19 final (code 90.7, json 75.8,
math 87.5, prose 39.0, counting 113.0, structure 98.6) by 8-16%, and C6 aggregate by 24%. The counting and
structure "wins" from the first boots were inside the noise band.

Two measurement points that change how the comparison should be read:

- **Prompt tokenization differs by about 30%.** The same prefill text is 3,814 GLM tokens vs 2,950 DeepSeek
  tokens at the 2000 target, and 121,681 vs 93,335 at the 64000 target. Prefill tok/s therefore flatters GLM;
  wall-clock TTFT is the honest number and it is worse than the tok/s gap suggests. Both are published.
- **Output tokenization is close** (chars per output token: code 3.35 GLM vs 3.15 DeepSeek, prose 4.24 vs
  4.83), so decode tok/s is roughly comparable. A chars/s table is published alongside it anyway, since it is
  the one throughput metric that does not depend on either vocabulary.

## Plan to the deadline

1. g12-ship9 (MNBT=8192, SPEC_K=9) booted 10:04, same 3-rep protocol.
2. Then `glm_hiconc.sh` on the serving lane: C8/C12/C16. This is the one axis where GLM has a structural
   advantage (max-num-seqs 64 against DeepSeek's 16), so it is where a real win could still be.
3. Then hand the fleet back to DeepSeek and **measure DeepSeek the same way** (3 reps, medians, plus the same
   C8/C12/C16 sweep) so the published comparison is medians against medians rather than medians against the
   single runs of 09-19. `ds_final.sh` does the boot and both measurements in one step.
4. DeepSeek stays serving. The handback is armed as a chained job with a hard 12:10 UTC deadline, so it happens
   even if the session running these experiments dies.

## Decision rule for k=7 vs k=9, written down before the numbers arrive

With per-cell spread of 1.05x-1.41x over three passes, a median-of-3 resolves differences of roughly 10% and
up, not less. So:

- If g12 (k=9) beats g11 (k=7) by more than 10% on a majority of the six named categories, ship k=9.
- If the two are inside 10% of each other, ship **k=7**: it is the documented default, and a shorter draft is
  strictly cheaper in verify compute at the concurrency levels where the aggregate is measured.
- Either way, publish both sets of medians with their spreads, and say the comparison is unresolved if it is.

This is written before the data so that the choice is not made by reading whichever direction the noise fell.

## 10:18 UTC - g12-ship9 serving, launcher hardened for the swap

g12-ship9 (MNBT=8192, SPEC_K=9) came up in 14 min 2 s. 3-rep measurement running.

While nothing was booting, made the video-warmup fix the launcher's default value of `VLLM_EXTRA` and pushed it
to all four nodes: md5 now 17bb1581 on Reddie, Spark4, Asusi and Bluey, `bash -n` clean on each. Without this a
bare `glm_boot.sh` swap back to GLM would sit for eight minutes with `/health` refusing connections while the
engine was already up, which is exactly the trap that cost this run a boot. `dist_glm.sh` prints the md5 per
node so a stale copy is visible before it causes a confusing failure.

g12 milestones: KV pool **3,515,369 tokens** (max concurrency 3.35x at 1,048,576 per request) against g11's
3,834,498, so **k=9 costs 8% of the KV pool** as well as verify compute. Graph capture 45 s / 5.09 GiB. RoCEnante
live on both routes: all-reduce up to 2 MB (NCCL above that) and the all-gather route with a 16 MiB shard cap.

## 10:31 UTC - g12 (k=9) measured, and the decision: ship k=7

g12-ship9 medians of 3 (spread in brackets): code 78.6 (1.15x), json 72.1 (1.24x), math 66.1 (1.41x),
prose 34.5 (1.12x), format 88.5 (1.29x), counting 91.1 (1.51x), reasoning 43.4 (1.27x), summary 43.2 (1.89x),
narrative 27.4 (1.33x). Aggregate C1 51.0 / C3 92.8 / C6 130.0.

Against g11 (k=7): code +1.5%, json +4.6%, prose +5.2%, format -2.9%, counting -5.9%, math -16.7%.
Aggregates all worse: C1 -6%, C3 -3.6%, C6 -9.3%. KV pool 8% smaller.

**Per the rule written down before the numbers: k=9 wins no named category by more than 10%, so k=7 ships.**
The aggregate losses and the KV cost are the only two effects that point the same way twice, and both favour
k=7, which is consistent with a longer draft costing verify compute at concurrency.

The idle test disagrees with the bench on counting, which is worth stating plainly: k=9 raises accepted tokens
per step on counting from 7.50 to 8.79 (+17%) and idle counting tok/s from 113.1 to 122.9, while the bench's
counting median moved the other way (96.8 -> 91.1) on a cell whose own spread was 1.51x. When two measurements
of the same quantity disagree by less than the noise, neither is evidence.

### Correction to the prefill reading

g11's prefill numbers were not representative, and I reported them too confidently earlier. Rep-1 cold prefill,
both boots, identical MNBT=8192:

| target | g11 (k=7) | g12 (k=9) |
|---|---|---|
| 2,000 (3,814 tok) | 1,025 tok/s, ttft 3.7 s | 1,794 tok/s, ttft 2.1 s |
| 8,000 (15,168 tok) | 1,486, ttft 10.2 s | 2,062, ttft 7.4 s |
| 32,000 (60,917 tok) | 1,858, ttft 32.8 s | 2,014, ttft 30.2 s |
| 64,000 (121,681 tok) | 1,748, ttft 69.6 s | 1,803, ttft 67.5 s |

Draft length should not move prefill by 75%. The two large cells agree within 3-8% between boots and the two
small ones do not, so **the small-target prefill cells are not trustworthy**: they are the first thing measured
after the warm-up pass and they carry whatever state it left. Only the 32K and 64K cells get published as
prefill, and at those sizes GLM sits at roughly 1,850-2,050 tok/s, which is close to DeepSeek's 2,007-2,048.
GLM's prefill is therefore near parity in tok/s and behind in wall-clock TTFT, which is the ~30% token
inflation rather than the engine.

## Revised plan, 10:31

The C8+ sweep must run on the config that ships, so the handback was re-gated (chain3 replaced by chain3v2 on a
new flag, hard deadline 12:15 UTC). Order now: g12's C8/C12/C16 sweep (running), g12's C24/C32 probe, then
**g13-ship7rep: a second independent boot of the k=7 shipping config**, which gets the same 3-rep screen plus
both sweeps. That second boot answers the methodological question this run keeps running into: does a median of
three passes actually reproduce across boots, or is the median itself unstable? Then DeepSeek comes back and is
measured the same way.

## 10:40 UTC - the high-concurrency hypothesis fails too

GLM at k=9, medians of 2 passes, mean across categories: C8 166.2, C12 194.9, C16 217.9. Per category:

| | C8 | C12 | C16 |
|---|---|---|---|
| coding | 208 | 283 | 269 |
| counting | 274 | 433 | 440 |
| tables | 316 | 312 | 366 |
| math | 214 | 252 | 295 |

DeepSeek's published C8 coding is 376 and C12 coding 475, so **GLM loses the concurrency axis by 40-68% as
well**, and it is already flattening between C12 and C16 (coding actually falls, 283 -> 269).

This was the one axis where GLM had a structural advantage on paper: it serves max-num-seqs 64 against
DeepSeek's 16. The reason the advantage does not materialise is the same bandwidth argument as at one stream,
only worse: each additional concurrent sequence routes to its own top-8 of 288 experts, so the union of experts
touched per step grows with concurrency and GLM streams more weight bytes per step. DeepSeek at 3.5 bpw with
Engram has far less to move and keeps scaling. A model whose decode is bandwidth-bound does not become
compute-bound just because you give it more streams; it becomes more bandwidth-bound.

The C24/C32 probe is still worth running, for one narrow claim only: DeepSeek's lane physically cannot serve
more than 16 concurrent streams, so if GLM's total keeps climbing past C16 there is a real capacity statement
to be made about serving many streams at once, which is a different thing from being faster.

Note on the repo's own "530 tok/s at C48" from 2026-08-31: that is a sum over mixed prompts on a different
harness, not this harness's per-category aggregate, and the two must not be put in the same table.

## 10:47 UTC - measured acceptance, and why k=9 cannot pay

`vllm:spec_decode_num_accepted_tokens_per_pos_total` at k=9, over the whole g12 session (dominated by the
C8-C32 passes). Cumulative acceptance by draft position, and the conditional survival from one position to the
next:

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| cumulative | 74.9% | 53.7% | 40.1% | 31.5% | 26.0% | 21.6% | 18.2% | 14.9% | 12.5% |
| conditional | - | 71.6% | 74.8% | 78.4% | 82.6% | 83.2% | 83.4% | 82.0% | 83.8% |

Overall acceptance 32.6% of 363,915 draft tokens; 2.93 accepted per draft; **3.93 tokens per step**.

The conditional survival is flat at roughly 0.8 after the first position, so tokens per step is
`1 + sum(cumulative)` and the tail is geometric. Dropping positions 7 and 8 gives k=7 an expected
**3.66 tokens per step against k=9's 3.93: +7% for k=9**, bought with 29% more draft and verify work per step
and 8% less KV pool. That is the whole k=7 vs k=9 question answered arithmetically, and it agrees with the
aggregate medians, which is the first time tonight two independent measurements of one effect have agreed.

It also sets the ceiling on speculation as a lever here. Even a perfect drafter for position 0 only would add
0.25 tokens per step. Getting code from 78 to DeepSeek's 91 tok/s needs tokens per step to rise by about 16%,
which at this survival curve means lifting conditional acceptance across every position, not adding positions.

## 10:47 UTC - C24 and C32: GLM's own ceiling

| | C24 | C32 |
|---|---|---|
| mean across categories | 211.0 | 230.7 |
| coding | 294 (22.0/stream) | 311 (21.3/stream) |
| counting | 497 (35.2/stream) | 459 (31.4/stream) |
| tables | 268 | 412 |
| prose | 119 (8.3/stream) | 128 (8.1/stream) |

GLM's best coding aggregate anywhere in the sweep is **311 at C32**, against DeepSeek's **475 at C12**. At twice
the concurrency DeepSeek's lane can physically reach, GLM is still 35% behind it. Counting tells the same story:
497 at C24 against DeepSeek's 639.9 at C12.

So the narrow capacity claim is the only one left standing, and it is worth stating precisely: **GLM serves 32
concurrent streams; the DeepSeek lane refuses more than 16.** That is a statement about how many users a lane
can hold, not about speed, and it should not be dressed up as one.

## 11:02 UTC - g13, second boot of the shipping config

Boot 859 s. KV pool **3,895,606 tokens** against g11's 3,834,498 on the identical config, so the pool figure
reproduces within 1.6%. That matters for the k=9 reading: g12's 3,515,369 is a 9% drop, comfortably outside
boot-to-boot variation, so the KV cost of k=9 is real rather than an artifact. Graph capture 45 s / 4.93 GiB.

## 11:25 UTC - the actual answer: a validated per-step cost model, and the 18 GiB that is costing the race

I had been calling decode "bandwidth-bound" and leaving it there. That was too coarse. Only ~33 ms of the 67.5 ms
step is routed-expert streaming, so something else owns the other half. Taking the step times the harness already
records at four draft lengths and fitting them against verify tokens t:

| k | verify tokens | measured step ms p50 | fit |
|---|---|---|---|
| none | 1 | 37.7 | 38.7 |
| 5 | 6 | 61.5 | 59.3 |
| 7 | 8 | 67.5 | 67.5 |
| 9 | 10 | 74.5 | 75.7 |

**step_ms = 34.6 + 4.12 x verify_tokens**, residuals within 2.2 ms across a 2x range of step time.

The slope is the routed experts and it checks out from the checkpoint independently: top-8 of 288 experts is
8/288 x 40.8 GiB/rank = 1.134 GiB = **4.46 ms** predicted at 273 GB/s against a measured 4.12, correctly a little
under because tokens in one verify batch share some experts.

The intercept is what I had missed. Summing the checkpoint by module class and dtype:

| module class | GiB | dtype |
|---|---|---|
| routed experts | 163.27 | U8 145.12 + F8_E4M3 18.14 |
| **attention** | **11.70** | **BF16** |
| dense MLP | 3.52 | BF16 |
| embeddings + head | 2.37 | BF16 |
| other | 0.43 | BF16 |

**18.01 GiB of non-expert weights, all bf16, read on every single step** regardless of how many tokens are in
flight. That is 4.50 GiB/rank at TP4 = **17.7 ms**, which accounts for the intercept almost exactly and leaves
16.9 ms of genuine non-weight overhead (drafter forward, aux hidden-state extraction from 5 target layers,
45 layers of all-reduce, KV reads, kernel launch, host bubble).

### What that predicts, at k=7 with code's measured 5.20 accepted tokens per step

| | step ms | code tok/s | vs DeepSeek 90.7 |
|---|---|---|---|
| as built, bf16 attention | 67.5 | 77.0 | -15% |
| **bf16 -> fp8** | 58.6 | 88.7 | -2% |
| **bf16 -> NVFP4** | 54.2 | 95.9 | **+6%** |
| fp8 plus half the overhead | 50.2 | 103.6 | +14% |

So GLM-5.3-Flash **can** beat DeepSeek-V4.1-Flash on this fleet, and the thing standing in the way is not a
launcher flag. It is 18 GiB of unquantized attention and dense weights in both NVFP4 packs on this fleet. The
keys pack and the nvidia pack both carry the same 11.70 GiB of bf16 attention, so this could not be tested by
swapping checkpoints tonight, and re-quantizing plus re-validating a 182 GiB checkpoint is not something to
start two hours before a hard handback deadline.

This is the single highest-value next step for this repo, it is quantified, and it is falsifiable: quantize the
1,327 bf16 non-expert tensors to fp8 and the model says code lands at 88.7 tok/s, within 2% of DeepSeek.

### Why k=7 is special, and it is not tuning

The drafter's own config says `dflash_config.block_size: 8`. DFlash2 is a block-diffusion drafter with a block
of 8, so k=7 fills exactly one block (1 target token + 7 drafts). **k=9 spills into a second diffusion block**,
which is why it costs more than the linear fit predicts, why its acceptance tail collapses to 12.5% at position 8,
and why it gives back 9% of the KV pool. k=5 underfills the block and wastes it. The right default was not found
by search, it is a property of the drafter, and it should be documented as such rather than as a tuned value.

## 11:21 UTC - the median protocol reproduces, and the fp8 route is priced out

### g13 vs g11: two independent boots of the identical shipping config

| category | g11 median | g13 median | delta |
|---|---|---|---|
| code | 77.4 | 74.9 | -3.2% |
| json | 68.9 | 66.8 | -3.0% |
| math | 79.4 | 69.3 | -12.7% |
| prose | 32.8 | 33.8 | +3.0% |
| structure | 91.1 | 86.8 | -4.7% |
| counting | 96.8 | 94.7 | -2.2% |
| reasoning | 55.9 | 56.7 | +1.4% |
| summary | 49.7 | 49.3 | -0.8% |
| narrative | 28.5 | 29.0 | +1.8% |
| C1 / C3 / C6 aggregate | 54.5 / 96.3 / 143.3 | 52.6 / 101.6 / 144.8 | -3.5% / +5.5% / +1.0% |

g13's within-boot spread also tightened to 1.01x-1.26x from g11's 1.05x-1.41x. **Medians of three passes
reproduce across boots to within 5% on eight of nine categories**, against single runs that differed by 23-28%
on the same lane. The protocol works, which is what makes the DeepSeek comparison worth publishing at all.

Mean of the two boots against DeepSeek: code 76.2 vs 90.7 (-16%), json 67.9 vs 75.8 (-10%), math 74.4 vs 87.5
(-15%), prose 33.3 vs 39.0 (-15%), structure 89.0 vs 98.6 (-10%), counting 95.8 vs 113.0 (-15%). A consistent
10-16% deficit, stable across boots.

### Why the fp8 conversion was built, priced, and then not booted

Having found the 17.7 ms of bf16 non-expert weights, the obvious move was to quantize them tonight. The
conversion is cheap to stage: hardlink the 120 original shards, write one new fp8 shard, rewrite the index
(vLLM's `filter_duplicate_safetensors_files` keeps only indexed files), which is about 4 minutes of I/O rather
than rewriting 78 GiB. The script is in `tools/mkfp8.py` and its dry run measures 2.6-2.9% per-tensor relative
error, normal for per-tensor fp8 with no outlier handling.

Then the module breakdown priced it, and the answer is that it cannot win tonight:

| non-expert bf16 module | GiB | quantize? |
|---|---|---|
| `self_attn.o_proj` x46 | 3.625 | yes |
| `self_attn.q_proj` / `k_proj` / `v_proj` x34 | 6.375 | yes |
| `mlp.shared_experts.{gate,up,down}` x43 | 2.016 | yes |
| `q_b_proj` / `kv_b_proj` / `q_a_proj` / `kv_a_proj_with_mqa` x12 | 1.125 | yes |
| dense `mlp.{gate,up,down}` x3 | 0.843 | yes |
| `embed_tokens` + `lm_head` | 2.364 | **no** - embedding is a lookup, not a matmul, and 2.8% error on lm_head moves logits directly |
| `visual.merger.*` | 0.234 | **no** - not on the decode path |
| `self_attn.indexer.*` | 0.141 | **no** - DSA top-k selection; error changes which KV blocks are attended |
| `mlp.gate` (MoE router) x43 | 0.094 | **no** - error flips expert selection |
| KDA `f_b_proj` / `g_b_proj` gates | 0.132 | **no** - gating, not projection |

13.98 GiB is safely quantizable, 3.95 GiB is not. At 273 GB/s:

| | saved per rank | step | code tok/s | vs DeepSeek 90.7 |
|---|---|---|---|---|
| as built | - | 67.6 ms | 77.0 | -15% |
| fp8 on the safe 13.98 GiB | 1.7 GiB -> 6.9 ms | 60.7 ms | 85.7 | **-6%** |
| NVFP4 on the safe 13.98 GiB | 2.6 GiB -> 10.3 ms | 57.2 ms | 90.8 | **+0%** |
| NVFP4 plus half the 16.9 ms overhead | | 48.8 ms | 106.6 | +17% |

**fp8 reaches 86, NVFP4 reaches a tie.** Beating DeepSeek needs the NVFP4 re-quantization *and* work on the
16.9 ms of non-weight overhead. So booting an fp8 build tonight would have cost roughly 75 minutes of the
handback margin to land at -6%, on a checkpoint whose quality I could only have validated with the same thin
7-prompt gate this write-up already criticises for missing the W4A4 landmine. It was not booted, on arithmetic
rather than on time. The script and the module list are in the repo so the NVFP4 version is a one-shot job for
whoever picks it up.

## 11:31 UTC - the NVFP4-attention build, and why it was worth the margin after all

My earlier decision to skip this was based on fp8 arithmetic, and it was the wrong comparison. fp8 halves the
bytes and lands at -6%; NVFP4 quarters them and scales **every** category by 67.6/57.2 = 1.182:

| | GLM measured | x1.182 predicted | DeepSeek | predicted delta |
|---|---|---|---|---|
| code | 76.2 | 90.1 | 90.7 | -0.7% |
| json | 67.9 | 80.3 | 75.8 | +5.9% |
| math | 74.4 | 87.9 | 87.5 | +0.5% |
| prose | 33.3 | 39.4 | 39.0 | +1.0% |
| counting | 95.8 | 113.2 | 113.0 | +0.2% |
| structure | 89.0 | 105.2 | 98.6 | +6.7% |

Five of six at or above DeepSeek is worth a boot, so it was built.

**The format was verified before anything was written**, against tensors this checkpoint already stores in it:
`ops.scaled_fp4_quant(w, 448*6/amax, is_sf_swizzled_layout=False)` reproduced a stored expert's packed weight
bytes **exactly** and its fp8 block scales at **1.0000 identical**, and the implied `weight_scale_2` matched the
stored `3.77837e-05` to every digit. Low nibble first. The swizzled layout matched only 12% of scale bytes, so
that flag matters and is not guesswork.

334 tensors, 13.98 GiB bf16 -> 3.93 GiB NVFP4 in 126 s. Excluded on correctness, not convenience:
`embed_tokens` and `lm_head` (a lookup, and logits), `self_attn.indexer.*` (chooses which KV blocks are
attended), `mlp.gate` (chooses experts), the KDA `f_b`/`g_b` gates, `eh_proj`, `visual.*`. 16 patterns stay
ignored.

Assembly avoided rewriting 78 GiB: hardlink the 120 originals, drop in one new shard, retarget 334 names in the
index (vLLM's `filter_duplicate_safetensors_files` keeps only indexed files), zero unreferenced shards.

**`quant_algo` NVFP4 -> W4A16_NVFP4**, selecting `ModelOptNvFp4W4A16LinearMethod`, described in vLLM's own
source as FP4 Marlin GEMM with bf16 activations. It needs no `input_scale`, so the same edit that makes the new
attention weights loadable is also the fix for the W4A4 uninitialised-scale landmine documented above. No engine
patch was needed at all; those modules were only ever excluded by config.

## 11:28 UTC - a real trap, found the hard way: the DeepSeek restore needs settle time

chain4 cleared the handback hold a moment before my re-assert, so the handback fired early and **failed**:

```
== rank 3 Bluey ==
MemAvailable 15 GiB < 100 GiB, refusing to boot
boot_dsv41_tp4x exit=4
no new head container on Reddie: stopping
```

The DS launcher's memory guard ran seconds after a 100 GiB GLM lane was stopped, before the fleet had reclaimed
it. Exit 4 is a **refusal, not damage** - the guard did its job - but it costs a boot, and it had never been hit
before because the restore has never previously run directly after a GLM lane rather than after another DS lane
or an idle fleet.

Fix, now in `tools/ds_handback.sh`: stop GLM everywhere, drop caches, then **poll every node until the minimum
MemAvailable is at least 100 GiB** before calling the restore, with one automatic retry. Anyone swapping a large
lane out on this fleet needs this step, in either direction.

## 11:47 UTC - both NVFP4 boots failed, and the cause was my assumption, not the quantization

g14 (11:34) and g15 (11:47) both died at the same line:

```
vllm/models/glm5next/nvidia/model.py:877  weight_loader(param, loaded_weight, shard_id)
vllm/model_executor/layers/linear.py:932  param.load_merged_column_weight(...)
vllm/model_executor/parameter.py:175      assert param_data.shape == loaded_weight.shape
AssertionError
```

I read the first failure as a fused-layer scale problem, and fixed two real constraints for g15 (shared global
scale per fused group, `weight_scale_2` shape `(1,)` instead of a bare scalar, all six `in_proj_qkvbfg_a`
constituents quantized instead of three). Those fixes are probably correct. They were masked, because the actual
cause was upstream of all of them:

**`filter_duplicate_safetensors_files` filters by FILE, not by tensor name.**

```python
for weight_name in weight_map:
    weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))
```

Every one of the 120 original shards is still referenced by the index, because each still holds routed-expert
tensors that must be loaded. So all 121 files are kept, and the loader then iterates **every tensor inside
them** - including the stale bf16 copies of the 412 tensors I had retargeted to the new shard. A stale bf16
`(out, in)` is then handed to the now-packed NVFP4 parameter of shape `(out, in/2)`, and the shape assert fires.

The hardlink-plus-index-rewrite trick is only valid when a stale tensor's shard contains nothing else that is
needed. Here 47 shards mix both. I asserted in this file earlier that the index filters by name; that was wrong,
and it cost both boots. The conversion itself was never implicated: the format was verified byte-for-byte
against the checkpoint's own expert tensors before either boot.

**The remaining fix is mechanical, not conceptual:** rewrite those 47 shards omitting the 412 superseded bf16
tensors (78.6 GiB read, about 65 GiB written, and 158 GB was free), hardlink the other 73 unchanged, keep the
new NVFP4 shard. Everything else - the group-completeness check, the shared scales, the `(1,)` shape, the
`W4A16_NVFP4` selection that also disarms the W4A4 landmine - is already written and verified in `tools/`.

**Stopping here on the two-failure rule.** Both failures were clean refusals during weight load with nothing
damaged and no partial state, but two in a row is the line, so the fleet goes back to DeepSeek rather than into
a third attempt on my own initiative. DeepSeek boot started 11:47:29 with 116 GiB free on every node.

## 11:55 UTC - attempt 3: remove the stale tensors instead of hoping the index hides them

Tony's call to keep going, so the shard rewrite was done properly. `tools/mkdir3.py` builds
`keys-glm53-nvfp4-attn3`:

- 73 shards that contain no superseded tensor: hardlinked, no copy
- 47 shards that mix superseded bf16 with tensors still needed: **rewritten without the dead tensors**
  (92 s, 42,582 tensors kept, **412 dropped**, about 65 GiB written)
- the verified NVFP4 shard and the corrected index and config carried over from attn2

Then three checks that should have existed before the first boot:

```
VERIFY duplicate names across files: 0   indexed names missing: 0   total names present: 113898
index entries pointing at a file that lacks the tensor: 0
```

Cumulative state of the three attempts:

| | g14 | g15 | g16 |
|---|---|---|---|
| all 6 `in_proj_qkvbfg_a` constituents quantized | no, 3 of 6 | yes | yes |
| one shared global scale per fused group | no | yes | yes |
| `weight_scale_2` shape | `()` | `(1,)` | `(1,)` |
| stale bf16 duplicates removed | no | no | **yes, 412** |
| duplicate tensor names in the checkpoint | 412 | 412 | **0** |

A note on the two-failure rule: I stopped and restored after g15 and did not start g16 on my own initiative.
Both failures were refusals during weight load with no partial state, and the third attempt was Tony's call with
the cause diagnosed rather than guessed.

Also logged: the first automated handback attempt after g15 reported `CHAIN6 first handback failed, retrying
once`, and its retry is what brought DeepSeek up. The retry in `ds_handback.sh` earned its place on the day it
was written.

## 12:23 UTC - g17 and the conclusive finding: the bf16 projections are hardcoded in the model, not the checkpoint

g17 excluded the KDA fused group and quantized the remaining 7.42 GiB. It failed differently again:

```
vllm/models/glm5next/nvidia/model.py:917  param = params_dict[name]
KeyError: 'layers.0.self_attn.o_proj.weight_scale'
```

The checkpoint now supplied NVFP4 scales for `o_proj`, but **the model never created a quantized parameter for
it**. The reason is in the image's own source:

```
vllm/models/glm5next/nvidia/kda.py:172    vllm_config.quant_config = None
vllm/models/glm5next/nvidia/model.py:331  quant_config=None,  # MLA projections are BF16 in checkpoint
vllm/models/glm5next/nvidia/model.py:1090 quant_config=None,
```

**The GLM-5.3-Flash implementation hardcodes its attention projections as bf16.** KDA nulls the quant config for
its entire submodule tree; the MLA path passes `quant_config=None` explicitly, with a comment asserting the
checkpoint is bf16 there - true of both NVFP4 packs on this fleet. So the `ignore` list was never the gate. No
checkpoint rebuild can reach those weights, and the four boots walked down the chain one step at a time:

| boot | failed at | cause | fixed by |
|---|---|---|---|
| g14 | `parameter.py:175` | stale bf16 duplicates (index filters by FILE, not name) + incomplete fused groups | g16 |
| g15 | `parameter.py:175` | same duplicates, masking the fused-group fixes | g16 |
| g16 | `kda.py:114` -> `parameter.py:175` | KDA loads `f_a`/`g_a` as replicated shards with `output_size *= tp_size`; does not line up with packed params | g17 excluded the group |
| g17 | `model.py:917` | `o_proj` and the rest are built with `quant_config=None` in the model source | **needs a source change** |

**So the honest conclusion of the night is a bounded one, and it has a number on it.** The 18 GiB of bf16
non-expert weights cost a measured 17.7 ms of a 67.6 ms step, the fitted model predicts that NVFP4 on 13.88 GiB
of them is worth x1.179 - enough to put five of the six named categories at or above DeepSeek - and the thing
standing in the way is three lines of `quant_config=None` in `glm5next/nvidia/`, not the quantization, not the
checkpoint format, and not a launcher flag. The conversion tooling is verified byte-for-byte correct and is in
`tools/` ready for whoever changes those lines.

GLM-5.3-Flash TP4 therefore **did not beat DeepSeek-V4.1-Flash tonight** on any of CODE, JSON, PROSE, MATH,
counting or structure. The fleet is back on DeepSeek.

## 12:44 UTC - g18 SERVES, and five of the six named categories beat DeepSeek

The blocker was two lines, and they were bind-mountable the whole time. Patched copies of the image's own
`glm5next/nvidia/kda.py` and `model.py`, mounted read-only on all four nodes behind an opt-in `NVFP4_PATCH=1`:

```
kda.py:172    vllm_config.quant_config = None            -> removed (it stripped quant for the whole KDA tree)
model.py:331  quant_config=None,  # MLA ... BF16          -> quant_config=vllm_config.quant_config,
```

Left alone deliberately, because they really are bf16 here and are in the checkpoint's `ignore` list:
`model.py:1090` (vision tower; quantizing it yields NaN image features) and `attention.py:263` (indexer
`wk_weights_proj`). `is_layer_skipped()` honours the ignore list per layer, so `indexer.*`, `f_b`/`g_b`,
`q_a`/`kv_a` and `eh_proj` stayed bf16 as intended.

Boot 841 s. `Detected ModelOpt NVFP4 checkpoint (quant_algo=W4A16_NVFP4)`, `Using MarlinNvFp4LinearKernel for
NVFP4 GEMM`, **model weights 43.76 GiB/rank**, KV pool 3,895,606 tokens, graphs 45 s / 4.92 GiB.
**Quality gate PASS 5/5.**

### The measurement, medians of 2 passes (spread in brackets)

| C1 per-stream tok/s | DeepSeek | GLM bf16 (2 boots) | **GLM NVFP4** | vs DeepSeek | vs GLM bf16 |
|---|---|---|---|---|---|
| code | 90.7 | 76.2 | 89.1 (1.08x) | -1.8% | +17% |
| json | 75.8 | 67.9 | **77.5** (1.27x) | **+2.2%** | +14% |
| math | 87.5 | 74.4 | **96.7** (1.02x) | **+10.5%** | +30% |
| prose | 39.0 | 33.3 | **41.8** (1.02x) | **+7.2%** | +26% |
| counting | 113.0 | 95.8 | **114.9** (1.02x) | **+1.7%** | +20% |
| structure | 98.6 | 89.0 | **109.5** (1.13x) | **+11.1%** | +23% |
| C1 aggregate | 61.3 | 53.6 | **62.5** | **+2.0%** | +17% |
| C3 aggregate | 128.9 | 99.0 | 107.7 | -16% | +9% |

**How strong each claim is:** math, prose and structure beat DeepSeek by more than their measured spread. json,
counting and code are at parity within noise (+2.2%, +1.7%, -1.8% against spread up to 1.27x). C3 aggregate is
still DeepSeek's by 16%, so this is a single-stream win, not a concurrency win.

### The cost model was right

Predicted: step 67.6 -> 57.2 ms, x1.179 on every category. Measured idle step 55-59 ms, and per-category gains of
x1.14 to x1.30 (mean about x1.22, slightly better than predicted). The idle probe moved code 75.9-78.1 -> 97.7
and counting 113.1-114.5 -> 138.0. Weights fell 2.6 GiB/rank short of arithmetic because `embed_tokens`,
`lm_head`, the indexer, the router and the gates stayed bf16 on purpose.

So the night's chain closes: 18.01 GiB of bf16 non-expert weights -> 17.7 ms of a 67.6 ms step -> quantize
13.88 GiB of them -> five of six categories above DeepSeek. The prediction was made before the build existed and
held.

## 12:50 UTC - the needle catches what the quality gate could not

Quantizing q/k/v/o is exactly the change that should degrade long-context retrieval before it degrades a
short-prompt gate, so the needle was run for that reason:

| needle | prompt tokens | prefill | result |
|---|---|---|---|
| 65K | 65,360 | 2,004 tok/s | **PASS**, exact: `COPPER-LANTERN-8315` |
| 131K | 131,258 | 1,965 tok/s | **FAIL**: `COPPER-LANTERN-815` - one digit dropped - then the model began second-guessing itself in the output |

**So this build is not safe to serve for long-context work on this evidence**, whatever the speed says. The
5/5 quality gate passed it; a 131K retrieval did not. That is the same class of mistake this write-up warns about
in the W4A4 section: a short gate cannot clear a numerics change.

Two caveats in both directions. There is no bf16 GLM needle at 131K from tonight to compare against, so it is not
yet established that this is a regression rather than a pre-existing limit of the lane (the repo's issue #14
concerns long-context behaviour). And the failure is one character with visible self-correction, not a collapse.
A bracket at 98K and a repeat at 131K with depth 0.6 are running to separate a length threshold from a
depth artifact.

What is safe to claim right now: **the speed result is real and measured, and the long-context quality of this
build is an open question with one concrete failure against it.** It should not be served to anything doing
100K-plus retrieval until that is resolved, and the resolution needs a bf16 baseline needle at the same lengths.

## 13:05 UTC - the pack we served is abliterated, and the NVFP4 build quantized the tensors that do it

Checking `ABLIT_META.json` in the served checkpoint (prompted by a question about a sibling build on the Hub):

```
method: dealign-oproj-transplant          style: safety-anchor-early + late-mtp-oproj
parent: LibertAIDAI/GLM-5.3-Flash-NVFP4   donor: dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4
tensor: model.language_model.layers.{L}.self_attn.o_proj.weight   min_layer 15  max_layer 45
n_edited: 31   edit_mtp: True   mean_rel_fro: 0.126   early_stock: [0, 14]   experts: NVFP4 passthrough
```

So every boot tonight, bf16 and NVFP4 alike, served an abliterated model. The abliteration is narrow: 31
`o_proj` tensors transplanted from an uncensored donor, layers 0-14 deliberately stock as a safety anchor.

**And `o_proj` is the biggest single thing the NVFP4 build quantizes** - 3.625 GiB of the 13.88 GiB. g18 therefore
applies ~2-3% per-tensor quantization error on top of a transplant that already moved those tensors by 12.6%
relative Frobenius, on 31 of 45 layers. Neither of those was designed with the other in mind.

Consequences, neither of them addressed tonight:

1. **Alignment behaviour on g18 is untested.** The quality gate covers counting, JSON, code, math and prose. None
   of those probe refusal, so whether the abliteration survived quantization intact, was blunted, or was
   amplified is simply unknown. It should not be assumed in either direction.
2. **This is the most specific suspect for the 131K needle miss.** The needle failed at 131K depth 0.3 while 65K,
   98K and 131K depth 0.6 were exact. Those transplanted layers are already 12.6% off their original weights
   before NVFP4 adds its own error, and `o_proj` sits directly on the attention output path.

Cheapest experiment to separate the two: rebuild excluding `o_proj` for layers 15-45. That gives up about 2.8 ms
of the 10.4 ms saved (so roughly x1.13 instead of x1.18) and makes the NVFP4 change orthogonal to the
abliteration. If the needle then passes at 131K depth 0.3, the interaction is confirmed.

Also corrected in the write-up: the credits named "the keys build" generically and did not name the actual parent
and donor. Note the sibling `drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-43-mtp-l45` on the Hub is
**RedHat-parented** with layers 15-43, so it is a different lineage from what we serve, not the same pack.
