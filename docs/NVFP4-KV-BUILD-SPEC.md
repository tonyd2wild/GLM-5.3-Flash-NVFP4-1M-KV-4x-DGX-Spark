# nvfp4_ds_mla: 4-bit packed KV for GLM-5.3 on the SM120 sparse path — build spec

> **PARKED — unfinished working spec from 2026-08-27, TP2-era numbers.** The NVFP4 KV lane
> was subsequently brought up successfully; see the KV-lanes section of the
> [README](../README.md) and `docs/OPEN-PROBLEMS.md`. This file is retained for its build
> reasoning only. It references a `v10` Dockerfile that does not exist in `docker/`.
>
> **Current config:** TP4, `--kv-cache-memory 25769803776` (24 GiB/rank) = **3,895,606 fp8 tokens** at 1,048,576 context, DFlash2 k=7, with `flusher-unconditional.sh`. See the [README](../README.md).

Status: IN PROGRESS (started 2026-08-27 ~04:30 UTC, Knox). Goal: halve the MLA KV bytes
(432 B/token/layer vs 656 packed / 528 plain fp8) → ~600K tokens in the proven 4.14 GiB
slab, ~900K-1M with a 5.5-6 GiB slab. Foundation: the working NoPE zero-pad shim (v10).

## Key design decision: NVFP4 storage + BF16 compute (v1)

The SM120 sparse decode kernel has two compute modes (`ComputeTraits<MT, CM>`): FP8
(native fp8 MMA m16n8k32) and **BF16 ("IO dequants FP8 KV -> BF16 in smem", bf16 MMA)**.
v1 rides the BF16 mode: the only kernel change is the IO-warp dequant step (unpack e2m1
nibbles x group scale x global scale -> bf16) — ALL MMA machinery is reused. fp4-MMA is
a v2 speed project, not a v1 correctness project.

## Layout: ModelType::GLM_NVFP4, 432 B/token (16B-aligned regions)

| Offset | Bytes | Content |
|---|---|---|
| 0 | 256 | 512 dims packed e2m1, 2 per byte |
| 256 | 32 | 32 x FP8-E4M3 group scales (group of 16 dims) |
| 288 | 4 | 1 x FP32 per-token global scale (2nd level) |
| 292 | 12 | pad to 304 (16B aligned) |
| 304 | 128 | BF16 rope region — zeros for GLM (DeepSeek-ABI compat, keeps the shim pattern) |

`KV_GMEM_STRIDE=432, KV_SMEM_STRIDE=304 (bulk copy nope+scales+pad), SCALE_INLINE=true`.

## Work items

1. **flashinfer csrc**: `ModelType::GLM_NVFP4` (model_type.h) + `KVCacheTraits<GLM_NVFP4>`
   (all constants above) + the nvfp4 dequant routine in the BF16-mode IO path of
   `decode_dsv3_2_kernel.cuh` (and the prefill kernels) + dispatch: route GLM_NVFP4
   through ComputeMode::BF16 in `sparse_mla_sm120_decode_dsv3_2.cu` / `_prefill.cu`
   (2048 AND 2176 topk, reusing tonight's TOPK templating).
2. **flashinfer python**: `_MODEL_TYPE_GLM_NVFP4`, plumb model-type + scale handling in
   `_sparse_mla_sm120.py` / `_core.py` (the fp8-path scale-format machinery is bypassed —
   scales are consumed in-dequant).
3. **Triton cache writer** (vllm side, in our `do_kv_cache_update` shim override): bf16
   -> group-amax -> e4m3 scales + fp32 global -> e2m1 nibble pack -> layout write; rope
   region zeroed. Standalone-probeable.
4. **vllm plumbing**: CacheDType "nvfp4_ds_mla", page accounting 432B, canonicalize for
   the SM120 backend, `get_kv_cache_shape`, impl dtype acceptance.
5. **Probe ladder**: (a) writer roundtrip vs reference quant; (b) kernel probe — synthetic
   packed cache through the flashinfer API vs bf16 reference; (c) boot on lane 2; (d)
   needle tests 33K/131K/262K; (e) bench; (f) slab ladder 4.14 -> 5.5 -> 6.5 GiB.

## Risks
e2m1 KV quality on this model (needle + logprob checks gate each step); IO dequant
throughput in BF16 mode (correctness first, tune later); prefill orchestrator BF16-mode
coverage for V_HAS_ROPE=false; scale-format assumptions buried in the fp8 path.

Existence proof for the target: a third party serves GLM-5.3 with 4-bit KV (B12X kernels,
368 B/token, needle-verified 1.04M ctx) on 2 Sparks.
