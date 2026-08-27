# Draft issue for vllm-project/vllm

## Title

[Feature] NoPE variant of the `fp8_ds_mla` packed KV layout (`pe_dim=0`) so GLM-5.3-Flash-class NoPE sparse-MLA models can use fp8 KV cache (SM120/SM121 Blackwell)

## Body

### Summary

`--kv-cache-dtype fp8_ds_mla` (and the fp8 path of the sparse-MLA backends generally) hardcodes the DeepSeek geometry `pe_dim=64` in the packed KV layout. NoPE sparse-MLA models — GLM-5.3-Flash (`glm5_next`: `kv_lora_rank=512`, `qk_rope_head_dim=0`, DeepSeek-style sparse indexer) — therefore cannot use fp8 KV at all: the cache write kernel asserts and kills the engine.

```
concat_and_cache_mla, csrc/cache_kernels.cu:866: pe_dim must be 64 for fp8_ds_mla
```

This is not hypothetical: RTX PRO 6000 Blackwell (SM120, 4x GPU) users hit exactly this crash serving GLM-5.3-Flash with the official image, and removing `--kv-cache-dtype fp8` did not help (auto re-selected the same path): https://huggingface.co/zai-org/GLM-5.3-Flash/discussions/19

We hit the same wall on 2x NVIDIA DGX Spark (GB10, SM121), where fp8 KV matters most — 262K context on unified-memory parts, fp8 = 2x KV capacity.

### What works today (our GB10 deployment, for context)

We run GLM-5.3-Flash on 2x DGX Spark (CUDA 13.0, FlashInfer 0.6.18.dev20260819) with a locally patched vLLM day-0 image. **bf16 KV works end-to-end** after three narrow patches:

1. **Backend capability gate**: the SM90-only `FLASHINFER_MLA_SPARSE` sparse-MLA backend's supported-capability set extended from `{9}` to `{9, 12}`;
2. **FA2 kernel selection**: force FlashInfer's `BatchMLAPagedAttentionWrapper(backend="fa2")` on SM12x (the FA2 MLA path runs correctly on SM121; the SM90-specific paths do not);
3. **Versioned-gate scoping**: the FlashInfer version gates scoped so the SM12x branch doesn't fall into Hopper-only sub-paths.

So the sparse indexer, paged MLA attention, and NoPE (`kpe=0`) geometry are all fine on SM12x in bf16. Only the fp8 KV story is blocked, on both ends:

- **trtllm/native fp8 route**: `fp8_ds_mla` packed layout → the `pe_dim must be 64` assert above. This is an architecture-independent layout assumption, it bites SM120 and SM121 alike.
- **FlashInfer FA2 fp8 route**: FlashInfer's fp8 MLA FA2 kernels are genuinely Hopper-only — relaxing its SM90 guard yields a CUDA `invalid argument` at kernel launch on SM121. (Filed separately against flashinfer-ai/flashinfer; a vLLM-side layout fix is still needed for the day FlashInfer lands SM12x fp8 MLA, and would immediately help Hopper NoPE users too if any path exercises it there.)

### Proposal

Add a NoPE variant of the `fp8_ds_mla` packed layout — `pe_dim=0` alongside the existing `pe_dim=64` — through `concat_and_cache_mla` and the `FLASHINFER_MLA_SPARSE` / SM12x sparse-MLA backend plumbing. Two viable designs, in order of preference:

1. **True `pe_dim=0` layout variant.** The packed entry becomes scale + 512B of fp8 ckv with no rope tail. Cleanest: no wasted bytes (the 64-elem bf16 pe tail is 128B/token/layer ≈ 20% overhead at `ckv=512`), and the read-side kernels simply skip the pe segment. Requires templating/branching `concat_and_cache_mla` and the corresponding gather/attention consumers on `pe_dim ∈ {0, 64}`.

2. **Zero-padded pe fallback.** Keep the 64-wide pe slot and write zeros for NoPE models. Mathematically identity for NoPE attention (q_pe is empty/zero, so the pe dot-product contributes 0 to logits); touches only the cache-write path and model-runner shape handling, and reuses every existing pe_dim=64 kernel unmodified. Costs the 128B/token overhead but unblocks fp8 immediately and is a strictly smaller diff.

Either way, backend selection should stop offering/accepting `fp8_ds_mla` for NoPE models until the chosen variant exists — a startup-time "fp8 KV not yet supported for pe_dim=0, falling back to bf16 KV" is strictly better than the current mid-init CUDA-side assert (which also fires with `--kv-cache-dtype auto` in some configurations per the HF thread).

### Repro

- Model: `zai-org/GLM-5.3-Flash` (arch `glm5_next`, NoPE MLA: `kv_lora_rank=512`, `qk_rope_head_dim=0`, sparse indexer)
- Hardware A: 4x RTX PRO 6000 Blackwell (SM120), image `vllm/vllm-openai:glm53-flash`, `-tp 4 --kv-cache-dtype fp8` → `pe_dim must be 64 for fp8_ds_mla` (HF discussion 19)
- Hardware B: 2x DGX Spark GB10 (SM121), CUDA 13.0, patched day-0 image as described above, `--kv-cache-dtype fp8_ds_mla` → same assert in `concat_and_cache_mla`
- Same configs with bf16 KV: works (on SM121 only with the SM12x backend-gate patches above; stock selection currently finds no sparse backend, cf. #45317)

### Related

- https://huggingface.co/zai-org/GLM-5.3-Flash/discussions/19 — SM120 field report of this assert
- #45317 — sparse-MLA models can select no attention backend on SM121 (bf16 gate issue; our patch 1 above)
- #51920 — `FlashInferMLASparseSM120Impl` startup crash on GB10 with `fp8_ds_mla` (GLM-5.2, `pe_dim=64` — shows the pe_dim=64 fp8 route is otherwise viable on SM12x, which makes the NoPE layout the remaining gap)
- #35577 — TRITON_MLA fp8 KV request for SM12x (same "fp8 = 2x context on consumer Blackwell" motivation)
- flashinfer-ai/flashinfer#3170 — SM121 support audit (SM120/SM121 share compute capability; please gate `is_sm12x`, not SM120-only)

We can test candidate PRs on both bf16-working GB10 nodes and provide traces/benchmarks.
