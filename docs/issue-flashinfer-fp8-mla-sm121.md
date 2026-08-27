# Draft issue for flashinfer-ai/flashinfer

## Title

[Feature] FP8 `kv_data_type` support for `BatchMLAPagedAttentionWrapper` (FA2 path) on SM12x (SM120/SM121, consumer Blackwell / GB10) — incl. NoPE (`head_dim_kpe=0`)

## Body

### Motivation

We are running **GLM-5.3-Flash** (NoPE MLA + DeepSeek-style sparse indexer, `kv_lora_rank=512`, `qk_rope_head_dim=0`) on **2x NVIDIA DGX Spark (GB10, SM121)** via a locally patched vLLM. After extending vLLM's `FLASHINFER_MLA_SPARSE` backend from SM90-only to SM12x, the **bf16 KV cache path works end-to-end on SM121** through `BatchMLAPagedAttentionWrapper` with `backend="fa2"`. The remaining gap is **fp8 KV cache**: at 262K context on a 128GB unified-memory part, fp8 e4m3 KV would double usable KV capacity (and effective batch/context), which is the difference between a comfortable long-context deployment and constant preemption.

Today fp8 `kv_data_type` for MLA is hard-gated to Hopper, and the gate appears to be genuine (not just conservative), so this is a kernel-support request rather than a "please relax the check" request.

### Current behavior

1. **The guard.** `flashinfer/mla/_core.py` rejects fp8 KV up front:

   > `FP8 kv_data_type for MLA requires an SM90 GPU`

   so on SM121 any `plan()` with `kv_data_type=torch.float8_e4m3fn` fails immediately.

2. **Relaxing the gate does not work** (expected, but confirming it is load-bearing): with the SM90 check patched out, the FA2 fp8-MLA kernel launch fails on SM121 with CUDA `invalid argument` at dispatch. So the fp8 path is compiled/tuned Hopper-only (SM90a SASS / smem layout assumptions), and there is currently no SM12x fp8 MLA kernel behind this wrapper.

3. **bf16 baseline works** on the *same* wrapper, same shapes, same machine — so the plumbing (plan/run, paging, sparse indexer interaction, NoPE `head_dim_kpe=0`) is all fine on SM121; only the fp8 kernel is missing.

### Repro

Environment:

- 2x NVIDIA DGX Spark (GB10, SM121, aarch64, unified memory)
- CUDA 13.0
- FlashInfer `0.6.18.dev20260819` (nightly), JIT path
- Torch bf16 q, fp8 e4m3 KV

Shape (GLM-5.3-Flash NoPE MLA):

- `head_dim_ckv = 512` (`kv_lora_rank`)
- `head_dim_kpe = 0` (NoPE — no RoPE sub-head)
- `num_heads = 32`
- `page_size = 1`
- `q_data_type = torch.bfloat16`, `kv_data_type = torch.float8_e4m3fn`

```python
import torch, flashinfer

wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
    torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
    backend="fa2",
)
wrapper.plan(
    qo_indptr, kv_indptr, kv_indices, kv_len_arr,
    num_heads=32, head_dim_ckv=512, head_dim_kpe=0, page_size=1,
    causal=True, sm_scale=sm_scale,
    q_data_type=torch.bfloat16,
    kv_data_type=torch.float8_e4m3fn,   # <- raises "FP8 kv_data_type for MLA requires an SM90 GPU"
)
```

- As shipped: `ValueError` from the SM90 guard in `mla/_core.py`.
- With the guard relaxed to admit `major == 12`: kernel launch fails with `CUDA error: invalid argument` on SM121.
- Identical call with `kv_data_type=torch.bfloat16`: works, correct output, stable under load (we run it in production behind vLLM).

### Request

Support fp8 e4m3 `kv_data_type` in the `BatchMLAPagedAttentionWrapper` FA2 path on SM12x (SM120 and SM121), including the NoPE configuration `head_dim_kpe=0`.

Notes that may make this cheaper than it looks:

- Per the SM121 support audit (#3170), SM120/SM121 share compute capabilities, so one FA2 fp8-MLA port should cover RTX PRO 6000 Blackwell (SM120) and GB10 (SM121); please gate with an `is_sm12x`-style predicate rather than SM120-only.
- The XQA MLA decode kernel (`mla_sm120.cu`, see #2655) is already **fp8-e4m3-only on SM120** — i.e., fp8 MLA math on SM12x tensor cores is a solved problem in-tree; what's missing is fp8 KV in the general FA2 paged wrapper (which is what vLLM's sparse-MLA backend for GLM-5.3-Flash / DeepSeek-V3.2-family models drives, and which supports incremental prefill, not just decode).
- The trtllm-gen `fp8_ds_mla` route is not a substitute here for two reasons: SM12x cubins are not shipped (NVIDIA/TensorRT-LLM#11799 tracked the artifact gap), and the `fp8_ds_mla` packed layout hardcodes `pe_dim=64`, so NoPE models (`head_dim_kpe=0`) are rejected regardless of arch (see the vLLM-side `concat_and_cache_mla` assert; reported by RTX PRO 6000 users at https://huggingface.co/zai-org/GLM-5.3-Flash/discussions/19).

If a dedicated fp8 FA2 kernel is not on the roadmap, an interim documented position (e.g., "SM12x MLA fp8 KV: use per-page dequant-to-bf16 epilogue" or similar) would still help downstream projects stop rediscovering this boundary one CUDA `invalid argument` at a time.

### Downstream context

- vLLM sparse-MLA backend selection on SM121: vllm-project/vllm#45317, #51920
- GLM-5.3-Flash fp8 KV failure reports on consumer Blackwell: https://huggingface.co/zai-org/GLM-5.3-Flash/discussions/19

Happy to test nightlies/branches on GB10 hardware (we have 2x DGX Spark on tap and an existing repro harness).
