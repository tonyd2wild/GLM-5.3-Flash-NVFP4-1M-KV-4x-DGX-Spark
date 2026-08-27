# GLM-5.3-Flash-NVFP4 TP2 Deployment Report — Reddie + Spark4

Deployment date: 2026-08-26
Endpoint: `http://100.113.138.96:8000/v1`
Model: `glm-5.3-flash` (262,144-token context)
Final status: **SERVING on Reddie + Spark4 as vLLM TP2, coherent output verified, tool calling verified**

Deployed by Knox (Claude, 5080). This was the hardest deploy this fleet has done: **seven distinct day-0 bugs**, each root-caused and fixed. Nobody else on the internet has GLM-5.3-Flash running on DGX Spark as of this writing (checked: vLLM/SGLang/FlashInfer trackers, HF discussions, NVIDIA forums, Reddit, X).

## 1. The model

- **LibertAIDAI/GLM-5.3-Flash-NVFP4** — 182 GiB, 120 safetensors, arch `Glm5NextForConditionalGeneration` (`glm5_next`), NVFP4 weight-only quant (routed-expert FFNs only; attention/KDA/shared-experts/router/embeddings stay BF16) of zai-org/GLM-5.3-Flash (320B total / 18B active, released the same day).
- Architecture is exotic: 45 layers = **34 KDA linear-attention layers** (recurrent state, no KV growth) + **11 DeepSeek-sparse-attention MLA layers** with a kpool top-k indexer (`index_topk=2048`, `index_kpool=4`), **NoPE MLA** (`qk_rope_head_dim=0`, `kv_lora_rank=512`), MoE with shared experts, mHC hyper-connections, native MTP head.
- Weights on Reddie `/var/tmp/glm-5.3-flash-nvfp4` (local NVMe), NFS-exported to Spark4 at the same path.

## 2. Engine and image

Both current vLLM stock images lack `glm5_next`; the vLLM PR authors publish a per-model day-0 image. SGLang's day-0 GLM image targets the base FP8 model, and the NVFP4 quant card names vLLM as the lane. Selected base:

```text
vllm/vllm-openai:glm53-flash-arm64-cu130
digest: sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce
vllm: 0.1.dev20051+g487ecf187 (registers Glm5NextForConditionalGeneration + Glm5NextMTPModel)
```

The final deployed image is a five-patch derivative, built up as failures were root-caused:

```text
radixark/vllm-glm53-flash:sm121-v7
```

Patch stack (each with a refuse-if-source-changed guard):
1. **SM90 NoPE sparse-MLA backend extended to SM121** (`flashinfer_mla_sparse_sm90.py`: capability gate 9 → {9,12}; wrapper `backend="fa3"` → fa2 off-Hopper; FlashInfer≥0.6.18 feature gate scoped to fp8 KV) + added to the capability-12 candidate list in `platforms/cuda.py`. Reason: the only stock SM12x sparse backend (`FLASHINFER_MLA_SPARSE_SM120`) hard-requires the packed `fp8_ds_mla` cache layout, which hard-requires DeepSeek's `pe_dim=64`; GLM's NoPE MLA has `pe_dim=0` → `concat_and_cache_mla` assert death in warmup. The FA2 kernel was probed directly on Reddie's GPU with GLM's real shape (32 heads/rank, ckv 512, kpe 0, page_size 1) before patching: PASS, and numerically correct vs a torch reference (rel err ≈ bf16 rounding).
2. **FlashInfer upgraded to the 0.6.18 nightly** (`flashinfer-python==0.6.18.dev20260819` + matching `flashinfer-cubin`; `flashinfer-jit-cache` 0.6.17 uninstalled so no stale AOT kernels load). Reason: 0.6.17's FA2 MLA scheduler produces **NaN for 64–256-row batches on SM121** (bisected standalone: 1–6 rows clean, 512+ clean, 64/128/256 NaN — normal prompts land exactly in the kill zone). 0.6.18 nightly: all bisect cases clean.
3. **NCCL restored to 2.30.7** — the FlashInfer nightly silently downgraded `nvidia-nccl-cu13` to 2.29.7, which fails `ncclCommInitRank` with "internal error" on the Spark IB fabric (two consecutive rendezvous deaths before diagnosis).
4. **nvidia-cutlass-dsl restored to 4.6.2** — the nightly also left a mixed 4.7.0/4.6.2 install that ICE'd the CuTeDSL warmup (`cute-to-nvvm` internal compiler error at ~91% load).
5. **PDL disabled on SM12x** (`is_arch_support_pdl`: `major >= 9` → `major in (9,10)`) + **indexer hardening**: kpool top-k destination `torch.empty` → `torch.full(-1)` (the top-k kernels only guarantee the first `min(k, valid)` entries; rows with fewer valid pools carried uninitialized int32 pool ids → in-range-but-bogus token indices → MLA gathers uninitialized KV → NaN lottery), and the pool-expansion Triton kernel now clamps `pid < pool_len` so any residual garbage id degrades to `-1` (handled natively by the sparse kernels). Both defects match open upstream issues (vLLM #51562 class, #53635 class).

## 3. Exact final launch

Launcher `~/launch-glm53-vllm-tp2.sh` on both nodes (identical; rank arg differs). Serve args:

```bash
vllm serve /models/glm-5.3-flash-nvfp4 \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --max-num-seqs 6 --block-size 2304 --moe-backend marlin \
  --enforce-eager \
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --distributed-executor-backend mp \
  --nnodes 2 --node-rank <0|1> \
  --master-addr 192.168.192.2 --master-port 29521 \
  [--headless on rank 1]
```

Docker: `--network host --ipc host --shm-size 32g --ulimit memlock=-1 --cap-add IPC_LOCK --device /dev/infiniband`, model ro-mounted, NCCL fabric env: `NCCL_NET=IB NCCL_IB_HCA=rocep1s0f0 NCCL_IB_GID_INDEX=3 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_ADDR_FAMILY=AF_INET NCCL_SOCKET_IFNAME=enp1s0f0np0` (+GLOO/TP/MN on the same iface), `NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 NCCL_CROSS_NIC=0 NCCL_IB_MERGE_NICS=0`, `TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Sequence (worker first, both after `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`):
```bash
# Spark4 (192.168.192.4), via Reddie hop:  ~/launch-glm53-vllm-tp2.sh 1
# wait ~25 s, confirm Up
# Reddie (192.168.192.2):                  ~/launch-glm53-vllm-tp2.sh 0
```

Key non-obvious flags:
- `--block-size 2304`: vLLM's hybrid mamba/attention block aligner picks 2176 on this backend, whose kpool storage block (544) tiles by 32 but not 64 — and DeepGEMM's paged-MQA on arch-12 fp8 accepts **only** 64-entry pool pages. 2304 = multiple of 256 (kpool·64) and of 128 (MLA alignment); the aligner pads the mamba page up ~6% to match.
- `--gpu-memory-utilization 0.85`: 0.78/0.80 starve the bf16 KV cache (0.4/0.64 GiB free vs 1.65 GiB needed at 131K). 0.85 credit: barrydeen's independently published recipe (repo has since vanished; archived locally).
- `--moe-backend marlin`: the quant card's "known-good on sm_121" fallback. FLASHINFER_CUTLASS also runs, but marlin was kept for the validated final config.
- KV cache dtype: **bf16** (~1 KB/token/MLA-layer). fp8 KV is supported by the patched backend once on FlashInfer ≥0.6.18 — staged as phase-2 (halves cache, ~doubles token capacity).

## 4. Every failure and fix (chronological)

| # | Boot | Symptom | Root cause | Fix |
|---|---|---|---|---|
| 1 | 1 | KV sizing death: 0.4 GiB free, 1.15 GiB needed | gmu 0.78 too low for 91 GB/rank weights + 131K bf16 KV | gmu 0.80 |
| 2 | 2 | Warmup death: `pe_dim must be 64 for fp8_ds_mla` | Only SM12x sparse backend forces DeepSeek packed cache; GLM is NoPE (pe=0) | Patch 1: SM90 NoPE backend → SM121 (FA2, probed first) |
| 3 | 3 | KV sizing death again (0.64 vs 1.65 GiB) | bf16 cache on the SM90 backend needs more than the packed fp8 layout | gmu 0.85 (credit barrydeen) |
| 4 | 4 | Warmup death: DeepGEMM assert `arch 12 … block_kv == 64` | Block aligner ignorant of arch-12 pool-page restriction (kpool 4 × 2176-block → 32-tiles) | `--block-size 2304` |
| 5 | 5 | SERVES but NaN logits → deterministic garbage ("locklock") | FlashInfer 0.6.17 FA2 MLA scheduler NaN on 64–256-row batches on SM121 (bisected standalone) | Patch 2: FlashInfer 0.6.18 nightly |
| 6 | v3 | `ncclCommInitRank: internal error` ×2 | Nightly silently downgraded NCCL 2.30.7→2.29.7; 2.29.x breaks on this IB fabric | Patch 3: pin NCCL 2.30.7 |
| 7 | v4 | ~91% load death: CuTeDSL `cute-to-nvvm` ICE | Nightly left mixed cutlass-dsl 4.7.0/4.6.2 | Patch 4: pin 4.6.2 family |
| 8 | v5/v7 | Residual garbage on some boots | (a) worker rank silently on a stale image — in-place remote sed edits of the worker launcher failed silently for several generations; (b) defense-in-depth: PDL race surface + indexer uninitialized top-k | Verify `IMAGE=` on BOTH nodes each generation; Patch 5 |

Diagnostic tooling built along the way (all reusable): direct FlashInfer kernel probes with the model's real geometry (run BEFORE patching arch gates — the Qwen sm121-qsa method), a standalone NaN bisect harness over batch shapes, and an env-gated forward-hook NaN localizer baked into a debug image (`GLM53_NAN_DEBUG=1` names the first module emitting non-finite values; it fingered `layers.3.self_attn.mla_attn` and exonerated MHC/TileLang via A/B probes vs torch reference at 5 and 4096 tokens).

Also hit and documented: two Codex-era traps confirmed still true — never let a new rank rendezvous with a dying one (teardown BOTH before either relaunch), and capture `docker logs` before `rm -f`.

## 5. Engine receipts (final v7 boot)

```text
Using FLASHINFER_MLA_SPARSE_SM90 attention backend out of potential backends:
  ['FLASHINFER_MLA_SPARSE_SM90', 'FLASHINFER_MLA_SPARSE_SM120']
Using 'MARLIN' NvFp4 MoE backend
vLLM is using nccl==2.30.7
Loading weights took 610.43 seconds
Available KV cache memory: 7.19 GiB
GPU KV cache size: 603,144 tokens
Maximum concurrency for 262,144 tokens per request: 2.30x
init engine (profile, create kv cache, warmup model) took 147.74 s
Application startup complete.
```

End-to-end boot: ~14 minutes (10 min weights over NVMe/NFS + ~2.5 min warmup JIT). GPU temps at idle-serve: 44 °C both nodes.

## 6. Validation

- Greedy `2+2` → coherent reasoning. Identity → "I'm GLM… developed by Z.ai". Long-form generation coherent. Logprobs finite and sane (during the NaN era this endpoint literally could not serialize logprobs).
- Tool calling: `tools`-bearing request returns structured `tool_calls` (`glob` with `{"path": "C:/tmp/*.html"}`), `finish_reason: tool_calls`.

## 7. Benchmark (3 streaming runs, 200 tokens, temp 0, thinking off)

| Run | TTFT | Total | Tokens | Decode tok/s |
|---:|---:|---:|---:|---:|
| 1 | 0.351 s | 14.05 s | 200 | 14.53 |
| 2 | 0.239 s | 14.16 s | 200 | 14.30 |
| 3 | 0.226 s | 13.93 s | 200 | 14.52 |

Median: **0.239 s TTFT, 14.30 tok/s decode**. No MTP yet — the checkpoint ships a native MTP head (`Glm5NextMTPModel` is registered in the image), and MTP is the expected ~1.5–1.6× decode lift (phase 2), as is fp8 KV (phase 2b, needs the 0.6.18 stack we now run). `--enforce-eager` also leaves CUDA-graph headroom on the table.

## 8. Fleet state (end of deploy)

| Node | Serving | H3 `comfy-h3-x` |
|---|---|---|
| Bluey | Qwen3.8-Flash-Next NVFP4 (SGLang TP2 head, :8000) — untouched | DOWN |
| Asusi | Qwen worker — untouched | DOWN |
| Reddie | **GLM-5.3-Flash NVFP4 (vLLM TP2 head, :8000)** | DOWN (stopped for GLM) |
| Spark4 | **GLM worker** | DOWN (stopped for GLM) |

DS4: DOWN everywhere (standing order: no restore). Both new models serving simultaneously. OMP registration `glm53-nvfp4` validated (262K ctx, thinking levels detected).

## 9. README notes for the repo

1. vLLM is the lane for the NVFP4 quant; stock vLLM lacks `glm5_next` until PR #53906 merges. Pin the day-0 image digest and ship the sm121-v7 patch stack (or upstream it: the SM120-backend NoPE gap, the block-aligner arch-12 pool-page rule, the indexer `torch.empty`, and the PDL gate all deserve upstream issues).
2. FlashInfer: require ≥0.6.18 on SM121 (FA2 MLA NaN below that) and **audit transitive pins after installing the nightly** — it downgrades NCCL (fabric-fatal) and skews cutlass-dsl (warmup-fatal).
3. Never edit the worker's launcher through chained quoting — copy the whole file, then `grep '^IMAGE'` on BOTH nodes before every launch. Two of tonight's "mystery" boots were a silent image mismatch between ranks.
4. Startup expectations: ~14 min; 120 shards; ~91 GB/rank weights; 603K KV tokens at gmu 0.85/262K; worker-first with full teardown of both ranks before either relaunch.
5. `max_tokens` includes reasoning tokens when thinking is on; pass `chat_template_kwargs: {"enable_thinking": false}` to disable per-request.
6. Phase-2 backlog: MTP (`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`, doubles load time — draft head reloads the checkpoint), fp8 KV cache, CUDA graphs, chunked-prefill kpool alignment (round scheduler chunks to multiples of 4 or accept mild selection-quality degradation on straddled pools).

## Credits

- Model: zai-org/GLM-5.3-Flash. Quant: LibertAIDAI/GLM-5.3-Flash-NVFP4 (whose card's sm_121 marlin fallback note and serve flags were used directly).
- **barrydeen** (glm53-flash-dgx-spark): the gmu 0.85 + 131K reference config and the quantization-coverage table. Used with credit per Tony's directive.
- vLLM PR #53906 authors for the day-0 image; FlashInfer for the 0.6.18 SM90-NoPE MLA path.

---

# Addendum: Phase 2 — fp8 KV cache + MTP-4 (same day, 2026-08-26 evening)

Final flagship config SERVING at 21:21 UTC: image **radixark/vllm-glm53-flash:sm121-v8**, serve args add
`--kv-cache-dtype fp8_e4m3 --kv-cache-memory 4445787956 --speculative-config '{"method":"mtp","num_speculative_tokens":4}'`.

## What happened between v7 and v8

1. **MTP-4 alone (bf16 KV) died on its first request.** Boot succeeded, served, then rank 0 was killed with no Python traceback. dmesg: `NV_ERR_NO_MEMORY` — the GB10 unified-memory OOM the fleet doctrine warns about. The draft head adds ~5 GB of weights; gmu 0.85 left no slack (vLLM's own startup log said usage exceeded the requested budget and suggested the exact `--kv-cache-memory` values). KV with bf16+MTP was down to 275,941 tokens (1.05x at 262K).
2. **fp8 KV turned out to be a two-line fix, not a Hopper exclusive.** FlashInfer's "FP8 kv_data_type requires SM90" guard hides a smem over-request: the fa2 fp8 branch forces CTA_TILE_KV=32 (written for Hopper's 228 KB smem); on GB10 that doubles the DISPATCH_SMEM_CONFIG tile and asks cudaFuncSetAttribute for 117,312 B against a 101,376 B opt-in max → cudaErrorInvalidValue. Fix (mla.cuh): cap `EFF_CTA_TILE_KV = min(CTA_TILE_KV, 32)` — fp8 gets TKV=16 on 100 KB devices (91,680 B, fits; 190 regs, 0 spills; grid = num_sm cooperative launch unchanged) — plus the `_core.py` gate accepting major 12. Verified standalone on the GPU: all batch shapes clean, rel-err 0.0054 vs fp32 reference. To our knowledge the first fp8 KV for a NoPE-MLA model on any consumer Blackwell device.
3. The two fixes are complementary: fp8 halves KV bytes, relieving exactly the memory pressure that killed the MTP boot; the pinned `--kv-cache-memory` removes the gmu edge-riding permanently.

## Final benchmarks (3 streaming runs, 200 tokens, temp 0, thinking off)

| Config | TTFT (median) | Decode (median) | Peak | KV tokens | Concurrency @262K |
|---|---:|---:|---:|---:|---:|
| v7 bf16, no MTP | 0.239 s | 14.30 tok/s | 14.53 | 603,144 | 2.30x |
| bf16 + MTP-4 | — | died (OOM) | — | 275,941 | 1.05x |
| **v8 fp8 + MTP-4** | **0.289 s** | **21.77 tok/s** | **22.69** | **507,041** | **1.93x** |

+52% decode over baseline. fp8 stores ~2x tokens per GiB (507K from a 4.14 GiB pinned budget vs 603K from 7.19 GiB bf16).

MTP metrics under real traffic: mean acceptance length 2.5–2.9; per-position acceptance ≈ [0.74, 0.47, 0.27, 0.15]; avg draft acceptance 37–47%. Position 4 is nearly free-riding → `num_speculative_tokens=3` is a candidate micro-tune. Raising `--kv-cache-memory` toward the ~10.7 GiB vLLM reports available would put fp8 KV near 1M tokens with MTP resident (untested; do as a controlled change — the OOM line on GB10 is real).

## Validation on the final config

Coherent greedy output, correct identity, structured tool calls (`glm47` parser, `finish_reason: tool_calls`), finite logprobs, no NaN. Thinking is ON by default (`glm45` parser separates it); disable per-request with `chat_template_kwargs: {"enable_thinking": false}` or bake `--default-chat-template-kwargs` at launch.
