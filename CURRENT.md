# CURRENT recipe (read this first)

**GLM-5.3-Flash NVFP4, tensor-parallel 4 across four DGX Spark, 1M context, DFlash2 drafter, CUDA graphs.**
Updated 2026-09-02. If a file in this repo disagrees with this page, this page wins and the file is history.

## The one launcher

`launch-glm53-tp4-24g.sh <rank>` on every node, **workers first: rank 3, then 2, then 1, then rank 0 (the head)**.

| rank | node | ring IP | role |
|---|---|---|---|
| 0 | Reddie | 192.168.192.2 | head, serves `:8000`, model id `glm-5.3-flash` |
| 1 | Spark4 | 192.168.192.4 | worker |
| 2 | Asusi | 192.168.192.3 | worker |
| 3 | Bluey | 192.168.192.1 | worker |

Before launching on every node: `flusher-unconditional.sh` running, `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`, and after any reboot check `nvidia-smi --query-gpu=clocks.sm --format=csv` under load (a post-reboot node can sit at ~611 MHz and halve every number).

## What it runs

| | value |
|---|---|
| image | `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` |
| weights | `/var/tmp/models/GLM-5.3-Flash-NVFP4-redhat` = **RedHatAI/GLM-5.3-Flash-NVFP4 (compressed-tensors)**. ModelOpt and abliterated NVFP4 builds corrupt token IDs (vLLM #54150); the launcher refuses them unless `ALLOW_MODELOPT=1`. |
| drafter | `incoai/GLM-5.3-Flash-DFlash2`, `num_speculative_tokens: 7` |
| context | `--max-model-len 1048576` |
| KV | `--kv-cache-dtype fp8_e4m3 --kv-cache-memory 25769803776` (24 GiB per rank), `--block-size 2304` |
| concurrency | `--max-num-seqs 64 --max-num-batched-tokens 16384` |
| graphs | `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'` (NOT `--enforce-eager`; eager belongs only to the b12x/NVFP4-KV lane and the topkfix image) |
| MoE | `--moe-backend marlin`, `--gpu-memory-utilization 0.85` |
| vision | on, `chat_template_mm.jinja` mounted from the weights dir |
| watchdog | `fleet_watchdog.sh` (hard-codes the launcher name; do not rename the launcher) |

## What to expect (measured 2026-09-02, this exact config, isolated, temperature 0)

| | |
|---|---|
| boot | healthy in ~16 min (13 min load + 96 s graph capture, 6.45 GiB of graphs) |
| KV pool | `GPU KV cache size: 3,834,498 tokens` |
| real prompts, single stream | prose **31** tok/s, narrative 32, code **75**, reasoning 73, JSON 80, HTML 94, summary 60 (40-prompt battery, median) |
| real prompts, mixed load | 4 streams 74 tok/s aggregate (first token 0.94 s); 16 streams 100 tok/s (first token 2.2 s) |
| fresh 1.6K prompt, first token | 0.91 s single stream |
| cold prefill | 2.0K tok/s at 7K tokens, 3.0K at 28K, 3.8K at 110K, 4.8K at 182K (fresh prompt, no cache) |
| power at 16 streams | 156 W across four GPUs, 2.99 tokens per joule |
| ceiling (counting prompt, max draft acceptance, not a decode number) | 100 tok/s single stream, 817 tok/s aggregate at 48 streams |

Side by side with DeepSeek V4 Flash Vision at the same TP4 config, same day: https://github.com/tonyd2wild/GLM-5.3-Flash-EXL3-on-2x-NVIDIA-DGX-Spark/blob/main/results/h2h_tp4.md

## How we quote numbers

Decode is quoted from real prompts (prose, code, and the other categories above). The counting prompt is reported only as the drafter's acceptance ceiling and labeled as such. Prefill is quoted cold, from a fresh prompt at a fixed length, never from a cached context. Every number carries the prompt type and the temperature.

## Sweep harness

`tools/exp-glm53-tp4.sh` is the same recipe with every knob exposed as an env var (`EAGER`, `CUDAGRAPH_MODE`, `MAX_NUM_SEQS`, `MAX_BATCHED`, `SPEC_JSON`, `EXP_NAME`). Defaults equal the recipe above. Use it for one-knob experiments; serve with the launcher.

## Do not use (history, moving to `archive/` in the next commit)

`launch-glm53-tp4-dflash2-topkfix.sh` and `docker/topkfix/` (eager-only alternate lane, gmu 0.72, seqs 16), `docker/Dockerfile.glm53-sm121-v2` and `-v9` as build targets (v8 is the published base), `overlay-dflash2/` (duplicate of `docker/dflash2-overlay/`), `docs/DEPLOY-REPORT.md` (a TP2 report), `docs/GB10-KV-MEMORY-LADDER.md`, `docs/BENCH-C1-C6-DFLASH2.md`, `docs/NVFP4-KV-BUILD-SPEC.md`, `probes/bench_c1c6.py`, `probes/bench_glm53.py`.

Two Sparks only: the sibling repo `GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark` (TP2, 262K, enforce-eager, KV pinned 6 GiB).
