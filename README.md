# GLM-5.3-Flash NVFP4 · 1M-Token KV · 4x DGX Spark · 36 tok/s

[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B / A18B MoE, released 2026-08-26) serving across **all four NVIDIA DGX Spark (GB10) nodes** at tensor-parallel 4, using the [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) quant — deployed the same day the model dropped.

**As far as we can tell: the first TP4 `glm5_next` deployment outside NVIDIA B200 hardware, the first fp8 KV cache for a NoPE-MLA model on any consumer Blackwell part, and a 1.26-million-token KV pool on $16K of desk hardware.**

## Numbers

| Metric | TP4 flagship |
|---|---|
| Decode | **35.7 tok/s** generic median · **up to 63.8 tok/s** warmed on structured/agentic output (MTP acceptance runs hot — [re-bench below](#warmed-streaming-re-bench--the-357-is-a-floor-not-the-ceiling-2026-08-27)) |
| TTFT | **0.204 s median** |
| Context | **1,048,576 (model-native 1M) — launcher default** · the 1.26M-token KV pool physically holds a full 1M-token request. Cap --max-model-len lower (e.g. 300000) for a snappier multi-user endpoint |
| KV pool | **1,263,415 tokens fp8** — 4.82 concurrent full-context requests (or one ~1M-token context) |
| Speculative decode | native MTP head, 4 draft tokens |
| KV dtype | fp8_e4m3 (our FlashInfer SM12x unlock — see below) |
| Boot | ~12 min (quarter weights per rank) |

Progression on the same hardware pair count: 14.3 tok/s (day-1 bf16 TP2) → 21.8 (fp8+MTP TP2) → **35.7 (TP4)**.

### Warmed streaming re-bench — the 35.7 is a floor, not the ceiling (2026-08-27)

The 35.7 median above is a **generic 200-token greedy** number. Re-benched **warmed + streaming** (decode = `(completion_tokens − 1) / (t_last − t_first)`, measured off-box over the tailnet, temp 0), throughput is strongly **content-regime dependent** because the MTP head's draft-acceptance rate swings with how predictable the output is:

| Prompt | Decode (warmed) |
|---|---:|
| 🔢 **count 1→100** (structured) | **63.8 tok/s peak · ~61 median** (6 runs) |
| 🔤 alphabet ×8 | ~60 |
| 💻 code continuation | ~53 |
| 📝 freeform 400-word essay | ~37 |

So the honest picture: **freeform prose ≈ 37 (that's where the "36" comes from), but structured / list / code / tool-argument output — what agents actually generate — runs 53–64 tok/s** as MTP acceptance approaches 100%. Real agentic workloads live in the high-acceptance zone, so **~55–64 tok/s is the number that matters in production**, roughly 1.7× the headline. TTFT stays ~0.2 s across all regimes.

## What's in here

- [`launch-glm53-vllm-tp4.sh`](launch-glm53-vllm-tp4.sh) — the 4-rank launcher (head serves `:8000`; run worker-first: rank 3 → 2 → 1, head 0 last). Full NCCL fabric env included.
- [`cache_flusher.sh`](cache_flusher.sh) — **required sidecar** on every node during boot. GB10's driver fails allocations against page-cache-full memory; this holds the cache down through the 182 GiB shard read. Mechanism + measurements: [docs/GB10-KV-MEMORY-LADDER.md](docs/GB10-KV-MEMORY-LADDER.md).
- [`docker/`](docker/) — the **sm121-v8 image patch stack** (8 Dockerfiles, applied v1→v8 on the day-0 `vllm/vllm-openai:glm53-flash-arm64-cu130`). The vendor image dies five different ways on GB10; these fix: the NoPE-MLA backend gap, a FlashInfer FA2 NaN kernel bug, two dependency downgrades the FlashInfer nightly sneaks in (NCCL, cutlass-dsl), a PDL race surface, uninitialized indexer top-k memory, and the fp8-KV shared-memory tile bug.
- [`docs/DEPLOY-REPORT.md`](docs/DEPLOY-REPORT.md) — every failure, root cause, and receipt from the deploy day (eight kernel-level bugs).
- [`docs/`](docs/) issue drafts — upstream-ready reports for the FlashInfer fp8-MLA SM12x gap and the vLLM NoPE `fp8_ds_mla` layout gap.
- [`probes/`](probes/) — the debugging kit: probe a FlashInfer kernel with your model's real geometry before trusting arch-gate patches, a NaN bisect harness, kernel-vs-torch A/B, and the benchmark script.

## Quickstart (4 nodes)

One node owns the weights on local NVMe and NFS-exports them; the other three mount at the same path.

```bash
# every node: image + weights visible + flusher
docker load < sm121-v8.tar          # or build docker/Dockerfile.glm53-sm121* v1->v8 in order
ls /var/tmp/glm-5.3-flash-nvfp4/config.json
nohup ./cache_flusher.sh > flusher.log 2>&1 &
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

# workers first, ~20 s apart, head LAST
./launch-glm53-vllm-tp4.sh 3   # worker
./launch-glm53-vllm-tp4.sh 2   # worker
./launch-glm53-vllm-tp4.sh 1   # worker
./launch-glm53-vllm-tp4.sh 0   # head — serves http://<head>:8000/v1
```

Edit the rank→IP map at the top of the launcher for your fabric. Thinking is off by default (`--default-chat-template-kwargs`); re-enable per request with `chat_template_kwargs: {"enable_thinking": true}`. Tool calling ships enabled (`glm47` parser).

## Fast loading: InstantTensor (added 2026-08-27)

**Status: experimental — 15x load speedup measured, but NOT stable in our multi-node TP2 topology** (a rank dies silently ~1 min post-load in every v9 boot, at any KV size, including budgets that are 100% stable on v8; cf. eugr/spark-vllm-docker#29 for the same multi-node class of problem). The shipped launchers do NOT enable it; the stable image remains v8. The v9 image adds the InstantTensor direct-I/O loader (`--load-format instanttensor`): loads drop from ~10 minutes to 40-100 seconds. Two things to know: its pip install silently downgrades NCCL to a fabric-fatal version (v9 re-pins 2.30.7 in the same layer), and because direct I/O never fills the page cache, it also defeats the first layer of the GB10 KV-allocation wall -- the full story and the remaining (unsolved) second wall are in [docs/GB10-KV-MEMORY-LADDER.md](docs/GB10-KV-MEMORY-LADDER.md). Credit: jack6464 (NVIDIA forum) for the pointer.

## Hard-won rules (each one cost us a boot)

1. Tear down **all** ranks before relaunching **any** — a fresh rank that rendezvouses with a dying one hangs.
2. Verify `grep '^IMAGE'` matches on every node before every launch; copy launcher files whole, never sed over ssh.
3. Run the cache flusher during every boot; on TP2-class per-rank weights, take vLLM's suggested `--kv-cache-memory` verbatim (the ladder study shows why bigger dies).
4. Reboot a node that has been through many boot cycles — GB10's driver accumulates allocation-pool degradation that eventually kills even proven configs.
5. Capture `docker logs` before `docker rm -f`.

## Why TP4 (beyond speed)

At TP2 each rank carries ~97 GiB of weights and the GB10 driver can only reliably grant ~4.5 GiB of KV afterward (measured across six controlled boots — see the ladder study). At TP4, weights drop to ~50 GiB per rank and the KV ceiling simply dissolves: the 9 GiB slab allocates with ~60 GiB of slack, giving every one of the six request slots full 262K context simultaneously.

## vLLM v0.28.0 status (checked 2026-08-27)

**Not viable for GLM-5.3 yet**: the `glm5_next` architecture is NOT in the v0.28.0 release
(PR vllm-project/vllm#53906 still open/unmerged at check time), and no rebased day-0 image
exists (all `vllm/vllm-openai:glm53-flash*` tags still date to the original 2026-08-26 push).
The day-0 image used here is itself a main-branch dev snapshot (`0.1.dev20051`) cut around
the 0.28 branch point -- i.e. this stack already runs 0.28-era engine code plus the GLM
support 0.28 lacks. Upgrade path when it opens: watch the PR and the Docker Hub tags; the
patch stack here is guarded string-patches that apply-or-refuse loudly, so porting to a new
base is mechanical (apply v1->v10 in order, fix whichever guards fire, ladder through the
experiment lane before production).

## Credits

Model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · Quant: [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) · barrydeen (gmu reference + quant table) · vLLM [PR #53906](https://github.com/vllm-project/vllm/pull/53906) authors for the day-0 image · FlashInfer 0.6.18 · Deployed and debugged by Knox (Claude) for [@tonyd2wild](https://github.com/tonyd2wild). Companion deep-dive repo: [GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark).
