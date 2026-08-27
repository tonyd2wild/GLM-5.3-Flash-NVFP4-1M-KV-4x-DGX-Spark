# Why the fleet kept "randomly" dying — SM121 crash forensics (2026-08-27)

A day of repeated engine deaths under real traffic turned out to be **two separate
diseases**, misdiagnosed as one. Both are now fixed and both fixes ship in this repo.
If your GB10/DGX Spark GLM deployment dies "randomly" under agent traffic, it is
almost certainly one of these.

## Disease 1 — persistent_topk hard-crashes ANY decode past ~24K context

**The big one.** Deterministic, unrelated to memory, and it masqueraded as OOM all day.

- The DSA sparse indexer routes decode-time top-k to `torch.ops._C.persistent_topk`
  whenever `select_k in (512, 1024, 2048)` (`sparse_attn_indexer_kpool.py`, ~line 812).
  GLM-5.3-Flash has `index_topk=2048, index_kpool=4` → `select_k=512` → always eligible.
- The kernel sizes its CTA grid to the sequence's candidate set. Past roughly ~20K tokens
  of context it oversubscribes GB10's SM budget (`total_ctas=124 > num_sms*occupancy=48`),
  and its `FilteredTopK` fallback **requires 128KB smem/block — SM121 has 99KB (101,376 B)**.
  There is no third path: it raises `RuntimeError` → `EngineDeadError` → endpoint dead
  until a full fleet relaunch.
- Receipts from two independent crashes: both killing requests were **MTP decode steps at
  32,760–32,763 computed tokens**; `kv_cache_usage=0.018` (1.8% of pool) at death; dmesg
  clean in the death window. Identical stack: `sample_tokens → speculator.propose →
  mtp_block → self_attn → indexer → persistent_topk → topk.cu:138`.
- Why every KV-size theory failed: 38G, 32G, 24G configs all "passed gates then died in
  real use" — because the gates prefilled 20K tokens (just UNDER the trigger) and real
  conversations crossed 24K within minutes. Pool size was never the variable.

**Fix (shipped):** `docker/sparse_attn_indexer_kpool_sm121.py` — gate the persistent
kernel on `torch.cuda.get_device_properties(0).multi_processor_count >= 78` so small-SM
parts take the existing `top_k_per_row_decode` fallback (no smem wall). Deploy by
bind-mount, no image rebuild:

```
-v $HOME/patches/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro
```

Verified: a 28.8K-token-context request that previously killed the engine on its first
decode step now completes ("DEEP-DECODE-OK"), engine healthy after.

**Upstream ask:** `topk.cu:138` should fall back to a multi-wave persistent variant
instead of raising when smem < 128KB.

## Disease 2 — phantom KV backing above ~16 GiB/rank on TP4

With disease 1 patched, a second failure surfaced: at 24 GiB/rank KV, combined load
(one 28.8K deep-decode + three concurrent 20K prefills) produced NVRM
`NV_ERR_NO_MEMORY` followed by `CUDA error: operation not permitted` on the head rank.
Same GB10 pattern documented in the TP2 KV hunt: the reservation "succeeds" but physical
backing at the pool's far edges does not exist; touching it under real load faults.

**Durable config: 16 GiB/rank = 2,516,582 fp8 KV tokens** (2.4x full-1M-context
concurrency), which passed the full gate suite. The head rank (API server + engine core
+ NFS export duty on top of its shard) is always the binding constraint.

## Gate design — what "passing" must mean on GB10

A config is production-ready ONLY after all of these, in one session, engine healthy after:

1. **Deep decode past the topk wall**: single request, 28–32K-token prompt, ≥100 decoded
   tokens (catches disease 1 — a 20K gate sits under the trigger and proves nothing).
2. **Concurrent prefills**: 3x simultaneous ~20K prompts (catches head-rank transients).
3. **Repeat deep decode** after the pool has been touched (catches phantom backing).
4. Vision probe (the checkpoint's stock chat template can't take images — see the
   template notes in the README).
5. `/health` returns 200 after all of the above. **Never probe `/v1/models` for
   liveness — it returns 200 with a dead engine.** `/health` returns 503 on
   `EngineDeadError`.

## Self-healing (shipped: `fleet_watchdog.sh`)

vLLM v1 has no engine-restart path, and Docker restart policies cannot recover a
multi-node mp deployment (workers exit 0 on head death; the dead head often never exits;
blind restarts race the stale TCPStore rendezvous). The watchdog runs on the head as a
systemd service: probe `/health` every 60s; on 3 consecutive failures, tear down ALL
ranks, run the GB10 memory ritual (drop_caches + compact_memory), relaunch workers-first,
wait for healthy. A crash becomes a ~15-minute unattended recovery.

## Timeline of the day (for the record)

| Time (UTC) | Event | Real cause |
|---|---|---|
| ~04:58 | TP2 prod died mid-prefill (19K prompt) | genuine NVRM pressure; fixed by flush-before-launch |
| 05:00–07:30 | TP2 KV hunt: 672K record, 6G+ first-touch deaths | phantom backing (disease 2, TP2 flavor) |
| ~14:12 | TP4 32G died "under traffic" | **disease 1** (decode at 32.8K ctx) |
| ~14:34 | TP4 24G died "under traffic" | **disease 1** (identical signature) |
| ~15:06 | TP4 24G + patch died under combined load | **disease 2** (24G phantom edge) |
| 15:21 | TP4 16G + patch: all gates pass | — |
