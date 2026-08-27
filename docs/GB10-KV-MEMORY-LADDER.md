# The GB10 KV-Memory Ladder: why your "free" memory is a lie

Empirical study from 2026-08-26, six controlled boots of GLM-5.3-Flash (fp8 KV + MTP-4,
~96.6 GiB weights + ~5 GB draft head per rank) on 2x DGX Spark, TP2.

## The ladder

| `--kv-cache-memory` | KV tokens | Result |
|---:|---:|---|
| 4.14 GiB (vLLM's own suggested number) | 507,041 | **WORKS — every time (3/3)** |
| 5.5 GiB | 672,606 | allocated on head, worker died (NVRM OOM) |
| 6.5 GiB | 796,779 | died — NVRM OOM on head |
| 7.5 GiB | 920,953 | allocated on head twice, a rank died every time (5 attempts total ≥5.5 GiB, 0 survived) |

Mitigations tried, in escalating order — none moved the ceiling:
1. `drop_caches` before launch (baseline ritual) — defeated: the 182 GiB shard read refills page cache during load.
2. **Cache-flusher sidecar** ([`cache_flusher.sh`](../cache_flusher.sh)) holding Cached < 40 GiB through the load — got the head's 7.5 GiB allocation to succeed for the first time, but a rank still died in warmup.
3. **Full node reboot** before the attempt (fresh 117 GiB free pool) — the freshly-rebooted node still OOM'd during warmup.
4. **cgroup memory cap** (`--memory 112g`) forcing continuous in-container reclaim — the *other* node OOM'd at the same instant its KV line printed.

## The mechanism (confirmed by live measurement + NVIDIA's own docs)

GB10 has no VRAM: every GPU allocation is host-DRAM through the NVIDIA kernel driver
(`_memdescAllocInternal`). The driver's allocation path **fails fast instead of reclaiming
clean page cache** — it needs the request covered by *truly free* pages (MemFree), not
"available" ones (MemAvailable). Live sampling during our boots showed MemFree oscillating
at **0.7–4.4 GiB** through shard load with everything else parked in page cache; the one
successful 4.14 GiB allocation squeaked through riding a concurrent ~4.4 GiB kernel reclaim.
Anything bigger loses that race on some rank, some boot — `NVRM: NV_ERR_NO_MEMORY ...
_memdescAllocInternal @ mem_desc.c:1359` in dmesg (check `journalctl -k`, dmesg rotates).

A compounding trap: since vLLM PR #35356, integrated-GPU "free memory" is
`psutil.virtual_memory().available` — which **includes reclaimable page cache**. So vLLM
reports ~109 GiB free and suggests KV sizes the driver then can't deliver. The only number
that ever worked was the conservative one vLLM's profiler computed (`--kv-cache-memory`
suggestion in the startup log). **Use that number. It is not advisory.**

NVIDIA references: KB a_id 5776 ("Application is experiencing memory issues even though I am
within the memory capacity"), KB a_id 5728 ("Unexpected Available Memory Reporting"), and the
forum thread on `_memdescAllocInternal` cascades under UMA fragmentation.

## Practical rules for big models on Spark pairs

1. Take vLLM's suggested `--kv-cache-memory` verbatim. Every larger value we tried died.
2. Run the cache flusher through load anyway — it demonstrably improves the allocation's odds
   and costs nothing.
3. NFS-loading workers are the weak node: NFS client memory resists reclaim harder than local
   page cache. Expect the worker to fail first.
4. Capture `docker logs` BEFORE any teardown; NVRM failures surface minutes after the real event.
5. Paths to bigger pools, untested/future: NVIDIA's ongoing GB10 UMA OOM-handling driver work,
   dropping the MTP draft head (frees ~5 GiB and its allocation pressure: bf16 no-MTP hit 603K,
   fp8 no-MTP would exceed 1M), O_DIRECT model loading (fastsafetensors `nogds` — currently
   hangs in multi-node), and decode-context-parallel KV sharding once glm5_next's sparse stack
   supports it.

## Update: InstantTensor changes the game (2026-08-27, ~02:00 UTC)

Prompted by jack6464's DGX-Spark forum post, we added the **InstantTensor** direct-I/O loader
(`pip install instanttensor` + `--load-format instanttensor`; Dockerfile v9). Results on TP2:

1. **Load time: 610 s → 40–100 s** (local NVMe ~40 s at ~5 GB/s; the NFS worker ~70–100 s).
   Every experiment now costs ~6 minutes instead of ~21.
   ⚠️ Its pip install **downgrades `nvidia-nccl-cu13` to 2.29.7** (the fabric-fatal version,
   same sabotage as the FlashInfer nightly) — the v9 Dockerfile re-pins 2.30.7 in the same layer.
2. **The ladder's first wall falls**: because direct I/O never touches the page cache
   (Cached stays ~1–9 GiB through the whole load vs 24 GiB saturated before), the previously
   impossible **7.5 GiB / 920,953-token KV slab now ALLOCATES successfully on both ranks, every
   attempt.** This confirms the page-cache mechanism end-to-end.
3. **A second, deeper wall exists**: in three post-InstantTensor attempts (cgroup 112g, 118g,
   and uncapped), a rank still died **silently ~1–2 minutes after successful allocation** —
   exit code None, no NVRM error, no OOM-kill, nothing in dmesg. With ~97 GiB weights + 7.5 GiB
   KV per rank, something in the driver/warmup path kills the process without a trace. Unresolved;
   likely needs NVIDIA's ongoing GB10 UMA work or smaller per-rank weights (TP4 sidesteps it
   entirely — its 9 GiB slabs run stable at ~50 GiB weights/rank).
4. **Practical TP2 recipe**: v9 image, `--load-format instanttensor`, and vLLM's suggested
   `--kv-cache-memory` (4.14 GiB → 507K tokens) — the stable pool with 15x faster boots.
   The old cache-flusher sidecar becomes optional insurance rather than a requirement.

Credit: jack6464's post on the NVIDIA developer forum (GLM-5.3-Flash thread, post #22) for
pointing at InstantTensor on this hardware.
