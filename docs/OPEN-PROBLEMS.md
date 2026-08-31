# Open problems — things we broke and could not fix

> **Updated 2026-08-29.** Three entries below have since been resolved on the TP4 fleet.
> They are kept with their original text and a **RESOLVED** note, because the reasoning that
> was wrong is as useful as the fix:
>
> | # | was | now |
> |---|---|---|
> | — | "phantom KV backing above 16 GiB is a hardware ceiling" | **page cache.** An *unconditional* flusher took 24 GiB/rank through the full gate suite (3,895,606 tokens, +54.8 %). A threshold-triggered flusher can sit below its threshold and still starve the NVRM allocator. |
> | 2 | NVFP4 KV dies on chunked prefill | **serves.** `nvfp4_ds_mla` + `--max-num-batched-tokens 8192` ran stably at 270K context on this fleet. The 2048-token auto-derived batch budget looks implicated in the original failure. |
> | 7 | CUDA graphs untested with DFlash2 | **works.** `cudagraph_mode: FULL` runs with the DFlash2 drafter. Still check `spec_decode_num_accepted_tokens_per_pos_total` after any graph-enabled boot — vllm#53030 pins acceptance at exactly 1.00 silently. |
>
> Problem 1 (the rank-1 KV asymmetry) and problem 3 (InstantTensor multi-node) are **still open**
> and are the highest-value contributions.


Everything here is reproducible on our hardware and unresolved as of 2026-08-28. If you
have a Spark pair (or any GB10/SM121 box) and want to attack one, these are the real
frontiers — PRs very welcome. Each entry gives the symptom, what we already ruled out, and
the cheapest next probe, so you do not repeat our dead ends.

Hardware context for all of it: 2× or 4× NVIDIA DGX Spark (GB10, SM121), 121.7 GiB
**unified** memory per node (no discrete VRAM — the GPU allocates from system RAM),
vLLM `0.1.dev20051+g487ecf187`, GLM-5.3-Flash NVFP4 (~181 GiB, ~91 GiB/rank at TP2).

---

## 1. The TP worker rank profiles 4–5 GiB less KV headroom than the head

**This is the one that caps our pool, and we never explained it.**

vLLM builds the KV pool from `min(num_blocks)` across ranks
(`kv_cache_utils.py:2554`), so the worker's number is the one that counts. Measured on
both of our node pairs, in the same boot:

| pair | rank 0 (head) | rank 1 (worker) | gap |
|---|---|---|---|
| Bluey / Asusi | 8.54 GiB | 5.54 GiB | −3.00 |
| Reddie / Spark4 | 6.25 GiB | 2.29 GiB | −3.96 |
| Reddie / Spark4, **no drafter** | 11.07 GiB | 6.27 GiB | −4.80 |

Closing it would be worth roughly **+54 % pool**.

Ruled out:
- **Weights** — both ranks log the identical `Model loading took 90.65 GiB`.
- **NFS** — one pair reads weights over NFS from the head, the other has them local on
  both nodes (`findmnt` confirms ext4/nvme). The gap appears in both.
- **The DFlash2 drafter** — removing `--speculative-config` entirely does not close it
  (the gap actually widens slightly).
- **Startup free memory** — both nodes report ~110 GiB free before load.
- **Node identity** — it follows the *rank*, not the machine.

Next probe: boot once with `VLLM_LOGGING_LEVEL=DEBUG` and capture `gpu_worker.py:552-560`
on both ranks — `peak_activation_memory` and `non_kv_cache_memory` per rank. That names
which term differs. Candidates we could not distinguish: NCCL buffer allocation asymmetry,
CUDA context differences, or something in the `mp` executor's worker path.

---

## 2. NVFP4 KV cache: serves and drafts, dies on chunked prefill — **RESOLVED 2026-08-29**

> **Resolved.** On the TP4 fleet, `nvfp4_ds_mla` with `--kv-cache-dtype-skip-layers sliding_window`
> and `--max-num-batched-tokens 8192` served a 270K-context deployment with the DFlash2 drafter
> without the chunked-prefill kill. The original failures ran the auto-derived 2048-token batch
> budget, which is the leading suspect. Original analysis below.


The [drowzeys NVFP4-KV stack](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated)
(4-bit `nvfp4_ds_mla` MLA cache, 368 B/token/layer vs our 512) ported cleanly and **worked**
— 334,161-token pool, 35.9 tok/s, 0.563 draft acceptance, aux layers wired correctly.

Then any prompt long enough to require chunked prefill **hard-kills the rank-0 worker**:

| prompt | result |
|---|---|
| 1,468 tok | OK |
| 2,977 tok | OK (single chunk) |
| **5,897 tok** | **worker killed** |
| 49,891 tok | worker killed |

Both failures die on the **first** scheduled chunk (`num_computed_tokens=0`,
`total_num_scheduled_tokens=4089`). No Python traceback, no CUDA error, no OOM-killer
entry — the head reports `RuntimeError: cancelled` out of `shm_broadcast.py`.

Prime suspect: on this stack the DFlash2 drafter cannot achieve an exact KV page fit
(`2,637,824 / 2048 = 1288 = 8 × 161`, not a multiple of any kernel block size), so it falls
back to a **standalone** per-layer drafter KV path that never runs on the fp8 route. That
path had never executed on real hardware before this port.

Next probe, in order: (a) force the drafter to block 64 so no kernel-block split occurs;
(b) if it still dies, drop `--speculative-config` on the same image and retest 5,897
tokens — that isolates the drafter from the b12x kernels entirely.

Worth knowing before you invest: measured honestly, the format win is **1.36×**, not the
1.78× the byte counts suggest, and **1.17×** once the standalone drafter's pages are
counted. Our fp8 route is already on the 512 B/token NoPE record, not the 656 B packed one.

---

## 3. InstantTensor direct-I/O loader: 15× faster loads, silently unstable multi-node

`--load-format instanttensor` drops weight loading from ~10 minutes to 40–100 seconds and
leaves the page cache empty (which also sidesteps problem 5 below). In **all four** of our
TP2 boots with it, a rank died silently ~1 minute after loading completed — exit code
`None`, nothing in `dmesg`, at every KV budget including one that is otherwise rock stable.

This matches the known instability class for direct-I/O loaders on Spark
(eugr/spark-vllm-docker#29 reports fastsafetensors hanging in cluster mode). Single-node or
TP4 may fare better; we did not test those.

Fixing this is the single highest-value contribution to this repo — it removes the
15-minute boot tax that makes every other experiment here expensive.

---

## 4. UVM driver livelock under memory pressure (workaround, no fix)

On these unified-memory nodes, if the kernel pages vLLM out during load or warmup, the
process enters an unrecoverable spin:

- one `VLLM::Worker` thread at 100–210 % CPU, minutes of accumulated CPU time
- the `UVM GPU` kernel thread hot alongside it
- **`nvidia-smi` reports ~96 % GPU utilization at ~10 W** — busy-looking, doing nothing
- shard loading frozen at a reproducible point (48/120 for us, every time)
- `shm_broadcast.py:801  No available shared memory broadcast block found in 60 seconds`

It never recovers. Twice it took the whole node unreachable over SSH (ping fine, sshd
unable to fork) and needed a power cycle.

**Workaround:** `vm.swappiness=0` on every node, and cycle swap (`swapoff -a && swapon -a`)
before launching. Do **not** disable swap entirely — with no swap at all the worker is
killed outright during MoE marlin repack, which has no valve for that spike.

**The workaround does not survive a reboot.** Put it in `/etc/sysctl.d/` rather than
applying it by hand; we lost three boots to exactly that omission.

Also unresolved: after a livelock, the GPU stayed at 96 % / 10 W **even after the container
was killed** — the driver state did not clear. We did not establish whether a module reload
recovers it or a reboot is required.

---

## 5. Pinning `--kv-cache-memory` silently removes the activation reservation — **still true**

> Still true, and still the sharpest edge here. But note what we got wrong *around* it: we
> concluded that pinning above 16 GiB/rank was unsafe on principle. The real variable was page
> cache. With an unconditional flusher, a pinned 24 GiB/rank passes deep decode at 41K context
> and 3x concurrent 32,879-token prefills. Pinning still skips the activation reservation — so
> you must gate it yourself, which is the point.


Not a bug so much as a very sharp edge, documented here because it cost us most of a night
and the failure mode is deeply misleading.

`gpu_worker.py:475-495`: when `kv_cache_memory_bytes` is set, vLLM **still runs the profile
pass** but never subtracts the measured peak, and `--gpu-memory-utilization` is ignored
entirely. You get exactly the pool you asked for, with no headroom reserved for a real
forward pass.

The result passes every cheap check — allocates, warms up, answers a short prompt at full
speed — and then dies on the first long request. We reproduced it at 7.5 GiB, at 300K
context, at 700K context, and at 12 GiB (which locked the node).

Use the profiler. If you must pin, pin at the value vLLM itself suggests in the startup log
(`Replace gpu_memory_utilization config with --kv-cache-memory=N`), and validate with a
≥28K-token prompt, not a 500-token generation.

---

## 6. `--gpu-memory-utilization 0.87` is not reachable on every node

Mia's recipe uses 0.87 and reports ~+145K tokens per +0.01 util. On our nodes it can refuse
outright before loading anything:

```
ValueError: Free memory on device cuda:0 (104.46/121.69 GiB) on startup is less than
desired GPU memory utilization (0.87, 105.87 GiB)
```

vLLM raises if free < requested. Whether 0.87 clears depends on the node's startup free
memory, which varies several GiB between otherwise identical machines. When it *did* clear,
it pushed the host into swap (3 GiB paged out while serving) and produced 10-second decode
stalls — the throughput average looked like a uniform slowdown but was actually bimodal
(36.8 tok/s windows interleaved with zeros).

0.85 is what we ship. Whether the extra utilization is recoverable with a more aggressive
pre-launch flush is untested.

---

## 7. Smaller open items

- **Vision is not speculated.** The drafter logs `does not support external multimodal
  embeddings … using text-only draft inputs`. Image requests work but get no speedup.
- **CUDA graphs with DFlash2 — RESOLVED 2026-08-29, CONFIRMED +22% 2026-08-31:** `cudagraph_mode: FULL` runs with the
  drafter on this fleet. The trap below still applies — check the metric after any graph boot.
  Original note: **CUDA graphs untested with DFlash2.** We run `--enforce-eager`; step time is a flat
  ~78 ms regardless of acceptance, which makes it the real throughput ceiling. Others run
  graphs successfully on this model. **Trap:** vllm#53030 — piecewise-graph
  `BatchDescriptor` collision silently pins acceptance at exactly 1.00. Check
  `vllm:spec_decode_num_accepted_tokens_per_pos_total` after any graph-enabled boot.
  **2026-08-31 measured on the fp8/marlin lane:** dropping `--enforce-eager` (required
  only by the b12x NVFP4-KV kernels) gives `FULL_AND_PIECEWISE` capture (40s, 0.86 GiB)
  that coexists cleanly with DFlash2 — +22% decode (26.7 -> 32.5 tok/s), accept ratio
  27-35% (NOT pinned at 1.00), zero Xid. Full method + probe battery:
  `docs/CUDA-GRAPHS-DFLASH2-FP8-TP4-2026-08-31.md`.
- **Acceptance is workload-bound, not config-bound.** 0.70+ on structured/math output,
  ~0.33 on freeform prose, measured on the same engine minutes apart. Benchmarks that
  quote one number without the prompt mix are not comparable — including ours before we
  started publishing the harness.
- **`temperature: 0` is free throughput** (+13–21 %): vLLM's rejection sampler does an
  exact top-1 match at temp 0 but a probabilistic ratio test above it, and with greedy
  drafting the draft probability is pinned to 1, making the T>0 test strictly harder.

---

## How to help

The two that would change the most, in order:

1. **InstantTensor multi-node stability** (#3) — removes the ~20-minute boot tax for everyone.
2. **The rank-1 KV asymmetry** (#1) — worth roughly another 54 % pool, and likely an upstream
   vLLM fix rather than anything GB10-specific.

Also unproven by us: **28 and 32 GiB/rank**. Another operator holds 28 GiB at 4,545,221 tokens
on the same hardware ([tonyliu312](https://github.com/tonyliu312/GLM-5.3-Flash-DFlash2-TP4-1M-Context)).
We ship 24. If you ladder higher and it survives the gate suite, that is a PR worth sending.

Reproduction material is in this repo: `overlay-dflash2/` (patches + a simulation harness
that models the KV geometry without booting), `probes/` (kernel probes, the allocation-wall
prober, and the c1–c6 benchmark), and `docs/SM121-CRASH-FORENSICS-2026-08-27.md` for the
two bugs we *did* fix, as a template for how we chase these.
