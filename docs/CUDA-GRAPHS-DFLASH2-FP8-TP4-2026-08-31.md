# CUDA Graphs + DFlash2 on the fp8/marlin TP4 lane — 2026-08-31

## TL;DR

Dropping `--enforce-eager` on the **fp8-KV / marlin** TP4 lane (NOT the b12x NVFP4-KV
path) enables `cudagraph_mode: FULL_AND_PIECEWISE`, which captures cleanly alongside
DFlash2 and gives **~+22% single-stream decode** with no regression, no Xid, and no
spec-acceptance degradation. Backups + rollback on all 4 ranks.

## Why this matters / how it differs from the NVFP4-KV lane

Tony's recipe ships `--enforce-eager` in every launcher, with the README noting it is
*"required by the b12x kernels."* That is lane-specific: the b12x `B12X_MLA_SPARSE`
backend (drowzeys/keys Zero-RoPE + NVFP4-KV, `nvfp4_ds_mla`) is not CUDA-graph-capture
safe. **Our live lane is fp8-KV + `--moe-backend marlin`** — the b12x binding does not
apply. So dropping eager there is not contradicting Tony; it's a different kernel path.

Tony's own OPEN-PROBLEMS marks *"CUDA graphs with DFlash2 — RESOLVED 2026-08-29:
`cudagraph_mode: FULL` runs with the drafter."* We confirmed this at TP4 with
FULL_AND_PIECEWISE.

## The change (exactly two flags, everything else untouched)

- **Remove** `--enforce-eager`
- **`--max-num-batched-tokens` 8192 -> 16384**
- gmu 0.68, max-num-seqs 6, max-model-len 1048576, fp8 KV, DFlash2 k=7 — unchanged.

Launchers edited identically on all 4 ranks; backups
`launch-glm53-vllm-tp4-redhat-1m.sh.bak-p0-20260831`.

## Boot evidence

- Engine: `enforce_eager=False`, `cudagraph_mode: FULL_AND_PIECEWISE (2,1)`,
  capture sizes [1..96], `compile_ranges_endpoints: [16384]`.
- **Graph capture: PIECEWISE 15/15 + FULL 6/6, 40s, 0.86 GiB.** Drafter graphs also
  captured 6/6. `Graph capturing finished in 40 secs, took 0.86 GiB`.
- KV pool: 3,834,498 tokens (vs 3,895,606 eager) — the ~0.86 GiB graph buffer.
- **No Xid 43/13**, no EngineDead, all 4 ranks Up.

## The critical guard — vLLM#53030

Tony's doc warns: piecewise-graph `BatchDescriptor` collision can silently pin DFlash2
acceptance at exactly **1.00**. After this graphs boot, spec accept ratio was
**27-35%** across runs (eager baseline was 25.1%) — i.e. NOT pinned, graphs and the
drafter coexist. **Always re-check the accept metric after any graph-enabled boot.**

## Probe battery (same harness, eager -> graphs)

| Metric | Baseline (eager) | After (graphs) | Δ |
|---|---|---|---|
| P1 decode tok/s (median of 5) | ~26.7 | **32.5** | **+22%** |
| Spec accept ratio | 25.1% | 27.2-35% | improved |
| KV pool (tokens) | 3,895,606 | 3,834,498 | -1.6% |
| P4 tool + strict JSON | pass | pass | — |
| P5 concurrency (3-stream) | pass | pass (~116 tok/s agg) | — |
| Graph capture + mem | n/a | clean, 0.86 GiB | — |

## Verdict / caveats

- **Net-positive, below the +35% SHIP bar.** +22% decode, zero regression, no instability.
- H1 and H2 ship together, so the +22% is a combined effect — graphs are the likely
  driver, but not cleanly attributable.
- 24h soak still pending at time of writing (stable to 29+ min).

## Rollback

Restore `.bak-p0-20260831` launchers on all 4 ranks, relaunch worker-first. Same ~15-20
min cold start; radix cache re-warms ~30-60 min.
