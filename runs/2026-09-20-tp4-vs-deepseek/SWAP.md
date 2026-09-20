# Swapping the four Sparks between GLM-5.3-Flash TP4 and DeepSeek-V4.1-Flash TP4

The fleet runs one model at a time: each lane wants ~100 GiB of the 121 GiB on every node. Both launchers
stay installed, so a swap is one command in either direction. Run everything as root on the head (Reddie).

## To GLM-5.3-Flash (1M context, NVFP4 + DFlash2)

```bash
MNBT=8192 SPEC_K=7 bash /root/glm_boot.sh glm-serve
```

That line is the whole configuration. The multimodal video-warmup fix is now a **launcher default**, so a
bare boot no longer hangs for eight minutes before `/health` answers; if you set `VLLM_EXTRA` yourself, add to
it rather than replacing it, or you lose the fix.

`glm_boot.sh` stops whatever is running on all four nodes, drops page cache, then launches rank 3, 2, 1 and
finally the head with 20 s between them, and polls `/health` until the API answers. Boot is about 14 minutes,
of which 5.5 is the checkpoint read (rank 0 local, ranks 1-3 over NFS) and 5.5 is engine init plus CUDA graphs.

Every lever is an env knob on `glm53_tp4.sh`, so the whole configuration is on one line. Defaults are the
shipping ones. The knobs that matter:

| knob | default | note |
|---|---|---|
| `MNBT` | 16384 | 8192 for decode-weighted traffic, 16384 for long prompts |
| `SPEC_K` | 7 | DFlash2 draft length; `SPEC=none` turns speculation off |
| `SEQS` | 64 | aggregate ceiling; do not lower without re-measuring C6+ |
| `MAXLEN` | 1048576 | KV pool scales with it |
| `KV_MEM` | 25769803776 | 24 GiB pinned; the pool report is `max_concurrency x max_model_len` |
| `MOE_BACKEND` | marlin | **keep marlin**, see the correctness section in the write-up |
| `ROCE` / `PREFIX_FIX` | 1 / 1 | b12x one-shot all-reduce and the #18 prefix-cache repair |
| `GMU` | 0.85 | |
| `BLOCK` | 2304 | mamba page alignment, do not change casually |

Verify it is really up, not just answering on the config endpoint:

```bash
curl -s -m 30 -X POST http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"count to 5"}],"max_tokens":40}'
```

`/v1/models` returns 200 from the config even with a dead engine behind it, so it is not a liveness check.

## To GLM-5.3-Flash with NVFP4 attention (the fast build, 4 of 6 categories above DeepSeek)

```bash
MODEL_DIR=keys-glm53-nvfp4-attn3 NVFP4_PATCH=1 MNBT=8192 SPEC_K=7 bash /root/glm_boot.sh glm-nvfp4
```

This is the same launcher with two extra knobs. `MODEL_DIR` points at the build whose 13.88 GiB of non-expert
projections are NVFP4 (W4A16) instead of bf16, and `NVFP4_PATCH=1` bind-mounts two patched glm5next files that
stop `quant_config` being forced to `None` for those projections - without it the build cannot load, because a
packed `(out, in/2)` weight will not go into a bf16 `(out, in)` parameter. The patched files live at
`$PATCH_HOME/patches/nvfp4/{kda.py,model.py}` on every node and the launcher refuses to start if they are
missing.

Measured against the bf16 build: structure +30%, math +29%, prose +27%, counting +21%, code +12%, JSON +5%,
step time 67.6 -> 57 ms, weights 43.76 GiB/rank. Quality gate 5/5.

**Caveat before serving it:** one of four needles dropped a digit at 131K depth 0.3 (65K, 98K and 131K depth 0.6
were exact). Do not put this build on 100K-plus retrieval work until that is settled against a bf16 baseline.

## To DeepSeek-V4.1-Flash (500K context, EXL3 3.5 bpw + DSpark)

```bash
bash /root/sr_boot.sh sr2-e12-final500k-s16-go.sh ds-serve
```

Boot is about 7.5 minutes. To boot it and re-measure it in one step, `bash /root/ds_final.sh <label>`, which
also runs the 3-rep median screen and the high-concurrency sweep and leaves the lane serving.

## Measuring either lane the same way

```bash
bash /root/glm_screen3.sh <label> 3 1,3,6 2000,8000,32000,64000   # GLM
bash /root/ds_screen3.sh  <label> 3 1,3,6 2000,8000,32000,64000   # DeepSeek
bash /root/glm_hiconc.sh <label>   /   bash /root/ds_hiconc.sh <label>
python3 /root/mktable.py <glm_dir> <glm_label> <ds_dir> <ds_label> 3
```

Both screens discard one warm-up pass, run three measured passes and report per-cell medians with the
max/min spread, because single passes on these lanes are not resolvable (see the write-up). Only rep 1's
prefill numbers are cold: this build has no prefix-cache reset endpoint, so reps 2 and 3 hit the cache.

## Before you swap: the launcher must be on every node

All four nodes launch from their own copy of `/root/glm53_tp4.sh`, so they must agree:

```bash
bash /root/dist_glm.sh
```

prints the md5 and a syntax check for each node and re-pushes the head's copy. As of this run all four are
`17bb1581`. A node running a stale copy fails in non-obvious ways rather than refusing to start.

## Two things that will bite you

- **Launch worker-first, head last.** Ranks 1-3 with `--headless`, then rank 0. The head gives up on the
  others long before they finish reading the checkpoint if you start it first.
- **Do not edit a harness script while it is running.** bash reads a script incrementally, so an overwrite
  mid-run corrupts the copy that is executing. Edit between experiments.
