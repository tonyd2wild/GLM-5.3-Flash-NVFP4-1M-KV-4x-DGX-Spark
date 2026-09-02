# GLM-5.3-Flash TopK oversubscription — exact `torch.topk` routing fix

> ## ⚠️ This image must run with `--enforce-eager`
>
> **Do not pair the topkfix image with `cudagraph_mode: FULL_AND_PIECEWISE`.** Measured on a
> 4-Spark fleet 2026-09-02: all three worker ranks freeze at the same instant on
> `Breakable CUDA graph enabled` and never reach serving. The tell is
> **96% GPU utilisation at ~18 W** — a spin-wait on a collective that will never complete
> (see [`docs/FIELD-NOTES-4NODE-100G.md`](FIELD-NOTES-4NODE-100G.md)) — with the head rank
> logging `shm_broadcast: No available shared memory broadcast block` once a minute.
>
> This is a genuine incompatibility, not a flaw in the patch: this fix was validated
> 2026-08-28, and the CUDA-graph path only became the recommended fp8/marlin lane on
> 08-31. With `--enforce-eager` the image is clean — verified decoding 113,918-token
> prompts at C1 and C2 with **no** SM-count bind-mount and zero `persistent_topk`,
> `oversubscribe`, `FilteredTopK` or `EngineDead` signatures on any rank.
>
> **The trade-off, measured on the same fleet** (seqs 64, batched 16384, temperature 0,
> median of 3):
>
> | prompt | graphs + SM-count bind-mount | topkfix + eager |
> |---|---|---|
> | count-to-100 | 105.58 | 80.07 |
> | code | 77.32 | 58.09 |
> | prose | 31.53 | 21.85 |
>
> About **−25%**, which is the cost of eager, not of the patch. Main lane keeps the
> bind-mount and the graphs; this lane trades speed for a baked fix and no mount to forget.



**Status: SHIPPED AND VERIFIED LIVE on a 4-Spark GB10 fleet (2026-08-28).** This is the
second, cleaner fix for **Disease 1** (`persistent_topk` smem wall) documented in
[`SM121-CRASH-FORENSICS-2026-08-27.md`](SM121-CRASH-FORENSICS-2026-08-27.md). Where the
original fix gated the fused top-k kernels behind an SM-count check (forcing small-SM parts
onto the fallback kernel), this fix routes the fused top-k **through exact `torch.topk`** on
sm_12x — the approach upstream vLLM itself converged on (PR #49897) and the community
`dots3-note-gb10-vllm` runtime uses. Chat canary OK, **4/4 concurrent long-context requests
OK**, engine clean (no `EngineDeadError` / topk errors) after load.

## Why the fleet still crashed (recap of Disease 1)

GLM-5.3-Flash uses a **kpool** sparse-attention indexer (`sparse_attn_indexer_kpool.py`,
`index_topk=2048`, `index_kpool=4` → `select_k = 512`). On GB10 (sm_121, 48 SMs, ~99 KB /
101376 B smem per block), the fused top-k kernels are unusable:

- `torch.ops._C.persistent_topk` (decode) and `torch.ops._C.top_k_per_row_prefill` (prefill)
  launch CTAs proportional to the **logits width == `max_model_len`**, not the prompt length.
- At large context, `total_ctas` exceeds GB10's 48 SMs, tripping a `FilteredTopK` fallback
  that requires **≥128 KB smem/block** — SM121 has only 101376 B.
- Crash: `RuntimeError: launch_persistent_topk, topk.cu: persistent_topk would oversubscribe
  and the FilteredTopK fallback requires >=128KB smem per block (have 101376).
  total_ctas=60 > num_sms*occupancy=48 (TopK=512)` → `EngineDeadError` → endpoint dead.

## Why this is the cleaner fix than the SM-count gate

The shipped `sparse_attn_indexer_kpool_sm121.py` gate forces small-SM parts onto the existing
`top_k_per_row_decode` fallback. It works but:

- **Depends on SM-count detection** and uses the fallback kernel rather than the intended path.
- The prefill `top_k_per_row_prefill` path was not addressed the same way.
- It diverges from the direction upstream took (exact `torch.topk`).

## The fix — `docker/topkfix/patch_kpool_topk.py`

A small Python patcher applied at image build. It:

1. Adds `_USE_TORCH_TOPK = current_platform.is_device_capability_family(120)` (sm_12x only).
2. Adds `_torch_decode_topk` and `_torch_prefill_topk` — exact `torch.topk` replacements that
   mask invalid positions with `-inf`, run `torch.topk(masked, k, dim=-1)`, and preserve the
   `-1` padding sentinel (the downstream `expand_pools_and_append_tail` recognizes `tok < 0`).
3. Wires them into **both** call sites in `sparse_attn_indexer_kpool.py` (decode before
   `persistent_topk`, prefill before `top_k_per_row_prefill`), passing the **exact same
   arguments** the fused kernels received so the pool-granular masking contract is preserved.
4. On non-sm_12x, behavior is byte-identical to stock (fused branches untouched).

**kpool-specific note:** logits/seq_lens are **pool-granular** (compress_ratio ==
`index_kpool`); selection is on pools (`select_k = topk_tokens // index_kpool`), matching the
kernel contract. This is the crucial difference from the community `dots3-note-gb10-vllm`
patch, which targets the **non-kpool** `sparse_attn_indexer.py` and does **not** fix GLM-5.3.

### Build (overlay, ~40 s/node)

```bash
# Base = the existing DFlash2 overlay image already on your nodes.
FROM ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
ARG VLLM=/usr/local/lib/python3.12/dist-packages/vllm
COPY docker/topkfix/patch_kpool_topk.py /tmp/patch_kpool_topk.py
RUN python3 /tmp/patch_kpool_topk.py && \
    python3 -c "import ast,pathlib; p=pathlib.Path('$VLLM/model_executor/layers/sparse_attn_indexer_kpool.py'); ast.parse(p.read_text()); t=p.read_text(); assert '_USE_TORCH_TOPK' in t and t.count('_torch_decode_topk')>=2 and t.count('_torch_prefill_topk')>=2, 'topkfix markers missing'; print('topkfix apply + AST check OK')"
```

```bash
cd docker/topkfix && docker build -t ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v12-dflash2-topkfix .
```

Because the patch touches only ONE file, distribute by pushing the patcher + Dockerfile to
each node and building the tiny overlay from each node's **existing local base** (base layers
are shared → ~40 s). Do **not** `docker save | docker load` a full 31 GB image across the
fleet (collapses to ~2 MB/s through a relay and crawls).

### Launcher

[`launch-glm53-tp4-dflash2-topkfix.sh`](../launch-glm53-tp4-dflash2-topkfix.sh) — drop-in for
`launch-glm53-tp4-24g.sh` using the topkfix image. **Crucially it does NOT bind-mount
`sparse_attn_indexer_kpool_sm121.py`** — the torch.topk fix is baked into the image, and a
bind-mount would overwrite it. Launch worker-first (rank 3→2→1→0, head last).

## Verification (live, 2026-08-28)

- `docker inspect` → image `sm121-v12-dflash2-topkfix`.
- Resolved `Glm5NextForConditionalGeneration` + `DFlash2DraftModel`, Eagle3 aux layers
  `(6,15,25,34,43)`, spec-decode warmup completed.
- Chat canary returns "OK".
- **Stress probe:** 4 concurrent ~2000-token-prompt requests all completed OK (5.7–6.4 s
  each), healthy after — the shape that previously killed the engine now passes.
- `docker logs --since`: clean (no `persistent_topk` / `oversubscribe` / `EngineDeadError`).

## Upstream context

- Upstream vLLM PR **#49897** (SM12x prefill top-k through `torch.topk`) — still open.
- Issue **#49896** (SM12x `top_k_per_row_prefill` NaN/garbage → illegal memory access) — open.
- Issue **#51782** (`persistent_topk` silently drops top-k candidates on coarse histogram
  bins) — open; a *correctness* defect beyond the crash, and another reason exact `torch.topk`
  is the right routing on sm_12x.
- Community `jjang-ai/dots3-note-gb10-vllm` uses the same `torch.topk` routing for the
  **non-kpool** indexer; this patches the **kpool** indexer GLM-5.3 actually uses.

## Rollback

Rel launch with `launch-glm53-tp4-24g.sh` (the SM-count gate + `sm121-v11-dflash2` image).
