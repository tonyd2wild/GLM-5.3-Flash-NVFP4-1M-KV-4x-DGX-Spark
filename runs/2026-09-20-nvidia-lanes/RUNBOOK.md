# Exact replication runbook

Every command, in order, with the values used. Everything referenced here is in this folder. Upstream is MIT
and ungated throughout, so no access requests and no weights are hosted by us.

## 0. What you need first

| | |
|---|---|
| hardware | 4x DGX Spark (GB10, 121 GiB unified each), RoCEv2 fabric |
| image | `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`, digest `sha256:35c6f70ffcba62fd67d7b9d4b4e8300ad177201792ce9cdb1ea18fd449bc23b6` |
| base checkpoint | [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4), 190.4 GiB, 33 shards |
| drafter | `GLM-5.3-Flash-DFlash2` at `/var/tmp/models/GLM-5.3-Flash-DFlash2` |
| donor (Lane B only) | [`dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4`](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4), fetched by range request, 2.56 GiB |
| free disk | ~200 GiB per lane you build |
| prereq on every node | `$HOME/patches/sparse_attn_indexer_kpool.py` (from `docker/sparse_attn_indexer_kpool_sm121.py` in the repo root) and `chat_template_mm.jinja` in the weights dir |

**The image digest matters.** `patches/kda.py` and `patches/model.py` are edited copies of two files from that
exact image. The launcher verifies the image's own copies still hash to `2a20e453...` and `b6c8eb2d...` and
refuses to boot otherwise, because mounting stale copies over a rebuilt image would silently run old model code.
If you are on a different image, regenerate with `tools/mkpatch.py` and update the hashes in `glm53_tp4.sh`.

## 1. Put the patched model files on every node

```bash
install -d $HOME/patches/nvfp4
cp patches/kda.py patches/model.py $HOME/patches/nvfp4/
# repeat on all four nodes; they must be identical
```

Without these the build **cannot load at all**: `kda.py:172` and `model.py:331` force `quant_config=None` on the
attention projections, so vLLM builds them BF16 and a packed 4-bit weight has nowhere to go.

## 2. Lane A: official nvidia weights + NVFP4 attention

```bash
# quantize the non-expert projections: 403 tensors, 13.04 GiB -> 3.67 GiB, about 70 s on one GPU
SRC=/var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia OUT=/scratch/nv-attn python3 tools/mknvfp4b.py

# assemble: hardlink the 2 clean shards, rewrite the 31 holding superseded tensors, write index and config
SRC=/var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia DST=/var/tmp/models/nvidia-glm53-attn \
NEWDIR=/scratch/nv-attn python3 tools/build_nv.py

# use the ignore list proven to boot (do NOT hand-derive one, see the traps section in README)
python3 tools/fix_ignore.py /var/tmp/models/nvidia-glm53-attn
```

`build_nv.py` prints `VERIFY dups 0 | indexed-missing 0 | index-points-at-wrong-file 0`. All three must be zero.

## 3. Lane B: the same, plus the dealignai transplant

```bash
# fetch ONLY the tensors dealignai actually changed. It diffs each o_proj against nvidia stock and skips the
# identical ones, so it discovers the set rather than trusting a documented range. Took layers 12-44, 33
# tensors, 2.56 GiB, in 329 s; skipped 0-11 and 45.
NV=/var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia OUT=/scratch/dealign python3 tools/fetch_dealign.py

# quantize with the donor substituted BEFORE quantization, so the abliterated o_proj is what gets quantized
SRC=/var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia SUBST=/scratch/dealign/dealign_oproj.safetensors \
OUT=/scratch/nv-ablit python3 tools/mknvfp4b.py     # must print "donor tensors used: 33"

SRC=/var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia DST=/var/tmp/models/nvidia-glm53-ablit-attn \
NEWDIR=/scratch/nv-ablit python3 tools/build_nv.py
python3 tools/fix_ignore.py /var/tmp/models/nvidia-glm53-ablit-attn
```

**Order is not optional.** Substitute, then quantize. Reversed, you would be copying BF16 tensors over NVFP4
ones, which cannot work.

## 4. Boot

```bash
bash tools/glm_settle.sh          # stop everything, drop caches, WAIT for >=105 GiB free on every node
MODEL_DIR=nvidia-glm53-attn NVFP4_PATCH=1 MNBT=8192 SPEC_K=7 MAXLEN=500000 \
  VLLM_EXTRA='--limit-mm-per-prompt {"image":4,"video":0}' \
  bash tools/glm_boot.sh my-label
```

Swap `MODEL_DIR=nvidia-glm53-ablit-attn` for Lane B. The settle step is not optional: booting straight after
stopping a ~100 GiB lane races the memory reclaim and vLLM refuses with
`Free memory on device cuda:0 (103.04/121.69 GiB) ... less than desired`.

Expect on the way up: `quant_algo=W4A16_NVFP4`, `Using MarlinNvFp4LinearKernel`, 43.76 GiB/rank of weights,
`GPU KV cache size: 3,532,196 tokens`, boot about 12 minutes.

## 5. Measure

```bash
export BENCH_MODEL=glm-5.3-flash
bash tools/lane_run.sh <label> <MODEL_DIR>   # gate, idle, C1-C6 x3 with prefill, C8-C16, C24-C32, acceptance
python3 tools/compare_lanes.py               # tables, judged against each cell's own spread
python3 tools/corrupt_probe.py               # vLLM #54150 token-corruption check
bash tools/probe_prefill.sh <label>          # cold prefill on a QUIET lane
```

**Run these one at a time.** Two measurements here were thrown away because something else was running against
the endpoint: a prefill probe taken during a 190 GiB shard rewrite, and Lane B's rep 1 taken while the
corruption probe was generating. Single passes also carry a plus-or-minus 25% band, so use the 3-rep medians.

## 6. Verify you built the same thing

```bash
python3 tools/mkchecksums.py /var/tmp/models/nvidia-glm53-attn
```

Compare against `checksums-laneA.json` / `checksums-laneB.json`. The untouched shards come straight from nvidia
and verify against the upstream checkpoint; only the new shard, index and config are ours.

`tools/verify_ablit.py` attempts to confirm the donor weights ended up quantized, but note its limit: NVFP4
introduces about 9.4% Frobenius error while the abliteration edit is only 2-4%, so the noise exceeds the signal
and the test is **inconclusive by construction**. The reliable check is `mknvfp4b.py` reporting
`donor tensors used: 33`.
