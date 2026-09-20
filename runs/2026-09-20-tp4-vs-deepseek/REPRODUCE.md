# Reproducing the NVFP4-attention build

You do not need the 172 GiB build, five boots, or a 65 GiB download. You need **about 4 GiB plus roughly two
minutes of local disk work**, on the right base checkpoint.

## What is actually new

The build is the base checkpoint with 412 tensors replaced. Of its 121 shards:

- **73 are byte-identical to the base** and are hardlinked, not copied
- **47 are rewritten**, purely to *remove* the superseded bf16 tensors
- **1 is new**: `model-nvfp4attn2-00001-of-00001.safetensors`, **3.90 GiB**, holding the 412 quantized weights
  plus their `weight_scale` and `weight_scale_2` tensors

So the only novel bytes are that one 3.90 GiB shard, the rewritten `model.safetensors.index.json`, and the
edited `config.json`.

## Why you cannot just drop the new shard in

This is the part that cost two failed boots here, so it is worth being blunt about.

`filter_duplicate_safetensors_files()` filters by **file**, not by tensor name:

```python
for weight_name in weight_map:
    weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))
```

Every original shard stays referenced by the index, because each still holds routed-expert tensors that must
load. So all of them are kept, and vLLM then iterates **every tensor inside them**, including the stale bf16
copies of the 412 tensors you just retargeted. A stale bf16 `(out, in)` is then handed to the packed NVFP4
parameter of shape `(out, in/2)` and the loader asserts:

```
vllm/model_executor/layers/parameter.py:175  assert param_data.shape == loaded_weight.shape
AssertionError
```

Hardlink-plus-retarget is only valid when a stale tensor's shard contains nothing else that is needed. Here 47
shards mix both. **The rewrite is mandatory**, but it is local I/O against your own copy, not a download: 47
shards, 42,582 tensors kept, 412 dropped, measured at **92 seconds**.

## Steps

1. Get the base checkpoint this was built from (see below: it must be the right one).
2. Download the new shard, the index and the config.
3. Run `tools/build4.py`, which hardlinks the clean shards, rewrites the 47 dirty ones without the superseded
   tensors, drops the new shard in, and writes the index and config. It then verifies zero duplicate names, zero
   missing indexed names and zero index entries pointing at a file that lacks the tensor. Those three checks are
   the ones whose absence cost the two boots.
4. Put `patches/kda.py` and `patches/model.py` on every node and boot with `NVFP4_PATCH=1`. Without the patch
   the build cannot load at all: `kda.py:172` and `model.py:331` force `quant_config=None` on these projections,
   so the layer is constructed bf16 and a packed 4-bit weight has nowhere to go.

## It is base-specific, and that matters

The quantized tensors were produced **from one specific checkpoint's weights**:
`keys-glm-5.3-flash-nvfp4-ablit-l15-45-anchorstock`, whose parent is `LibertAIDAI/GLM-5.3-Flash-NVFP4`.

- It will **not** work on the RedHat, nvidia or stock LibertAI packs. Those are different weights, and for
  nvidia a different layout entirely (33 shards, 147,661 tensors). Dropping this shard onto them gives you
  another model's attention weights.
- That base is **abliterated**: 31 `layers.{15..45}.self_attn.o_proj.weight` tensors transplanted from
  `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4`. `o_proj` is 3.625 GiB of the 13.88 GiB quantized here, so **the
  new shard contains abliterated weights**. Know that before you redistribute it.

To build it on a different base, run `tools/mknvfp4b.py` then `tools/build4.py` against that checkpoint. The
conversion takes about 70 seconds on one GPU. `mknvfp4b.py` refuses to write if any fused group is incomplete,
which is the check that stops you producing a layer mixing NVFP4 and bf16 constituents.

## Known gaps, if you want to help

- **No bf16 needle baseline at 1M context.** One needle dropped a digit at 131K depth 0.3 on a 1M lane and
  passed at 500K, so the fault tracked `max_model_len` rather than the quantization, but nobody has run the
  unquantized build at 1M to confirm it is not pre-existing. Run `v41needle.py --targets 65536,131072,262144
  --depth 0.3` on a stock 1M lane and the answer is settled.
- **Refusal behaviour after quantization is untested.** `o_proj` carries the abliteration and is the largest
  thing quantized. METHOD.md in the base pack reports 32/32 bypass on a Refusal32 set for the unquantized build;
  nobody has re-run it here.
