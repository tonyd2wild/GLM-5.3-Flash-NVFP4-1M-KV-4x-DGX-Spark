#!/usr/bin/env python3
"""Assemble /var/tmp/models/keys-glm53-nvfp4-attn from the original pack plus the new NVFP4 shard.

Hardlinks the 120 original shards (no copy, same filesystem), moves the new shard in, rewrites the index so
the 334 quantized names point at it, and edits config.json:
  - drop the now-quantized module patterns from the `ignore` list (including the fused aliases, since
    vLLM's is_layer_skipped resolves packed layers through packed_modules_mapping)
  - quant_algo NVFP4 -> W4A16_NVFP4, which selects ModelOptNvFp4W4A16LinearMethod (FP4 Marlin GEMM with
    bf16 activations). This needs no input_scale, so it also avoids the W4A4 uninitialised-scale trap that
    this checkpoint's declared config otherwise walks into.
"""
import json,os,sys,shutil,collections
SRC="/var/tmp/models/keys-glm-5.3-flash-nvfp4-ablit-l15-45-anchorstock"
DST="/var/tmp/models/keys-glm53-nvfp4-attn"
NEWDIR="/var/tmp/glm53-vllm-cache/nvfp4attn"
man=json.load(open(os.path.join(NEWDIR,"manifest.json")))
NEW=man["new_shard"]; qnames=set(man["quantized"])
os.makedirs(DST,exist_ok=True)
n_link=0
for fn in os.listdir(SRC):
    s=os.path.join(SRC,fn); d=os.path.join(DST,fn)
    if os.path.isdir(s) or fn=="model.safetensors.index.json" or fn=="config.json": continue
    if os.path.exists(d): continue
    try: os.link(s,d); n_link+=1
    except OSError: shutil.copy2(s,d); n_link+=1
print("linked %d files"%n_link)
tgt=os.path.join(DST,NEW)
if not os.path.exists(tgt):
    os.rename(os.path.join(NEWDIR,NEW),tgt)   # same filesystem, instant
print("new shard in place: %.2f GiB"%(os.path.getsize(tgt)/2**30))

idx=json.load(open(os.path.join(SRC,"model.safetensors.index.json")))
wm=idx["weight_map"]; moved=0
for n in qnames:
    base=n[:-len(".weight")]
    wm[n]=NEW; wm[base+".weight_scale"]=NEW; wm[base+".weight_scale_2"]=NEW; moved+=1
idx["weight_map"]=wm
json.dump(idx,open(os.path.join(DST,"model.safetensors.index.json"),"w"),indent=1)
refs=set(wm.values())
print("index: %d entries, %d shards referenced, %d tensors retargeted"%(len(wm),len(refs),moved))
unref=[f for f in os.listdir(DST) if f.endswith(".safetensors") and f not in refs]
print("unreferenced shards left on disk (vLLM filters by index): %d"%len(unref))

cfg=json.load(open(os.path.join(SRC,"config.json")))
q=cfg["quantization_config"]
DROP={"*.self_attn.q_proj","*.self_attn.k_proj","*.self_attn.v_proj","*.self_attn.o_proj",
      "*.self_attn.q_a_proj","*.self_attn.q_b_proj","*.self_attn.kv_a_proj_with_mqa","*.self_attn.kv_b_proj",
      "*.mlp.shared_experts.gate_proj","*.mlp.shared_experts.up_proj","*.mlp.shared_experts.down_proj",
      "*.mlp.gate_proj","*.mlp.up_proj","*.mlp.down_proj",
      "*.mlp.gate_up_proj","*.mlp.shared_experts.gate_up_proj"}
before=list(q["ignore"])
q["ignore"]=[x for x in before if x not in DROP]
q["quant_algo"]="W4A16_NVFP4"
cfg["quantization_config"]=q
json.dump(cfg,open(os.path.join(DST,"config.json"),"w"),indent=1)
print("ignore: %d -> %d  (dropped: %s)"%(len(before),len(q["ignore"]),", ".join(sorted(set(before)&DROP))))
print("kept ignored:",", ".join(q["ignore"]))
print("quant_algo ->",q["quant_algo"])
