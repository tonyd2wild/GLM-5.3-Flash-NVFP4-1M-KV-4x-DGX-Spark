#!/usr/bin/env python3
"""Quantize the non-expert 2-D projection weights of the GLM-5.3-Flash NVFP4 pack to NVFP4 (W4A16).

Why: those bf16 tensors are read on EVERY decode step regardless of token count. The fitted per-step model
(step_ms = 34.6 + 4.12 x verify_tokens, intercept = 17.7 ms of bf16 weights + 16.9 ms overhead) prices the
13.98 GiB of safely-quantizable projections at 10.3 ms of a 67.6 ms step.

Format verified byte-for-byte against this checkpoint's own expert tensors:
  weight        uint8          (out, in/2)    fp4 e2m1, low nibble first
  weight_scale  float8_e4m3fn  (out, in/16)   NON-swizzled layout
  weight_scale_2 float32       scalar         = amax / (448 * 6) = 1 / input_global_scale
Round-trip through ops.scaled_fp4_quant(..., is_sf_swizzled_layout=False) reproduced both stored byte arrays
exactly, so the layout is not guesswork.

Writes ONLY the new shard. The caller assembles the model dir by hardlinking the originals.
"""
import json,os,sys,struct,collections,time
import torch
from safetensors.torch import safe_open, save_file
from vllm import _custom_ops as ops

SRC=os.environ.get("SRC","/models/glm-5.3-flash-nvfp4")
OUT=os.environ.get("OUT","/cache/nvfp4attn")
FP8MAX,FP4MAX=448.0,6.0

# Safe to quantize: large standard linear projections on the decode path.
KEEP=(".self_attn.q_proj",".self_attn.k_proj",".self_attn.v_proj",".self_attn.o_proj",
      ".self_attn.q_a_proj",".self_attn.q_b_proj",".self_attn.kv_a_proj_with_mqa",".self_attn.kv_b_proj",
      ".mlp.shared_experts.gate_proj",".mlp.shared_experts.up_proj",".mlp.shared_experts.down_proj",
      ".mlp.gate_proj",".mlp.up_proj",".mlp.down_proj")
# Never: embeddings/head (lookup, and logits), DSA indexer (picks KV blocks), MoE router (picks experts),
# KDA f/g gates, eh_proj, vision tower, and anything already quantized.
NEVER=(".experts.","indexer","embed_tokens","lm_head",".mlp.gate.",".b_proj",".f_a_proj",".g_a_proj",
       ".f_b_proj",".g_b_proj",".eh_proj","visual.")

idx=json.load(open(os.path.join(SRC,"model.safetensors.index.json")))
wm=idx["weight_map"]
shards=collections.defaultdict(list)
for n,f in wm.items(): shards[f].append(n)

targets={}
for f,names in shards.items():
    with open(os.path.join(SRC,f),"rb") as fh:
        k=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(k))
    for name in names:
        t=hdr.get(name)
        if not t or t["dtype"]!="BF16" or not name.endswith(".weight"): continue
        if any(x in name for x in NEVER): continue
        if not any(name.endswith(k+".weight") for k in KEEP): continue
        sh=t["shape"]
        if len(sh)!=2 or sh[1]%16 or min(sh)<64: continue
        targets[name]=(f,tuple(sh),t["data_offsets"][1]-t["data_offsets"][0])
gib=sum(v[2] for v in targets.values())/2**30
print("targets: %d tensors, %.2f GiB bf16 -> ~%.2f GiB nvfp4"%(len(targets),gib,gib/4*1.125),flush=True)
byf=collections.defaultdict(list)
for n,(f,_,_) in targets.items(): byf[f].append(n)
print("spread over %d source shards"%len(byf),flush=True)
if os.environ.get("LIST","0")=="1":
    pats=collections.Counter()
    import re
    for n in targets: pats[re.sub(r"\.layers\.\d+\.",".layers.N.",n).split("language_model.")[-1]]+=1
    for p,c in pats.most_common(): print("   x%-3d %s"%(c,p))
    raise SystemExit(0)

os.makedirs(OUT,exist_ok=True)
new={}; t0=time.time(); done=0
for f in sorted(byf):
    with safe_open(os.path.join(SRC,f),framework="pt") as fh:
        for n in byf[f]:
            w=fh.get_tensor(n).cuda()
            gs=(FP8MAX*FP4MAX/w.abs().max().float()).clamp(min=1e-12)
            q,bs=ops.scaled_fp4_quant(w.bfloat16(),gs,is_sf_swizzled_layout=False)
            base=n[:-len(".weight")]
            new[n]=q.cpu()
            new[base+".weight_scale"]=bs.cpu()
            new[base+".weight_scale_2"]=(1.0/gs).float().cpu().reshape(())
            del w,q,bs
            done+=1
    if done % 60 < 24: print("  %d/%d  %.0fs"%(done,len(targets),time.time()-t0),flush=True)
torch.cuda.empty_cache()
NEW="model-nvfp4attn-00001-of-00001.safetensors"
save_file(new,os.path.join(OUT,NEW),metadata={"format":"pt"})
print("wrote %s: %d tensors, %.2f GiB, %.0fs"%(NEW,len(new),os.path.getsize(os.path.join(OUT,NEW))/2**30,time.time()-t0))
json.dump({"new_shard":NEW,"quantized":sorted(targets),"keep":list(KEEP)},open(os.path.join(OUT,"manifest.json"),"w"),indent=1)
print("manifest written")
