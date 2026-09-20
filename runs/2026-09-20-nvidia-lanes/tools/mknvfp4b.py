#!/usr/bin/env python3
"""NVFP4 the non-expert projections of GLM-5.3-Flash, corrected for vLLM's FUSED-layer rules.

g14 failed at parameter.py:175 `assert param_data.shape == loaded_weight.shape` in load_merged_column_weight.
Reading ModelOptNvFp4W4A16LinearMethod.create_weights gave both reasons:
  weight_scale_2 = PerTensorScaleParameter(data=torch.empty(len(output_partition_sizes)))   -> shape (N,), so a
      constituent must load a shape-(1,) tensor, not a bare scalar
  process_weights_after_loading: if torch.unique(layer.weight_scale_2).numel() != 1: raise  -> every constituent
      of a fused layer must carry the SAME global scale
And glm5next/nvidia/model.py:776-781 shows in_proj_qkvbfg_a fuses SIX tensors (q,k,v,b,f_a,g_a), not three, so
q/k/v cannot be quantized unless b/f_a/g_a are too.

So: quantize in fused GROUPS with one shared amax per group, and store weight_scale_2 as shape (1,).
"""
import json,os,sys,struct,collections,time,re
import torch
from safetensors.torch import safe_open, save_file
from vllm import _custom_ops as ops

SRC=os.environ.get("SRC","/models/glm-5.3-flash-nvfp4")
OUT=os.environ.get("OUT","/cache/nvfp4attn2")
FP8MAX,FP4MAX=448.0,6.0

# fused target -> constituent suffixes that MUST share one global scale (from packed_modules_mapping)
FUSED={"in_proj_qkvbfg_a":[".self_attn.q_proj",".self_attn.k_proj",".self_attn.v_proj",
                           ".self_attn.b_proj",".self_attn.f_a_proj",".self_attn.g_a_proj"],
       "gate_up_proj":[".gate_proj",".up_proj"]}
# standalone layers, each its own scale
SINGLE=[".self_attn.o_proj",".self_attn.q_b_proj",".self_attn.kv_b_proj",".down_proj"]
# never: lookup/logits, expert router, DSA indexer, the a-projections that fuse into fused_qkv_a_proj,
# the f_b/g_b gates, eh_proj, vision, and anything already quantized
NEVER=(".experts.","indexer","embed_tokens","lm_head",".mlp.gate.",".f_b_proj",".g_b_proj",
       ".eh_proj","visual.",".q_a_proj",".kv_a_proj_with_mqa",".fused_qkv_a_proj",
       )

idx=json.load(open(os.path.join(SRC,"model.safetensors.index.json")))
wm=idx["weight_map"]
shards=collections.defaultdict(list)
for n,f in wm.items(): shards[f].append(n)
meta={}
for f,names in shards.items():
    with open(os.path.join(SRC,f),"rb") as fh:
        k=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(k))
    for name in names:
        t=hdr.get(name)
        if t: meta[name]=(f,t["dtype"],tuple(t["shape"]),t["data_offsets"][1]-t["data_offsets"][0])

def eligible(name):
    if not name.endswith(".weight"): return None
    f,dt,sh,nb=meta[name]
    if dt!="BF16" or len(sh)!=2 or sh[1]%16 or min(sh)<16: return None
    if any(x in name for x in NEVER): return None
    for tgt,parts in FUSED.items():
        for p in parts:
            if name.endswith(p+".weight"):
                # group key = everything up to the constituent suffix
                return (tgt, name[:-len(p+".weight")])
    for p in SINGLE:
        if name.endswith(p+".weight"): return ("single", name)
    return None

groups=collections.defaultdict(list)
for name in meta:
    g=eligible(name)
    if g: groups[g].append(name)
tot=sum(meta[n][3] for v in groups.values() for n in v)
print("groups: %d, tensors: %d, %.2f GiB bf16"%(len(groups),sum(len(v) for v in groups.values()),tot/2**30),flush=True)
byk=collections.Counter()
for (tgt,_),v in groups.items(): byk[tgt]+=len(v)
print("  by fused target:",dict(byk),flush=True)
if os.environ.get("LIST","0")=="1":
    pats=collections.Counter()
    for v in groups.values():
        for n in v: pats[re.sub(r"\.layers\.\d+\.",".layers.N.",n).split("language_model.")[-1]]+=1
    for p,c in pats.most_common(): print("   x%-3d %s"%(c,p))
    bad=[ (tgt,pref,len(v)) for (tgt,pref),v in groups.items() if tgt in FUSED and len(v)!=len(FUSED[tgt])]
    print("\nINCOMPLETE fused groups (would mix quantized and bf16 constituents): %d"%len(bad))
    for b in bad[:10]: print("   ",b)
    raise SystemExit(0)

# refuse to write a group that is missing a constituent: vLLM would see a partially quantized fused layer
bad=[(tgt,pref) for (tgt,pref),v in groups.items() if tgt in FUSED and len(v)!=len(FUSED[tgt])]
if bad:
    print("ABORT: %d incomplete fused groups, e.g. %s"%(len(bad),bad[:3])); raise SystemExit(3)

# SUBST: quantize from donor tensors instead of the base checkpoint's own, so an abliterated o_proj is what
# gets quantized rather than stock. Order matters: substitute first, quantize second.
SUBST=os.environ.get("SUBST","")
sub={}; used_sub=0
if SUBST:
    with safe_open(SUBST,framework="pt") as _fh:
        for _k in _fh.keys(): sub[_k]=_fh.get_tensor(_k)
    print("substitution donor: %d tensors from %s"%(len(sub),SUBST),flush=True)
    _miss=[k for k in sub if k not in meta]
    if _miss: print("  WARNING %d donor tensors absent from the base: %s"%(len(_miss),_miss[:3]),flush=True)
os.makedirs(OUT,exist_ok=True)
new={}; t0=time.time(); done=0
for (tgt,pref),names in sorted(groups.items()):
    ten={}
    for n in sorted(names):
        f=meta[n][0]
        if n in sub:
            ten[n]=sub[n].cuda(); used_sub+=1
        else:
            with safe_open(os.path.join(SRC,f),framework="pt") as fh: ten[n]=fh.get_tensor(n).cuda()
    amax=max(t.abs().max().float().item() for t in ten.values())     # ONE amax for the whole fused group
    gs=torch.tensor(FP8MAX*FP4MAX/max(amax,1e-12),dtype=torch.float32,device="cuda")
    s2=(1.0/gs).reshape(1).float().cpu()                              # shape (1,), not a bare scalar
    for n,w in ten.items():
        q,bs=ops.scaled_fp4_quant(w.bfloat16(),gs,is_sf_swizzled_layout=False)
        base=n[:-len(".weight")]
        new[n]=q.cpu(); new[base+".weight_scale"]=bs.cpu(); new[base+".weight_scale_2"]=s2.clone()
        done+=1
    del ten
    if done%120<8: print("  %d tensors  %.0fs"%(done,time.time()-t0),flush=True)
torch.cuda.empty_cache()
NEW="model-nvfp4attn2-00001-of-00001.safetensors"
save_file(new,os.path.join(OUT,NEW),metadata={"format":"pt"})
print("wrote %s: %d tensors, %.2f GiB, %.0fs"%(NEW,len(new),os.path.getsize(os.path.join(OUT,NEW))/2**30,time.time()-t0))
json.dump({"new_shard":NEW,"quantized":sorted(n for v in groups.values() for n in v)},
          open(os.path.join(OUT,"manifest.json"),"w"),indent=1)
print("manifest written; donor tensors used: %d"%used_sub)
