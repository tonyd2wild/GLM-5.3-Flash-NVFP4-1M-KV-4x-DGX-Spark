#!/usr/bin/env python3
"""WHERE did Blackfrost edit the weights? (Their method is undisclosed.)

Both packs are NVFP4 quantizations of the same zai-org GLM-5.3-Flash base, so for an UNTOUCHED tensor the two
differ only by quantization noise. Measuring that noise on classes nobody abliterates (gate_proj, up_proj, q/k/v)
gives a floor; a class sitting well above the floor was edited. This predicts whether Blackfrost touched the
12,384 routed-expert down_proj tensors carrying 0.81 of the refusal signal, or something narrower like the 46
o_proj tensors (0.03) that the keys lane poked and which degenerated.
Read-only, CPU only, samples a handful per class: the lane must not be disturbed.
"""
import json,os,sys,collections
import torch
from safetensors import safe_open
A=sys.argv[1] if len(sys.argv)>1 else "/models/nv"          # nvidia stock
B=sys.argv[2] if len(sys.argv)>2 else "/models/bf"          # blackfrost
# fp4 e2m1: magnitude LUT indexed by (exp<<1)|mant, sign in bit 3
LUT=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.],dtype=torch.float32)
def deq(w,s,s2):
    """NVFP4 -> float32. w uint8 (out, in/2) two fp4 per byte, low nibble first; s fp8 per 16-group; s2 scalar."""
    lo=w & 0xF; hi=(w >> 4) & 0xF
    def val(n):
        mag=LUT[(n & 0x7).long()]
        return torch.where((n & 0x8).bool(), -mag, mag)
    out=torch.stack([val(lo),val(hi)],dim=-1).reshape(w.shape[0],-1)
    sc=s.float().repeat_interleave(16,dim=1)[:, :out.shape[1]]
    return out*sc*float(s2.reshape(-1)[0])
def load(d,name):
    idx=json.load(open(os.path.join(d,"model.safetensors.index.json")))["weight_map"]
    if name not in idx: return None
    got={}
    for suf in ("","_scale","_scale_2"):
        k=name if suf=="" else name+suf
        if k not in idx: return None
        with safe_open(os.path.join(d,idx[k]),framework="pt",device="cpu") as f:
            got[suf]=f.get_tensor(k)
    return got
idxA=json.load(open(os.path.join(A,"model.safetensors.index.json")))["weight_map"]
idxB=json.load(open(os.path.join(B,"model.safetensors.index.json")))["weight_map"]
CLASSES=[("expert down_proj", lambda k: ".experts." in k and k.endswith("down_proj.weight")),
         ("expert gate_proj",  lambda k: ".experts." in k and k.endswith("gate_proj.weight")),
         ("expert up_proj",    lambda k: ".experts." in k and k.endswith("up_proj.weight")),
         ("attn o_proj",       lambda k: k.endswith("self_attn.o_proj.weight")),
         ("attn q_proj",       lambda k: k.endswith("self_attn.q_proj.weight")),
         ("shared down_proj",  lambda k: "shared_expert" in k and k.endswith("down_proj.weight"))]
print("%-20s %6s  %-10s %-10s %-10s"%("class","n","rel_delta","min","max"))
print("-"*64)
for label,pred in CLASSES:
    common=[k for k in idxA if pred(k) and k in idxB]
    if not common:
        print("%-20s %6d  (not comparable: absent in one pack)"%(label,0)); continue
    common.sort()
    step=max(1,len(common)//6)
    picks=common[::step][:6]
    rels=[]
    for k in picks:
        ta=load(A,k); tb=load(B,k)
        if ta is None or tb is None: continue
        try:
            wa=deq(ta[""],ta["_scale"],ta["_scale_2"]); wb=deq(tb[""],tb["_scale"],tb["_scale_2"])
        except Exception as e:
            print("   %s deq failed %r"%(k,e)); continue
        if wa.shape!=wb.shape: continue
        rels.append(((wa-wb).norm()/wa.norm().clamp_min(1e-9)).item())
    if rels:
        print("%-20s %6d  %-10.5f %-10.5f %-10.5f"%(label,len(rels),sum(rels)/len(rels),min(rels),max(rels)))
    else:
        print("%-20s %6d  (no comparable samples)"%(label,0))
print("\nReading: the lowest row is the quantization-noise floor. A class far above it was EDITED.")
