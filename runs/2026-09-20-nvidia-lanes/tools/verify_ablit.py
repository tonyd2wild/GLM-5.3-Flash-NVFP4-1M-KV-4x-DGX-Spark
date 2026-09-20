#!/usr/bin/env python3
"""End-to-end check that Lane B's quantized o_proj came from the dealignai donor, not nvidia stock.

Dequantizes the NVFP4 tensor out of Lane B's new shard and compares it against both candidates. Quantization
adds a few percent of error, so the donor should be much closer than stock if the substitution worked. If they
come out equidistant, the substitution silently did nothing.
"""
import json,os,struct,sys
import torch
from safetensors.torch import safe_open
E2M1=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.])
def deq(w,s,s2):
    lo=(w & 0xF).to(torch.long); hi=(w >> 4).to(torch.long)
    val=lambda n: E2M1[n & 7]*torch.where((n>>3).bool(),-1.,1.)
    out=torch.stack([val(lo),val(hi)],dim=-1).reshape(w.shape[0],-1)
    blk=s.float()*s2.float()
    return (out.reshape(w.shape[0],-1,16)*blk.unsqueeze(-1)).reshape(w.shape[0],-1)
def get(M,name):
    wm=json.load(open(os.path.join(M,"model.safetensors.index.json")))["weight_map"]
    if name not in wm: return None
    with safe_open(os.path.join(M,wm[name]),framework="pt") as fh: return fh.get_tensor(name)
LB="/var/tmp/models/nvidia-glm53-ablit-attn"; NV="/var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia"
DON="/var/tmp/glm53-vllm-cache/dealign/dealign_oproj.safetensors"
print("%-8s %-12s %-12s %-12s"%("layer","vs donor","vs stock","verdict"))
with safe_open(DON,framework="pt") as dfh:
    for L in (12,20,33,44,3,45):
        n="model.language_model.layers.%d.self_attn.o_proj.weight"%L
        b=n[:-len(".weight")]
        w=get(LB,n); s=get(LB,b+".weight_scale"); s2=get(LB,b+".weight_scale_2")
        if w is None or w.dtype!=torch.uint8:
            print("%-8d %s"%(L,"not quantized in Lane B (unexpected)" if w is not None else "absent")); continue
        dq=deq(w,s,s2)
        stock=get(NV,n).float()
        donor=dfh.get_tensor(n).float() if n in dfh.keys() else None
        ed=((dq-donor).norm()/donor.norm()).item() if donor is not None else float("nan")
        es=((dq-stock).norm()/stock.norm()).item()
        if donor is None:
            v="donor has no tensor for this layer (expected for 0-11,45); matches stock" if es<0.05 else "UNEXPECTED"
            print("%-8d %-12s %-12.4f %s"%(L,"-",es,v)); continue
        v="FROM DONOR" if ed<es*0.5 else ("from stock" if es<ed*0.5 else "ambiguous")
        print("%-8d %-12.4f %-12.4f %s"%(L,ed,es,v))
