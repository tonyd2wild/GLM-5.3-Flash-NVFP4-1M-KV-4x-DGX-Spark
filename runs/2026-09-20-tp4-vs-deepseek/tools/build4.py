#!/usr/bin/env python3
"""One-shot build of the NVFP4 model dir: hardlink clean shards, rewrite dirty ones without the superseded
bf16 tensors, add the NVFP4 shard, write the index and the edited quant config. Combines what assemble.py and
mkdir3.py did separately, so only ONE new model dir is created (disk is at 96%)."""
import json,os,struct,collections,time,shutil
from safetensors.torch import safe_open, save_file
R="/out"
SRC=R+"/keys-glm-5.3-flash-nvfp4-ablit-l15-45-anchorstock"
DST=R+"/keys-glm53-nvfp4-attn4"
NEWDIR="/newshard"
man=json.load(open(os.path.join(NEWDIR,"manifest.json")))
NEW=man["new_shard"]; superseded=set(man["quantized"])
provided=set()
for n in superseded:
    b=n[:-len(".weight")]
    provided |= {n,b+".weight_scale",b+".weight_scale_2"}
orig=json.load(open(os.path.join(SRC,"model.safetensors.index.json")))
wm=dict(orig["weight_map"])
shards=collections.defaultdict(list)
for n,f in orig["weight_map"].items(): shards[f].append(n)
dirty=sorted({orig["weight_map"][n] for n in superseded})
clean=sorted(set(shards)-set(dirty))
print("targets %d tensors | %d clean shards (hardlink), %d dirty (rewrite)"%(len(superseded),len(clean),len(dirty)),flush=True)
os.makedirs(DST,exist_ok=True); t0=time.time()
for fn in os.listdir(SRC):
    s=os.path.join(SRC,fn); d=os.path.join(DST,fn)
    if os.path.isdir(s) or fn in ("model.safetensors.index.json","config.json"): continue
    if fn.endswith(".safetensors") and fn in dirty: continue
    if not os.path.exists(d):
        try: os.link(s,d)
        except OSError: shutil.copy2(s,d)
print("linked aux + clean shards %.0fs"%(time.time()-t0),flush=True)
kept=drop=0
for i,f in enumerate(dirty,1):
    out={}
    with safe_open(os.path.join(SRC,f),framework="pt") as fh:
        for n in fh.keys():
            if n in provided: drop+=1; continue
            out[n]=fh.get_tensor(n); kept+=1
    save_file(out,os.path.join(DST,f),metadata={"format":"pt"}); del out
    if i%10==0 or i==len(dirty): print("  %d/%d shards  kept %d dropped %d  %.0fs"%(i,len(dirty),kept,drop,time.time()-t0),flush=True)
ns=os.path.join(DST,NEW)
if not os.path.exists(ns):
    try: os.link(os.path.join(NEWDIR,NEW),ns)
    except OSError: shutil.copy2(os.path.join(NEWDIR,NEW),ns)
for n in superseded:
    b=n[:-len(".weight")]
    wm[n]=NEW; wm[b+".weight_scale"]=NEW; wm[b+".weight_scale_2"]=NEW
orig["weight_map"]=wm
json.dump(orig,open(os.path.join(DST,"model.safetensors.index.json"),"w"),indent=1)
cfg=json.load(open(os.path.join(SRC,"config.json"))); q=cfg["quantization_config"]
DROP={"*.self_attn.o_proj","*.self_attn.q_b_proj","*.self_attn.kv_b_proj",
      "*.mlp.gate_proj","*.mlp.up_proj","*.mlp.down_proj","*.mlp.gate_up_proj",
      "*.mlp.shared_experts.gate_proj","*.mlp.shared_experts.up_proj",
      "*.mlp.shared_experts.down_proj","*.mlp.shared_experts.gate_up_proj"}
before=list(q["ignore"]); q["ignore"]=[x for x in before if x not in DROP]
q["quant_algo"]="W4A16_NVFP4"; cfg["quantization_config"]=q
json.dump(cfg,open(os.path.join(DST,"config.json"),"w"),indent=1)
print("ignore %d -> %d, quant_algo %s"%(len(before),len(q["ignore"]),q["quant_algo"]))
present=collections.defaultdict(set)
for fn in sorted(os.listdir(DST)):
    if not fn.endswith(".safetensors"): continue
    with open(os.path.join(DST,fn),"rb") as fh:
        k=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(k))
    for n in hdr:
        if n!="__metadata__": present[n].add(fn)
dups={n for n,v in present.items() if len(v)>1}
missing=[n for n in wm if n not in present]
bad=[n for n in wm if n in present and wm[n] not in present[n]]
print("VERIFY dups %d | indexed-missing %d | index-points-at-wrong-file %d | names %d"%(len(dups),len(missing),len(bad),len(present)))
print("done %.0fs"%(time.time()-t0))
