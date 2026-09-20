#!/usr/bin/env python3
"""Build keys-glm53-nvfp4-attn3: the NVFP4 build with the superseded bf16 tensors actually REMOVED.

g14/g15 failed because filter_duplicate_safetensors_files() filters by FILE, not tensor name. All 120 original
shards stay referenced (they hold the experts), so the loader still read the stale bf16 copies of the 412
retargeted tensors and fed a (out, in) bf16 into a (out, in/2) packed NVFP4 parameter.

So: hardlink the shards that contain none of the superseded tensors, and REWRITE the ones that do, omitting
them. Reuses the already-verified NVFP4 shard and the already-correct index/config from attn2.
"""
import json,os,struct,collections,time,shutil
import torch
from safetensors.torch import safe_open, save_file
R="/out"
SRC=R+"/keys-glm-5.3-flash-nvfp4-ablit-l15-45-anchorstock"; A2=R+"/keys-glm53-nvfp4-attn2"; DST=R+"/keys-glm53-nvfp4-attn3"
man=json.load(open("/newshard/manifest.json"))
NEW=man["new_shard"]; superseded=set(man["quantized"])
# every name the new shard provides (weights + both scales)
provided=set()
for n in superseded:
    b=n[:-len(".weight")]
    provided |= {n, b+".weight_scale", b+".weight_scale_2"}
idx=json.load(open(os.path.join(A2,"model.safetensors.index.json")))
wm=idx["weight_map"]
orig=json.load(open(os.path.join(SRC,"model.safetensors.index.json")))["weight_map"]
shards=collections.defaultdict(list)
for n,f in orig.items(): shards[f].append(n)
dirty=sorted({orig[n] for n in superseded})
clean=sorted(set(shards)-set(dirty))
print("shards: %d clean (hardlink), %d dirty (rewrite)"%(len(clean),len(dirty)),flush=True)
os.makedirs(DST,exist_ok=True)
t0=time.time()
for fn in os.listdir(SRC):
    s=os.path.join(SRC,fn); d=os.path.join(DST,fn)
    if os.path.isdir(s) or fn.endswith(".safetensors") or fn in ("model.safetensors.index.json","config.json"): continue
    if not os.path.exists(d):
        try: os.link(s,d)
        except OSError: shutil.copy2(s,d)
for f in clean:
    d=os.path.join(DST,f)
    if not os.path.exists(d):
        try: os.link(os.path.join(SRC,f),d)
        except OSError: shutil.copy2(os.path.join(SRC,f),d)
print("linked aux files and %d clean shards  %.0fs"%(len(clean),time.time()-t0),flush=True)
kept_tot=drop_tot=0
for i,f in enumerate(dirty,1):
    out={}
    with safe_open(os.path.join(SRC,f),framework="pt") as fh:
        for n in fh.keys():
            if n in provided: drop_tot+=1; continue
            out[n]=fh.get_tensor(n); kept_tot+=1
    save_file(out,os.path.join(DST,f),metadata={"format":"pt"})
    del out
    if i%8==0 or i==len(dirty): print("  rewrote %d/%d shards  kept %d dropped %d  %.0fs"%(i,len(dirty),kept_tot,drop_tot,time.time()-t0),flush=True)
ns=os.path.join(DST,NEW)
if not os.path.exists(ns):
    try: os.link(os.path.join(A2,NEW),ns)
    except OSError: shutil.copy2(os.path.join(A2,NEW),ns)
shutil.copy2(os.path.join(A2,"model.safetensors.index.json"),os.path.join(DST,"model.safetensors.index.json"))
shutil.copy2(os.path.join(A2,"config.json"),os.path.join(DST,"config.json"))
# verify: no name is now provided by two files, and every indexed name exists somewhere
present=collections.defaultdict(set)
for fn in sorted(os.listdir(DST)):
    if not fn.endswith(".safetensors"): continue
    with open(os.path.join(DST,fn),"rb") as fh:
        k=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(k))
    for n in hdr:
        if n!="__metadata__": present[n].add(fn)
dups={n:v for n,v in present.items() if len(v)>1}
missing=[n for n in wm if n not in present]
print("VERIFY duplicate names across files: %d   indexed names missing: %d   total names present: %d"%(len(dups),len(missing),len(present)))
if dups: print("  e.g.",list(dups.items())[:3])
if missing: print("  e.g. missing",missing[:5])
bad=[n for n in wm if n in present and wm[n] not in present[n]]
print("index entries pointing at a file that lacks the tensor: %d"%len(bad))
if bad: print("  e.g.",[(n,wm[n],sorted(present[n])) for n in bad[:3]])
print("done %.0fs"%(time.time()-t0))
