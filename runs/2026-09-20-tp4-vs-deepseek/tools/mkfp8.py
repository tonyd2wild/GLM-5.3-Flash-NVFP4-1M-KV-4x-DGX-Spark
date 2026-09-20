#!/usr/bin/env python3
"""Quantize the non-expert 2-D bf16 weights of the GLM-5.3-Flash NVFP4 pack to fp8_e4m3, weight-only,
per-tensor scales, dynamic activations (no calibration needed).

Why: those tensors are 17.90 GiB read on EVERY decode step regardless of token count, which the fitted
per-step model prices at 17.7 ms of a 67.5 ms step. Halving them is worth ~8.9 ms/step.

Strategy that avoids rewriting 78 GiB: hardlink all 120 original shards into the new dir, write ONE new
shard holding the quantized tensors plus their scales, and rewrite model.safetensors.index.json to point
those names at it. vLLM's filter_duplicate_safetensors_files() keeps only files the index references, so
the stale bf16 copies inside the hardlinked shards are never read.
"""
import json,os,sys,struct,collections,shutil,time
import torch
from safetensors.torch import safe_open, save_file

SRC=sys.argv[1] if len(sys.argv)>1 else "/var/tmp/models/keys-glm-5.3-flash-nvfp4-ablit-l15-45-anchorstock"
DST=sys.argv[2] if len(sys.argv)>2 else "/var/tmp/models/keys-glm53-nvfp4-fp8attn"
DRY=os.environ.get("DRY","0")=="1"
LIMIT=int(os.environ.get("LIMIT","0"))     # dry-run: only this many tensors
FP8=torch.float8_e4m3fn
FMAX=448.0

idx=json.load(open(os.path.join(SRC,"model.safetensors.index.json")))
wm=idx["weight_map"]
shards=collections.defaultdict(list)
for n,f in wm.items(): shards[f].append(n)

# pick targets: 2-D, bf16, not a routed expert, min dim >= 64
targets={}
for f,names in shards.items():
    with open(os.path.join(SRC,f),"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(n))
    for name in names:
        t=hdr.get(name)
        if not t or t["dtype"]!="BF16": continue
        if ".experts." in name: continue
        sh=t["shape"]
        if len(sh)!=2 or min(sh)<64: continue
        targets[name]=f
print("targets: %d tensors in %d shards"%(len(targets),len(set(targets.values()))))
if LIMIT: targets=dict(list(targets.items())[:LIMIT]); print("DRY LIMIT -> %d"%len(targets))

if DRY:
    # validate the numerics on a sample without writing the model
    byf=collections.defaultdict(list)
    for n,f in targets.items(): byf[f].append(n)
    worst=0.0
    for f,names in list(byf.items())[:3]:
        with safe_open(os.path.join(SRC,f),framework="pt") as fh:
            for n in names[:6]:
                w=fh.get_tensor(n)
                amax=w.abs().max().float()
                s=(amax/FMAX).clamp(min=1e-12)
                q=(w.float()/s).clamp(-FMAX,FMAX).to(FP8)
                deq=q.float()*s
                rel=((deq-w.float()).norm()/w.float().norm()).item()
                worst=max(worst,rel)
                print("   %-64s %s amax %.4f rel_err %.5f"%(n[:64],tuple(w.shape),amax.item(),rel))
    print("worst relative error on sample: %.5f"%worst)
    raise SystemExit(0)

os.makedirs(DST,exist_ok=True)
t0=time.time()
# 1. hardlink everything except the index (and shards are hardlinked too)
for fn in os.listdir(SRC):
    s=os.path.join(SRC,fn); d=os.path.join(DST,fn)
    if os.path.exists(d): continue
    if os.path.isdir(s): continue
    if fn=="model.safetensors.index.json": continue
    try: os.link(s,d)
    except OSError: shutil.copy2(s,d)
print("linked originals in %.1fs"%(time.time()-t0))

# 2. quantize, grouped by source shard so each is opened once
NEW="model-fp8attn-00001-of-00001.safetensors"
out={}; byf=collections.defaultdict(list)
for n,f in targets.items(): byf[f].append(n)
done=0
for f in sorted(byf):
    with safe_open(os.path.join(SRC,f),framework="pt") as fh:
        for n in byf[f]:
            w=fh.get_tensor(n)
            s=(w.abs().max().float()/FMAX).clamp(min=1e-12)
            out[n]=(w.float()/s).clamp(-FMAX,FMAX).to(FP8)
            out[n.rsplit(".",1)[0]+".weight_scale"]=s.reshape(1).to(torch.float32)
            done+=1
    print("  %s -> %d/%d  %.0fs"%(f,done,len(targets),time.time()-t0),flush=True)
save_file(out,os.path.join(DST,NEW),metadata={"format":"pt"})
print("wrote %s (%.2f GiB) in %.0fs"%(NEW,os.path.getsize(os.path.join(DST,NEW))/2**30,time.time()-t0))

# 3. rewrite the index: retarget quantized names, add the scales, drop shards no longer referenced
for n in targets: wm[n]=NEW
for n in list(out):
    if n.endswith(".weight_scale"): wm[n]=NEW
idx["weight_map"]=wm
refs=set(wm.values())
for fn in os.listdir(DST):
    if fn.endswith(".safetensors") and fn not in refs:
        print("  unreferenced shard left on disk (vLLM will skip it):",fn)
json.dump(idx,open(os.path.join(DST,"model.safetensors.index.json"),"w"),indent=1)
print("index rewritten: %d entries, %d shards referenced"%(len(wm),len(refs)))
