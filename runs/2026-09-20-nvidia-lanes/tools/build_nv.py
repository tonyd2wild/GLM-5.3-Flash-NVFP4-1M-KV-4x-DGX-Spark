#!/usr/bin/env python3
"""Assemble an NVFP4-attention build on the nvidia checkpoint.

Same shape as build4.py, with one difference that matters: nvidia's ignore list uses BROAD per-layer wildcards
(`layers.N.self_attn*` x45, `...shared_experts*` x42). Dropping those wholesale would un-ignore the DSA indexer
and the KDA f_b/g_b gates, which stay bf16 and are not in the new shard, and vLLM would then look for
quantized params that do not exist. So we drop the broad entries and add back narrow ones for exactly the
modules that remain bf16.

Env: SRC, DST, NEWDIR (conversion output with manifest.json).
"""
import json,os,struct,collections,time,shutil,re,sys
from safetensors.torch import safe_open, save_file
SRC=os.environ.get("SRC","/out/GLM-5.3-Flash-NVFP4-nvidia")
DST=os.environ["DST"]
NEWDIR=os.environ.get("NEWDIR","/newshard")
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
print("targets %d tensors | %d clean shards, %d to rewrite"%(len(superseded),len(clean),len(dirty)),flush=True)
os.makedirs(DST,exist_ok=True); t0=time.time()
for fn in os.listdir(SRC):
    s=os.path.join(SRC,fn); d=os.path.join(DST,fn)
    if os.path.isdir(s) or fn in ("model.safetensors.index.json","config.json"): continue
    if fn.endswith(".safetensors") and fn in dirty: continue
    if not os.path.exists(d):
        try: os.link(s,d)
        except OSError: shutil.copy2(s,d)
print("linked aux + %d clean shards  %.0fs"%(len(clean),time.time()-t0),flush=True)
kept=drop=0
for i,f in enumerate(dirty,1):
    out={}
    with safe_open(os.path.join(SRC,f),framework="pt") as fh:
        for n in fh.keys():
            if n in provided: drop+=1; continue
            out[n]=fh.get_tensor(n); kept+=1
    save_file(out,os.path.join(DST,f),metadata={"format":"pt"}); del out
    if i%8==0 or i==len(dirty): print("  %d/%d shards  kept %d dropped %d  %.0fs"%(i,len(dirty),kept,drop,time.time()-t0),flush=True)
ns=os.path.join(DST,NEW)
if not os.path.exists(ns):
    try: os.link(os.path.join(NEWDIR,NEW),ns)
    except OSError: shutil.copy2(os.path.join(NEWDIR,NEW),ns)
for n in superseded:
    b=n[:-len(".weight")]
    wm[n]=NEW; wm[b+".weight_scale"]=NEW; wm[b+".weight_scale_2"]=NEW
# sorted so the index is byte-reproducible: retargeting iterates a set, and unsorted insertion order made
# two otherwise-identical builds produce different file hashes
orig["weight_map"]={k:wm[k] for k in sorted(wm)}
json.dump(orig,open(os.path.join(DST,"model.safetensors.index.json"),"w"),indent=1,sort_keys=True)
# --- config: narrow the ignore list ------------------------------------------------
cfg=json.load(open(os.path.join(SRC,"config.json"))); q=cfg["quantization_config"]
before=list(q.get("ignore") or q.get("exclude_modules") or [])
BROAD=re.compile(r"\.self_attn\*$|\.shared_experts\*$")
kept_ig=[x for x in before if not BROAD.search(x)]
NARROW=["lm_head","model.language_model.embed_tokens",
        "*.self_attn.f_b_proj","*.self_attn.g_b_proj","*.self_attn.fused_qkv_a_proj",
        "*.self_attn.q_a_proj","*.self_attn.kv_a_proj_with_mqa",
        "*.self_attn.indexer.wk_weights_proj","*.self_attn.indexer.wk",
        "*.self_attn.indexer.weights_proj","*.self_attn.indexer.wq_b",
        "*.mlp.gate","*.eh_proj","model.visual.*"]
kept_ig=[]   # replace nvidias broad per-layer wildcards wholesale with the proven list
for x in NARROW:
    if x not in kept_ig: kept_ig.append(x)
q["ignore"]=kept_ig; q["quant_algo"]="W4A16_NVFP4"; cfg["quantization_config"]=q
json.dump(cfg,open(os.path.join(DST,"config.json"),"w"),indent=1)
print("ignore %d -> %d (dropped %d broad, added %d narrow); quant_algo %s"%(
    len(before),len(kept_ig),len(before)-len([x for x in before if not BROAD.search(x)]),len(NARROW),q["quant_algo"]))
# --- verify -------------------------------------------------------------------------
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
