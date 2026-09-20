#!/usr/bin/env python3
"""Fetch only the o_proj tensors dealignai actually edited, by HTTP range request.

dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 is ~180 GiB. The abliteration is 33 BF16 o_proj tensors in layers
12-44 (determined empirically by diffing against nvidia's stock weights, not from a documented range). Each is
64 MiB, so this pulls about 2.06 GiB instead of the whole pack.

safetensors carries a JSON header with each tensor's byte offsets, so we read the header, then range-read just
the tensor payloads, and write them to one local safetensors file.
Output: $OUT/dealign_oproj.safetensors + manifest.
"""
import json,os,struct,urllib.request,re,time,sys
import torch
from safetensors.torch import save_file
REPO="https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4/resolve/main"
NV=os.environ.get("NV","/models/nvidia")
OUT=os.environ.get("OUT","/out")
os.makedirs(OUT,exist_ok=True)
def get(url,rng=None,timeout=300):
    h={"Range":"bytes=%d-%d"%rng} if rng else {}
    return urllib.request.urlopen(urllib.request.Request(url,headers=h),timeout=timeout).read()
print("fetching dealignai index...",flush=True)
d_wm=json.loads(get(REPO+"/model.safetensors.index.json"))["weight_map"]
nv_wm=json.load(open(os.path.join(NV,"model.safetensors.index.json")))["weight_map"]
def nv_sample(name,k=64):
    f=nv_wm[name]
    with open(os.path.join(NV,f),"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; h=json.loads(fh.read(n)); t=h[name]
        fh.seek(8+n+t["data_offsets"][0]); return fh.read(k),t["dtype"],tuple(t["shape"])
names=sorted([n for n in d_wm if n.endswith("self_attn.o_proj.weight")],
             key=lambda n:int(re.search(r"layers\.(\d+)\.",n).group(1)))
hdr={}
def head(f):
    if f not in hdr:
        n=struct.unpack("<Q",get(REPO+"/"+f,(0,7)))[0]
        hdr[f]=(n,json.loads(get(REPO+"/"+f,(8,8+n-1))))
    return hdr[f]
out={}; t0=time.time(); skipped=[]
for nm in names:
    f=d_wm[nm]; n,h=head(f); t=h[nm]
    s=8+n+t["data_offsets"][0]; e=8+n+t["data_offsets"][1]-1
    nb,ndt,nsh=nv_sample(nm)
    if (t["dtype"],tuple(t["shape"]))!=(ndt,nsh):
        print("  SKIP %s: shape/dtype mismatch %s %s vs %s %s"%(nm,t["dtype"],tuple(t["shape"]),ndt,nsh)); continue
    first=get(REPO+"/"+f,(s,s+63))
    if first==nb:
        skipped.append(int(re.search(r"layers\.(\d+)\.",nm).group(1))); continue   # identical to stock
    raw=get(REPO+"/"+f,(s,e))
    assert len(raw)==t["data_offsets"][1]-t["data_offsets"][0], "short read on "+nm
    out[nm]=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(t["shape"])
    print("  %-62s %6.1f MiB  %.0fs"%(nm.split("language_model.")[-1],len(raw)/2**20,time.time()-t0),flush=True)
print("\nfetched %d tensors (%.2f GiB); identical-to-stock layers skipped: %s"%(len(out),
      sum(v.numel()*2 for v in out.values())/2**30, skipped))
save_file(out,os.path.join(OUT,"dealign_oproj.safetensors"),metadata={"format":"pt"})
json.dump({"tensors":sorted(out),"source":"dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4",
           "license":"MIT","skipped_identical_layers":skipped},
          open(os.path.join(OUT,"dealign_manifest.json"),"w"),indent=1)
print("wrote dealign_oproj.safetensors in %.0fs"%(time.time()-t0))
