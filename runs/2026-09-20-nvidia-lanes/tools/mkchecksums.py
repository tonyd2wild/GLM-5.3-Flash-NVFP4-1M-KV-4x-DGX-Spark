#!/usr/bin/env python3
"""Emit a verification manifest for a build, so anyone can confirm they produced the same thing without
downloading weights from us.

Hashes only what this recipe creates or changes: the new NVFP4 shard, the rewritten index, the edited config,
and (for the uncensored lane) the donor tensor file. The 30-odd untouched shards come from the upstream
checkpoint and are verifiable against it.
"""
import json,os,sys,hashlib,struct
D=sys.argv[1]; OUT=sys.argv[2] if len(sys.argv)>2 else None
def sha(p,bs=1<<22):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        while True:
            b=f.read(bs)
            if not b: break
            h.update(b)
    return h.hexdigest()
idx=json.load(open(os.path.join(D,"model.safetensors.index.json")))
wm=idx["weight_map"]
newshards=sorted({f for f in set(wm.values()) if "nvfp4attn" in f or "attn" in f.lower() and f.startswith("model-nvfp4")})
man={"dir":os.path.basename(D),"files":{}}
for f in ["model.safetensors.index.json","config.json"]+newshards:
    p=os.path.join(D,f)
    if not os.path.exists(p): continue
    e={"bytes":os.path.getsize(p),"sha256":sha(p)}
    if f.endswith(".json"):
        # canonical hash: key order in the index is an artifact of how it was written, not of content
        e["sha256_canonical"]=hashlib.sha256(
            json.dumps(json.load(open(p)),sort_keys=True,separators=(",",":")).encode()).hexdigest()
    man["files"][f]=e
q=json.load(open(os.path.join(D,"config.json")))["quantization_config"]
man["quant_algo"]=q.get("quant_algo"); man["ignore_entries"]=len(q.get("ignore",[]))
man["total_tensors"]=len(wm)
qnames=sorted(n for n in wm if n.endswith(".weight_scale_2"))
man["quantized_modules"]=len(qnames)
print(json.dumps(man,indent=1))
if OUT:
    json.dump(man,open(OUT,"w"),indent=1); print("\nwrote",OUT,file=sys.stderr)
