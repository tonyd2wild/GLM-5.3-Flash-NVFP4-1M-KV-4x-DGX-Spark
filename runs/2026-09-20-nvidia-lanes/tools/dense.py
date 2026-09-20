import json,collections,os
M="/var/tmp/models/keys-glm-5.3-flash-nvfp4-ablit-l15-45-anchorstock"
idx=json.load(open(os.path.join(M,"model.safetensors.index.json")))
wm=idx["weight_map"]
# read dtype+shape per tensor from each shard header (no data read)
import struct
shards=collections.defaultdict(list)
for name,f in wm.items(): shards[f].append(name)
info={}
for f in shards:
    p=os.path.join(M,f)
    with open(p,"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]
        hdr=json.loads(fh.read(n))
    for name in shards[f]:
        t=hdr.get(name)
        if t: info[name]=(t["dtype"],t["shape"],t["data_offsets"][1]-t["data_offsets"][0])
DT=collections.Counter(); BY=collections.Counter()
cls=collections.defaultdict(lambda: collections.Counter())
def bucket(n):
    if ".mlp.experts." in n or ".experts." in n: return "routed_experts"
    if ".shared_expert" in n or ".mlp." in n: return "dense_mlp"
    if any(k in n for k in (".self_attn.",".attn.",".attention.")): return "attention"
    if "indexer" in n or "kda" in n.lower() or "linear_attn" in n: return "kda_indexer"
    if "embed" in n or "lm_head" in n: return "embed_head"
    return "other"
for n,(dt,sh,nb) in info.items():
    DT[dt]+=1; BY[dt]+=nb
    b=bucket(n); cls[b][dt]+=nb
tot=sum(BY.values())
print("total checkpoint bytes %.1f GiB across %d tensors"%(tot/2**30,len(info)))
print("\nby dtype:")
for dt,nb in BY.most_common(): print("  %-10s %8.2f GiB  (%d tensors)"%(dt,nb/2**30,DT[dt]))
print("\nby module class (GiB):")
for b,c in sorted(cls.items(),key=lambda kv:-sum(kv[1].values())):
    print("  %-16s %8.2f  | %s"%(b,sum(c.values())/2**30," ".join("%s %.2f"%(d,v/2**30) for d,v in c.most_common())))
nonexp=sum(sum(c.values()) for b,c in cls.items() if b!="routed_experts")
print("\nnon-expert (read every step regardless of token count): %.2f GiB total, %.2f GiB per rank at TP4"%(nonexp/2**30,nonexp/2**30/4))
bf=sum(c.get("BF16",0)+c.get("F16",0)+c.get("BFLOAT16",0) for b,c in cls.items() if b!="routed_experts")
print("of which bf16/f16: %.2f GiB total, %.2f GiB per rank"%(bf/2**30,bf/2**30/4))
print("\nper-rank bytes at 273 GB/s -> ms: non-expert %.1f ms"%(nonexp/4/273e9*1000))
