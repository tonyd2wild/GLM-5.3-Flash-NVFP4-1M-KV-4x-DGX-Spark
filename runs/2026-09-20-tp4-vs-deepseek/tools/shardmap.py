import json,collections,os,struct
M="/var/tmp/models/keys-glm-5.3-flash-nvfp4-ablit-l15-45-anchorstock"
wm=json.load(open(os.path.join(M,"model.safetensors.index.json")))["weight_map"]
shards=collections.defaultdict(list)
for n,f in wm.items(): shards[f].append(n)
bf_by_shard=collections.Counter(); bytes_by_shard=collections.Counter(); tot_by_shard=collections.Counter()
twod=0; twod_bytes=0; other=collections.Counter()
for f,names in shards.items():
    with open(os.path.join(M,f),"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(n))
    for name in names:
        t=hdr.get(name)
        if not t: continue
        nb=t["data_offsets"][1]-t["data_offsets"][0]
        tot_by_shard[f]+=nb
        if t["dtype"]=="BF16" and ".experts." not in name and ".mlp.experts." not in name:
            bf_by_shard[f]+=1; bytes_by_shard[f]+=nb
            if len(t["shape"])==2 and min(t["shape"])>=64: twod+=1; twod_bytes+=nb
            else: other[tuple(t["shape"]) if len(t["shape"])<2 else "2d-small"]+=1
print("shards total %d; shards containing non-expert bf16: %d"%(len(shards),len(bf_by_shard)))
print("non-expert bf16 tensors: %d, of which 2-D with min dim>=64: %d (%.2f GiB of %.2f GiB)"%(
    sum(bf_by_shard.values()),twod,twod_bytes/2**30,sum(bytes_by_shard.values())/2**30))
print("bytes that would have to be REWRITTEN if we rewrite whole shards: %.1f GiB"%(sum(tot_by_shard[f] for f in bf_by_shard)/2**30))
print("\ntop shards by non-expert bf16 content:")
for f,c in bf_by_shard.most_common(8):
    print("   %s  %4d tensors  %6.2f GiB bf16 of %6.2f GiB shard"%(f,c,bytes_by_shard[f]/2**30,tot_by_shard[f]/2**30))
print("\nnon-2D / small non-expert bf16 shapes (left alone):", dict(other.most_common(6)))
