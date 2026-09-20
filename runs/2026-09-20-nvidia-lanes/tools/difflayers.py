import json,os,struct,urllib.request,re,collections
NV="/var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia"
REPO="https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4/resolve/main"
nv_wm=json.load(open(os.path.join(NV,"model.safetensors.index.json")))["weight_map"]
print("fetching dealignai index...",flush=True)
d_idx=json.load(urllib.request.urlopen(REPO+"/model.safetensors.index.json",timeout=120))
d_wm=d_idx["weight_map"]
names=sorted([n for n in d_wm if n.endswith("self_attn.o_proj.weight")],
             key=lambda n:int(re.search(r"layers\.(\d+)\.",n).group(1)))
print("o_proj tensors: dealignai %d | nvidia %d"%(len(names),len([n for n in nv_wm if n.endswith("self_attn.o_proj.weight")])),flush=True)
hdrcache={}
def dl_head(f):
    if f in hdrcache: return hdrcache[f]
    r=urllib.request.Request(REPO+"/"+f,headers={"Range":"bytes=0-7"})
    n=struct.unpack("<Q",urllib.request.urlopen(r,timeout=60).read())[0]
    r=urllib.request.Request(REPO+"/"+f,headers={"Range":"bytes=8-%d"%(8+n-1)})
    h=json.loads(urllib.request.urlopen(r,timeout=180).read())
    hdrcache[f]=(n,h); return hdrcache[f]
def dl_sample(name,k=64):
    f=d_wm[name]; n,h=dl_head(f); t=h[name]; s=8+n+t["data_offsets"][0]
    r=urllib.request.Request(REPO+"/"+f,headers={"Range":"bytes=%d-%d"%(s,s+k-1)})
    return urllib.request.urlopen(r,timeout=60).read(), t["dtype"], tuple(t["shape"])
def nv_sample(name,k=64):
    if name not in nv_wm: return None,None,None
    f=nv_wm[name]; p=os.path.join(NV,f)
    with open(p,"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; h=json.loads(fh.read(n)); t=h[name]
        fh.seek(8+n+t["data_offsets"][0]); return fh.read(k), t["dtype"], tuple(t["shape"])
diff=[];same=[];missing=[]
for nm in names:
    L=int(re.search(r"layers\.(\d+)\.",nm).group(1))
    db,ddt,dsh=dl_sample(nm); nb,ndt,nsh=nv_sample(nm)
    if nb is None: missing.append(L); continue
    if (ddt,dsh)!=(ndt,nsh): print("  SHAPE/DTYPE MISMATCH layer %d: %s %s vs %s %s"%(L,ddt,dsh,ndt,nsh)); continue
    (diff if db!=nb else same).append(L)
print("\nlayers where dealignai DIFFERS from nvidia stock (these are the abliterated ones):")
print("  ",diff)
print("layers identical to stock:")
print("  ",same)
if missing: print("in dealignai but not nvidia:",missing)
print("\n-> transplant set: %d tensors, %.2f GiB"%(len(diff),len(diff)*67108864/2**30))
