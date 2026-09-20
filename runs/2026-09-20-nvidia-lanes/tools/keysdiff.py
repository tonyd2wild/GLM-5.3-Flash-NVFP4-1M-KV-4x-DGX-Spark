#!/usr/bin/env python3
"""Was the keys abliteration really o_proj only?

keys-ablit = LibertAI body + (claimed) dealignai o_proj L15-45. If its routed-expert down_proj tensors are
byte-identical to LibertAI stock, then o_proj really was the only edit, and the reported 32/32 refusal bypass
came from ~3.5% of the intervention. If they differ, the expert weights were the active ingredient all along.
"""
import json,os,struct
STOCK="/var/tmp/glm-5.3-flash-nvfp4"                                     # LibertAI, unmodified
KEYS="/var/tmp/models/keys-glm-5.3-flash-nvfp4-ablit-l15-45-anchorstock" # LibertAI + dealignai o_proj
def head(M,f):
    with open(os.path.join(M,f),"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; return 8+n, json.loads(fh.read(n))
def sample(M,name,k=96):
    wm=json.load(open(os.path.join(M,"model.safetensors.index.json")))["weight_map"]
    if name not in wm: return None
    base,hdr=head(M,wm[name]); t=hdr[name]
    with open(os.path.join(M,wm[name]),"rb") as fh:
        fh.seek(base+t["data_offsets"][0]); return fh.read(k)
wm=json.load(open(os.path.join(KEYS,"model.safetensors.index.json")))["weight_map"]
import re
def probe(pattern,label,limit=8):
    names=[n for n in wm if re.search(pattern,n)][:limit]
    diff=same=0
    for n in names:
        a=sample(STOCK,n); b=sample(KEYS,n)
        if a is None or b is None: continue
        if a==b: same+=1
        else: diff+=1
    print("  %-44s checked %2d  identical %2d  DIFFER %2d"%(label,len(names),same,diff))
    return diff
print("keys-ablit vs LibertAI stock, first bytes of each tensor:")
d1=probe(r"layers\.(1[5-9]|2\d|3\d|4[0-5])\.self_attn\.o_proj\.weight$","o_proj L15-45 (the claimed edit)")
d2=probe(r"layers\.(1[5-9]|2\d|3\d|4[0-4])\.mlp\.experts\.\d+\.down_proj\.weight$","routed-expert down_proj (the real lever)")
d3=probe(r"layers\.\d+\.mlp\.shared_experts\.down_proj\.weight$","shared-expert down_proj")
d4=probe(r"embed_tokens\.weight$","embed_tokens")
print()
if d2==0 and d1>0:
    print("  -> keys really did edit ONLY o_proj. Their 32/32 claim rests on ~3.5% of the intervention.")
elif d2>0:
    print("  -> expert down_proj ALSO differ: o_proj was not the active ingredient.")
