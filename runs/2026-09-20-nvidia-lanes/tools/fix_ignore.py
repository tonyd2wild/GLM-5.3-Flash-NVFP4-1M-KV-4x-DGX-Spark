#!/usr/bin/env python3
"""Use the ignore list that is empirically proven to boot (from the working LibertAI NVFP4-attention build)
rather than a hand-derived one. Same architecture, same module names, same quantized set, so the same list
applies. Hand-deriving from nvidia's broad wildcards produced an entry for the fused target
in_proj_qkvbfg_a, which made vLLM build that layer bf16 and the packed weights could not load.
"""
import json,sys
PROVEN=["lm_head","model.language_model.embed_tokens",
        "*.self_attn.f_b_proj","*.self_attn.g_b_proj","*.self_attn.fused_qkv_a_proj",
        "*.self_attn.q_a_proj","*.self_attn.kv_a_proj_with_mqa",
        "*.self_attn.indexer.wk_weights_proj","*.self_attn.indexer.wk",
        "*.self_attn.indexer.weights_proj","*.self_attn.indexer.wq_b",
        "*.mlp.gate","*.eh_proj","model.visual.*"]
for D in sys.argv[1:]:
    p=D+"/config.json"
    c=json.load(open(p)); q=c["quantization_config"]
    before=len(q.get("ignore",[]))
    q["ignore"]=list(PROVEN); q["quant_algo"]="W4A16_NVFP4"
    c["quantization_config"]=q
    json.dump(c,open(p,"w"),indent=1)
    print("%-34s ignore %d -> %d, quant_algo %s"%(D.split("/")[-1],before,len(PROVEN),q["quant_algo"]))
