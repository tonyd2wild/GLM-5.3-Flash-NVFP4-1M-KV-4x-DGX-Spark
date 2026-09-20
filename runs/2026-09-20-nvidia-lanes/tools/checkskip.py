import json
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
D="/models/m"
c=json.load(open(D+"/config.json")); ig=c["quantization_config"]["ignore"]
pm={"qkv_proj":["q_proj","k_proj","v_proj"],"gate_up_proj":["gate_proj","up_proj"],
    "in_proj_qkvbfg_a":["q_proj","k_proj","v_proj","b_proj","f_a_proj","g_a_proj"]}
tests=[
 ("model.language_model.layers.20.self_attn.in_proj_qkvbfg_a","MUST be quantized (its 6 constituents are)"),
 ("model.language_model.layers.20.self_attn.o_proj",          "MUST be quantized"),
 ("model.language_model.layers.20.mlp.shared_experts.gate_up_proj","MUST be quantized"),
 ("model.language_model.layers.20.self_attn.indexer.wk",      "must stay bf16"),
 ("model.language_model.layers.20.self_attn.f_b_proj",        "must stay bf16"),
 ("model.language_model.layers.20.self_attn.q_a_proj",        "must stay bf16"),
 ("model.language_model.layers.20.mlp.gate",                  "must stay bf16"),
 ("lm_head",                                                  "must stay bf16"),
 ("model.visual.blocks.0.attn.qkv",                           "must stay bf16"),
]
print("ignore entries: %d"%len(ig))
bad=0
for prefix,want in tests:
    sk=is_layer_skipped(prefix,ig,pm)
    need_skip = want.startswith("must stay")
    ok = (sk==need_skip)
    if not ok: bad+=1
    print("  %-62s skipped=%-5s %s  %s"%(prefix.split("language_model.")[-1],sk,want,"OK" if ok else "<-- WRONG"))
print("\n%s"%("all correct" if bad==0 else "%d WRONG - do not boot"%bad))
