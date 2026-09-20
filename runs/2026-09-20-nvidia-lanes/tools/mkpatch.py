#!/usr/bin/env python3
"""Emit patched kda.py and model.py that let the quantized attention projections actually be built as NVFP4.

Both g16 and g17 failed for one reason: glm5next hardcodes these projections bf16
  kda.py:172    vllm_config.quant_config = None      (whole KDA submodule tree)
  model.py:331  quant_config=None                    (MLA projections)
so a packed NVFP4 (out, in/2) weight cannot load into a bf16 (out, in) parameter.

Deliberately NOT touched, because these really are bf16 in every pack on this fleet and are also in the
checkpoint's ignore list:
  model.py:1090      vision tower (quantizing it yields NaN image features)
  attention.py:263   indexer wk_weights_proj
"""
import os,shutil
SRCD="/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia"
OUT="/patches"
os.makedirs(OUT,exist_ok=True)
# --- kda.py: stop nulling the quant config for the KDA submodule tree
p=os.path.join(SRCD,"kda.py"); s=open(p).read()
old="""        saved_quant_config = vllm_config.quant_config
        vllm_config.quant_config = None
        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config"""
new="""        # NVFP4-attn build (2026-09-20): the KDA projections ARE quantized in this
        # checkpoint, so do NOT strip the quant config. Modules that remain bf16
        # (indexer.*, f_b/g_b, q_a/kv_a, eh_proj) are protected by the checkpoint's
        # own `ignore` list, which is_layer_skipped() honours per layer.
        super().__init__(config, vllm_config, prefix)"""
assert old in s, "kda.py anchor missing"
s=s.replace(old,new); open(os.path.join(OUT,"kda.py"),"w").write(s)
# --- model.py: pass the real quant config to the MLA projections
p=os.path.join(SRCD,"model.py"); s=open(p).read()
old2="                quant_config=None,  # MLA projections are BF16 in checkpoint"
new2="                quant_config=vllm_config.quant_config,  # NVFP4-attn build: MLA projections are quantized"
assert s.count(old2)==1, "model.py anchor count %d"%s.count(old2)
s=s.replace(old2,new2); open(os.path.join(OUT,"model.py"),"w").write(s)
print("wrote kda.py and model.py to",OUT)
for f in ("kda.py","model.py"):
    print(" ",f,os.path.getsize(os.path.join(OUT,f)),"bytes")
