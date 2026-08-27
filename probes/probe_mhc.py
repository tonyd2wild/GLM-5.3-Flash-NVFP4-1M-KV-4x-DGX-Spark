import torch
import vllm.model_executor.layers.mhc  # registers custom ops
from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch

dev = torch.device("cuda")
torch.manual_seed(0)
T, HC_MULT, HIDDEN = 4096, 4, 6144
HC3 = 2 * HC_MULT + HC_MULT * HC_MULT

residual = (torch.randn(T, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.5)
fn = torch.randn(HC3, HC_MULT * HIDDEN, dtype=torch.float32, device=dev) * 0.02
hc_scale = torch.rand(3, dtype=torch.float32, device=dev) + 0.5
hc_base = torch.randn(HC3, dtype=torch.float32, device=dev) * 0.1
args = (residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 3)

ref = mhc_pre_torch(*args)
tl = torch.ops.vllm.mhc_pre_tilelang(*args, 1, None, 0.0)
torch.cuda.synchronize()

names = ["post_mix", "comb_mix", "layer_input"]
worst = 0.0
for n, r, t in zip(names, ref, tl):
    d = (r.float() - t.float()).abs().max().item()
    scale = r.float().abs().max().item() or 1.0
    rel = d / scale
    worst = max(worst, rel)
    print(f"{n}: max_abs={d:.6f} rel={rel:.6f}")
print("nan_in_tilelang:", any(torch.isnan(t.float()).any().item() for t in tl)); print("VERDICT:", "MATCH" if worst < 0.03 else "MISMATCH")
