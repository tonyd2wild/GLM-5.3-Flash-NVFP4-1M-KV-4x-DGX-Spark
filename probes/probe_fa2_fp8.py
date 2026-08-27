import math
import torch
from flashinfer.mla import BatchMLAPagedAttentionWrapper

dev = torch.device("cuda")
torch.manual_seed(0)
H, CKV, KPE, TOPK = 32, 512, 0, 2048
SM_SCALE = 1.0 / math.sqrt(256)

def run_case(toks, len_fn, label, check_ref=False):
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    kv_indices = torch.zeros(toks * TOPK, dtype=torch.int32, device=dev)
    w = BatchMLAPagedAttentionWrapper(
        ws,
        qo_indptr=torch.zeros(toks + 1, dtype=torch.int32, device=dev),
        kv_indptr=torch.zeros(toks + 1, dtype=torch.int32, device=dev),
        kv_indices=kv_indices,
        kv_len_arr=torch.full((toks,), TOPK, dtype=torch.int32, device=dev),
        use_cuda_graph=True,
        backend="fa2",
    )
    qo_cpu = torch.arange(toks + 1, dtype=torch.int32)
    kv_cpu = (qo_cpu * TOPK).to(torch.int32)
    lens_cpu = torch.tensor([len_fn(i) for i in range(toks)], dtype=torch.int32)
    NP = 65536
    kv_bf16 = torch.randn(NP, 1, CKV, dtype=torch.bfloat16, device=dev) * 0.5
    ckv_cache = kv_bf16.to(torch.float8_e4m3fn)
    kpe_cache = torch.zeros(NP, 1, KPE, dtype=torch.float8_e4m3fn, device=dev)
    q_nope = torch.randn(toks, H, CKV, dtype=torch.bfloat16, device=dev) * 0.5
    q_pe = torch.zeros(toks, H, KPE, dtype=torch.bfloat16, device=dev)
    g = torch.Generator(device="cpu").manual_seed(1)
    idx = torch.zeros(toks, TOPK, dtype=torch.int32)
    for i in range(toks):
        L = int(lens_cpu[i])
        idx[i, :L] = torch.randperm(NP, generator=g)[:L].to(torch.int32)
    kv_indices.copy_(idx.reshape(-1).to(dev))
    w.plan(qo_cpu, kv_cpu, kv_indices, lens_cpu, H, CKV, KPE, 1, False, SM_SCALE,
           q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn)
    out = w.run(q_nope, q_pe, ckv_cache, kpe_cache, ckv_scale=1.0, kpe_scale=1.0)
    torch.cuda.synchronize()
    bad = int((~torch.isfinite(out.float())).sum())
    msg = f"{label}: toks={toks} bad={bad}"
    if check_ref and bad == 0:
        i = 0
        L = int(lens_cpu[0])
        sel = idx[0, :L].long()
        k = ckv_cache[sel, 0, :].float()
        qi = q_nope[0].float()
        attn = torch.softmax(qi @ k.T * SM_SCALE, dim=-1)
        ref0 = attn @ k
        rel = ((out[0].float() - ref0).abs().max() / ref0.abs().max()).item()
        msg += f" rel_err={rel:.4f}"
    print(msg, "NAN!" if bad else "clean", flush=True)

run_case(3, lambda i: 64, "fp8-small", check_ref=True)
run_case(64, lambda i: i + 1, "fp8-causal-64")
run_case(128, lambda i: i + 1, "fp8-causal-128")
run_case(1024, lambda i: min(i + 1, 2048), "fp8-causal-1024")
run_case(6, lambda i: 2048, "fp8-decode6")
