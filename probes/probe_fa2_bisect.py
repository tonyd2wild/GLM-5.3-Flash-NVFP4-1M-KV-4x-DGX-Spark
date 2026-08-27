import math
import torch
from flashinfer.mla import BatchMLAPagedAttentionWrapper

dev = torch.device("cuda")
torch.manual_seed(0)
H, CKV, KPE, TOPK = 32, 512, 0, 2048
SM_SCALE = 1.0 / math.sqrt(256)

def run_case(toks, len_fn, label):
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
    ckv_cache = torch.randn(NP, 1, CKV, dtype=torch.bfloat16, device=dev) * 0.5
    kpe_cache = torch.zeros(NP, 1, KPE, dtype=torch.bfloat16, device=dev)
    q_nope = torch.randn(toks, H, CKV, dtype=torch.bfloat16, device=dev) * 0.5
    q_pe = torch.zeros(toks, H, KPE, dtype=torch.bfloat16, device=dev)
    g = torch.Generator(device="cpu").manual_seed(1)
    idx = torch.zeros(toks, TOPK, dtype=torch.int32)
    for i in range(toks):
        L = int(lens_cpu[i])
        idx[i, :L] = torch.randperm(NP, generator=g)[:L].to(torch.int32)
    kv_indices.copy_(idx.reshape(-1).to(dev))
    w.plan(qo_cpu, kv_cpu, kv_indices, lens_cpu, H, CKV, KPE, 1, False, SM_SCALE,
           q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    out = w.run(q_nope, q_pe, ckv_cache, kpe_cache)
    torch.cuda.synchronize()
    bad = int((~torch.isfinite(out.float())).sum())
    print(f"{label}: toks={toks} bad={bad}", "NAN!" if bad else "clean", flush=True)

run_case(6, lambda i: 2048, "decode6-kv2048")
run_case(1, lambda i: 2048, "decode1-kv2048")
run_case(64, lambda i: 64, "t64-kv64")
run_case(64, lambda i: 2048, "t64-kv2048")
run_case(128, lambda i: i + 1, "causal-128")
run_case(256, lambda i: i + 1, "causal-256")
run_case(512, lambda i: i + 1, "causal-512")
