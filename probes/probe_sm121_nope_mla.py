import math
import torch

print("GPU capability:", torch.cuda.get_device_capability())
from flashinfer.mla import BatchMLAPagedAttentionWrapper

dev = torch.device("cuda")
NUM_HEADS = 32   # 64 total / TP2
CKV = 512        # kv_lora_rank
KPE = 0          # qk_rope_head_dim (NoPE)
TOPK = 2048      # index_topk
TOKS = 4
KVLEN = 128
SM_SCALE = 1.0 / math.sqrt(256 + KPE)  # qk_head_dim 256

for backend in ("fa3", "fa2", "auto"):
    try:
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
        kv_indices = torch.zeros(TOKS * TOPK, dtype=torch.int32, device=dev)
        w = BatchMLAPagedAttentionWrapper(
            ws,
            qo_indptr=torch.zeros(TOKS + 1, dtype=torch.int32, device=dev),
            kv_indptr=torch.zeros(TOKS + 1, dtype=torch.int32, device=dev),
            kv_indices=kv_indices,
            kv_len_arr=torch.full((TOKS,), TOPK, dtype=torch.int32, device=dev),
            use_cuda_graph=True,
            backend=backend,
        )
        qo_cpu = torch.arange(TOKS + 1, dtype=torch.int32)
        kv_cpu = (qo_cpu * TOPK).to(torch.int32)
        lens_cpu = torch.full((TOKS,), KVLEN, dtype=torch.int32)
        w.plan(
            qo_cpu, kv_cpu, kv_indices, lens_cpu,
            NUM_HEADS, CKV, KPE, 1, False, SM_SCALE,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
        )
        q_nope = torch.randn(TOKS, NUM_HEADS, CKV, dtype=torch.bfloat16, device=dev)
        q_pe = torch.zeros(TOKS, NUM_HEADS, KPE, dtype=torch.bfloat16, device=dev)
        ckv_cache = torch.randn(TOKS * TOPK, 1, CKV, dtype=torch.bfloat16, device=dev)
        kpe_cache = torch.zeros(TOKS * TOPK, 1, KPE, dtype=torch.bfloat16, device=dev)
        out = w.run(q_nope, q_pe, ckv_cache, kpe_cache)
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(out.float()).all())
        print(f"{backend}: PASS shape={tuple(out.shape)} dtype={out.dtype} finite={finite}")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:280]
        print(f"{backend}: FAIL {type(e).__name__}: {msg}")
