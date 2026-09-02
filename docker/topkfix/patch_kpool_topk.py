#!/usr/bin/env python3
"""Custom sm_12x torch.topk routing for GLM-5.3-Flash kpool sparse-attention indexer.

Fixes the GB10 crash:
  launch_persistent_topk ... persistent_topk would oversubscribe and the
  FilteredTopK fallback requires >=128KB smem per block (have 101376).
  total_ctas=60 > num_sms*occupancy=48 (TopK=512)

Root cause: the fused indexer top-k kernels (persistent_topk decode,
top_k_per_row_prefill prefill) launch CTAs proportional to logits width
(== max_model_len), so large context oversubscribes GB10's 48 SMs and trips a
FilteredTopK fallback needing >=128KB smem/block (GB10 has ~99KB/101376B).
Exact torch.topk is context-length independent.

GLM-5.3 kpool: logits/seq_lens are POOL-granular (compress_ratio == index_kpool);
selection is on pools (select_k = topk_tokens // index_kpool), matching the
kernel contract. The -1 padding sentinel is preserved for the expand kernels.
"""
import sys
from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = VLLM / "model_executor/layers/sparse_attn_indexer_kpool.py"

ANCHOR_HELPERS = "RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024\n"

HELPERS = '''RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# GB10 / consumer Blackwell (sm_12x): the fused indexer top-k kernels
# (persistent_topk decode / top_k_per_row_prefill prefill) launch CTAs sized
# to logits width (== max_model_len), so large context oversubscribes the 48
# SMs and crashes on the FilteredTopK fallback (needs >=128KB smem/block;
# GB10 exposes ~99KB/101376B). torch.topk is exact and context-length
# independent. kpool logits/seq_lens are POOL-granular; selection is on pools
# (select_k = topk_tokens // index_kpool), matching the fused-kernel contract.
_USE_TORCH_TOPK = current_platform.is_device_capability_family(120)


def _torch_decode_topk(logits, seq_lens, topk_indices, topk_tokens):
    R, N = logits.shape
    dev = logits.device
    k = min(int(topk_tokens), N)
    cols = torch.arange(N, device=dev)
    # seq_lens is 2D (B, next_n) for native spec decode; logits rows are the
    # flattened (B*next_n) tokens in the same order.
    sl = seq_lens.reshape(-1)[:R].to(torch.long)
    valid = cols[None, :] < sl[:, None]
    masked = torch.where(valid, logits, torch.full_like(logits, float("-inf")))
    _, idx = torch.topk(masked, k, dim=-1)
    sel_valid = torch.gather(valid, 1, idx)
    idx = torch.where(sel_valid, idx.to(torch.int32),
                      torch.full_like(idx, -1, dtype=torch.int32))
    topk_indices[:R, :k] = idx
    if k < topk_tokens:
        topk_indices[:R, k:] = -1


def _torch_prefill_topk(logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices, topk_tokens):
    R, N = logits.shape
    dev = logits.device
    k = min(int(topk_tokens), N)
    cols = torch.arange(N, device=dev)
    ks = cu_seqlen_ks[:R].to(torch.long)[:, None]
    ke = cu_seqlen_ke[:R].to(torch.long)[:, None]
    valid = (cols[None, :] >= ks) & (cols[None, :] < ke)
    masked = torch.where(valid, logits, torch.full_like(logits, float("-inf")))
    _, idx = torch.topk(masked, k, dim=-1)
    sel_valid = torch.gather(valid, 1, idx)
    idx = torch.where(sel_valid, idx.to(torch.int32),
                      torch.full_like(idx, -1, dtype=torch.int32))
    topk_indices[:R, :k] = idx
    if k < topk_tokens:
        topk_indices[:R, k:] = -1
'''

# Decode path: insert torch.topk branch before the fused persistent_topk call.
DECODE_OLD = """        if current_platform.is_cuda() and select_k in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk("""

DECODE_NEW = """        if _USE_TORCH_TOPK:
            _torch_decode_topk(logits, seq_lens, topk_dst, select_k)
        elif current_platform.is_cuda() and select_k in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk("""

# Prefill path: insert torch.topk branch before the is_xpu()/top_k_per_row_prefill.
PREFILL_OLD = """            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_prefill(  # type: ignore[attr-defined]
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )
            else:
                torch.ops._C.top_k_per_row_prefill("""

PREFILL_NEW = """            if _USE_TORCH_TOPK:
                _torch_prefill_topk(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    select_k,
                )
            elif current_platform.is_xpu():
                xpu_ops.top_k_per_row_prefill(  # type: ignore[attr-defined]
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )
            else:
                torch.ops._C.top_k_per_row_prefill("""


def main():
    src = TARGET.read_text()

    # 1. helpers (anchor must appear once)
    assert src.count(ANCHOR_HELPERS) == 1, "anchor RADIX_TOPK_WORKSPACE_SIZE not unique"
    src = src.replace(ANCHOR_HELPERS, HELPERS, 1)

    # 2. decode
    assert src.count(DECODE_OLD) == 1, "decode hunk not found/unique"
    src = src.replace(DECODE_OLD, DECODE_NEW, 1)

    # 3. prefill
    assert src.count(PREFILL_OLD) == 1, "prefill hunk not found/unique"
    src = src.replace(PREFILL_OLD, PREFILL_NEW, 1)

    TARGET.write_text(src)
    print("PATCH_OK: _torch_decode_topk/_torch_prefill_topk wired into kpool indexer")
    # sanity
    assert "_USE_TORCH_TOPK" in src
    assert src.count("_torch_decode_topk") >= 2
    assert src.count("_torch_prefill_topk") >= 2
    print("SANITY_OK: helpers + both call sites present")


if __name__ == "__main__":
    main()
