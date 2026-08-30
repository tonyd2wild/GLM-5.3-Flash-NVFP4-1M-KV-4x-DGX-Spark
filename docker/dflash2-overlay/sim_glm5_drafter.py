"""Simulation harness for patch_glm5_drafter_group.py.

Drives the patched kv_cache_utils end to end on CPU with two geometries:

GEOMETRY A -- the real lane-1 numbers (boot 8, ~/lane1_fail8.log):
  MLA block 4608 @ 512 B/token -> mla_page 2,359,296; drafter SWA 4 kv heads
  x 128 head fp8 -> 1024 B/token. Exact fit: drafter block rescales to 2304,
  real page == mla_page, NO page_size_padded, rides MLA tensors 0-4.
  Includes a runner-side reshape repro of the boot-8 crash: FlashInfer picks
  kernel block 64 (manager block 2304 -> ratio 36); with the contiguous view
  the required storage must EQUAL the MLA tensor. (The old padded strided
  view demanded ratio x more -- setStorage out of bounds.)

GEOMETRY B -- a mismatched drafter (3 kv heads -> 768 B/token; exact-fit
  block 3072 fails the LCM guard vs MLA block 4608): standalone mode.
  Drafter keeps its spec, gets compact per-layer tensors, and its pages are
  charged in the per-block cost everywhere.

Asserts groups, layout detection, tensor emission, storage-bound safety, and
that _pool_bytes_per_block / emission / max-mem accounting agree.
"""

from types import SimpleNamespace

import torch

from vllm.v1.core import kv_cache_utils as K
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache

MLA_BLOCK = 4608
MLA_PAGE = 2_359_296  # 4608 * 512 B/token (real lane-1 fp8 MLA geometry)

vllm_config = SimpleNamespace(
    parallel_config=SimpleNamespace(
        pipeline_parallel_size=1, decode_context_parallel_size=1
    ),
    model_config=SimpleNamespace(max_model_len=262144),
    cache_config=SimpleNamespace(
        num_gpu_blocks_override=None, mamba_cache_mode="none"
    ),
    max_in_flight_tokens=8192,
    speculative_config=None,
)


def build_spec(draft_kv_heads: int) -> dict:
    spec: dict = {}
    for i in range(34):
        spec[f"model.layers.{i}.kda"] = MambaSpec(
            block_size=16, shapes=((128, 128),), dtypes=(torch.float32,)
        )
    for i in range(11):
        spec[f"model.layers.{i}.mla"] = MLAAttentionSpec(
            block_size=MLA_BLOCK, num_kv_heads=1, head_size=512, dtype=torch.uint8
        )
        spec[f"model.layers.{i}.indexer"] = MLAAttentionSpec(
            block_size=MLA_BLOCK,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.uint8,
            compress_ratio=4,
        )
        spec[f"model.layers.{i}.kpool_tail"] = KpoolTailSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=4,
        )
    for i in range(5):
        spec[f"drafter.layers.{i}.attn"] = SlidingWindowSpec(
            block_size=2048,
            num_kv_heads=draft_kv_heads,
            head_size=128,
            dtype=torch.uint8,
            sliding_window=2048,
        )
    return spec


def run_geometry(spec: dict, label: str):
    groups = K._get_kv_cache_groups_glm5_next(vllm_config, spec)
    assert groups is not None, f"[{label}] glm5_next path rejected the model"
    layout = K._glm5_next_tensor_layout(groups)
    assert layout is not None, f"[{label}] layout detection failed"
    (
        attn_g,
        mamba_gs,
        mla_names,
        idx_names,
        mla_page,
        idx_page,
        tail_names,
        tail_page,
        draft_g,
    ) = layout
    assert mla_page == MLA_PAGE
    cfg = K.get_kv_cache_config_from_groups(vllm_config, groups, AVAIL)
    pb = K._pool_bytes_per_block(vllm_config, groups)
    assert cfg.num_blocks == AVAIL // pb, (
        f"[{label}] emission and _pool_bytes_per_block disagree"
    )
    need = K._max_memory_usage_bytes_from_groups(vllm_config, groups)
    covered = set()
    for t in cfg.kv_cache_tensors:
        covered.update(t.shared_by)
    assert covered == set(spec.keys()), set(spec.keys()) - covered
    print(
        f"[{label}] groups={len(groups)} num_blocks={cfg.num_blocks} "
        f"tensors={len(cfg.kv_cache_tensors)} bytes/block={pb} "
        f"max-mem={need} fits={need <= AVAIL}"
    )
    return groups, layout, cfg, pb, need


def bind_and_reshape(cfg, layer_name: str, d_spec, kernel_block: int):
    """Emulate _allocate_kv_cache + _reshape_kv_cache for one drafter layer.

    Returns (view, raw) after asserting the view's required storage fits the
    tensor the layer is actually bound to -- the exact boot-8 failure mode.
    """
    tensor = next(t for t in cfg.kv_cache_tensors if layer_name in t.shared_by)
    raw = torch.zeros(tensor.size, dtype=torch.int8)
    assert raw.numel() % d_spec.page_size_bytes == 0
    num_blocks = raw.numel() // d_spec.page_size_bytes
    assert num_blocks == cfg.num_blocks
    ratio = d_spec.block_size // kernel_block
    kernel_num_blocks = num_blocks * ratio
    shape = (kernel_num_blocks, 2, kernel_block, d_spec.num_kv_heads, d_spec.head_size)
    content = 2 * kernel_block * d_spec.num_kv_heads * d_spec.head_size
    required = kernel_num_blocks * content
    assert required <= raw.numel(), (
        f"boot-8 regression: view needs {required} B but tensor has "
        f"{raw.numel()} B"
    )
    view = _reshape_attention_kv_cache(
        raw, d_spec, shape, tuple(range(5)), kernel_num_blocks, packing=None
    )
    assert view.shape == shape
    return view, raw


AVAIL = 4_445_787_956  # lane-1 kv_cache_memory_bytes

# ===========================================================================
# GEOMETRY A: real lane-1 numbers -> EXACT FIT
# ===========================================================================
spec_a = build_spec(draft_kv_heads=4)  # 1024 B/token, matches boot 8
assert spec_a["model.layers.0.mla"].page_size_bytes == MLA_PAGE
groups_a, layout_a, cfg_a, pb_a, _ = run_geometry(spec_a, "A/exact-fit")
draft_g = layout_a[8]
assert draft_g is groups_a[-1]
d0 = next(iter(draft_g.kv_cache_spec.kv_cache_specs.values()))
assert type(d0) is SlidingWindowSpec
assert d0.block_size == 2304, d0.block_size  # 2,359,296 // 1024
assert d0.page_size_padded is None, "padded drafter spec = boot-8 crash"
assert d0.page_size_bytes == d0.real_page_size_bytes == MLA_PAGE
print(f"[A] drafter: block 2048->{d0.block_size}, real page == mla_page, no padding")

# per-block cost unchanged vs drafterless baseline
base_spec = {k: v for k, v in spec_a.items() if not k.startswith("drafter.")}
base_groups = K._get_kv_cache_groups_glm5_next(vllm_config, base_spec)
assert K._glm5_next_tensor_layout(base_groups)[8] is None
assert pb_a == K._pool_bytes_per_block(vllm_config, base_groups)
print(f"[A] bytes/block unchanged by drafter: {pb_a}")

# tensor wiring: 22 tensors, drafter layer i rides MLA tensor i, none standalone
assert len(cfg_a.kv_cache_tensors) == 22
draft_names = list(draft_g.layer_names)
mla_names = layout_a[2]
for i in range(11):
    t = cfg_a.kv_cache_tensors[i]
    assert t.size == MLA_PAGE * cfg_a.num_blocks
    assert t.shared_by[0] == mla_names[i]
    if i < 5:
        assert t.shared_by[-1] == draft_names[i]
    else:
        assert not any(n in draft_names for n in t.shared_by)

# boot-8 repro: FlashInfer kernel block 64 (2304 -> ratio 36), contiguous view
view, raw = bind_and_reshape(cfg_a, draft_names[0], d0, kernel_block=64)
assert view.numel() == raw.numel()  # exact fit: view covers the MLA tensor fully
# kernel block j of manager block b starts at b * mla_page + j * kernel_page
kernel_page = 2 * 64 * 4 * 128
assert view[36].data_ptr() - view[0].data_ptr() == MLA_PAGE  # mgr block 1
assert view[1].data_ptr() - view[0].data_ptr() == kernel_page  # within block 0
print(
    f"[A] runner reshape OK under kernel split 2304->64 (x36): required "
    f"{view.numel()} B == tensor {raw.numel()} B; blocks stay page-aligned"
)

# ===========================================================================
# GEOMETRY B: 768 B/token drafter -> exact fit 3072 fails LCM guard vs 4608
# -> STANDALONE
# ===========================================================================
spec_b = build_spec(draft_kv_heads=3)
groups_b, layout_b, cfg_b, pb_b, need_b = run_geometry(spec_b, "B/standalone")
draft_gb = layout_b[8]
db = next(iter(draft_gb.kv_cache_spec.kv_cache_specs.values()))
assert db.block_size == 2048 and db.page_size_padded is None  # spec untouched
draft_page_b = db.page_size_bytes
assert draft_page_b == 2048 * 768
idx_page = layout_b[5]
assert pb_b == 11 * MLA_PAGE + 11 * idx_page + 5 * draft_page_b, pb_b
assert len(cfg_b.kv_cache_tensors) == 27  # 11 MLA + 11 idx + 5 drafter
draft_names_b = list(draft_gb.layer_names)
for i, t in enumerate(cfg_b.kv_cache_tensors[22:]):
    assert t.shared_by == [draft_names_b[i]]
    assert t.size == draft_page_b * cfg_b.num_blocks
for t in cfg_b.kv_cache_tensors[:22]:  # no drafter on MLA/idx tensors
    assert not any(n in draft_names_b for n in t.shared_by)
# runner reshape on the standalone tensor with a kernel split (2048 -> 64)
view_b, raw_b = bind_and_reshape(cfg_b, draft_names_b[0], db, kernel_block=64)
assert view_b.numel() == raw_b.numel()
print(
    f"[B] standalone: 5 compact tensors of {draft_page_b} B/block, per-block "
    f"cost includes them, reshape OK under kernel split"
)

print("\nALL SIMULATION CHECKS PASSED")
