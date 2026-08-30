#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SM121-PORT of vLLM PR #52816 (DFlash2: local convolution + candidate
selector, merge commit b389ac294) onto radixark/vllm-glm53-flash:sm121-v8
(vLLM 0.1.dev20051+g487ecf187).

Run INSIDE the image at build time, AFTER copying the new files in:

    cp qwen3_dflash2.py  $VLLM/model_executor/models/qwen3_dflash2.py
    cp -r dflash2/       $VLLM/v1/worker/gpu/spec_decode/dflash2/
    python3 patch_registry_and_select.py [optional-vllm-root]

Edits (all anchored string replacements, asserted, ast-checked, idempotent):
  1. model_executor/models/registry.py         DFlash2DraftModel entry
  2. v1/worker/gpu/spec_decode/__init__.py     route DFlash2 drafts to the
                                               DFlash2 speculator
  3. model_executor/models/qwen3_dflash.py     decoder_layer_cls / model_cls
                                               subclass hooks + is_causal
  4. v1/worker/gpu/spec_decode/speculator.py   draft_logits_spec hook
  5. config/vllm.py                            force V2 model runner for a
                                               DFlash2 draft
  6. model_executor/layers/logits_processor.py get_top_k_tokens (verbatim
                                               from the PR)
"""

import ast
import os
import sys

VLLM_ROOT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/usr/local/lib/python3.12/dist-packages/vllm"
)


def patch_file(rel_path: str, edits: list[tuple[str, str, str]]) -> None:
    """Apply (marker, old, new) edits to VLLM_ROOT/rel_path.

    marker: substring whose presence means the edit is already applied (skip).
    old:    anchor text that must occur exactly once; replaced by new.
    """
    path = os.path.join(VLLM_ROOT, rel_path)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    changed = False
    for marker, old, new in edits:
        if marker in src:
            print(f"[skip] {rel_path}: already applied ({marker[:48]!r})")
            continue
        assert old in src, f"ANCHOR NOT FOUND in {rel_path}:\n{old[:200]!r}"
        assert src.count(old) == 1, (
            f"ANCHOR NOT UNIQUE ({src.count(old)}x) in {rel_path}:\n{old[:200]!r}"
        )
        src = src.replace(old, new)
        changed = True
        print(f"[edit] {rel_path}: applied ({marker[:48]!r})")
    ast.parse(src, filename=path)  # syntax gate before writing
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"[ok]   {rel_path}: written, ast.parse clean")
    else:
        print(f"[ok]   {rel_path}: no changes needed, ast.parse clean")


# --------------------------------------------------------------------------
# 1. registry.py: DFlash2DraftModel entry
# --------------------------------------------------------------------------
patch_file(
    "model_executor/models/registry.py",
    [
        (
            '"DFlash2DraftModel"',
            '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n',
            '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n'
            "    # SM121-PORT PR#52816\n"
            '    "DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),\n',
        ),
    ],
)

# --------------------------------------------------------------------------
# 2. spec_decode/__init__.py: speculator selection
# --------------------------------------------------------------------------
patch_file(
    "v1/worker/gpu/spec_decode/__init__.py",
    [
        (
            "DFlash2Speculator",
            '    if speculative_config.method == "dflash":\n'
            "        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (\n",
            '    if speculative_config.method == "dflash":\n'
            "        # SM121-PORT PR#52816: route DFlash2 drafts (declared by\n"
            "        # architecture, or by a dflash_config carrying a candidate\n"
            "        # selector) to the DFlash2 speculator. On the plain DFlash path\n"
            "        # such a checkpoint would silently draft as DFlash1.\n"
            "        _draft_cfg = speculative_config.draft_model_config\n"
            '        _dflash_cfg = getattr(_draft_cfg.hf_config, "dflash_config", None) or {}\n'
            '        if "DFlash2DraftModel" in (_draft_cfg.architectures or []) or (\n'
            '            "selector_rank" in _dflash_cfg\n'
            "        ):\n"
            "            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (\n"
            "                DFlash2Speculator,\n"
            "            )\n"
            "\n"
            "            return DFlash2Speculator(vllm_config, device)\n"
            "        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (\n",
        ),
    ],
)

# --------------------------------------------------------------------------
# 3. qwen3_dflash.py: subclass hooks + explicit is_causal resolution
# --------------------------------------------------------------------------
patch_file(
    "model_executor/models/qwen3_dflash.py",
    [
        (
            "SM121-PORT causal",
            '    """``dflash_config.causal`` overrides all layers; else only SWA'
            ' layers causal."""\n'
            '    override = (getattr(config, "dflash_config", None) or {}).get("causal")\n',
            '    """Resolve explicit causality before falling back to legacy layer'
            ' defaults."""\n'
            "    # SM121-PORT causal (PR#52816, extended): honor a top-level `is_causal`\n"
            "    # and a dflash_config-level `is_causal` (the GLM53 DFlash2 drafter ships\n"
            "    # the latter) before the legacy `causal` key.\n"
            '    dflash_cfg = getattr(config, "dflash_config", None) or {}\n'
            '    is_causal = getattr(config, "is_causal", None)\n'
            "    if is_causal is None:\n"
            '        is_causal = dflash_cfg.get("is_causal")\n'
            "    if is_causal is not None:\n"
            "        return bool(is_causal)\n"
            '    override = dflash_cfg.get("causal")\n',
        ),
        (
            "decoder_layer_cls = DFlashQwen3DecoderLayer",
            "class DFlashQwen3Model(nn.Module):\n"
            "    hf_to_vllm_mapper = WeightsMapper(\n",
            "class DFlashQwen3Model(nn.Module):\n"
            "    # SM121-PORT PR#52816: subclass hook for DFlash2.\n"
            "    decoder_layer_cls = DFlashQwen3DecoderLayer\n"
            "\n"
            "    hf_to_vllm_mapper = WeightsMapper(\n",
        ),
        (
            "self.decoder_layer_cls(",
            "                DFlashQwen3DecoderLayer(\n"
            "                    current_vllm_config,\n",
            "                self.decoder_layer_cls(  # SM121-PORT PR#52816\n"
            "                    current_vllm_config,\n",
        ),
        (
            "model_cls = DFlashQwen3Model",
            "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
            '    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):\n',
            "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
            "    # SM121-PORT PR#52816: subclass hook for DFlash2.\n"
            "    model_cls = DFlashQwen3Model\n"
            "\n"
            '    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):\n',
        ),
        (
            "self.model = self.model_cls(",
            "        self.model = DFlashQwen3Model(\n"
            "            vllm_config=vllm_config,\n",
            "        self.model = self.model_cls(  # SM121-PORT PR#52816\n"
            "            vllm_config=vllm_config,\n",
        ),
    ],
)

# --------------------------------------------------------------------------
# 4. spec_decode/speculator.py: draft_logits_spec hook
# --------------------------------------------------------------------------
patch_file(
    "v1/worker/gpu/spec_decode/speculator.py",
    [
        (
            "self.draft_logits_spec(",
            "            self.draft_logits = torch.zeros(\n"
            "                self.max_num_reqs,\n"
            "                self.num_speculative_steps,\n"
            "                self.vocab_size,\n"
            "                dtype=vllm_config.model_config.head_dtype,\n"
            "                device=device,\n"
            "            )\n",
            "            # SM121-PORT PR#52816: dtype/fill via draft_logits_spec so\n"
            "            # DFlash2 can cache a sparse fp32/-inf distribution.\n"
            "            dtype, fill = self.draft_logits_spec(vllm_config)\n"
            "            self.draft_logits = torch.full(\n"
            "                (\n"
            "                    self.max_num_reqs,\n"
            "                    self.num_speculative_steps,\n"
            "                    self.vocab_size,\n"
            "                ),\n"
            "                fill,\n"
            "                dtype=dtype,\n"
            "                device=device,\n"
            "            )\n",
        ),
        (
            "def draft_logits_spec(",
            "    def _validate_local_argmax_reduction(self) -> None:\n",
            "    def draft_logits_spec(\n"
            "        self, vllm_config: VllmConfig\n"
            "    ) -> tuple[torch.dtype, float]:\n"
            '        """Dtype and fill for the cached proposal distribution.\n'
            "\n"
            "        Speculators that write only a subset of columns each step\n"
            "        override this. (SM121-PORT PR#52816)\n"
            '        """\n'
            "        return vllm_config.model_config.head_dtype, 0.0\n"
            "\n"
            "    def _validate_local_argmax_reduction(self) -> None:\n",
        ),
    ],
)

# --------------------------------------------------------------------------
# 5. config/vllm.py: force the V2 model runner for a DFlash2 draft
# --------------------------------------------------------------------------
patch_file(
    "config/vllm.py",
    [
        (
            "_is_dflash2_draft():",
            "        if self._dflash_needs_multi_kv_group():\n"
            "            return True\n",
            "        if self._dflash_needs_multi_kv_group():\n"
            "            return True\n"
            "\n"
            "        # SM121-PORT PR#52816: the DFlash2 candidate selector exists only\n"
            "        # in the V2 speculator; on V1 the same checkpoint would silently\n"
            "        # draft as DFlash1. Force V2 as for dspark.\n"
            "        if self._is_dflash2_draft():\n"
            "            return True\n",
        ),
        (
            "def _is_dflash2_draft(",
            "    def _dflash_needs_multi_kv_group(self) -> bool:\n",
            "    def _is_dflash2_draft(self) -> bool:\n"
            '        """SM121-PORT PR#52816: whether the DFlash draft is a DFlash2\n'
            "        one, by the same signals the speculator selection uses\n"
            '        (v1/worker/gpu/spec_decode/__init__.py)."""\n'
            "        spec = self.speculative_config\n"
            '        if spec is None or spec.method != "dflash":\n'
            "            return False\n"
            '        draft_config = getattr(spec, "draft_model_config", None)\n'
            "        if draft_config is None:\n"
            "            return False\n"
            '        if "DFlash2DraftModel" in (draft_config.architectures or []):\n'
            "            return True\n"
            "        dflash_cfg = (\n"
            '            getattr(draft_config.hf_config, "dflash_config", None) or {}\n'
            "        )\n"
            '        return "selector_rank" in dflash_cfg\n'
            "\n"
            "    def _dflash_needs_multi_kv_group(self) -> bool:\n",
        ),
    ],
)

# --------------------------------------------------------------------------
# 6. logits_processor.py: get_top_k_tokens (verbatim from the merged PR)
# --------------------------------------------------------------------------
patch_file(
    "model_executor/layers/logits_processor.py",
    [
        (
            "_flashinfer_topk",
            "from vllm.platforms import current_platform\n",
            "from vllm.platforms import current_platform\n"
            "\n"
            "# SM121-PORT PR#52816: vocab-parallel top-k for the DFlash2 candidate\n"
            "# selector -- verbatim from the merged logits_processor.py (b389ac294).\n"
            "from collections.abc import Callable\n"
            "from functools import cache\n"
            "\n"
            "from vllm.logger import init_logger\n"
            "from vllm.utils.flashinfer import has_flashinfer\n"
            "\n"
            "logger = init_logger(__name__)\n"
            "\n"
            "\n"
            "@cache\n"
            "def _flashinfer_topk() -> (\n"
            "    Callable[..., tuple[torch.Tensor, torch.Tensor]] | None\n"
            "):\n"
            '    """FlashInfer\'s radix top-k, or None for torch.topk.\n'
            "\n"
            "    The top-k spans the vocabulary, where the radix kernel is about twice\n"
            "    torch.topk.\n"
            '    """\n'
            "    if not current_platform.is_cuda():\n"
            "        return None\n"
            "    if not has_flashinfer():\n"
            "        logger.info_once(\n"
            '            "flashinfer is unavailable; vocab-parallel top-k uses '
            'torch.topk, "\n'
            '            "at roughly half the speed."\n'
            "        )\n"
            "        return None\n"
            "    from flashinfer import top_k\n"
            "\n"
            "    return top_k\n"
            "\n"
            "\n"
            "def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:\n"
            "    impl = _flashinfer_topk()\n"
            "    if impl is None or not scores.is_cuda:\n"
            "        return torch.topk(scores, k, dim=-1)\n"
            "    return impl(scores, k, sorted=True, deterministic=True)\n",
        ),
        (
            "def get_top_k_tokens(",
            "    def extra_repr(self) -> str:\n",
            "    # SM121-PORT PR#52816: verbatim from the merged logits_processor.py.\n"
            "    def get_top_k_tokens(\n"
            "        self,\n"
            "        lm_head: VocabParallelEmbedding,\n"
            "        hidden_states: torch.Tensor,\n"
            "        k: int,\n"
            "        embedding_bias: torch.Tensor | None = None,\n"
            "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"
            '        """Vocab-parallel top-k without all-gathering full logits.\n'
            "\n"
            "        The `get_top_tokens` reduction widened from one token to k,\n"
            "        returning the values as well as the global ids. Communication is\n"
            "        O(batch * 2k * tp_size) rather than O(batch * vocab_size).\n"
            "\n"
            "        Scale and soft cap are applied to the k selected values rather\n"
            "        than the whole vocabulary; both are monotonic, so the selection\n"
            "        is the same and only k entries are touched.\n"
            '        """\n'
            "        if self.scale <= 0.0 and self.scale != 1.0:\n"
            "            raise ValueError(\n"
            '                "The local top-k reduction optimization is not supported '
            'for "\n'
            '                "non-positive logit scaling factors."\n'
            "            )\n"
            "\n"
            "        logits = self._apply_head(lm_head, hidden_states, embedding_bias)\n"
            "\n"
            "        # Mask out padding entries beyond org_vocab_size on this shard.\n"
            "        num_pad = lm_head.shard_indices.num_org_vocab_padding\n"
            "        if num_pad > 0:\n"
            '            logits[..., -num_pad:] = -float("inf")\n'
            "\n"
            "        values, ids = _topk(logits, k)\n"
            "        # Convert shard-local indices to global vocab indices.\n"
            "        ids = ids.to(torch.int64) + lm_head.shard_indices.org_vocab_start_index\n"
            "\n"
            "        if lm_head.tp_size > 1:\n"
            "            values = tensor_model_parallel_all_gather(values, dim=-1)\n"
            "            ids = tensor_model_parallel_all_gather(ids, dim=-1)\n"
            "            values, selected = _topk(values, k)\n"
            "            ids = ids.gather(-1, selected)\n"
            "\n"
            "        values = values.float()\n"
            "        if self.scale != 1.0:\n"
            "            values = values * self.scale\n"
            "        if self.soft_cap is not None:\n"
            "            values = torch.tanh(values / self.soft_cap) * self.soft_cap\n"
            "        return ids, values\n"
            "\n"
            "    def extra_repr(self) -> str:\n",
        ),
    ],
)

print("\nAll patches applied and ast-checked. VLLM_ROOT =", VLLM_ROOT)
