#!/usr/bin/env python3
"""
patch_glm_aux_capture.py -- DFlash2 / EAGLE-3 target-side glue for GLM-5.3-Flash.

Edits vllm/models/glm5next/nvidia/model.py IN PLACE (build-time, inside the
radixark/vllm-glm53-flash:sm121-v8 image) so the Glm5Next* target model can
feed aux hidden states to a DFlash2 draft model:

  1. imports EagleModelMixin + SupportsEagle3 from
     vllm.model_executor.models.interfaces
  2. Glm5NextModel gains EagleModelMixin (provides the
     `aux_hidden_state_layers` store + `_set_aux_hidden_state_layers`)
  3. Glm5NextForCausalLM and Glm5NextForConditionalGeneration declare
     SupportsEagle3 (the interface's concrete default methods implement
     set_aux_hidden_state_layers / get_eagle3_default_aux_hidden_state_layers,
     routing through `.language_model` / `.model` automatically)
  4. the decoder-layer loop captures aux hidden states with mHC contraction
     (deferred hc_post materialized, then hc_contract == mean over hc
     streams), mirroring vllm/models/deepseek_v4/nvidia/model.py
     DeepseekV4Model.forward (aux capture at ~lines 1160-1213:
     `if idx + 1 in self.aux_hidden_state_layers: ... aux_recon.mean(dim=1)`
     + sp_all_gather)
  5. Glm5NextModel.forward returns (hidden_states, aux_hidden_states) when
     any layer was captured, exactly like DeepseekV4Model.forward

Layer-index semantics: gpu_model_runner._get_eagle3_aux_layers_from_config
converts DFlash `target_layer_ids` to id+1 before calling
set_aux_hidden_state_layers ("# Add 1 to convert DFlash's aux layer id
semantics"). The capture below therefore tests `idx + 1 in
self.aux_hidden_state_layers` -- identical to deepseek_v4 -- which captures
the OUTPUT of 0-based decoder layer `idx`. Net effect for
target_layer_ids=[5,14,24,33,42]: outputs of layers 5,14,24,33,42 are
captured, each contracted to [num_tokens, hidden_size].

Usage:
    python3 patch_glm_aux_capture.py [--model-file PATH] [--dry-run]

Idempotent: re-running on an already-patched file is a no-op (exit 0).
Fails loudly (AssertionError, nonzero exit) if any anchor is missing.
"""

from __future__ import annotations

import argparse
import ast
import sys

DEFAULT_MODEL_FILE = (
    "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py"
)

MARKER = "DFLASH2-AUX-CAPTURE"

# ---------------------------------------------------------------------------
# Anchored edits. Every anchor must appear EXACTLY ONCE in the target file.
# ---------------------------------------------------------------------------

EDIT_IMPORTS_ANCHOR = """\
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
"""

EDIT_IMPORTS_NEW = """\
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)
"""

EDIT_MODEL_CLASS_ANCHOR = """\
class Glm5NextModel(nn.Module):
"""

EDIT_MODEL_CLASS_NEW = """\
class Glm5NextModel(nn.Module, EagleModelMixin):
"""

EDIT_LOOP_ANCHOR = """\
        for layer in self._active_layers:
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
"""

EDIT_LOOP_NEW = """\
        # DFLASH2-AUX-CAPTURE (EAGLE-3 aux hidden states; mirrors
        # DeepseekV4Model.forward in vllm/models/deepseek_v4/nvidia/model.py).
        aux_hidden_states: list[torch.Tensor] = []
        for idx, layer in enumerate(self._active_layers, start=self.start_layer):
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
            if idx + 1 in self.aux_hidden_state_layers:
                # `idx + 1` matches deepseek_v4: the runner already converted
                # DFlash target_layer_ids to id+1 semantics
                # (gpu_model_runner._get_eagle3_aux_layers_from_config), so
                # this captures the OUTPUT of 0-based decoder layer `idx`.
                if post is not None:
                    # Mid-stack mHC layer: its final hc_post is deferred to
                    # the next layer's fused pre. Materialize the multi-stream
                    # reconstruction here (pure op -- the deferred
                    # residual/post/comb state is not mutated), then contract
                    # hc streams exactly like the last layer does.
                    # hc_contract == mean over streams, the same contraction
                    # deepseek_v4 uses (aux_recon.mean(dim=1)).
                    aux_recon = layer.hc_post(hidden_states, residual, post, comb)
                    aux_hidden_state = hc_contract(aux_recon, layer.n)
                else:
                    # Last mHC layer (already hc_post + hc_contract'ed inside
                    # the layer) or a non-mHC layer: the output is already
                    # plain [num_tokens, hidden_size].
                    aux_hidden_state = hidden_states
                if self.is_sequence_parallel:
                    # Aux states are consumed at full-sequence granularity;
                    # gather the SP shard (deepseek_v4 pattern).
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[
                        :full_num_tokens
                    ]
                aux_hidden_states.append(aux_hidden_state)
"""

EDIT_RETURN_ANCHOR = """\
        hidden_states = self.norm(hidden_states)
        return hidden_states
"""

EDIT_RETURN_NEW = """\
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            # (final_hidden_states, list-of-aux) -- gpu_model_runner unpacks
            # this tuple when use_aux_hidden_state_outputs is set; identical
            # to DeepseekV4Model.forward's aux return.
            return hidden_states, aux_hidden_states
        return hidden_states
"""

EDIT_CAUSAL_LM_ANCHOR = """\
class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
"""

EDIT_CAUSAL_LM_NEW = """\
class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, SupportsEagle3, MixtureOfExperts, IsHybrid
):
"""

EDIT_COND_GEN_ANCHOR = """\
class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid
):
"""

EDIT_COND_GEN_NEW = """\
class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, SupportsEagle3
):
"""

EDITS: list[tuple[str, str, str]] = [
    ("import EagleModelMixin + SupportsEagle3", EDIT_IMPORTS_ANCHOR, EDIT_IMPORTS_NEW),
    ("Glm5NextModel gains EagleModelMixin", EDIT_MODEL_CLASS_ANCHOR, EDIT_MODEL_CLASS_NEW),
    ("decoder loop: aux hidden state capture + mHC contraction", EDIT_LOOP_ANCHOR, EDIT_LOOP_NEW),
    ("forward tail: return (hidden_states, aux_hidden_states)", EDIT_RETURN_ANCHOR, EDIT_RETURN_NEW),
    ("Glm5NextForCausalLM declares SupportsEagle3", EDIT_CAUSAL_LM_ANCHOR, EDIT_CAUSAL_LM_NEW),
    ("Glm5NextForConditionalGeneration declares SupportsEagle3", EDIT_COND_GEN_ANCHOR, EDIT_COND_GEN_NEW),
]


def patch_file(path: str, dry_run: bool = False) -> int:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if MARKER in text:
        print(f"[patch_glm_aux_capture] {path}: already patched ({MARKER} marker found); no-op.")
        return 0

    # Sanity: the file we expect (guards against pointing at the wrong tree).
    for required in ("class Glm5NextModel", "hc_contract", "sp_all_gather"):
        assert required in text, (
            f"ANCHOR PRECHECK FAILED: {required!r} not found in {path} -- "
            "is this really glm5next/nvidia/model.py?"
        )

    applied = []
    for name, anchor, replacement in EDITS:
        n = text.count(anchor)
        assert n == 1, (
            f"ANCHOR FAILED for edit [{name}]: expected exactly 1 occurrence, "
            f"found {n}. The upstream file has drifted -- re-derive the anchor "
            f"before building.\n--- anchor ---\n{anchor}\n--------------"
        )
        text = text.replace(anchor, replacement, 1)
        applied.append(name)

    # The patched source must still be valid Python.
    try:
        ast.parse(text, filename=path)
    except SyntaxError as e:
        raise AssertionError(f"POST-EDIT ast.parse FAILED for {path}: {e}") from e

    if dry_run:
        print(f"[patch_glm_aux_capture] DRY RUN -- {path} not written.")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"[patch_glm_aux_capture] {path}: {len(applied)} edits applied:")
    for name in applied:
        print(f"  - {name}")
    print("[patch_glm_aux_capture] ast.parse OK.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    ap.add_argument("--dry-run", action="store_true", help="validate anchors + parse, write nothing")
    args = ap.parse_args()
    return patch_file(args.model_file, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
