# GLM-5.3-Flash target-side glue for DFlash2 (aux hidden state capture)

`patch_glm_aux_capture.py` is a build-time script run INSIDE
`radixark/vllm-glm53-flash:sm121-v8` (or a derived image). It edits
`/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py`
in place with six anchored replacements, asserts every anchor matches exactly
once, `ast.parse`s the result, and is idempotent (marker: `DFLASH2-AUX-CAPTURE`).

Dockerfile usage:

```dockerfile
COPY patch_glm_aux_capture.py /opt/dflash2/
RUN python3 /opt/dflash2/patch_glm_aux_capture.py
```

Everything mirrors the in-image prior art:
`vllm/models/deepseek_v4/nvidia/model.py`, `DeepseekV4Model.forward`
(aux capture block ~lines 1160-1213) and `DeepseekV4ForCausalLM`'s
`SupportsEagle3` declaration (~line 1457).

## What the patch does

1. **Imports** `EagleModelMixin` and `SupportsEagle3` from
   `vllm.model_executor.models.interfaces`.
2. **`Glm5NextModel(nn.Module, EagleModelMixin)`** — the mixin provides the
   `aux_hidden_state_layers: tuple[int, ...] = ()` store and
   `_set_aux_hidden_state_layers()`. Default `()` means zero behavior change
   when no drafter is configured (the capture branch never fires, forward
   returns a plain tensor as before).
3. **`Glm5NextForCausalLM` + `Glm5NextForConditionalGeneration` declare
   `SupportsEagle3`.** The interface's concrete default methods do the
   routing: `set_aux_hidden_state_layers` looks for `.language_model` (the
   multimodal wrapper has one → lands on Glm5NextForCausalLM) then requires
   `.model` to be an `EagleModelMixin` (Glm5NextModel, after this patch).
   Both entry points work because the image serves
   `Glm5NextForConditionalGeneration`, whose inherited
   `Glm4vForConditionalGeneration.forward` calls `self.language_model.model(...)`
   and returns the result untouched — the `(hidden, aux)` tuple passes
   straight through to the runner. Same for `Glm5NextForCausalLM.forward`.
4. **Capture in the decoder loop** (see below).
5. **Return** `(hidden_states, aux_hidden_states)` from
   `Glm5NextModel.forward` when any layer was captured — the exact
   deepseek_v4 convention that `gpu_model_runner` unpacks when
   `use_aux_hidden_state_outputs` is set.

## How the capture works (and the mHC wrinkle)

GLM-5.3-Flash runs mHC (`mhc: true`, `hc_mult: 4`): between decoder layers the
hidden state is multi-stream, `[num_tokens, 4, hidden]` — and, unlike
deepseek_v4, each GLM layer **defers its final `hc_post`** to the next layer's
fused post+pre kernel. Mid-loop, the loop variables are
`(hidden_states, residual, post, comb)` where the true post-layer hidden state
has not been materialized yet.

The capture, per layer `idx` with `idx + 1 in self.aux_hidden_state_layers`:

- **`post is not None` (mid-stack mHC layer):**
  `aux_recon = layer.hc_post(hidden_states, residual, post, comb)` materializes
  the deferred reconstruction (`MHCPostOp` → `mhc_post_tilelang`, a pure op:
  the deferred loop state is not mutated, so the ongoing forward is
  bit-identical to unpatched). Then
  `aux_hidden_state = hc_contract(aux_recon, layer.n)`.
- **`post is None`:** the layer already produced a plain
  `[num_tokens, hidden]` tensor — the LAST mHC layer does its own
  `hc_post` + `hc_contract` inside `Glm5NextDecoderLayer.forward`, and non-mHC
  / MTP layers never expand. Use `hidden_states` directly. (With
  `target_layer_ids=[5,14,24,33,42]` on 45 layers this branch is never hit,
  but it keeps the code correct for arbitrary layer sets.)

### Contraction choice

`hc_contract(x, n)` **is** `x.mean(dim=1)` (see
`vllm/model_executor/layers/mhc.py` ~line 561) — literally the same
contraction deepseek_v4 uses for its aux states (`aux_recon.mean(dim=1)`,
line 1179). So the aux hidden state handed to the drafter is exactly "what
the target model itself would consider the plain hidden state after layer
idx": `hc_post` recombines layer output with the 4 residual streams, mean
contracts the streams. Capturing after the final layer would reproduce the
model's own pre-norm final hidden state, by construction. Output shape:
`[num_tokens, 4096]` per tapped layer; 5 taps concatenate to
`[num_tokens, 20480]` in the DFlash speculator
(`torch.cat(aux_hidden_states, dim=-1)` → `model.combine_hidden_states`,
`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` ~line 341).

## The +1 indexing decision

`gpu_model_runner._get_eagle3_aux_layers_from_config` (~line 5645-5657)
converts DFlash draft-config `target_layer_ids` with an explicit
`i + 1` ("Add 1 to convert DFlash's aux layer id semantics") before calling
`set_aux_hidden_state_layers`. deepseek_v4's loop then tests
`if idx + 1 in self.aux_hidden_state_layers` where `idx` is the 0-based layer
index — i.e. an aux id of `k` means "the OUTPUT of 0-based layer `k-1`".

The patch copies that test verbatim (`idx + 1 in self.aux_hidden_state_layers`
with `enumerate(self._active_layers, start=self.start_layer)`). Net:
`target_layer_ids [5,14,24,33,42]` → runner sets `{6,15,25,34,43}` → we
capture the outputs of 0-based layers 5,14,24,33,42. Consistent end-to-end
with how the DFlash2 drafter was trained against deepseek_v4-style capture.
Do NOT re-apply +1 anywhere else (launcher config keeps raw
`target_layer_ids`; the runner owns the conversion).

## SP / all-gather (TP2) considerations

- Plain TP (TP2, no sequence-parallel MoE): every rank runs the full token
  batch; layer outputs and the mHC state are replicated full-size. No
  collectives needed in the capture path; each rank builds identical aux
  tensors. `layer.hc_post` inputs are the same tensors the next layer's fused
  kernel would consume, so there is no TP sharding to undo. The hc_fn/scale
  parameters are not TP-sharded (fp32 per-layer params).
- Sequence-parallel MoE mode (`use_sequence_parallel_moe`): mid-loop tensors
  are SP-sharded over tokens. The capture contracts on the shard, then
  `sp_all_gather(aux_hidden_state)[:full_num_tokens]` — exactly deepseek_v4
  line 1181. Gathering the contracted `[shard, 4096]` (not the `[shard, 4,
  4096]` recon) keeps the collective 4x smaller.
- PP is gated off for GLM5Next (no `make_empty_intermediate_tensors`), so the
  non-last-rank IntermediateTensors path never runs; the aux list would simply
  be dropped there, same as deepseek_v4.
- CUDA graphs: the capture adds tensor ops only (one `mhc_post` + mean + an
  optional gather per tapped layer per forward) — same op classes deepseek_v4
  traces today, so piecewise capture is unaffected.

Cost note: 5 extra `mhc_post_tilelang` calls + 5 means per forward. That is
the same overhead deepseek_v4 pays; negligible against the layer stack.

## Runtime validation — acceptance telemetry, not crashes

**A wrong contraction (or off-by-one layer tap) does NOT crash.** Shapes stay
`[num_tokens, 4096]`, the drafter happily consumes garbage, and the only
symptom is acceptance length collapsing — decode "works" but speedup vanishes.
Validate with telemetry, in this order:

1. **Static wiring check (before serving):** in a throwaway container, apply
   the patch and confirm import + interfaces (the patch's own verification):
   `Glm5NextModel` has `EagleModelMixin` in its MRO, both top classes have
   `supports_eagle3 == True`, `aux_hidden_state_layers` default is `()`.
2. **Startup log:** the runner logs
   `Using auxiliary layers from speculative config: (6, 15, 25, 34, 43)`.
   If you instead see the model-default `(2, n//2, n-3)` line, the draft
   config's `dflash_config.target_layer_ids` didn't reach the runner.
3. **Acceptance-length telemetry:** run a real prompt batch and read vLLM's
   spec-decode metrics (Prometheus `vllm:spec_decode_num_accepted_tokens*` /
   the periodic "SpecDecoding metrics" log line: draft acceptance rate,
   per-position acceptance, mean acceptance length).
   - **Good:** mean acceptance length ~4-5.8 (upstream DFlash2 reports this
     range on GLM-class targets); per-position acceptance decaying gently.
   - **Broken glue:** acceptance length pinned near 1.0-1.5 and first-position
     acceptance under ~50% → the drafter is seeing wrong features. Check, in
     order: layer-id off-by-one (compare against capturing raw ids without the
     runner's +1 — i.e. don't "fix" the +1 twice), contraction (mean vs sum vs
     single-stream), missing `hc_post` reconstruction (feeding the deferred
     `x` without recombining residual streams), and SP gather (token
     misalignment shows as acceptance collapsing only at batch sizes where SP
     sharding kicks in).
4. **Greedy sanity:** temperature 0 with and without the drafter must produce
   identical text (spec decode is lossless by construction — target verifies).
   If outputs differ, the bug is NOT this capture glue; look at the
   sampler/verifier side.
5. **Baseline invariance:** with no speculative config, patched vs unpatched
   image must be bit-identical (capture branch is dead; forward returns a
   plain tensor). A quick logprob diff on a fixed prompt confirms.

## Anchors (fail-loud contract)

Each anchor must appear exactly once or the script AssertionErrors with the
edit name — anchors chosen against `sm121-v8`:

| Edit | Anchor |
|---|---|
| imports | the 4-name `from vllm.model_executor.models.interfaces import (...)` block |
| model class | `class Glm5NextModel(nn.Module):` |
| decoder loop | the 4-line `for layer in self._active_layers:` loop body |
| forward tail | `hidden_states = self.norm(hidden_states)\n        return hidden_states` |
| causal LM bases | the 2-line `class Glm5NextForCausalLM(...)` header |
| cond-gen bases | the 2-line `class Glm5NextForConditionalGeneration(...)` header |

If a future image drifts (reformat, renamed loop var), the build fails at
patch time — never silently at runtime.
