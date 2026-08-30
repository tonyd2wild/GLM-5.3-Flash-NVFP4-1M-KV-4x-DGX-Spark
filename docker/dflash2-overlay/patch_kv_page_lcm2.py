"""SUPERSEDED no-op -- kept only so existing Dockerfile RUN chains still pass.

The candidate-search page unification that used to live here patched the
GENERIC uniform-page path (`unify_kv_cache_spec_page_size`) to let the DFlash2
drafter's page shrink onto the target's. That path is no longer reachable for
GLM-5.3-Flash + DFlash2: `patch_glm5_drafter_group.py` teaches the GLM-5-Next
fast path (`_get_kv_cache_groups_glm5_next`) to partition the drafter's
SlidingWindowSpec layers into their own KV cache group that slot-shares the
MLA tensors, so the model never falls through to the generic path at all.
And for this model the generic path is unservable regardless (any uniform
page rescales the kpool tail's block away from its pool size and warmup dies
at `assert tail_kv_cache.shape[2] == pool_size`), so the search here was dead
weight that also perturbed page unification for every other model. Removed
2026-08-28.  # SM121-PORT
"""

print("kv_page_candidate: superseded by patch_glm5_drafter_group.py; no-op")
