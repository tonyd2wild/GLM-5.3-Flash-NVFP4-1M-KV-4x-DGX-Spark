# Tools

Everything used to produce this run. Scripts run as root on the head node.

## Measuring

| script | what it does |
|---|---|
| `v41bench.py` | the benchmark. Nine prompt categories across concurrency levels, plus cold-prefill cells. Produces the JSONs in `../results/`. Takes `--model`, so it measures either lane |
| `sr_quality.py` | the quality gate: counting, JSON, code, arithmetic, prose. Honours `BENCH_MODEL` |
| `idletest.py` | short idle and back-to-back probes; reports tok/s, accepted tokens per step and **step ms**, which is what the cost model is fitted on. Honours `BENCH_MODEL` |
| `v41needle.py` | long-context retrieval. `--targets 65536,131072 --depth 0.3`. Honours `BENCH_MODEL` |
| `probe_prefill.sh` | a single unique long prompt, cold every time |
| `specacc.sh` | cumulative draft acceptance from `/metrics`, including per-position survival |
| `glm_screen3.sh` / `ds_screen3.sh` | the 3-rep median protocol on each lane. One discarded warm-up, three measured passes, per-cell medians plus max/min spread |
| `glm_hiconc.sh` / `glm_hiconc2.sh` / `ds_hiconc.sh` | concurrency sweeps: C8/C12/C16, then C24/C32 |
| `glm_fullsweep.sh` | the whole characterisation in one job: gate, idle, C1-C6 with prefill, then C8 through C32 |
| `mktable.py`, `final.py`, `agg3.py` | build the comparison tables from the JSONs |

**Single passes on these lanes carry a plus-or-minus 25% band.** Two boots of an identical config differed by
23% on code. Use the 3-rep median scripts, not one pass, or you will measure noise. Only **rep 1's** prefill
cells are cold: this build has no prefix-cache reset endpoint, so later reps hit the cache.

## Building the NVFP4 checkpoint

| script | what it does |
|---|---|
| `dense.py` | sums a checkpoint by module class and dtype. This is what found the 18.01 GiB of bf16 |
| `shardmap.py` | maps target tensors to shards and prices the rewrite |
| `mknvfp4b.py` | quantizes the non-expert projections to NVFP4 in **fused groups** with one shared global scale per group. Refuses to write if any fused group is incomplete |
| `build4.py` | assembles the model dir: hardlink clean shards, rewrite the ones holding superseded tensors, drop in the new shard, write index and config, then verify no duplicate names, nothing missing, nothing mis-pointed |
| `mkpatch.py` | regenerates `../patches/{kda.py,model.py}` from an image. Run this if you are on a different image digest |
| `mkfp8.py` | the fp8 variant. Measured and priced, then not booted once the arithmetic showed it capped below parity |

See [`../REPRODUCE.md`](../REPRODUCE.md) for the order and the trap that costs two boots if you skip the rewrite.

## Serving and swapping

`glm53_tp4.sh` is the launcher, every lever an env knob. `glm_boot.sh` does worker-first 3->2->1->0 and polls
`/health`, never `/v1/models`, which returns 200 from config with a dead engine behind it. `ds_handback.sh` and
`glm_takeover.sh` swap between the two models **and wait for the fleet to reclaim the outgoing lane's ~100 GiB
before the incoming workers allocate**, which is not optional: skipping it produced a refused boot here.

## Credit

`v41bench.py`, `sr_quality.py`, `idletest.py`, `v41needle.py` and `probe_prefill.sh` come from the
DeepSeek-V4.1-Flash work in the sibling repo and were parameterised with `BENCH_MODEL` so one harness measures
both models on identical prompts.
