# Runbook: knapcio's stack on this fleet (2026-09-29)

The speed stack is **[knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)**
(MIT; vLLM-derived files Apache-2.0), pinned at commit **`770d115`**. This repository does not copy
its code. It records how we run it on our four Sparks, our two lanes (500K default, 262K), and what we
measured. knapcio's stack is itself built on this repo's v11 image, RoCE port and prefix-cache repair
(see his `CREDITS.md`).

Everything below is what was actually run on 2026-09-29, in order. Times are for our fleet.

## 0. Before you start

- Four DGX Sparks on a switched RoCE fabric, our image
  `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` on every node.
- `nvidia/GLM-5.3-Flash-NVFP4` at knapcio's pinned revision (`config.json` sha256 starts `e23c5d98`)
  and `incoai/GLM-5.3-Flash-DFlash2` at revision `bf582e4` (`config.json` starts `c4aeac01`).
  Check yours: `sha256sum config.json`.
- Run `gputest.sh` (TP2 repo, `speed-night-2026-09-18/`) on every node first. Under 50 TFLOPS means
  a power-clamped GPU after an unclean reset; only an AC power cycle clears it.

## 1. Clone at the pinned commit, same path on every node

```bash
sudo mkdir -p /srv/glm53-knapcio && sudo chown $(id -u):$(id -g) /srv/glm53-knapcio
git clone https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4.git /srv/glm53-knapcio
git -C /srv/glm53-knapcio checkout 770d1153062aa916b06591411c61d5c593ea0f03
```

Not under `/var/tmp` on our fleet: his launcher refuses to start if **any** container, running or
stopped, bind-mounts a path that overlaps the overlay directory, and one of our old containers mounts
all of `/var/tmp`. Pick a path nothing else mounts.

## 2. Build the image on every node (about a minute)

```bash
cd /srv/glm53-knapcio
docker build --platform linux/arm64 \
  --build-arg BASE_IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 \
  --build-arg ROCE_HCA=rocep1s0f0 \
  -f Dockerfile.roce -t glm53-roce:v11-b58f34ea .
```

His Dockerfile pins our image by digest; only nodes that pulled it from GHCR carry the digest, so we
build from the local tag (same image, loaded rather than pulled).

## 3. Convert the weights (CPU only, about 50 seconds per node)

On every node that holds the base checkpoint (ours: Reddie and Bluey), with the serving engine stopped:

```bash
docker run --rm --network none --memory 48g --cpus 8 --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /srv/glm53-knapcio:/recipe:ro -v /var/tmp/models:/models --entrypoint bash \
  ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 -c '
    set -e
    bash /recipe/scripts/build_lossless8.sh /models/GLM-5.3-Flash-NVFP4-nvidia /models/glm-quant-mix
    python3 /recipe/scripts/drafter_fp8.py /models/GLM-5.3-Flash-DFlash2 /models/GLM-5.3-Flash-DFlash2-fp8blk'
sha256sum /var/tmp/models/glm-quant-mix/lossless8/config.json          # f14dc13ce3be...  (his accepted hash)
sha256sum /var/tmp/models/GLM-5.3-Flash-DFlash2-fp8blk/config.json     # 15bc842939ff...  (his accepted hash)
```

The target is assembled with hardlinks, so base and output must share a filesystem. Both of our
conversions matched his published config hashes exactly. Use `--rm` or remove the converter container
afterwards: a stopped container that mounts `/srv/glm53-knapcio` also trips his preflight.

## 4. Put the weights and drafter where every rank can read them

- Copy the FP8 drafter (2.2 GB) to every node at the same path.
- Nodes without room for a local copy read the converted weights over NFS through a symlink at the
  same path, e.g. on Spark4: `ln -sfn /mnt/reddie-models/glm-quant-mix /var/tmp/models/glm-quant-mix`
  (Asusi reads Bluey's copy the same way). The hardlinked files are ordinary files over NFS.

## 5. Launch a lane (from the head)

```bash
cp env.500k env.262k /srv/glm53-knapcio/          # from this folder; edit HOSTS/IPS/paths first
cd /srv/glm53-knapcio
ENV_FILE=env.500k bash start.sh serve              # the default; env.262k for the 262K lane
bash start.sh status                               # until health 200
```

- Cold first boot: **542 s** (every kernel JIT-compiles). Warm reboot: **167 to 235 s**.
- It serves `glm-5.3-flash` on `0.0.0.0:8000`, the same name and port as this repo's earlier recipes,
  so existing clients keep working. (His default is `GLM-5.3-Flash-FP8` on loopback `:8093`.)

To relaunch: `bash start.sh stop` stops **and preserves** the containers. His preflight then refuses the
same names and mounts, so save the logs and `docker rm -f` the stopped `glm53k-r*` containers on every
node before `serve` again.

## 6. Gotchas we hit (each one cost a boot)

- **Keep every draft length in the spec table.** His `SPEC_TABLE` ends in odd-looking entries
  (`[28,28,6],[29,29,4],[30,30,2],[31,32,1]`). They are there so the table contains every depth 1 to 7,
  which makes vLLM capture a FULL CUDA graph for every verify length 2 to 8. Device-side draft selection
  clones those graphs. We replaced the table with `[[1,1,7],[2,2,5],[3,8,5],[9,32,4]]`, the log said
  `DEVSELECT_BUILD_FAILED: no FULL 1-request graph for M=2`, and single-stream prose fell **48%**.
- **Profile knobs resolve when the profile is sourced.** `TRUNC_COST`, `DEVSELECT`, `PF3_ARM`,
  `GATHER_ROUTE` and `L2PF_V2` are consumed inside `profiles/current.env`; setting them after
  `source profiles/current.env` in the env file does nothing. Put them on the command line
  (`TRUNC_COST=r12 ENV_FILE=env.500k bash start.sh serve`). Plain variables (`MAX_MODEL_LEN`,
  `PORT`, `SERVED_NAME`, `HOST_BIND`) can be overridden after the `source` line.
- **Thinking cannot be switched off** in his chat template (`reasoning_effort` low, high or max). Our env files set `DEFAULT_EFFORT=low`.
  Send `"chat_template_kwargs": {"reasoning_effort": "low"}` for short answers.
- **Uncensored lane:** this stack runs on `nvidia/GLM-5.3-Flash-NVFP4` (censored). Whether his 8-bit
  conversion applies to the Blackfrost weights is untested; the uncensored lane stays on this repo's
  previous recipe until it is.

## 7. What we tried and did not keep

| change | result |
|---|---|
| deeper draft table, all depths kept (`[[1,1,7],[2,2,5],[3,8,5],[9,27,4],[28,28,6],[29,29,3],[30,30,2],[31,32,1]]`) | within ±3% of his table in every cell |
| scheduler `mode: batch-max` (live via `profiles/levers_policy.json`, no reboot) | within ±2% |

His draft policy is at a local optimum on our fleet. Not yet tried: the second RoCE rail (he runs two;
ours is addressed on two of four nodes), the clock cap off (he measured +4 to 8% prefill uncapped; we
keep the cap for the hard power-off mitigation), `TRUNC_COST=r12`.

## Files here

| path | what |
|---|---|
| `env.500k`, `env.262k` | the two lanes' env files for his `start.sh` |
| `bench/bench_tp2_night.py` | the speed-night harness (10 prompts, C1-C6 mixed, cold prefill, long context), with `--effort` |
| `bench/needle.py` | three needles at 10/50/90% depth of a salted document; cold prefill + retrieval |
| `bench/kvtest.py` | N concurrent long requests; samples KV usage and every node's free memory each second, aborts below 5 GiB (2026-10-02) |
| `results/` | every run's raw JSON |
| `charts/make_charts.py`, `charts/*.svg` | the README charts (no dependencies), light and dark |
