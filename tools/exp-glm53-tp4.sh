#!/bin/bash
# tools/exp-glm53-tp4.sh: the SWEEP HARNESS for this recipe, not the serving launcher.
# Same image, weights, ranks, KV pin, block size and drafter as launch-glm53-tp4-24g.sh, with
# every tunable exposed as an env var so an experiment can flip one knob at a time (EAGER,
# CUDAGRAPH_MODE, MAX_NUM_SEQS, MAX_BATCHED, SPEC_JSON, EXP_NAME picks a separate compile cache).
# Defaults equal the current recipe, so a bare run reproduces launch-glm53-tp4-24g.sh.
# Site-specific: workers read weights over NFS from the head at /mnt/reddie-models; edit for yours.
# For serving, use launch-glm53-tp4-24g.sh (fleet_watchdog.sh hard-codes that name).
# exp-glm53-tp4.sh <rank>  — experiment launcher for GLM-5.3-Flash TP4.
#
# Byte-identical to ~/revert-glm53-redhat-tp4.sh (the known-good recipe) EXCEPT that
# the levers under test are read from the environment. Unset => baseline behaviour,
# so an unparameterised run reproduces the known-good config exactly.
#
# Levers (export before running, same values on ALL FOUR nodes):
#   EXP_NAME        label, also picks the per-experiment compile cache dir.
#                   REQUIRED for k sweeps: vLLM's SpeculativeConfig.compute_hash()
#                   omits num_speculative_tokens (vllm#53366), so sweeping k in one
#                   cache dir silently reuses the wrong Inductor artifact.
#   SPEC_JSON       full --speculative-config JSON. default: baseline dflash k=7
#   EAGER           1 => --enforce-eager (baseline). 0 => use CUDAGRAPH_MODE
#   CUDAGRAPH_MODE  PIECEWISE | FULL_DECODE_ONLY | FULL_AND_PIECEWISE | FULL
#                   NOTE: 34 KDA layers are UNIFORM_SINGLE_TOKEN_DECODE, so FULL is
#                   auto-downgraded and cannot graph a 1+K verify batch. PIECEWISE
#                   is the mode to try first.
#   MAX_NUM_SEQS    default 64 (the current recipe)
#   MAX_BATCHED     default 16384 (the current recipe)
#   NCCL_EXTRA      extra "-e K=V" docker env pairs, e.g. "-e NCCL_ALGO=Tree -e NCCL_PROTO=LL"
#   VLLM_EXTRA      extra vllm serve args
set -euo pipefail
NODE_RANK="${1:?usage: exp-glm53-tp4.sh <0|1|2|3>}"

EXP_NAME="${EXP_NAME:-baseline}"
EAGER="${EAGER:-0}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_BATCHED="${MAX_BATCHED:-16384}"
NCCL_EXTRA="${NCCL_EXTRA:-}"
VLLM_EXTRA="${VLLM_EXTRA:-}"
# NOTE: do NOT write this as "${SPEC_JSON:-{...}}". Bash matches the first unescaped `}`
# as the end of the parameter expansion, so the trailing brace leaks out as a literal and
# is appended to whatever the caller passed -- producing `...true}}` and a hard argparse
# failure on every rank. Plain conditional assignment instead.
if [ -z "${SPEC_JSON:-}" ]; then
  SPEC_JSON='{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7}'
fi

IMAGE="${IMAGE:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}"
NAME="vllm_glm53"
MODEL_DIR="GLM-5.3-Flash-NVFP4-redhat"
CACHE_HOST_PATH="/var/tmp/glm53-vllm-cache"
HEAD_IP="192.168.192.2"; MPORT="29521"; PORT="8000"

case "$NODE_RANK" in
  0) HOST_IP=192.168.192.2; HEADLESS="";           MODEL_HOST="/var/tmp/models/$MODEL_DIR" ;;
  1) HOST_IP=192.168.192.4; HEADLESS="--headless"; MODEL_HOST="/mnt/reddie-models/$MODEL_DIR" ;;
  2) HOST_IP=192.168.192.3; HEADLESS="--headless"; MODEL_HOST="/mnt/reddie-models/$MODEL_DIR" ;;
  3) HOST_IP=192.168.192.1; HEADLESS="--headless"; MODEL_HOST="/mnt/reddie-models/$MODEL_DIR" ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac

test -f "$MODEL_HOST/config.json" || { echo "MODEL MISSING at $MODEL_HOST" >&2; exit 3; }
mkdir -p "$CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true

# graph mode vs eager
if [ "$EAGER" = "1" ]; then
  GRAPH_ARGS="--enforce-eager"
else
  GRAPH_ARGS="--compilation-config {\"cudagraph_mode\":\"$CUDAGRAPH_MODE\"}"
fi

# shellcheck disable=SC2086
# SKIP_INDEXER_MOUNT=1 for images with the PR#3 topk fix baked in: mounting the
# SM-count-gated file over a baked fix silently reverts the thing under test.
if [ "${SKIP_INDEXER_MOUNT:-0}" = "1" ]; then
  INDEXER_MOUNT=""
else
  INDEXER_MOUNT="-v $HOME/patches/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro"
fi

docker run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST:/models/glm-5.3-flash-nvfp4:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -v /var/tmp/models/GLM-5.3-Flash-DFlash2:/models/dflash2-draft:ro \
  ${INDEXER_MOUNT} \
  -e VLLM_HOST_IP=$HOST_IP -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_CACHE_ROOT="/cache/vllm-$EXP_NAME" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=192.168.192.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  $NCCL_EXTRA \
  "$IMAGE" \
    /models/glm-5.3-flash-nvfp4 \
    --served-model-name glm-5.3-flash --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --tensor-parallel-size 4 --gpu-memory-utilization 0.85 --max-model-len 1048576 \
    --max-num-seqs "$MAX_NUM_SEQS" --block-size 2304 --moe-backend marlin \
    --speculative-config "$SPEC_JSON" \
    --kv-cache-dtype fp8_e4m3 --kv-cache-memory 25769803776 \
    $GRAPH_ARGS \
    --max-num-batched-tokens "$MAX_BATCHED" --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 --chat-template /models/glm-5.3-flash-nvfp4/chat_template_mm.jinja \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --distributed-executor-backend mp --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" $HEADLESS $VLLM_EXTRA

echo "launched $NAME rank=$NODE_RANK exp=$EXP_NAME eager=$EAGER graph=$CUDAGRAPH_MODE seqs=$MAX_NUM_SEQS"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited" >&2; exit 1; }
