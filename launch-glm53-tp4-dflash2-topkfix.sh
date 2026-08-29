#!/usr/bin/env bash
set -euo pipefail
# GLM-5.3-Flash-NVFP4, TP4, DFlash2 spec decode — with the exact torch.topk top-k fix
# baked into the image. This is the launcher for the sm121-v12-dflash2-topkfix image,
# which replaces the Disease-1 SM-count gate (sparse_attn_indexer_kpool_sm121.py
# bind-mount) with exact torch.topk routing — see docs/TOPK-OVERSUSCRIPTION-FIX.md.
#
# NOTE: do NOT bind-mount $HOME/patches/sparse_attn_indexer_kpool.py with this image —
# the torch.topk fix is baked in and a bind-mount would overwrite it.
#
# Head = Reddie (192.168.192.2). EDIT FOR YOUR FABRIC: MODEL_HOST_PATH, the rank->IP
# map, and the NCCL block. --memory 112g assumes 128 GB nodes.
# PREREQS on every node:
#   /var/tmp/models/GLM-5.3-Flash-DFlash2/   <- the drafter weights
#   $MODEL_HOST_PATH/chat_template_mm.jinja  <- required for vision (mount is :ro)
# Launch WORKER-FIRST: 3 -> 2 -> 1 -> head 0.
NODE_RANK="${1:?usage: launch-glm53-tp4-dflash2-topkfix.sh <0|1|2|3>}"

IMAGE="ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v12-dflash2-topkfix"
NAME="vllm_glm53"
MODEL_HOST_PATH="/var/tmp/models/keys-glm-5.3-flash-nvfp4-ablit-l15-45-anchorstock"
MODEL_PATH="/models/glm-5.3-flash-nvfp4"
CACHE_HOST_PATH="/var/tmp/glm53-vllm-cache"
HEAD_IP="192.168.192.2"
MPORT="29521"
PORT="8000"
SERVED_NAME="glm-5.3-flash"

case "$NODE_RANK" in
  0) HOST_IP=192.168.192.2; HEADLESS="" ;;
  1) HOST_IP=192.168.192.4; HEADLESS="--headless" ;;
  2) HOST_IP=192.168.192.3; HEADLESS="--headless" ;;
  3) HOST_IP=192.168.192.1; HEADLESS="--headless" ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac

test -f "$MODEL_HOST_PATH/config.json"       || { echo "MISSING: $MODEL_HOST_PATH/config.json" >&2; exit 3; }
test -f "$MODEL_HOST_PATH/chat_template_mm.jinja" || { echo "MISSING: chat_template_mm.jinja in the weights dir (vision will 500)" >&2; exit 3; }
test -f /var/tmp/models/GLM-5.3-Flash-DFlash2/config.json || { echo "MISSING: drafter weights at /var/tmp/models/GLM-5.3-Flash-DFlash2" >&2; exit 3; }
mkdir -p "$CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v /var/tmp/models/GLM-5.3-Flash-DFlash2:/models/dflash2-draft:ro \
  -v "$CACHE_HOST_PATH:/cache" \
  -e VLLM_HOST_IP=$HOST_IP \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=192.168.192.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.72 \
    --max-model-len 1048576 --max-num-seqs 16 --block-size 2304 \
    --moe-backend marlin \
    --speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7}' \
    --kv-cache-dtype fp8_e4m3 --kv-cache-memory 25769803776 \
    --enforce-eager \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 \
    --chat-template "$MODEL_PATH/chat_template_mm.jinja" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --distributed-executor-backend mp \
    --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP tp4-dflash2-topkfix port=$PORT name=$SERVED_NAME"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited; inspect with: docker logs $NAME" >&2
  exit 1
}
