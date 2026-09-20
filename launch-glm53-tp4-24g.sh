#!/usr/bin/env bash
set -euo pipefail
# GLM-5.3-Flash-NVFP4 ABLITERATED, TP4 across four Sparks. Head = Reddie (192.168.192.2).
# EDIT FOR YOUR FABRIC: MODEL_HOST_PATH, the rank->IP map below, and the NCCL block
# (NCCL_IB_HCA, NCCL_IB_ADDR_RANGE, NCCL_SOCKET_IFNAME/GLOO/TP/MN). --memory 112g
# assumes 128 GB nodes.
# PREREQS on every node, or this boots and then fails in non-obvious ways:
#   $HOME/patches/sparse_attn_indexer_kpool.py   <- cp from docker/sparse_attn_indexer_kpool_sm121.py
#   /var/tmp/models/GLM-5.3-Flash-DFlash2/       <- the drafter weights
#   $MODEL_HOST_PATH/chat_template_mm.jinja      <- required for vision (mount is :ro)
# Vision ON (mm template), thinking OFF. Launch WORKER-FIRST: 3 -> 2 -> 1 -> head 0.
NODE_RANK="${1:?usage: launch-glm53-tp4-24g.sh <0|1|2|3>}"

IMAGE="ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2"
NAME="vllm_glm53"
# Checkpoint. The documented default is RedHatAI/GLM-5.3-Flash-NVFP4 (compressed-tensors)
# because the ModelOpt builds emit intermittent corrupted token IDs (vLLM #54150, README
# table above). This launcher previously hard-coded the abliterated ModelOpt path, so the
# shipped default did not match the documented one. Override with MODEL_HOST_PATH=... only
# for the legacy/abliterated builds, and set ALLOW_MODELOPT=1 to get past the guard below.
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/var/tmp/models/GLM-5.3-Flash-NVFP4-redhat}"
if [ -f "$MODEL_HOST_PATH/config.json" ] && [ "${ALLOW_MODELOPT:-0}" != "1" ]; then
  _q=$(python3 -c "import json;print(json.load(open('$MODEL_HOST_PATH/config.json')).get('quantization_config',{}).get('quant_method',''))" 2>/dev/null || echo "")
  if [ "$_q" = "modelopt" ]; then
    echo "REFUSING: $MODEL_HOST_PATH is a ModelOpt build (quant_method=modelopt)." >&2
    echo "  ModelOpt NVFP4 emits intermittent corrupted token IDs (vLLM #54150)." >&2
    echo "  Use RedHatAI/GLM-5.3-Flash-NVFP4, or set ALLOW_MODELOPT=1 to override." >&2
    exit 5
  fi
fi
MODEL_PATH="/models/glm-5.3-flash-nvfp4"
CACHE_HOST_PATH="/var/tmp/glm53-vllm-cache"
HEAD_IP="192.168.192.2"
MPORT="29521"
PORT="8000"

case "$NODE_RANK" in
  0) HOST_IP=192.168.192.2; HEADLESS="" ;;
  1) HOST_IP=192.168.192.4; HEADLESS="--headless" ;;
  2) HOST_IP=192.168.192.3; HEADLESS="--headless" ;;
  3) HOST_IP=192.168.192.1; HEADLESS="--headless" ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac

# Fail loudly on missing prereqs rather than letting Docker create empty dirs over them.
# The RoCEv2 GID index is not stable across link bounces and reboots, so look it up
# instead of trusting NCCL_IB_GID_INDEX below. A stale index fails at init with
# "unhandled system error" on one rank and nothing useful from the head; an unset one
# makes the collective hang silently. See docs/FIELD-NOTES-4NODE-100G.md.
GIDX=3
DETECTED_GIDX=""
for _i in 0 1 2 3 4 5 6 7; do
  _t=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/$_i 2>/dev/null)
  _g=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/$_i 2>/dev/null)
  case "$_t" in *"RoCE v2"*)
    case "$_g" in *ffff*) DETECTED_GIDX=$_i; break ;; esac ;;
  esac
done
[ -n "$DETECTED_GIDX" ] && GIDX=$DETECTED_GIDX
echo "using NCCL_IB_GID_INDEX=$GIDX"

test -f "$MODEL_HOST_PATH/config.json"       || { echo "MISSING: $MODEL_HOST_PATH/config.json" >&2; exit 3; }
test -f "$MODEL_HOST_PATH/chat_template_mm.jinja" || { echo "MISSING: chat_template_mm.jinja in the weights dir (vision will 500)" >&2; exit 3; }
test -f "$HOME/patches/sparse_attn_indexer_kpool.py" || { echo "MISSING: \$HOME/patches/sparse_attn_indexer_kpool.py -- cp it from docker/sparse_attn_indexer_kpool_sm121.py. Without it the engine dies on every decode past ~24K context." >&2; exit 3; }
test -f /var/tmp/models/GLM-5.3-Flash-DFlash2/config.json || { echo "MISSING: drafter weights at /var/tmp/models/GLM-5.3-Flash-DFlash2" >&2; exit 3; }
mkdir -p "$CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -e VLLM_HOST_IP=$HOST_IP \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=$GIDX \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=192.168.192.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -v $HOME/patches/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro \
  -v /var/tmp/models/GLM-5.3-Flash-DFlash2:/models/dflash2-draft:ro \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 1048576 \
    `# max-num-seqs 6 -> 64: the old value was the binding constraint on aggregate` \
    `# throughput, not the fabric. Aggregate was still climbing monotonically when it hit` \
    `# the cap. 6 -> 64 took best aggregate 183 -> 519 tok/s. Single stream is unaffected.` \
    `# See docs/SPEED-RUN-2026-08-31.md.` \
    --max-num-seqs 64 --block-size 2304 --moe-backend marlin \
    `# 8192 -> 16384: prefill +36-56% (114K prompt: 1194 -> 1863 tok/s, TTFT 95s -> 61s).` \
    `# Costs ~3% aggregate at C48. Worth it for agentic/long-prompt traffic; set back to` \
    `# 8192 if you only care about aggregate decode.` \
    --max-num-batched-tokens 16384 \
    --speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7}' \
    --kv-cache-dtype fp8_e4m3 --kv-cache-memory 25769803776 \
    `# CUDA graphs: use FULL_AND_PIECEWISE on this (fp8 KV + marlin) lane.` \
    `# CORRECTION 2026-09-02 -- an earlier revision of this file said "DO NOT REMOVE` \
    `# --enforce-eager" on the strength of a -19% measurement. That measurement was real` \
    `# but tested the WRONG MODE: plain PIECEWISE, which forces piecewise graphs onto the` \
    `# decode path. FULL_AND_PIECEWISE uses FULL for uniform decode batches and piecewise` \
    `# only for mixed/prefill, and measures FASTER than eager on all three prompt types:` \
    `#   count-to-100 101.6 -> 105.6   code 72.0 -> 77.3   prose 26.9 -> 31.5` \
    `#   best aggregate 503.3 -> 530.0 tok/s` \
    `# --enforce-eager is a property of the NVFP4-KV/b12x lane, NOT of this model: the b12x` \
    `# kernels require it, marlin does not. Keep eager ONLY on the b12x lane, and on the` \
    `# topkfix image (docs/TOPK-OVERSUSCRIPTION-FIX.md), which deadlocks under graphs.` \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 --chat-template /models/glm-5.3-flash-nvfp4/chat_template_mm.jinja \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --distributed-executor-backend mp \
    --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP tp4 kv=24GiB mnbt=16384 seqs=64 graphs=FULL_AND_PIECEWISE"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited; inspect with: docker logs $NAME" >&2; exit 1; }
