#!/usr/bin/env bash
set -uo pipefail
# GLM-5.3-Flash NVFP4 + DFlash2, TP4 across the four Sparks. Head = Reddie (192.168.192.2).
# Based on tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark launch-glm53-tp4-24g.sh, with:
#   1. per-rank model path: rank 0 reads /var/tmp/models locally, ranks 1-3 over NFS at
#      /mnt/reddie-models (Spark4 has 51 GB free and Asusi 42 GB, so local copies do not fit);
#   2. every lever as an env knob so one experiment = one env change;
#   3. the 2026-09-18 TP2 night's two shipped levers (b12x RoCE, #18 prefix-cache repair) on by
#      default, both already staged in ~/patches on all four nodes.
# Launch WORKER-FIRST: 3 -> 2 -> 1 -> head 0. Poll /health, never /v1/models.
NODE_RANK="${1:?usage: glm53_tp4.sh <0|1|2|3>}"

IMAGE="${IMAGE:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}"
NAME=vllm_glm53
MODEL_DIR="${MODEL_DIR:-keys-glm-5.3-flash-nvfp4-ablit-l15-45-anchorstock}"
MODEL_PATH="/models/glm-5.3-flash-nvfp4"
EXP_NAME="${EXP_NAME:-glm53tp4}"
CACHE_HOST_PATH="/var/tmp/glm53-vllm-cache"
HEAD_IP=192.168.192.2
MPORT=29521
PORT=8000
# knobs
GMU="${GMU:-0.85}"; MAXLEN="${MAXLEN:-1048576}"; SEQS="${SEQS:-64}"; MNBT="${MNBT:-16384}"
BLOCK="${BLOCK:-2304}"; MOE_BACKEND="${MOE_BACKEND:-marlin}"; SPEC_K="${SPEC_K:-7}"
KV_MEM="${KV_MEM:-25769803776}"; KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
CG="${CG:-FULL_AND_PIECEWISE}"     # or EAGER
# The multimodal warmup builds a dummy VIDEO sized to max_model_len. At 1M context that is an enormous
# synthetic video normalized single-threaded on CPU: the engine comes up fully, graphs and all, while /health
# still refuses connections for about eight minutes. video:0 skips it and keeps image support.
# See runs/2026-09-20-tp4-vs-deepseek/README.md. Override VLLM_EXTRA only if you add to this, not replace it.
VLLM_EXTRA="${VLLM_EXTRA:---limit-mm-per-prompt {\"image\":4,\"video\":0\}}"; NCCL_EXTRA="${NCCL_EXTRA:-}"
PATCH_HOME="${PATCH_HOME:-$HOME}"   # patches live in the node tonyspark home, not /root
SPEC="${SPEC:-dflash}"   # SPEC=none disables speculative decoding entirely
if [ "$SPEC" = "none" ]; then SPEC_ARGS=""; else
  SPEC_ARGS="--speculative-config {\"method\":\"dflash\",\"model\":\"/models/dflash2-draft\",\"num_speculative_tokens\":$SPEC_K}"
fi

case "$NODE_RANK" in
  0) HOST_IP=192.168.192.2; HEADLESS=""; MODEL_HOST_PATH="/var/tmp/models/$MODEL_DIR" ;;
  1) HOST_IP=192.168.192.4; HEADLESS="--headless"; MODEL_HOST_PATH="/mnt/reddie-models/$MODEL_DIR" ;;
  2) HOST_IP=192.168.192.3; HEADLESS="--headless"; MODEL_HOST_PATH="/mnt/reddie-models/$MODEL_DIR" ;;
  3) HOST_IP=192.168.192.1; HEADLESS="--headless"; MODEL_HOST_PATH="/mnt/reddie-models/$MODEL_DIR" ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac

# RoCEv2 GID index is not stable across link bounces; look it up.
GIDX=3
for _i in 0 1 2 3 4 5 6 7; do
  _t=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/$_i 2>/dev/null)
  _g=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/$_i 2>/dev/null)
  case "$_t" in *"RoCE v2"*) case "$_g" in *ffff*) GIDX=$_i; break ;; esac ;; esac
done

test -f "$MODEL_HOST_PATH/config.json" || { echo "MISSING $MODEL_HOST_PATH/config.json" >&2; exit 3; }
test -f "$MODEL_HOST_PATH/chat_template_mm.jinja" || { echo "MISSING chat_template_mm.jinja (vision 500s)" >&2; exit 3; }
test -f "$PATCH_HOME/patches/sparse_attn_indexer_kpool.py" || { echo "MISSING ~/patches/sparse_attn_indexer_kpool.py (dies past ~24K ctx)" >&2; exit 3; }
if [ -n "${NVFP4_PATCH:-}" ]; then
  for f in kda.py model.py; do
    test -f "$PATCH_HOME/patches/nvfp4/$f" || { echo "MISSING: \$PATCH_HOME/patches/nvfp4/$f -- NVFP4_PATCH=1 needs the patched glm5next files (they stop quant_config being forced to None for the attention projections). See runs/2026-09-20-tp4-vs-deepseek." >&2; exit 3; }
  done
  echo "NVFP4_PATCH on: attention projections will be built from the checkpoint quant config"
fi
test -f /var/tmp/models/GLM-5.3-Flash-DFlash2/config.json || { echo "MISSING drafter /var/tmp/models/GLM-5.3-Flash-DFlash2" >&2; exit 3; }
mkdir -p "$CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

# #18 prefix-cache repair (kv_cache_coordinator.py) and b12x RoCEnante one-shot all-reduce.
PREFIX_MOUNT=""
if [ "${PREFIX_FIX:-1}" = "1" ] && [ -f "$PATCH_HOME/patches/kv_cache_coordinator.py" ]; then
  PREFIX_MOUNT="-v $PATCH_HOME/patches/kv_cache_coordinator.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py:ro"
fi
ROCE_MOUNTS=""; ROCE_ENV=""
if [ "${ROCE:-1}" = "1" ] && [ -d "$PATCH_HOME/patches/glm-roce/b12x" ]; then
  R="$PATCH_HOME/patches/glm-roce"; D=/usr/local/lib/python3.12/dist-packages
  ROCE_MOUNTS="-v $R/b12x:$D/b12x:ro -v $R/b12x-1.3.0.dist-info:$D/b12x-1.3.0.dist-info:ro -v $R/b12x-roce:/opt/b12x-roce:ro
    -v $R/b12x_roce_all_reduce.py:$D/vllm/distributed/device_communicators/b12x_roce_all_reduce.py:ro
    -v $R/cuda_communicator.py:$D/vllm/distributed/device_communicators/cuda_communicator.py:ro
    -v $R/parallel_state.py:$D/vllm/distributed/parallel_state.py:ro
    -v $R/gpu_worker.py:$D/vllm/v1/worker/gpu_worker.py:ro
    -v $R/envs.py:$D/vllm/envs.py:ro"
  ROCE_ENV="-e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB -e VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB -e VLLM_ROCE_ALLGATHER_ENABLE=1 -e B12X_ROCE_HCA=rocep1s0f0 -e B12X_ROCE_GID_INDEX=$GIDX -e B12X_ROCE_SPIN_LIMIT=300000000 -e B12X_ROCE_CACHE_DIR=/opt/b12x-roce/cache"
fi
if [ "$CG" = "EAGER" ]; then GRAPH_ARGS="--enforce-eager"; else GRAPH_ARGS="--compilation-config {\"cudagraph_mode\":\"$CG\"}"; fi

# shellcheck disable=SC2086
docker run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  --oom-score-adj 500 \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -v $PATCH_HOME/patches/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro \
  ${NVFP4_PATCH:+-v $PATCH_HOME/patches/nvfp4/kda.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py:ro} \
  ${NVFP4_PATCH:+-v $PATCH_HOME/patches/nvfp4/model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py:ro} \
  -v /var/tmp/models/GLM-5.3-Flash-DFlash2:/models/dflash2-draft:ro \
  $PREFIX_MOUNT $ROCE_MOUNTS \
  -e VLLM_HOST_IP=$HOST_IP -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_CACHE_ROOT="/cache/vllm-$EXP_NAME" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e MAX_JOBS=2 \
  -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=$GIDX \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=192.168.192.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  $ROCE_ENV $NCCL_EXTRA \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAXLEN" \
    --max-num-seqs "$SEQS" --block-size "$BLOCK" --moe-backend "$MOE_BACKEND" \
    --max-num-batched-tokens "$MNBT" \
    $SPEC_ARGS \
    --kv-cache-dtype "$KV_DTYPE" --kv-cache-memory "$KV_MEM" \
    $GRAPH_ARGS \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 --chat-template $MODEL_PATH/chat_template_mm.jinja \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --distributed-executor-backend mp \
    --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS ${VLLM_EXTRA_EP:+--enable-expert-parallel} $VLLM_EXTRA
echo "launched $NAME rank=$NODE_RANK host=$HOST_IP model=$MODEL_DIR image=$IMAGE gmu=$GMU maxlen=$MAXLEN seqs=$SEQS mnbt=$MNBT block=$BLOCK moe=$MOE_BACKEND k=$SPEC_K spec=$SPEC kv=$KV_MEM cg=$CG roce=${ROCE:-1} prefixfix=${PREFIX_FIX:-1} gidx=$GIDX"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited" >&2; docker logs --tail 40 "$NAME" >&2; exit 1; }
