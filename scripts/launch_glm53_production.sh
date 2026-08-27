#!/usr/bin/env bash
# GLM-5.3-Flash PRODUCTION LAUNCH (the "boot 8" formula, frozen 2026-08-27).
# Verified: c1 23.4 / c4 42.4 / c8 60.5 tok/s agg; decode-only 23.7 (community ruler).
# FP8 weights + FP8 KV + flashinfer_sparse_mla DSA (sm121 patch stack) + NEXTN adaptive
# MTP + decode graphs bs<=8 + chunked-prefill 2048 + gmu 0.80 (KV pool 1.23M tok).
# Full provenance of every flag: ../LEDGER.md. Edit NODES/SSH_HOSTS below for your fabric.
set -uo pipefail
NODES=(192.168.100.11 192.168.100.12 192.168.100.13 192.168.100.14)
SSH_HOSTS=(gx10-1 gx10-2 gx10-3 gx10-4)
# Opt step 5 (ladder step 2): 0xSero patch stack rebuilt for sm121 -- unlocks
# flashinfer_sparse_mla DSA + FP8 KV for GLM on major-12 devices. FP8 weights unchanged.
IMAGE="glm53-sglang-sm121:20260827"
NAME="glm53_sglang"
PORT=8213
DIST_PORT=29653
WEIGHTS_DIR="/var/tmp/glm-legacy/hf"
MODEL_DIR="/cache/huggingface/hub/glm53-flash-fp8"
HEAD="${NODES[0]}"

ENVV=(
  -e "HF_HOME=/cache/huggingface" -e "HF_HUB_OFFLINE=1"
  -e "NCCL_NET=IB" -e "NCCL_IB_DISABLE=0"
  -e "NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0"
  -e "NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0"
  -e "GLOO_SOCKET_IFNAME=enp1s0f0np0"
  -e "NCCL_IB_GID_INDEX=3"
  -e "NCCL_MAX_NCHANNELS=4" -e "NCCL_MIN_NCHANNELS=4"
  -e "NCCL_CROSS_NIC=1" -e "NCCL_CUMEM_ENABLE=0"
  -e "NCCL_IGNORE_CPU_AFFINITY=1" -e "NCCL_DEBUG=WARN"
  -e "PYTHONUNBUFFERED=1"
  # GB10 JIT pitfalls from the fleet kit (SGLang on GB10, Ornith era)
  -e "MAX_JOBS=4" -e "NVCC_THREADS=1" -e "FLASHINFER_NVCC_THREADS=1"
)
BASE=(
  --cap-add IPC_LOCK --ulimit memlock=-1:-1
  --ulimit nofile=1048576:1048576
  --network host --ipc host --shm-size 10gb --gpus all
  --entrypoint python3
  --device /dev/infiniband:/dev/infiniband
  -v "$WEIGHTS_DIR:/cache/huggingface"
  # persist FlashInfer/tilelang JIT cache across relaunches
  -v "/var/tmp/glm53-jit-cache:/root/.cache"
  # GB10 sm121 caps dynamic smem ~99KB/block; stock DSA tilelang kernel double-buffers
  # a 64x512 bf16 KV tile (148KB request). Overlay sets num_stages=1 (~84KB, fits).
  -v "/var/tmp/glm53-overlays/tilelang_kernel.py:/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py:ro"
)
SERVE=(
  -m sglang.launch_server
  --model-path "$MODEL_DIR"
  --served-model-name glm-5.3-flash
  --host 0.0.0.0 --port "$PORT"
  --trust-remote-code
  --tp-size 4 --nnodes 4
  --dist-init-addr "$HEAD:$DIST_PORT"
  --kv-cache-dtype fp8_e4m3
  --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla
  # Opt step 1: decode CUDA graphs ON (prefill graphs auto-disable for KDA hybrid;
  # sglang #36550 replay bug only bites cold prefill >262K, our ctx is 32K).
  --reasoning-parser glm45
  # Opt step 2: native MTP head, cookbook low-latency recipe (NEXTN 5/1/6).
  --speculative-algorithm NEXTN
  --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
  # Opt step 3 (flags mined from 0xSero/glm-5.3-flash-sglang-sm120): adaptive draft
  # depth, GB10-native CUTLASS MoE runner (matches fleet-kit note), model-card tool parser.
  --speculative-adaptive
  # flashinfer_cutlass MoE runner CRASHES with Fp8MoEMethod in this build
  # (AttributeError 'runner', boot 3 2026-08-27); it is an NVFP4-lane lever only.
  --tool-call-parser glm47
  --chunked-prefill-size 2048
  --context-length 32768
  --max-running-requests 8
  --mem-fraction-static 0.80
)
docker_run_cmd() {
  local rank="$1"
  local cmd=(docker run -d --name "$NAME" "${BASE[@]}" "${ENVV[@]}"
             "$IMAGE" "${SERVE[@]}" --node-rank "$rank")
  local out="" t; for t in "${cmd[@]}"; do out+=" $(printf '%q' "$t")"; done; echo "$out"
}
if [ "${DRY_RUN:-0}" = "1" ]; then docker_run_cmd 0; exit 0; fi
NEED_GB=95
for h in "${SSH_HOSTS[@]}"; do
  ssh -o BatchMode=yes "$h" bash -s <<EOF
set -uo pipefail
docker rm -f $NAME glm53_slot >/dev/null 2>&1 || true
for pid in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  sudo -n kill -9 "\$pid" 2>/dev/null || true
done
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
avail=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
if [ "\$avail" -lt "$((NEED_GB*1024*1024))" ]; then
  echo "[preflight \$(hostname -s)] ABORT: \$((avail/1048576))G < ${NEED_GB}G"; exit 1
fi
echo "[preflight \$(hostname -s)] OK \$((avail/1048576))G"
EOF
  [ $? -ne 0 ] && { echo "PREFLIGHT FAILED on $h -- abort"; exit 1; }
done
for r in 3 2 1 0; do ssh -o BatchMode=yes "${SSH_HOSTS[$r]}" "$(docker_run_cmd $r)"; sleep 5; done
echo "launched; ready when: curl -s http://$HEAD:$PORT/v1/models"
