#!/usr/bin/env bash
# Bring up both models and the MinerU service. Run from /workspace/POC3/pod.
#
#   export POC3_API_KEY=<the same key you put in config.yaml pod.api_key>
#   bash serve_vllm.sh
#
# RunPod template must expose HTTP ports 8000, 8001 and 8002. They become
#   https://<POD_ID>-8000.proxy.runpod.net   (Qwen2.5-VL-7B)
#   https://<POD_ID>-8001.proxy.runpod.net   (Qwen3-VL-8B)
#   https://<POD_ID>-8002.proxy.runpod.net   (MinerU)
set -euo pipefail

: "${POC3_API_KEY:?set POC3_API_KEY - the RunPod proxy is public}"
export HF_HOME=${HF_HOME:-/workspace/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /workspace/logs

# --host 0.0.0.0 is not optional: binding 127.0.0.1 gives a proxy 502 that looks
# exactly like "the model is still loading", and you can lose an hour to it.
HOST=0.0.0.0

wait_ready () {  # port name
  local port=$1 name=$2
  echo "    waiting for $name on :$port ..."
  for _ in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "    $name ready on :$port"; return 0
    fi
    sleep 5
  done
  echo "    $name never became ready on :$port - see /workspace/logs"
  exit 1
}

# ---------------------------------------------------------------------------
# Sequential, health-gated startup. This is the single most likely bring-up
# failure if you skip it: both servers profile FREE VRAM at startup to size their
# KV cache, so launched together each one counts the other's not-yet-allocated
# memory as free and they collectively over-commit. Waiting costs a few minutes
# and removes the failure entirely.
#
# 0.42 each of 96 GB is ~40 GB apiece, ~80 GB total, leaving ~14 GB for MinerU's
# pipeline models and CUDA graph overhead. Do not raise it "because it fits" -
# MinerU shares this GPU.
#
# --max-model-len 98304 must match models[].max_model_len in config.yaml. It is
# well above the ~40k a PDF+MD request needs, because MinerU markdown size is not
# known until it runs and a context overflow is discovered only after the images
# have already been uploaded. Both models support it natively (Qwen2.5-VL 128k,
# Qwen3-VL 256k) so no rope scaling is involved, and at 0.42 utilisation the KV
# cache has room for many times this length at --max-num-seqs 4.
# ---------------------------------------------------------------------------

echo "==> 1/3 Qwen2.5-VL-7B on :8000"
nohup vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --served-model-name qwen2.5-vl-7b \
  --host $HOST --port 8000 \
  --api-key "$POC3_API_KEY" \
  --dtype bfloat16 \
  --max-model-len 98304 \
  --limit-mm-per-prompt '{"image": 24}' \
  --gpu-memory-utilization 0.42 \
  --max-num-seqs 4 \
  --swap-space 8 \
  > /workspace/logs/vllm_7b.log 2>&1 &
wait_ready 8000 "Qwen2.5-VL-7B"

echo "==> 2/3 Qwen3-VL-8B on :8001"
nohup vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl-8b \
  --host $HOST --port 8001 \
  --api-key "$POC3_API_KEY" \
  --dtype bfloat16 \
  --max-model-len 98304 \
  --limit-mm-per-prompt '{"image": 24}' \
  --gpu-memory-utilization 0.42 \
  --max-num-seqs 4 \
  --swap-space 8 \
  > /workspace/logs/vllm_8b.log 2>&1 &
wait_ready 8001 "Qwen3-VL-8B"

echo "==> 3/3 MinerU on :8002"
cd "$HERE"
nohup uvicorn mineru_server:app --host $HOST --port 8002 --workers 1 \
  > /workspace/logs/mineru.log 2>&1 &
wait_ready 8002 "MinerU"

echo
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
echo
echo "all three services are up. From the Windows box:"
echo "  python src/preflight.py --pod"
echo
echo "logs: /workspace/logs/{vllm_7b,vllm_8b,mineru}.log"
echo
echo "NOTES"
echo "  --limit-mm-per-prompt must be >= the PDF's page count or vLLM 400s the"
echo "  request. It is 24 here; raise it AND models[].max_images in config.yaml"
echo "  together if your document has more pages."
echo
echo "  If 8B rejects --mm-processor-kwargs or a log flag on your vLLM build,"
echo "  drop the flag rather than pinning a version for it. The client-side"
echo "  render.max_long_side_px cap already bounds the image token budget."
