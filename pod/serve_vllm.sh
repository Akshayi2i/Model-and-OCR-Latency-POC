#!/usr/bin/env bash
# Bring up Qwen3-VL-8B and the MinerU service. Run from /workspace/POC3/pod.
#
#   export POC3_API_KEY=<the same key you put in config.yaml pod.api_key>
#   bash serve_vllm.sh
#
# RunPod template must expose HTTP ports 8002 and 8003. They become
#   https://<POD_ID>-8003.proxy.runpod.net   (Qwen3-VL-8B)
#   https://<POD_ID>-8002.proxy.runpod.net   (MinerU)
# Port 8001 is NOT usable: RunPod's own nginx holds it on this template.
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
# vLLM first, behind a health gate, then MinerU: vLLM sizes its KV cache from
# FREE VRAM at startup, so MinerU must not be loading at the same moment.
#
# 0.70 of 96 GB is ~67 GB for vLLM: ~17 GB of weights and ~45 GB of KV cache,
# room for several 131k-token sequences. The remaining ~29 GB is MinerU's
# (its pipeline models need well under 10 GB) plus CUDA overhead. Do not raise
# it "because it fits" - MinerU shares this GPU.
#
# --max-model-len 131072 must match models[].max_model_len in config.yaml. A
# window of chunking.window_pages pages in PDF+MD with the schema prompt is far
# below this; the headroom is for larger windows and long outputs. Qwen3-VL
# supports 256k natively, so no rope scaling is involved.
#
# --limit-mm-per-prompt image:32 must be >= chunking.window_pages and match
# models[].max_images. It bounds ONE request (one window), not the document.
#
# --enable-prompt-tokens-details makes vLLM report how many prompt tokens came
# from its prefix cache. Every window and every document of a LOB shares one
# system prompt (the schema block is most of it), so after the first request
# that prefill is largely free; the timing report shows how much.
# ---------------------------------------------------------------------------

echo "==> 1/2 Qwen3-VL-8B on :8003"
nohup vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl-8b \
  --host $HOST --port 8003 \
  --api-key "$POC3_API_KEY" \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --limit-mm-per-prompt '{"image": 32}' \
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 8 \
  --enable-prompt-tokens-details \
  > /workspace/logs/vllm_8b.log 2>&1 &
wait_ready 8003 "Qwen3-VL-8B"

echo "==> 2/2 MinerU on :8002"
cd "$HERE"
nohup uvicorn mineru_server:app --host $HOST --port 8002 --workers 1 \
  > /workspace/logs/mineru.log 2>&1 &
wait_ready 8002 "MinerU"

echo
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
echo
echo "both services are up. From the Windows box:"
echo "  python src/preflight.py --pod"
echo
echo "logs: /workspace/logs/{vllm_8b,mineru}.log"
echo
echo "NOTES"
echo "  If this vLLM build rejects --enable-prompt-tokens-details, drop the flag:"
echo "  the timing report then shows '-' for cached tokens and nothing else changes."
