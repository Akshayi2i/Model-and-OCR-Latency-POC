#!/usr/bin/env bash
# One-time pod setup. Run from /workspace/POC3/pod.
#
#   export POC3_API_KEY=<something long and random>
#   bash setup_pod.sh
#
# Order matters here: the version floor is asserted BEFORE any weights are
# fetched, because discovering that vLLM cannot load Qwen3-VL after a 17 GB
# download is a slow way to learn it.
set -euo pipefail

: "${POC3_API_KEY:?set POC3_API_KEY first - the RunPod proxy is public, anyone with the pod id can reach these ports}"

# Weights on the volume, NOT the container disk: Qwen3-VL-8B (~17 GB) + MinerU
# models (~2-4 GB), and a mid-download disk-full costs half an hour.
export HF_HOME=${HF_HOME:-/workspace/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" /workspace/logs /workspace/poc3_ocr

echo "==> installing"
pip install --no-cache-dir -r "$(dirname "$0")/requirements-pod.txt"

echo "==> asserting version floor before downloading any weights"
python - <<'PY'
import sys
from importlib.metadata import version

def ge(v, floor):
    def parts(s):
        out = []
        for p in s.split(".")[:3]:
            num = "".join(c for c in p if c.isdigit())
            out.append(int(num) if num else 0)
        return out
    return parts(v) >= parts(floor)

fail = False
for pkg, floor, why in (
    ("vllm", "0.11.0", "Qwen3VLForConditionalGeneration is not registered below this"),
    ("transformers", "4.57.0", "Qwen3-VL's processor and config land in 4.57"),
):
    v = version(pkg)
    ok = ge(v, floor)
    print(f"  {pkg:14} {v:12} {'OK' if ok else 'TOO OLD'}   (need >= {floor}: {why})")
    fail = fail or not ok

import vllm.model_executor.models.registry as reg
names = set(getattr(reg, "_MULTIMODAL_MODELS", {}) ) | set(getattr(reg, "_TEXT_GENERATION_MODELS", {}))
for arch in ("Qwen3VLForConditionalGeneration",):
    known = arch in names
    print(f"  arch {arch:38} {'registered' if known else 'NOT REGISTERED'}")
    fail = fail or not known

if fail:
    print("\nThis vLLM cannot serve Qwen3-VL. Upgrade vllm and transformers.")
    sys.exit(1)
print("\nQwen3-VL is serveable from this install")
PY

echo "==> pre-downloading model weights"
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3-VL-8B-Instruct",):
    print(f"  {repo}")
    snapshot_download(repo, resume_download=True)
print("weights ready")
PY

echo "==> pre-downloading MinerU pipeline models"
# Fetching these now means /health can answer models_ready truthfully, and the
# first OCR job does not fail with the exit-0-wrote-nothing signature.
if ! mineru-models-download -s huggingface -m pipeline; then
  echo "  huggingface source failed (often rate limiting); trying modelscope"
  MINERU_MODEL_SOURCE=modelscope mineru-models-download -s modelscope -m pipeline
fi

echo
echo "setup complete. Next:"
echo "  bash $(dirname "$0")/serve_vllm.sh"
echo
echo "Then, on the Windows box, put the pod id in config.yaml and run:"
echo "  python src/preflight.py --pod"
