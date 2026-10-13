#!/usr/bin/env bash
# The whole test as one job ON THE POD: preflight, the three cases in parallel,
# a retry pass, the reports. Talks to vLLM and mineru-api on 127.0.0.1, so it
# needs no laptop, no proxy and no internet once started.
#
# Start it detached, so closing the terminal / the laptop does not stop it:
#     cd /workspace/POC3/Model-and-OCR-Latency-POC
#     nohup setsid bash pod/run_all.sh > /dev/null 2>&1 &
# Watch it:
#     bash pod/status.sh
#
# Re-running it is safe: finished documents and windows are skipped, so a second
# start after an interruption only does what is left.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-/workspace/POC3/venv}"
SERVE_CONF="${SERVE_CONF:-/workspace/serve.conf}"
LABEL="${LABEL:-3 cases in parallel on the pod (localhost, vLLM batching)}"
CASES="${CASES:-md_only pdf_only pdf_md}"
LOGS="$REPO/output_policies/logs"
mkdir -p "$LOGS"
MAIN_LOG="$LOGS/run_all.log"
exec >> "$MAIN_LOG" 2>&1
cd "$REPO"

say () { echo "[$(date -u '+%F %T') UTC] $*"; }
say "==== run_all.sh starting in $REPO (pid $$)"
rm -f "$LOGS/DONE" "$LOGS/FAILED"

# --- environment ---------------------------------------------------------------
if [ ! -f "$SERVE_CONF" ]; then say "missing $SERVE_CONF"; touch "$LOGS/FAILED"; exit 1; fi
# shellcheck disable=SC1090
source "$SERVE_CONF"
export POC3_API_KEY="${API_KEY:?API_KEY not set in $SERVE_CONF}"
export POC3_VL_BASE_URL="http://127.0.0.1:${VL_PORT:-8000}"
export POC3_OCR_BASE_URL="http://127.0.0.1:${MINERU_PORT:-8001}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

if [ ! -x "$VENV/bin/python" ]; then
  say "creating venv $VENV"
  python3 -m venv "$VENV" && "$VENV/bin/pip" install -q -r requirements.txt \
    || { say "venv setup failed"; touch "$LOGS/FAILED"; exit 1; }
fi
PY="$VENV/bin/python"
"$PY" -c "import yaml, requests, pymupdf, jsonschema" \
  || { say "python deps missing in $VENV: pip install -r requirements.txt"; touch "$LOGS/FAILED"; exit 1; }

# --- preflight: no GPU time is spent on a setup that cannot finish ---------------
for t in 1 2 3 4 5 6 7 8 9 10; do
  if "$PY" src/preflight.py --pod > "$LOGS/preflight_pod.log" 2>&1; then break; fi
  say "preflight --pod not GO yet (attempt $t) - services still starting? retrying in 60s"
  sleep 60
done
if ! grep -q "PREFLIGHT: GO" "$LOGS/preflight_pod.log"; then
  say "preflight --pod is NO-GO - see $LOGS/preflight_pod.log"; touch "$LOGS/FAILED"; exit 1
fi
if ! "$PY" src/preflight.py --local > "$LOGS/preflight_local.log" 2>&1; then
  say "preflight --local is NO-GO - see $LOGS/preflight_local.log"; touch "$LOGS/FAILED"; exit 1
fi
say "preflight GO (pod and local)"

# --- the three cases, in parallel ------------------------------------------------
run_pass () {   # pass-number
  local pids=() c
  for c in $CASES; do
    "$PY" src/runner.py --only "$c" --label "$LABEL" >> "$LOGS/run_$c.log" 2>&1 &
    pids+=($!)
    say "pass $1: started $c (pid $!)"
  done
  for p in "${pids[@]}"; do wait "$p"; say "pass $1: pid $p exited rc=$?"; done
}
run_pass 1
# A window whose REQUEST failed (timeout, server error) is retried by simply
# running again; everything finished is skipped.
say "retry pass for any window that failed on the network"
run_pass 2

# --- reports ---------------------------------------------------------------------
"$PY" src/report.py > "$LOGS/report.log" 2>&1
say "report.py exited rc=$? - output_policies/report/comparison.md and timing.md"
touch "$LOGS/DONE"
say "==== run_all.sh finished"
