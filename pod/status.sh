#!/usr/bin/env bash
# Progress of pod/run_all.sh: cells finished per case, what the runner(s) are
# doing, whether the job is still alive. Safe to run any time, from any terminal.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$REPO/output_policies/logs"
CASES="${CASES:-md_only pdf_only pdf_md}"
declare -A DIR=([md_only]="MD only" [pdf_only]="PDF only" [pdf_md]="PDF+MD")

echo "== job"
if pgrep -f "pod/run_all.sh" > /dev/null; then echo "running (pid $(pgrep -f pod/run_all.sh | tr '\n' ' '))"
elif [ -f "$LOGS/DONE" ]; then echo "FINISHED - reports in output_policies/report/"
elif [ -f "$LOGS/FAILED" ]; then echo "STOPPED with an error - tail $LOGS/run_all.log"
else echo "not running"; fi
tail -n 3 "$LOGS/run_all.log" 2>/dev/null

for c in $CASES; do
  d="$REPO/output_policies/8B/${DIR[$c]}"
  n=$(find "$d" -maxdepth 2 -name '*.meta.json' -not -path '*/_windows/*' 2>/dev/null | wc -l)
  w=$(find "$d" -path '*/_windows/*' -name '*.meta.json' 2>/dev/null | wc -l)
  echo; echo "== $c: $n documents done, $w windows done"
  [ -f "$LOGS/run_$c.log" ] && grep -vE "page images cached|markdown cache hit" "$LOGS/run_$c.log" | tail -n 2 | cut -c1-200
done

if [ -f "$LOGS/run_sequential.log" ]; then
  echo; echo "== sequential runner (one request at a time)"
  grep -vE "page images cached|markdown cache hit" "$LOGS/run_sequential.log" | tail -n 4 | cut -c1-200
fi

# Per policy x case model time, for every cell finished so far.
"${VENV:-/workspace/POC3/venv}/bin/python" - "$REPO" <<'PY' 2>/dev/null
import json, sys
from pathlib import Path
root = Path(sys.argv[1]) / "output_policies" / "8B"
cases = [("MD only", "md_only"), ("PDF only", "pdf_only"), ("PDF+MD", "pdf_md")]
rows = {}
for d, c in cases:
    for m in (root / d).glob("*/*.meta.json"):
        meta = json.loads(m.read_text(encoding="utf-8"))
        if meta.get("status") == "ERROR":
            continue
        rows.setdefault((meta.get("n_pages") or 0, meta["doc_id"]), {})[c] = meta
if rows:
    print("\n== model time per policy and case (s; s/page)")
    print(f"   {'policy':24} {'pages':>5}  {'MD only':>16}  {'PDF only':>16}  {'PDF+MD':>16}")
    for (pages, doc), by in sorted(rows.items()):
        cell = lambda c: (f"{by[c]['model_latency_s']:7.0f} ({by[c]['latency_per_page_s']:5.1f})"
                          if c in by and by[c].get("model_latency_s") is not None else f"{'-':>16}")
        print(f"   {doc:24} {pages:5d}  {cell('md_only'):>16}  {cell('pdf_only'):>16}  {cell('pdf_md'):>16}")
PY

echo; echo "== GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
