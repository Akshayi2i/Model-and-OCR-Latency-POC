#!/usr/bin/env bash
# Progress of pod/run_all.sh: cells finished per case, what each runner is doing,
# whether the job is still alive. Safe to run any time, from any terminal.
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
  echo; echo "== $c: $n/10 documents done, $w windows done"
  grep -vE "page images cached|markdown cache hit" "$LOGS/run_$c.log" 2>/dev/null | tail -n 2 | cut -c1-200
done
echo; echo "== GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
