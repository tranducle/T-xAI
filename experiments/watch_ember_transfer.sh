#!/bin/bash
# Five-minute watchdog over the E9 run.
#
# supervise_ember_transfer.sh already restarts a failed run; this only *observes*, so that
# progress can be read without attaching to the process and without a foreground
# poll every two minutes. It appends one line per interval and stops on its own
# once the supervisor is gone, so it cannot outlive the job it is watching.
#
# The distinction it records is the one that matters when a fit runs long: a
# process burning CPU is computing, a process at ~0% CPU with a growing page-in
# count is thrashing. Only the second is a reason to intervene.
#
# Usage: watch_ember_transfer.sh [interval_seconds]

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
INTERVAL="${1:-300}"
WATCH_LOG="$ROOT/results/e9_watch.log"

mkdir -p "$ROOT/results"

while true; do
  SUP_PID="$(pgrep -f 'supervise_ember_transfer.sh' | head -1)"
  PY_PID="$(pgrep -f 'run_ember_transfer.py' | head -1)"

  if [ -z "$SUP_PID" ] && [ -z "$PY_PID" ]; then
    printf '%s supervisor and worker both gone; watchdog stopping\n' \
           "$(date '+%Y-%m-%d %H:%M:%S')" >> "$WATCH_LOG"
    exit 0
  fi

  if [ -n "$PY_PID" ]; then
    STATS="$(ps -o %cpu=,rss=,etime=,state= -p "$PY_PID" | \
             awk '{printf "cpu %6s%%  rss %5.2f GB  elapsed %-9s state %s", $1, $2/1048576, $3, $4}')"
  else
    STATS="worker not running (supervisor $SUP_PID between attempts)"
  fi

  STAGES="$(python3 -c "
import json, pathlib
p = pathlib.Path('$ROOT/results/E9_ember2018.json')
if not p.exists():
    print('no results file yet')
else:
    d = json.loads(p.read_text())
    print('full_auc=%s groups=%d/9 curve=%d/4' % (
        ('%.6f' % d['auc']) if 'auc' in d else 'pending',
        len(d.get('auc_by_feature_group_alone', {})),
        len(d.get('learning_curve', {}))))
" 2>/dev/null || echo 'results unreadable')"

  printf '%s %s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$STATS" "$STAGES" \
      >> "$WATCH_LOG"
  sleep "$INTERVAL"
done
