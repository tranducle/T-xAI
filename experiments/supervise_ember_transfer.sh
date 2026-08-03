#!/bin/bash
# Supervise run_ember_transfer.py: retry on failure, resume from the results file.
#
# E9 writes results/E9_ember2018.json after every stage, so a retry picks up at
# the first unfinished stage rather than refitting the expensive full model.
#
# The one failure this loop treats specially is running out of memory. E9 sits
# close to the RAM ceiling by design (see its docstring), and retrying an OOM at
# the same training size would just OOM again — so on exit 137 (SIGKILL, which is
# how macOS ends a process the memory manager gives up on) the training size is
# halved and recorded, rather than the run being repeated identically.
#
# Usage: supervise_ember_transfer.sh [train_size]

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TRAIN_SIZE="${1:-300000}"
LOG="$ROOT/results/e9.log"
STATUS="$ROOT/results/E9_STATUS.txt"
MAX_ATTEMPTS=5
HEARTBEAT_SEC=60

mkdir -p "$ROOT/results"
ATTEMPT=1

note() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

heartbeat() {
  while true; do
    sleep "$HEARTBEAT_SEC"
    {
      printf 'state       : RUNNING (attempt %s/%s, train_size %s)\n' \
             "$ATTEMPT" "$MAX_ATTEMPTS" "$TRAIN_SIZE"
      printf 'stages done : %s\n' \
             "$(python3 -c "
import json,pathlib
p=pathlib.Path('$ROOT/results/E9_ember2018.json')
if not p.exists(): print('none yet')
else:
    d=json.loads(p.read_text())
    g=d.get('auc_by_feature_group_alone',{})
    c=d.get('learning_curve',{})
    print(f\"full_auc={'yes' if 'auc' in d else 'no'} groups={len(g)}/9 curve={len(c)}/4\")
" 2>/dev/null || echo '?')"
      printf 'rss (top)   : %s\n' "$(ps -o rss= -p "${PY_PID:-0}" 2>/dev/null | awk '{printf "%.2f GB", $1/1048576}')"
      printf 'updated     : %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
      printf 'last log    : %s\n' "$(tail -1 "$LOG" 2>/dev/null)"
    } > "$STATUS"
  done
}

note "E9 supervisor start: train_size=$TRAIN_SIZE"
heartbeat &
HEARTBEAT_PID=$!
trap 'kill $HEARTBEAT_PID 2>/dev/null' EXIT

while [ "$ATTEMPT" -le "$MAX_ATTEMPTS" ]; do
  note "attempt $ATTEMPT/$MAX_ATTEMPTS at train_size=$TRAIN_SIZE"
  python3 "$HERE/run_ember_transfer.py" --train-size "$TRAIN_SIZE" --log "$LOG" \
      >/dev/null 2>>"$LOG" &
  PY_PID=$!
  wait "$PY_PID"
  RC=$?

  if [ "$RC" -eq 0 ]; then
    note "E9 COMPLETE (exit 0) on attempt $ATTEMPT at train_size=$TRAIN_SIZE"
    { printf 'state       : DONE\n'
      printf 'train_size  : %s\n' "$TRAIN_SIZE"
      printf 'attempts    : %s\n' "$ATTEMPT"
      printf 'finished    : %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    } > "$STATUS"
    exit 0
  fi

  if [ "$RC" -eq 137 ] || [ "$RC" -eq 9 ]; then
    TRAIN_SIZE=$((TRAIN_SIZE / 2))
    note "killed (exit $RC), almost certainly out of memory; halving to $TRAIN_SIZE"
    if [ "$TRAIN_SIZE" -lt 25000 ]; then
      note "FATAL: train_size fell below 25000; stopping"
      printf 'state       : FAILED (out of memory below 25000 rows)\n' > "$STATUS"
      exit 1
    fi
    # A partial stage-1 result computed at the old size would be mixed with
    # stages computed at the new one. Drop it so every number in the file comes
    # from one training size.
    rm -f "$ROOT/results/E9_ember2018.json"
  else
    note "attempt $ATTEMPT failed with exit $RC; retrying in 30s"
    sleep 30
  fi
  ATTEMPT=$((ATTEMPT + 1))
done

note "GAVE UP after $MAX_ATTEMPTS attempts"
printf 'state       : FAILED (max attempts)\n' > "$STATUS"
exit 1
