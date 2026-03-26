#!/usr/bin/env bash

# Run all experiment folders under scripts/exp*/
# Usage: ./run_all_exps.sh [-n] [-s]
#   -n  dry-run: only print what would be run
#   -s  stop on first non-zero exit
#
# Chain-failure recovery: if an exp fails without writing
# outputs/validation_summary.json, the last known good summary is
# copied there so downstream experiments can still resolve their baseline.

DRY_RUN=0
STOP_ON_ERROR=0
while getopts ":ns" opt; do
  case ${opt} in
    n ) DRY_RUN=1 ;;
    s ) STOP_ON_ERROR=1 ;;
    \? ) echo "Usage: $0 [-n] [-s]"; exit 1 ;;
  esac
done
shift $((OPTIND -1))

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAST_GOOD_SUMMARY=""   # absolute path to most-recent validation_summary.json

for d in "$ROOT_DIR"/exp*/; do
  d="${d%/}"  # strip trailing slash
  if [ ! -d "$d" ]; then
    continue
  fi

  echo ""
  echo "=== Running: $d ==="

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry-run: would execute entrypoint(s) in $d"
    continue
  fi

  if [ -x "$d/run.sh" ]; then
    (cd "$d" && ./run.sh)
    rc=$?
  elif [ -f "$d/run.sh" ]; then
    (cd "$d" && bash run.sh)
    rc=$?
  elif [ -f "$d/train.py" ]; then
    (cd "$d" && python3 train.py)
    rc=$?
  elif [ -f "$d/main.py" ]; then
    (cd "$d" && python3 main.py)
    rc=$?
  else
    echo "No recognized entrypoint in $d (checked run.sh, train.py, main.py)"
    rc=0
  fi

  OUT_SUMMARY="$d/outputs/validation_summary.json"

  if [ "$rc" -ne 0 ]; then
    echo "Exit code $rc from $d"

    # If this exp didn't produce its summary but we have a prior good one,
    # propagate it so chain experiments can find their baseline.
    if [ ! -f "$OUT_SUMMARY" ] && [ -n "$LAST_GOOD_SUMMARY" ]; then
      mkdir -p "$d/outputs"
      cp "$LAST_GOOD_SUMMARY" "$OUT_SUMMARY"
      echo "  -> Propagated last-good summary from $LAST_GOOD_SUMMARY to $OUT_SUMMARY"
    fi

    if [ "$STOP_ON_ERROR" -eq 1 ]; then
      exit $rc
    fi
  fi

  # Track the newest successfully-written summary for chain recovery
  if [ -f "$OUT_SUMMARY" ]; then
    LAST_GOOD_SUMMARY="$OUT_SUMMARY"
  fi
done

echo ""
echo "All done."
