#!/usr/bin/env bash
set -euo pipefail

export PREFIX='xxx'
export ADDR='xxxx'
export SETS_REAL='01,AB,12,BC,23,CD,34,DC,43,ED,54,FE01,6578,0F12,76,10,87,2134,98AB,3245,A9BC,4356,BACD'
export GPUI="${GPUI:-0}"
export GPUX="${GPUX:-}"
export ROTOR_MASKED_GPU_FASTPATH="${ROTOR_MASKED_GPU_FASTPATH:-auto}"
export ROTOR_MASKED_GPU_BATCH_STEPS="${ROTOR_MASKED_GPU_BATCH_STEPS:-auto}"
export ROTOR_MASKED_GPU_AUTOTUNE="${ROTOR_MASKED_GPU_AUTOTUNE:-1}"
# 0A1B2C3D4E5F60718293A4B
python3 - <<'PY'
import os, sys
s = os.environ['SETS_REAL']
n = len(s.split(','))
print("suffix group count =", n)
if n != 23:
    print("ERROR: suffix group count must be 23", file=sys.stderr)
    sys.exit(1)
PY

cmd=(./Rotor -g --gpui "$GPUI")
if [[ -n "$GPUX" ]]; then
  cmd+=(--gpux "$GPUX")
fi
cmd+=(-m address --coin BTC \
  --prob-profile none \
  --prefix "$PREFIX" \
  --suffixsets "$SETS_REAL" \
  -o found_12Js.txt \
  "$ADDR")

printf 'Running: '
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}" | tee run_12Js.log
