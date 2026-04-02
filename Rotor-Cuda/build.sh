#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_BUILD="${GPU_BUILD:-1}"
CCAP="${CCAP:-89}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
CLEAN_FIRST="${CLEAN_FIRST:-1}"

echo "[*] build dir   : $(pwd)"
echo "[*] GPU_BUILD   : $GPU_BUILD"
echo "[*] CCAP        : $CCAP"
echo "[*] JOBS        : $JOBS"
echo "[*] CLEAN_FIRST : $CLEAN_FIRST"

if [[ "$GPU_BUILD" == "1" ]] && ! command -v /usr/local/cuda/bin/nvcc >/dev/null 2>&1; then
  echo "[!] nvcc not found at /usr/local/cuda/bin/nvcc" >&2
  exit 1
fi

if [[ "$CLEAN_FIRST" == "1" ]]; then
  make clean
fi

if [[ "$GPU_BUILD" == "1" ]]; then
  make -j"$JOBS" gpu=1 CCAP="$CCAP" all
else
  make -j"$JOBS" all
fi

echo
echo "[*] Build done."
if [[ -f ./Rotor ]]; then
  ls -lh ./Rotor
  echo "[*] Linked CUDA runtime:"
  ldd ./Rotor | grep cudart || echo "    (no libcudart found)"
fi
