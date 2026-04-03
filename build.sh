#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

cd "$PROJECT_DIR"

GPU_BUILD="${GPU_BUILD:-1}"
CCAP="${CCAP:-auto}"
GPU_INDEX="${GPU_INDEX:-0}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
CLEAN_FIRST="${CLEAN_FIRST:-1}"

detect_ccap() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi

  local raw
  raw="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n "$((GPU_INDEX + 1))p" | tr -d '[:space:]' || true)"
  if [[ -z "$raw" ]]; then
    return 1
  fi

  raw="${raw//./}"
  if [[ ! "$raw" =~ ^[0-9]+$ ]]; then
    return 1
  fi

  printf '%s\n' "$raw"
}

if [[ "$GPU_BUILD" == "1" && "$CCAP" == "auto" ]]; then
  if detected_ccap="$(detect_ccap)"; then
    CCAP="$detected_ccap"
  else
    echo "[!] Unable to auto-detect GPU compute capability; falling back to CCAP=89" >&2
    echo "[!] If runtime still reports 'no kernel image is available', rebuild with explicit CCAP." >&2
    CCAP="89"
  fi
fi

echo "[*] build dir   : $(pwd)"
echo "[*] GPU_BUILD   : $GPU_BUILD"
echo "[*] CCAP        : $CCAP"
echo "[*] GPU_INDEX   : $GPU_INDEX"
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

if [[ "$GPU_BUILD" == "1" ]]; then
  cat <<EOF
[*] If runtime still says "no kernel image is available for execution on the device",
    your CCAP is wrong for the target GPU.
    Rebuild explicitly, e.g.:
      CCAP=75 "$SCRIPT_DIR/build.sh"   # T4 / RTX 20xx / GTX 16xx
      CCAP=86 "$SCRIPT_DIR/build.sh"   # RTX 30xx / A10 / A40
      CCAP=89 "$SCRIPT_DIR/build.sh"   # RTX 40xx
      CCAP=90 "$SCRIPT_DIR/build.sh"   # H100 / H20 class
EOF
fi
