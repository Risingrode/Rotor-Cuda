#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# =========================
# Masked search test script
# 可通过环境变量覆盖下面参数
# =========================

export PREFIX="${PREFIX:-D0AC934BA9987E529BF3150373B63BD06849D740A}"
export ADDR="${ADDR:-17SGFkZyNGbtjKqP3GnEH5CVRzQ6YAqcbU}"
export SETS_REAL="${SETS_REAL:-01,AB,12,BC,23,CD,34,DC,43,ED,54,FE01,6578,0F12,76,10,87,2134,98AB,3245,A9BC,4356,BACD}"
# 对应示例尾部: 0A1B2C3D4E5F60718293A4B

export COIN="${COIN:-BTC}"
export SEARCH_MODE="${SEARCH_MODE:-address}"
export PROB_PROFILE="${PROB_PROFILE:-none}"

export USE_GPU="${USE_GPU:-1}"
export GPUI="${GPUI:-0}"
export GPUX="${GPUX:-256,512}"
export CPU_THREADS="${CPU_THREADS:-4}"

export OUT_FILE="${OUT_FILE:-found_mask_test.txt}"
export LOG_FILE="${LOG_FILE:-run_mask_test.log}"
export ROTOR_BIN="${ROTOR_BIN:-./Rotor}"
export EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -x "$ROTOR_BIN" ]]; then
  echo "[!] Rotor binary not found or not executable: $ROTOR_BIN" >&2
  echo "[!] Build first, e.g.:" >&2
  echo "    make all" >&2
  echo "    # or GPU build on target host" >&2
  echo "    make gpu=1 CCAP=89 all" >&2
  exit 1
fi

python3 - <<'PY'
import os, sys

prefix = os.environ["PREFIX"].strip()
addr = os.environ["ADDR"].strip()
sets_raw = os.environ["SETS_REAL"].strip()
sets = [x.strip() for x in sets_raw.split(",") if x.strip()]

print("[*] prefix len         =", len(prefix))
print("[*] suffix group count =", len(sets))
print("[*] target address     =", addr)

if len(sets) != 23:
    print("[!] ERROR: suffix group count must be 23", file=sys.stderr)
    sys.exit(1)

if len(prefix) + len(sets) != 64:
    print(f"[!] ERROR: prefix len ({len(prefix)}) + suffix groups ({len(sets)}) must equal 64", file=sys.stderr)
    sys.exit(1)

bad = [c for c in prefix if c not in "0123456789abcdefABCDEF"]
if bad:
    print("[!] ERROR: prefix contains non-hex chars", file=sys.stderr)
    sys.exit(1)
PY

echo "[*] USE_GPU      = $USE_GPU"
echo "[*] GPUI         = $GPUI"
echo "[*] GPUX         = $GPUX"
echo "[*] PROB_PROFILE = $PROB_PROFILE"
echo "[*] OUT_FILE     = $OUT_FILE"
echo "[*] LOG_FILE     = $LOG_FILE"
echo

cmd=(
  "$ROTOR_BIN"
  -m "$SEARCH_MODE"
  --coin "$COIN"
  --prob-profile "$PROB_PROFILE"
  --prefix "$PREFIX"
  --suffixsets "$SETS_REAL"
  -o "$OUT_FILE"
)

if [[ "$USE_GPU" == "1" ]]; then
  cmd+=(-g --gpui "$GPUI" --gpux "$GPUX")
else
  cmd+=(-t "$CPU_THREADS")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra=( $EXTRA_ARGS )
  cmd+=("${extra[@]}")
fi

cmd+=("$ADDR")

echo "[*] Running command:"
printf '    %q ' "${cmd[@]}"
echo
echo

"${cmd[@]}" | tee "$LOG_FILE"
