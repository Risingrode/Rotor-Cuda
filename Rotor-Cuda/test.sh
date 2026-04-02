#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# =========================
# Masked search test script
# 默认跑“必命中”自测样例
# 原始无命中样例可通过 CASE=sample_nomatch 切换
# 仍可通过环境变量覆盖 PREFIX / ADDR / SETS_REAL
# =========================

CASE="${CASE:-selfhit}"

case "$CASE" in
  selfhit)
    default_prefix='4226D9811DDB6397F1DBB6BF73359A6C5B04DB6B5'
    default_addr='14JiAhBSdHjEPG92jRuKVJcABaGV7R6vBy'
    default_sets='AF,26,E7,05,3C,8F,94,7D,A5,2C,8E,D4,8A,B6,4D,49,A7,1E,35,B8,12,C3,0F'
    ;;
  sample_nomatch)
    default_prefix='D0AC934BA9987E529BF3150373B63BD06849D740A'
    default_addr='17SGFkZyNGbtjKqP3GnEH5CVRzQ6YAqcbU'
    default_sets='01,AB,12,BC,23,CD,34,DC,43,ED,54,FE01,6578,0F12,76,10,87,2134,98AB,3245,A9BC,4356,BACD'
    ;;
  *)
    echo "[!] Unknown CASE: $CASE" >&2
    echo "[!] Supported CASE values: selfhit, sample_nomatch" >&2
    exit 1
    ;;
esac

export PREFIX="${PREFIX:-$default_prefix}"
export ADDR="${ADDR:-$default_addr}"
export SETS_REAL="${SETS_REAL:-$default_sets}"

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

if [[ "$USE_GPU" == "1" ]]; then
  if ! ldd "$ROTOR_BIN" 2>/dev/null | grep -q 'libcudart'; then
    echo "[!] Rotor binary does not appear to be built with GPU support." >&2
    echo "[!] Current binary is missing libcudart dependency." >&2
    echo "[!] Rebuild on the target host, e.g.:" >&2
    echo "    make clean" >&2
    echo "    make gpu=1 CCAP=89 all" >&2
    exit 1
  fi
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
print("[*] case               =", os.environ.get("CASE", "selfhit"))

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
echo "[*] CASE         = $CASE"
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
