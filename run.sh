export PREFIX='xxx'
export ADDR='xxxx'
export SETS_REAL='01,AB,12,BC,23,CD,34,DC,43,ED,54,FE01,6578,0F12,76,10,87,2134,98AB,3245,A9BC,4356,BACD'
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

./Rotor -g --gpui 0 --gpux 256,512 -m address --coin BTC \
  --prob-profile none \
  --prefix "$PREFIX" \
  --suffixsets "$SETS_REAL" \
  -o found_12Js.txt \
  "$ADDR" | tee run_12Js.log
