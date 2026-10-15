#!/usr/bin/env bash
# Localize Tagore's "illegal memory access" in GNN-Descent.
#   ./debug_tagore.sh $ANNS_DATA/sift-1M/base.fvecs
# 1) plain runs at 100k and 1M points (is it scale-dependent?)
# 2) compute-sanitizer memcheck at 100k (which kernel / source line?)
set -uo pipefail
FVECS=${1:?usage: $0 base.fvecs}
: "${TAGORE:=$HOME/Tagore}"
: "${GPU:=0}"
OUT=${OUT:-$PWD/tagore_debug}; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
DIM=$(python -c "import numpy as np,sys; print(int(np.fromfile(sys.argv[1],dtype=np.int32,count=1)[0]))" "$FVECS")
SAN=$(command -v compute-sanitizer || ls /usr/local/cuda/bin/compute-sanitizer 2>/dev/null || true)
export CUDA_VISIBLE_DEVICES=$GPU

run() {  # n, extra-prefix...
  local n=$1; shift
  local sub="$OUT/sub_${n}.fvecs"
  [[ -f $sub ]] || head -c $(( n * (DIM + 1) * 4 )) "$FVECS" > "$sub"
  "$@" python "$HERE/build.py" tagore --tagore-build "$TAGORE/build-eval" --fvecs "$sub" \
      --n "$n" --dim "$DIM" --prune CAGRA --raw-out "$OUT/raw_$n" --out "$OUT/graph_$n.u32bin"
}

for n in 100000 1000000; do
  echo "=== plain run n=$n ==="
  run $n 2>&1 | grep -E "time of|CUDA error|illegal|@@RESULT|Error" ; echo "exit=${PIPESTATUS[0]}"
done

if [[ -n "$SAN" ]]; then
  echo "=== compute-sanitizer memcheck n=100000 (slow) ==="
  run 100000 "$SAN" --tool memcheck --print-limit 5 --show-backtrace device 2>&1 \
    | grep -vE "^\s*$" | grep -A12 -E "Invalid|Error|ERROR SUMMARY" | head -60
else
  echo "compute-sanitizer not found (conda install -c nvidia cuda-sanitizer-api, or use /usr/local/cuda/bin)"
fi
