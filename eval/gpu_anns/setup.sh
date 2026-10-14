#!/usr/bin/env bash
# Build the three source-built systems inside the active conda env (env.yml).
# cuVS CAGRA comes from the conda package.
#
#   REPOS=~ CUDA_ARCH=80 ./setup.sh          # repos at ~/Jasper ~/Tagore ~/ScaleGANN
#
# CUDA_ARCH defaults to the compute capability of GPU 0 (e.g. 89 for L40S).
set -euo pipefail

: "${REPOS:=$HOME}"
: "${JASPER:=$REPOS/Jasper}"
: "${TAGORE:=$REPOS/Tagore}"
: "${SCALEGANN:=$REPOS/ScaleGANN}"
: "${JOBS:=$(nproc)}"
if [[ -z "${CUDA_ARCH:-}" ]]; then
  CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | tr -d '.' | head -1)
fi
echo "CUDA_ARCH=$CUDA_ARCH  CONDA_PREFIX=${CONDA_PREFIX:-<none>}"
[[ -n "${CONDA_PREFIX:-}" ]] || { echo "activate the gpu-anns-eval conda env first"; exit 1; }

for r in "$JASPER" "$TAGORE" "$SCALEGANN"; do
  [[ -d "$r" ]] || { echo "missing repo: $r"; exit 1; }
  echo "  $(basename "$r"): $(git -C "$r" rev-parse --abbrev-ref HEAD) @ $(git -C "$r" rev-parse --short HEAD)"
done

THRUST_CFG=$(find "$CONDA_PREFIX" -name thrust-config.cmake 2>/dev/null | head -1)
[[ -n "$THRUST_CFG" ]] || { echo "thrust-config.cmake not found (install cuda-cccl)"; exit 1; }

echo "=== Jasper (create_index, run_query) ==="
cmake -S "$JASPER" -B "$JASPER/build" -DCMAKE_BUILD_TYPE=Release \
  -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DJASPER_CUDA_ARCHS="$CUDA_ARCH" \
  -DThrust_DIR="$(dirname "$THRUST_CFG")" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build "$JASPER/build" --target create_index run_query -j "$JOBS"

echo "=== Tagore (python module) ==="
cmake -S "$TAGORE" -B "$TAGORE/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" -DPYTHON_EXECUTABLE="$(which python)"
cmake --build "$TAGORE/build" --target Tagore -j "$JOBS"

echo "=== ScaleGANN (executeDiskPartition, executeDiskMerge) ==="
cmake -S "$SCALEGANN" -B "$SCALEGANN/build" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build "$SCALEGANN/build" --target executeDiskPartition executeDiskMerge -j "$JOBS"

echo "=== sanity ==="
python -c "import cuvs, cupy; print('cuvs', cuvs.__version__, 'cupy', cupy.__version__)"
python -c "import sys; sys.path.insert(0, '$TAGORE/build'); import Tagore; print('Tagore OK')"
ls -l "$JASPER/build/bin/create_index" "$JASPER/build/bin/run_query" \
      "$SCALEGANN/build/executeDiskPartition" "$SCALEGANN/build/executeDiskMerge"
echo "done"
