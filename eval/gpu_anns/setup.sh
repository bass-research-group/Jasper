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

# Resolve the host compilers to absolute paths. Jasper's CMakeLists takes
# CMAKE_{C,CXX}_COMPILER from $CC/$CXX verbatim, and conda's activation sets
# bare names that CPM sub-projects then fail to find. Fall back to system gcc.
for v in CC CXX; do
  val="${!v:-}"
  if [[ -n "$val" ]] && p=$(command -v "$val" 2>/dev/null); then export "$v=$p"; else unset "$v"; fi
done
if [[ -z "${CC:-}" || -z "${CXX:-}" ]]; then
  export CC=$(command -v gcc) CXX=$(command -v g++)
fi
echo "CC=$CC ($("$CC" -dumpfullversion))  CXX=$CXX  nvcc=$(command -v nvcc)"
COMPILERS=(-DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_CUDA_HOST_COMPILER="$CXX")

THRUST_CFG=$(find "$CONDA_PREFIX" -name thrust-config.cmake 2>/dev/null | head -1)
[[ -n "$THRUST_CFG" ]] || { echo "thrust-config.cmake not found (install cuda-cccl)"; exit 1; }

# build-eval/ is ours (your own build/ dirs are untouched); start clean so no stale
# CMake cache keeps a bad compiler. CLEAN=0 to reuse.
if [[ "${CLEAN:-1}" == 1 ]]; then rm -rf "$JASPER/build-eval" "$TAGORE/build-eval" "$SCALEGANN/build-eval"; fi

echo "=== Jasper (create_index, run_query) ==="
cmake -S "$JASPER" -B "$JASPER/build-eval" -DCMAKE_BUILD_TYPE=Release "${COMPILERS[@]}" \
  -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DJASPER_CUDA_ARCHS="$CUDA_ARCH" \
  -DThrust_DIR="$(dirname "$THRUST_CFG")" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build "$JASPER/build-eval" --target create_index run_query -j "$JOBS"

echo "=== Tagore (python module) ==="
cmake -S "$TAGORE" -B "$TAGORE/build-eval" -DCMAKE_BUILD_TYPE=Release "${COMPILERS[@]}" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" -DPYTHON_EXECUTABLE="$(which python)"
cmake --build "$TAGORE/build-eval" --target Tagore -j "$JOBS"

echo "=== ScaleGANN (executeDiskPartition, executeDiskMerge) ==="
cmake -S "$SCALEGANN" -B "$SCALEGANN/build-eval" "${COMPILERS[@]}" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build "$SCALEGANN/build-eval" --target executeDiskPartition executeDiskMerge -j "$JOBS"

echo "=== sanity ==="
python -c "import cuvs, cupy; print('cuvs', cuvs.__version__, 'cupy', cupy.__version__)"
python -c "import sys; sys.path.insert(0, '$TAGORE/build-eval'); import Tagore; print('Tagore OK')"
ls -l "$JASPER/build-eval/bin/create_index" "$JASPER/build-eval/bin/run_query" \
      "$SCALEGANN/build-eval/executeDiskPartition" "$SCALEGANN/build-eval/executeDiskMerge"
echo "done"
