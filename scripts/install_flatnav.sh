#!/usr/bin/env bash
# Build FlatNav's Python extension from the pinned submodule.
#
#   bash scripts/install_flatnav.sh
#
# FlatNav itself is unmodified — third_party/flatnav is pinned at upstream's v0.1.2-rc1 tag.
# The only local change is a build-configuration patch: FlatNav's setup.py unconditionally
# passes -march=native and AVX2 flags, which produces a binary that dies with SIGILL on a CPU
# without AVX2+FMA. The paper's machine (Intel Xeon E5-2620, Sandy Bridge) is such a CPU.
#
# The scalar build is selected automatically from /proc/cpuinfo. Force it either way:
#
#   FLATNAV_SCALAR_BUILD=1 bash scripts/install_flatnav.sh   # scalar, portable, slower
#   FLATNAV_SCALAR_BUILD=0 bash scripts/install_flatnav.sh   # -march=native
#
# This changes throughput, not recall.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FLATNAV_DIR="${ROOT}/third_party/flatnav"
PATCH_FILE="${ROOT}/patches/flatnav-scalar-build.patch"
ENV_NAME="${RGANN_ENV:-rgann}"

# conda is often absent from a non-interactive shell's PATH, so look for it where the
# installers put it. Override with CONDA=/path/to/conda for anything unusual.
if [ -n "${CONDA:-}" ]; then
  :
elif command -v conda >/dev/null 2>&1; then
  CONDA="conda"
else
  for candidate in "${HOME}/miniforge3" "${HOME}/miniconda3" "${HOME}/anaconda3" /opt/conda; do
    if [ -x "${candidate}/bin/conda" ]; then
      CONDA="${candidate}/bin/conda"
      break
    fi
  done
fi

if [ -z "${CONDA:-}" ]; then
  echo "error: conda not found. Run scripts/setup_ubuntu.sh (or setup_macos.sh), or set CONDA=/path/to/conda" >&2
  exit 1
fi

run_in_env() { "${CONDA}" run -n "${ENV_NAME}" --no-capture-output "$@"; }

cpu_lacks_avx2_fma() {
  # Only Linux exposes /proc/cpuinfo. Elsewhere, assume the CPU is modern enough and let the
  # import check in rgann.indexes.flatnav catch it if not.
  if [ ! -r /proc/cpuinfo ]; then
    return 1
  fi
  local flags
  flags="$(awk '/^flags/ { print; exit }' /proc/cpuinfo)"
  [ "${flags}" = "${flags#*avx2}" ] || [ "${flags}" = "${flags#*fma}" ]
}

scalar_build="${FLATNAV_SCALAR_BUILD:-auto}"
if [ "${scalar_build}" = "auto" ] || [ -z "${scalar_build}" ]; then
  if cpu_lacks_avx2_fma; then
    scalar_build=1
  else
    scalar_build=0
  fi
fi

cd "${ROOT}"

# In a git checkout the submodule may simply not be initialised yet. In an image built by
# COPY there is no .git at all and the sources are already present, so asking git for them
# fails with "not a git repository" and takes the whole build down with it.
if [ ! -f "${FLATNAV_DIR}/python-bindings/setup.py" ]; then
  if git -C "${ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    git submodule update --init --recursive third_party/flatnav
  else
    echo "error: ${FLATNAV_DIR} has no sources and this is not a git checkout." >&2
    echo "       Populate it before running this script (git submodule update --init --recursive)." >&2
    exit 1
  fi
fi

# ninja is named explicitly rather than left to arrive as a transitive dependency of
# scikit-build: FlatNav's build selects the Ninja generator, and where it is missing the
# build dies with "CMake was unable to find a build program corresponding to Ninja".
run_in_env pip install --no-cache-dir scikit-build "cmake<4" pybind11 ninja

SETUP_PY="${FLATNAV_DIR}/python-bindings/setup.py"
# Re-running must not stack patches. Where git is available, resetting to the pinned state is
# the cleanest way to guarantee that; where it is not, the patch is detected and skipped
# instead, keyed on a string only the patch introduces.
if git -C "${FLATNAV_DIR}" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "${FLATNAV_DIR}" checkout -- python-bindings/setup.py
fi

# Upstream's setup.py hard-codes `-arch x86_64` for macOS, which was true of every Mac when
# v0.1.2-rc1 was tagged. On Apple Silicon it makes the compiler target x86_64 while scikit-build
# sets CMAKE_OSX_ARCHITECTURES=arm64, and CMake refuses the mismatch:
#   The CXX compiler targets architectures "x86_64;arm64" but CMAKE_OSX_ARCHITECTURES is "arm64"
# Build for whatever the host actually is. Keyed on the replacement text so re-runs are safe.
if [ "$(uname -s)" = "Darwin" ] && ! grep -q 'platform.machine()' "${SETUP_PY}"; then
  echo "FlatNav: building for $(uname -m) rather than upstream's hard-coded x86_64"
  patch -p1 -d "${FLATNAV_DIR}" <"${ROOT}/patches/flatnav-macos-arch.patch"
fi

if [ "${scalar_build}" = "1" ]; then
  echo "FlatNav: scalar build (no AVX2/FMA on this CPU)"
  if grep -q -- '-DNO_SIMD_VECTORIZATION' "${SETUP_PY}"; then
    echo "FlatNav: scalar patch already applied, not reapplying"
  else
    patch -p1 -d "${FLATNAV_DIR}" <"${PATCH_FILE}"
  fi
  # -march=native would still emit AVX2 on a host that has it; pin to the paper's baseline.
  sed -i.bak 's/-march=native/-march=sandybridge -mno-fma/g' "${SETUP_PY}"
  rm -f "${SETUP_PY}.bak"
  export NO_SIMD_VECTORIZATION=1
else
  echo "FlatNav: native SIMD build (-march=native)"
  unset NO_SIMD_VECTORIZATION
fi

run_in_env bash -c "cd '${FLATNAV_DIR}/python-bindings' && pip install --no-cache-dir --no-build-isolation ."

run_in_env python -c "import flatnav; from flatnav.index import create; print('flatnav', flatnav.__version__)"
