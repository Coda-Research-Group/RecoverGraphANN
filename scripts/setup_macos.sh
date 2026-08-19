#!/usr/bin/env bash
# Setup on macOS.
#
#   bash scripts/setup_macos.sh
#
# RoarGraph is deliberately NOT installed here. Its distance kernels are written directly
# against x86 SIMD intrinsics (third_party/RoarGraph/include/efanna2e/distance.h), which Apple
# Silicon does not have.
#
# An Intel Mac IS x86-64 and might well build it — the skip is not a claim that it cannot.
# No Intel Mac was available to check, and a build step nobody has run is worse than a
# documented gap. See the platform table in the README.
#
# Everything else works: HNSW, FAISS NSG, FlatNav, all figures, Table 1, the test suite and
# `make quick`.
#
# Recall reproduces on macOS. Throughput does not — the paper's QPS numbers are specific to
# the machine in the README's hardware note.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${RGANN_ENV:-rgann}"

log() { printf '\n==> %s\n' "$1"; }

if [ "$(uname -s)" != "Darwin" ]; then
  echo "error: this script is for macOS; on Linux use scripts/setup_ubuntu.sh" >&2
  exit 1
fi

# --- 1. Toolchain ------------------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  echo "error: Homebrew not found. Install it from https://brew.sh and re-run." >&2
  exit 1
fi

log "Installing build dependencies"
brew install libomp cmake

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found. Install Miniforge: brew install --cask miniforge" >&2
  exit 1
fi

CONDA_ROOT="$(conda info --base)"
CONDA="${CONDA_ROOT}/bin/conda"

# --- 2. Environment ----------------------------------------------------------------------
if "${CONDA}" env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  log "Updating conda environment '${ENV_NAME}'"
  "${CONDA}" env update -n "${ENV_NAME}" -f "${ROOT}/environment.yml" --prune
else
  log "Creating conda environment '${ENV_NAME}'"
  "${CONDA}" env create -n "${ENV_NAME}" -f "${ROOT}/environment.yml"
fi

run_in_env() { "${CONDA}" run -n "${ENV_NAME}" --no-capture-output "$@"; }

# --- 3. Pinned third-party sources -------------------------------------------------------
log "Checking out submodules at their pinned commits"
git -C "${ROOT}" submodule update --init --recursive third_party/hnswlib third_party/flatnav

# --- 4. Native backends (RoarGraph excluded, see header) ---------------------------------
log "Building hnswlib (instrumented fork)"
run_in_env pip install --no-cache-dir "${ROOT}/third_party/hnswlib"

log "Building FlatNav"
bash "${ROOT}/scripts/install_flatnav.sh"

log "Installing rgann"
run_in_env pip install --no-cache-dir -e "${ROOT}"

# --- 5. Verify ---------------------------------------------------------------------------
log "Verifying backends"
run_in_env python - <<'PY'
import faiss
import flatnav
import hnswlib

import rgann

print(f'faiss     {faiss.__version__}')
print(f'flatnav   {flatnav.__version__}')
print('hnswlib   ok (instrumented build)' if hasattr(hnswlib.Index, 'get_all_links')
      else 'hnswlib   WRONG BUILD: get_all_links missing, Table 1 cannot be produced')
print('RoarGraph skipped on macOS (needs x86-64; untested on Intel Macs)')
print(f'rgann     {rgann.__version__}')
PY

cat <<EOF

Setup complete (without RoarGraph).

  Activate:  conda activate ${ENV_NAME}
  Datasets:  bash scripts/download_data.sh
  Smoke run: make quick

Figure 3's RoarGraph curves need Linux/x86-64. Everything else runs here.

EOF
