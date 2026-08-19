#!/usr/bin/env bash
# One-command setup on a freshly installed Ubuntu 24.04 LTS.
#
#   bash scripts/setup_ubuntu.sh
#
# Installs build tools, Miniforge, the `rgann` conda environment, and the three native index
# backends (hnswlib, FlatNav, RoarGraph) from the pinned submodules under third_party/.
#
# Needs sudo for apt only. Set SKIP_APT=1 if the packages are already present.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${RGANN_ENV:-rgann}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniforge3}"

log() { printf '\n==> %s\n' "$1"; }

# --- 1. System packages ------------------------------------------------------------------
# libboost is RoarGraph's; libomp/OpenMP is used by every backend; cmake<4 is FlatNav's.
if [ "${SKIP_APT:-0}" != "1" ]; then
  log "Installing system packages (sudo required)"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    build-essential g++ cmake git curl ca-certificates \
    libomp-dev libboost-all-dev
fi

# --- 2. Miniforge ------------------------------------------------------------------------
if [ ! -d "${CONDA_ROOT}" ]; then
  log "Installing Miniforge into ${CONDA_ROOT}"
  installer="$(mktemp -t miniforge.XXXXXX.sh)"
  curl --fail --location --output "${installer}" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "${installer}" -b -p "${CONDA_ROOT}"
  rm -f "${installer}"
else
  log "Miniforge already present at ${CONDA_ROOT}"
fi

# conda is not on PATH in a non-interactive shell; call it by absolute path throughout.
CONDA="${CONDA_ROOT}/bin/conda"

# --- 3. Environment ----------------------------------------------------------------------
if "${CONDA}" env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  log "Updating conda environment '${ENV_NAME}'"
  "${CONDA}" env update -n "${ENV_NAME}" -f "${ROOT}/environment.yml" --prune
else
  log "Creating conda environment '${ENV_NAME}'"
  "${CONDA}" env create -n "${ENV_NAME}" -f "${ROOT}/environment.yml"
fi

run_in_env() { "${CONDA}" run -n "${ENV_NAME}" --no-capture-output "$@"; }

# --- 4. Pinned third-party sources -------------------------------------------------------
log "Checking out submodules at their pinned commits"
git -C "${ROOT}" submodule update --init --recursive

# --- 5. Native backends ------------------------------------------------------------------
log "Building hnswlib (instrumented fork)"
run_in_env pip install --no-cache-dir "${ROOT}/third_party/hnswlib"

log "Building FlatNav"
bash "${ROOT}/scripts/install_flatnav.sh"

log "Building RoarGraph"
run_in_env pip install --no-cache-dir pybind11
run_in_env pip install --no-build-isolation -e "${ROOT}/third_party/RoarGraph/pyroar"

log "Installing rgann"
run_in_env pip install --no-cache-dir -e "${ROOT}"

# --- 6. Verify ---------------------------------------------------------------------------
log "Verifying backends"
run_in_env python - <<'PY'
import faiss
import flatnav
import hnswlib
from RoarGraph import IndexRoarGraph, Metric

import rgann

print(f'faiss     {faiss.__version__}')
print(f'flatnav   {flatnav.__version__}')
print('hnswlib   ok (instrumented build)' if hasattr(hnswlib.Index, 'get_all_links')
      else 'hnswlib   WRONG BUILD: get_all_links missing, Table 1 cannot be produced')
print('RoarGraph ok')
print(f'rgann     {rgann.__version__}')
PY

cat <<EOF

Setup complete.

Miniforge was installed in batch mode, which deliberately does not touch your shell
configuration, so 'conda' is NOT on your PATH yet and 'conda activate' will not be found.
Pick one:

  This shell only:  export PATH="${CONDA_ROOT}/bin:\$PATH"
  Permanently:      ${CONDA} init bash     # then open a new shell

Then:

  Activate:  conda activate ${ENV_NAME}      (or prefix with: ${CONDA} run -n ${ENV_NAME})
  Datasets:  bash scripts/download_data.sh
  Smoke run: make quick

EOF
