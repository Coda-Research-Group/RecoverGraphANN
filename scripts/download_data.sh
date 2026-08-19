#!/usr/bin/env bash
# Download the two VIBE attention datasets used in the paper and verify their checksums.
#
#   bash scripts/download_data.sh                 # both datasets
#   bash scripts/download_data.sh yi-128-ip       # just one
#
# ~635 MB total. Re-running skips files that already verify, so it is safe to resume.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${ROOT}/data"
MANIFEST="${DATA_DIR}/MANIFEST.sha256"

# Pinned revision, not `main`: the dataset repo is still being updated upstream.
REVISION="c8723b8f8c2ac64afb181aa1f4fd5f078bfca286"
BASE_URL="https://huggingface.co/datasets/vector-index-bench/vibe/resolve/${REVISION}"

ALL_DATASETS=("yi-128-ip" "llama-128-ip")

if [ "$#" -gt 0 ]; then
  datasets=("$@")
else
  datasets=("${ALL_DATASETS[@]}")
fi

if command -v sha256sum >/dev/null 2>&1; then
  checksum_of() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  # macOS ships shasum rather than sha256sum.
  checksum_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  echo "error: neither sha256sum nor shasum found" >&2
  exit 1
fi

expected_for() {
  local name="$1"
  awk -v file="${name}.hdf5" '$2 == file { print $1 }' "${MANIFEST}"
}

mkdir -p "${DATA_DIR}"

for name in "${datasets[@]}"; do
  target="${DATA_DIR}/${name}.hdf5"
  expected="$(expected_for "${name}")"

  if [ -z "${expected}" ]; then
    echo "error: ${name}.hdf5 is not listed in ${MANIFEST}" >&2
    exit 1
  fi

  if [ -f "${target}" ] && [ "$(checksum_of "${target}")" = "${expected}" ]; then
    echo "ok       ${name}.hdf5 (already downloaded, checksum matches)"
    continue
  fi

  echo "download ${name}.hdf5"
  curl --fail --location --progress-bar --output "${target}.part" "${BASE_URL}/${name}.hdf5"

  actual="$(checksum_of "${target}.part")"
  if [ "${actual}" != "${expected}" ]; then
    echo "error: checksum mismatch for ${name}.hdf5" >&2
    echo "  expected ${expected}" >&2
    echo "  actual   ${actual}" >&2
    echo "  the download is corrupt, or upstream changed the file. Not installing it." >&2
    rm -f "${target}.part"
    exit 1
  fi

  mv "${target}.part" "${target}"
  echo "ok       ${name}.hdf5"
done

echo
echo "Datasets are in ${DATA_DIR}."
