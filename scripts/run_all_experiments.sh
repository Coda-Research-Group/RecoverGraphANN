#!/usr/bin/env bash
# Run every timed experiment in the paper and collect the timings in one file.
#
#   bash scripts/run_all_experiments.sh            # everything, 1-2 days
#   bash scripts/run_all_experiments.sh --quick    # 10k subsample, minutes
#   SKIP_ROARGRAPH=1 bash scripts/run_all_experiments.sh   # macOS, or no x86-64
#
# Resumable: an artifact whose results CSV already exists is skipped, so an interrupted run
# continues where it stopped. Delete the CSV to force a rerun.
#
# Reviewer timings are not expected to match ours — see the hardware note in the README.
# Every stage appends one line to results/timings.csv so the comparison is a single diff.
#
# RGANN_HOSTNAME sets the `hostname` recorded in every results row, and in this file's own
# hostname column. Leave it unset to record the machine's own name; set it when the results
# are going to be published from a machine whose name should not be.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${RGANN_ENV:-rgann}"
RESULTS="${ROOT}/results"
TIMINGS="${RESULTS}/timings.csv"
FIGURES="${ROOT}/figures"

# The stage runs from ROOT, so the commands take repo-relative paths. That keeps the
# command recorded in timings.csv runnable by whoever reads the file, and keeps one
# machine's directory layout out of a published result.
REL_RESULTS="results"
REL_FIGURES="figures"

QUICK_FLAG=""
if [ "${1:-}" = "--quick" ]; then
  QUICK_FLAG="--quick"
fi

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

mkdir -p "${RESULTS}" "${FIGURES}"

if [ ! -f "${TIMINGS}" ]; then
  echo "artifact,command,started_utc,finished_utc,wall_seconds,exit_code,hostname" >"${TIMINGS}"
fi

# Run one stage, skip it if its output already exists, and record how long it took.
stage() {
  local artifact="$1" output="$2"
  shift 2

  if [ -f "${output}" ]; then
    echo "skip     ${artifact} (${output} exists; delete it to rerun)"
    return 0
  fi

  local started finished start_epoch end_epoch status
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_epoch="$(date +%s)"

  echo
  echo "==> ${artifact}"
  set +e
  "${CONDA}" run -n "${ENV_NAME}" --no-capture-output "$@"
  status=$?
  set -e

  end_epoch="$(date +%s)"
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s,"%s",%s,%s,%s,%s,%s\n' \
    "${artifact}" "$*" "${started}" "${finished}" \
    "$((end_epoch - start_epoch))" "${status}" "${RGANN_HOSTNAME:-$(hostname)}" >>"${TIMINGS}"

  if [ "${status}" -ne 0 ]; then
    echo "error: ${artifact} failed with exit ${status}; see ${TIMINGS}" >&2
    exit "${status}"
  fi
}

cd "${ROOT}"

# Figures 1 and 2 read the HDF5 files directly and take seconds; they come first so an
# obvious data problem surfaces before a multi-hour build.
stage "figure1-norms" "${FIGURES}/norm_distribution_panels.pdf" \
  python experiments/fig1_norm_distributions.py --output "${REL_FIGURES}"

stage "figure2-inner-products" "${FIGURES}/inner_product_distribution_panels_raw.pdf" \
  python experiments/fig2_inner_product_distributions.py --output "${REL_FIGURES}"

stage "table1-degree" "${RESULTS}/table1_degree.csv" \
  python experiments/table1_degree.py --csv "${REL_RESULTS}/table1_degree.csv" ${QUICK_FLAG}

stage "ablation-l2" "${RESULTS}/ablation_l2_normalization.csv" \
  python experiments/ablation_l2_normalization.py --csv "${REL_RESULTS}/ablation_l2_normalization.csv" ${QUICK_FLAG}

FIG3_BACKENDS="hnsw-hnswlib nsg-faiss flatnav roargraph"
if [ "${SKIP_ROARGRAPH:-0}" = "1" ]; then
  FIG3_BACKENDS="hnsw-hnswlib nsg-faiss flatnav"
  echo "note: RoarGraph skipped by SKIP_ROARGRAPH=1; Figure 3 will have no RoarGraph curve"
fi

# shellcheck disable=SC2086  # the backend list and quick flag are intentionally word-split
stage "figure3-recall-qps" "${RESULTS}/fig3_recall_qps.csv" \
  python experiments/fig3_recall_qps.py --csv "${REL_RESULTS}/fig3_recall_qps.csv" \
  --output "${REL_FIGURES}" --backends ${FIG3_BACKENDS} ${QUICK_FLAG}

stage "table1-render" "${RESULTS}/table1_degree.tex" \
  python experiments/table1_degree.py --render --csv "${REL_RESULTS}/table1_degree.csv" \
  --output "${REL_RESULTS}/table1_degree.tex"

echo
echo "All artifacts written. Timings: ${TIMINGS}"
column -s, -t "${TIMINGS}" 2>/dev/null || cat "${TIMINGS}"
