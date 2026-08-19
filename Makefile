# Entry points, named after the paper artifacts they produce.
#
#   make quick      ~5 min   all four backends on a 10k subsample; proves the pipeline works
#                            (measured: 324 s in Docker on the paper's Xeon E5-2620)
#   make figures    seconds  regenerate Figures 1-3 from the committed CSVs, no index builds
#   make table      seconds  regenerate Table 1 from the committed CSVs
#   make ablation   ~30 min  re-measure the section 6 L2-normalization ablation
#   make all        ~3.5 h   the canonical run: every paper number, from scratch
#   make test       seconds  unit tests
#   make check      seconds  lint + patch-drift guard + tests
#   make release-check       is everything the README promises actually present?
#
# Everything runs inside the conda environment; override with CONDA_ENV=... if you renamed it.

CONDA_ENV ?= rgann
CONDA     ?= conda
RUN       := $(CONDA) run -n $(CONDA_ENV) --no-capture-output
PYTHON    := $(RUN) python

RESULTS := results
FIGURES := figures

.PHONY: help quick figures table ablation all test check lint check-patches release-check clean docker docker-quick

help:
	@awk '/^# Entry points/,/^$$/ { sub(/^#[[:space:]]?/, ""); print }' Makefile

# --- Paper artifacts ----------------------------------------------------------------------

$(FIGURES)/norm_distribution_panels.pdf:
	$(PYTHON) experiments/fig1_norm_distributions.py --output $(FIGURES)

$(FIGURES)/inner_product_distribution_panels_raw.pdf:
	$(PYTHON) experiments/fig2_inner_product_distributions.py --output $(FIGURES)

$(FIGURES)/efsearch_recall_qps_2x2.pdf: $(RESULTS)/fig3_recall_qps.csv
	$(PYTHON) experiments/fig3_recall_qps.py --csv $< --output $(FIGURES)

figures: $(FIGURES)/norm_distribution_panels.pdf \
         $(FIGURES)/inner_product_distribution_panels_raw.pdf \
         $(FIGURES)/efsearch_recall_qps_2x2.pdf

# Not a prerequisite on the CSV: if it is missing, the script's own error names the command
# that produces it, which is more use than make's "No rule to make target".
table:
	$(PYTHON) experiments/table1_degree.py --render \
		--csv $(RESULTS)/table1_degree.csv --output $(RESULTS)/table1_degree.tex

# The one artifact that has to be measured rather than rendered: §6 compares the paper's
# transformation against plain L2 normalization, so it builds indexes in both spaces.
ablation:
	$(PYTHON) experiments/ablation_l2_normalization.py \
		--csv $(RESULTS)/ablation_l2_normalization.csv

# --- Running the experiments --------------------------------------------------------------

quick:
	$(PYTHON) experiments/table1_degree.py --quick
	$(PYTHON) experiments/fig3_recall_qps.py --quick
	@echo
	@echo "Quick run complete. These numbers are from a 10k subsample and are NOT the paper's."

all:
	bash scripts/run_all_experiments.sh

# --- Checks --------------------------------------------------------------------------------

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

check-patches:
	$(PYTHON) scripts/check_patches.py

check: lint check-patches test

# Not part of `check`: this is expected to report gaps for most of the artifact's life.
# It has to come back clean before tagging a release or minting a DOI.
release-check:
	$(PYTHON) scripts/check_release_ready.py

# --- Docker ---------------------------------------------------------------------------------
# ISA=native on a modern CPU; ISA=scalar on anything without AVX2+FMA (the paper's machine).

ISA ?= native

docker:
	docker build --build-arg ISA=$(ISA) -t recovergraphann:latest .

docker-quick: docker
	docker run --rm -v "$(PWD)/data:/app/data" recovergraphann:latest make quick

clean:
	rm -rf $(RESULTS)/quick $(FIGURES)/quick
	rm -rf .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
