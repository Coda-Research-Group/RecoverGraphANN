# Raw run logs

The per-run metrics CSVs the paper's numbers were originally read out of, copied verbatim from
the machine that produced them. 46 files, May–June 2026.

They are committed as evidence, not as inputs: nothing in this repository reads them except
`tests/test_paper_numbers_trace.py`, which checks that every cell of Table 1 still resolves to
a row here. The reproducible pipeline writes to `results/*.csv` instead.

## Table 1's source

All 16 cells come from **`2026-05-26_09-14-58-index-metrics.csv`** (138 rows), filtered to

    algorithm = hnsw-hnswlib
    use_hnsw_pruning_rule_on_X = True
    hnsw_ef_construction = 500
    hnsw_ef_search = 1000
    hnsw_learn_insert_order = none

with these columns:

| Table 1 | column |
|---|---|
| `acc` | `selected_neighbors_avg_{X,X_asymmetric}` |
| `deg` | `degree_avg_{X,X_asymmetric}` |
| Recall@10 | `recall_{X,X_asymmetric}_avg_at_k` |

`degree_lowest_layer_avg_*` is the level-0 degree and reproduces none of the printed cells.

## Two things to know before reading these files

**`2026-06-03_10-26-36-index-metrics.csv` is mislabelled.** All 32 of its rows carry
`use_asymmetric_transformation=True`, but they are the **L2-normalization ablation**, not the
Bachrach transformation. The transform was edited in place rather than selected by a flag, so
the column records which branch of the code ran, not which transformation. Its `M=48`,
`efSearch=1000` rows are the source of §6's 13.7/27.5 with 80.2% recall on `yi-128-ip` and
21.5/41.7 with 47.3% on `llama-128-ip`.

Any query over these logs that filters on `use_asymmetric_transformation` must exclude this
file, or it will mix two different transformations. This is exactly what
`--normalization {none,bachrach,l2}` exists to prevent, and why
`experiments/ablation_l2_normalization.py` regenerates the ablation rather than citing these
rows.

**Every file has `git_dirty=True`.** The harness recorded the working tree as dirty on every
run, because experiment selection was done by editing `main()`. The `git_commit` column is
therefore a rough locator, not an exact one — which is the other reason the artifact replaces
that workflow with named entry points.

## Provenance columns

Each row carries `run_timestamp_utc`, `git_commit`, `git_dirty` and `hostname`. Every run was
single-threaded (`num_threads_build = num_threads_search = 1`) on the reference machine, an
Intel Xeon E5-2620 @ 2.00 GHz. The `hostname` column reads `reference-machine`: the real name
identifies private infrastructure and says nothing a reader can use.
