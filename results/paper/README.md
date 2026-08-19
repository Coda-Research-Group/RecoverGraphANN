# The paper's own CSVs

The two hand-assembled files the camera-ready was written from, copied verbatim from the
machine that produced them. They are **inputs to nothing**. They are here
so that a rerun has something to be diffed against, and so the paper's numbers remain
inspectable after the pipeline replaces them.

| File | Feeds |
|---|---|
| `efSearch.csv` (256 rows) | Figure 3, and the query-agnostic numbers quoted in §Results |
| `different-m.csv` (36 rows) | Table 1's Recall@10 column |

Their reproducible replacements are `results/fig3_recall_qps.csv` and
`results/table1_degree.csv`.

## How they were made

By hand, from the per-run logs in `results/logs/`. That is the problem this artifact exists to
fix: there is no script that turns those logs into these files, so the only record of which
rows were selected was the person who selected them.

## Reduced to the query-agnostic rows

As shipped, this file also carried a `type` column separating `query-agnostic` rows from
`ood-aware` ones — RoarGraph given the real query sample. This artifact evaluates the
query-agnostic setting only, so those 64 rows and the column are gone.
[VIBE](https://github.com/vector-index-bench/vibe) benchmarks RoarGraph with the query sample
and is the reference for that.

`different-m.csv` covers nine values of `M`; Table 1 uses four of them.
