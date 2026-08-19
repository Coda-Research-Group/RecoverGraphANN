# The third-party patches

Two of the three index backends carry local changes. Each is shipped three ways — as a pinned
submodule, as a standalone diff in `patches/`, and documented here — so that a reader can
inspect the instrumentation without cloning anything, and so the change survives a fork being
renamed or deleted.

`make check-patches` regenerates each diff from the commit its submodule is pinned at and
fails on any byte difference, so the copies cannot drift apart.

| Submodule | Upstream | Base commit | Local change |
|---|---|---|---|
| `third_party/hnswlib` | [nmslib/hnswlib](https://github.com/nmslib/hnswlib) | `3f342966` (v0.8.0) | build-time instrumentation |
| `third_party/RoarGraph` | [matchyc/RoarGraph](https://github.com/matchyc/RoarGraph) | `78bf05cf` | Python bindings, portable build |
| `third_party/flatnav` | [BlaiseMuhirwa/flatnav](https://github.com/BlaiseMuhirwa/flatnav) | `a5383a44` (`v0.1.2-rc1`) | none to the source; two build patches applied at install time, see below |

---

## hnswlib — instrumenting the neighbor-selection heuristic

`patches/hnswlib-instrumentation.patch`, against `nmslib/hnswlib@3f342966` — the commit
upstream's `v0.8.0` tag points at. Four files, +296/-31, in two commits: `get_all_links`
alone, then the instrumentation and the `enable_pruning` switch together (they share the
`addPoint` signature, so splitting them further yields a commit that does not build).

This is the most reusable thing in the repository: it is what makes Table 1's `acc` column
possible, and no released ANN library reports it.

### The question it answers

HNSW does not connect a new point to the nearest candidates it found. It runs
`getNeighborsByHeuristic2`, which walks the candidate pool in increasing distance and keeps a
candidate only if it is closer to the query than to any already-selected neighbour. On
attention-derived data the first selected neighbour is the highest-inner-product candidate,
which is also the highest-norm one — and its large norm then makes almost every remaining
candidate look closer to *it* than to the inserted point. So they are all discarded.

Stock hnswlib gives you no way to see this happen. You can observe that recall is bad and
that the graph is sparse, but not that the heuristic is where the sparsity comes from. The
patch makes the pruning countable.

### What it adds

**`hnswlib/hnswlib.h`** — a `AddPointMetrics` struct with four counters, plus `reset()` and
`add()` so per-thread instances can be merged:

| Field | Meaning |
|---|---|
| `candidates_visited` | nodes popped during the build-time greedy search |
| `candidates_pruned` | candidates the heuristic rejected |
| `mutual_connect_sizes` | per `mutuallyConnectNewElement` call: *(candidate pool size on entry, selected-neighbour count)* |
| `heuristic_good_not_good_counts` | per `getNeighborsByHeuristic2` call: *(kept, discarded)* |

**Table 1's `acc` is the mean of the second element of `mutual_connect_sizes`**, and its
companion `candidate_pool_avg` is the mean of the first. The pool sits at ~499 out of
`efConstruction = 500` in every configuration, which is what makes `acc` interpretable: the
heuristic sees the same number of candidates either way, and the transformation changes only
how many survive.

**`hnswlib/hnswalg.h`** — threads an optional `AddPointMetrics*` through `searchBaseLayer`,
`getNeighborsByHeuristic2`, `mutuallyConnectNewElement`, `updatePoint` and both `addPoint`
overloads, incrementing the counters at the points where candidates are visited and rejected.
The pointer defaults to `nullptr`, so an uninstrumented call path is unchanged.

**`hnswlib/bruteforce.h`** — `BruteforceSearch::addPoint` returns an empty `AddPointMetrics`,
purely so it still satisfies `AlgorithmInterface`. Brute force has no heuristic to measure.

**`python_bindings/bindings.cpp`** — two additions:

- `add_items` now returns
  `(candidates_visited, candidates_pruned, mutual_connect_sizes, heuristic_good_not_good_counts)`.
  **This is a breaking change to the upstream signature**, which returns nothing. Code written
  against stock hnswlib still runs; code that unpacks the return value does not work against
  stock hnswlib. `rgann.indexes.hnsw` checks for the fork explicitly rather than letting this
  fail obscurely.
- `Index.get_all_links()` returns `{label: [[level-0 neighbours], [level-1 neighbours], ...]}`
  for every live point, taking `appr_alg->global` while it traverses. **Table 1's `deg` is the
  mean over labels of the total neighbour count across all levels** — not the level-0 count,
  which is roughly 0.4 lower on these datasets.

### The `enable_pruning` switch

The same patch adds `enable_pruning` (default `true`) to `addPoint` and `add_items`. Setting
it to `false` skips `getNeighborsByHeuristic2` and connects the raw candidates.

This is not an experiment the paper reports, but it is how the diagnosis was confirmed: with
pruning off, `acc` rises to `M`, `deg` rises with it, and recall improves sharply on raw data
— which is what pins the failure on the heuristic rather than on the data alone.

Measured on a 10 000-row prefix of `yi-128-ip`, raw space, `M = 48`, `efConstruction = 500`,
`efSearch = 1000`, k = 10, single-threaded:

| | `acc` | `deg` | Recall@10 |
|---|---:|---:|---:|
| pruning on | 2.62 | 4.69 | 67.58% |
| **pruning off** | **47.76** | **60.17** | **89.90%** |

`acc` goes to 47.76 against `M = 48`: with the heuristic disabled the graph keeps essentially
every candidate it is offered, so what the raw-space graph lacks is not candidates but
acceptances. Reproduce it with `enable_pruning: False` in the build parameters. (A 10k prefix
is an easier problem than the full database, so the recall column is higher throughout than
Table 1's — the comparison is between the two rows, not against the paper.)

### Reusing it on its own

The patch is independent of everything else here:

```shell
git clone https://github.com/nmslib/hnswlib && cd hnswlib
git checkout 3f342966
git apply /path/to/patches/hnswlib-instrumentation.patch
pip install .
```

---

## RoarGraph — bindings and a portable build

`patches/roargraph-pyroar-bindings.patch`, against `matchyc/RoarGraph@78bf05cf`.
Six files, +355/-389 — it deletes more than it adds.

1. **`pyroar/` — Python bindings.** Upstream RoarGraph is a set of command-line tools that
   read and write binary files, so running it under the same harness as the other three
   indexes would mean staging every dataset through disk. `pyroar` is a pybind11 extension
   exposing build and search over numpy arrays directly.
2. **In-memory entry points on `IndexBipartite`** — `BuildRoarGraphwithData`,
   `SetLearnBaseKNN`, `SearchRoarGraphPy` — beside the file-based ones, which are untouched.
3. **A build that runs on the paper's CPU** — see below.

### Why the build had to change

Upstream's `DistanceL2::compare` and `DistanceInnerProduct::compare` end in an unconditional
AVX-512 block (`_mm512_loadu_ps`). The paper's Intel Xeon E5-2620 is Sandy Bridge: it has AVX,
but neither AVX-512 nor AVX2, so upstream's kernels cannot run there at all. The patch
replaces them with plain AVX (`_mm256_*`) loops, relaxes `-march=native`, and links a generic
FAISS so the extension builds across heterogeneous hosts.

The commit is pinned exactly rather than tracked as a branch, because this is the build the
published measurements were taken on and the artifact's job is to run that build.

### Kept deliberately small

An earlier version of this branch carried 718 lines of commented-out code and two functions
nothing called. They are gone, which is why the patch now deletes more than it adds. The cut
was verified rather than assumed: all four RoarGraph curves of Figure 3 were re-measured
against the rebuilt extension, and all 64 rows came back identical in `recall_avg`,
`avg_node_degree` and `accepted_candidates_avg`.

---

## FlatNav — unmodified source, two patched build flags

`third_party/flatnav` is pinned at `a5383a44`, which is exactly upstream's `v0.1.2-rc1` tag
(verify: `gh api repos/BlaiseMuhirwa/flatnav/git/ref/tags/v0.1.2-rc1`). The submodule points
at `Coda-Research-Group/flatnav`, a fork holding that identical commit — kept for durability,
not for changes, since FlatNav is the only backend whose source has no other copy here. See
[`third_party/README.md`](../third_party/README.md).

**The checked-out source is upstream's; the build is not.** Two patches are applied by
`scripts/install_flatnav.sh` at install time, each conditional on the host, and both touching
only `python-bindings/setup.py`:

| patch | applied when | what it changes |
|---|---|---|
| `flatnav-scalar-build.patch` | `/proc/cpuinfo` lacks `avx2`/`fma` | adds a `NO_SIMD_VECTORIZATION` escape hatch, and pins `-march` to the paper's baseline |
| `flatnav-macos-arch.patch` | `uname -s` is `Darwin` | builds for `platform.machine()` instead of a hard-coded `x86_64` |

Neither touches the algorithm — they decide which instructions get emitted and for which
architecture. Both are hand-written against upstream's own `setup.py` rather than being diffs
between two commits, which is why `make check-patches` does not cover them: there is no pair
of commits to regenerate them from. They also leave the submodule's working tree dirty while
installed; `install_flatnav.sh` resets it to the pinned state on each run.

`patches/flatnav-macos-arch.patch`: upstream's `setup.py` hard-codes
`-arch x86_64` for macOS, which was true of every Mac when `v0.1.2-rc1` was tagged. On Apple
Silicon the compiler then targets x86_64 while scikit-build sets `CMAKE_OSX_ARCHITECTURES` to
`arm64`, and CMake refuses to configure at all:

```
The CXX compiler targets architectures "x86_64;arm64"
but CMAKE_OSX_ARCHITECTURES is "arm64"
```

The patch substitutes `platform.machine()`, so the build follows the host. It is applied only
on Darwin, and only when not already present, so re-runs are safe.

`patches/flatnav-scalar-build.patch`: FlatNav's build unconditionally passes `-march=native` and AVX2 flags, producing a
binary that dies with `SIGILL` on a CPU without AVX2+FMA — which the paper's Intel Xeon
E5-2620 is. The patch adds a `NO_SIMD_VECTORIZATION` escape hatch;
`scripts/install_flatnav.sh` applies it automatically when `/proc/cpuinfo` lacks the flags.

This affects throughput, not recall, and is recorded per run so a QPS comparison can account
for it.
