# Pinned third-party sources

Three index implementations, each pinned to an exact commit rather than a branch tip. Two of
them carry local changes; the third does not.

| Directory | Repository | Pinned commit | Local changes |
|---|---|---|---|
| `hnswlib/` | Coda-Research-Group/hnswlib, branch `RecoverGraphANN` | `f9739395` | build-time instrumentation, off `nmslib/hnswlib` `v0.8.0` (`3f342966`) |
| `RoarGraph/` | Coda-Research-Group/RoarGraph, branch `RecoverGraphANN` | `c2145e04` | Python bindings and a portable build, off `matchyc/RoarGraph@78bf05cf` |
| `flatnav/` | Coda-Research-Group/flatnav, branch `RecoverGraphANN` | `a5383a44` (`v0.1.2-rc1`) | none to the source; two build patches at install time |

Branch tips rather than tags for the two forks, because a tag can be moved and a branch in
this repository's own organisation cannot drift without someone noticing. The submodule
pointer is a commit either way.

## FlatNav is unmodified, and forked anyway

`a5383a44` is exactly what upstream's `v0.1.2-rc1` tag points at. Check it yourself:

```shell
gh api repos/BlaiseMuhirwa/flatnav/git/ref/tags/v0.1.2-rc1 --jq .object.sha
```

`Coda-Research-Group/flatnav` is a fork of that repository holding the identical commit, and
the submodule points at the fork. Not because the source was changed — it is byte-identical to
upstream — but because FlatNav is the one backend with **no copy of its source anywhere else
in this artifact**. hnswlib and RoarGraph survive their upstreams disappearing, since `patches/`
carries their complete diffs. FlatNav had no such fallback: `a5383a44` is a pre-1.0 release
candidate, exactly the kind of tag that gets tidied away, and if it went the backend would be
unbuildable.

The `RecoverGraphANN` branch in the fork exists only to keep that commit reachable from a
ref this organisation controls, rather than from a tag someone else can move.

The source is untouched; the build is patched twice, by `scripts/install_flatnav.sh` at
install time:

- `patches/flatnav-scalar-build.patch`, on CPUs without AVX2+FMA — which includes the paper's
  own machine, so the published numbers come from this build
- `patches/flatnav-macos-arch.patch`, on macOS, so the extension is compiled for the host
  architecture rather than upstream's hard-coded `x86_64`

Both change which instructions get emitted and for which architecture, not what the algorithm
does. Neither is covered by `make check-patches`: they are written against upstream's own
`setup.py`, so there is no pair of commits to regenerate them from. They are applied to the
working tree, which `install_flatnav.sh` resets to the pinned state on each run.

## Why the patches are also kept outside this directory

`patches/` carries each fork's diff against its named upstream commit, and
[`docs/PATCHES.md`](../docs/PATCHES.md) documents them hunk by hunk. That duplication is
deliberate: a submodule pointer is useless if the fork is renamed or deleted, submodules do
not travel inside a source archive, and a reader should be able to see what was instrumented
without cloning anything.

`make check-patches` regenerates each diff from the pinned commit and fails on a byte
difference, so the two copies cannot drift apart.

## Getting them

```shell
git submodule update --init --recursive
```

RoarGraph has submodules of its own (`robin-map`, `DiskANN`), which is why `--recursive`
matters — without it the build fails on a missing `tsl/robin_map.h`.
