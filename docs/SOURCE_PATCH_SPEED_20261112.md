# Source Patch Speed Experiments

## Contract

Opt-in native Rust experiments on top of `9fa2aefe5de10f64e22848aabb1b6be497bb8f3f`
and the uncommitted QLIC block-context work. Normal encoder defaults, JPEG
recompression, decoder semantics and Python ABI are unchanged. No new bitstream
features are introduced. Verification compares decoded original pixels,
including hidden RGB and extra channels, not compressed-byte equality.

Source Add Patch encodes a signed reference atlas followed by a signed residual
and a standard Patch Dictionary. The reference and residual have different
training domains; replacing both learners is not equivalent to accelerating
patch detection. The detector may return no plan. In that case the reference
learner is unused and residual options apply to the ordinary frame.

## Routes

All routes retain the same e9 transforms and entropy backend unless a listed
option changes its MA learner or gathering. This is the native projected
candidate, not an outer multi-candidate scheduler or an official libjxl result.

- `baseline`: native e9 without patches.
- `source`: existing Source Add Patch, unchanged learner configuration.
- `source-ref8/ref16`: block-local MA only for the reference atlas.
- `source-res8/res16`: block-local MA only for the residual (or no-plan frame).
- `source-both8/both16`: block-local MA for both parts.
- `baseline-parallel/source-parallel`: existing sparse parallel MA gathering;
  no replacement of the usual MA split search.
- `source-ref16-parallel`: atlas block tree plus parallel gathering in the
  ordinary residual learner. The other `*-parallel` hybrids are also available.

Block-local scoring now forks independent subplans of at least 65,536 samples
inside the caller's existing pool. Histogram merging, rate comparisons and tie
order remain unchanged. Small subplans execute serially. This is a work-grain
threshold, not a new leaf or image limit. At one worker it remains serial.

Parallel gathering uses `GatherOptions` with at most the requested workers
(capped by its existing 64-worker limit), sparse property calculation and
unchanged sample probability. No stripe approximation or lower effort is used.
Worker-local deduplication can change the training record layout and final tree;
file size is measured, not assumed unchanged. It can also use more memory.

## Reproduction

```powershell
cargo build --release -p rs-jxl-cli --example patch_lab -j 4
target/release/examples/patch_lab.exe input.png --output source-parallel.jxl --mode source --effort 9 --threads 8 --parallel-gather
target/release/examples/patch_lab.exe input.png --output hybrid.jxl --mode source --effort 9 --threads 8 --block-size 16 --block-target reference --parallel-gather
```

`--block-target` requires `--block-size`. Its default is `residual`; `off` mode
allows only that target. Separate reference/residual receipts distinguish
configuration from actual learner invocation. They count all learner calls,
including losing backend candidates. Outputs are create-new only.

To inspect inclusive stage times, set `RS_JXL_PROFILE=1` and pass `--profile`.
`encode.patch_reference`, `encode.patch_residual`, `encode.patch_analysis`,
`patch.read_planes`, `patch.assemble` and `patch.dictionary` isolate the patch
stages. Nested/parallel profile totals overlap; do not add them as wall time.
Timing benchmarks scrub inherited profiling/experiment environment variables.

The plotter consumes a complete verified report without rerunning compression:

```powershell
python -B tools/plot_patch_tradeoffs.py out/source-patch-speed-20261112/focused5.json --out out/source-patch-speed-20261112/tradeoffs.png --svg --per-case --pareto
```

Each point is one experiment. X is native encoding time (log scale by default),
Y is bpp. The corpus uses sums of per-image median times and
`8 * sum(encoded bytes) / sum(original pixels)`, not the mean of individual bpp.
Startup, PNG input/output and oracle decoding are excluded. Per-case panels and
an audit CSV preserve regressions hidden by aggregation. PNG and SVG are both
available; `--x-scale linear` changes only presentation. Reports with failed,
missing or unverified trials are rejected rather than silently intersected.

## Measured Results

Report: `out/source-patch-speed-20261112/focused5.json`. Five targeted inputs:
paper `9bde`, text screenshot `7dba`, editor `3dd`, `image.png`, and
`test_img/png/wallhaven-49yekn.png`. Total: 7,759,446 pixels. Eleven routes,
three interleaved repetitions, e9, eight workers: 165 official pixel checks,
all passed. This is a focused screenshot experiment, not a broad-corpus claim.
Encoder executable SHA256:
`ebc4fa2389889cc9e3a5681696eacc61bcfc278346430543aa5d24dfd0d2e7c9`.

| Route | Total bytes | bpp | Native seconds | MPix/s |
|---|---:|---:|---:|---:|
| baseline | 1,653,606 | 1.704870 | 3.4989 | 2.2177 |
| baseline-parallel | 1,653,606 | 1.704870 | 3.6398 | 2.1318 |
| source | 1,454,394 | 1.499482 | 9.5437 | 0.8130 |
| source-parallel | 1,454,394 | 1.499482 | 8.3256 | 0.9320 |
| source-ref8 | 1,474,309 | 1.520015 | 9.5706 | 0.8108 |
| source-ref16 | 1,473,886 | 1.519579 | 9.6668 | 0.8027 |
| source-res8 | 1,832,246 | 1.889048 | 3.8689 | 2.0056 |
| source-res16 | 1,832,484 | 1.889294 | 4.0267 | 1.9270 |
| source-both8 | 1,852,161 | 1.909581 | 3.6938 | 2.1006 |
| source-both16 | 1,851,976 | 1.909390 | 3.8861 | 1.9967 |
| source-ref16-parallel | 1,473,886 | 1.519579 | 8.1210 | 0.9555 |

Source parallel gathering reduced aggregate time by **12.76%**, with unchanged
file sizes on all five inputs. It does not imply parallel gathering always wins:
the no-patch baseline got 4.03% slower in this run. Individual small timings
also varied; the report retains min/max and every trial.

The reference16/parallel hybrid is only 2.46% faster than source-parallel while
using 1.34% more bytes. The both16 hybrid is 27.34% larger than original Source
Patch, and is both larger and slower than the no-patch baseline in aggregate.
Do not promote these hybrids to defaults. On `49yekn` no atlas was produced:
reference block receipts correctly record `configured=true, used=false`.

Per-image source comparison (the size is unchanged with parallel gathering):

| Input | Bytes | bpp | Source seconds | Parallel seconds |
|---|---:|---:|---:|---:|
| Paper9bde | 283,434 | 1.557193 | 3.541421 | 3.035899 |
| Text7dba | 9,519 | 0.148096 | 0.275360 | 0.269794 |
| Editor3dd | 106,886 | 0.400033 | 1.433265 | 1.351161 |
| image.png | 347,113 | 1.759795 | 3.325092 | 2.778269 |
| wallhaven-49yekn | 707,442 | 2.729329 | 0.968583 | 0.890464 |

Plots and audit data live beside the report: `tradeoffs.png/.svg/.csv` and
`tradeoffs_per_case.png/.svg`. The dashed Pareto frontier describes these
observed medians only; it is not a statistical significance guarantee.

### Profiling Follow-Up

Separate, non-trace diagnostic runs on the original Paper9bde PNG produced a
238x232 atlas and 1,107 patch positions. These profiled runs are not included
in the timing plot (the harness uses its common pixel-preserving PNG payload).

| Inclusive stage | Original source seconds | Parallel gather seconds |
|---|---:|---:|
| Patch analysis | 0.2672 | 0.2647 |
| Reference frame | 0.2381 | 0.2340 |
| Residual frame | 3.0548 | 2.4952 |
| MA gather, both frames | 1.6013 | 1.0244 |
| Worker gathering | 1.5992 | 0.1688 |
| Append worker samples | not used | 0.8536 |
| MA prepared training | 0.6959 | 0.7176 |

Parallel collection exposes a serial sample-append bottleneck; the residual
still dominates, not patch dictionary writing (~0.003 s). Investigating this
merge is a better next target than replacing the accurate residual learner
with the current four-predictor spatial tree. Profiles are stored as
`paper-profile-source.log` and `paper-profile-parallel.log`.

## Validation

- 17 block-context unit tests, including serial/parallel plan and receipt
  consistency, signed samples, ragged layouts and native roundtrips.
- 5 experiment CLI contract tests; 19 benchmark harness tests; 20 plot tests.
- Dedicated official-decoder regression: 64 main/extra-channel checks across
  8/16-bit RGBA, hidden RGB, signed atlas/residual and reference/residual/both
  block placement. Explicit reference receipts must activate when requested.
- Release experiment executable built locally. No Linux or macOS runtime
  claims are made by this Windows experiment.
