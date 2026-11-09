# MA Strategy Experiments, 2026-11-09

## Scope and Decision

Three experiments follow the pixel-lossless speed work: model-aware MA split
scoring, cross-candidate training-sample reuse, and small-node batching. The
starting revision is `24a625944925c15a81c26b34023f99c95aa0a8f3`. These are Windows
Release, e9, eight-thread experiments on a Ryzen 9 7945HX, not JPEG reconstruction
or lossy encoding changes.

The useful result is histogram model regularization: several images get
both smaller and faster, but others grow. It remains an explicit development
option. Small-node batching was implemented and tested, then removed after
mixed results and a slowdown on the large case. No production scoring default,
effort, sample count, search space, resource ceiling, Python ABI, decoder, or
installed CLI option changes here.
Compressed JXL byte identity is not an acceptance criterion.

## Model Cost Scoring

The encoder already contains the optional `DescriptionCost` implementation in
`modular/ma_scoring.rs`. This experiment exposes it through the development-only
`encoder_lab` example and measures its effect rather than adding a second copy.
The histogram cost added to both the parent and candidate children is:

```text
C(histogram) = weight * 0.5 * (active_symbols - 1) * log2(total_weight + 1)
weight = 0, 0.25, 0.5, or 1
C = 0 when active_symbols <= 1
```

This is a model-complexity regularizer, not an exact estimate of clustered wire
bits. The existing split threshold still applies. Actual tree tokens, histogram
clustering, LZ77 and entropy coding can change the final ranking. In particular,
reducing model overhead can increase residual payload. It is not a completed
implementation of a clustering-aware final-rate objective.

`--model-costs` reads the per-encode `ModelCostReceipt` from the returned native
artifact, including tree bits and entropy-header bits. The native preparation
step creates a fresh collector; reading the original input options would report
the wrong collector. The receipt does not cover transforms, weighted-predictor
headers, palette data, residual payload or container framing.

### Timing and Size

Times below are medians of native encoder calls, excluding process startup, PNG
loading, output IO and verification, but including encoder/context setup. These
are not directly comparable to the earlier reusable NumPy/CFFI context timings.
PNG metadata is retained, also unlike that earlier NumPy input experiment.
MPix counts spatial pixels, not channels.

The 4K image has five measured runs plus one warmup per setting. Other rows use
three measured runs plus one warmup. The four-image initial screening used two
runs without warmup and is not sufficient for small timing claims. Each report
retains every timing, size, command and binary/input identity. Model settings
are interleaved; the later harness explicitly rotates the leading variant so
two-way A/B trials cannot accidentally run in a fixed order.

| Image | Setting | Bytes | bpp | Encode seconds | MPix/s | Size versus off |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wallhaven-mlzoy1, 3840x2160 | off | 14,179,178 | 13.6759 | 5.8734 | 1.412 | baseline |
| wallhaven-mlzoy1 | half | 14,086,041 | 13.5861 | 4.0697 | 2.038 | -0.657% |
| wallhaven-mlzoy1 | full | 14,100,844 | 13.6004 | 3.5668 | 2.325 | -0.552% |
| clover, 1024x1024 | off | 1,673,489 | 12.7677 | 1.3213 | 0.794 | baseline |
| clover | half | 1,639,976 | 12.5120 | 0.7549 | 1.389 | -2.003% |
| clover | full | 1,643,716 | 12.5406 | 0.6462 | 1.623 | -1.779% |
| wallhaven-49yekn, 1920x1080 | off | 707,442 | 2.7293 | 0.9975 | 2.079 | baseline |
| wallhaven-49yekn | half | 694,990 | 2.6813 | 0.8091 | 2.563 | -1.760% |
| wallhaven-49yekn | full | 696,855 | 2.6885 | 0.7644 | 2.713 | -1.497% |
| kodim19, 512x768 | off | 436,950 | 8.8898 | 0.7905 | 0.497 | baseline |
| kodim19 | half | 438,658 | 8.9245 | 0.7771 | 0.506 | +0.391% |
| kodim19 | full | 441,849 | 8.9894 | 0.6364 | 0.618 | +1.121% |
| Alpha icon, 512x512 | off | 34,061 | 1.0395 | 0.4209 | 0.623 | baseline |
| Alpha icon | half | 34,365 | 1.0487 | 0.4235 | 0.619 | +0.893% |
| Alpha icon | full | 34,765 | 1.0609 | 0.3915 | 0.670 | +2.067% |

The 4K full-cost result reduces time by 39.27%, with a smaller file, rather than
exchanging size for speed. Its measured ranges are 5.712-6.092 s off and
3.423-3.705 s full. Conversely, the Kodak and icon regressions rule out enabling
the same weight globally. The three-run clover timings are noisier: off spans
1.069-1.382 s and full 0.623-0.736 s, still a substantial local improvement.

Other coverage includes a webpage, PDF-reader screenshot, game UI, grayscale
image and a tiny 16-bit image. Full cost saves 1.519% on the webpage, but only
0.080% on the PDF; its initial PDF timing is slightly worse. Gray full cost
grows 0.377%. Do not treat tiny-image MPix/s as sustained codec throughput or
select an automatic policy from these eleven images.

### Why Some Images Improve

Diagnostic runs on the 4K image retain the same 135 groups and 16,219,035 total
distinct training records across those groups:

| Setting | Split-search calls | Total leaves | Tree bits | Entropy-header bits |
| --- | ---: | ---: | ---: | ---: |
| off | 148,263 | 74,199 | 1,082,431 | 569,505 |
| half | 18,547 | 9,341 | 155,998 | 318,970 |
| full | 10,325 | 5,230 | 100,845 | 266,173 |

The half setting saves about 147 KB of emitted model data but only 93,137 bytes
overall: the remaining stream grows and offsets part of that saving. This is
evidence of excessive low-payoff splitting on this case, not proof that fewer
leaves always mean a smaller file. For example, the webpage's full-cost file
shrinks even though its measured tree-plus-entropy-header total grows slightly.

Further work should calibrate costs for sampling fraction, group/shared-tree
scope and final clustering, then validate on a separate corpus. Do not encode
every weight at runtime and call the result a cheap pre-analysis policy.

Shared-tree and independent-group images have different outcomes in this small
matrix, but that does not establish a safe routing rule. In particular, image
size alone does not explain texture, transparency, entropy-header reuse or
clustering behavior.

## Cross Candidate Reuse

Default e9 has one final candidate: `SourceSearch::Disabled` in the CLI and FFI.
RCT and weighted-predictor selection happen before MA sampling. Evaluating
several transform/predictor costs is not the same as gathering and training a
new MA tree for each combination. The 25 and 135 gathers seen in these profiles
belong to distinct spatial groups, not duplicate candidates.

Relevant ownership boundaries:

- `modular/orchestration.rs::finish_stream` selects/applies transforms and WP.
- `modular/frame.rs` owns shared versus local group training.
- `modular/training.rs` prepares quantiles and sample columns.
- `modular/ma_prepared.rs` consumes mutable prepared samples.
- The optional scheduler `CandidateCache` retains immutable transformed layouts
  and WP headers; it is not a cross-candidate cache of mutable training samples.

An explicit outer candidate search could reuse raw training columns when only
later LZ77/entropy choices differ. A valid key would need the actual transformed
channel contents/layout, complete WP header, predictor/property ordering,
sampling domain and RNG/stream identity, quantization thresholds and channel
metadata. Matching image dimensions or a few public options is insufficient.
Each learner would still need a private permutation or mutable column copy;
caller sample limits must also remain enforced on a cache hit.

No new sample cache is added: there is no matching repeated work demonstrated
on the default route measured here. For an outer-search experiment, instrument
domain/schema/content fingerprints first and verify actual column equality on
prospective hits before estimating saved work versus copying and retained RAM.

## Small Node Availability

`RS_JXL_PROFILE=1` with `RS_JXL_PROFILE_MA_NODES=1` adds static record-count and
predictor-count buckets around the complete split-search call. It also records
local DFS frontier-width markers. Their durations are not useful; their call
counts measure availability. No per-node environment read, timing or queue scan
occurs when these diagnostics are disabled.

| Image | All split calls | Calls with at most 256 records | Fraction of inclusive split time in those nodes | DFS snapshots with at least 4 small ready nodes |
| --- | ---: | ---: | ---: | ---: |
| Tecnick | 20,169 | 10,972 | 18.1% | 2,179 |
| PDF | 19,701 | 14,824 | 19.1% | 2,577 |
| wallhaven-mlzoy1 | 148,263 | 71,641 | 15.1% | 7,849 |

These are distinct stored records, not summed sample weights. Inclusive parallel
scope times overlap and must not be summed into wall time. Frontier snapshots
can observe the same pending node repeatedly; they include 1-record terminal
nodes and do not prove cache/scratch eligibility. Width at least eight occurs
only 15, 9 and 36 times respectively. Nodes with at most 16 records account for
negligible time in these samples. Group-level parallelism already supplies work.

The rejected batch experiment evaluated 4-8 independent, unsearched 2-256-record
nodes and joined them before any sample/index mutation. DFS commit order, budget
decisions, predictor tie order and error delivery were retained. Cached nodes,
deferred histogram subtraction, detached subtrees, active dynamic frontiers,
one-thread scopes and explicit `parallel_nodes=false` were excluded. It did not
change the MA scoring objective or omit any search. The experimental controls,
deferred-error field and batch scheduler are no longer in the source tree.

Five measured runs plus one warmup per variant used the same frozen executable,
strictly alternating off/on order, no profiling and unchanged model-cost scoring
(`off`). File sizes were unchanged, and every output passed original-pixel
verification:

| Image | bpp | Batch off seconds | Batch on seconds | Time change | Off -> on MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| kodim19 | 8.8898 | 0.7677 | 0.7767 | +1.18% | 0.512 -> 0.506 |
| Tecnick | 8.5050 | 1.0840 | 0.9901 | -8.66% | 1.328 -> 1.454 |
| PDF | 2.2665 | 1.7315 | 1.7385 | +0.40% | 0.841 -> 0.838 |
| clover | 12.7677 | 1.3144 | 1.2566 | -4.40% | 0.798 -> 0.834 |
| wallhaven-mlzoy1 | 13.6759 | 5.1614 | 5.5674 | +7.87% | 1.607 -> 1.490 |

The sum of per-image median times rises 10.059 -> 10.329 seconds (+2.69%).
Diagnostic runs confirm 8, 521, 771 and 2,569 real batch dispatches on Kodak,
Tecnick, PDF and the 4K image respectively; this was not a disabled-path test.
Split-search call counts stayed unchanged. Tecnick's local improvement is worth
recording, but the aggregate result does not justify more permanent scheduling
and error-state complexity. Neither the small PDF difference nor a general
speedup is established by these noisy timing ranges.

Likely costs include repeated pending-stack scans, job/lock setup, additional
scratch leases and synchronization inside already-parallel group work. These
are implementation-level explanations, not isolated causal measurements. Any
future batch design should address those costs rather than just increase the
number of tasks.

## Focused Validation

- The experimental batch build passed 221 MA tests, including six new cases for
  disjointness, readiness, maximum batch width, deferred errors, caller budgets
  and unchanged DFS decisions. Those six batch-specific tests are archived
  alongside the rejected implementation.
- The retained source passed 215 MA tests after batch removal. The development
  example passed 21 parsing/adapter tests, and the measurement harness passed
  six tests including actual native timing/size validation and alternating order.
- The model and batch reports contain 209 independent official-decoder checks
  against original pixels, including warmups and profiling runs. The rebuilt
  final example and cleaned harness passed six more checks on RGB, transparent
  and 16-bit PNGs (`off`/`full`), for 215 successful checks in total.
- The final Release example was rebuilt after removing batching. The installed
  CLI and Python native library were not replaced by experiment executables.

Tests comparing a scheduling primitive's selected tree prove the intended
algorithmic equivalence of that primitive, not a requirement to preserve old
compressed bytes when changing the model-cost objective.

## Evidence and Reproduction

Development builds and reports are local under `out/ma-strategy-20261109/`:

- `encoder-lab.exe`: frozen model-cost/diagnostic build, before batch code.
- `node-profile.json`: off-cost node/frontier diagnostic coverage.
- `model-screen.json`: initial four-image/four-weight screen.
- `model-holdout.json`: seven additional images, three weights.
- `model-recheck.json`: five-run 4K confirmation.
- `model-profile.json`: half/full search counts and emitted model costs.
- `final-smoke.json`: final rebuilt example/harness, three inputs and two modes;
  this is correctness evidence, not another performance median.
- `encoder-lab-batch.exe`, `batch-profile.json`, `batch-timing.json`: frozen
  rejected batch build, activation evidence and five-run A/B measurements.
- `batch-experiment.patch`, `ma_small_batch_tests.rs`, `benchmark_ma_batch.py`:
  archived implementation/test/driver for that rejected experiment. The patch
  is relative to the starting revision and includes the retained diagnostics;
  it is not a patch to apply blindly to the current source. The archived driver
  needs the repository `tools` directory on its Python import path and the
  frozen batch binary. Its experimental environment switch is process-start
  configuration, not a current public CLI/Python option.

Every generated stream, including warmups, is decoded by the independent
official libjxl 0.12 `djxl` and compared with the original PNG integer samples,
including invisible RGB and 16-bit words. Generated images are temporary and
removed after the run. Hashes identify inputs and executables only. They never
constrain the compressed output or substitute for original-pixel comparisons.

```powershell
cargo build --release --offline -p rs-jxl-cli --example encoder_lab
target/release/examples/encoder_lab.exe INPUT.png --effort 9 --threads 8 --description-cost half --model-costs --output OUTPUT.jxl
python -B tools/benchmark_ma_models.py --encoder target/release/examples/encoder_lab.exe --decoder PATH/TO/djxl.exe --case INPUT.png --modes off half full --effort 9 --threads 8 --repeats 5 --warmups 1 --out out/ma-models-local.json
python -B -m unittest discover -s tools -p test_benchmark_ma_models.py
cargo test --offline -p rs-jxl --lib modular::ma_ -- --test-threads=8 --quiet
cargo test --offline -p rs-jxl-cli --example encoder_lab
```

`--description-cost` and `--model-costs` belong to `encoder_lab`, not the
installed CLI/Python API. The native Rust `source_ma_scoring.description` field
already exists. Model receipts deliberately reject example `--search` and
`--scan-only`; summing full trials and probes would not describe the final file.
`--profile` on the benchmark enables diagnostics, invalidating its timings as a
speed benchmark. This is targeted Windows validation, not a full-corpus or
cross-platform acceptance run.
