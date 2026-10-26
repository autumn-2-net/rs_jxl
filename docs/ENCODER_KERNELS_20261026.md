# Encoder Kernel Work, 2026-10-26

This round targets backend work, not candidate pruning or a smaller MA budget.
The baseline is the local corpus-repair-v3 build, frozen under
`out/encoder-kernels-20261026/encoder-lab-before.exe` and
`out/encoder-kernels-20261026/rs-jxl-before.exe`. Binary hashes and input/payload
hashes are recorded by the benchmark reports. The worktree is not a clean commit.

## Measurement

- Native Windows x86-64 release build, 16 encoder workers, effort 9.
- Full content-pilot search with the unchanged strict/default MA learning budget.
- Matrix: `out/corpus-repair-20261026/pilot.json`.
- Primary time is the complete encoder call, including all probes and candidates,
  excluding executable startup, source image decode, and file I/O.
- Independent acceptance uses official libjxl 0.12 `djxl` against original pixel
  samples, including hidden RGB and 16-bit samples. Compressed JXL equality is
  not an acceptance condition.
- Profile stage totals are inclusive and aggregate concurrent tasks. They must
  not be added or compared directly with wall time.

## Before Profile

`out/encoder-kernels-20261026/profile-before/report.json` uses
`RS_JXL_PROFILE=1`. These are diagnostic runs, not uninstrumented medians.

| Image | Pixels (MP) | Call s | MPix/s | bpp | Split sweeps | Full-candidate scopes s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floral_frame_PNG270.png | 18.061716 | 23.038674 | 0.784 | 2.259042 | 538153 | 22.521480 |
| pizza_PNG44037.png | 12.8232 | 39.217861 | 0.327 | 11.116226 | 1972533 | 38.347688 |
| wallhaven-mlzoy1.png | 8.2944 | 29.019489 | 0.286 | 13.675351 | 1534802 | 28.170247 |

All three have four full-candidate calls. The full-candidate work dominates the
whole search. The repeated MA split scan is the largest inclusive backend stage.

`profile-detail-before/report.json` additionally enables
`RS_JXL_PROFILE_DETAIL=1` for wallhaven. It records 133,124,543 predictor scoring
calls and 132,416,195 uncached predictor histogram collections. Timer overhead
raises the complete call to 32.338167 s; do not report this as normal performance.

## Changes

1. Build the ordered list of live property thresholds and cumulative weights
   once per property. Reuse it for every predictor and refinement pass, instead
   of rescanning empty root quantizer bins for each predictor. The second
   iteration retains the direct sweep for default-scoring nodes with fewer
   than four active predictors, which cannot amortize the extra table well.
2. Collect adjacent compacted predictor columns together for the 14-predictor
   Variable route, not only the two-predictor route. Histograms remain separate.
3. Reduce small serial-node candidate results immediately instead of allocating
   a per-property result vector. Keep ordered ties, drain failures, and do not
   publish partial caches.
4. Prevalidate the gather schema width, reuse exact static channel/group bins,
   and specialize the token/count part of HybridUintConfig(4, 1, 2).

No sampling, prediction tools, threshold candidates, leaf budgets, scheduler
routes, entropy objective, or bitstream format are changed.

### Layout Experiment Not Promoted

`layout-trial/report.json` compares the existing explicit
`--histograms bins` tool with the default on wallhaven, using the new binary.
The single uninstrumented calls are 27.513144 s and 27.003439 s respectively,
both 14,178,604 bytes. A 1.85% single-run difference is insufficient evidence to
change a default. This experiment remains opt-in.

## Validation

`cargo test --offline -p rs-jxl --lib modular:: -- --test-threads=8`:
480 passed, one opt-in microbenchmark ignored. This includes compact histograms,
paired predictors, alternative scoring/refinement, cache reuse, error cleanup,
weighted sample collection, wide/static quantizers and MA budget modes.
The added multi-failure test verifies ordered error selection and complete cache
cleanup in the serial candidate-reduction path.

`cargo build --offline --release -p rs-jxl-cli --bin rs-jxl --example encoder_lab`
passed. The existing unused `Tools::apply` warning in the lab target remains.

### First Iteration: Uninstrumented Slow Cases

Reports: `before-r0`/`after-r0`, `after-r1`/`before-r1`, then
`before-r2`/`after-r2`, all beneath `out/encoder-kernels-20261026/`.
Each report has one run per image; the following takes the median of three runs
per binary. All 18 outputs passed official original-pixel comparison.

| Image | Before s | After s | Time change | After MPix/s | bpp | Bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floral_frame_PNG270.png | 21.178700 | 21.607064 | +2.02% | 0.835917 | 2.259042 | 5100273 |
| pizza_PNG44037.png | 36.002108 | 34.447671 | -4.32% | 0.372252 | 11.116226 | 17818198 |
| wallhaven-mlzoy1.png | 26.347627 | 25.049593 | -4.93% | 0.331119 | 13.675351 | 14178604 |

Sizes are unchanged in all measured runs. Timing ranges in seconds:

| Image | Before range | After range |
| --- | ---: | ---: |
| floral | 20.161917..22.386597 | 21.260908..23.268572 |
| pizza | 35.464451..36.628730 | 34.009979..37.298985 |
| wallhaven | 25.080255..26.841044 | 24.697788..25.354040 |

The sum of per-image medians decreases from 83.528435 s to 81.104328 s (2.90%).
This is a modest measured improvement, not a universal speedup or a 1 MPix/s
result. Floral has a measured 2.02% regression; the overlapping ranges do not
establish whether that is persistent. Do not label it a win or claim absence of
timing regressions. The remaining dominant backend work is still MA learning.

### First Iteration: Independent Holdout

`holdout-before/report.json` and `holdout-after/report.json` contain 36 distinct
PNG images across icons, textures, photos, manga, screenshots and 16-bit patterns.
All 72 outputs passed official pixel checks. Total size remains 24,159,501 bytes
for 36,363,908 pixels (5.315051 bpp).

Total call time increases from 30.204567 s to 31.546342 s (+4.44%), with throughput
decreasing from 1.203921 to 1.152714 MPix/s. These are single runs, not medians.
Consequently the unconditional split-table path is not accepted as a general
speedup. The first-iteration executables are retained as `encoder-lab-v1.exe`
and `rs-jxl-v1.exe`; the next iteration limits table reuse by active predictor
count without changing any candidate or search threshold.

`cli-smoke/report.json` passed both checkpoints: 16-bit PNG exact samples and
`flower_cropped.jpg` byte-exact JPEG reconstruction through official djxl.
The JPEG output is 166,741 bytes (1.233291 bpp); these CLI timings are process
wall time and are not mixed with the encoder-call table above.

### Second Iteration: Work-Gated Threshold Reuse

The revised release build restricts threshold-table reuse to at least four
active predictors or non-default scoring/refinement. The direct two-predictor
sweep remains available. No search settings or parallel-routing settings change.

`holdout-v2/report.json`, followed by `holdout-control-v2/report.json` using the
frozen before binary, verifies all 72 outputs. Total call time is 30.372412 s
before versus 29.953900 s after (-1.38%); throughput is 1.197268 versus
1.213996 MPix/s. Bytes and weighted bpp remain 24,159,501 and 5.315051.
These are single-run aggregate measurements, not evidence of a stable speedup
on every image. The initial +4.44% aggregate regression did not recur in this
paired check, but small per-image timing differences still occur.

`slow-v2/report.json` checks the three slow cases with the revised build:

| Image | Call s | MPix/s | bpp | Bytes |
| --- | ---: | ---: | ---: | ---: |
| floral_frame_PNG270.png | 20.098563 | 0.898657 | 2.259042 | 5100273 |
| pizza_PNG44037.png | 36.081766 | 0.355393 | 11.116226 | 17818198 |
| wallhaven-mlzoy1.png | 24.416288 | 0.339708 | 13.675351 | 14178604 |

All three outputs pass official original-pixel comparison. These are one-run
measurements, not medians directly comparable with the three-run first-iteration
table. Pizza remains in the old baseline timing range; this is not a solution
to the slow-case throughput problem. `cli-smoke-v2/report.json` also passes
16-bit sample equality and byte-exact JPEG reconstruction for both checkpoints.

## Actual MA Parallel Route

The three slow-image profiles were already running parallel MA. This is not
inferred from the CLI thread count alone: `ma.subtrees` records dispatched
subtree batches, and `execution.cooperative` records their arena capacity.

| Image | Subtree batches | Batches with multiple worker tasks | Subtree jobs | Maximum worker tasks |
| --- | ---: | ---: | ---: | ---: |
| floral_frame_PNG270.png | 111 | 110 | 5482 | 16 |
| pizza_PNG44037.png | 302 | 298 | 23215 | 16 |
| wallhaven-mlzoy1.png | 265 | 265 | 24893 | 16 |

These counts cover all candidates/probes, not just the winning encoding. All
cooperative-arena entries in these profiles report 16-worker capacity. They
establish that parallel dispatch is used, not that all workers stay occupied
or that the decomposition is optimal; per-worker busy/wait time is not recorded.

Relevant routing:

- `scheduler/pilot.rs` enables routed tools; `scheduler/native_tools.rs::apply`
  enables balanced whole-subtree scheduling and the 8192-record entry gate.
  This also applies to normal CLI `--source-pilot`, not just the lab executable.
- `modular/ma_train.rs::find_best_split_with_properties` dispatches properties
  when parallel execution is enabled, there are at least 1024 records,
  records * predictors * properties >= 65536, and more than one worker exists.
  Above-threshold parent-histogram work and some ready sibling searches also
  dispatch parallel tasks.
- `modular/ma_parallel.rs::finish` runs multiple detached subtrees concurrently.
  Each detached subtree deliberately uses a serial inner learner. It does not
  recursively spawn property jobs; this avoids multiplying nested worker work.
- The tested profiles contain no `ma.channel_trees` entry. Forced channel-tree
  partitioning is a separate, disabled tool, not a prerequisite for MA parallelism.
- Explicit `MaBudgetMode::Bounded` serializes channel-root/detached-subtree
  budget consumption. This test matrix uses default strict mode, not that path.
- Some probe encodes use an independent/null runner internally, while the
  scheduler runs the probe encodes concurrently. That is a different parallel
  layer, not evidence that full-image MA accidentally became serial.

The remaining suspects are repeated histogram/scoring work, small-task overhead,
shared-resource contention and tail imbalance. Existing inclusive stage totals
cannot separate those effects. Do not describe the root cause as "parallel MA
was disabled" or claim full parallel efficiency without additional measurements.
