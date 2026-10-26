# Encoder Scout Experiments, 2026-10-25

## Scope

Pixel-lossless encoding and original-byte JPEG transcoding only. Original
decoded samples (including invisible RGB), or original reconstructed JPEG
bytes, are the correctness criterion. Matching a previous compressed stream
is not an optimization objective. No lossy encoder or decoder changes.

## Port Status Clarification

The local optimized C++ source HEAD was checked and remains
`b866b3defec13494bfbaf705db93581cda118e37`, the Rust port's frozen source.
Implementation coverage, default activation and measured benefit are distinct.
Neither the existing tests nor this experiment establish that every C++
optimization has identical activation, performance and platform coverage.

Current MA implementation already includes property-level split parallelism,
ready-child parallel evaluation, optional whole-subtree jobs, histogram reuse
and subtraction, worker scratch reuse, and physical sample-column partitioning.
The default split sweep requires at least 1024 records, at least 65536 units
of record/predictor/property work, and more than one available worker. Nested
independent candidates deliberately use the serial execution scope instead of
creating additional pools.

`tools/ma_simd.rs::DefaultScorer` already dispatches to native SSE2, AVX2 and
AVX2+FMA for scoring and fused count movement, with scalar fallback. The initial
audit had no AVX512 backend; the later explicit AVX512 and AVX2 reciprocal
experiments below are now implemented but remain off by default. Optional
predictor/property deduplication, sparse histograms,
statistical racing, channel-tree partitioning and detached subtrees are not
all enabled by default. See `MA_PORT_GAPS.md`, `PORTING_KERNELS.md` and
`PIXEL_LOSSLESS_PORT_STATUS.md` for the audited implementation scope.

## Experimental Pixel Route

`DriverOptions::scout()` / CLI `--source-scout` is an explicit experiment for
source effort 9/10 with `--source-search`. It does not replace the default
source scheduler or expose a different lossy policy. Unsupported effort values
fail. Small images or unsupported sampling geometry retain the existing source
route; source-input restrictions still apply.

The route starts with retained full-image anchors, then uses bounded coverage
and palette-feasibility scanning, cached coarse ranking and full-fidelity crop
scores. The candidate pool includes automatic backends, explicit predictor/RCT
families, palette variants up to learning fraction 1.0, and the anchor's
neighborhood. Cheap scouting parameters do not replace final parameters.

The revised route adds one full-image learned-palette witness (e9, learning
fraction 0.01) when the initial literal output is at most 85% of the regular
output. This addresses a measured mismatch: MA overhead reverses the ranking
of learned/unlearned palette candidates on small crops versus the full image.
This witness counts toward the challenger budget; it can lose and is not a
guarantee that palette variants dominate. Fractions 0, 0.01, 0.03, 0.1 and 1
remain eligible. The measured full-image winner anchors subsequent screening.

The revised e9 path can finish after a material (at least 5%) literal gain,
but only after a cross-family challenge. It scores the source cross-channel
variable-predictor representations against the learned literal at full probe
fidelity, and admits the best competitive rival for a real full-frame encode.
If that rival beats the retained full-image winner, one RCT sibling is also
encoded: cropped RCT rankings are not assumed to transfer to the full image.
Strong within-family improvement alone is not evidence that other families
cannot win. This is still a search-budget heuristic, not a bound on unseen
candidates. Weak literal gains continue through the broad frontier; e10
retains the same witness/challenge prefix and keeps deeper exploration. No
image names or benchmark identities are used in routing.

Up to three full-frame challengers are nominated at e9. A challenger can survive
because its total is within 2.5% of the current witness, or because it improves
a region accounting for at least 10% of witness bytes by more than 5%. Full
results update the retained winner; sample estimates never become valid byte
bounds. Real finalized-byte bounds can abort a losing candidate. Timeouts and
resource errors are failures, not successful bounded encodes.

At e10 the e9 prefix is retained, up to five challengers are considered, and
deeper winner/automatic candidates follow. This does not guarantee matching
the legacy scheduler's winner. Region sampling can miss whole-image effects;
compression-size changes must be measured before any default promotion.

## Baseline Diagnostic

The paper screenshot `9bde6aaf390109f97622d35ff1e900ee.png` used seven full
encodes and 134 probes in the starting encoder. An instrumented eight-thread
run recorded 22.883 seconds in `cli.total`, 17.769 inclusive worker seconds
in shared LZ77, 6.231 in MA learning and 3.442 in MA split sweeps. These scopes
overlap and diagnostics add overhead: do not add them or use them as normal
throughput measurements. They show that MA SIMD alone is not the whole issue.

The first diagnostic used the CLI default 1,048,576-sample resource limit and
failed with `grouped MA selected-sample budget exceeded`. It is not a timing
result. The complete diagnostic and before/after benchmark use the same
33,554,432 ceiling as previous encoder comparisons, without reducing the
source sampling fraction. Both logs remain under `out/encoder-scout-20261025`.

## Measurement Harness

`tools/benchmark_encoder_checkpoints.py` compares frozen Rust executables,
serially rotating before/after order. Source effort is 9, encoder threads 8;
timing is CLI wall time including IO, excluding independent decode checks.
Warmups are checked but excluded from medians. PNG snapshots retain original
sample values while removing metadata unsupported by comparison decoders.
JPEG is original-byte transcoding, never pixel recompression.

Every output is independently checked with official djxl. JSON and Markdown
reports retain bytes, bpp, MPix/s, commands, diagnostics and binary/input
identities. Temporary image outputs are removed. A passed harness run means
correct outputs and complete measurements, not necessarily an improvement.
`--after-arg=--source-scout` is PNG-only; JPEG flags are separate.

This document records an experiment, not full-corpus acceptance. Measurements
and final retention decisions are appended after the frozen-binary comparisons.

## Rejected First Scout Gate

`out/encoder-scout-20261025/scout-v1-gate/REPORT.md` records one timed round,
no warmup, eight threads. All outputs passed independent original-pixel checks,
but the route failed compression-quality acceptance:

| Input | Before bytes | Scout v1 bytes | Before s | Scout v1 s |
| --- | ---: | ---: | ---: | ---: |
| Tecnick RGB_OR_1200x1200_023 | 1,530,900 | 1,530,900 | 6.760 | 4.990 |
| clover | 1,586,564 | 1,586,564 | 10.625 | 7.960 |
| news.ycombinator.com | 58,672 | 62,279 | 5.361 | 2.289 |
| paper 9bde6aaf | 146,658 | 193,964 | 24.132 | 11.706 |

Screenshot size regressions are +6.147% and +32.256%. These are rejected
tradeoffs, not retained wins. The paper baseline's winner is the e9 learned
palette with fraction 0.01; v1's cropped ranking excluded it. V2 adds the
full-image witness above. Gates are diagnostic samples, not corpus guarantees.

## JPEG Search Reuse

The ordinary JPEG source-search entry now uses one per-invocation AC cache.
The key binds an immutable JPEG input and the resolved coefficient-order,
CfL, entropy/LZ policy, geometry and resource bounds. Adjacent e3/e5/e7
candidates share the same AC policy for normal-sized frames; e9/e10 share a
second group. Tiny-image coefficient-order gates remain part of resolution.

Only encoded AC fragments are retained, with their exact bit lengths. Quant
tables and DC/auxiliary streams remain candidate-specific. No JPEG trial is
removed; independent complete outputs still determine the winner. Direct
single-backend transcoding remains uncached. Source e1/e2 also use the direct
path because their prefix contains only one candidate. At most one policy group is
retained, evicted before rebuilding; coefficient/token/model working buffers
are not retained across candidates. Errors cannot publish a partial cache.

Primitive-equivalence tests intentionally compare cached and uncached fragments
and outputs. This tests the reuse implementation, not a constraint that future
compression algorithms must emit the old JXL stream.

## Second Scout Gate

`out/encoder-scout-20261025/scout-v2-gate/REPORT.md` used the same single-round
gate. Both screenshots returned the learned-palette witness at full trial 3,
but still executed 64 probes and two additional full trials with no benefit.
Web size improved from 58,672 to 58,238 bytes with 5.175 -> 5.163 seconds;
paper size matched at 146,658 bytes but took 23.127 -> 26.783 seconds. Thus
restoring the correct route alone did not improve paper throughput. The final
revision above eliminates post-witness search only after a measured material
gain, leaving the default scheduler unchanged.

The first real JPEG cache gate (`wallhaven-vpyekp.jpg`) retained 92,752 bytes
and passed official original-byte reconstruction, with 0.475 -> 0.426 seconds.
Single-round timings are preliminary and are not the final performance claim.

## V3 Repeated Mixed-Sample Results

`out/encoder-scout-20261025/scout-v3-mixed/REPORT.md` contains one validated
warmup plus three timed rounds per checkpoint on ten inputs. All 80 outputs
passed official djxl validation; no timeout or failed sample was excluded.
The comparator is the Rust encoder frozen before this change, **not C++ or
official cjxl**. Pixel results use the explicit scout; JPEG uses ordinary
source search. Times include CLI startup and file IO but not verification.

| Input | Before s | After s | After MPix/s | After bpp | Byte change |
| --- | ---: | ---: | ---: | ---: | ---: |
| news.ycombinator.com | 5.154 | 2.146 | 0.895 | 0.242500 | -0.740% |
| paper 9bde6aaf | 26.098 | 2.991 | 0.487 | 0.805742 | 0 |
| Tecnick RGB_OR_1200x1200_023 | 6.482 | 4.530 | 0.318 | 8.505000 | 0 |
| clover | 10.062 | 8.288 | 0.127 | 12.104523 | 0 |
| kodim19 | 8.763 | 8.931 | 0.044 | 8.889730 | 0 |
| grayscale cvo9xd | 1.307 | 1.304 | 0.192 | 2.887168 | 0 |
| 16-bit vgqcws | 0.176 | 0.176 | 0.023 | 24.550781 | 0 |
| JPEG wallhaven-yqxzqx | 14.173 | 12.456 | 1.184 | 4.364689 | 0 |
| JPEG progressive wallhaven-9oo8k1 | 3.643 | 3.342 | 2.824 | 0.958587 | 0 |
| JPEG wallhaven-vpyekp | 0.423 | 0.418 | 5.874 | 0.301927 | 0 |

The three small PNGs take the legacy fallback; their timing variation is not
evidence of scout acceleration. The smallest JPEG gain is within ordinary
short-run timing variation. The two screenshots informed the rule revision,
so they are not holdout evidence. This ten-image experiment cannot establish
full-corpus non-regression or justify replacing the default pixel scheduler.

The subsequent holdout gate **rejected v3 for compression quality**: `1eb7e7b9`
grew from 351,302 to 417,020 bytes (+18.707%), despite 36.253 -> 8.091 seconds.
The old scheduler's winner was not another literal learning fraction: profiling
identified a cross-channel variable predictor with RCT 6, group dimension 1024,
learning fraction 0.1 and two prior-channel properties. V3 stopped before
testing that family. This motivated the cross-family challenge above; the v3
table must not be read as final quality acceptance.

Other single-round v3 holdouts were exact and retained size: `3dd595a3`
80,020 bytes (6.010 -> 2.563 seconds), 4096-square RGBA `146845733_p0`
1,329 bytes (6.031 -> 6.127 seconds), and the repeated JPEG sentinel 92,752
bytes. The ordinary JPEG cache is independent of this rejected pixel heuristic.

The frozen v3 binary predates the final e1/e2 direct-path guard; the measured
e9 route is unchanged by that guard. Final-binary holdout checks are recorded
separately below. The input and frozen binary identities are in JSON reports.

## Verification And Use

Targeted checks after implementation:

- `cargo test --locked --offline -p rs-jxl --lib scheduler:: -- --test-threads=8`:
  176 passed after the cross-family regression fix.
- `cargo test --locked --offline -p rs-jxl --lib jpeg:: -- --test-threads=8`:
  341 passed, 9 explicit external-oracle tests ignored.
- `cargo test --locked --offline -p rs-jxl-cli --bin rs-jxl -- --test-threads=8`:
  85 passed, 2 explicit external-data tests ignored.
- `python -B -m unittest discover -s tools -p test_benchmark_encoder_checkpoints.py`:
  18 passed.
- `cargo clippy --locked --offline --workspace --all-targets -- -D warnings`:
  passed.

Tests include strong/weak palette feedback, cross-channel family challenges,
e10 retaining the e9 witness,
unsupported effort rejection, JPEG e1/e2 retaining their single backend-3
trial, fragment alignment/tails, immutable image and policy keys, eviction,
coefficient/original-byte reconstruction, and exact final-container limits on
cache hits and misses. Full workspace runtime tests were not repeated; the
external-oracle matrix above is separate from the ignored unit-test entries.

Example opt-in pixel command after building the release CLI:

```powershell
.\target\release\rs-jxl.exe encode input.png output.jxl --source-effort 9 --source-search --source-scout --source-sample-limit 33554432 --threads 8
```

Do not add `--source-scout` to JPEG transcoding: matching AC reuse is already
part of its ordinary `--source-search` path. Pixel scouting is experimental,
does not promise a size bound relative to legacy search, and remains opt-in.
The measurements do not certify e10 performance or untested platforms.

## RCT Feedback Revision

The first cross-family gate (`scout-family-gate`) reduced the `1eb7e7b9`
regression to 353,441 vs 351,302 bytes (+0.609%) at 18.332 vs 34.879 seconds.
That still exceeds the intended 0.25% tradeoff, so it was not accepted as the
final policy. Diagnostics showed the cropped ranking selected RCT 0, whereas
the full-image winner used RCT 6 within the same variable-predictor family.

The final revision confirms one alternate RCT only when a completed family
challenger actually beats the retained full-image result. Losing challengers
do not trigger another complete encode. This remains within the three-witness
e9 budget and is part of e10's retained prefix. The regression test deliberately
reverses the crop and full-image RCT ranking. This change does not modify
ordinary default pixel search, JPEG search, decoder code or SIMD kernels.

## Final Affected-Case Gate

`out/encoder-scout-20261025/scout-feedback-gate/REPORT.md` records the final
release executable (`rs-jxl-feedback-scout.exe`), one measured round, no warmup,
eight threads. All 12 outputs across six before/after pairs passed independent
official djxl validation. No final-gate image grew. Single-round times below
are diagnostic, not warmed median claims; do not substitute the faster but
rejected v3 screenshot timings for these corrected results.

| Input | Before s | After s | After MPix/s | After bpp | After bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| paper 1eb7e7b9 | 33.976 | 25.944 | 0.139 | 0.780255 | 351,302 |
| paper 9bde6aaf | 20.714 | 12.261 | 0.119 | 0.805742 | 146,658 |
| news.ycombinator.com | 4.940 | 2.278 | 0.843 | 0.242500 | 58,238 |
| Tecnick RGB_OR_1200x1200_023 | 6.130 | 4.215 | 0.342 | 8.505000 | 1,530,900 |
| JPEG wallhaven-vpyekp | 0.424 | 0.386 | 6.371 | 0.301927 | 92,752 |
| screenshot 3dd595a3 | 5.506 | 3.303 | 0.647 | 0.299484 | 80,020 |

Only the news screenshot changes size (-0.740%); the others retain their
baseline sizes. The formerly held-out `1eb7e7b9` now informed the repair, so
this final run is regression evidence, not an independent generalization test.
The final pixel rule still needs a broader untouched corpus and repeated timing
before any default promotion. No claim is made that it always matches legacy
compression size. The corrected paper timings also show significant backend
work remains even after reducing search.

The separate JPEG cache is retained in ordinary source search. Its repeated
mixed-sample e9 measurements above showed unchanged sizes and roughly 12.1%
and 8.3% elapsed-time reductions on the large sequential/progressive examples;
the tiny JPEG gain was not decisive. The final source adds only an e1/e2
direct-path guard relative to that measured JPEG implementation.

All builds/checks were local Windows x86_64. No C++ source, decoder, CI,
neural model, ARM kernel or Python binding was changed by this encoder work.
No commit or push was made.

## Hotspot Acceleration Follow-Up (In Progress)

The current target is **1 MPix/s for the entire e9 scout**, not just the selected
candidate or an isolated kernel. It has not been reached on the slow paper
screenshots. Do not interpret a fast single-candidate measurement as acceptance.

The first backend revision specializes token trees per channel while retaining
wire context IDs, avoids unused predictor/history work, vectorizes MA score
tails, compiles LZ symbol prices once per model, and rejects provably impossible
chain extensions. The scout also installs the existing exact gather, channel
tokenization and bounded LZ stream execution tools. It does not enable striped
sampling or reduce chain depth.

`hotspots-v2-gate/report.json` compares frozen before/after binaries at eight
threads, one measured round without warmup. Every output passed official djxl
integer-pixel comparison, including invisible RGB. All sizes stayed unchanged.

| Input | Before s | After s | After MPix/s | After bpp | Bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| paper 1eb7e7b9 | 27.807 | 18.297 | 0.197 | 0.780255 | 351302 |
| paper 9bde6aaf | 13.083 | 10.840 | 0.134 | 0.805742 | 146658 |
| Tecnick RGB_OR_1200x1200_023 | 4.630 | 4.763 | 0.302 | 8.505000 | 1530900 |
| news.ycombinator.com | 2.310 | 2.057 | 0.934 | 0.242500 | 58238 |

The remaining bottlenecks differ. On paper, a losing cross-channel candidate
alone costs about seven seconds; the retained literal candidate costs about
two. On Tecnick, a single automatic e9 candidate takes about 1.2 seconds while
the scout performs 69 probe encodes. Fixed automatic e9 is not a general fix:
on the clover texture it grows from 1586564 to 1673489 bytes (about 5.5%).

The development-only `rs-jxl-cli/examples/encoder_lab.rs` entry point and
`tools/benchmark_encoder_tools.py` compare explicit backend tools or full scout
runs. The example reports encoder-call timing separately from the harness's
whole-process timing. JSON matrices avoid Windows command-line JSON quoting.
Reports retain binary identities for reproducibility, not as compressed-output
correctness requirements. Temporary JXL/decoded PNG files are removed.

An initial immutable-chain batch experiment was rejected as a default: paper
improved only slightly, while 1eb increased from 21.35 to 46.09 seconds with
unchanged size. Profiling isolated a 19.19-second speculative batch traversing
long copies whose interiors the sequential DP would skip. The revised experiment
bounds speculative match extension and requeries the complete candidate set only
when consumed. This is not a timeout or truncation of final matches. It remains
opt-in pending measured benefit.

Further experimental work includes larger original-pixel strip probes, a
clipped-entropy sparse-event MA sweep, and already ported alternative LZ tools.
These are not promoted based on primitive tests alone. The exact weighted-
predictor flat shortcut retains the per-pixel state update and checks all five
neighbors plus signed error history. Core verification after these changes:
1602 tests passed, 13 ignored; the subsequent speculative-boundary revision
passed all nine targeted chain-batch tests. External performance/oracle gates
are recorded separately as they complete.

### Representation And Parallelism Findings

The paper probe error was not only fixed-header overhead. The source palette
policy has an area-dependent cap. The complete 9bde image accepts a 3614-entry
RGBA tuple palette; the small crops can reject it and encode four spatial
channels instead. Such probes compare the wrong representation. Increasing
strip height to 128/256 rows did not reliably fix cost or ranking. The diagnostic
`--probe-palette` bypasses this rejection only in probes and explicitly records
a policy gap. It does not preserve whole-image palette index assignments and
must not be presented as exact representation replay.

`candidate-palette-proof-gate/report.json` (v5, eight threads, one round) reduced
the complete paper scout to 2.060 s, 0.707 MPix/s, 145940 bytes, 0.801798 bpp with
indexed matching. This still misses the end-to-end target.

`candidate-parallel-tools-gate/report.json` isolates the real 1eb cross6 candidate:

| Tools | Wall s | MPix/s | Bytes | bpp |
| --- | ---: | ---: | ---: | ---: |
| bucket LZ, event MA | 4.563 | 0.789 | 351934 | 0.781659 |
| plus MA subtrees | 3.425 | 1.052 | 351934 | 0.781659 |
| plus parallel WP presets | 3.288 | 1.095 | 351934 | 0.781659 |
| plus predictor/property dedup and sparse histograms | 3.158 | 1.141 | 352010 | 0.781827 |

These are single-candidate measurements, NOT full-scout acceptance. The bucket
output is about 0.18% larger than the old 351302-byte winner. Indexed/suffix
matching on this family exceeds the 0.25% size tolerance, so it cannot replace
the old matcher globally. Indexed matching on the paper literal family instead
improves both time and size (2.099 to 0.992 seconds, 146658 to 145940 bytes).
Every measured row above passed official djxl comparison of original samples.

With MA subtrees and WP parallelism, the 1eb candidate's profile is approximately
3.24 seconds: MA learning 0.70, shared LZ 1.35 (raw/fitted histogram construction
0.47), preparation 0.36, entropy clustering 0.25. Inclusive and parallel scopes
overlap; their totals must not be added as independent wall time. Sixteen threads
improved the same candidate only to 3.080 seconds, so increasing threads alone
is not the next primary optimization.

The next frozen experiment adds bounded dense LZ histogram counting with sorted
sparse finalization, parallel post-merge entropy-cluster pair evaluation, and an
opt-in pilot-first family frontier. The frontier executes full-strength probes
before any full frame and confirms near/global or material regional rivals.
Only complete encoded outputs can win. Sampling remains heuristic, not a byte
bound or a guarantee against unseen candidates. These changes passed 1631 core
tests (13 ignored); external acceptance is still pending.

### Pilot-First v6 Gate

`candidate-pilot-gate/report.json` compares three experimental routes on six
fixed inputs. One round, eight threads, no warmup. All 18 outputs were decoded
by official djxl and matched original integer samples. The following are
**complete scout** measurements (wall time includes process startup/input/output):

| Input | Route | Wall s | MPix/s | Bytes | bpp |
| --- | --- | ---: | ---: | ---: | ---: |
| paper 9bde | indexed, wide strips | 1.360 | 1.070 | 145940 | 0.801798 |
| paper 1eb | buckets/events, wide strips | 3.460 | 1.041 | 351934 | 0.781659 |
| news.ycombinator.com | indexed, wide strips | 1.641 | 1.171 | 57749 | 0.240464 |
| editor 3dd | indexed, wide strips | 1.087 | 1.966 | 77651 | 0.290618 |
| Tecnick 023 | indexed, wide strips | 2.970 | 0.485 | 1530900 | 8.505000 |
| clover | buckets/events, wide strips | 3.195 | 0.328 | 1586553 | 12.104439 |

This table uses different explicit backend choices; it is not yet a validated
single automatic configuration or proof of a general 1 MPix/s floor. Sixteen-row
probes are rejected as a general shortcut: paper improved, but 1eb regressed to
8.287 seconds and the web screenshot to 2.778 seconds through extra full work.
Index matching on cross6 is also excluded from the proposed automatic route
because its 352221 bytes exceed the 0.25% size-growth budget.

The natural-image trace confirms a representation/scale issue: all three thin
clover probes slightly favor automatic, but the actual complete neighborhood
candidate is 5.2% smaller. Both are executed. Tecnick also executes both, yet
automatic really wins. Arbitrarily forcing either route would discard quality.
The next experiment adds measured tree/model bit receipts, separate from sample
tokens, to test amortized probe ranking, and square-context probes. Such ranking
is explicitly an estimate, never a real compressed-output byte bound.

The retained JPEG sentinel remains byte-reversible: v5/v6 both output 92752 bytes
(0.301927 bpp), with measured wall times 0.385/0.368 seconds. Report:
`v6-jpeg-sentinel/report.json`. No JPEG scheduling policy was changed here.

### Model Costs And v8 Gate

Tree/model receipts measure emitted tree and entropy-header bits, not pixel
tokens or entire DC sections. The experimental score amortizes only these bits
by sample/full pixel counts. It is never passed to the output-budget API. Native
invocations have independent collectors, including patch-reference encoding.

On clover this corrects the probe ordering: neighborhood wins all three regions,
matching the complete-frame result. A conservative three-region consensus then
avoids the second full encode. An eight-row first stage can discard clearly
losing families before the larger probes. e10 retains all candidate families;
coarse/fine stages share cumulative work accounting and release coarse resident
samples before loading fine samples.

`candidate-consensus-gate/report.json`, v8, eight threads, one round:

| Input | Complete pipeline wall s | MPix/s | Bytes | bpp |
| --- | ---: | ---: | ---: | ---: |
| clover, staged | 1.683 | 0.623 | 1586564 | 12.104523 |
| Tecnick 023, staged | 1.977 | 0.729 | 1530900 | 8.505000 |

All outputs passed official djxl pixel verification. The one-megapixel target
is NOT reached here. Uniformly shrinking the fine probes is rejected: eight-row
fine probes made Tecnick run extra candidates and take 5.750 seconds, whereas
128-square context probes took 1.214 seconds. Likewise model correction applied
to screenshots made 1eb take 6.840-8.120 seconds. These are screening regressions,
not acceptable global defaults.

The v9 experiment restricts model correction to non-literal inputs and retains
raw strip scores for literal evidence. It is available explicitly through
`scheduler::pilot::encode` and CLI `--source-pilot` (with `--source-search
--source-effort 9` or 10); ordinary source/scout defaults are unchanged. It does
not guarantee the smallest result among candidates that were not fully encoded.

Exact backend work in v9 includes compact LZ rings/reused query scratch,
predictor-level MA SIMD dispatch, allocation-free refinement minima, masked
short entropy histograms, row-addressed WP cost evaluation and exact small-error
WP weight lookup. The profiling collector is sharded to avoid serializing MA
workers on one diagnostic mutex. Core gate: 1668 passed, 13 ignored, no failures.
Performance and wider pixel verification for these changes are pending.

### Unified Pilot v9/v10 Gate

The following v10 runs use ONE content-aware policy, effort 9, 16 threads,
without profiling. Wall time includes executable startup and image I/O; the
encoder-call measurement is reported separately. Each output passed official
djxl reconstruction against every original integer sample, including alpha.
These are single-round diagnostics, not repeated medians or a universal floor.

| Input | Encoder s | Wall s | Wall MPix/s | Bytes | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| paper 1eb | 2.588 | 2.668 | 1.350 | 351934 | 0.781659 |
| paper 9bde | 1.011 | 1.063 | 1.370 | 145969 | 0.801957 |
| editor 3dd | 1.332 | 1.389 | 1.539 | 77651 | 0.290618 |
| Tecnick 023 | 0.987 | 1.045 | 1.379 | 1530900 | 8.505000 |
| clover | 1.110 | 1.162 | 0.902 | 1586564 | 12.104523 |
| sparse RGBA 4096 | 1.313 | 1.551 | 10.818 | 1286 | 0.000613 |
| wallhaven 49yekn | 1.608 | 1.670 | 1.242 | 698853 | 2.696192 |
| Kodak 19 | 1.598 | 1.630 | 0.241 | 436948 | 8.889730 |
| game Blobby | 1.131 | 1.162 | 0.411 | 190018 | 3.184194 |
| mail icon 512 | 0.339 | 0.369 | 0.711 | 27736 | 0.846436 |
| grayscale 500 | 0.517 | 0.545 | 0.459 | 90224 | 2.887168 |

Reports: `pilot-v10-first/report.json`, `pilot-v10-holdout/report.json` under
`out/encoder-scout-20261025`. The web screenshot is intentionally not presented
as an accepted speedup: v10 outputs 60319 bytes versus v9's 57748 (+4.45%),
despite improving wall time from about 1.6 to 0.65 seconds. Static palette
probes wrongly excluded a useful learned sibling. That guard is being fixed.
The sparse RGBA omission in v9 (3657 bytes) was fixed by adding a static-tree
candidate at the SAME e9 backend effort, not by falling back to lower effort.

v10 backend changes: speculative entropy-cluster distance caches preserve
sequential center updates and recompute every changed center; exact three-token
LZ prefix rejection; shared compact-pair MA collection. The pilot uses coarse
regional agreement to avoid unnecessary fine probes, and enables existing
coverage on 2^17..2^19-pixel images. The explicit helper now retains resource
ceilings and uses the requested worker budget. Ordinary search stays unchanged.
Core gate: 1686 passed, 13 ignored; benchmark harness: 20 passed.

Remaining hotspots from instrumented v10 traces (inclusive scopes overlap):

- Clover: one full encode, approximately 1.09 s; MA 0.31 s, shared LZ 0.30 s,
  preparation 0.13 s, entropy model 0.13 s. Cluster assignment fell from about
  0.095 to 0.038 s. The complete scout, not this isolated winner, must pass.
- Kodak 19: an unnecessary cross-channel full candidate costs 0.86 s. Even
  without it, the automatic candidate takes about 0.57 s: MA gather 0.185 s
  and training 0.174 s. Thus screening alone cannot reach 1 MPix/s here.
- Game screenshot: three full candidates, two proven bounded losers, cost
  roughly 0.80 s plus 0.42 s probes. Bounded failures are never outputs.

Further exact kernels and finalist scheduling are under test. No result below
1 MPix/s is counted as meeting the requested target, and JXL output-byte
identity is not the correctness criterion.

### v11/v12 Refinement

Exact row-addressed neighbor loads, 16-byte MA decisions, unit-multiplier token
fast paths, dynamic-property quantization lookup tables and spatial-prefix masks
retain the original integer operations. v11 also adds sampled LZ diagnostics;
diagnostic clocks are not used for acceptance timings. v11 is an intermediate
binary: its parallel finalist prefilter could discard a candidate before a
pending winner changed the score reference. v12 fixes this with a peek/commit
boundary and an explicit rank-inversion regression test.

The v12 pilot uses at most two full finalists per wave on the existing pool.
Only a real completed baseline supplies the shared cutoff. Probe roots can
cooperate with their inner jobs under a 256 MiB scratch **estimate** (not an RSS
guarantee). No global pool, timeout fallback, smaller sample fraction or reduced
backend effort is introduced. Core gate: 1711 passed, 13 ignored.

`pilot-v12-gate/report.json`, sixteen threads, one round, full pipeline:

| Input | Encoder s | Wall s | Wall MPix/s | Bytes | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| clover | 1.068 | 1.120 | 0.936 | 1586564 | 12.104523 |
| Tecnick 023 | 0.865 | 0.916 | 1.572 | 1530900 | 8.505000 |
| Kodak 19 | 0.715 | 0.748 | 0.526 | 436948 | 8.889730 |
| game Blobby | 0.837 | 0.869 | 0.549 | 190018 | 3.184194 |
| web news | 1.166 | 1.220 | 1.575 | 57748 | 0.240460 |
| paper 1eb | 2.527 | 2.605 | 1.383 | 351934 | 0.781659 |

The web size regression is gone. Clover and the smaller holdouts still fail
the throughput target; no universal 1 MPix/s claim is warranted. Additional
formal CLI checks (`main-v12-sentinel/REPORT.md`) pass original-pixel comparisons
on paper, editor, RGBA, icon and grayscale, and exact original JPEG recovery.
The JPEG sentinel is 92752 bytes / 0.301927 bpp / 0.390 s / 6.308 MPix/s.
Single-round time differences are diagnostic, not statistical regression bounds.

The next exact backend experiments target the 32-observation sparse histogram
probation (over 99% of clover's sparse updates occur before later dense promotion),
repeated BTreeMap penalty lookups in LZ seed scoring, and repeated MA collector
hash computation. Any collector hash cache is bounded and retains exact equality
checks after a hash match. Large inputs must retain the uncached path.

### v13/v14 Histogram And Gather Gate

The bounded-context raw LZ histogram path skips sparse probation only after
checking every input context. A shared dense literal-penalty prefix replaces
repeated ordered-map queries, with the original fallback for wide contexts.
On instrumented clover, histogram counting fell from about 60 to 14 ms and
shared LZ from 225 to 165 ms. Inclusive scopes are not additive.

The MA hash cache is **off by default**. The experimental lab `--hash-cache`
switch controls it independently. Same-binary v14 A/B measurements did not
show a consistent gain: Tecnick was 0.848 s without it versus 0.900 s with it;
Kodak was 0.744 versus 0.764 s, with identical sizes. These are single rounds,
not confidence intervals. Report: `pilot-v14-hash-ab/report.json`.

The native pilot now merges exactly equivalent RCT0/RCT6 cross-channel trials
only for images with fewer than three TOTAL channels. Other parameters must
match; gray plus two extras is not eligible. Generic callback searches and
ordinary source defaults are unchanged. Tests cover these distinctions.

`main-v14-natural-repeat/REPORT.md`: formal CLI, e9 + source pilot, 16 threads,
one validated warmup and three alternating before/after repetitions. Below are
medians including startup and I/O. Every result passed official djxl original
sample equality. These changes do not establish a universal throughput floor.

| Input | v12 wall s | v14 wall s | v14 MPix/s | Bytes | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| clover | 1.120014 | 1.073796 | 0.977 | 1586564 | 12.104523 |
| Kodak 19 | 0.818067 | 0.802321 | 0.490 | 436948 | 8.889730 |
| Tecnick 023 | 0.960000 | 0.910216 | 1.582 | 1530900 | 8.505000 |
| grayscale 500 | 0.455497 | 0.406964 | 0.614 | 90224 | 2.887168 |

Core tests: 1730 passed / 13 ignored; CLI: 86 passed / 2 ignored; lab: 27 passed;
all integration-test targets also pass `cargo check -p rs-jxl --tests`.
The remaining gather hotspot is particularly visible on Kodak: 768852 selected
samples produce 768852 records (no successful merges, not a proof of key
uniqueness), but still pay hashing, merging and per-sample atomic accounting.
When all visited pixels fit the local sample limit, selected pixels necessarily
fit as well; this can eliminate local CAS without presampling or weakening a
shared budget. That proof-based bypass and a separately gated native AVX-512 MA
kernel are the next experiments, not accepted results yet.

### v15 Exact Budget Bypass And SIMD Experiment

Source gather now proves `selected <= visited <= local_limit` when possible,
eliminating local per-sample CAS only in that case. Tight local limits and all
shared limits retain the original debit behavior. No sampling sites, predictors,
weights or error ceilings are removed. Tests cover serial/parallel sample
equivalence and failure accounting.

An opt-in `RS_JXL_MA_AVX512=1` backend computes 16 histogram bins while retaining
eight logical accumulator lanes. Runtime AVX2/FMA/AVX512F/DQ checks guard it;
small/sparse kernels retain AVX2. It is read once per process and defaults OFF.
Core gate: 1742 passed / 13 ignored. Performance of the AVX512 experiment was
NOT established by the early harness runs: see the environment correction below.

Three-repeat formal CLI medians, warmup excluded, v14 vs v15:

| Mode | Input | v14 wall s | v15 wall s | v15 MPix/s | Bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| Default AVX2 | clover | 1.043135 | 1.047833 | 1.001 | 1586564 |
| Default AVX2 | Kodak 19 | 0.829041 | 0.762442 | 0.516 | 436948 |
| Default AVX2 | Tecnick 023 | 0.942431 | 0.939784 | 1.532 | 1530900 |
| Attempted AVX512, actually default | clover | 1.036543 | 1.089544 | 0.962 | 1586564 |
| Attempted AVX512, actually default | Kodak 19 | 0.759930 | 0.790167 | 0.498 | 436948 |
| Attempted AVX512, actually default | Tecnick 023 | 0.909522 | 0.953726 | 1.510 | 1530900 |

Reports: `main-v15-default-repeat`, `main-v15-avx512-repeat`. Both harnesses
removed inherited `RS_JXL_*` variables. Thus the AVX512-labelled run exercised
the default backend, not AVX512. Its slowdown MUST NOT be attributed to AVX512.
The earlier speed/regression interpretation is withdrawn. Raw reports remain
unchanged for auditability. Times from separate A/B sessions also must not be
treated as a paired isolated SIMD measurement.
Every output passed official djxl pixel equality and sizes are unchanged.
The 1.001 result has effectively no safety margin and is not stable attainment.

The v15 single-stage lab experiment was rejected by driver validation before
encoding: coarse consensus incorrectly remained enabled without staged probes.
That experimental option is fixed in v16 with an actual encode regression test;
the failed `pilot-v15-stage-ab` run is not an accepted benchmark.

### v16 Experiments Under Validation

- Exact RCT output-expression costs are shared across all 19 candidates:
  57 channel scans become 14, and fully cached candidates need no temporary
  transform (18 transforms become 5). Full signed-i32 wrapping expressions,
  candidate order and cost-combination order are covered by reference tests.
- Weighted prediction and history update share coordinate validation and row
  offsets in internal callers. Public alternating predict/update behavior is
  unchanged. Tests compare returned values and complete state, including flat
  mismatches, wrapping histories, stripe warm-up and exhausted coordinates.
- `GatherOptions::concat_batches` / lab `--concat-batches` defaults OFF. It only
  applies to one parallel terminal wave with records >=99% of weight and target
  column payload <=64 MiB. Worker-local deduplication remains; cross-batch rows
  retain their weights. Weighted integer histograms are preserved, but
  record-count-dependent training heuristics may change. This is not an RSS cap.
  All other cases use the original merge. Per-wave profiling now separates
  `gather.batch_collect` from `gather.append`.
- `PilotOptions::single_stage` / lab `--pilot-single-stage` defaults OFF. It
  bypasses coarse screening, not fine candidate families or final output
  validation. A real encode regression covers its driver-option compatibility.

Core gate at this stage: 1762 passed / 13 ignored. Final performance, file size
and official-decoder compatibility are pending; these are not claimed wins.

The v16 natural/screen matrices subsequently verified all four combinations on
ten inputs with official djxl and original integer samples. All sizes matched
the preceding accepted sizes. Single-stage probing is NOT a global default:
it slowed clover from 1.055 to 1.263 s and Tecnick from 0.853 to 1.033 s.

With staged probes and bounded sample concatenation (single-round diagnostics):

| Input | Wall s | Wall MPix/s | Bytes | bpp |
| --- | ---: | ---: | ---: | ---: |
| clover | 0.9836 | 1.066 | 1586564 | 12.104523 |
| Kodak 19 | 0.6343 | 0.620 | 436948 | 8.889730 |
| Tecnick 023 | 0.8770 | 1.642 | 1530900 | 8.505000 |
| grayscale 500 | 0.3355 | 0.745 | 90224 | 2.887168 |
| paper 9bde | 0.9738 | 1.495 | 145969 | 0.801957 |
| paper 1eb | 2.5572 | 1.409 | 351934 | 0.781659 |
| editor 3dd | 1.5446 | 1.384 | 77651 | 0.290618 |
| web news | 1.2020 | 1.598 | 57748 | 0.240460 |
| icon 512 | 0.3466 | 0.756 | 27736 | 0.846436 |
| game Blobby | 0.8573 | 0.557 | 190018 | 3.184194 |

Reports: `pilot-v16-natural-matrix`, `pilot-v16-screen-matrix`. The larger
representative images pass in this round, but the small holdouts still fail;
this is not an all-images floor. Instrumented Kodak with concatenation spends
29 ms gathering (21 ms workers, 8 ms append), versus about 160 ms previously.
Its full candidate takes 338 ms, but front-end work still takes about 260 ms.
The next controlled experiments target probe scheduling, small initial finalist
waves and allocation-free exact ANS population-cost evaluation.

### v17 Small-Image Scheduling And Exact ANS Costs

`fast_population_cost` now uses bounded stack scratch while retaining all three
normalization shifts and the original f32 selection order. Cluster pair scoring
no longer clones a combined histogram for every comparison. Core tests: 1781
passed / 13 ignored; CLI 86 / 2 ignored; lab 28 passed; integration targets check.

The pilot helper fills a missing gather configuration with bounded terminal
concatenation only for non-literal inputs. Explicit caller settings are retained.
The lab always supplies explicit gather settings, so `--concat-batches` is still
needed there. Ordinary source encoding defaults are unchanged.

Four scheduling experiments remain explicit: independent probe roots, a single
fine probe stage, quiet arena notifications, and a bounded initial full-candidate
wave. The initial wave handles at most three eligible consecutive candidates,
only through 2^19 pixels and the existing 256 MiB scratch estimate. It completes
all submitted candidates before committing real sizes in canonical order; probe
scores are never used as byte bounds. This estimate is not a hard RSS limit.

`pilot-v17-small-matrix/report.json`, one round, 16 threads, official djxl verified:

| Input | Serial-initial wall s | Initial-wave wall s | Wave MPix/s | Bytes | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak 19 | 0.6561 | 0.6449 | 0.610 | 436948 | 8.889730 |
| game Blobby | 0.9017 | 0.5867 | 0.814 | 190018 | 3.184194 |
| icon 512 | 0.3762 | 0.3471 | 0.755 | 27736 | 0.846436 |
| grayscale 500 | 0.3330 | 0.2234 | 1.119 | 90224 | 2.887168 |
| clover | 0.9554 | 0.9982 | 1.050 | 1586564 | 12.104523 |

The clover image is outside the initial-wave gate: its difference is run noise,
not evidence of an algorithm change. Independent probes regressed the game case
to 1.010 s. Quiet notifications did not give a consistent benefit. No universal
1 MPix/s claim follows: Kodak, icon and game still fail the complete-call target.

Instrumented Kodak: full candidate 347 ms, front end about 258 ms; train_prepared
longest 159 ms and gather longest 31 ms. Inclusive times across parallel tasks
must not be added. The full candidate itself is already over 1 MPix/s, but the
complete search is not. Game initial-wave timing remains limited by concurrent
MA sweeps and entropy clustering; its exact winner is retained.

### Rejected Scoring Shortcuts

`pilot-v17-scoring` compares the existing clipped incremental scoring, count-log
tables and dense property bins. None provided a reliable overall speedup.
`pilot-v18-search` adds lab-only `--shannon`, `--quantile-splits 32|64`, and
`--refine-quantiles`. These are alternative search algorithms, not lower effort
or a claim of C++ bitstream identity. All 20 outputs passed original sample
equality with official djxl. No alternative is promoted by these results:

- Kodak quantile32: 436843 bytes vs 436948, wall 0.636 s vs 0.668 s; still slow.
- Icon quantile32: 28045 bytes vs 27736 (+1.11%), wall 0.350 s vs 0.352 s. Reject.
- Game quantile64: 189993 bytes vs 190018, wall 0.550 s vs 0.599 s. An isolated
  candidate result, not sufficient justification for a global default.
- Clover quantile32: 1584828 bytes vs 1586564, wall 1.041 s vs 0.985 s. Slower.
- Shannon and refined quantiles did not consistently improve complete-call time.

Next work targets repeated short LZ comparisons, small-node MA overhead and
possible overlap of useful full-candidate execution with front-end analysis.

### v17 Repeated CLI Gate

`main-v17-repeat/REPORT.md` compares v16 and v17, one validated warmup plus three
alternating repetitions. Every output passed original sample equality (including
the 4096-square RGBA image and the 16-bit sentinel) or exact JPEG reconstruction.
All sizes are unchanged. The new non-literal helper gather default is included.

| Input | v16 wall s | v17 wall s | v17 MPix/s | bpp |
| --- | ---: | ---: | ---: | ---: |
| paper 9bde | 1.113519 | 1.081658 | 1.346 | 0.801957 |
| paper 1eb | 2.697119 | 2.809394 | 1.282 | 0.781659 |
| editor 3dd | 1.425457 | 1.537007 | 1.391 | 0.290618 |
| web news | 1.251436 | 1.287971 | 1.492 | 0.240460 |
| Tecnick 023 | 0.954856 | 0.909085 | 1.584 | 8.505000 |
| clover | 1.093251 | 1.007966 | 1.040 | 12.104523 |
| Kodak 19 | 0.812801 | 0.664653 | 0.592 | 8.889730 |
| RGBA 4096 | 3.406178 | 3.301370 | 5.082 | 0.000613 |
| JPEG vpyekp | 0.354525 | 0.383409 | 6.410 | 0.301927 |
| 16-bit 64-square sentinel | 0.167921 | 0.162087 | 0.025 | 24.550781 |

Some screenshot/JPEG medians increased in this run; this is not an across-the-
board speedup. The individual runs show substantial variation (editor v17
1.290--1.602 s). Repeat same-binary controls before attributing modest changes
to one kernel. The 64-square sentinel is for correctness, not a large-image
throughput test; process and setup costs dominate its 4096 pixels.

### Explicit Experiment Environments

The development matrix now accepts `--encoder-env RS_JXL_NAME=value`; the
checkpoint harness accepts separate `--before-env` / `--after-env` assignments.
All inherited codec variables are still removed, but explicit variables are
passed only to that encoder and recorded in the report. Decoder environments
remain clean. Do not enable benchmark experiments merely by setting PowerShell
environment variables. Example: `--encoder-env RS_JXL_MA_RCP=1`.

The reciprocal experiment changes only AVX2/FMA rational-log division using one
Newton correction. It preserves integer counts, the probability floor and the
small-histogram table; floating scoring may choose different lossless trees.
It is opt-in and unaccepted until end-to-end size, speed and pixel checks.

### Measurement Contract And v19 Diagnostics

The user now accepts excluding process startup, input decoding and file I/O.
The primary throughput is **all encoder-call work**, including scanning, probes,
candidate selection, pool setup and final encoding. The lab already times that
boundary and reports `seconds` / `mpix_per_second`; process wall measurements
remain secondary. Do not subtract the pilot, or report only the winning trial.
For tiny correctness sentinels, report absolute milliseconds as well: the
64-square image contains only 0.004096 MP and is not a large-image speed floor.

v19 adds an explicit automatic-full/front overlap experiment, with owned result
handoffs, a shared bounded arena, and the existing scratch estimates. A completed
automatic candidate is retained once, never encoded again or used as a fabricated
probe score. Error paths drain submitted work. Overlap changes the critical path;
subtracting the full-candidate time from elapsed time no longer measures the front.

`pilot-v19-overlap/report.json` is a single-round diagnostic, not a stable median:

| Input | Initial-wave call s | Overlap call s | Output bytes |
| --- | ---: | ---: | ---: |
| Kodak 19 | 0.6114 | 0.4077 | 436948 |
| game Blobby | 0.5398 | 0.8944 | 190018 |
| icon 512 | 0.3240 | 0.5077 | 27736 |
| grayscale 500 | 0.2184 | 0.2944 | 90224 |

Unconditional overlap is rejected. Kodak benefits but the other classes lose
their initial full-candidate wave and regress. v20 therefore tests a cheap scan
gate, without running another image scan. Its separate quarter-sampling experiment
changes only probe learning, records the adaptation, and preserves full candidate
learning and hard sample/leaf resource limits. Neither is an accepted default yet.

The explicit-environment SIMD reruns are `pilot-v19-avx512-overlap` and
`pilot-v19-rcp-overlap`. AVX512 and reciprocal scoring show mixed results rather
than a consistent complete-call benefit, so neither becomes a default. All v19
matrix outputs passed official djxl comparison with original samples. Core v19:
1807 passed / 13 ignored; CLI 86 / 2 ignored; lab 30 passed; harness 23 passed.

### v20: Reject Global Probe Downsampling

Core tests: 1814 passed, 13 ignored; CLI/lab: 86/30 passed, 2 CLI ignored.
`pilot-v20-front-matrix` completed 25 verified outputs. The five variants separate
initial waves, content-gated overlap, quarter probe learning and earlier
14-predictor cache admission. Cache admission has no stable benefit and stays off.

Quarter sampling initially improved Kodak19 (539 to 377 ms) and Blobby (533 to
458 ms), with unchanged sizes. It also made the icon slower (325 to 535 ms):
full encodes increased from one to three. Held-out Kodak01/06 showed the same
problem. On paper1eb it raised complete-call time from 2.46 to 7.94 seconds with
no size gain. **Do not enable global quarter sampling.** Cheaper probes are not
an optimization if their ranking causes extra expensive full trials.

Reports `pilot-v20-holdout` and `pilot-v20-routing` keep all raw results. The
first is intentionally marked failed: after 12 verified outputs, the benchmark
PNG reader rejected the sub-byte `3dbrick2.png` fixture before encoding. This is
a harness input-format limit, not a decoder mismatch. The second completed all
27 outputs, including the remaining screenshots, an RGB game and 16-bit sentinel.

Examples from the routing run (single rounds, not medians):

| Input | Initial call ms | Initial MPix/s | Overlap call ms | Bytes |
| --- | ---: | ---: | ---: | ---: |
| Kodak01 | 482.7 | 0.815 | 373.1 | 454216 |
| Kodak03 | 719.2 | 0.547 | 725.2 | 327569 |
| Kodak06 | 478.7 | 0.821 | 388.8 | 418729 |
| Absolver 1200x675 | 1950.3 | 0.415 | 1872.0 | 717614 |
| paper1eb | 2460.8 | 1.464 | 2450.9 | 351934 |
| 16-bit 64-square | 104.6 | 0.039 | 112.4 | 12570 |

The tiny sentinel remains low-throughput even without process overhead. It runs
nine full candidates in the legacy small-input path; its approximately 105 ms
call is not proof of a low large-image throughput floor. In contrast, Absolver
is a real remaining bottleneck: instrumented automatic full takes 392 ms, but
a losing YCoCg MA trial consumes 1207 ms before its real output bound rejects it.
Its three probe scores are [27898, 36416, 47270], against the automatic leader's
[26060, 35349, 47677]. Weak regional evidence admits costly unsuccessful work.

Detailed per-predictor MA timings in the v20 profiles are intrusive: roughly
0.4--1 million scopes on the small cases, several million on Absolver. Each
scope updates a thread-sharded map, and those updates are included in parent
timings, not the child itself. Use these logs to locate work, not as accepted
speed measurements. The next experiment restricts fine instrumentation to an
explicit detail switch and tests near-tie pilot decisions separately from the
integer histogram kernel. No final-size bound is inferred from probe scores.

### v21 Holdout Findings

Core 1826 passed / 13 ignored; CLI 86 / 2 ignored; lab 31 passed;
all integration targets compile; benchmark-harness tests 32 passed.
The lab now stops its call timer before printing search/representation receipts.

`PilotOptions::near_tie_consensus` / `--pilot-near-tie-consensus` is a separate
opt-in. For e9 automatic leaders, it rejects non-Zero alternatives that have no
regional estimated advantage greater than 1% and are at least 0.5% worse in total.
It is independent of model correction. e10, explicit non-automatic leaders,
static/learned literal witnesses and real output bounds are preserved. This is
a work-selection heuristic, not a bound on the final compression ratio.

`pilot-v21-select-matrix`: Absolver call fell from 1829 to 674 ms (1.202 MPix/s),
with the same 717614-byte output. `pilot-v21-kodak24`: all 48 outputs verified and
all 24 sizes unchanged. However Kodak11/21 regressed from 1112/1037 to 1533/1472
ms. The already completed overlapped automatic result disabled the initial wave;
the pilot then encoded two losing contenders serially. These are real remaining
regressions, not startup costs. No v21 selection default is promoted.

`pilot-v21-kernels` tests existing predictor/property deduplication and sparse
histograms; neither consistently removes the slow path. The sparse variant grows
the icon by nine bytes for only a small timing difference, so it is not promoted.
`pilot-v21-pair14` explicitly enables `RS_JXL_MA_CACHE_PAIR14=1`: the new integer
cache collector processes fourteen identity predictor columns in seven paired
passes, retaining all original counts, cache ceilings and subtraction behavior.
It has no stable complete-call gain in this round and remains off by default.

Per-predictor profiling now additionally requires `RS_JXL_PROFILE_DETAIL=1`;
`RS_JXL_PROFILE=1` alone retains broad stages/cache timings. Neither variable
belongs in a speed acceptance run. Both the pair-kernel switch and detail switch
are cached per process, so A/B experiments must start fresh processes.

### v22/v23: Reuse Completed Work Without Serializing Challengers

Measurement contract: the primary metric is the complete encoder call, including
scan, probes, pool setup, candidate selection and final encoding. Only CLI startup,
input image decoding and output I/O are excluded. Report latency as well as
MPix/s: a 64x64 image taking 110 ms is not evidence of large-image throughput.
These experiments use 16 threads, source effort 9, on the same Windows host.
They compare Rust scheduling variants, not a new official-libjxl benchmark.

v22 fixes a real scheduling defect after automatic/probe overlap. A completed
automatic result is committed once, retained as an incumbent and removed from
pending work. The remaining rank leader and its eligible peers go through the
resource-bounded parallel finalist path. They use the actual completed output's
byte count for early termination, never an estimated probe byte count. Pending
rejections wait until the preceding wave commits, so a new winner can change the
comparison reference. Allocation limits and error handling remain in force.

The separate empirical filter for overlapped continuous-tone trials retains
automatic and both literal witnesses. Other challengers need at least 2% total
estimated improvement, or at least 5% improvement in two material regions (each
at least 10% of the automatic score). Incomplete/zero/mismatched vectors cannot
justify rejection. This is not a proof that final size stays within 0.25% of an
exhaustive search result. New corpora still need independent size measurements.

`pilot-v22-kodak24` verified 48 outputs, with unchanged size for every Kodak image.
The two earlier slow outliers no longer run serial losing candidates: Kodak11
was 1292 -> 506 ms, Kodak21 1112 -> 438 ms in this single-round diagnostic.
`pilot-v22-repeat` then verified 112 outputs across 14 mixed cases, with one
verified warmup and three measured repeats per variant. All output sizes were
unchanged in that matrix. Timing variation was not uniformly positive: Blobby
was 536 -> 581 ms, the editor screenshot 1254 -> 1328 ms, and RGBA4096
3037 -> 3262 ms. Do not report only the favorable cases.

The additional `pilot-v22-holdout` exposed a compression regression despite
passing pixel checks: `20XX_Menu_picture.png` changed from 83134 to 83748 bytes
(+0.739%), saving only about 20 ms. Its automatic crop scores were
[13153, 8474, 2390] and neighborhood scores [13274, 8680, 2455], yet neighborhood
won on the complete image. The near-tie rule must not prune this flat-content
alternative. v23 gates near-tie screening on the existing continuous-tone scan;
flat/low-color images keep the prior checks, without another scan or a filename
exception. A deliberately scan-less custom driver retains its explicit opt-in
behavior. A synthetic regression test reproduces the probe/full rank reversal.

The content-pilot helper and CLI `--source-pilot` preset enable bounded initial
waves, content/resource-gated automatic overlap and content-gated near-tie
screening. Ordinary source search is unchanged. Quarter probe sampling, earlier
cache admission, pair14, reciprocal scoring and AVX512 scoring stay disabled.
No final effort, MA sampling fraction, sample/leaf ceiling, or decoder contract
is weakened.

#### v23 Repeated Encoder-Call Measurements

`out/encoder-scout-20261025/pilot-v23-repeat/report.json` records 18 inputs,
144 independently verified outputs, one warmup plus three repeats, with variant
order alternating. Both variants use the same v23 binary and full backend
effort. `initial` has bounded initial waves; `select` additionally enables the
accepted overlap and content-gated screening. Times below are medians of complete
calls, not averages of per-image rates. All 18 sizes were unchanged, including
the repaired 20XX case. No instrumentation environment flags were enabled.

| Input | Initial ms | Select ms | Select MPix/s | Bytes | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak11 | 1182.1 | 481.3 | 0.817 | 407439 | 8.289368 |
| Kodak21 | 1170.3 | 463.5 | 0.848 | 435181 | 8.853780 |
| Kodak19 | 636.5 | 441.7 | 0.890 | 436948 | 8.889730 |
| Absolver, 1200x675 | 2052.8 | 739.4 | 1.096 | 717614 | 7.087546 |
| Mail icon, 512x512 | 367.1 | 374.4 | 0.700 | 27736 | 0.846436 |
| Blobby, 797x599 | 571.2 | 598.4 | 0.798 | 190018 | 3.184194 |
| paper9bde, 1264x1152 | 976.2 | 1005.6 | 1.448 | 145969 | 0.801957 |
| paper1eb, 2560x1407 | 2620.6 | 2615.6 | 1.377 | 351934 | 0.781659 |
| Editor screenshot | 1241.5 | 1238.5 | 1.726 | 77651 | 0.290618 |
| Web screenshot | 1208.9 | 1157.9 | 1.659 | 57748 | 0.240460 |
| Tecnick023, 1200x1200 | 841.9 | 817.5 | 1.762 | 1530900 | 8.505000 |
| Clover, 1024x1024 | 914.4 | 946.9 | 1.107 | 1586564 | 12.104523 |
| RGBA4096 | 3199.1 | 3071.9 | 5.461 | 1286 | 0.000613 |
| 16-bit 64x64 sentinel | 108.9 | 111.8 | 0.037 | 12570 | 24.550781 |
| Address-book icon | 366.8 | 378.9 | 0.692 | 40045 | 1.222076 |
| Open-document icon | 289.4 | 289.8 | 0.905 | 12852 | 0.392212 |
| 20XX, 634x406 | 430.8 | 435.1 | 0.592 | 83134 | 2.583767 |
| Gray, 500x500 | 224.1 | 224.8 | 1.112 | 90224 | 2.887168 |

This is not a universal 1 MPix/s guarantee. Every >=1 MP image in this mixed
matrix exceeds that rate, as does Absolver, but small inputs still have fixed
training/search overhead. The 64x64 sentinel's 112 ms is real encoder work, not
CLI startup. Preserve the small latency regressions in the record: Blobby +27 ms,
Clover +33 ms, paper9bde +29 ms. These short runs do not establish significance
for small timing differences; they do establish the large multi-trial savings
on Kodak11/21 and Absolver without the observed size loss.

v22 core tests passed 1834 with 13 ignored; CLI 86 with 2 ignored, lab 31 and
Python harness 32 passed. After the v23 content gate, all 269 scheduler tests
passed, including the new low-color rank-reversal regression. The final full core
suite passed 1835 tests with 13 ignored (16.25 seconds); all integration targets
also compiled. Release compilation and `git diff --check` passed. Binary hashes
in reports identify executables only;
codec acceptance compares original samples, including hidden RGB and 16-bit
precision. Equality of compressed JXL bytes is not an optimization requirement.

#### Additional v23 Acceptance And Limits

`pilot-v23-kodak24/report.json` completed 48 verified outputs with no size changes
on all 24 photos. This single-round holdout still exposes Kodak02: 1000 -> 1030 ms,
0.382 MPix/s at 768x512, 403664 bytes, 8.212565 bpp. The conservative content gate
preserves its additional candidate work. It is a remaining hotspot, not startup
overhead, and a reason not to claim that all small photographs meet 1 MPix/s.

`pilot-v23-cli/report.json` checks the actual CLI/helper preset, not just lab
flags. It compares frozen v21/v23 CLIs, with one verified warmup and three measured
repeats per version. All 56 outputs passed independent official decoding: PNG
samples match exactly and all reconstructed JPEG bytes match the original.
Temporary encoded/decoded files were removed by the harness. These figures
include process startup and I/O, so they are secondary to the call table above:

| Input | v21 wall ms | v23 wall ms | v23 bytes | bpp |
| --- | ---: | ---: | ---: | ---: |
| Kodak11 | 1672.0 | 642.1 | 407439 | 8.289368 |
| Absolver | 1999.5 | 781.9 | 717614 | 7.087546 |
| paper9bde | 1062.5 | 1090.9 | 145969 | 0.801957 |
| RGBA4096 | 3782.8 | 3518.0 | 1286 | 0.000613 |
| 16-bit 64x64 | 164.4 | 183.2 | 12570 | 24.550781 |
| 20XX | 791.1 | 567.5 | 83134 | 2.583767 |
| JPEG wallhaven-vpyekp | 368.3 | 368.8 | 92752 | 0.301927 |

The three final reports contain 248 verified encodes (240 PNG pixel checks,
8 JPEG original-byte checks), across 39 distinct PNGs and one JPEG. All observed
sizes match their respective baseline sizes. This is targeted mixed-content
acceptance, not a full-corpus or cross-platform benchmark. Some sub-second cases
remain below 1 MPix/s; no timing or compressed-size guarantee is inferred from
this finite corpus. The ordinary source-search policy remains available; the
new defaults are confined to the explicitly selected content-pilot preset.
