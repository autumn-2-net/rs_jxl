# Source References, Delta Palette and Content Features

## Scope

Three independent encoder experiments, based on `8de84d7`. They retain standard
JPEG XL syntax and do not change production defaults. Distribution leaf offset
remains opt-in. No decoder algorithm or custom predictor is introduced.

The comparison uses complete native source-e9 candidates, not the outer
multi-candidate scheduler. All variants in a comparison share the non-experimental
options. Release builds, eight encoder threads, serial trials, rotating variant
order. Native timings exclude process startup, image IO and independent decoder
checks. The timed region is `rs_jxl::encode`; projecting native options and
serializing receipts are also outside it. Other local workloads are not stopped;
small timing differences are not
treated as demonstrated speedups without repeated controls.
Palette timings include opt-in telemetry, including delta-index scans; they are
instrumented timings, not an uninstrumented production-speed claim. Decoder
provenance is an explicitly supplied local official build plus a recorded hash,
not independent binary authentication by the harness.

Correctness means official libjxl 0.12 `djxl` reproduces every original integer
sample, including RGB under alpha zero. JXL byte/hash identity is not required.
Original/binary hashes are provenance and mutation checks only. Larger files,
slower encodes, inactive tools and declined candidates remain in the reports.

## Frozen Inputs

`out/representation-study-20261113/` contains the frozen manifests and reports.
`study.json` has 144 original PNGs, seed 20261113:

| Stratum | Images |
| --- | ---: |
| Document and web | 40 |
| Manga | 24 |
| Game and UI | 24 |
| Icon renders | 24 |
| Natural photographs | 32 |

There are 72 calibration and 72 holdout images. Original hashes and declared
source groups are disjoint across the two sets. This is not a claim of perceptual
deduplication or never having benchmarked these images before: historical
overlaps are recorded in `protocol.json`. Images are not resized or cropped for
the compression comparison. The corpus includes the 94.835 MP Graveyard poster.

`known12.json` is a separate historical Source Patch positive-control set.
`patch24.json` is a preselected, stratified holdout subset, excluding known12.
`sensitivity24.json` is separately sampled from calibration before its results,
seed 20261114: document/web 7, game/UI 4, icons 4, manga 4, photographs 5.

An initial full-frame run on all 144 images was interrupted on the large poster.
Forcing full-frame disables ordinary native grouping and made this a stress
test. `palette-full-1024.json` preserves its 25 complete cases/175 verified outputs
and interruption note; it is NOT a completed corpus result. Only that benchmark
and its child were stopped, and its temporary images were removed. The primary
study instead retains native grouping on the SAME 144 originals. Full-frame and
capacity controls use the preselected calibration subset, not a wins-only set.

## Source Reference Selection

`SourcePatchConfig.representative` is an experimental native option:

| Mode | Selection |
| --- | --- |
| `First` | Existing minimum `(x,y)` representative, default |
| `Medoid` | Lowest sampled absolute difference to background-relative instances |
| `MinEntropy` | Lowest approximate atlas plus occurrence residual coding cost |

The existing detector, quantized fingerprint grouping, occurrence lists, stable
group ordering and atlas rectangle packing are unchanged. Each occurrence keeps
its own detected background. The selected atlas uses exact signed integer
differences; the complete signed residual frame is rebuilt and replay-checked.
Extra channels are untouched by Add. Floating color falls back to First and is
reported explicitly rather than treating float words as integer samples.

The entropy proxy uses JXL signed packing, HybridUint(4,1,2), extra-bit counts,
and empirical symbol entropy. Per channel it considers Zero, Left, Top and JXL
clamped Gradient. Atlas candidates are background-relative with zero padding;
observations retain ORIGINAL background and global neighbor values before
subtracting the candidate. Stripping both backgrounds would bias Zero and
boundary prediction and is covered by a regression test.

This is an isolated-occurrence proxy, not final ANS size. It omits interactions
with overlapping groups, right/bottom exterior prediction, packed-atlas neighbors,
full MA trees, RCT, LZ77 and entropy headers. Full encoded byte counts decide
whether a proposal actually helps; no proxy-score decrease is called a byte win.

Default experimental sampling budgets are 8 candidates, 16 scoring occurrences,
and 128 pixel locations per group. Zero requests exhaustive work for that axis;
it is not an input/decoder restriction. The deeper control uses 32/64/0, examining
every pixel of each existing patch without enlarging the detector's rectangles.
First is always eligible and wins score ties. Sampling is deterministic and
includes endpoints. `sampled_pixels` counts stencil locations per group, not all
candidate-by-occurrence operations. `selection_seconds` measures only selection;
complete encode time includes origin retention and all downstream effects.

`emitted_plans` records successful plan construction and replay validation. In a
successful single-encode `patch_lab` invocation, no later fallback discards that
plan, so it also demonstrates emission. It is not a generic decoder/frame counter.

## Delta Palette Ablation

The seven independent modes are Absolute, ExactDelta(Gradient/Left/Top), and
ExactZero(Gradient/Left/Top). Experiments use the same capacity, grouping and
non-mode options within each comparison. The primary capacity is 1024; separate
full-frame controls compare 1024 and 10000 on sensitivity24.

An invocation-owned optional `PaletteReceipt` records global and local policy
attempts: requested/actually attempted kernels, delta production, absolute
fallback, policy rejection, acceptance and references to delta columns.
Configuration alone is not activation. A zero column that is never referenced
is not counted as a used delta tool. Full-frame still has local AC palettes.
Receipt counting is disabled in ordinary encodes without a sink.

Accepted preparation is not a file-size win. Local offsets are physical channel
positions, not original RGB stream IDs; reference counts can count pixels at
multiple transformations and are not unique source coverage. Failed encodes can
have intermediate events from subsequently rolled-back preparations.

## Input-Only Routing Hypothesis

This is an offline C61-inspired content-analysis experiment, not a C61 codec
port and not a new default scheduler. It uses 21 input-only features: equality,
flatness, directional differences, sampled color cardinality, and scalar/tuple
residual statistics for JXL-compatible Left, Top and clamped Gradient. Invisible
RGB and alpha are retained. No file sizes, output sizes, path, hash, source family
or encoder timings enter the feature vector.

Features use up to nine seeded 64x64 original-resolution windows. Original-image
decoding/validation and feature calculation are timed separately. This Python
offline extractor is NOT a claimed cheap native implementation; its full decode
cost is included when estimating end-to-end routing overhead.

A constant route or one-threshold decision stump minimizes total encoded bytes
on calibration only. Baseline wins equal byte counts. Holdout labels do not tune
thresholds. Evaluation reports total bytes, per-image regret, false skips/enables,
and both exhaustive-search and single-selected-route time estimates. An oracle's
smallest files do not imply those files can be chosen for free.
This first classifier compares only the seven measured palette routes. It does
not claim a complete router over Source Patch, cross-channel MA or other tools.

## Reproduction

Build examples without changing the production CLI defaults:

```powershell
cargo build --release -p rs-jxl-cli --example patch_lab --example palette_delta_lab
python -B tools/benchmark_patch_representatives.py --help
python -B tools/benchmark_palette_delta.py --help
python -B tools/analyze_representation_routes.py --help
python -B tools/summarize_representation_study.py --help
```

Benchmarks accept prebuilt executables and an explicitly supplied official
`djxl`. They verify manifest-declared input hashes before trials, snapshot
original pixels, validate every encoded output, and remove temporary images.
Reports are create-only. The scripts do not build codecs implicitly.

Python 3.10 was used with NumPy and Pillow; multichannel 16-bit checks additionally
use the existing optional OpenCV oracle, and plots require Matplotlib. Source
fixtures exercise 8/16-bit gray, GA, RGB and RGBA; the 144 real originals are
8-bit samples after lossless packed-PNG expansion, not a 144-image 16-bit claim.

The following records the measured configuration. Choose NEW `--out` paths for
reruns; existing reports are deliberately never overwritten:

```powershell
$lab = 'out/representation-study-20261113'
$djxl = '../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe'
python -B tools/benchmark_palette_delta.py --encoder "$lab/palette-delta-lab.exe" --decoder $djxl --manifest "$lab/study.json" --palette-colors 1024 --threads 8 --repeats 1 --timeout 900 --out "$lab/palette-native-1024.json"
python -B tools/benchmark_patch_representatives.py --encoder "$lab/patch-representative-lab.exe" --decoder $djxl --manifest "$lab/patch24.json" --effort 9 --threads 8 --candidates 32 --occurrences 64 --pixels 0 --parallel-gather --repeats 1 --out "$lab/source-holdout24.json"
python -B tools/benchmark_palette_delta.py --encoder "$lab/palette-delta-lab.exe" --decoder $djxl --manifest "$lab/sensitivity24.json" --full-frame --palette-colors 1024 --threads 8 --repeats 1 --timeout 900 --out "$lab/palette-control24-1024.json"
python -B tools/benchmark_palette_delta.py --encoder "$lab/palette-delta-lab.exe" --decoder $djxl --manifest "$lab/sensitivity24.json" --full-frame --palette-colors 10000 --threads 8 --repeats 1 --timeout 900 --out "$lab/palette-control24-10000.json"
python -B tools/analyze_representation_routes.py extract --manifest "$lab/study.json" --out "$lab/features.json"
python -B tools/analyze_representation_routes.py analyze --manifest "$lab/study.json" --features "$lab/features.json" --report "$lab/palette-native-1024.json" --baseline absolute --out "$lab/palette-native-analysis.json"
```

Known12 uses the same Source command with its own manifest: 8/16/128 budgets and
two repetitions including Off, then 32/64/0 with one repetition of only First,
Medoid and entropy. The Source synthetic run uses `--synthetic`, 0/0/0 budgets
and one repetition; the 18-case palette synthetic run uses `--synthetic`,
`--full-frame`, capacity 1024 and one repetition. All use eight encoder threads.

Summaries show arithmetic mean per-image bpp, pixel-weighted bpp, total bytes,
summed native time and MPix/s separately. Charts use mean bpp/time and total
MiB/time, without error bars. Incomplete cohorts are labeled partial and cannot
silently stand in for the full frozen dataset.

## Results

### Source Positive Controls

All 96 official-decoder checks passed for known12 (four routes, two repetitions).
The deeper 32/64/0 run added 36 checks (three routes, one repetition). Both runs
use the same encoder, originals and unchanged detector; deep results follow:

| Route | Total bytes | Native seconds | Mean image bpp | Wins/ties/losses vs First |
| --- | ---: | ---: | ---: | --- |
| First | 2,517,598 | 28.090 | 0.861601 | baseline |
| Medoid | 2,517,523 | 27.680 | 0.861612 | 2/9/1 |
| MinEntropy | 2,518,305 | 28.317 | 0.862641 | 1/6/5 |

Medoid saves only 75 bytes (0.00298%); entropy adds 707 bytes (0.02808%). The
per-image oracle over these THREE routes saves only 500 bytes, and requires
encoding all three. Including the Off route answers a different question:
whether to use Source Patch at all, not which representative to use.

Of 1,143 retained groups, only 79 have different sampled representatives. Deep
Medoid changes nine groups; deep entropy changes 28. The default/deep Medoid
byte totals are identical. On the Paper9bde and gmessages regressions, the deep
run examines every candidate, occurrence and rectangle pixel, so their failure
cannot be attributed to sampling budgets. The entropy proxy decreases by about
1,141 bits overall while the complete files grow by 707 bytes. This demonstrates
an objective mismatch but does not isolate which downstream stage causes it.
Two-occurrence groups also tie under summed L1 Medoid regardless of sampling.

The 8/16-bit gray/GA/RGB/RGBA fixtures add 32 verified outputs and demonstrate
actual representative changes with exact invisible-RGB reconstruction. They
are activation/correctness controls, not evidence of broad compression gains.
Single deep repetitions and shared-host timings do not establish a speedup.

### Source Holdout

All 96 checks passed on the preselected 24-image holdout at 32/64/0. Source plans
were emitted on six images; 18 were inactive. There were 118 groups, 15 with
different sampled representatives. Medoid and entropy each changed six groups
across two images, preserving the same atlas area and dictionary positions.

| Route | Total bytes | Native seconds | Mean image bpp | Wins/ties/losses vs First |
| --- | ---: | ---: | ---: | --- |
| First | 12,889,533 | 26.945 | 4.160151 | baseline |
| Medoid | 12,889,530 | 27.087 | 4.160162 | 1/22/1 |
| MinEntropy | 12,889,493 | 27.123 | 4.160154 | 1/22/1 |

The 3/40-byte savings do not justify promoting either selection objective. This
does not prove that every possible reference-selection method is unhelpful;
these two objectives and the tested groups offer little practical benefit.
Restricting the denominator to the six emitted images does not change that
conclusion: First is 1,361,100 bytes, Medoid -0.000220%, entropy -0.002939%.
The oracle excluding Off saves 54 bytes (0.003967% of active-image bytes).
Only two images change: NOAA `5338.png` saves 41/54 bytes for Medoid/entropy;
Lilith `7106.png` adds 38/14 bytes respectively. The other 22 tie First.

### Native Palette, 144 Images

All 1,008 outputs passed official exact-pixel decoding, with no input/binary
mutation. The retained grouping request produced 96 groupwise and 48 full-image
encodes per mode. One original triggered libpng's known-incorrect-sRGB-profile
warning; it did not cause a sample mismatch.

| Mode | Total bytes | Change vs Absolute | Mean image bpp | Native seconds | Wins/ties/losses | Images using delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Absolute | 145,714,903 | baseline | 4.186932 | 165.653 | baseline | 0 |
| Delta Gradient | 150,811,288 | +3.498% | 4.353166 | 160.906 | 6/61/77 | 80 |
| Delta Left | 150,977,729 | +3.612% | 4.332132 | 161.537 | 4/66/74 | 76 |
| Delta Top | 150,322,351 | +3.162% | 4.343341 | 159.321 | 4/64/76 | 77 |
| Zero Gradient | 149,800,995 | +2.804% | 4.326641 | 174.786 | 10/67/67 | 67 |
| Zero Left | 148,543,702 | +1.941% | 4.306731 | 174.409 | 6/67/71 | 73 |
| Zero Top | 148,403,929 | +1.845% | 4.307445 | 176.005 | 7/67/70 | 71 |

Baseline pixel-weighted bpp is 2.755943 and throughput is 2.553 MPix/s. The
per-image seven-route oracle reaches 145,161,372 bytes (-0.380%), mean bpp
4.176125 and weighted bpp 2.745474. It improves 13 images and ties 131, at
1,172.617 seconds of ALL-candidate native work (7.079x baseline), not free choice.

This is not simply a failure to activate the kernels: Delta Gradient is used on
80 images and loses on 75 of those; Zero Gradient is used on 67 and loses on 65.
Some wins occur without any delta indices in the final preparations. Such wins
must not automatically be attributed to predictive palette representation:
requesting a different kernel can also change which absolute transform the
existing cost gate accepts or rejects.

This comparison is therefore a **palette-policy mode ablation**, not an isolated
replacement of identical accepted transform scripts. ExactDelta retries Absolute
when its kernel declines, but not when a produced delta candidate subsequently
fails the policy cost test. ExactZero inserts a delta column before costing,
including when no pixels match. A rejection changes subsequent channel-range
attempts. `inactive` does not mean unchanged behavior, and `fallback_only` does
not prove the same accepted absolute script as the baseline.

The oracle's 553,531-byte saving separates as follows:

| Winning output classification | Images | Saved bytes |
| --- | ---: | ---: |
| Delta used | 5 | 188,970 |
| Inactive | 6 | 115,168 |
| Absolute fallback only | 2 | 249,393 |

Thus 65.9% of savings are from outputs with no accepted delta table. Even the
remaining 34.1% is activation-associated, not causally isolated benefit. Every
fixed mode also regresses in aggregate when restricted to its delta-used images.
Examples:

- CLIC `f67f11de...`, Zero Gradient: -248,093 bytes, no accepted delta; absolute
  accepts fall from 13 to one.
- `6826.png`, Zero Gradient: -105,748 bytes, no palette accepted, versus five
  absolute accepts in baseline. Delta Left on the SAME image adds 1,958,990 bytes
  (+47.80%) with 139 accepted delta tables.
- `1550.png`, Delta Top: -112,005 bytes with 10 local delta tables; admission
  changes still prevent attributing the entire saving to one mechanism.
- `window_03.png`, Delta Gradient: a cleaner small signal, one global absolute
  palette replaced by one delta palette covering 262,144 pixels, saving 296 bytes.

These observations point toward candidate-cost/admission errors worth examining,
not proof that the Delta Palette representation itself cannot be beneficial.

Descriptive strata expose where the aggregate comes from. "Best fixed alternative"
is selected AFTER observing each category, not an implementable image router:

| Stratum | Images | Best fixed alternative | Byte change | Seven-route oracle change |
| --- | ---: | --- | ---: | ---: |
| Document/web | 40 | Zero Top | +3.1311% | -0.1319% |
| Game/UI | 24 | Delta Left | +0.2773% | 0.0000% |
| Icon renders | 24 | Delta Left | +21.8144% | -0.3199% |
| Manga | 24 | Zero Gradient | +0.1249% | -0.0271% |
| Natural photographs | 32 | Zero Gradient | -0.5543% | -1.1946% |

The natural-photo win is consistent with the palette-admission confound above;
it is not evidence that photographs generally benefit from delta tables. Category
labels are not features in the calibration/holdout routing experiment.

### Full-Frame Sensitivity Controls

The fixed calibration subset contains 24 originals and 60.054 MP. At capacity
1024, all 168 outputs pass official sample verification. Removing the native
grouping request does NOT turn the fixed delta policies into overall byte wins:

| Mode | Bytes, cap 1024 | Change vs Absolute | Native seconds | Mean image bpp |
| --- | ---: | ---: | ---: | ---: |
| Absolute | 20,129,788 | baseline | 101.805 | 4.052462 |
| Delta Gradient | 22,324,213 | +10.901% | 83.683 | 4.359156 |
| Delta Left | 22,382,997 | +11.193% | 83.357 | 4.312000 |
| Delta Top | 21,889,550 | +8.742% | 82.360 | 4.374965 |
| Zero Gradient | 20,272,064 | +0.707% | 103.317 | 4.168934 |
| Zero Left | 20,201,069 | +0.354% | 103.148 | 4.181026 |
| Zero Top | 20,171,477 | +0.207% | 103.606 | 4.175219 |

The oracle improves three images and ties 21, saving 119,438 bytes (0.593%) at
661.277 seconds of all-candidate work (6.495x baseline). Whole-144 totals must
not be compared directly with these subset totals.
Restricting the PRIMARY report to the exact same 24 hashes gives native-grouping
Absolute 21,319,928 bytes, 26.619 seconds, mean bpp 4.151293. Full-frame Absolute
is 5.582% smaller but takes 3.825x as much native time. This is an existing
grouping-route effect, not Delta improvement or proof of a new production win.

Increasing capacity to 10000 on the SAME 24 full-frame originals also passes all
168 checks, but does not reverse the result:

| Mode | Bytes, cap 10000 | Change vs Absolute | Native seconds | Mean image bpp |
| --- | ---: | ---: | ---: | ---: |
| Absolute | 19,999,994 | baseline | 100.309 | 4.047856 |
| Delta Gradient | 22,336,561 | +11.683% | 82.892 | 4.372449 |
| Delta Left | 22,225,676 | +11.128% | 85.659 | 4.305457 |
| Delta Top | 21,741,402 | +8.707% | 83.913 | 4.368783 |
| Zero Gradient | 20,200,489 | +1.002% | 102.392 | 4.166423 |
| Zero Left | 20,112,574 | +0.563% | 99.896 | 4.177908 |
| Zero Top | 20,076,416 | +0.382% | 100.542 | 4.171879 |

Absolute improves by 129,794 bytes versus capacity 1024 (0.645%). The seven-route
oracle again improves only three images, saving the SAME 119,438 bytes (0.597%)
at 655.603 seconds of all-candidate work (6.536x baseline). Thus the tested
capacity restriction and grouping request do not alone explain the weak delta
results. This is not a proof about untested capacities or alternative admission
rules.

### Input-Only Routing Results

The feature set and source-disjoint split were frozen before fitting. Calibration
selects this one-threshold rule:

```text
top_residual_mean_abs <= 0.004363809567036692 -> ExactZero(Gradient)
otherwise                                    -> Absolute
```

It does not generalize to the holdout set:

| Split | Images | Absolute bytes | Chosen bytes | Change | Wins/ties/losses | Oracle bytes |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Calibration | 72 | 60,198,012 | 60,038,677 | -0.2647% | 3/67/2 | 59,899,019 |
| Holdout | 72 | 85,516,891 | 85,934,127 | +0.4879% | 0/62/10 | 85,262,353 |

The rule enables Zero Gradient six times on calibration and twelve times on
holdout. All twelve holdout enables are non-beneficial (ten losses and two ties).
It also misses five images where keeping Absolute forgoes an oracle improvement.
Holdout byte regret against the seven-route oracle is 671,774 bytes (0.788%).
The threshold was not retuned after looking at holdout results.

| Split | Baseline native seconds | Chosen-route native estimate | Offline input/feature overhead | Chosen plus overhead estimate |
| --- | ---: | ---: | ---: | ---: |
| Calibration | 78.563 | 79.066 | 8.893 | 87.958 |
| Holdout | 87.090 | 87.552 | 10.176 | 97.728 |

Chosen-route times above are estimates from the recorded single-variant trials,
plus separately measured feature overhead, NOT another measured end-to-end
encoding run. The classifier failed on bytes already, so it was not promoted or
rerun as a production pipeline. Running only one route naturally costs less than
exhaustively running seven, but this is not a speedup over the ordinary baseline.

Across 144 images, source reading/validation/decoding took 6.175 seconds and
feature calculation took 12.893 seconds (19.069 seconds combined). Median feature
calculation was 97.614 ms/image in this Python implementation. Decision evaluation
itself is negligible; calling the entire analysis free would be incorrect.
An integrated native extractor could reuse decoded pixels, but its performance
was not implemented or measured here.

This rejects this particular shallow rule and feature/sampling configuration,
not content analysis in general. It also does not test a router over all MA,
Source Patch and group-size strategies.

## Decision

- Keep the existing First representative and ordinary e9 defaults. Distribution
  leaf offset remains optional as requested.
- Keep Medoid and MinEntropy as explicit experimental native options. Deeper
  budgets and independent inputs did not demonstrate a useful aggregate gain;
  lowering the proxy cost is not evidence of a smaller final file.
- Do not automatically enable any tested Delta/Zero policy or the learned rule.
  Some files benefit, but every fixed alternative regresses in the primary
  aggregate, and the input-only selector fails holdout.
- The most actionable follow-up is to audit palette candidate admission against
  final coding cost. First compare identical preparation scopes with Absolute,
  delta and no-palette counterfactuals. The present telemetry identifies a
  confound; it does not replace that controlled causal experiment.

## Verification

Completed runs contain **1,730 official-decoder pixel checks**, all passing:

| Run | Checks |
| --- | ---: |
| Source synthetic | 32 |
| Source known12, repeated | 96 |
| Source known12, deeper selection | 36 |
| Source holdout24 | 96 |
| Palette synthetic | 126 |
| Palette primary144 | 1,008 |
| Palette full-frame24, cap 1024 | 168 |
| Palette full-frame24, cap 10000 | 168 |

The interrupted pilot's 175 passing outputs are excluded from this total and
from complete-corpus claims. Synthetic cases cover 8/16-bit gray, gray-alpha,
RGB/RGBA, hidden RGB and ordering/boundary cases; the real corpus is 8-bit.
Original and executable provenance checks passed, and temporary codec outputs
were removed. Reports, receipts and charts remain under `out/`.

Targeted Rust suites passed: 137 patch tests, 105 palette tests, five representative
tests after the background-proxy correction, and eight CLI example tests.
`cargo check --workspace --all-targets --offline -j4` passed. The six Python tool
test modules passed **71 tests**, including partial-run accounting, grouping
projection, input validation, source-disjoint analysis and plot-title layout.
These are Windows runtime results, not a Linux/macOS validation claim.

## Artifacts

The generated artifacts are local outputs, not embedded image data in source:

- [Primary144 arithmetic-mean bpp/time](../out/representation-study-20261113/palette-native-1024-final.bpp-v-time.png)
- [Primary144 total MiB/time](../out/representation-study-20261113/palette-native-1024-final.totalMiB-v-time.png)
- [Primary144 summary](../out/representation-study-20261113/palette-native-1024-final.summary.json)
  and [per-image CSV](../out/representation-study-20261113/palette-native-1024-final.cases.csv)
- [Source deeper-known12 summary](../out/representation-study-20261113/source-known12-deep-final.summary.json)
  and [Source holdout24 summary](../out/representation-study-20261113/source-holdout24-summary.summary.json)
- [Full-frame cap1024 summary](../out/representation-study-20261113/palette-control24-1024-summary.summary.json)
  and [cap10000 summary](../out/representation-study-20261113/palette-control24-10000-summary.summary.json)
- [Frozen input features](../out/representation-study-20261113/features.json)
  and [calibration/holdout evaluation](../out/representation-study-20261113/palette-native-analysis.json)

Plots use mean image bpp or total encoded MiB against summed per-image median
native encode time. They have no error bars and do not imply uncertainty estimates.
The oracle point includes ALL candidate work. Most comparisons have one repeat;
their tiny timing differences should not be interpreted as established gains.
