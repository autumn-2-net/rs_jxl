# Patch, Cross-Channel MA and Offset Reuse

## Scope

This follow-up retains the integer seed and delayed Delta computation from
[SOURCE_PATCH_STAGE_PROFILE_20261113.md](SOURCE_PATCH_STAGE_PROFILE_20261113.md).
It does not force Source Patch on, change the detector's admission policy, or
enable expensive refinement/cross-channel strategies by default.

All timings below are native encoder calls, excluding input decode, process
startup, file IO and independent pixel verification. Tests use Release, e9,
eight encoder threads, with encodes run serially. Size means actual JXL bytes.
Acceptance is decoded original samples including invisible RGB, not JXL hashes.
These are fixed native source candidates at e9, not the outer multi-candidate
scheduler or an official libjxl benchmark. Trials rotate variant order; local
background workloads are not stopped, so small timing differences need repeat
controls before being treated as wins.

## Parallel Gather Control

The 12-image target cohort uses the previously frozen optimized-detector
`profile-candidate.exe`. It compares patch-off, Source Add with full MA, and the
same Source Add policy with parallel sparse sample gathering. Two interleaved
repetitions, sums of per-image medians. All 72 official djxl 0.12 pixel checks
passed. No per-image size changed between the two Source modes.

| Route | Bytes | Mean image bpp | Weighted bpp | Native s | MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Patch off | 2801523 | 0.981864 | 0.706717 | 9.101890 | 3.484230 |
| Source, full MA | 2517598 | 0.861601 | 0.635094 | 36.349241 | 0.872455 |
| Source, full MA, parallel gather | 2517598 | 0.861601 | 0.635094 | 28.463636 | 1.114161 |

Parallel gather reduces this target cohort's Source-mode time by 21.69%.
Source remains much slower than patch-off; a successful dictionary is not
proof of a profitable complete encode. For example, `5333.png` grows from
93754 to 113869 bytes when Source is forced on. `ellipses.png` also grows.
`wallhaven-49yekn.png` is an illustration control with no Source size benefit;
its earlier Tile16 result belongs to a different detector.

| Image | Source bytes (both) | Serial gather s | Parallel gather s | Time change |
| --- | ---: | ---: | ---: | ---: |
| 9bde6aaf390109f97622d35ff1e900ee.png | 283434 | 3.6107 | 3.3239 | -7.94% |
| 7dba5bd64be12f08db25c3a78bcb80d7.png | 9519 | 0.3186 | 0.3068 | -3.69% |
| 3dd595a30a79d3ba6d107fa6f60c5d3d.png | 106886 | 1.2975 | 1.1594 | -10.64% |
| image.png | 347113 | 3.6550 | 3.0557 | -16.40% |
| wallhaven-49yekn.png | 707442 | 1.0640 | 0.9837 | -7.55% |
| SampleB.png | 17208 | 0.2689 | 0.2553 | -5.03% |
| ellipses.png | 4348 | 0.0910 | 0.0871 | -4.29% |
| news.ycombinator.com.png | 66584 | 1.1051 | 0.9758 | -11.69% |
| 5332.png | 652466 | 11.4576 | 8.4967 | -25.84% |
| 5333.png | 113869 | 9.2838 | 6.2331 | -32.86% |
| gmessages.png | 175244 | 3.8270 | 3.2165 | -15.95% |
| 8277.png | 33485 | 0.3701 | 0.3695 | -0.16% |

This is a deliberately enriched target cohort, not a representative dataset
compression claim. The wider independent cross-channel cohort excludes earlier
cross-channel30 and Source50 inputs by SHA256; it is not independent of the
earlier baseline/Offset200 experiment.

## Independent Cross-Channel Screen

Sixty images, six from each of ten strata, 252032758 pixels. One rotated timed
trial per route, 240 original-sample checks with official djxl 0.12: all passed.
The cohort excludes earlier cross-channel30/Source50 content by SHA256. Some
same-series images remain, so this is not a claim of 60 unrelated scenes.

| Route | Bytes | Mean image bpp | Weighted bpp | Native s | MPix/s | Smaller / equal / larger |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ordinary e9, patch off | 169857115 | 6.985042 | 5.391589 | 132.177626 | 1.906773 | - |
| Previous channel 1 | 164320680 | 6.906813 | 5.215852 | 144.209017 | 1.747691 | 47 / 12 / 1 |
| Previous channels 2 | 163211339 | 6.886596 | 5.180639 | 156.087353 | 1.614690 | 48 / 12 / 0 |
| Ordinary e9, parallel gather | 169857115 | 6.985042 | 5.391589 | 136.708828 | 1.843573 | 0 / 60 / 0 |

Prev1 saves 3.259% total bytes at 9.102% more observed time. Prev2 saves 3.913%
at 18.089% more time. Mean per-image bpp improves by less than total bytes:
large high-bit-depth images and documents contribute disproportionately.
`monarch.png` is the prev1 regression: 407506 -> 408405 bytes (+0.221%);
prev2 does not regress that image. No guarantee is inferred from 60 successes.

| Stratum (six images each) | Prev2 byte change | Prev2 time change |
| --- | ---: | ---: |
| Alpha/icons | -1.117% | +20.40% |
| Documents/web | -3.208% | +12.88% |
| Mixed formats/depths | -7.288% | +25.86% |
| Game/UI | -2.770% | +12.62% |
| Manga | -1.034% | +25.09% |
| Local mixed | -1.181% | +16.50% |
| High-resolution photos | -0.369% | +21.57% |
| Other photos | -0.430% | +22.43% |
| Textures | -0.596% | +3.92% |
| Urban | -0.611% | +18.21% |

The 94.835 MP `Graveyard_27x39_ENGLISH.png` changes from 42928156 bytes /
37.004s to 41377756 bytes / 40.847s with prev2. Parallel gather leaves its
size unchanged and takes 37.727s. Across the complete ordinary-image cohort,
gather is 3.43% slower in this single pass: no general speedup is demonstrated.
This contrasts with the repeated Source Patch control above. Neither full-MA
cross-channel search nor gather is promoted globally from this experiment.

The first run's checkpoint writer appended the final JSON after its last
checkpoint. The complete final record was extracted without rerunning encodes
or changing samples; the original is `holdout60-results.checkpoint-jsonlog`.
The writer now seeks/truncates at finalization, covered by its harness tests.

## Cross-Channel Patch Combination

The same 12 targets/controls, one rotated timed trial plus one separate profile
per route/image. All 144 outputs pass original-sample djxl checks. Profile
scopes confirm reference/residual emission on 11 images in every route;
`wallhaven-49yekn.png` emits none. Its cross-channel gain is ordinary-image MA,
not Patch residual compression. Do not treat the whole cohort as 12 Patch wins.

| Route | Bytes | Mean image bpp | Weighted bpp | Native s | MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 2517598 | 0.861601 | 0.635094 | 35.652795 | 0.889498 |
| Source + parallel gather | 2517598 | 0.861601 | 0.635094 | 27.368821 | 1.158730 |
| Source + prev2 residual | 2463380 | 0.834969 | 0.621417 | 40.394493 | 0.785084 |
| Source + prev2 both | 2452925 | 0.828683 | 0.618779 | 40.887730 | 0.775614 |
| Source + gather + prev2 residual | 2463380 | 0.834969 | 0.621417 | 34.930011 | 0.907904 |
| Source + gather + prev2 both | 2452925 | 0.828683 | 0.618779 | 35.258918 | 0.899434 |

Both+gather saves another 2.569% of Source bytes with 1.105% less observed
time than serial-gather Source, but costs 28.829% more time than Source+gather.
These are different comparisons. Eight images shrink, three tie and one grows
relative to Source. `5333.png` grows 113869 -> 114643 bytes (+0.680%); this
remains larger than patch-off's 93754 bytes. Cross-channel search does not fix
an unprofitable Patch representation. `ellipses.png` also remains larger than
patch-off despite improving over Source. No unconditional admission is added.

Useful cases: paper9bde changes 283434 -> 258661 bytes; `image.png` changes
347113 -> 329197; `news.ycombinator.com.png` changes 66584 -> 63730. The editor
screenshot stays 106886 bytes. Reference and residual must be evaluated jointly
where useful, not assumed to need identical settings on all images.

For `5332.png`, plain Source+gather takes 7.661s, while prev2+gather takes
12.736s. The profile loses `gather.append` and shows 8.564s in
`gather.batch_collect`, versus 0.928s for Source+gather. The existing Channels
scheduler falls back to serial when the whole estimated batch exceeds 1 GiB;
adding properties increases that estimate. This motivates a separate Waves
experiment, not raising the memory budget or changing sample sites. Profile
times are from separate diagnostic runs and are not part of the native table.

### Budgeted Gather Waves

A targeted follow-up runs those two slow documents and paper9bde through the
same freshly frozen binary, with one rotated timed trial and one diagnostic
trial per route. All 18 outputs pass official original-sample checks. This is
a diagnostic subset chosen after the first screen, not an independent corpus.

| Image | Source + gather s | Prev2 both + Channels s | Prev2 both + Waves s | Prev2 bytes (both schedulers) |
| --- | ---: | ---: | ---: | ---: |
| paper9bde | 3.235941 | 3.370875 | 3.544291 | 258661 |
| 5332.png | 7.920014 | 12.220343 | 9.355497 | 645240 |
| 5333.png | 5.674503 | 6.701781 | 6.651786 | 114643 |

For 5332, Waves reduces observed cross-channel encode time by 23.44%, without
changing the size or raising the memory estimate budget. The new `gather.plan`
diagnostic confirms the cause:

- Channels without cross-channel properties: 350 tasks, eight chunks,
  933172396 estimated bytes.
- Channels with prev2: 350 tasks, one chunk, 1140544040 estimated bytes,
  exceeding the unchanged 1073741824-byte budget.
- Waves with prev2: 329 tasks / eight chunks / 1070705377 estimated bytes,
  followed by 21 tasks / eight chunks / 69838663 estimated bytes.

The reference frame is small and does not cause the fallback. On the other
two images, the estimated residual batch already fits, so Waves does not
recover lost parallelism. Paper9bde is 5.14% slower in this single pass; 5333
is nearly unchanged. Across these three selected images, Waves saves 12.30%
time versus Channels+prev2, not versus Source without cross-channel search.
The option remains experimental. Estimated compact-record bytes are not a
measurement or bound on whole-process peak RSS.

## Distribution Offset Reuse

Twenty real images (128259930 pixels) plus eight synthetic 8/16-bit gray,
gray-alpha, RGB and RGBA fixtures: all 84 original-sample checks passed. The
table includes only the 20 real images. The frozen old binary includes the
same detector optimizations and predates the Offset reuse edits.

| Route | Bytes | Mean image bpp | Weighted bpp | Native s | MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Offset refinement | 69694537 | 6.576639 | 4.347081 | 66.340615 | 1.933355 |
| Old Distribution Offset | 68260862 | 6.541034 | 4.257658 | 93.189134 | 1.376340 |
| Reused Distribution Offset | 68260862 | 6.541034 | 4.257658 | 87.513065 | 1.465609 |

Observed total Offset time falls 6.091%, with every per-image size unchanged
relative to the old implementation. This is a work-reuse optimization, not a
new compression proposal. Refinement still costs 31.91% more time than no
refinement and saves 2.057% total bytes in this particular, large-document-heavy
subset. It is not a replacement estimate for the earlier 200-image study.

All 20 images hit raw-token reuse: 247445285 remapped pre-LZ literals across
1817 offset proposals, of which 1632 win actual serialization arbitration.
No no-LZ token-fit fast path activates in this ordinary-e9 matrix. Fitting still
visits the original pixels; the saved work is the second tokenization walk.
`Graveyard_27x39_ENGLISH.png` changes 58.392 -> 53.940s, while retaining the old
Offset size of 41619010 bytes. One-pass timings include regressions on some
smaller inputs; focused repeat controls are reported separately below.

Targeted repeat controls use three rotated repetitions, selected after the
screen to inspect one win and two apparent regressions. They are not an
independent corpus or a replacement for the original observations. All 27
additional original-sample checks pass and per-image sizes remain unchanged.

| Image | Old Offset median s | Reuse median s | Change |
| --- | ---: | ---: | ---: |
| 5023.png | 8.099934 | 7.847050 | -3.122% |
| wallhaven-n6qzmx.png | 3.336743 | 3.411666 | +2.245% |
| 0554.png | 2.655980 | 2.462713 | -7.277% |

The n6qzmx ranges overlap (old 3.151..3.587s; reuse 3.159..3.542s); this does
not establish a universal speed gain or a statistically significant regression.
0554 reverses its first-pass slowdown. Background workload and scheduling noise
remain relevant; the local semantic optimization is established by avoided
work, primitive tests and pixel checks, not by requiring identical JXL hashes.

## Native Reuse and Experimental Controls

Distribution refinement remains opt-in. The new native path retains the
baseline's pre-LZ literal streams for offset-only, unit-multiplier tree trials.
It remaps packed signed residuals using the leaf's old/new offsets, then runs
the original LZ parser again with the same group boundaries, distance widths
and shared-stream batch policy. It does not reuse old matches or ANS models.
Training and predictor choices are unchanged. Actual serialized cost still
selects the original tree when the proposal is not strictly smaller.

This removes the second tokenization's property/predictor pixel traversal on
eligible e9 domains, not the separate fitting walk, LZ parse, model building or
output serialization. The cache extends pre-LZ token lifetimes, using O(tokens)
additional retained memory; it is not a claim of reduced peak memory.

Literal-only/no-LZ Distribution fits can also use baseline tokens instead of
the pixel fitter, after validating stream/context association and a conservative
sample bound. Weighted predictors and wide/multiplier cases retain the original
fitter when the unwrapped residual cannot be proven recoverable from the token.
The counters `offset_token_fits`, `offset_reused_tokens` and
`offset_raw_reused_tokens` distinguish those paths. Hash identity is not an
acceptance requirement; primitive token-mapping tests use a fresh native
tokenizer as a local correctness oracle.

`patch_lab` now accepts `--previous-channels 0|1|2` and
`--channel-target residual|reference|both`. Omitted controls preserve defaults.
For patch-off, residual means the ordinary main image. For Source mode, it is
only actually a residual when a dictionary/reference was emitted. Reference
controls change the raw reference sampling policy because atlas resolution
recomputes the effective policy later. `--parallel-gather` remains independent.
No block-plan flags are used in the full-MA comparisons.

`--parallel-gather --gather-waves` selects the existing ordered Waves scheduler
with the unchanged 1 GiB compact-record estimate. It does not raise the budget,
split a channel into learning stripes, or change sample sites. Oversized single
channels may still run serially. `gather.plan` notes under `RS_JXL_PROFILE=1`
expose mode, task/chunk counts and estimated bytes; they are absent in normal
timed runs. The byte estimate is not an RSS cap or measurement.

The new `tools/benchmark_patch_ma.py` supports explicit route lists, all-cases
aggregates, per-case checkpointed reports and separate `--profile` diagnostics.
It never calls a failed subset a successful aggregate. Supported effort values
match the experiment CLI (4 through 10); this comparison uses 9 throughout.

## Validation and Adoption

The six complete raw reports contain 585 successful official djxl 0.12
original-sample checks: 72 gather controls, 240 cross-channel screen trials,
144 Patch combination trials, 84 Offset trials including synthetic fixtures,
27 Offset repeat controls, and 18 gather-wave trials. Diagnostic outputs are
also checked but excluded from timing. These are encoding trials, not 585
distinct images. No encoder or verification failures were omitted.

Focused tests passed: 38 Patch source tests, 74 Offset-related Rust tests,
six `patch_lab` tests, nine Patch harness tests, 14 refinement harness tests,
and 24 summary tests. Release experiment binaries build successfully. This
does not claim a full workspace, cross-platform or metadata regression run;
the installed CLI and wheel were not rebuilt for this experiment.

- Keep integer seed comparisons, single-channel reads and delayed Delta
  construction on the existing Source detector path by default. Preserve its
  admission rules; this does not force Patch on.
- Keep raw-token reuse inside the opt-in Distribution refinement. The actual
  serialized-cost fallback remains. It saves work but retains additional
  tokens in memory; peak RSS has not been measured.
- Keep prev1/prev2 and gather-wave composition opt-in. The ordinary-image
  prev2 screen is promising, but the Patch combination has a concrete size
  regression and gather has no demonstrated universal speed advantage.
- Preserve plain/non-Patch candidates when comparing representations. A
  dictionary's existence is not evidence that its final file is smaller.

## Artifacts

The results above describe the first follow-up, committed as `16e1d82`. A
subsequent Offset fitting implementation is described below; its measurements
are kept separate from this first experiment.

Ignored local results are under `out/patch-ma-followup-20261113/`:

- `gather12.json`: raw runs, per-image medians, input/binary identities.
- `patch12.json`: five known screenshot/illustration targets and seven images
  whose previous diagnostic runs actually emitted reference/residual frames.
- `holdout60.json`: six deterministic samples from each of ten strata; excluded
  earlier cross-channel30/Source50 content, not just matching filenames.
- `offset20.json`: first two of those preselected cases per stratum, including
  the 94.835 MP document; selection predates this run's outcomes.
- `bin/offset-before.exe`: current detector optimization, old Offset algorithm.
- `holdout60-results.json`, `patch12-results.json`, `offset20-results.json`:
  complete raw trials, verification receipts and timing provenance.
- `offset20-real.json`: explicit projection of the 20 predetermined real
  inputs from the Offset report, excluding synthetic fixtures, not failures.
- `offset-repeat3-results.json`, `waves3-results.json`: targeted controls.
- `bin/offset-after.exe`, `bin/patch-ma.exe`, `bin/patch-ma-waves.exe`: frozen
  experiment builds; the Waves binary also adds the gather-plan diagnostic.
- `figures/{holdout60,patch12,offset20,waves3}.summary.{json,md}`: independent
  cohort summaries. Cohorts are never pooled into a synthetic total.
- `figures/{holdout60,patch12,offset20,waves3}.bpp-v-time.png`: mean image bpp
  versus summed native time, without error bars.
- `figures/{holdout60,patch12,offset20,waves3}.totalMiB-v-time.png`: total actual
  size versus summed native time, using the same complete cohorts.

`patch_lab` is an experimental example, not a new installed CLI/API contract.
Its `selected_mode=source` means an enabled route only. Separate `--profile`
runs confirm actual reference/residual emission and are excluded from timing.

## Reproduction

Build the experiments once, freeze the binaries, and run measurement jobs
serially. The frozen pre-reuse `encoder_lab` is required for the third command;
using the new executable for both sides would not measure the change.

```powershell
python -B tools/benchmark_patch_ma.py --encoder out/patch-ma-followup-20261113/bin/patch-ma.exe --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --manifest out/patch-ma-followup-20261113/holdout60.json --variants off off-prev1 off-prev2 off-gather --effort 9 --threads 8 --repeats 1 --timeout 1800 --out out/patch-ma-followup-20261113/holdout60-results.json

python -B tools/benchmark_patch_ma.py --encoder out/patch-ma-followup-20261113/bin/patch-ma.exe --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --manifest out/patch-ma-followup-20261113/patch12.json --variants source source-gather source-prev2-residual source-prev2-both source-gather-prev2-residual source-gather-prev2-both --effort 9 --threads 8 --repeats 1 --profile --timeout 1800 --out out/patch-ma-followup-20261113/patch12-results.json

python -B tools/benchmark_refinement.py --encoder out/patch-ma-followup-20261113/bin/offset-after.exe --frozen out/patch-ma-followup-20261113/bin/offset-before.exe --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --manifest out/patch-ma-followup-20261113/offset20.json --synthetic --variants baseline frozen-offset-distribution offset-distribution --effort 9 --threads 8 --repeats 1 --timeout 1800 --out out/patch-ma-followup-20261113/offset20-results.json

python -B tools/benchmark_patch_ma.py --encoder out/patch-ma-followup-20261113/bin/patch-ma-waves.exe --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --manifest out/patch-ma-followup-20261113/waves3.json --variants source-gather source-gather-prev2-both source-waves-prev2-both --effort 9 --threads 8 --repeats 1 --profile --timeout 1800 --out out/patch-ma-followup-20261113/waves3-results.json

python -B tools/summarize_patch_ma_followup.py --report out/patch-ma-followup-20261113/waves3-results.json --baseline source-gather-prev2-both --out out/patch-ma-followup-20261113/figures/waves3 --plot
```

The benchmark scripts refuse to overwrite an existing report. Select a fresh
output path for a repeat. Synthetic offset cases are reported separately from
the 20 real images. No metadata roundtrip claim is made by these pixel tests.

## Subsequent Offset Fitting Work

After `16e1d82` was pushed, the user confirmed that Distribution refinement
must remain opt-in until the added cost is satisfactory. No default effort,
decoder, CI or Python ABI policy changes are part of this work.

The subsequent native implementation has three computational changes:

1. Fit Distribution samples from cached pre-LZ literals in ordinary e9, not
   just no-LZ streams. Pair literals with the original transformed samples to
   recover the signed predictor, then subtract in i64. This supports Weighted
   and signed extremes without replaying predictor state or tree properties.
2. Score independent leaves in ordered 64-leaf chunks on the existing execution
   pool. The shared Offset signaling histogram still uses the same sequential
   sweeps and BFS proposal order. The chunk size is dispatch granularity, not a
   learning, sample, leaf-count or candidate limit.
3. Cache the existing per-count entropy term lazily within each leaf's scoring
   call. At most 256 `(count, value)` entries (4 KiB) are retained per active
   scorer. Collisions recompute the original formula; candidate order and
   floating-point accumulation order are unchanged. No whole-image speedup is
   attributed to this cache in isolation.

For unit-multiplier leaves, let the packed token represent
`pixel - prediction - old_offset` modulo 2^32. Since all standard predictions
except Average4 return an i32 value, compute the prediction modulo 2^32, then
subtract that signed value from the original sample in i64. This preserves
the wide residual's mean and reservoir values. Average4 retains a conservative
source-value bound proving its prediction fits i32, otherwise the original
pixel fitter is used. No input values are clamped or rejected by that fallback.

### Offset-Dependent Routing Guard

Review found a separate correctness condition missed by the original reuse
path: zero-offset Gradient leaves can enable native LUT routing, whose
property-9 evaluation clips a wide gradient. Nonzero offsets disable that
route and use generic wrapping properties. Topology equality alone therefore
does not prove that a cached literal still belongs to the same context.

A compact iterative eligibility check conservatively accounts for static
channel/group specialization. If Gradient-LUT context routing may be involved,
cached fitting falls back to the original fitter, and offset remapping falls
back to fresh native tokenization. Both transitions are covered, including
returning to zero offset. A single static Gradient leaf without a property-9
decision remains eligible for reuse. No LUT or decoder semantics are changed.

Regressions exercise positive/negative integer overflow, threshold -8193 and
its clipping boundary, static splits above/below property 9, ineligible
thresholds, and unchanged buffers on fallback. The end-to-end refinement test
also checks that LZ and non-LZ trials really retokenize before serialization,
including when predictor refitting is requested.

The original 585-image-output verification did not exercise these custom-tree
edge cases. A passing ordinary corpus is not a proof that remapping every
topologically identical tree is safe. Correctness remains original decoded
pixels, not equality with an earlier compressed file.

### Follow-Up Artifacts

`out/offset-fit-20261113/` contains the frozen control, exploratory and guarded
executables. `real20-results.json` is the completed pre-guard pilot, not the
acceptance report for the guarded implementation. Its same-20-image comparison
observed a 2.44% reduction in total time with unchanged sizes. It is retained
as exploratory evidence, not pooled with later runs.

### Guarded Implementation Results

The final guarded executable was tested on the same predetermined 20 real
images (128.260 MP, ten strata), e9 and eight threads. All three variants use
every image. Native timings exclude process startup, input and official
decoder verification; this screen has one timed trial per image/variant.

| Variant | Actual bytes | Mean image bpp | Weighted bpp | Native seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| e9, Offset disabled | 69,694,537 | 6.576639 | 4.347081 | 68.300811 | 1.877868 |
| Previous Distribution Offset | 68,260,862 | 6.541034 | 4.257658 | 90.618923 | 1.415377 |
| Guarded token-fit Distribution Offset | 68,260,862 | 6.541034 | 4.257658 | 88.293313 | 1.452657 |

This is a 2.57% total native-time reduction versus the previous Offset
implementation, with every image's actual size unchanged. It is not a new
compression-ratio improvement. The unchanged 1,817 proposals and 1,632 wins
confirm that this work preserved the sampled search on this cohort; 2,029
fits used cached pre-LZ literals instead of replaying predictors/properties.

The 94.835 MP `Graveyard_27x39_ENGLISH.png` dominates the total: 57.916688 to
55.812177 seconds, both 41,619,010 bytes. The other 19 images together take
32.702235 versus 32.481136 seconds, only a 0.68% improvement. Negative timing
results are retained: e.g. the first screen's `Nuclear_Throne_screenshot_02`
takes 2.021262 versus 3.182301 seconds, despite lower fitted-Offset work time.
This is not evidence of a universal per-image speedup. Domain-local timing
sums can overlap under parallel execution and are not end-to-end wall time.

Relative to disabling Offset, this cohort saves 2.06% of total bytes but
still adds 29.27% native time. The arithmetic mean bpp is not the same as
aggregate size divided by total pixels; both are shown explicitly. Keep
Distribution opt-in, as agreed, rather than promoting it on this evidence.

Verification completed: 90 Offset-related Rust tests passed, with one
opt-in microbenchmark ignored; the final real-image screen passed 60 official
`djxl` pixel checks, and eight separate 8/16-bit gray, gray-alpha, RGB/RGBA
fixtures passed another 24 checks including hidden RGB. Synthetic inputs do
not enter the real-image bpp or timing totals. The ignored microbenchmark
means the entropy cache has not been measured in isolation. No full workspace,
cross-platform, peak-memory or Python-wheel acceptance is claimed here.

Final artifacts under `out/offset-fit-20261113/`:

- `final20-results.json`: guarded implementation, all real-image trials.
- `synthetic-results.json`: separate synthetic correctness trials.
- `final20.summary.{json,md}`: cohort and per-stratum summaries.
- `final20.bpp-v-time.png`: arithmetic mean bpp versus summed native time.
- `final20.totalMiB-v-time.png`: actual total size versus summed native time.
- `bin/before.exe`, `bin/guarded.exe`: frozen comparison executables.

### Targeted Timing Counterchecks

Four apparent regressions from the screen were repeated three times with
rotated variant order, still at e9/eight threads. These are deliberately chosen
counterexamples, not an independent random cohort, and are never pooled with
the 20-image total. Every size stayed unchanged and all 36 official pixel
checks passed. Per-image native medians:

| Image | Previous Offset s | Guarded Offset s | Time change |
| --- | ---: | ---: | ---: |
| 1247 | 2.189061 | 2.265211 | +3.48% |
| Nuclear Throne | 2.323423 | 2.327355 | +0.17% |
| SeisinkiVulnus | 0.915207 | 0.976973 | +6.75% |
| 0554 | 2.456667 | 2.504621 | +1.95% |
| Sum | 7.884357 | 8.074159 | +2.41% |

The initial Nuclear Throne slowdown of roughly 57% did not repeat at that
magnitude, but these medians do not establish a general speed win either.
Results are from a shared host without process-affinity isolation; no unrelated
workloads were stopped. Do not infer the cause from one stage timer, or claim
statistical significance from these short repeats. Actual-size fallback
protects size selection, not elapsed time. The current implementation is an
opt-in tradeoff, not grounds to enable Offset on every e9 encode.

`repeat4-results.json` and `repeat4.summary.{json,md}` retain all timed trials
and the four-image summary, including every slower measurement. Reproduce
with the same benchmark command using `--repeats 3` and four `--case` paths
for the named images instead of `--manifest` (use a fresh output path).

The `1247` and Nuclear Throne cases were also repeated three times at one
thread, in `serial2-results.json` and `serial2.summary.{json,md}`. Their summed
native medians are 23.720136 versus 22.896523 seconds (-3.47%), with identical
sizes and another 18 successful official pixel checks. `1247` improves from
12.627114 to 11.812830 seconds (-6.45%); Nuclear Throne is effectively flat
at 11.093022 versus 11.083694 seconds. This supports a workload-dependent
computation benefit, but does not isolate the three implementation changes
or prove that parallel scoring alone caused the eight-thread regressions.

Across the final screen, separate synthetic checks and targeted repeats,
138 encoded outputs passed official original-sample comparisons. Repeated
outputs are counted as checks, not additional distinct images. The earlier
unguarded pilot is excluded from this acceptance count.

Final screen and figure commands (not the targeted repeats):

```powershell
python -B tools/benchmark_refinement.py --encoder out/offset-fit-20261113/bin/guarded.exe --frozen out/offset-fit-20261113/bin/before.exe --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --manifest out/patch-ma-followup-20261113/offset20.json --variants baseline frozen-offset-distribution offset-distribution --effort 9 --threads 8 --repeats 1 --timeout 1800 --out out/offset-fit-20261113/final20-results.json

python -B tools/summarize_patch_ma_followup.py --report out/offset-fit-20261113/final20-results.json --baseline frozen-offset-distribution --manifest out/patch-ma-followup-20261113/offset20.json --out out/offset-fit-20261113/final20 --plot
```
