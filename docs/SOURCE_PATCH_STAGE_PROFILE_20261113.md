# Source Patch and Full-Pipeline Stage Profile

## Outcome

The local optimizations work, but their scope is too small to establish a whole-encode speed win.
On 50 stratified real images, the candidate detector took 2.299 s instead of
2.582 s (-10.96%). The unprofiled complete encode took 89.610 s instead of
89.385 s (+0.25%), with identical per-image sizes and exact original-pixel recovery.

This qualifies the earlier [eight-image experiment](SOURCE_PATCH_DETECTOR_SPEED_20261113.md):
a noisy outer Patch analysis total was insufficient to conclude that integer
seed comparisons or deferred Delta computation were ineffective.

Follow-up decision: retain the integer-seed and deferred-Delta computation
optimizations in the normal detector, as requested after reviewing these
measurements. Patch admission remains unchanged; diagnostics stay opt-in.
No MA, LZ77, packing policy, bitstream syntax or decoder was changed in this investigation.

## Scope and Method

- Base commit: `1029f1ad9a6fc7e879a20978c74fba66afd28b89`.
- Windows x86_64, matching Release `patch_lab` builds, e9, eight encoder threads.
- **Whole-pipeline Source Patch force-On**, using `patch_lab --mode source`.
  Non-patch options retain its normal e9 native projection. This is not a
  default-Auto scheduler benchmark or a JPEG byte-transcode benchmark.
  Forced patch preparation can expose costs that Auto would avoid.
- 50 images, 110,987,718 pixels, sampled from the prior 200-image manifest.
  Each of its ten strata contributes exactly one quarter of its cases.
  Seed 20261113; deterministic LCG/Fisher-Yates selection is recorded in
  `out/source-patch-detector-20261113/sample50.json`.
  No filtering by runtime, compression result or observed hotspot.
- Covers photos, documents/web, manga, game UI, icons/transparency, textures,
  high-resolution imagery, packed palette PNG and 8/16-bit RGB/gray inputs.
  Largest selected canvas is 4096x4096; no image resizing.
- Broad run: one unprofiled encode and one independent detailed diagnostic encode
  per image/version, interleaved versions. 200 official djxl pixel checks passed.
  This is a hotspot survey, not a statistically established timing regression test.
- Focused eight-image repeat: three unprofiled and five detailed diagnostic
  encodes per image/version. Another 128 official pixel checks passed.
  Native time excludes startup, input/output file IO and independent decoding.
- No builds ran during measurements. Temporary encoded/decoded files were removed
  by the harness. All 50 per-image byte sizes were unchanged; JXL hashes are not
  acceptance targets. Pixels, including hidden RGB, are the acceptance condition.

### How to Read Time

Unprofiled native time is the performance result. Diagnostic `encode.frames`
is the outer elapsed time in a different run.

Most MA/LZ77/entropy counters are **inclusive accumulated scope seconds**, summed
across calls and workers. They are not CPU time and not elapsed-time percentages.
Do not add parent and child rows, and do not divide these totals by wall time
to construct a pie chart. For example, baseline `ma.learn` accumulated 269.671 s,
while the diagnostic outer `encode.frames` total was 90.104 s. This is expected
with concurrent group encoding, not evidence of 269 seconds of serial delay.

The seven detector phases are sequential and non-overlapping within each detector
call. Boundary bookkeeping and temporary-buffer cleanup are included; some cleanup
lands in the component phase. Finalization also includes releasing detector
scratch buffers; it is not a pure Delta-extraction timer. In the eight-image repeat, stage medians are computed
per image before summation, so their sum need not exactly equal the median parent.

## Whole Encode

| Version | Bytes | Mean image bpp | Pixel-weighted bpp | Native s | MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 77,711,186 | 6.325920 | 5.601426 | 89.385150 | 1.241680 |
| candidate | 77,711,186 | 6.325920 | 5.601426 | 89.610395 | 1.238559 |

The +0.25% observed broad-run difference is too small for a speed conclusion
from one timing per image. It does not negate the separately measured detector
improvement. There is no measured compression-size change.

### Corpus Coverage

Baseline native results; accepted plans means nonempty retained Source Patch groups.

| Stratum | Images | MPix | Mean bpp | Native s | MPix/s | Accepted plans |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| alpha_icon | 5 | 3.459 | 2.3784 | 2.876 | 1.203 | 0 |
| document_web | 6 | 26.170 | 1.2233 | 27.491 | 0.952 | 4 |
| format_mixed | 5 | 18.384 | 6.6302 | 18.809 | 0.977 | 2 |
| game_ui | 4 | 0.490 | 1.9683 | 0.313 | 1.564 | 1 |
| manga | 4 | 3.866 | 7.3949 | 3.131 | 1.235 | 0 |
| mixed_local | 4 | 22.287 | 1.9528 | 8.079 | 2.759 | 0 |
| photo_highres | 8 | 23.648 | 9.1371 | 17.486 | 1.352 | 0 |
| photo_other | 6 | 5.538 | 7.6313 | 5.177 | 1.070 | 0 |
| texture | 4 | 4.196 | 15.9597 | 3.115 | 1.347 | 0 |
| urban | 4 | 2.950 | 8.9814 | 2.908 | 1.015 | 0 |

## Patch Breakdown

| Stage | Baseline ms | Candidate ms | Change |
| --- | ---: | ---: | ---: |
| 4x4 seed scan | 288.531 | 130.590 | -54.74% |
| Background buffers / seed queue | 215.280 | 218.885 | 1.67% |
| Background propagation | 374.571 | 322.703 | -13.85% |
| Components / border admission / scan | 1603.524 | 1535.885 | -4.22% |
| Fingerprint and eager Delta (baseline) | 31.252 | 20.459 | -34.53% |
| Fingerprint map lookup / insert | 1.765 | 1.701 | -3.59% |
| Final filtering / sort / deferred Delta / cleanup | 63.370 | 64.929 | 2.46% |

The two targeted stages save about 169 ms across the corpus, only 0.19% of the
89.4-second native baseline. Even removing both baseline stages entirely would
remove only about 320 ms in this diagnostic run. The exact wall benefit cannot
be inferred by adding timings from separate runs, but the scale explains why
whole-encode noise can obscure these savings.

The counter evidence agrees: both versions visit 12,805 component attempts,
scan 9,178 eligible rectangles and produce 1,552,257 fingerprint components.
Exact Delta sample generation drops from 1,552,257 to 170,571 values (-89.01%).
Both retain 548 groups. For these integer inputs the Delta counts represent
generated sample slots; nonfinite float inputs would need separate accounting.

### Outside the Detector

| Scope | Baseline s | Candidate s |
| --- | ---: | ---: |
| `patch.read_planes` | 1.164015 | 1.313695 |
| `patch.source.detect` | 2.581785 | 2.298748 |
| `patch.source.pack_dictionary` | 0.043238 | 0.045058 |
| `patch.source.subtract` | 0.118880 | 0.113534 |
| `patch.source.verify_replay` | 0.168390 | 0.158833 |
| `patch.assemble` | 0.006449 | 0.006872 |
| `encode.patch_analysis` | 4.261079 | 4.117453 |

`encode.patch_analysis` is the parent and must not be added to the other rows.
The listed children do not include every validation/destruction cost.

Plane extraction rose from 1.164 s to 1.314 s even though that source was not
modified. The detector saved 283 ms, while the complete Patch analysis saved
only 144 ms. This locates an offsetting observation, but does not prove a
cache, compiler-layout or scheduling cause. No hardware-counter/assembly study
was performed; do not present any of those explanations as established.

### Wasted Discovery Work

Only seven of the 50 forced-on inputs emitted a Patch plan. The other 43
consumed 3.267 s of the 4.261 s Patch analysis total. They account for 1.534 s
of the 1.604 s component phase. Eleven images have no seeds and return early.

This makes connected-component processing and admission more relevant than
BTreeMap replacement: total map lookup/insert is only 1.8 ms. An oversized
rejected component still has to be marked visited; stopping its traversal
without preserving that invariant would repeatedly rediscover it.

This result concerns the **forced-on diagnostic route**. It is not a claim that
default Auto wastes the same time on these 43 inputs.

### Focused Screenshots

The repeated eight-image run isolates the intended arithmetic improvement:

- Seed scan: 34.965 -> 14.224 ms.
- Fingerprint/eager Delta: 36.838 -> 21.104 ms.
- Complete detector: 290.493 -> 246.744 ms.
- Complete Patch analysis: 694.223 -> 681.612 ms.

Paper9bde specifically spends about 63 ms detecting patches, 96 ms packing the
atlas, and 68 ms checking exact decoder recovery in the baseline. Optimizing
the 5.4 ms seed scan cannot remove these later costs. Text7dba similarly spends
23.3 ms in replay checking versus 11.8 ms in detection.

The paper's packing hotspot is not universal: packing totals only 43 ms across
the broad corpus. The majority of broad images have no accepted plan.

## Full-Pipeline Hotspots

These rows are inclusive accumulated scope seconds, **not additive wall-time
components**. Parent/child entries are deliberately shown together to identify
which internal work should be examined next.

| Scope | Baseline accumulated s | Candidate accumulated s |
| --- | ---: | ---: |
| `ma.learn` | 269.671 | 272.984 |
| `ma.gather` | 59.545 | 58.141 |
| `ma.train_prepared` | 209.621 | 214.369 |
| `ma.split_sweep` | 190.966 | 194.132 |
| `ma.cache_collect` | 35.186 | 34.955 |
| `ma.cache_subtract` | 13.347 | 13.093 |
| `modular.weighted_header` | 35.865 | 36.063 |
| `modular.tokenize` | 19.668 | 19.391 |
| `lz77.parse_local` | 45.451 | 43.833 |
| `lz77.parse_shared` | 3.675 | 3.426 |
| `lz77.source.fast_seed` | 36.408 | 34.898 |
| `lz77.source.optimal` | 8.411 | 7.754 |
| `lz77.source.chain_batch_fill` | 1.634 | 1.628 |
| `entropy.model_build` | 36.840 | 35.121 |
| `entropy.cluster` | 29.975 | 28.645 |
| `entropy.cluster.refine` | 24.878 | 24.099 |
| `entropy.uint_config_search` | 2.046 | 1.869 |
| `entropy.write_tokens` | 4.650 | 4.569 |

### MA Is Not One Bottleneck

- On high-resolution photos and some 16-bit inputs, tree training and split
  sweeps dominate the accumulated MA work. `ma.split_sweep` is 190.966 s here.
  Histogram collection/subtraction are children; do not add their 35.186/13.347 s
  to that total.
- Large document scans are different. `5332.png` spends 7.685 s in `ma.gather`
  versus 1.427 s in `ma.train_prepared`; `5333.png` is 6.685 versus 0.458 s.
  These are shared-frame examples where more split-search optimization will
  miss the primary work.
- The 4096x4096 difference-frame input similarly has 8.676 accumulated seconds
  of gathering versus 2.280 of training. Weighted-predictor preparation and
  tokenization are also substantial. These phases deserve a separate kernel/
  data-flow investigation rather than another blanket tree-search rewrite.

### LZ77: Fast Matching, Not Always DP

`lz77.source.fast_seed` totals 36.408 accumulated seconds, compared with 8.411
for `lz77.source.optimal`. The focused screenshots had a much larger Optimal
component, so they were not representative of broad-corpus LZ77 work.

Examples of accumulated fast-seed time are 6.417 s for `6818.png`, 4.047 s for
`1230.hdr.png`, and 3.177 s for DIV2K `0748.png`. The raw receipts retain
existing sampled matcher statistics. Their sampled index/matching/DP durations
are neither wall totals nor automatically unbiased estimates of all matcher
work, especially with precomputed batch matching. `dp_init` is initialization,
not the DP loop.

### Entropy Modeling Has a Concentrated Outlier

Of 29.975 accumulated seconds in entropy clustering, 23.376 s (about 78%)
come from the single 12-MPix, 16-bit `1230.hdr.png`. Model construction on that
image is 24.842 s. This needs histogram cardinality/cluster-refinement analysis,
not a conclusion that ANS bit emission is slow everywhere.
Whole-corpus `entropy.write_tokens` is only 4.650 accumulated seconds.

## Slow Inputs

Baseline unprofiled native measurements; these are elapsed times, not the
accumulated work counters above.

| Image | MPix | bpp | Native s | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| diverse-downloads/imazen26_hdr/1230.hdr.png | 12.000 | 13.4676 | 13.526 | 0.887 |
| diverse-downloads/imazen26/5300-noaa-hurricane-documents/5332.png | 8.415 | 0.6203 | 12.238 | 0.688 |
| diverse-downloads/imazen26/5300-noaa-hurricane-documents/5333.png | 8.415 | 0.1083 | 9.175 | 0.917 |
| diverse-downloads/imazen26/6800-ia-scans-manuscript-text/6818.png | 6.120 | 4.5910 | 4.088 | 1.497 |
| local-test/png/wallhaven-9oz33w.png | 5.165 | 5.9153 | 3.943 | 1.310 |
| local-test/mt_lay/t1/146845733_p2.png | 16.777 | 0.0054 | 3.811 | 4.402 |
| diverse-downloads/gb82/sc-png/gmessages.png | 4.447 | 0.3153 | 3.777 | 1.177 |
| DIV2K/0216.png | 3.329 | 14.5075 | 3.194 | 1.042 |
| DIV2K/0687.png | 3.133 | 10.0123 | 2.568 | 1.220 |
| DIV2K/0748.png | 3.133 | 9.0946 | 2.396 | 1.308 |
| DIV2K/0123.png | 3.133 | 8.4633 | 2.050 | 1.529 |
| DIV2K/0619.png | 3.109 | 7.6178 | 2.037 | 1.526 |

## Next Work

1. Separate fast LZ77 matcher preparation, probing, extension and spatial/ring
   candidates on the photo/document cohort before spending more effort on DP.
2. Investigate MA sample gathering and weighted-predictor preparation on the
   two large document scans; use split/histogram work as the target on photos.
3. Investigate entropy cluster refinement on the 16-bit HDR-derived PNG as an
   outlier, with unaffected 8-bit/16-bit controls.
4. For Source Patch, prioritize connected-component traversal on rejected
   regions, plane extraction, and screenshot-specific atlas packing.
   Exact replay must stay mandatory; accelerate it rather than disable it.
5. Keep integer seed/deferred Delta as valid local candidates, but measure them
   with their downstream costs. Alternative legal atlas placements are allowed;
   neither old C++ placement identity nor JXL byte identity is an optimization goal.

No further algorithm optimization was applied during this analysis.

## Reproduction and Artifacts

The optional `RS_JXL_PATCH_SOURCE_PROFILE=1` switch requires `RS_JXL_PROFILE=1`.
It selects detailed local accounting; disabled code does not take per-component
clocks. Normal production defaults leave it off. It emits one JSON record per
detector invocation, including an early no-seed return.

```powershell
python -B tools/benchmark_source_patch_detector.py --encoder baseline=out/source-patch-detector-20261113/bin/profile-baseline.exe --encoder candidate=out/source-patch-detector-20261113/bin/profile-candidate.exe --decoder path/to/official/djxl.exe --manifest out/source-patch-detector-20261113/sample50.json --effort 9 --threads 8 --repeats 1 --profile --profile-detail --profile-repeats 1 --timeout 1800 --out out/source-patch-detector-20261113/new-stages50.json
python -B tools/summarize_source_patch_profile.py out/source-patch-detector-20261113/new-stages50.json --out out/source-patch-detector-20261113/new-summary.json
```

Local ignored receipts:

- `stages8.json`, `stages8-summary.json`: repeated focused cohort.
- `sample50.json`, `stages50.json`, `stages50-summary.json`: broad cohort,
  input and binary hashes, all raw stage logs, per-image metrics and checks.
- `profile-candidate.rs`: exact profiled candidate source used for its frozen
  example executable. Candidate equivalence tests remain in the earlier
  archived experiment patch.
- `bin/profile-baseline.exe`, `bin/profile-candidate.exe`: frozen executables.

The benchmark refuses to overwrite reports. No image or encoded corpus is
committed. Debug validation: 29 focused native tests and 23 Python harness/
summary tests passed. A second agent reviewed phase boundaries and the
inclusive/parallel timing interpretation.
