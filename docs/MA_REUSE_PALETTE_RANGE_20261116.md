# Coarse MA Continuation, Palette Pilots, and Narrow Prediction

Date: 2026-11-16. Measurements were made in an isolated checkout.
Baseline source: production `0addc02` plus its existing uncommitted ANS
normalization patch. That retained ANS change is committed separately. The
three experiments are retained as opt-in implementations; no effort preset
enables them. Local unversioned artifacts are under the sibling project at
`jxl_python/out/ma-reuse-routing-range-20261116/` (`work/`, `prior.patch`,
`paired-e9/`, `paired-threads/`, and the explicit input manifests).

## Independent Experiments

- `coarse`: `RS_JXL_MA_COARSE_CONTINUE=1`. Initialize up to three levels using
  native Legacy split scoring and twelve-quantile proposals across the configured
  properties. Resume the original e9 learner on unfinished nodes using the same
  prepared samples, quantizers, parent histograms and histogram budget. A coarse
  rejection is checked with the native learner before treating it as saturated.
  Cached native decisions are consumed before subtree detachment. This changes
  the initial topology, not the final tree depth or sample budget. Explicit
  seeds/lookahead/channel-tree experiments and tiny pilots are excluded.
- `palette`: `--palette-pilot route`. At each existing eligible global/local
  Palette attempt, compare the raw representation with the already-built table
  and index using the native four-leaf MA/Token/LZ pilot. It can both reject old
  acceptances and rescue old rejections. Reuse raw-plane scores while the input
  state is unchanged; invalidate them on every accepted transform. Continue
  encoding the accepted candidate without rebuilding its table/index. Explicit
  force/direct/zero/replay choices retain precedence. Pilot errors fall back to
  the old decision. This does NOT compare every possible palette mode, and the
  pilot tree is not reused as the final MA tree because its predictor set and
  quantization differ. It is not an exact final-size cost model.
- `narrow`: `RS_JXL_PREDICT_NARROW=1`. Scan the actual immutable transformed plane
  once, prove the arithmetic range separately for each requested predictor, and
  use eight i32 AVX2 lanes with signed i64 output. Scalar edges and wide i64
  fallback remain. No input-bit-depth shortcut, decoder change, or ARM work.
  The channel scan and output widening are included in native timing.

All switches are opt-in. The previous rejected histogram batching experiment
is NOT part of this checkout.

## Validation and Measurement

350 focused Rust tests passed: MA training (290), residual SIMD (9), tokens
(17), and Palette (34). The retained ANS change also passed all 192 entropy
tests. Three Python runner checks passed. SIMD tests cover
all 42 RCTs, negative rounding and ties, extreme i32 fallback, padding and tails.

`cases.json` selects ten original-resolution images across photos, documents,
UI, illustration, transparency, 16-bit texture and HDR, including previous
Palette counterexamples. Each trial uses ordinary e9 with only one experiment
changed. Same Release binary, rotated order, two repetitions, eight threads.
Summaries sum the per-image median native encode time. Native time excludes
input/output I/O, process startup and oracle verification. All emitted files
are decoded with independent official libjxl 0.12 djxl and compared to original
pixel samples, including alpha and invisible RGB. Compressed-byte identity is
not a correctness target.

Weighted bpp is `8 * total_bytes / total_pixels`; mean bpp gives each image
equal weight. Stage scopes overlap when nested/concurrent and are not wall-time
percentages. Profiling runs are separate from timed runs. These targeted samples
are not evidence of corpus-wide improvement.

## Results

All 128 complete encodes passed official pixel comparison. No timeouts or
encoder failures occurred. `paired-e9/` holds 80 trials; `paired-threads/` holds
48 supplemental trials on the photo, paper screenshot and 16-bit linen texture.

| e9, eight threads, ten images | Total bytes | Weighted bpp | Mean bpp | Native s | MPix/s | Size delta | Time delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 50,852,540 | 7.720446 | 8.036020 | 28.7545 | 1.833 | - | - |
| Coarse continuation | 50,871,536 | 7.723330 | 8.039663 | 28.5570 | 1.845 | +0.037% | -0.69% |
| Palette Route | 50,863,679 | 7.722137 | 8.039603 | 30.3728 | 1.735 | +0.022% | +5.63% |
| Narrow AVX2 | 50,852,540 | 7.720446 | 8.036020 | 29.2805 | 1.800 | 0% | +1.83% |

The different thread counts below use only the three-image subset, not the
ten-image population above. Each value again sums two-run per-image medians.

| e9, three images | 1 thread s | Delta | 4 threads s | Delta |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 29.9490 | - | 9.2437 | - |
| Coarse continuation | 29.9915 | +0.14% | 9.0765 | -1.81% |
| Palette Route | 31.0232 | +3.59% | 8.7910 | -4.90% |
| Narrow AVX2 | 29.3856 | -1.88% | 9.1396 | -1.13% |

## Interpretation

None is enabled by default. Small timing deltas are not stable evidence of a
speedup: for example, the two HDR coarse runs were 15.6006 and 13.9266 seconds.
Excluding HDR, coarse continuation was about 4.1% slower on the other nine
images. Coarse topology worsened size on eight inputs and improved only two
by 12 and 19 bytes. It does reuse preparation, but this initialization is not
a demonstrated improvement over ordinary e9's decisions.

Palette Route reduced `49yekn` by 3,792 bytes (0.553%) and `6826` by 1,643 bytes
(0.040%), but worsened the paper screenshot by 0.990%, transparent `t1/p0` by
15.584%, and screenshot `7dba` by 13.012%. The latter two are small files: the
absolute increases are 1,693 and 1,638 bytes. On transparent `t1/p0`, 342 old
acceptances were vetoed and 460 raw-plane scores reused. Thus caching is active,
but it cannot fix a misleading local objective. The score omits final MA/WP,
cross-channel clustering, full-history LZ and later transform interactions.
Those are hypotheses for the misranking, not separately isolated causes here.

Narrow AVX2 changed no file sizes and passed range/kernel/official pixel tests,
but did not establish an end-to-end speed improvement at e9. A range scan and
i64 output widening are not free, while fixed/pair and Weighted paths are not
replaced. We did not measure their individual shares in this run. Following
the request to wrap up, no e5 or further profiling sweep was started.

The transparent `window_03` in this set is not the black-background version
used by the older fixed-input Palette report. These results use ordinary e9,
not that report's forced Delta modes, so its exact sizes should not be compared.

## Reproduction

Build `portfolio_lab` in Release. Use the existing
`tools/benchmark_ma_kernels.py` with `--manifest`, `--decoder`, and four
`--variant NAME=EXECUTABLE` arguments pointing at that same executable, then:

```text
--experiment coarse=coarse --experiment narrow=narrow
--experiment palette=palette --threads 8 --effort 9 --repeats 2
```

Keep `base` without an experiment assignment. A manifest contains
`{"cases": [{"source": "path/to/original.png"}]}`. Output directory must be new.
The runner cleans inherited experimental environment settings and records
explicit variants, binary provenance, input provenance, original dimensions,
native receipts and every independent pixel-check result. It deletes encoded
and decoded test outputs after validation. No compressed hash target is used.
