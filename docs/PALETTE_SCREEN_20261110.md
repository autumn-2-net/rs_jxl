# Sampled Palette-Order Screening

## Contract

`PaletteOrder::ScreenedRuns` is an opt-in native Rust experiment. Production
defaults, source selectors 0..9, the Python/C ABI, decoder and JPEG transcode
are unchanged. It chooses an order before MA learning, not by encoding every
candidate. It is not a guarantee against increased final compressed size.

```rust
use rs_jxl::modular::palette::{PaletteOptions, PaletteOrder, PaletteTools};

let palette = PaletteOptions {
    tools: PaletteTools {
        order: PaletteOrder::ScreenedRuns,
        ..Default::default()
    },
    ..Default::default()
};
// Set EncodeOptions.modular.palette_policy = Some(palette).
```

The ordinary palette admission policy still applies. Ordered absolute palettes
with at least three channels are eligible; geometry overrides and successful
exact-delta palettes retain their old behavior. Implicit colors are not
permuted. Duplicate promotions use the final mapping's last-assignment rule.
Every channel participates, including invisible RGB and alpha. Exact-zero
delta prefixes are still inserted by the existing transform after ordering.

## Preflight V1

The rule was fixed before measuring the 200-image sample:

1. Retain the existing stable default luma order as the fallback.
2. Count full-tuple raster run starts once, sharing counts among RunFrequency,
   lexical, Hot32 and Hot128. Deduplicate identical candidate permutations.
3. Collect at most 16 contiguous 64x64 original-resolution tiles on a stratified
   4x4 grid. Decode color identities once, then remap those IDs for each order.
   No thumbnail resizing, quantization, or per-candidate full index image.
4. Use the existing native SIMD `cost::estimate_cost` for gradient-context
   HybridUint residual cost. Score the complete palette table as well as sampled
   indices. Expand each tile's estimate to its stratum's pixel population.
5. Split strata into two checkerboard folds. Require more than 3% estimated
   improvement in the combined score and in each nonempty fold. Among qualifying
   alternatives, choose the smallest estimated total; otherwise use default.

An unavailable baseline cost falls back to default; an unavailable alternative
cost rejects that alternative. The estimation kernel's resource/arithmetic
limits are not additional restrictions on otherwise valid encoder inputs.

Tile edges use the cost kernel's edge handling; interior neighborhoods remain
intact. This approximation, the gradient-only predictor model, subsequent MA
learning/LZ77 and entropy clustering can all disagree with final byte size.
Fold agreement is a consistency check, not a statistical confidence bound.
Tiny images may have only one nonempty fold. No file-name exceptions or trained
per-image thresholds are involved.

This first version still performs a separate run scan rather than fusing counts
into ordinary palette discovery. Timing is instrumented before deciding whether
that additional refactor is worthwhile.

## Reproduce

```powershell
cargo build --release -p rs-jxl-cli --example palette_lab
python tools/sample_palette_manifest.py --manifest out/ma-corpus500-20261109/sample.json --count 200 --seed 20261110 --out out/palette-screen-20261110/sample200.json
python tools/benchmark_palette_runs.py --encoder target/release/examples/palette_lab.exe --decoder PATH_TO_OFFICIAL_DJXL --manifest out/palette-screen-20261110/sample200.json --modes default screened --effort 9 --threads 8 --repeats 1 --profile --timeout 1200 --out out/palette-screen-20261110/sample200-v1.json
```

Each mode performs exactly one complete native encode, excluding process startup,
PNG input decoding, file I/O and official decoder verification. The lab compares
one fixed projected backend candidate, not the outer production scheduler's
best-of-search result. Non-palette settings are identical; all patch detectors
are disabled in both modes. Full-frame/10,000-color overrides are used only in
the small diagnostic, not silently applied to the 200-image comparison.

`--profile` records `palette.order.screen`, scan/sample/cost sub-stages and the
selected-order counters. Stage totals are inclusive and can overlap across
threads; do not add them or call their sum elapsed wall time. The JSON preserves
raw profiles and estimates, including rejected choices. Timing includes enabled
profiling overhead in both variants.

All outputs are decoded by official djxl and compared against original PNG
samples, including 16-bit values and hidden RGB. Temporary encoded/decoded files
are removed. Binary and source hashes identify provenance, not optimization
targets. Reports refuse overwrite.

## Four-Image Diagnostic

Windows x86_64, e9, eight native threads, one trial per mode. All modes use
`--full-frame --palette-colors 10000`. No threshold was adjusted after this run.

| Image | Default bytes | Forced runs | Screened | Screen / default | Screen scope ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paper9bde | 269,794 | 253,096 | 253,096 | -6.19% | 33.674 |
| Editor3dd | 110,581 | 94,654 | 94,654 | -14.40% | 17.140 |
| Web8279 | 32,130 | 31,948 | 32,130 | 0.00% | 10.235 |
| Kenney Preview | 53,669 | 56,961 | 53,669 | 0.00% | 10.780 |

Screening retains the two larger gains, rejects Preview's 6.13% size regression,
and also misses Web8279's small 0.57% gain. These four existing diagnostic images
are not independent proof of generalization. Twelve official pixel checks pass.
Report: `out/palette-screen-20261110/preflight-v1.json`.

Diagnostic lab SHA256 (before adding the cost-limit fallback, with the same
scores, candidates and admission thresholds):
`1d39f6ffb26517b8ce07f3e306e2c71927964c088a7c6d195eae9023b55330c8`.
Oracle: official libjxl 0.12 djxl, SHA256
`c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.

## 200-Image Validation

The fixed manifest and measured report are kept under
`out/palette-screen-20261110/`: `sample200.json` and `sample200-v1.json`.
All 400 encodes and original-pixel checks passed; temporary outputs were removed.
There were no encoder/decoder failures, estimation-limit fallbacks, or timeouts.

Seed `20261110` samples the existing 500-image stratified PNG manifest without
reading its encoded results. Quotas are proportional: 32 high-resolution photos,
24 other photos, 16 urban, 16 manga, 16 mixed local, 24 document/web, 16 game/UI,
20 alpha/icon, 16 texture and 20 format controls. All 200 original file hashes
were verified before encoding; duplicate paths and hashes are rejected.

The selected formats include 173 8-bit, 19 16-bit and eight packed 1/2/4-bit PNGs.
Packed PNG data remains packed at encoder input; only the independent expected
pixel representation expands to 8 bits using the established corpus adapter.
No image is resized or quantized. The largest image is 8,100 x 11,708 pixels;
it is retained, not silently excluded to improve timings. The per-process
1200-second harness timeout is a safety guard, never encoder scheduling policy.

Final measured lab SHA256:
`4548825fc7960ea459604d90ff46cebff8329ca0e66a694f0ebc584ec84ea06d`.
Sample manifest SHA256:
`4d85762afd88f92a7354eb86d6e0d255aeec455973dd271571fa2296768f2229`.
Total population: 511,979,597 pixels. One rotated trial per mode, eight native
threads, no concurrent codec jobs or compilation during measurement. These are
single-trial observations, not confidence intervals or stable speed promises.

### Aggregate

| Metric | Default | Screened | Change |
| --- | ---: | ---: | ---: |
| Bytes | 337,302,699 | 336,194,738 | -0.32848% |
| MiB | 321.6769 | 320.6203 | -1.0566 MiB |
| Native seconds, summed | 274.237 | 280.823 | +2.4015% |
| bpp, pixel weighted | 5.27056 | 5.25325 | -0.01731 |
| MPix/s, aggregate | 1.86692 | 1.82314 | -2.3452% |

49 images become smaller, 143 have equal sizes, and eight become larger.
Gross savings are 1,132,716 bytes and gross growth is 24,755 bytes; net savings
are 1,107,961 bytes. Equal size is not a JXL byte-identity assertion.

The biggest input, Graveyard, accounts for 551,116 saved bytes, almost half of
the net saving. Excluding it explicitly as a sensitivity check (not as the
headline result), the other 199 images shrink 0.18916%; their measured native
time is 237.881s vs 236.802s. Small timing differences may be noise. The
full 200-image result, including this expensive case, remains the primary one.

### Per Stratum

| Stratum | Images | Smaller / equal / larger | Byte change |
| --- | ---: | ---: | ---: |
| High-resolution photos | 32 | 4 / 28 / 0 | -0.00844% |
| Other photos | 24 | 4 / 20 / 0 | -0.65145% |
| Urban | 16 | 2 / 14 / 0 | -0.06686% |
| Manga | 16 | 3 / 12 / 1 | -0.00622% |
| Mixed local | 16 | 7 / 9 / 0 | -0.99989% |
| Document/web | 24 | 16 / 6 / 2 | -1.01860% |
| Game/UI | 16 | 6 / 7 / 3 | -0.72066% |
| Alpha/icon | 20 | 2 / 17 / 1 | -0.00635% |
| Texture | 16 | 0 / 16 / 0 | 0.00000% |
| Format controls/mixed | 20 | 5 / 14 / 1 | -0.24468% |

Document/web without Graveyard is -0.34688%. Group labels come from the frozen
input manifest, not from a classifier or a decision based on encoded results.

Selected wins include 7017.png (plot: 31,354 -> 25,994 bytes, -17.10%),
Blobby_volley_2_alpha_6.png (212,022 -> 193,427, -8.77%), 5222.png (document:
211,427 -> 193,714, -8.38%), and windows95.png (11,406 -> 10,567, -7.36%).
These examples do not replace the complete aggregate or regression list.

### All Size Regressions

| Input | Default bytes | Screened bytes | Increase | Percent |
| --- | ---: | ---: | ---: | ---: |
| Kenney tile_0056.png | 133 | 139 | 6 | +4.51128% |
| EmpireOriginal.png | 5,723 | 5,946 | 223 | +3.89656% |
| imazen26 manuscript 6814.png | 2,426,340 | 2,442,973 | 16,633 | +0.68552% |
| imazen26 manuscript 6809.png | 2,522,282 | 2,530,059 | 7,777 | +0.30833% |
| Threes_development_process_art_mockup_07.png | 18,110 | 18,123 | 13 | +0.07178% |
| TetsuSan.png | 277,260 | 277,339 | 79 | +0.02849% |
| imazen26 clipart 9029.png | 111,698 | 111,703 | 5 | +0.00448% |
| phone_PNG48891.png | 2,890,849 | 2,890,868 | 19 | +0.00066% |

Small absolute byte counts are not hidden by rounding percentage regressions to
zero. No thresholds were changed or cases removed after observing these results.

### Screening Work

94/200 images reach the selector. At least one alternative is chosen during
palette preparation on 59 images; 57 images have a changed final size. Attempts
can later be rejected by the ordinary palette admission policy, so these
counters are not proof that every selected palette is present in the stream.

Across all attempts: 1,530 default, 526 runs, 359 lexical, 166 Hot32 and 175
Hot128 decisions. The combined screening scope totals 23.005s of **inclusive
per-thread duration**, not 23s added to encoding wall time. Among the 94 active
images, the per-image summed scope median is 19.105ms, p95 793.594ms, maximum
10.043s on Graveyard's many groups. These sums overlap across workers.

| Nested work | Inclusive summed seconds |
| --- | ---: |
| Sample gathering / palette-map lookups | 13.690 |
| Full run-start scan | 5.282 |
| Candidate index/table cost estimation | 2.344 |

The screen is therefore not uniformly a negligible cost. Repeated lookup and
sample gathering, not just sorting or the SIMD cost kernel, are the main direct
targets for reducing it. Fusing run counts into discovery alone would not remove
the largest measured component.

Graveyard is 94.835 MPix: 42,928,156 -> 42,377,040 bytes (-1.284%), but native
encoding takes 36.356 -> 44.021s. Its summed MA learning scope also increases
142.260 -> 168.529s and local LZ77 scope 43.641 -> 48.960s (both overlap across
threads). Changed index streams affect backend work as well as screening cost;
the entire wall-time difference must not be labeled sorting overhead.

### What Still Fails

EmpireOriginal is a concrete proxy mismatch, not a sampled-pixel correctness
problem. Default is estimated at 77,018 bits; lexical is 71,678 bits, with both
folds predicting improvement. It passes the 3% margin, yet final size grows
3.90%. For tile_0056, all 18x18 pixels are sampled: the proxy predicts 270 ->
257 bits, but actual output grows 133 -> 139 bytes. Increasing sampling density
alone cannot fix that latter failure.

The proxy omits the actual MA predictor/context choices, tree/histogram headers,
and final LZ77/entropy decisions. Both sampling error and model error must be
distinguished. Candidate follow-up work is a cheap multi-predictor/token-cost
cross-check and better accounting for table/model overhead, followed by fresh
validation. This run does not establish that either will eliminate regressions.

## Decision And Checks

Keep `ScreenedRuns` opt-in, not the default scheduler or Python/C numeric API.
It rejects some known harmful orders and achieves real local wins, but the
requested "screen out every growing image" condition is not satisfied.
Do not advertise 192/200 non-growing cases as selection accuracy: many images
never reach this selector, or retain their default order.

- 11 focused palette-order unit tests and 12 integration tests pass on the final
  code; the integration matrix includes 24 native RGB8/RGBA16 roundtrips.
- 10 benchmark-harness tests, including original packed-input retention, and
  five deterministic sampler tests pass.
- The lab CLI argument test, workspace/all-target compilation and formatting
  checks pass. Clippy passes with the same pre-existing unrelated
  `needless_borrow` exemption used by the earlier experiment.
- The 200-image run verifies 400 streams with official djxl. The separate
  four-image diagnostic verifies 12 more. No codestream-identity optimization
  target is used.
- Windows runtime validation only; no Linux/macOS runtime claim.
