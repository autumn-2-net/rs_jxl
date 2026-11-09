# Shallow MA Plans and Shared-Preparation Warm Starts

Status: explicit native experiments. Production effort defaults are unchanged.
This work concerns pixel-lossless encoding, not JPEG reconstruction or lossy
encoding. No fNBLI predictor, context ID, or rANS layout is emitted.

## Two Complementary Experiments

`ModularOptions.fast_context` builds a complete shallow tree without exhaustive
MA refinement. It scans actual transformed channels with the standard JXL
predictors. `PredictorSet::Basic` preserves Zero/Left/Top/Gradient;
`PredictorSet::Full` additionally scores all other standard predictors, including
the actual Weighted predictor with each stream's WP header and fresh state per
view. Full requires more work and memory; it is not a free SIMD speedup.

`auto12` compares a single leaf with partitions on property 10 (`Left-TopLeft`,
vertical) or property 11 (`TopLeft-Top`, negative horizontal). Thresholds are
sampled proposals; residual histograms are accumulated over the full population.
`static` fits constant leaf offsets and rescores them. `adaptive` adds a causal
bias estimate as another **static offset proposal**. The decoder never needs a
new adaptive predictor. `scan=scalar/avx2/avx512` selects the fused Basic4 feature
kernel; in Full mode the extra stateless rows use the native residual dispatcher,
and Weighted retains its native causal state.

`ModularOptions.source_ma_seed` instead creates and refines a tree on the ordinary
MA prepared sample columns. It does **not** run the complete fast-context image
scan and then gather the image again. It shares the strategy, not the literal
full-population offset tree: prepared residual tokens do not preserve the raw
residuals needed to refit those offsets.

| Seed mode | Behavior |
| --- | --- |
| `Only` | Emit the shallow seed; no exhaustive suffix search. |
| `Refine` | Keep initial partitions and resume normal MA learning at their leaves. |
| `Repair` | Also audit initial splits with normal MA search; replace a subtree when a fine split beats its entire estimated seed cost. |

Seed proposals use available properties 0, 10 and 11 and the configured standard
predictor set. They use 12-quantile threshold proposals where the source quantile
gates apply; small/static domains retain exact threshold evaluation. Every leaf
can choose any configured predictor. Refinement retains all configured MA
properties and predictors, not just the seed dictionary.

Sampling, quantization and residual preparation happen once per learning
invocation, not once for an entire image's independent groups. Parent histograms
are passed to children, with one sibling computed by subtraction. Eligible
property histograms can be retained within the existing cache budget and reused
by repair. Refinement resumes existing learner state instead of calling the
pixel gatherer. No cross-image or cross-candidate cache is introduced.

The seed depth (default 3, explicit 1..8) only bounds initial work. It does not
cap final refined depth or add a production leaf limit. Existing caller/decoder
limits, strict/bounded behavior, forced multiplier regions and constant suffixes
remain in effect. `split_bits` (default 24) is an approximate sampled-bit seed
split cost, **not** actual final clustered/header bits or a byte-size guarantee.

## Native and Lab Interfaces

```rust
use rs_jxl::modular::ma_train::{SeedMode, SeedOptions};

let mut options = rs_jxl::EncodeOptions::default();
options.modular.source_ma_seed = Some(SeedOptions {
    mode: SeedMode::Refine,
    depth: 3,
    ..Default::default()
});
```

Examples use the development `encoder_lab` executable, not the installed CLI:

```text
encoder_lab image.png --effort 3 --threads 8 --model-costs
encoder_lab image.png --effort 3 --threads 8 --model-costs --fast-context auto12 --context-bias static --context-predictors full
encoder_lab image.png --effort 9 --threads 8 --model-costs --ma-seed only --ma-seed-depth 3
encoder_lab image.png --effort 9 --threads 8 --model-costs --ma-seed refine
encoder_lab image.png --effort 9 --threads 8 --model-costs --ma-seed repair
```

Seed and fast-context experiments cannot be combined. Fixed/custom trees,
candidate preparation reuse, histogram-binning and channel-tree experiments also conflict with
seeding. An explicit lab experiment overrides the source low-effort fixed tree;
without the switch the original e2/e3 path is unchanged. Other effort policies,
including sampling and available properties, remain unchanged. A low-effort
seed may therefore have fewer available directional properties than e9.

Receipts report seed learners, prepared records, attempted seed splits, refined
leaves, repaired/rejected branches, predictor changes and histogram reuse events.
Zero prepared records must not be described as substantial training work.
These counters describe operations, not independent image scans or final leaves.
Profile scopes include `ma.seed.build`, `ma.seed.refine` and `ma.seed.audit`.

## Measurement

```text
python tools/benchmark_ma_seed.py --encoder path/to/encoder_lab --decoder path/to/official/djxl --case photo.png --case screenshot.png --efforts 2 3 5 9 --threads 8 --warmups 1 --repeats 3 --out out/ma-seed/confirmation.json
```

The six default variants are baseline, fast-basic, fast-full, seed-only,
seed-refine and seed-repair. Every result is decoded by the supplied official
`djxl` and compared against original PNG samples, including invisible RGB and
16-bit values. The benchmark uses real encoding, not estimated-size wins.

Native timings include analysis, preparation, learning and bitstream emission.
Process startup, PNG input decoding, output I/O and validation are excluded.
Reports include bytes, bpp, MPix/s, model bits, native receipts and same-effort
baseline ratios. Input/binary hashes identify the experiment, not an encoded
byte-identity objective. Reports are never overwritten. A subprocess timeout is
a test failure, not an encoder fallback strategy.

Do not infer fNBLI parity or corpus-wide acceptance from these measurements.
Full predictor scoring may improve a shallow tree yet cost more than ordinary
low-effort encoding. Seed partitions may restrict later choices; repair may
recover size while costing more than an ordinary full search. These are measured
tradeoffs, not default-policy assumptions.

## Results

Windows x86-64, Ryzen 9 7945HX, 8 encoder threads, optimized Release build.
Baseline means the existing **Rust direct native candidate at the same effort**,
not official C++ encoding or the outer multi-candidate scheduler. The official
`djxl` 0.12.0 (`a1ebdf2`) is the independent pixel oracle.

Local evidence (ignored `out/` artifacts, not required runtime dependencies):

- `out/ma-seed-20261109/screen.json`: 11 images x 4 efforts x 6 variants,
  264 original-pixel checks, all passed. One measured run per configuration.
- `out/ma-seed-20261109/confirm-e9.json`: 6 images x 6 variants x
  (1 warmup + 3 measured runs), 144 checks, all passed.
- `out/ma-seed-20261109/confirm-e3.json`: the 4K image x 6 variants x
  (1 warmup + 3 measured runs), 24 checks, all passed.
- Frozen executable: `out/ma-seed-20261109/encoder-lab.exe`, SHA256
  `e612367bf619d6da7271ab69e032c9855a7216c922c80ebe9766ac90bccf0005`.

All 432 outputs passed. No encoded-byte equality was required. Native Modular
tests: 569 passed, one pre-existing ignored test; low-effort projection check:
one passed; lab CLI: 23 passed; the three benchmark-helper suites: 29 passed.
Native tests include 8/16/31-bit samples, invisible RGB, all WP presets,
scalar/SIMD residual checks, empty views, multiplier/constant grafting,
unrestricted refinement properties and strict-budget/model-cost boundaries.
The real-file official matrix covers 8/16-bit PNGs; 31-bit validation here uses
the maintained native decoder. Float, multiframe, JPEG reconstruction and Linux
runtime acceptance are not claimed by this experiment. Some source PNGs carry
libpng's known incorrect-sRGB-profile warning; pixel verification still passed.

### Repeated e9 Results

Times are medians of the three measured runs, excluding warmup. `Full` means
`auto12 + static bias + all 14 standard predictors`. These are complete native
encoding times, not just the fast scan. Negative size change is better.

| Image / strategy | Bytes | Size change | Seconds | bpp | MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K `wallhaven-mlzoy1`, baseline | 14,179,178 | - | 5.580 | 13.676 | 1.487 |
| 4K, Basic4 shallow | 14,306,851 | +0.900% | 1.753 | 13.799 | 4.732 |
| 4K, Full shallow | 13,924,746 | -1.794% | 2.517 | 13.431 | 3.295 |
| 4K, seed-only | 14,194,754 | +0.110% | 2.257 | 13.691 | 3.675 |
| 4K, seed-refine | 14,193,624 | +0.102% | 5.191 | 13.690 | 1.598 |
| 4K, seed-repair | 14,182,003 | +0.020% | 5.340 | 13.679 | 1.553 |
| Clover, baseline | 1,673,489 | - | 1.128 | 12.768 | 0.930 |
| Clover, Full shallow | 1,677,373 | +0.232% | 0.407 | 12.797 | 2.576 |
| Web `news.ycombinator.com`, baseline | 110,876 | - | 0.482 | 0.462 | 3.990 |
| Web, Full shallow | 105,784 | -4.593% | 0.401 | 0.440 | 4.794 |
| Kodak 19, baseline | 436,950 | - | 0.756 | 8.890 | 0.520 |
| Kodak 19, Full shallow | 480,152 | +9.887% | 0.303 | 9.769 | 1.300 |
| Kodak 19, seed-repair | 431,245 | -1.306% | 0.765 | 8.774 | 0.514 |
| PDF `9bde...`, baseline | 412,541 | - | 1.578 | 2.267 | 0.923 |
| PDF, Full shallow | 523,317 | +26.852% | 1.293 | 2.875 | 1.126 |
| PDF, seed-refine | 410,421 | -0.514% | 1.548 | 2.255 | 0.940 |
| PDF, seed-repair | 409,112 | -0.831% | 1.602 | 2.248 | 0.909 |
| Alpha icon, baseline | 34,061 | - | 0.350 | 1.039 | 0.749 |
| Alpha icon, Full shallow | 56,488 | +65.844% | 0.539 | 1.724 | 0.486 |
| Alpha icon, seed-refine | 33,658 | -1.183% | 0.388 | 1.027 | 0.676 |

The 4K seed receipts show the same 16,219,035 prepared records across 135 learning
invocations for all three seed modes. Refine resumes 1,080 leaves and records
2,970 histogram reuse events; Repair replaces 462 seed branches and records
3,320 reuse events. Reuse is real, but it does not remove the subsequent fine
search. It is therefore **not** a general large speedup over normal MA learning.

### Low-Effort Check

Repeated e3 results on the same 4K image:

| Strategy | Bytes | Seconds | bpp | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 14,532,233 | 0.467 | 14.016 | 17.775 |
| Basic4 shallow | 14,288,193 | 0.802 | 13.781 | 10.338 |
| Full shallow | 13,936,325 | 1.633 | 13.442 | 5.078 |
| Seed-only | 14,398,851 | 0.666 | 13.888 | 12.449 |
| Seed-refine | 14,311,382 | 0.642 | 13.803 | 12.920 |
| Seed-repair | 14,311,281 | 0.628 | 13.803 | 13.205 |

Full saves 4.10% relative to e3 but takes 3.50x the time. On this image it is
also smaller and faster than ordinary e9; that cross-effort observation does
not make it a drop-in e3 replacement or establish a general winner.

For completeness, the single-run screen's Full shallow size changes relative
to each effort's own baseline were:

| Image | e2 | e3 | e5 | e9 |
| --- | ---: | ---: | ---: | ---: |
| 4K `wallhaven-mlzoy1` | -5.36% | -4.10% | -2.19% | -1.79% |
| Kodak 19 | -6.02% | +1.48% | +2.96% | +9.89% |
| Tecnick 023 | -5.06% | +0.33% | +1.53% | +4.87% |
| Game UI | -13.04% | -26.18% | -6.33% | +12.90% |
| Web | -4.47% | -8.30% | -4.37% | -4.59% |
| Clover | -8.04% | +0.74% | +0.73% | +0.23% |
| Alpha icon | +1.81% | +19.69% | +27.43% | +65.84% |
| Gray photo | -4.08% | +9.59% | +10.73% | +13.46% |
| Small 16-bit photo | -2.82% | -1.87% | -13.72% | -14.42% |
| PDF screenshot | -6.10% | -20.04% | +5.14% | +26.85% |
| `wallhaven-49yekn` | -22.26% | -4.34% | -4.44% | +7.82% |

These percentages do not include a time penalty; low-effort Full commonly costs
more than the original fixed tree. The raw screen retains times, bpp and MPix/s
for every variant, including all baseline e2/e3/e5 runs.

### Decision

Keep both tools opt-in. Full standard-predictor statistics improve the previous
Basic4 fast route on important cases, but sparse/transparent and detailed
content still need richer context partitioning. Shared-preparation warm starts
recover useful size on some images, but retaining normal refinement also retains
much of its cost. Repair is not a universally better choice than Refine.

Do not replace production low-effort or e9 routing from these results. A future
admission rule must be checked on held-out images and include counterexamples;
choosing the smallest row after encoding all six variants is an oracle, not a
cheap implementable router. No fNBLI throughput/compression equivalence was
measured or claimed.
