# Static MA Leaf Offset Experiment

Follow-up: the [fixed 100-image test](MA_LEAF_OFFSET_CORPUS100_20261110.md)
measured a 0.2064% weighted byte reduction for Entropy, with 400/400 official
pixel checks passing. Its broader sample has smaller gains than the focused
eight-real-image results below. Defaults remain unchanged.

## Scope

This opt-in native Rust experiment fits a fixed bias after MA learning. It keeps
the learned topology, properties and predictors. It does not introduce a new
predictor, adaptive decoder state or nonstandard entropy coder. JPEG coefficient
reconstruction and lossy encoding are not changed. Default behavior is unchanged.

Ordinary learned leaves start at offset zero, but the encoder already uses
nonzero offsets for verified constants and experimental fast-context trees.
This is therefore an extension of offset fitting, not the first support for the
standard field. It borrows the static bias idea, not the complete online CDCR or
Multi-ctx2 method. Context threshold seeding is a separate future experiment.

## Algorithm

`modular/leaf_offset.rs` proposes offsets and `modular/refinement.rs` arbitrates
actual serialized results. No Python preprocessing participates in encoding.

1. Traverse original transformed channel samples using the existing causal
   JXL predictors. Map contexts to leaf nodes in breadth-first order.
2. Only consider existing zero-offset, unit-multiplier leaves. Preserve existing
   offsets and nonunit multipliers; sampled evidence cannot prove divisibility.
3. Accumulate a full-population signed mean of wide residuals in i128. Keep a deterministic
   per-leaf reservoir for the median and entropy estimate. The default capacity
   is 1024 i64 residuals per eligible leaf. This bounds only fitting statistics, not
   input dimensions, encoded leaves or MA training.
4. Compare zero against rounded mean, sampled lower median, or both and their
   +/-1 and +/-2 neighbors. Mean rounding is nearest, ties away from zero.
   Wide mean/median proposals are mapped modulo 2^32 to a legal signed offset;
   averaging itself is performed before narrowing. Preserve full-population
   zero-residual leaves even when offset-token label merging predicts savings.
5. Rank using signed residual packing and HybridUint(4,1,2) token entropy plus
   extra bits, scaled to the leaf population. Add a marginal estimate for the
   shared tree offset-context entropy and extra bits, updating that histogram
   after each proposed replacement in BFS order. Zero remains on ties.
6. If any leaves improve that proxy, retokenize and serialize one proposed tree.
   Retain it only if actual serialized tree/model/residual cost is smaller than
   the original. Shared sections include their TOC size costs. This is not a
   second MA training pass. Predictor refitting, when also enabled, is an
   independent alternative rather than a combinatorial joint search.

The proxy is not the final bit cost: actual entropy clustering/configuration,
tree coding, LZ77 and interacting offset frequencies can change the result.
Reservoir estimates may also miss a useful bias. The exact arbitration prevents
adopting a locally estimated win that encodes larger; complete-file sizes are
checked independently by the benchmark.

Residual arithmetic preserves the encoder contract: subtract predictor and
offset in the wide domain, then narrow modulo 2^32 for unit multipliers. Fitting
uses the equivalent wrapped residual subtraction. No float pixel conversion is
introduced; floating point is only used for entropy cost estimates.

Nonzero offsets can disable gradient/WP/LUT fast paths. Generic property
traversal also differs from the specialized gradient LUT at extreme overflowing
gradients. The proxy uses generic traversal, and the proposed tree is always
retokenized using its final encoding path. This is another reason to measure
complete outputs and decode correctness rather than trust proxy improvements.

## Controls

Native Rust:

```rust
use rs_jxl::modular::refinement::{OffsetMode, OffsetOptions};

let mut options = rs_jxl::EncodeOptions::default();
options.modular.refinement.offsets = Some(OffsetOptions {
    mode: OffsetMode::Entropy,
    samples_per_leaf: 1024,
});
```

`None` disables fitting. A zero capacity is invalid. Explicit fixed/custom trees
and fast-context plans conflict with fitting so their semantics are not silently
overridden. This experimental control is not added to the Python ABI yet.

Lab CLI, built with `cargo build --release -p rs-jxl-cli --example encoder_lab`:

```powershell
.\target\release\examples\encoder_lab.exe input.png --effort 9 --threads 8 --leaf-offset entropy --leaf-offset-samples 1024 --model-costs --output output.jxl
```

Modes: `mean`, `median`, `entropy`. The switch cannot be combined with the outer
`--search`, `--scan-only` or `--fast-context` experiments. Diagnostics report
proposal/win counts, changed leaves, retained samples, population, candidates,
fitting time, model time and writing time. Proposed leaves are not necessarily
accepted leaves. `saved_bits` is measured serialized domain savings.
Phase timers sum work across domains, which can execute concurrently. They are
not a disjoint wall-time breakdown and should not be added to infer elapsed
time. The top-level `seconds` field measures the whole native encoding call.

## Validation

`tools/benchmark_refinement.py` supports `offset-mean`, `offset-median` and
`offset-entropy`, alongside baseline and the existing refinement experiments.
It checks original PNG samples, including transparent RGB, against official
`djxl`, and rejects complete-file size regressions. Encoded-byte identity is
not a correctness requirement or compression target.

Encoder timing is the native call, excluding CLI startup and image I/O.
Decoder wall timing includes startup and PNG I/O and is not a decoder-kernel
benchmark. All candidate trials execute serially with rotated variant order.
The process timeout is a harness failure guard, not an encoder search policy.

## Results

Windows x64, release build, effort 9, 8 encoder threads, single native candidate
(not the outer scheduler). No LZ77 experimental switches or tree threshold
changes were enabled. The baseline is this same binary with offset fitting off,
not a comparison against official cjxl's encoder.

Artifacts, relative to the repository root:

- `out/ma-leaf-offset-20261110/focused-e9.json`: 8 real images plus 8 synthetic
  fixtures, four modes, one observation each; 64 official pixel checks.
- `out/ma-leaf-offset-20261110/recheck-e9.json`: PDF screenshot, code screenshot
  and photo, four modes, three rotated-order repetitions; 36 official checks.
- Encoder SHA256: `6c7c47d7cd7549b6e61b38e05c705e707b2f23cfb0745c670157ce9c22f39156`.
- Official libjxl 0.12 djxl SHA256:
  `c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.

All **100/100** official decode comparisons passed, including 8/16-bit gray,
gray-alpha, RGB and RGBA with invisible RGB. No complete-file growth occurred.
Several proposed domains were rejected by actual encoding; for example the
`n6qzmx` entropy run proposed 10 domains but accepted only one. Synthetic
fixtures are correctness coverage, not representative corpus compression gains.

### Real-image sizes

The eight real images contain 19,939,406 pixels. Savings are relative to the
same per-image baseline, not PNG or a separately optimized scheduler result.

| Image | Baseline bytes | Mean bytes / saving | Median bytes / saving | Entropy bytes / saving |
| --- | ---: | ---: | ---: | ---: |
| PDF `9bde6aaf...` | 412,541 | 404,919 / 1.848% | 404,598 / 1.925% | 404,248 / 2.010% |
| Code `3dd595a3...` | 126,449 | 126,155 / 0.233% | 125,331 / 0.884% | 126,040 / 0.323% |
| `wallhaven-n6qzmx` | 459,153 | 459,108 / 0.010% | 459,113 / 0.009% | 459,150 / 0.001% |
| Celeste | 177,696 | 177,525 / 0.096% | 177,399 / 0.167% | 177,575 / 0.068% |
| Wikipedia | 750,804 | 748,951 / 0.247% | 745,945 / 0.647% | 749,037 / 0.235% |
| Photo `wallhaven-mlzoy1` | 14,179,178 | 14,125,759 / 0.377% | 14,127,024 / 0.368% | 14,092,560 / 0.611% |
| 16-bit linen displacement | 1,841,476 | 1,839,507 / 0.107% | 1,838,674 / 0.152% | 1,838,161 / 0.180% |
| Gray-alpha cursor | 14,678 | 14,531 / 1.001% | 14,491 / 1.274% | 14,438 / 1.635% |
| **Total** | **17,961,975** | **17,896,455 / 0.365%** | **17,892,575 / 0.386%** | **17,861,209 / 0.561%** |

Aggregate bpp: baseline **7.206624**, mean **7.180336**, median **7.178780**,
entropy **7.166195**. The photo dominates the weighted size total. This small,
deliberately mixed sample does not establish broad-corpus gains.

### Repeated native timings

Median of three observations per image/mode. Startup, input decoding and file
I/O are excluded; fitting, optional second tokenization and exact selection are
included. These are full native candidate timings, not offset-fit-only timings.

| Image | Mode | Seconds | MPix/s | bpp |
| --- | --- | ---: | ---: | ---: |
| PDF | Baseline | 1.382 | 1.054 | 2.266510 |
| PDF | Mean | 2.375 | 0.613 | 2.224634 |
| PDF | Median | 2.388 | 0.610 | 2.222871 |
| PDF | Entropy | 2.521 | 0.578 | 2.220948 |
| Code | Baseline | 0.421 | 5.079 | 0.473250 |
| Code | Mean | 0.562 | 3.802 | 0.472150 |
| Code | Median | 0.581 | 3.679 | 0.469066 |
| Code | Entropy | 0.647 | 3.306 | 0.471719 |
| Photo | Baseline | 4.975 | 1.667 | 13.675905 |
| Photo | Mean | 6.414 | 1.293 | 13.624382 |
| Photo | Median | 6.333 | 1.310 | 13.625602 |
| Photo | Entropy | 6.127 | 1.354 | 13.592361 |

The exploratory PDF entropy timing was **9.055 s**, but the three follow-ups
were **2.521 / 2.265 / 2.726 s**, all with the same output size and exact decoded
pixels. The cause of the initial timing outlier is not established; do not use
it as evidence of a stable 9-second backend regression. The raw first-pass
record is preserved, not silently replaced. Other five images have only one
timing observation, so small speed differences there should not be interpreted.

Median djxl process wall time for baseline versus entropy: PDF **0.380/0.380 s**,
code **0.214/0.215 s**, photo **1.982/1.960 s**. These include PNG I/O and startup;
they do not prove unchanged decoder kernel throughput or preservation of SIMD
fast paths. No obvious end-to-end decode regression appeared in these checks.

### Decision

Keep this as a **default-off native experiment**. The standard offset field
provides measurable compression gains, but fitting and exact arbitration cost
more than the baseline, and some images gain almost nothing. Do not run all
three modes automatically. The entropy proxy is not globally optimal: median
beats it on the code screenshot and Wikipedia despite a smaller candidate set,
because shared models, LZ77 and interactions between leaves affect actual size.

Useful next work is a cheap domain-level rejection gate and reducing/reusing
the extra tokenization work, not changing JXL's predictor semantics. Context-3
threshold seeding and low-effort fixed-tree integration remain separate,
unimplemented experiments in this change.

### Reproduction

```powershell
python -B tools/benchmark_refinement.py --encoder target/release/examples/encoder_lab.exe --decoder <official-djxl-path> --case <input.png> --synthetic --variants baseline offset-mean offset-median offset-entropy --effort 9 --threads 8 --repeats 3 --out out/leaf-offset-new-run.json
```

The report path must be new. Fixtures and encoded outputs are temporary; only
JSON reports remain. Native tests cover wide means, wrapping token semantics,
BFS mapping, constant and multiplier protection, deterministic reservoirs,
weighted history, invalid inputs, shared signaling deltas and actual-cost
arbitration. Existing leaf refitting tests also pass after sharing its input
validation helper.
