# MA Leaf Offset Distribution Experiment

This extends the default-off [static offset experiment](MA_LEAF_OFFSET_20261110.md).
The earlier [100-image comparison](MA_LEAF_OFFSET_CORPUS100_20261110.md) saved
0.2064% with 26.59% more native encoding time. This change does not promote that
experiment into ordinary effort policies, change decoder semantics, or modify
JPEG byte reconstruction.

## New Proposal Mode

`OffsetMode::Distribution` keeps the learned topology, predictors and multipliers.
It collects the same original transformed samples and full-population wide mean
as the existing modes. Nonunit multipliers and proven all-zero residual leaves
are preserved. The follow-up fast-context combination can also refit an existing
nonzero unit-multiplier offset, keeping that original offset as the first candidate.

1. Keep the original offset, zero and the wide mean/median neighborhoods; add zero's neighborhood,
   four dominant sampled residual values and the 1/8, 1/4, 3/4, 7/8 quantiles.
   Neighborhoods contain the center and +/-1, +/-2 with signed wrapping.
   A nonzero original offset also supplies its own neighborhood.
2. Group identical residuals modulo 2^32 and score their multiplicities instead
   of retokenizing every repeated reservoir sample for every proposal.
3. Read each BFS leaf context's actual baseline literal HybridUint configuration
   from the already-built model. This differs from the legacy fitter's fixed
   `(4,1,2)` residual estimate. The candidate model may choose another configuration,
   so this is still an estimate, not a promise of exact coding costs.
4. Reuse cached residual scores for up to three alternating coordinate sweeps.
   Each replacement updates a shared offset-token histogram, including nonzero
   to nonzero changes. No pixels are rescanned during these sweeps.
5. Retokenize and serialize at most one proposed offset tree per domain, retaining
   it only when the existing exact arbitration measures fewer serialized bits.
   The original output remains available. The new mode does not automatically
   encode all older offset modes or guarantee beating those alternative modes.

The shared offset-signaling estimate deliberately remains the legacy `(4,1,2)`
Shannon/extra-bit proxy. It does **not** use the final tree entropy configuration,
tree-token LZ77, table description or complete clustered residual model. These
are remaining ranking limitations, not codec restrictions. Actual serialization
is authoritative and full-file sizes are independently checked by the benchmark.

The maximum candidate count and reservoir capacity bound only optional fitting
work. They do not restrict legal input depth, image size, tree size or MA training.
Dense and sparse scoring scratch support legal wide HybridUint alphabets without
allocating an entire wide alphabet for each candidate. No unsafe code, external
compression library, floating-point pixel conversion or decoder change is added.

## Controls

Native Rust:

```rust
use rs_jxl::modular::refinement::{OffsetMode, OffsetOptions};

let mut options = rs_jxl::EncodeOptions::default();
options.modular.refinement.offsets = Some(OffsetOptions {
    mode: OffsetMode::Distribution,
    samples_per_leaf: 1024,
});
```

Lab CLI:

```powershell
cargo build --release -p rs-jxl-cli --example encoder_lab
.\target\release\examples\encoder_lab.exe input.png --effort 9 --threads 8 --leaf-offset distribution --model-costs --output output.jxl
```

The original `mean`, `median` and `entropy` modes remain unchanged. This is an
explicit native experiment, not a new Python ABI field. Fixed/custom trees and
the lab's outer search remain excluded. Fast-context plans can now opt into
post-fitting; see [the combination experiment](FAST_CONTEXT_OFFSETS_20261111.md).
The measurements below predate that combination and retain their original binary.

## Measurement Plan

Compare baseline, `offset-entropy` and `offset-distribution` at e9/T8 in one frozen
release binary. Use original pixels including invisible RGB and official libjxl
0.12 `djxl` as the oracle. Native-call time excludes startup and image IO.
Identical JXL bytes are not a correctness criterion or optimization goal.

- Eight existing focused real images plus eight synthetic 8/16-bit channel cases.
- Thirty additional preselected stratified entries from the existing 100-image
  manifest, seed `20261111`. Ten categories, selected before candidate results.
- Repeat representative timing comparisons separately. Keep all original results;
  do not replace regressions with better-looking runs.

## Results

Windows x64, release encoder, e9/T8, fixed backend candidate (not the outer
scheduler). No builds or concurrent encoder benchmarks ran during measurements.
The benchmark's shared PNG adapter supplies identical metadata-free snapshots
and checks original samples, including packed PNG expansion and invisible RGB.
This accepts pixel reconstruction, not metadata preservation. Encoded outputs
and decoded images are temporary, not committed or retained as dataset copies.

### Fixed 30-Image Sample

The sample contains 68,188,326 pixels. Each mode has one timed observation per
image; variant order rotates. bpp is `8 * sum(bytes) / sum(pixels)`. Seconds sum
native calls, not total harness wall time. Small timing differences are not
precision speedup evidence.

| Mode | Complete bytes | bpp | Saving vs no offsets | Native seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| No offsets | 35,390,526 | 4.152092 | 0% | 30.6996 | 2.2211 |
| Existing Entropy | 35,310,326 | 4.142683 | 0.2266% | 40.6191 | 1.6787 |
| Distribution | 35,210,215 | 4.130938 | 0.5095% | 39.9652 | 1.7062 |

Distribution saves another **100,111 bytes / 0.2835%** versus Entropy on this
sample: **22 smaller, six equal, two larger**. The two regressions versus Entropy
are `woman.png` (+30 bytes) and `7394483.png` (+40 bytes), both falling back to the
no-offset output. No image grows relative to the no-offset baseline.

Encoding remains about **30.2% slower than no offsets**. The measured 1.6% lower
summed time versus Entropy is small and based on one observation per image;
do not claim a general speed improvement from it.

### Earlier Focused Images

The eight real images contain 19,939,406 pixels. Synthetic correctness fixtures
are excluded from these compression totals.

| Mode | Complete bytes | bpp | Saving vs no offsets |
| --- | ---: | ---: | ---: |
| No offsets | 17,961,370 | 7.206381 | 0% |
| Existing Entropy | 17,860,604 | 7.165952 | 0.5610% |
| Distribution | 17,651,402 | 7.082017 | 1.7257% |

Six of eight beat Entropy. `Celeste_screenshot_02.png` is 121 bytes larger than
Entropy (equal to baseline); `en.wikipedia.org.png` is 1,317 bytes larger than
Entropy (still below baseline). The large photo dominates this group's weighted
total, so 1.73% is not a representative general-corpus expectation.

### Repeated Native Timings

Three rotated observations per mode, all independently decoded and verified.
Sizes agree with the initial observations. Medians include fitting, optional
retokenization, model rebuilding and exact arbitration, but exclude startup/IO.

| Image | Mode | Bytes | bpp | Seconds | MPix/s |
| --- | --- | ---: | ---: | ---: | ---: |
| PDF `9bde6aaf...` | No offsets | 412,539 | 2.266499 | 1.3854 | 1.0511 |
| PDF `9bde6aaf...` | Entropy | 404,246 | 2.220937 | 2.4319 | 0.5988 |
| PDF `9bde6aaf...` | Distribution | 400,026 | 2.197752 | 2.2526 | 0.6464 |
| `wallhaven-mlzoy1` | No offsets | 14,178,604 | 13.675351 | 5.2392 | 1.5832 |
| `wallhaven-mlzoy1` | Entropy | 14,091,986 | 13.591807 | 6.6079 | 1.2552 |
| `wallhaven-mlzoy1` | Distribution | 13,889,288 | 13.396304 | 6.8063 | 1.2186 |
| Wikipedia | No offsets | 750,804 | 1.558102 | 2.0458 | 1.8843 |
| Wikipedia | Entropy | 749,037 | 1.554435 | 3.1514 | 1.2232 |
| Wikipedia | Distribution | 750,354 | 1.557168 | 3.3146 | 1.1630 |

## Validation and Disposition

- **165/165 official decoder pixel checks:** 16 focused cases x three modes,
  30 corpus cases x three modes, and three repeated cases x three modes x three
  observations. Synthetic cases cover 8/16-bit gray, gray-alpha, RGB and RGBA.
- **20 offset tests**, **eight refinement tests**, **one model-configuration
  mapping test**, and **13 Python harness tests** pass. The added BFS test
  switches compact/direct configurations and asserts which actual leaf changes;
  repeatability alone is not used to prove context alignment.
- Integer wrapping, wide means, full-zero preservation, nonunit multipliers,
  sparse wide alphabets, deterministic quantiles, nonzero-to-nonzero signaling
  deltas, and rejection of larger serialized proposals are covered.
- The oversized release test build was cancelled; native tests passed using the
  normal test profile. All compression measurements use the optimized release
  CLI. The final added test changes no production code in the frozen binary.
- Formatting and diff checks pass. No production default, decoder, Python ABI,
  JPEG transcoder or old Entropy policy is changed. No new encoded-tree or
  input-size limit is introduced.

**Retain as a default-off native experiment.** It improves the measured offset
proposal quality at roughly the same cost as the earlier experiment, but the
additional work versus ordinary encoding is still too large to apply globally.
Do not claim it always beats Entropy or automatically run both fitting modes.
Reducing/reusing the full-pixel fitting pass and improving the shared-model
estimate are separate future work; neither is implemented by these results.

## Artifacts and Reproduction

All local receipts are under `out/ma-leaf-offset-distribution-20261111/`:

- `sample30.json`: fixed stratified selection from the prior 100-image manifest.
- `focused-e9.json`, `corpus30-e9.json`, `recheck-e9.json`: raw measurements.
- `results-summary.json`, `audit_results.py`: audited aggregates and checks.
- `encoder-lab-distribution.exe`: frozen release executable, SHA256
  `246963788c0ec479310c8627c3852ae348eaf873a41b08bf462859dc468bf11b`.
- Official libjxl 0.12 decoder SHA256:
  `c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.

These hashes establish binary identities, not encoded-output equality goals.

```powershell
python -B tools/benchmark_refinement.py --encoder target/release/examples/encoder_lab.exe --decoder <official-djxl-path> --manifest out/ma-leaf-offset-distribution-20261111/sample30.json --variants baseline offset-entropy offset-distribution --effort 9 --threads 8 --repeats 1 --timeout 600 --out out/leaf-offset-distribution-new-run.json
```

Use a new report path. The timeout is a subprocess failure guard, not an encoder
deadline or a reason to discard difficult images. No timeout occurred in this run.
