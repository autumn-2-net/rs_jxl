# Shallow MA Plans With Distribution Offsets

Status: opt-in native experiment. Default efforts, JPEG byte reconstruction,
decoder semantics and Python ABI are unchanged.

## Purpose

The earlier [shallow-plan experiment](MA_WARM_START_EXPERIMENT_20261109.md)
removed exhaustive MA split search, but sometimes produced larger files.
This combines those trees with [distribution offset fitting](MA_LEAF_OFFSET_DISTRIBUTION_20261111.md).
It is not a new tree learner and does not resume exhaustive split search.

The topology and predictor of each shallow leaf remain frozen. Distribution
fitting now considers existing nonzero offsets on unit-multiplier leaves as
well as zero offsets. It retains the original offset as the first candidate
and may replace it with zero or another signed value. Perfect residual-zero
leaves, including nonzero constant predictors, and nonunit multipliers are
preserved. Mean/median/entropy modes retain their old zero-offset eligibility.

Candidate offsets are absolute offsets relative to the standard predictor,
not corrections added to the previous offset. Wide residuals are narrowed
only with the same modulo-32-bit semantics as tokenization. No pixel rounding,
custom prediction rule or decoder extension is introduced.

Each domain first encodes its original shallow tree, then fits offsets with
the actual baseline literal HybridUint configurations and tests at most one
new offset tree. Actual serialized tree/model/residual size decides acceptance;
ties retain the original. This guarantees the local cost decision, not a global
compression optimum. The corpus harness additionally checks complete-file size
against each variant's own shallow baseline, never against an unrelated e9 tree.

This adds a full selected-predictor traversal and potentially a second backend
encode. It can recover volume without full MA training, but is not free and
must be timed. Basic4 uses Zero/Left/Top/Gradient with no initial bias scan;
Full14 uses all standard predictors plus the existing static-mean bias scan.
The tests keep other effort policies unchanged.

## Controls

```rust
use rs_jxl::modular::fast_context::{Bias, Mode, Options, PredictorSet};
use rs_jxl::modular::refinement::{OffsetMode, OffsetOptions};

let mut options = rs_jxl::EncodeOptions::default();
options.modular.fast_context = Some(Options {
    mode: Mode::Auto12,
    bias: Bias::Static,
    predictors: PredictorSet::Full,
    ..Default::default()
});
options.modular.refinement.offsets = Some(OffsetOptions {
    mode: OffsetMode::Distribution,
    ..Default::default()
});
```

Lab CLI (not the installed CLI):

```powershell
.\target\release\examples\encoder_lab.exe input.png --effort 9 --threads 8 --fast-context auto12 --context-predictors full --context-bias static --leaf-offset distribution --model-costs --output output.jxl
```

Explicit fixed/custom trees are still protected from offset fitting. The lab's
outer search and scan-only mode remain separate experiments. Fast-context
receipts describe the initial shallow tree; refinement receipts separately
report offset proposals, accepted domains, saved bits and fitting work.

## Validation

Measurements and the acceptance decision are recorded below after running the
frozen release binary. Timings include the native encoder's analysis, tree
building, offset fitting and all trial encodes. They exclude CLI startup,
input PNG decoding, output IO and independent official decoder validation.
Correctness compares original samples, including invisible RGB and 16-bit
values. Encoded JXL byte identity is not a correctness criterion.

## Initial Offset Combination Results

Frozen binary `out/fast-context-offsets-20261111/encoder-lab.exe`, SHA256
`dd3a6e42e41eabf8e1d5310020ea883290eb43fe6f5cb5106591ad49d118eb18`.
Official libjxl 0.12 djxl is the independent decoder. These results precede
the separate cross-channel candidate below and must not be attributed to it.

`focused-e9.json` contains three real images and eight 8/16-bit channel fixtures;
`corpus30-e9.json` uses the previous fixed stratified sample, seed 20261111,
covering ten content categories. All **205/205** original-pixel checks passed.
Neither offset combination exceeded its corresponding shallow baseline's file
size. One native timing observation per configuration, rotated serial order,
8 encoder threads, e9; small time differences are not precision speedup claims.

The 30-image sample contains 68,188,326 pixels:

| Strategy | Bytes | Weighted bpp | Native seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Ordinary Rust e9 | 35,390,526 | 4.152092 | 35.871 | 1.901 |
| Basic4, no bias | 38,948,338 | 4.569502 | 30.530 | 2.234 |
| Basic4 + distribution | 38,906,518 | 4.564596 | 50.225 | 1.358 |
| Full14 + static mean | 37,674,390 | 4.420040 | 35.759 | 1.907 |
| Full14 + distribution | 37,665,467 | 4.418993 | 56.323 | 1.211 |

Relative to its own fast baseline, Distribution saves 0.1074% for Basic4
(19 smaller, 11 equal) at 64.51% more time; Full14 saves 0.0237%
(20 smaller, 10 equal) at 57.51% more time. Relative to ordinary e9, the
resulting files are still 9.93% and 6.43% larger on this sample.

The fast learner is not necessarily a fast complete encoder: on brochure
`5020.png`, Basic4 takes 12.03s and Full14 12.74s versus ordinary e9's 6.72s;
post-fitting increases these to 22.13s and 21.73s. A worse residual model can
leave expensive backend work even after eliminating full MA search. Conversely,
the focused 4K `wallhaven-mlzoy1.png` remains favorable to Full14: 13,924,172
bytes / 2.707s versus ordinary e9's 14,178,604 bytes / 5.835s. Distribution only
saves a further 12,052 bytes there and takes 3.745s. These are counterexamples
to treating either the shallow learner or offset fitting as a universal win.

Decision: retain the explicit experiment, **do not enable post-fitting by
default for fast encoding**. Offset corrections cannot recover context
partitions absent from the shallow tree. Improving its original scan/partition
choice is a more relevant speed-preserving direction.

## Cross-Channel Candidate

`fast_context::Options.cross_channel` / lab `--context-cross-channel` adds one
Auto12 candidate on standard MA property 18: the absolute clamped-gradient
residual of the nearest compatible earlier transformed channel. It competes
with the existing single-leaf and spatial axes using the same estimated cost.
This is cross-channel context selection, not a new pixel predictor or a
replacement for the existing reversible color transform.

Reference matching is confined to each actual Modular stream and requires
equal width, height, hshift and vshift. Missing references contribute zero;
metadata channels retain their original indices. An allocation-free feature
row is shared by histogram accumulation. No extra complete encoding trial or
exhaustive MA search is introduced. The existing static bias option still
performs its existing second statistics scan.

Native callers set `modular.previous_channels >= 1` to make references
available; zero remains respected. The lab raises the candidate's maximum
previous-channel count to at least one before source-policy projection, so
sampling and tokenization agree. Gray inputs with no earlier channel simply
have no cross-channel candidate. `cross_channel_plans` counts selected channel
plans, not encoded pixels or whole trees.

The old candidate remains in the estimated-cost competition, but that is
**not** the exact serialized fallback used by post-fitting. The benchmark must
report size increases as well as savings and time changes. Defaults stay off.

### Cross-Channel Measurements

Frozen binary `out/fast-context-offsets-20261111/encoder-lab-cross.exe`, SHA256
`5ae196797bd0e5d13478c4e58c7eb59b32fdd092d7cb1aa11ec01f57fc037675`.
Reports: `cross-focused-e9.json` and `cross-corpus30-e9.json` in that directory.
Same 30-image manifest, e9/T8, rotated serial trials, one observation each;
all 150 corpus and 55 focused outputs passed official djxl pixel verification.
The cross variants do not enable Distribution refitting.

| Plan | Total bytes | Weighted bpp | Native seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Ordinary e9 | 35,390,526 | 4.152092 | 39.102 | 1.744 |
| Basic4 | 38,948,338 | 4.569502 | 34.425 | 1.981 |
| Basic4 + cross | 38,652,805 | 4.534830 | 31.731 | 2.149 |
| Full14 + static mean | 37,674,390 | 4.420040 | 39.757 | 1.715 |
| Full14 + static mean + cross | 37,327,980 | 4.379398 | 40.401 | 1.688 |

Against the paired fast plan, Basic saves 0.759% and Full saves 0.919%.
Both have 22 smaller and 8 equal files, none larger in this sample. This is
not a non-growth guarantee. Timing is a single pass, not a significance claim;
for example Full on `7060.png` saves 5.23% but takes 1.824s versus 1.172s.
The totals still exceed ordinary e9 by 9.22% and 5.47% respectively.

The focused PDF screenshot `9bde6aaf390109f97622d35ff1e900ee.png` changes from
523,315 to 455,078 bytes with Full, 1.609 to 1.647 seconds, while ordinary e9
remains smaller at 412,539 bytes. Keep cross-channel selection opt-in: it is
more promising than global post-fitting here, but does not replace learned e9.
