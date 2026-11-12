# Static Joint-Gradient MA Contexts

Status: opt-in native experiment. Default effort policies, JPEG reconstruction,
decoder syntax and the Python ABI are unchanged. This is an independently
implemented C29-inspired context partition, not a C29 port.

## Source and Compatibility

The [C29 author's discussion](https://encode.su/printthread.php?page=4&pp=30&t=4141)
reports that a good context function plus efficiently coded static statistics
can outperform lightweight context mixing in that implementation. This is not
a general performance theorem or a measured result for rs_jxl.

The [dated C29 source](https://github.com/a1024/ABAC/blob/beeb0434dce01cdd6326620e271c5170e2be7f0d/ppmcodec/backup/c29-20250303.c)
uses a recurrent residual-activity state for context selection. Static tables
do not mean stateless contexts. That state is not a standard JXL MA property;
it cannot simply be added without changing decoder semantics. ABAC is GPLv3;
no implementation was copied or linked into this repository.

Instead this experiment uses existing standard JXL properties:

- Property 0: channel dispatch.
- Property 10: Left - TopLeft, wrapping with existing native semantics.
- Property 11: TopLeft - Top, the negative horizontal difference.
- Leaves: Zero, Left, Top or Gradient; offset 0; multiplier 1.

The serialized output remains an ordinary MA tree followed by the existing
JXL entropy stream. No custom predictor, recurrent context state or new ANS
format is emitted. These two gradient properties are a proxy for local
activity, not C29's actual context function.

## Initial Replacement Experiment

`modular/fast_context/activity.rs` is a separate learner selected explicitly
through `fast_context::Mode::Activity25` or `Activity81`:

1. Reuse the existing scalar/AVX2/AVX512 feature scan. Every original transformed
   sample contributes to four signed-residual HybridUint histograms, including
   hidden RGB. Independent stream views retain their own neighbor boundaries.
2. Quantize both signed gradients with fixed thresholds. Activity25 uses
   `[-5, -1, 0, 4]`; Activity81 uses
   `[-65, -17, -5, -1, 0, 4, 16, 64]`. Zero has its own bin.
3. Accumulate a 5x5 or 9x9 joint grid per transformed channel. This differs from
   Auto12, which chooses a single gradient axis with sampled quantile thresholds.
4. Evaluate a fixed balanced hierarchy of rectangles bottom-up. A rectangle can
   become a leaf or split on either gradient at its fixed midpoint. Reuse merged
   histograms for both axis orders; do not sort pixels or repeat sample scans.
5. Choose a leaf predictor using the existing entropy + extra bits + approximate
   model cost. Add the same 24-bit split estimate as Auto12; ties keep the leaf.
   This removes partitions without sufficient estimated gain.
6. Encode the selected tree once. There is no full-e9 fallback, output-size
   competition or hidden second encoding in a variant's timing.

There are at most 81 terminal grid cells and 289 rectangle states per channel
for Activity81, independent of pixel count. This bounds this experimental
topology only; it does not impose a new cap on ordinary MA training. Existing
explicit caller/decoder leaf budgets are honored; strict budgets still error.
Adaptation is limited to predictor selection, axis order and merges, not an
arbitrary optimal MA tree. Thresholds are not bit-depth-scaled, which can be a
compression limitation on high-depth or widely distributed transformed data.

Bias fitting, Full14 and cross-channel partitioning are not implemented for
these two replacement modes; contradictory requests are rejected rather than ignored.
Existing Auto12/full-predictor/cross-channel behavior is untouched.

## Full-Effort Initialization

The replacement experiment does not establish what this idea can achieve at
e9: it deliberately removes ordinary split search and only uses four predictors.
Do not interpret its results as restricting static contexts to low efforts.

`ma_train::SeedOptions.topology` now also accepts `SeedTopology::Activity25` and
`Activity81`. The default remains the existing Quantile initializer. Grid seeds
reuse the ordinary prepared sample population and its weighted residual tokens,
not the replacement learner's four-predictor scan. Every predictor supplied by
the effort policy participates. Grid thresholds are mapped to legal existing
quantizer boundaries, and metadata/multiplier domains retain their skeleton.

- `SeedMode::Refine`: continue the normal MA learner below the initial regions.
- `SeedMode::Repair`: additionally allow ordinary split search to replace an
  initial branch when its cost estimate beats that branch's entire initial tree.
- Refinement keeps the original property/predictor sets, sampling policy and
  explicit caller budgets. Seed depth does not cap the final tree. Cross-channel
  properties remain available when enabled by the caller's normal options.
- Histograms from the grid are reused for initial leaf states; no second image
  gather or four-predictor restriction is introduced. The fixed grid constrains
  initialization only, not the subsequent search.

These are alternative e9 training plans, not a promise that the estimate chooses
the smallest actual codestream. They are measured as raw encodes, without
silently replacing a losing result with ordinary e9 output.

## Reproduction

```powershell
cargo build --release -p rs-jxl-cli --example static_context_lab -j 4
.\target\release\examples\static_context_lab.exe input.png --output activity.jxl --effort 9 --threads 8 --mode activity81 --model-costs
```

For strict official-decoder validation and a bounded matched corpus:

```powershell
python -B tools/benchmark_static_context.py --encoder target/release/examples/static_context_lab.exe --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --case input.png --max-pixels 16000000 --effort 9 --threads 8 --repeats 2 --model-costs --out out/static-context/run.json
python -B tools/plot_patch_tradeoffs.py out/static-context/run.json --out out/static-context/tradeoffs.png --svg --per-case --pareto
```

The harness supports `--manifest` (existing `cases[].original` format) and
`--synthetic` (six RGB/RGBA 8/16-bit controls). It never builds, resizes or silently
skips admitted images. Temporary JXL/decoded PNGs are deleted; the JSON report
records original input and binary provenance. A timeout is a harness failure,
never a timing-based encoder policy or a successful sample.

Each variant uses the same direct native effort projection, with patches Off
and previous_channels=0 to isolate context learning. This is not the outer
multi-candidate scheduler. Native timing includes projection, all analysis,
tree generation and encoding; it excludes process startup, input PNG decoding,
output IO and the independent official djxl check. Corpus bpp is
`8 * sum(encoded_bytes) / sum(width * height)`, not an average of image bpp.
Timings aggregate per-image medians. Actual emitted model costs are recorded
separately; estimated costs are not labeled as measured bit savings.

## Replacement Measurements

Windows, AMD Ryzen 9 7945HX (16 cores / 32 logical processors), Release native
code, e9, 8 encoder threads, two rotated repetitions per real image. The frozen
30-image sample is `out/ma-leaf-offset-distribution-20261111/sample30.json`,
68,188,326 pixels across ten categories. It includes 1/2-bit packed inputs and
16-bit samples; packed pixels are expanded only in the independent oracle.

`out/static-context-20261112/corpus30.json` records these pixel-weighted results:

| Route | Bytes | Weighted bpp | Native seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: |
| Ordinary Rust e9 | 35,390,526 | 4.152092 | 29.318 | 2.326 |
| Existing Basic Auto12 | 38,948,338 | 4.569502 | 19.673 | 3.466 |
| Activity25 replacement | 37,575,090 | 4.408390 | 15.196 | 4.487 |
| Activity81 replacement | 37,369,381 | 4.384256 | 15.898 | 4.289 |

Activity81 is 4.05% smaller and takes 19.19% less time than Auto12 here. Against
ordinary e9 it is 5.59% larger and takes 54.23% as long. Only two images beat e9
in size; some transparent assets and packed documents regress considerably.
For example `muzzle_05.png` grows 33.96% and also takes about 22% longer. These
are reasons not to promote the replacement route as a general e9 default.

Earlier eight-image matched results are in `focused8.json` (64 outputs), six
synthetic controls in `synthetic.json` (24 outputs). The 30-image run adds 240
outputs. All 328 outputs passed official djxl versus original-sample checks;
encoded JXL identity is not an acceptance requirement. Temporary codec outputs
were removed. Local MA tests additionally cover 31-bit samples; this is distinct
from the official 8/16-bit PNG oracle coverage.

The 30-image binary SHA256 is
`4fd85718bf589d94a12c5d5c10e05423e9419cff6faa5958d662c8bc042e2e79`.
Official 0.12 djxl SHA256 is
`c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.
The earlier binary differs only by the subsequent scan/prune profiling-scope
separation; each report records its own exact binary identity.

Plots: `out/static-context-20261112/corpus30.png` and `focused8.png`, with SVG,
CSV and the focused per-image panels beside them. These initial plots use
pixel-weighted bpp. The requested larger 200-image comparison instead uses
arithmetic mean image bpp and image-bootstrap error bars; do not mix the axes.

## Matched 200-Image Comparison

The frozen stratified sample is
`out/static-context-corpus200-20261112/sample200.json`: 200 distinct images,
619,577,505 pixels, ten categories, 18 sixteen-bit PNGs and packed PNG inputs.
Selection does not use the experiments' compressed sizes. Images are not resized.

The matrix contains 30 points, or 6,000 encodes at one repetition:

| Family | Plans |
| --- | --- |
| Ordinary Rust | source efforts 1 through 9 |
| Official cjxl 0.12 | efforts 1 through 9, distance 0 |
| Shallow replacements at e9 | Auto12, Activity25, Activity81 |
| Full e9 initialization | Activity25 Refine, Activity81 Refine, Activity81 Repair |
| Leaf offset experiments at e9 | ordinary Distribution; Auto12 + Distribution |
| Full-predictor shallow experiments at e9 | Full14 Static; Full14 Static + Distribution |
| Full e9 split search experiments | Lookahead width 2; Lookahead width 4 |

The expensive 42-RCT joint search is excluded. All points use eight encoder
threads and automatic patches. These whole-pipeline points are not interchangeable
with the patches-Off isolation measurements above. A replacement plan still uses
the e9 transform/entropy policy, but replaces its MA trainer; it is not ordinary
full-search e9. Refine/Repair retain the full trainer after initialization.

```powershell
python -B tools/benchmark_context_corpus.py --manifest out/static-context-corpus200-20261112/sample200.json --encoder target/release/examples/static_context_lab.exe --cjxl ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/cjxl.exe --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --threads 8 --outdir out/static-context-corpus200-20261112/run
python -B tools/plot_context_corpus.py out/static-context-corpus200-20261112/run/summary.json --out out/static-context-corpus200-20261112/tradeoffs.png --pareto
```

The runner rotates plan order per image and runs encode trials sequentially to
avoid competing timed processes. It independently decodes every output with
official djxl and compares original samples, including invisible RGB. Packed PNGs
are expanded only for checking, not before encoding. Output files are temporary.

`trials.jsonl` checkpoints each attempt. `progress.json` tracks verification and
failures. Resume with the same command plus `--resume`; input, binary, source and
configuration fingerprints must still match. Failed trials are never counted as
successes or silently dropped. No chart is accepted from an incomplete matrix.

The Y coordinate is `mean(8 * bytes_i / (width_i * height_i))`, giving each image
equal weight. Vertical error bars are a paired image-bootstrap 95% confidence
interval, not variation of a codec's quality. Pixel-weighted bpp, total bytes,
total pixels and MPix/s are retained separately in CSV/JSON. Bootstrap intervals
describe this sampled corpus; they do not prove generality beyond its strata.

The X coordinate sums native encode times, excluding process startup, input PNG
decoding, file I/O and independent verification. Rust reports the native timer.
Official cjxl 0.12 reports MP/s rounded to three decimal places; its native time
is reconstructed from that rate and known image dimensions. CSV includes the
corresponding rounding bounds, not a fabricated exact timer or wall-time fallback.

## Completed 200-Image Results

The frozen matrix completed with **6,000/6,000 independently verified outputs**,
zero failures and zero timeouts. Identity checks passed. Wall time including
PNG preparation, official decoding, comparison and temporary I/O was about
163.57 minutes; the sum of all native encode timers was 96.11 minutes. Neither
number is a single configuration's time on the chart.

Selected complete-cohort points (one repetition, eight threads):

| Plan | Mean image bpp | Weighted bpp | Total bytes | Native seconds | MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rust e6 | 6.810007 | 6.020838 | 466,296,945 | 89.622 | 6.913 |
| Rust e9 | 6.626979 | 5.779851 | 447,633,230 | 365.171 | 1.697 |
| Official cjxl 0.12 e9 | 6.614241 | 5.791775 | 448,556,657 | 470.475 | 1.317 |
| Activity81 replacement | 6.905210 | 6.067136 | 469,882,604 | 159.638 | 3.881 |
| Activity81 Refine | 6.664376 | 5.815729 | 450,411,831 | 344.481 | 1.799 |
| Activity81 Repair | 6.634572 | 5.790597 | 448,465,462 | 396.193 | 1.564 |
| Ordinary e9 + Distribution offsets | 6.595476 | 5.740202 | 444,562,542 | 508.648 | 1.218 |

The results do not support promoting the new grid or lookahead routes:

- Activity81 replacement takes 43.72% of ordinary Rust e9 time, but total bytes
  grow 4.97%. Ordinary Rust e6 is both faster and smaller on this corpus, including
  mean bpp, total bytes and weighted bpp. The other tested shallow replacements
  are also dominated by Rust e6 on these aggregate measures.
- Activity25/81 Refine grow total bytes 0.383%/0.621%, while reducing observed
  time by only 3.31%/5.67%. Activity81 Repair grows bytes 0.186% and time 8.50%.
  These are complete e9 refinement trials, not four-predictor replacement trials.
- Lookahead2/4 grow bytes 0.008%/0.010% and take 4.69%/9.97% more time. Some
  individual images improve, but neither aggregate tradeoff is attractive.
- Distribution offsets reduce total bytes 0.686% and mean bpp 0.475%, but take
  39.29% more time. They win on 151 images, tie on 49 and lose on none. This is
  consistent with their existing serialized-domain acceptance check, not proof
  of a zero-regression guarantee for all future inputs.
- Rust e9 takes 22.38% less native time than the official e9 baseline here and
  produces 0.206% fewer total bytes, but its **mean image bpp is slightly higher**.
  The two size metrics weight the cohort differently; do not claim it wins both.

The 95% paired bootstrap interval for offset mean-bpp change versus Rust e9 is
`[-0.039620, -0.024051]` bpp. The corresponding interval for Activity81 Refine is
`[+0.030750, +0.044299]` bpp. These describe this cohort, without correction for
multiple experiment comparisons or a claim of population-wide guarantees.
One timing repetition does not establish a runtime confidence interval.

Artifacts under `out/static-context-corpus200-20261112/`:

- `tradeoffs.png`, `tradeoffs.svg`: arithmetic mean bpp and 95% image-bootstrap
  intervals versus summed native encode time, all 30 points.
- `tradeoffs.csv`: aggregate and individual-image rows, weighted bpp, byte totals,
  MPix/s and official timer rounding bounds.
- `tradeoffs-mean-only.png`, `tradeoffs-mean-only.svg`: mean image bpp versus
  summed native encode time, without error bars.
- `tradeoffs-total-size.png`, `tradeoffs-total-size.svg`: summed compressed MiB
  versus summed native encode time, without error bars. The Pareto frontier uses
  total bytes rather than mean bpp. Each new chart also has a matching CSV.
- `RESULTS.md`: all points, paired changes, wins/ties/losses and worst regressions.
- `run/summary.json`, `run/trials.jsonl`, `run/identity.json`: complete receipts,
  independent pixel-verification evidence and frozen provenance.

The native executable SHA256 is
`76790239ff2a802b4cc4b65020fbd48621eb599db3b2ba98726ebdd5e532e9b1`.
The run fingerprint is
`57e51aaeb135e8f455900207a00f83e0ffa48ad65ff27f248174bb434675fdb8`.
Several source PNGs emitted libpng warnings about invalid iCCP metadata; all
original sample comparisons still passed. This measures pixel losslessness,
not preservation of source metadata.

Validation also includes 33 fast-context tests, 21 seed tests, six CLI tests,
73 Python harness/plot/report tests, 24 new-seed synthetic official-decoder checks
and the 90-output three-format matrix pilot. Benchmark tools require the existing
NumPy, OpenCV and Matplotlib development dependencies; no Python runtime or wheel
dependency was added. Default encoder policies and the Python ABI remain unchanged.

To regenerate the human-readable table from a completed run:

```powershell
python -B tools/report_context_corpus.py out/static-context-corpus200-20261112/run/summary.json --out out/static-context-corpus200-20261112/RESULTS-new.md
```

To replot existing measurements without rerunning any encoders:

```powershell
python -B tools/plot_context_corpus.py out/static-context-corpus200-20261112/run/summary.json --out out/static-context-corpus200-20261112/tradeoffs-mean-only.png --no-error-bars --pareto
python -B tools/plot_context_corpus.py out/static-context-corpus200-20261112/run/summary.json --out out/static-context-corpus200-20261112/tradeoffs-total-size.png --metric total-mib --no-error-bars --pareto
```

These commands preserve the original interval charts. Add `--overwrite` only
when regenerating an existing destination. Point-only exports omit bootstrap
intervals in both the figure and CSV; the original measurements are unchanged.
