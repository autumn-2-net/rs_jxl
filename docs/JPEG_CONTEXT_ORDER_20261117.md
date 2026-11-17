# JPEG block context and coefficient order experiments

These opt-in experiments extend the JPEG AC work on `7b69a48`. They borrow
the proposed grouping/order ideas, not PJA's predictors or private coding
format. Every candidate still uses standard JXL VarDCT contexts, coefficient
permutations and entropy coding. Original JPEG coefficients, metadata and
reconstruction information are preserved. Ordinary encoding defaults do not
change, and no multi-encode tournament is added to ordinary e9.

On the existing 45-image corpus, none of the 12 individual policies reduces
aggregate size. Retain these as explicit experiments, not defaults. This does
not exhaust possible Block Context Map searches or coefficient permutations.

## Block context candidates

`jpeg::tuning::Options::block_context` accepts `BlockContextPolicy`:

| CLI value | Meaning |
| --- | --- |
| `source` | Existing source-derived DC thresholds and context map |
| `channel` | No DC thresholds; independent color channels |
| `shared-chroma` | Source DC cuts and luma bins; corresponding paired chroma bins share a context |
| `paired-luma` | Existing source thresholds; adjacent luma bins share a context |
| `quantile4`, `quantile5` | Up to four/five equal-mass DC bins, independent color channels |
| `activity4`, `activity5` | Four/five weighted-quantile bins; each block weighs one plus its nonzero AC count |

Quantiles use exact signed adjusted luma DC values from borrowed coefficient
blocks, including padded MCU blocks. The existing reversible-DC preparation
requires the encoded DC to equal adjusted DC, so the cuts and the decoder's
bucket comparisons use the same integer coordinates. Strict `DC > threshold`
semantics retain ties in the lower bin; ties can reduce the number of bins.
Activity is only a proposal heuristic, not an exact entropy-cost calculation.

Five bins with three independent channels need 15 block contexts, within the
existing 16-context model. Grayscale retains one shared synthetic chroma
context. Map layout is Y, X, B, with 13 coefficient-order classes; only DCT8 is
populated. `DcContexts::from_parts` validates thresholds, map shape and dense
context IDs. New policies are prepared before DC groups and AC tokenization;
the same model reaches qdc construction and header serialization.

## AC order candidates

`jpeg::tuning::Options::ac_order` accepts `AcOrderPolicy`:

| CLI value | Meaning |
| --- | --- |
| `source` | Preserve the existing adaptive order and signaling state |
| `zigzag` | Default DCT8 order with no custom-order signaling |
| `edge-first`, `edge-last` | Stable partition of the current adaptive AC order |
| `band-edge-first`, `band-edge-last` | Increasing frequency band `(u+v)/3`, then the selected edge preference |

The 63 AC positions comprise 14 axis-edge positions (`u == 0 || v == 0`) and
49 interior positions. DC remains first. Position indexes address already
transposed JXL blocks; there is no second transpose. Each channel is reordered
independently, preserving its source rank within a category. Custom order
signaling is written only when at least one resulting channel is nondefault.
The same order is used for tokenization and serialization.

This deliberately tests fixed, bounded candidates. It is not an implementation
of PJA, nor a guarantee that early edges cost less: scan position and remaining
nonzero counts influence JXL's existing coefficient contexts.

## Reproduction and measurement

Build the native example with:

```powershell
cargo build --release -p rs-jxl-cli --example jpeg_entropy_lab
```

For a single explicit experiment:

```powershell
.\target\release\examples\jpeg_entropy_lab.exe input.jpg --effort 9 --threads 8 --cooperative --symmetric-ma --cfl-peak --block-context activity4 --ac-order band-edge-first
```

The example verifies native byte-exact JPEG reconstruction, prints a JSON
receipt, and writes JXL only if `--output` is supplied. For paired measurement
with independent official reconstruction:

```powershell
python tools/benchmark_jpeg_transcode.py --encoder lab.exe --baseline lab.exe --decoder djxl.exe --manifest cases.json --out experiment.json --effort 9 --threads 8 --repeats 1 --baseline-arg=--cooperative --baseline-arg=--symmetric-ma --baseline-arg=--cfl-peak --candidate-arg=--cooperative --candidate-arg=--symmetric-ma --candidate-arg=--cfl-peak --candidate-arg=--block-context=activity4
```

Every invocation performs one complete encode, not a cached candidate choice.
The control is the previous explicit cooperative + symmetric MA + CfL peak
combination. AC histogram sets remain one and AC LZ remains source unless
explicitly stated. Timing includes native JPEG parsing and encoding, excluding
process startup, disk I/O, dimension inspection and reconstruction. Trial order
alternates. Small timing changes from single-repeat runs are diagnostic only.

Both native reconstruction and official djxl 0.12 must recover the entire
original JPEG file, not just pixels. Temporary outputs are removed. Reports
retain input/binary hashes, commands, actual block-context/threshold counts,
and 12 disjoint writer-derived bit costs summing to the complete output size.
Bpp uses total JXL bits divided by total original pixels; the tiny metadata
fixtures make an unweighted per-image bpp mean unsuitable here.

Explicit overrides cannot use the source-search AC cache. The legacy
`--paired-luma` flag cannot be combined with a new block-context override;
`--block-context paired-luma` is its unambiguous equivalent.

## Results

Release on Windows x86_64, e9, eight threads, one repeat per paired trial.
All 12 alternatives were first checked on seven diagnostic inputs, then each
was independently run on the full existing 45-image manifest. This includes
grayscale, RGB, 4:4:4/4:2:2/4:2:0/4:4:0, progressive, restart markers,
non-interleaved scans, metadata fixtures, illustrations and large photographs.
The 16 diagnostic inputs share some content; the remaining inputs are nine
project JPEGs and 20 CC12m originals, not a random population sample.

The shared control is 27,372,627 B, 2.061960835 weighted bpp, across
106,200,376 pixels. Original JPEGs total 33,406,801 B. Positive changes below
mean larger JXL files; times are total native seconds over all 45 images.

| Policy | JXL bytes | Change | Weighted bpp | Control / trial seconds | Trial MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared chroma | 27,379,575 | +0.025383% | 2.062484223 | 17.790 / 17.403 | 6.102 |
| Paired luma | 27,389,626 | +0.062102% | 2.063241358 | 17.550 / 17.671 | 6.010 |
| Channel only | 27,482,499 | +0.401394% | 2.070237416 | 17.741 / 17.079 | 6.218 |
| Quantile 4 | 27,403,229 | +0.111798% | 2.064266062 | 18.004 / 17.855 | 5.948 |
| Quantile 5 | 27,406,906 | +0.125231% | 2.064543048 | 17.892 / 18.059 | 5.881 |
| Activity 4 | 27,407,710 | +0.128168% | 2.064603613 | 17.945 / 17.947 | 5.918 |
| Activity 5 | 27,415,640 | +0.157139% | 2.065200974 | 17.894 / 18.110 | 5.864 |
| Zigzag | 27,411,944 | +0.143636% | 2.064922557 | 17.781 / 18.078 | 5.874 |
| Edge first | 27,384,436 | +0.043142% | 2.062850399 | 17.911 / 18.533 | 5.730 |
| Edge last | 28,601,394 | +4.489036% | 2.154522993 | 17.613 / 23.746 | 4.472 |
| Band edge first | 27,476,562 | +0.379704% | 2.069790186 | 18.058 / 18.140 | 5.855 |
| Band edge last | 27,463,129 | +0.330630% | 2.068778288 | 17.670 / 18.364 | 5.783 |

The zigzag experiment still computes the source adaptive proposal before
replacing it. Its timing therefore does not measure an optimized direct-zigzag
fast path. This does not affect the byte-size comparison.

### Why some wins do not aggregate

Shared chroma improves 31 images, worsens 12, and ties two. It saves
11,137.625 byte equivalents in AC model/order bits, but adds 18,169.375 in
AC data. Global signaling and alignment account for the remainder of the
6,948-byte total regression. The DC stream, auxiliary model and reconstruction
payload are unchanged. Channel-only makes the same trade more aggressively:
39,306 B saved in AC models versus 149,773.125 B added to AC data.

| Subset | Control bytes | Shared chroma delta | Channel-only delta |
| --- | ---: | ---: | ---: |
| 16 diagnostics | 2,579,213 | -823 B | +2,920 B |
| 9 project JPEGs | 22,620,524 | +10,066 B | +110,663 B |
| 20 CC12m originals | 2,172,890 | -2,295 B | -3,711 B |

The nine project inputs account for 82.64% of baseline bytes. Shared chroma's
small-image gains cannot offset them. Conversely, all four quantile/activity
policies worsen all 20 CC12m samples: their aggregate AC payload improves by
roughly 509..1,639 byte equivalents, but AC model bits grow by 15,179..23,536.
Balancing DC sample counts or AC activity is not the same objective as minimizing
the final conditional model and data cost.

Zigzag reduces `wallhaven-zpq1dg.jpg` by 26,089 B and the illustration
`145702319_p0.jpg` by 4,304 B, but increases `wallhaven-yqxzqx.jpg` by 43,084 B.
Putting all 49 interior coefficients before the 14 axis-edge coefficients
worsens every input: AC data grows by 1,234,071.875 byte equivalents despite
slightly cheaper AC models. This is not just permutation-signaling overhead.

An after-the-fact oracle selecting among the control and all 12 individual
policies totals 27,328,754 B, saving 43,873 B (0.160281%). This is only the
headroom among tested candidates, not an achieved one-pass router or an upper
bound on JPEG/JXL compression. No such tournament is enabled by default.

### Combinations and threads

Both promising small-sample combinations were also tested on all 45 originals:

| Combination | JXL bytes | Change | Weighted bpp | Control / trial seconds | Trial MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared chroma + edge first | 27,388,223 | +0.056977% | 2.063135671 | 17.799 / 18.025 | 5.892 |
| Shared chroma + zigzag | 27,413,508 | +0.149350% | 2.065040372 | 17.783 / 17.326 | 6.130 |

Neither combination recovers an aggregate ratio gain. Including both in the
hindsight oracle yields 27,326,382 B, -46,245 B (-0.168946%), still not enough
to justify a default 15-encode tournament including the control.

Shared chroma + edge first was repeated twice with one thread on the seven
diagnostic inputs. Median-per-image summed times were 3.997 -> 4.096 s
(control -> experiment), or 4.698 -> 4.585 MPix/s. The matched seven inputs
from the eight-thread combination run were 2.368 -> 2.431 s, or
7.931 -> 7.725 MPix/s. Both thread counts produced the same sizes:
3,473,378 -> 3,477,411 B (+0.116112%). Thread dispatch therefore does not
rescue this candidate's ratio, and no speed win is established.

Additional seven-input checks combined shared chroma, edge first and four AC
histogram sets at e3 and e10. Both passed reconstruction: e3 totals
3,490,316 -> 3,494,983 B, e10 3,473,666 -> 3,479,082 B. These are compatibility
checks of combined features, not isolated attribution to any one policy.

## Validation and decision

- 406 JPEG-related Rust unit tests passed; nine external-environment tests
  remained ignored. New tests cover every context/order pair on RGB and gray
  fixtures, validated maps, DC boundaries, permutation signaling and immutable
  input coefficients.
- The native example CLI test and 23 benchmark-harness tests passed. Explicit
  context overrides require actual count receipts consistent with the input's
  grayscale/color layout. Identical repeated CLI arguments are normalized;
  conflicting modes are rejected.
- All 29 real benchmark reports passed: 1,484 encodes each verified through
  native reconstruction AND official djxl 0.12 against original JPEG bytes.
  This counts repeated controls, not 1,484 distinct images.
- Scoped Rust formatting and diff whitespace checks passed. No decoder,
  ordinary e9 defaults, Python bindings, build settings or CI changes.

Keep the policies opt-in. The data supports a small-image versus large-image
tradeoff, not a generally better context partition or scan order. Further work
would need measured conditional data-plus-model cost to guide selection, not
simply more DC bins or a fixed preference for edge coefficients. There is no
claim here that the current encoder is globally optimal or that every possible
context-map/order search has been exhausted.

Local artifacts are under `../jxl_python/out/jpeg-context-order-20261117/`
relative to the main `rs_jxl` checkout: frozen `candidate.exe`, 12 `screen-*`
reports, 14 `full-*` reports, serial/compatibility reports and `summarize.py`.
The original manifests remain in `../jxl_python/out/jpeg-opt-20261116/`.
Raw timing and per-image costs are retained for follow-up; no compressed
dataset or reconstructed JPEG copies are retained.
