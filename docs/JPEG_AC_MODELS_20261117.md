# JPEG AC model experiments

Based on `7b69a48`. Ratio is the primary objective. These are opt-in native
experiments; ordinary JPEG transcoding, source search, CLI and Python defaults
remain unchanged. No decoder changes or pixel re-encoding are involved.

## Compatibility boundary

JPEG AC coefficients use the standard VarDCT context model, HybridUint and
ANS, optionally with the existing LZ syntax. DC and auxiliary integer streams
use Modular/MA. Moving the JPEG AC coefficients to a Modular image is not a
compatible shortcut for byte-reversible JPEG reconstruction: libjxl explicitly
rejects JPEG output from a Modular frame.

The current encoder already emits one full frame with one pass. Disabling
progressive presentation cannot introduce a new AC-dependent DC predictor or
an arbitrary MA model for AC: the decoder would need to know those rules.

Multiple AC histogram groups ARE supported by the existing decoder. AC global
signals the number of groups; each AC image group signals its histogram-set
index, which offsets its original AC contexts. There is still one entropy
model/context map per pass: clustering can share tables between sets. These
are not separately serialized independent entropy headers for every image tile.

Reference decoder paths:

- [ProcessACGlobal](https://github.com/libjxl/libjxl/blob/main/lib/jxl/dec_frame.cc)
- [AC group selectors and context offsets](https://github.com/libjxl/libjxl/blob/main/lib/jxl/dec_group.cc)

## Implementation

`jpeg::tuning::Options::ac_histograms` requests at most 1..4 sets in this
experiment, default 1. Four is an experiment bound, not a JXL format limit.
`jpeg/ac_histogram_groups.rs` operates on the existing raw AC token streams:

1. Preserve the tokenizer's block/DC/channel conditioning. Pool nonzero-count
   prediction subcontexts separately from coefficient-density subcontexts.
2. Count HybridUint(4, 2, 0) symbols for each image group. Extra bits are not
   histogram-coded; the estimate does not mistake them for table gains.
3. Deterministically bisect dissimilar distributions using sparse Hellinger
   distance and up to six refinement passes. Admit a split using estimated
   conditional entropy savings less a model-overhead margin.
4. Assign dense IDs, offset original contexts, then perform the normal entropy
   clustering/model construction and stream writing. Coefficient order and
   values are unchanged.

The scratch histogram is at most 32 KiB, not a full raw-context histogram per
image group. Empty, identical, or estimated-unprofitable distributions can
collapse to fewer sets. `ac_histograms_used` reports the actual count.

The admission estimate is deliberately a heuristic: pooled contexts and model
overhead do not predict the exact final size. Each invocation encodes one
candidate only; there is no hidden baseline tournament or size fallback.

`ac_lz` independently selects `source`, `off`, `fast`, `optimal`, `fast-forced`
or `optimal-forced`. The fast/optimal parsers use the existing Lz77b3w3t and
Optc256 tools on AC integer tokens, with spatial distance multiplier zero.
Forced modes bypass the global estimated-gain gate, not individual match costs.
`ac_lz_enabled` confirms the wire model was enabled, rather than just requested.

## Actual wire costs

`jpeg::tuning::transcode_measured` returns encoded bytes and `SizeBreakdown`.
Counts come from the real writers, not a second encode or statistical estimates.
The 12 disjoint bit counts must sum exactly to eight times the container size.

On the existing 45 JPEG inputs, with the prior best explicit combination
(`--cooperative --symmetric-ma --cfl-peak`), the output is 27,372,627 bytes:

| Component | Percentage of complete output |
| --- | ---: |
| AC data | 91.119% |
| DC data | 8.082% |
| AC model and orders | 0.460% |
| Shared auxiliary MA/entropy model | 0.100% |
| AC metadata | 0.074% |
| JPEG reconstruction payload | 0.069% |
| Extracted Exif/XMP payloads | 0.042% |
| Other headers, quantization, padding and framing | 0.054% |

The shared auxiliary model is not attributed entirely to DC. AC data includes
per-group selectors; AC model includes the number of histogram sets. Header
cost includes ICC when present; metadata cost refers to extracted Exif/XMP.

## Measurement protocol

Release, Windows x86_64, e9, eight encoder threads unless noted. The 45-image
manifest contains 16 format/regression inputs, nine project JPEGs and 20 CC12m
originals, totaling 106,200,376 pixels and 33,406,801 original JPEG bytes.
Some fixtures share content; these results are not a population estimate.

Every paired control uses the same explicit cooperative + symmetric MA + CfL
combination. Thus gains from previous experiments cannot be attributed to AC
grouping. Native time includes JPEG parse and encode, excludes process startup,
disk I/O, dimension inspection and reconstruction. Trials alternate order.
Bpp is pixel-weighted: total JXL bytes * 8 / total original pixels.

Both native reconstruction and official djxl 0.12 must recover the complete
original JPEG bytes, including metadata. JXL byte identity is not a goal.
Temporary outputs are removed after verification. Binaries, input hashes and
the exact commands are preserved in the benchmark reports.

## LZ findings

On all 45 inputs, disabling AC LZ or using either gated fast/optimal parser
produces the same sizes as the control. Disabling LZ reduced measured time
19.319 -> 15.023 s; fast and optimal took 16.329 s and 16.287 s respectively,
against their paired controls of 18.818 s and 19.097 s. These are single-repeat
diagnostics, not confidence intervals. A gated parser may reject before full
matching, so this is not evidence that a full optimal parse is faster.

On seven diagnostic inputs, forcing the wire LZ model on produces:

| Parser | Control bytes | Trial bytes | Size change | Control / trial seconds |
| --- | ---: | ---: | ---: | ---: |
| Fast forced | 3,473,378 | 3,484,933 | +0.333% | 2.561 / 2.345 |
| Optimal forced | 3,473,378 | 3,476,438 | +0.088% | 2.426 / 4.089 |

Each worsened five of seven inputs. Neither is recommended as a default.
These are actual activated LZ runs, not merely flags rejected by admission.

## Histogram-group results

All 45-image controls are 27,372,627 bytes / 2.061960835 weighted bpp.
These runs use one repeat, so small timing differences include system noise.

| Requested sets | JXL bytes | Change vs control | Weighted bpp | Control / trial seconds | Trial MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 27,372,694 | +67 B (+0.000245%) | 2.061965882 | 18.959 / 19.309 | 5.500 |
| 3 | 27,373,480 | +853 B (+0.003116%) | 2.062025091 | 19.243 / 19.498 | 5.447 |
| 4 | 27,370,898 | -1,729 B (-0.006317%) | 2.061830591 | 18.915 / 19.486 | 5.450 |

- With two sets, 18 images actually use two; 27 collapse to one. Five improve,
  13 grow, 27 have unchanged size.
- With three sets, 12 use three, six use two and 27 use one. Five improve,
  13 grow, 27 have unchanged size.
- With four sets, 12 use four, six use two and 27 use one. Four improve,
  14 grow, 27 have unchanged size.

This is not a no-op experiment. At two sets the AC data saves 31,595.875 byte
equivalents, but AC model/orders grow by 31,673.750. At four sets the data saves
82,281.875, while model/orders grow by 80,554.750; alignment/header differences
account for the remainder. The resulting models describe the data better, but
their signaling costs consume nearly all the savings.

The strongest four-set improvement is `wallhaven-zpq1dg.jpg`, -11,582 B
(-0.2587%), followed by `wallhaven-yqxzqx.jpg`, -6,835 B (-0.0850%). More sets
are not monotonically better: `wallhaven-qrjmgl.jpg` improves by 620 B with two
sets but grows by 345 B with four.

An after-the-fact oracle choosing the smallest complete output among one, two,
three and four sets totals 27,353,434 B, saving 19,193 B (-0.070117%). This is
only the available gain among these tested candidates, not a deployable router
or an upper bound for all possible histogram assignments. No oracle selection
or four-encode tournament was added to production.

### Threads and combinations

On the seven diagnostic images, four sets increase total size by 3,306 B
(+0.09518%). Single-thread two-repeat time is 4.324 -> 4.503 s, throughput
4.343 -> 4.170 MPix/s. Eight-thread one-repeat time is 2.435 -> 2.663 s,
throughput 7.711 -> 7.051 MPix/s. These runs use different repeat counts and
were not simultaneous, so they do not establish exact thread-scaling ratios.

The same seven inputs also pass with four sets plus forced fast LZ, and with
four sets at e3. The combinations increase size by 14,492 B and 2,901 B against
their respective controls. They are compatibility checks, not recommended
policies. Native and official JPEG reconstruction both pass.

### Decision and validation

Keep histogram grouping optional. The four-set aggregate improvement is too
small to justify enabling this classifier unconditionally. Improving the
assignment objective or sharing tables more economically is a better next
experiment than blindly increasing the number of sets. Cheap pooled entropy
alone does not accurately price the final context map/table overhead.

- JPEG-related Rust tests: 389 passed, nine explicit external-oracle tests
  ignored; CLI option test passed; Python harness tests: 20 passed.
- This round includes 708 native + official byte-verified outputs: 340 LZ
  experiment outputs, 42 histogram-screen outputs, 270 full-matrix outputs,
  28 single-thread outputs, and 28 combination/e3 outputs.
- Real activated multi-set images and the forced LZ + multi-set combination
  were verified, not just fixtures that collapse back to a single set.
- No default policy was changed, and no claim of improvement over official
  cjxl is made: the paired control is our previously tested explicit policy.

## Reproduce

```powershell
cargo build --release -p rs-jxl-cli --example jpeg_entropy_lab
target/release/examples/jpeg_entropy_lab.exe image.jpg --effort 9 --threads 8 --cooperative --symmetric-ma --cfl-peak --ac-histograms 3
python -B tools/benchmark_jpeg_transcode.py --encoder candidate.exe --baseline candidate.exe --decoder djxl.exe --manifest cases.json --out results.json --effort 9 --threads 8 --repeats 3 --baseline-arg=--cooperative --baseline-arg=--symmetric-ma --baseline-arg=--cfl-peak --candidate-arg=--cooperative --candidate-arg=--symmetric-ma --candidate-arg=--cfl-peak --candidate-arg=--ac-histograms=3
```

Repeat `--baseline-arg` and `--candidate-arg` to describe each side explicitly.
Manifest format is `{"cases":[{"original":"path/to/original.jpg"}]}`.
Keep instrumentation off for timing. The `jpeg.ac.histogram_groups` profile
scope measures classification/context reassignment, excluding entropy fitting.

Local evidence: `out/jpeg-ac-20261117` under the sibling C++ workspace;
manifest: `out/jpeg-opt-20261116/cases.json`. No Linux runtime or wheel
acceptance is claimed for this experiment.
