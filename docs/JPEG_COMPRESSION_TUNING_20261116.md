# JPEG coefficient compression experiments

These are explicit encoder experiments on the Rust JPEG-to-JXL coefficient
path, based on `f503b4d`. Ordinary JPEG encoding, Python bindings and source
search retain their defaults. Compression ratio is the primary objective;
encoding time is a cost constraint, not the optimization target for this round.

## Algorithms

- `--cfl-peak`: replace the midpoint of the first and last maximum-scoring CfL
  bins with an actual maximum, breaking ties toward zero then the lower bin.
  Keep the existing strict improvement gate. Exclude bin 255: its scored factor
  is +128, which cannot be represented by the emitted signed byte. Existing
  fitting samples, quantization ratios and integer inverse remain unchanged.
  This only affects the existing three-component 4:4:4 CfL path. A better zero
  count is not a proof of a smaller final file.
- `--symmetric-ma`: use the existing symmetric MA learner and half description
  cost on learned DC, control-field and quantization auxiliary trees. Explicit
  fixed/custom trees remain unchanged. This does NOT apply Distribution Offset:
  auxiliary streams do not have the pixel refinement stage used by ordinary e9.
- `--paired-luma`: merge adjacent luma DC buckets in the AC block-context map.
  Preserve the DC thresholds, qdc decisions and chroma distinctions; renumber
  referenced contexts densely. Both tokenization and the serialized context
  header use the same map. This tests sharing statistical evidence and reducing
  model cost, not changing quantized coefficients.
- `--cooperative`: lend the existing encoder pool to auxiliary MA split search.
  This is an existing execution option, measured separately before combining it
  with each compression policy. It is not a new compression algorithm.

Each compression experiment performs one encode. There is no baseline fallback,
two-container tournament, JPEG pixel re-encode, or changed decoder syntax.
The source-search AC cache rejects policies that change CfL or AC contexts;
the explicit tuning entry point does not use that cache.

## Measurement

Windows x86_64 Release, e9. The 45 inputs comprise 16 existing format/regression
fixtures, nine additional project JPEGs, and 20 CC12m originals sampled with seed
20261116. They total 106,200,376 pixels and 33,406,801 original JPEG bytes.
There are 12 three-component 4:4:4 inputs; other inputs cover 4:2:0, 4:2:2,
4:4:0 and grayscale, progressive scans, restart markers and metadata.
Several format fixtures share content, so this is not a population estimate.

Paired baseline/candidate runs alternate order, with two repeats per image.
Native time includes JPEG parse and encode, excluding startup, disk I/O,
dimension inspection and reconstruction. Aggregate time sums per-image medians.
Weighted bpp is complete JXL bytes times eight divided by all input pixels.
The JSON also records arithmetic mean per-image bpp; the 1x1 metadata fixture
makes that mean unsuitable for assessing image compression in this matrix.

Every trial reconstructs the complete original JPEG with the native decoder and
official `djxl` 0.12, comparing all bytes, including metadata. JXL output identity
is not a correctness requirement. Original inputs are never modified; temporary
encoded/reconstructed files are removed after verification.

## Results and decision

Each row below has its own paired ordinary baseline. The baseline output is
27,397,679 bytes / 2.063848 weighted bpp in every 45-image run. The three
compression policies include cooperative execution; do not attribute their
entire time reduction to the new compression decision.

| Policy | JXL bytes | Size change | Ordinary / trial seconds | Smaller / same / larger |
| --- | ---: | ---: | ---: | ---: |
| Cooperative only | 27,397,679 | 0% | 44.850 / 33.614 | 0 / 45 / 0 |
| Symmetric MA + half description | 27,374,429 | -0.084861% | 44.677 / 18.722 | 42 / 1 / 2 |
| Actual CfL maximum | 27,396,124 | -0.005676% | 44.134 / 32.521 | 9 / 36 / 0 |
| Paired luma AC contexts | 27,414,636 | +0.061892% | 45.578 / 33.235 | 31 / 0 / 14 |
| MA + CfL combination, one repeat | 27,372,627 | -0.091438% | 44.043 / 17.899 | 43 / 1 / 1 |

The combination saves 25,052 bytes, with weighted bpp **2.061961** and measured
throughput **5.933 MPix/s**, versus 2.411 MPix/s in its paired ordinary run.
Its one regression is `video-001.q50.440.jpeg`, +4 bytes. The 20 CC12m inputs
all improve with MA, totaling -0.178704%; none of these sampled CC12m files is
CfL eligible. CfL alone saves 1,555 bytes across nine of twelve eligible files.

Paired luma contexts are **not recommended as a blanket policy**. They save
1,418 bytes in the CC12m subset, but increase `wallhaven-yqxzqx.jpg` by 10,530
bytes and `wallhaven-zpq1dg.jpg` by 3,261 bytes. More images improve than regress,
yet the total becomes larger. This is a candidate for a future context-cost
admission study, not a reason to enable unconditional merging.

Keep all new policies explicit for now. The strongest result is a small size
improvement accompanied by a substantial MA time reduction, not a breakthrough
in JPEG compression ratio. The useful next compression target is AC context
allocation/modeling; repeating fixed context merging or expanding HybridUint
search without measuring its gain is not justified by this round.

### Single-thread follow-up

On the seven diagnostic images, MA + CfL saves 4,673 bytes (-0.134357%).
Two-repeat single-thread time is 6.637 -> 4.110 seconds (-38.08%). The same seven
images in the one-repeat eight-thread combination run take 6.396 -> 2.331
seconds. Thus the policy improves serial execution too; its speed improvement
is not only extra parallelism. These runs were not simultaneous and do not
establish confidence intervals or whole-corpus thread scaling.

### Correctness

- All **850** measured outputs passed native and official `djxl` original-JPEG
  byte equality: 720 independent-policy outputs, 90 combination outputs, 28
  single-thread follow-up outputs, and 12 e3/e10 compatibility outputs.
- The e3/e10 checks use three small 4:4:4/progressive/metadata fixtures with all
  flags enabled. They are compatibility checks, not effort-level benchmarks.
- The initial seven-image t1/t8 profiling adds 14 successful reconstructions;
  its instrumented times are not included in the table above.
- JPEG unit tests: 376 passed, 9 explicit external-oracle tests ignored. These
  include all-policy resource-limit errors and serial/parallel reconstruction.
- The benchmark helper suite passes 41 tests; seven CLI conflicting/invalid
  option checks pass. Changed Rust files pass `rustfmt --check`.
- No Linux runtime or wheel
  acceptance is claimed in this round; the public ordinary defaults are unchanged.

## Reproduce

```powershell
cargo build --release -p rs-jxl-cli --example jpeg_entropy_lab
target/release/examples/jpeg_entropy_lab.exe image.jpg --effort 9 --threads 8 --cooperative --symmetric-ma
python -B tools/benchmark_jpeg_transcode.py --encoder candidate.exe --baseline baseline.exe --decoder djxl.exe --manifest cases.json --out results.json --effort 9 --threads 8 --repeats 2 --candidate-arg=--cooperative --candidate-arg=--symmetric-ma
```

Manifest format: `{"cases":[{"original":"path/to/original.jpg"}]}`. Relative
paths resolve against the manifest. Repeat `--candidate-arg` for combinations;
omit `--baseline` for diagnostic runs. `--profile` enables stage instrumentation
and should not be used for final timing comparisons. Reports refuse overwrite.

`jpeg.stage.*` scopes are disjoint outer stages. Nested `jpeg.ac.*`,
`jpeg.auxiliary.ma` and `jpeg.assembly.*` scopes are inclusive diagnostics;
do not sum them with their parents or interpret worker totals as wall time.

Local evidence is under `out/jpeg-opt-20261116` in the sibling C++ workspace;
input/executable hashes and exact commands are retained in each JSON report.
