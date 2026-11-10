# JPEG AC head/tail experiment

Status: opt-in experiment, not enabled by the ordinary JPEG API, CLI, Python
binding or source-search scheduler. Tested on Windows x86_64, Release, 2026-11-10.
The small measured saving does not justify enabling it by default.

## Scope and compatibility

Motivation: [DEA-J: A Context-Based Entropy Coding Scheme for JPEG Compression](https://www.mdpi.com/2079-9292/15/19/4578).
The paper uses previous-symbol-conditioned JPEG Huffman tables with separate
frequent-symbol and escape/tail treatment. That coder is not a drop-in JXL coder;
its reported savings against JPEG are not expected gains against JXL.

This experiment only borrows the head/tail idea through **standard HybridUint
configurations**. The head consists of small unsigned values, not an arbitrary
set of frequent symbols. It is not a DEA-J implementation or a new decoder model.

- Preserve quantized coefficients, coefficient order, CfL, AC context map,
  histogram clustering, LZ parse and group boundaries.
- Leave DC/auxiliary Modular coding and JPEG reconstruction metadata unchanged.
- Expand the AC HybridUint pool from the source's 28 candidates to 125: all 121
  divisions for the existing 8-bit ANS alphabet plus four retained source
  direct-only configurations. Alphabet reduction canonicalizes direct-only
  configurations using the existing rules.
- Count exact values for the expanded pool. The source's compressed counting
  representation only preserves the mappings in its original candidate set.
- Rank configurations using normalized histogram/header estimates plus extra
  bits and configuration costs. This estimate is not the acceptance criterion.
- Reuse prepared coefficients, auxiliary models and tokens; serialize the
  ordinary and candidate complete JXL containers. Accept only strictly smaller
  output, including all headers, alignment, TOC, metadata and reconstruction data.
  Equal or larger output keeps the ordinary result.
- Skip fixed, prefix and incremental entropy policies. Do not enable the
  experiment inside source-search AC payload caches.

No previous-symbol selector, Escape code, new ANS format or decoder change is
introduced. Allocation/internal errors are not silently converted to a fallback.
An optional candidate that exceeds the explicit output budget can be declined
when the ordinary container fits that budget.

## Results

Baseline: the ordinary Rust JPEG transcode path in the same executable, based on
`ec4f636`. This is not a fresh comparison with official libjxl.

The fixed matrix contains 16 format/regression files: baseline and progressive
JPEG, gray, RGB, 4:2:0/4:2:2/4:4:0/4:4:4, non-interleaved scans, restart markers,
Exif/XMP/ICC and two project images. Several fixtures share image content; this
is **not a representative dataset sample**.

Effort 9, 8 threads, 3 repeats per variant with alternating run order. Timing
covers native parsing and encoding, excluding process startup, file I/O,
dimension inspection and reconstruction. It includes per-call worker setup and
the candidate search/write overhead. Aggregate seconds sum per-image medians;
bpp is total complete-container bits divided by total pixels.

| Metric | Ordinary | Head/tail trial |
| --- | ---: | ---: |
| JXL bytes | 2,584,444 | 2,584,271 |
| Weighted bpp | 1.39965042 | 1.39955673 |
| JXL / original JPEG | 76.00868% | 76.00359% |
| Native seconds | 5.218410 | 5.407167 |
| MPix/s | 2.83074 | 2.73192 |

Saving: **173 bytes / 0.006694%**. Measured time increase: **3.617%**.
Thirteen files became smaller, three stayed the same, none became larger.
Timing is noisy at these small differences; no speed improvement is claimed.

| File | Ordinary bytes | Trial bytes | Saved bytes | Ordinary s | Trial s |
| --- | ---: | ---: | ---: | ---: | ---: |
| video-001.q50.420.jpeg | 3,379 | 3,378 | 1 | 0.009500 | 0.011607 |
| video-001.q50.420.progressive.jpeg | 3,207 | 3,206 | 1 | 0.009958 | 0.011835 |
| video-001.q50.444.jpeg | 3,945 | 3,943 | 2 | 0.011089 | 0.013907 |
| video-001.q50.422.jpeg | 3,572 | 3,572 | 0 | 0.009043 | 0.011469 |
| video-001.q50.440.jpeg | 3,567 | 3,566 | 1 | 0.009692 | 0.011801 |
| video-005.gray.q50.jpeg | 2,893 | 2,892 | 1 | 0.007966 | 0.009418 |
| video-005.gray.q50.progressive.jpeg | 2,826 | 2,825 | 1 | 0.007568 | 0.009981 |
| video-001.restart2.jpeg | 4,708 | 4,705 | 3 | 0.010935 | 0.013436 |
| 1x1_exif_xmp.jpg | 4,113 | 4,113 | 0 | 0.003383 | 0.003431 |
| sideways_bench.jpg | 11,796 | 11,796 | 0 | 0.034137 | 0.034336 |
| bicycles_restarts.jpg | 76,008 | 76,006 | 2 | 0.162071 | 0.171224 |
| flower.png.im_q85_rgb.jpg | 937,941 | 937,896 | 45 | 2.559631 | 2.665445 |
| flower_small.q85_420_non_interleaved.jpg | 43,285 | 43,281 | 4 | 0.120761 | 0.121837 |
| flower.png.im_q85_420_R13B.jpg | 455,965 | 455,920 | 45 | 1.187061 | 1.183820 |
| wallhaven-vpyekp.jpg | 92,752 | 92,742 | 10 | 0.201253 | 0.210902 |
| 145702319_p0.jpg | 934,487 | 934,430 | 57 | 0.874362 | 0.922717 |

The 1x1 fixture is metadata dominated; its per-file bpp is not an image
compression indicator. Per-file bpp, MPix/s, receipts, input hashes and executable
hashes are retained in the JSON report, not rounded for comparison.

## Correctness and verification

- 96 outputs in the e9 matrix: native reconstruction and official `djxl` 0.12
  both restored the **complete original JPEG bytes**, including metadata.
- Additional e3 and e10 single-thread checks on the small non-interleaved flower,
  `wallhaven-vpyekp.jpg` and the 1x1 metadata fixture: 12 more outputs passed both
  reconstruction checks. These single-repeat checks are not timing benchmarks.
- e3 retained its fixed source policy, with no expanded search; e10 exercised
  the candidate/fallback path.
- `cargo test -p rs-jxl --lib entropy::`: 165 passed.
- `cargo test -p rs-jxl --lib jpeg::`: 361 passed, 9 external-oracle tests ignored.
  Some tests overlap the entropy filter; these are not disjoint counts.
- Example argument test: 1 passed. Benchmark harness: 19 passed; reused harness
  helper tests: 20 passed. Formatting and diff checks passed.

No new Linux runtime test or full workspace/wheel acceptance is claimed.
Compressed JXL byte equality is not a correctness objective.

## Reproduce

```powershell
cargo build --release -p rs-jxl-cli --example jpeg_entropy_lab
.\target\release\examples\jpeg_entropy_lab.exe image.jpg --effort 9 --threads 8
.\target\release\examples\jpeg_entropy_lab.exe image.jpg --effort 9 --threads 8 --head-tail
python -B tools/benchmark_jpeg_head_tail.py --encoder target/release/examples/jpeg_entropy_lab.exe --decoder path/to/djxl.exe --case image.jpg --out out/jpeg-head-tail/result.json --effort 9 --threads 8 --repeats 3
```

Repeat `--case` for more inputs, or use `--manifest` with
`{"cases":[{"original":"relative/to/manifest/image.jpg"}]}`. Reports refuse
overwrite. Encoded/reconstructed images are temporary and removed; original
inputs are not modified. The harness timeout is a process guard, not a search
budget or production fallback.

Local evidence, ignored by Git:

- `out/jpeg-head-tail-20261110/cases.json`
- `out/jpeg-head-tail-20261110/results-e9-t8.json`
- `out/jpeg-head-tail-20261110/check-e3-t1.json`
- `out/jpeg-head-tail-20261110/check-e10-t1.json`

The explicit Rust entry is `jpeg::head_tail::transcode(input, options, receipt)`.
Create a fresh `Arc<jpeg::head_tail::Receipt>` per call. A snapshot reports the
candidate count, changed histograms, complete-container byte counts, selection
and search/write time. With an explicit candidate output budget failure,
`candidate` can be true while `candidate_bytes` is `None` and `selected` is false.
Ordinary `jpeg::transcode` never enables this experiment.

## Conclusion

The legal representation space can produce marginally smaller streams, but the
measured gain is too small for the added work. Retain this as a default-off
experiment. The result neither reproduces nor disproves DEA-J's different coder;
it only evaluates this limited, standard-JXL-compatible adaptation.
