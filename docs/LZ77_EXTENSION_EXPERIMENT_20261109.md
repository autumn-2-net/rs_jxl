# Exact LZ77 Extension Experiment

## Scope

This is an opt-in native Rust encoder experiment, behind the non-default
`experimental-lz77` Cargo feature. Production builds omit the extra matcher
fields and hot-loop branches. Candidate budgets, distance costs, parser
decisions, and JXL wire semantics are unchanged.
It does not link misa77, change the decoder, or introduce a new LZ77 format.

Reference: [misa77](https://github.com/welcome-to-the-sunny-side/misa77),
commit `882cfdad879f473afebe828ff19832bf9e4b1754`, specifically the equality-mask
first-mismatch idiom and independent candidate probes. The original MIT source
remains a research reference, not a runtime dependency. The kernels here are
original Rust `std::arch` implementations.

## Experiments

1. Scalar, AVX2, and AVX-512 exact longest-common-prefix extension over `u32`
   tokens. SIMD returns the first differing token from the comparison mask,
   avoiding a second scalar scan of a failing vector.
2. Four-way independent extension in fast-ring/spatial matching, with a shared
   current vector. Prefix and current-best endpoint rejection remain in place.
   Results are reduced in candidate order. Speculation may compare additional
   candidates but never drops a candidate or changes the tie rule.

Chain traversal remains sequential because zero-run chain selection depends on
the preceding match result. Its individual extensions use the selected kernel.
Immutable chain batches, bucket/index/suffix routes are intentionally excluded
from this experiment and rejected when combined with the switch.

The implementation accepts distance-one overlap and arbitrary slice lengths.
Loads stay within the input slices and scalar tails need no padding. No misa77
minimum-distance, maximum-match-length, or decompressor copy assumptions apply.
ISA support is checked at matcher construction; a forced unsupported ISA errors
instead of silently using another backend. Portable scalar remains available.
The packed `u16` kernel is currently a tested primitive only, not an active path.

## Controls

Development example, not the installed CLI or Python API:

```text
cargo build --release -p rs-jxl-cli --example encoder_lab --features rs-jxl/experimental-lz77
target/release/examples/encoder_lab input.png --effort 9 --threads 8 --match-extension avx2
target/release/examples/encoder_lab input.png --effort 9 --threads 8 --match-extension avx2 --batch-matches
```

An ordinary build rejects these experiment switches explicitly. With the Cargo
feature enabled, omitting both switches retains legacy matching, but does not
remove its runtime dispatch overhead. Supported kernel names are
`scalar`, `avx2`, and `avx512`. The batch flag requires an explicit kernel. The
switches cannot be combined with outer search, scan-only, another matcher, or
immutable `--chain-batch`. Candidates disabling LZ77 are rejected explicitly.

Native configuration is `tools::lz77::SourceConfig::{match_extension,
batch_fast_matches, match_extension_receipt}`. Receipts aggregate local counts
once on matcher destruction, not atomically inside every comparison. Disabled
receipts do not accumulate hot-loop counts. The lab requests receipts; its
timings therefore include diagnostic overhead, which was not separately timed.

JSON reports requested `mode`/`batch` separately from actual `active` and
`batch_active`, plus matcher, extension, and batch-group counts. Small streams
or careful seeds can be entirely chain-based: `batch=true, batch_active=false`
is not evidence for batching performance.

## Verification And Measurement

Kernel tests cover mismatch lanes/bits, alignments, exact tails, long matches,
all-equal data, immutable overlapping slices, and divergent four-way results.
Matcher tests cover exact length/tie rules and token replay. Full image checks
compare original samples, including hidden RGB, after official `djxl` decoding.
Compressed-byte equality is not an acceptance requirement.

`tools/benchmark_lz77_extension.py` runs fixed-candidate e9 experiments, rotating
variant order and comparing per-image medians. Native timings exclude startup,
PNG input/output, and oracle decoding. Reports retain bpp, MPix/s, byte sizes,
timings, commands, binary provenance, and activity counters. No outputs are
kept beyond the temporary verification directory. Timeouts are harness guards,
not encoder selection rules. This does not benchmark outer-scheduler throughput.

```text
python -B tools/benchmark_lz77_extension.py --encoder PATH_TO_ENCODER_LAB --decoder PATH_TO_OFFICIAL_DJXL --case input.png --effort 9 --threads 8 --repeats 3 --variants baseline scalar avx2 avx512 avx2-batch avx512-batch --out out/lz-extension/results.json
```

An optional `--frozen OLD_ENCODER_LAB --variants baseline frozen avx2` adds a
pre-change executable control. All other variants use the same new executable;
`baseline` is the new executable with the experimental switches omitted.

## Results

**Not promoted.** On this small sample the prototypes did not beat the frozen
pre-change executable consistently. Keeping only a runtime opt-out also showed
overhead, which is why the final source puts the hot-path hooks behind a
compile-time feature.

Windows x86_64, Ryzen 9 7945HX, 8 encoder threads, e9, 6 original PNGs totaling
15,274,438 pixels. The 16-bit displacement texture and gray-alpha cursor are
included. No compilation or other benchmark was running during measurements.
Three repetitions, rotated order, no discarded timed trials; table entries are
per-image medians. Startup, input decoding, output I/O, and oracle checking are
outside the native timer. Timing variability was substantial; this is a small
screen, not a statistically conclusive corpus result.

Raw reports (ignored local artifacts):

- `out/lz-extension-20261109/screen.json`: initial 42 encode/decode checks.
- `out/lz-extension-20261109/paired-r2.json`: 108 checks after bypassing empty
  cached suffixes, before adding the compile-time isolation.
- Every check passed official 0.12 `djxl` pixel comparison. All tested variants
  happened to have the same output sizes; output hashes were not compared.
- Frozen original executable: `out/ma-seed-20261109/encoder-lab.exe`.
- Experimental measured executable: `out/lz-extension-20261109/encoder-lab-r2.exe`.
  Hashes, full commands, sample ranges, and activity receipts are in the JSON.

| Variant | Sum of median seconds | Aggregate MPix/s | Time vs frozen original |
| --- | ---: | ---: | ---: |
| Frozen original | 9.1044 | 1.6777 | baseline |
| Experiment build, switches off | 9.7231 | 1.5709 | +6.80% |
| Scalar extension | 9.3212 | 1.6387 | +2.38% |
| AVX2 extension | 9.6059 | 1.5901 | +5.51% |
| AVX-512 extension | 9.4758 | 1.6119 | +4.08% |
| AVX2 four-way | 9.5141 | 1.6054 | +4.50% |

Aggregate size for every variant: 17,281,764 bytes, **9.051339 bpp**. Throughput
is total pixels divided by the sum of per-image medians, not the mean of rates.

| Image | JXL bytes | bpp | Frozen s | AVX2 s | AVX-512 s | AVX2 batch s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PDF `9bde6a...` | 412,541 | 2.26651 | 1.59710 | 1.84060 | 1.65121 | 1.65038 |
| VS Code `3dd595...` | 126,449 | 0.47325 | 0.50101 | 0.50960 | 0.53167 | 0.52442 |
| `wallhaven-49yekn` | 707,442 | 2.72933 | 1.00899 | 1.02249 | 1.03636 | 0.97844 |
| `wallhaven-mlzoy1` | 14,179,178 | 13.67590 | 5.55710 | 5.78785 | 5.77721 | 5.86611 |
| 16-bit `rough_linen_Displacement_1k` | 1,841,476 | 14.02196 | 0.28020 | 0.28454 | 0.32311 | 0.33694 |
| `cursor_PNG65` | 14,678 | 0.44794 | 0.16001 | 0.16077 | 0.15630 | 0.15785 |

The initially suspected empty-extension overhead was not the main cause:
skipping empty suffixes barely changed the counters on these images. On the 4K
image, batching made 77,004,673 extension calls versus 61,808,764 without batching
(+24.59%). There were 12,579,881 actual four-way batches. On the 16-bit texture,
there were no eligible four-way batches at all. Its batch-labelled timing must
not be interpreted as evidence for four-way SIMD performance.

Misa77's bounded byte matches and this parser's token histories are different
workloads. These results reject this particular integration as a default speed
optimization, not SIMD or independent-candidate comparison in general. Any next
attempt should first measure real extension lengths and dispatch costs, rather
than assume a wider vector automatically wins.

The experimental scalar/SIMD variants all include the same receipt mechanism;
SIMD did not beat scalar in aggregate either. However, comparison with frozen
production includes extra counters and dispatch infrastructure. A counter-free
timing path with a separate instrumented validation pass is still needed before
claiming the kernels themselves are slower or choosing any production backend.

## Final Isolation Checks

- Experimental feature enabled: 261 focused LZ77 tests passed.
- Feature disabled: 23 extension-filtered tests passed, including explicit
  rejection of unavailable experiments. Normal Release build passed; a real
  `--match-extension avx2` invocation failed with the expected Cargo-feature error.
- The lab CLI's 24 parser tests and the benchmark harness's 3 tests passed.
- Formatting and `git diff --check` passed. Strict Clippy reports four existing
  `needless_borrow` errors in untouched `modular/fast_context.rs` at lines
  500, 503, 506, and 514. This experiment did not edit or suppress them.
- No Linux execution, CI run, commit, or push was performed.

`out/lz-extension-20261109/default-control.json` checks the final **feature-off**
Release executable against the frozen original: 6 images x 2 builds x 2 repeats,
24 more successful official pixel checks. Total successful image checks across
the three reports: **174**. Sizes and aggregate bpp remain as above.

| Final ordinary-build control | Sum of median seconds | Aggregate MPix/s |
| --- | ---: | ---: |
| Frozen original in this paired run | 10.1873 | 1.4994 |
| Final feature-off build | 9.9229 | 1.5393 |

The approximately 2.6% time difference in this short check is **not** claimed as
a new optimization. The important structural change is that normal builds no
longer contain the experimental matcher fields or dispatch branches. Timing
noise and binary layout still prevent claiming exact performance equivalence.
