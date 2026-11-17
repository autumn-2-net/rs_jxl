# Standard JXL Decoder Reader Dispatch, 2026-11-17

This change accelerates selected Modular entropy-reader paths. It does not
establish a general speedup on ordinary images. The six real-image controls
remain effectively unchanged in the final single-core comparison. Encoding,
JXL syntax and the public decoder options are unchanged.

## Implementation

The maintained `vendor/jxl-rs/jxl` decoder already proves when every reachable
leaf of a pruned tree returns the same symbol without HybridUint extra bits.
The new dispatch reuses its constant reader for weighted-only trees and a
single Gradient leaf. LZ77 is excluded. A single Gradient leaf also uses the
existing `(4,2,0)` reader when the histogram configs permit it.

The constant path additionally checks the current entropy state once per
channel. A prefix singleton consumes no bits; an ANS singleton leaves its
state unchanged only when that state is at least 65536. The ordinary reader
must still handle lower initial states. For example, state `0x13` followed by
16 zero bits reaches final state `0x130000` after a singleton read. The new
guard preserves that operation, including truncated-input errors. The existing
Zero and FlatTree shortcuts are outside this change.

Prediction arithmetic, weighted error-history updates and final stream checks
are retained. No unsafe code, ISA requirement or decoder-environment switch
was added. The modified source files and rationale are recorded in
`vendor/jxl-rs/PROVENANCE.md`.

## Measurement

- Baseline standard codec: `d003ae07e97c84d17ba616c9e93c7e1bcd15c1f7`.
- Windows x86_64, Ryzen 9 7945HX, rustc 1.97.1, release thin LTO, one codegen unit.
- Public `decode::decode`, default Native output, one decoder thread. The timer
  includes decoder/context construction and output allocation, and excludes
  file IO, hashing and sample verification. There is no worker pool at one thread.
- Both benchmark programs use the identical harness source, SHA256
  `2dc79edf50044d17aaa6ed4f20243cb23dc3c1aeb63773d72c91a1f319538494`.
- Four sequential processes in before/after/after/before order. Every process
  has one warmup and seven timed decodes per case: 14 measurements per build.
  The final run pins each child process to logical CPU 2, affinity mask `4`.
- Twenty frozen inputs: 14 route probes/control images and six existing
  official libjxl 0.12 e9 streams, totaling 7,170,418 pixels. Every decode is
  checked against original integer samples, including invisible RGB and the
  authoritative alpha plane. These are development samples, not new holdouts.

Synthetic inputs are encoded once by the baseline program. The official JXL
files are copied unchanged, and expected pixels come from their original PNGs
without color conversion or alpha compositing. The existing manifest's PNG
and JXL SHA256 values were checked before use. Timing never re-encodes inputs.

The real controls are indices 0, 3, 5, 6, 7 and 8 of
`out/decoder-perf-20261024/pixel-inputs/manifest.json`: Kodak photo, webpage,
transparent icon, gray photo, 64x64 RGB16 photo and PDF/text screenshot.

## Results

Times are medians in milliseconds. Throughput counts spatial pixels, not channels.

| Sample | Before ms | After ms | Speedup | After MPix/s |
| --- | ---: | ---: | ---: | ---: |
| WP8 ANS singleton | 2.061 | 1.140 | 1.808x | 229.97 |
| WP8 non-unit leaf control | 1.914 | 1.911 | 1.001x | 137.18 |
| Gradient8 non-unit leaf control | 1.259 | 1.312 | 0.960x | 199.77 |
| Gradient8 varying, config 420 | 1.877 | 1.479 | 1.269x | 177.24 |
| Gradient8 varying, alternate config | 1.837 | 1.848 | 0.994x | 141.83 |
| WP16 prefix singleton | 1.721 | 1.090 | 1.580x | 240.57 |
| Gradient31 ANS singleton | 4.624 | 3.392 | 1.363x | 77.29 |
| WP8 alternate-config singleton | 2.149 | 1.054 | 2.039x | 248.69 |
| WP16 LZ input | 5.177 | 4.989 | 1.038x | 52.54 |
| Gradient8 prefix singleton | 1.040 | 0.579 | 1.798x | 452.95 |
| Ordinary RGBA8 with hidden RGB | 2.731 | 2.912 | 0.938x | 22.51 |
| Ordinary RGBA16 with hidden RGB | 9.090 | 9.036 | 1.006x | 7.25 |
| Ordinary RGBA31 with hidden RGB | 7.254 | 7.497 | 0.968x | 8.74 |
| Ordinary Gray16 | 1.216 | 1.227 | 0.991x | 53.41 |
| Kodak | 61.437 | 61.899 | 0.993x | 6.35 |
| Webpage | 56.292 | 56.816 | 0.991x | 33.82 |
| Transparent icon | 32.266 | 32.651 | 0.988x | 8.03 |
| Gray photo | 10.828 | 10.607 | 1.021x | 23.57 |
| Small RGB16 photo | 0.722 | 0.705 | 1.024x | 5.81 |
| PDF/text | 176.848 | 175.762 | 1.006x | 8.28 |

The six real-image medians sum to **338.392 -> 338.440 ms** for 4,286,834
pixels, or **12.67 MPix/s** after the change. This is effectively unchanged.
The constant-reader and single-Gradient probes improve substantially, but the
ordinary synthetic RGBA8 and RGBA31 controls take 6.6% and 3.3% longer. This is
not a blanket no-regression result. The real-image sample is small, and the
64x64 photo is dominated by fixed overhead.

Earlier unpinned runs were unstable: a version with an additional correctness
test linked into the after executable measured 2-3% more aggregate real-image
time, while an identical-harness unpinned run measured 4.5% higher throughput.
The extra test was moved to `decoder_hybrid_uint`, restoring byte-identical
benchmark source. The pinned run above is the primary result; the earlier
reports are retained, not silently averaged into it. Code layout and scheduling
are possible contributors, not established explanations. No broad-image gain
is claimed from these variations.

## Validation and Local Evidence

The route test checks 14 original 8/16/31-bit images, including non-unit leaves,
alternate HybridUint configs, LZ input, both entropy coders and invisible RGB.
Additional tests check low ANS initial states, the 65535/65536 boundary,
truncated normalization reads, and a singleton symbol that still carries two
extra bits. The benchmark verifies every timed and warmup output.

All 69 selected regression tests passed across ten integration targets:
`decode_hostile` (4), `decode_jpeg` (8), `decode_lossless_edges` (3),
`decode_streaming` (17), `decode_wide_integer` (7), `decoder_hybrid_uint` (19),
`decoder_parallel_pixels` (5), `decoder_reader_dispatch` (1),
`decoder_weighted_flat` (1, containing 180 combinations) and `gradient_interop`
(4). Three normal-run ignored entries are the hostile-test child entry point,
the explicit benchmark, and the optional external JPEG corpus. The benchmark
was run explicitly; no external JPEG-corpus acceptance is claimed here.

Rust 1.89 `cargo check` passed for the library and both changed test targets.
Changed Rust files pass rustfmt, and `git diff --check` passes. Strict Clippy
on rustc 1.97 did not pass: it stopped on 27 existing library lints in unchanged
encoder/scheduler files (including `duplicate_mod`, `collapsible_if` and
`manual_div_ceil`) before checking the integration targets. This change does
not claim a clean repository-wide Clippy result or alter those unrelated files.

```powershell
cargo test --offline -p rs-jxl --test decoder_reader_dispatch --test decoder_hybrid_uint --test decoder_weighted_flat --test decode_wide_integer --test decode_lossless_edges --test gradient_interop --test decoder_parallel_pixels --test decode_streaming --test decode_jpeg --test decode_hostile -- --test-threads=4
```

Local artifacts in `out/decoder-reader-20261117/`:

- `before.exe`, `after.exe`, `baseline-build.json`, `benchmark-harness.rs`.
- `real.tsv`, `fixtures/manifest.json`, `corpus-sha256.json`.
- `pinned-summary.json` and `pinned-{before,after}-*.json`: primary results.
- `full-identical-*.json`: the unpinned identical-harness comparison.
- `summary.json`, `repeat-*.json`: earlier measurements with the extra test.

The fixtures and executables are local development artifacts, not repository
data. `SAVE=1` requires a fresh directory; normal mode only reads frozen files.

```powershell
cargo test --offline --release -p rs-jxl --test decoder_reader_dispatch --no-run
# Copy each revision's produced test executable to before.exe / after.exe.
$env:RS_JXL_READER_BENCH_DIR = 'out/decoder-reader-rerun/fixtures'
$env:RS_JXL_READER_BENCH_REAL_TSV = 'out/decoder-reader-20261117/real.tsv'
$env:SAVE = '1'
& $before --ignored --exact native_reader_dispatch_speed --nocapture
Remove-Item Env:SAVE
$env:RS_JXL_READER_BENCH_RECEIPT = 'out/decoder-reader-rerun/before.json'
& $before --ignored --exact native_reader_dispatch_speed --nocapture
# Run after against the same directory with a distinct receipt path.
```

For the recorded comparison, `run_pair.py pinned` sets child affinity mask 4
when `RS_JXL_BENCH_AFFINITY=4` and captures the four sequential runs. Its summary
also records SHA256 values for both executables. Parallel decoding, other CPUs,
Linux and full-corpus throughput are outside this focused measurement.
