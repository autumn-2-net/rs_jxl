# JPEG Reconstruction Optimization: 2026-10-24

## Scope

This iteration targets original JPEG byte reconstruction from JXL, the largest
same-stream decoder gap in [the baseline](PERFORMANCE_BASELINE_20261024.md).
It does not change JXL encoding, compressed size, pixel rendering, MA search or
lossy policy. All runtime code remains native Rust. Official `djxl` is used only
as an independent development reference.

Correctness means **restoring the original JPEG bytes**, including progressive
scan choices, stuffing, padding, restart markers and metadata. Compressed JXL
byte identity is not an optimization target.

## Measured Hotspot

Opt-in `RS_JXL_PROFILE=1` scopes now separate coefficient decoding, metadata,
writer-model construction, validation, scan writing and total reconstruction.
`RS_JXL_PROFILE=trace` shows individual scans. Scopes are inclusive; do not sum
nested totals. Normal benchmarks leave profiling disabled.

Before optimization, five repetitions with eight requested threads gave:

| Input | Total seconds | Scan writing seconds | Scan share |
| --- | ---: | ---: | ---: |
| wallhaven-yqxzqx, sequential, 5120x2880 | 2.7581 | 2.0987 | 76.1% |
| wallhaven-9oo8k1, progressive, 4096x2304 | 1.1801 | 0.9888 | 83.8% |

The complete reconstruction writer, including non-scan work, accounted for
76.7% and 84.1% respectively. Metadata and writer-model construction were each
below one millisecond for all five repetitions, so they were not useful targets.

## Changes

- Sequential JPEG blocks use a dedicated writer instead of initializing a generic
  progressive coding state and passing through progressive band processing.
- Contiguous AC range checks preserve pre-emission validation while allowing
  compiler vectorization. Trailing-zero scanning is needed only for explicit
  preserved ZRL hints, including the permitted final overshoot.
- An empty progressive EOB flush returns immediately. Nonempty refinement data
  and error-poisoning behavior are preserved.
- Entropy writes form up to four complete bytes at once, use a word-wide `0xff`
  detection fast path and skip allocation checks when no byte is emitted.
  Stuffed writes still reserve their exact size before mutating output/state.
- Progressive refinement fuses range validation with magnitude preparation.
  For legal low-bit positions 0..10 the original category constraint is exactly
  `abs(coefficient) < 2048`; the equivalence is tested exhaustively.

There is no new unsafe code, GPU dependency, codec dependency or public API.
JPEG scan writing remains serial; requested worker counts principally affect
JXL coefficient decoding. These changes do not claim pixel-decoder acceleration.

## Before And After

Windows x86_64, Ryzen 9 7945HX, Rust 1.97.1, normal Release build. The same three
official-0.12-produced streams were decoded by the retained Rust baseline, the
optimized Rust executable and official `djxl` 0.12. Processes ran serially in
rotating order: one excluded warmup, three measured rounds, ten decodes per
invocation. Entries are median process wall time divided by ten, including
amortized process startup/input IO and decoder allocation. No output file writing
or verification is timed. C++ reuses its runner within repetitions; Rust creates
a runner per API call. That cost remains included on both Rust checkpoints.

Eight requested threads:

| Input | bpp | Before ms | After ms | Before MPix/s | After MPix/s | Official MPix/s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wallhaven-yqxzqx | 4.3686 | 556.002 | 343.327 | 26.521 | 42.949 | 83.988 | 1.619x |
| wallhaven-9oo8k1 | 0.9593 | 244.411 | 198.529 | 38.612 | 47.536 | 82.267 | 1.231x |
| wallhaven-vpyekp | 0.3020 | 31.264 | 21.221 | 78.609 | 115.807 | 197.969 | 1.473x |

Aggregate throughput is total pixels divided by the sum of per-image median
seconds, **not** the arithmetic mean of image rates. These three images total
26.640384 MPix and 9,276,600 JXL bytes: 2.7857 bpp, unchanged by decoder work.

| Requested threads | Before seconds | After seconds | Before MPix/s | After MPix/s | Official MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.471067 | 1.198536 | 18.110 | 22.227 | 46.351 |
| 8 | 0.831676 | 0.563077 | 32.032 | 47.312 | 88.011 |

Eight-thread aggregate time drops 32.3%, or 1.477x throughput. All six individual
input/thread comparisons improve; this small diagnostic sample is not proof of
absence of regressions on arbitrary inputs. Rust still takes 1.86x official time
in the eight-thread aggregate. The optimized C++ decoder is not remeasured in
this iteration; its prior same-stream baseline was 86.575 MPix/s.

Final five-repetition stage profiles (separate from the timed comparison):

| Input | Total seconds | Scan writing seconds | Coefficient stage seconds |
| --- | ---: | ---: | ---: |
| wallhaven-yqxzqx | 1.6679 | 1.0580 | 0.5665 |
| wallhaven-9oo8k1 | 0.9688 | 0.7728 | 0.1819 |

Scan writing remains about 63% and 80% of total reconstruction respectively.
Further writer batching/EOB overhead work is still warranted; this is not a
claim that decoder optimization is finished. Metadata remains negligible.

Artifacts: `out/decoder-perf-20261024/comparison/{report.json,REPORT.md}`.
The retained optimized executable is `out/decoder-perf-20261024/rs-jxl-optimized.exe`,
SHA256 `0a4590c4f1bed48a9665dd5bb56c765b705c2bac558e125fbe8307ab4eea72ec`.
The baseline identity and all commands are in the generated report. Binary hashes
identify benchmark checkpoints only. These generated artifacts are not committed.

## Verification

- 304 JPEG-focused unit tests passed, including all 65,536 two-byte stuffing
  inputs, pending-bit and exact-output-limit cases, sequential/generic block
  differential tests, and every `i16` value at each of 11 refinement bit planes.
- Nine existing optional external-encoder oracle tests were not run by that
  unit command. This iteration benchmarks real official-produced streams
  separately, with original-byte checks before timing.
- Six benchmark-driver contract tests passed: rotating execution order,
  correctness failures, timeouts, changed executables/inputs and cleanup.
- 36 decoder integration tests passed after the final change: 29 general
  decoding/multiframe/metadata cases, four isolated hostile-input batches and
  three progressive DC-refinement cases. The hostile child entry point is
  normally ignored and was invoked by the four parent tests.
- All 18 real-input original-JPEG comparisons passed (3 inputs, 3 decoders,
  2 thread counts). All 72 benchmark batches passed, including warmups;
  each batch performs ten reconstructions. No timeout or failed input was skipped.
- `cargo clippy --locked --offline --workspace --all-targets -- -D warnings`,
  `cargo fmt --all -- --check` and `git diff --check` passed.

This is Windows-only targeted validation, not a full corpus rerun or new
cross-platform acceptance. No commit or push was made.

## Reproduction

Build with the repository's normal Release profile (thin LTO, one codegen unit):

```powershell
cargo build --release --locked --offline -p rs-jxl-cli
cargo test --locked --offline -p rs-jxl --lib jpeg::
cargo test --locked --offline -p rs-jxl --test decode --test jpeg_dc_refinement --test decode_hostile
python -B -m unittest discover -s tools -p test_compare_decoder_checkpoints.py
```

The comparison tool uses pre-encoded inputs only; it does not build or encode.
Supply repeated `--case JXL ORIGINAL` pairs, two fixed Rust executables and the
official decoder, then choose a new output directory:

```powershell
python -B tools/compare_decoder_checkpoints.py `
  --before out/codec-subset-20261024/rs-jxl-baseline.exe `
  --after target/release/rs-jxl.exe `
  --official-djxl ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe `
  --case out/decoder-perf-20261024/large.jxl ../jxl_python/test_img/wallhaven-yqxzqx.jpg `
  --case out/decoder-perf-20261024/medium.jxl ../jxl_python/test_img/wallhaven-9oo8k1.jpg `
  --case out/decoder-perf-20261024/small.jxl ../jxl_python/test_img/wallhaven-vpyekp.jpg `
  --threads 1 8 --warmups 1 --repeats 3 --decode-reps 10 `
  --out out/decoder-checkpoints-new
```

The fixtures were produced by official 0.12 `cjxl INPUT OUTPUT -e 9
--lossless_jpeg=1 --num_threads=8`. There is no JPEG-to-pixels fallback.
Each decoder/thread/input combination first reconstructs a temporary JPEG and
compares it to the original byte for byte. Temporary decoded files are removed.
Timed invocations disable output; the returned JPEG buffer is still constructed.
Reports retain raw commands, binary identities and per-round wall times.
