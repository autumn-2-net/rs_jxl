# JPEG Reconstruction Follow-up, 2026-11-08

## Scope

Baseline: `352b129` with the repository's normal Release profile (thin LTO,
one codegen unit), Windows x86_64, Ryzen 9 7945HX. This change optimizes
**JXL-to-original-JPEG reconstruction**, not JPEG-to-pixels, pixel decoding,
encoding, or parameter selection. The existing encoder and compressed sizes
are unchanged. No new dependency or ABI change is required.

## Profile and Retained Change

A separate `decoder-profile` build identified the progressive initial-AC
writer as useful work to remove. On `wallhaven-9oo8k1.jpg`, one-thread profile
averages were 174.8 ms for reconstruction, including 86.8 ms for coefficient
decoding and 85.9 ms for JPEG rewriting. Nested initial-AC and refinement
scopes measured 32.2 and 39.3 ms respectively. Scope times overlap; they are
diagnostics, not additive independent benchmarks.

The initial-AC writer previously validated the selected spectral band, then
walked it again to emit nonzero coefficients. It now keeps a nonzero bit mask
from validation and visits only set bits during batched emission. Zero-run
lengths follow from gaps between those positions; the mask also gives the
trailing run length.

- Wide bands use the existing native Rust AVX2 classification machinery,
  specialized to omit refinement-only predicates.
- Bands shorter than 16 AC positions use the scalar implementation to avoid
  preparing all 64 zigzag coefficients. Other architectures keep this scalar
  fallback as well; no new ARM kernel was added.
- Original precision checks, signed point transforms, final ZRL overshoot,
  pending EOB/correction runs, and output limits are retained.
- Failed speculative batches still use the original scalar emitter, preserving
  missing-Huffman errors, partial writer state, and failed-state handling.
- AVX2 loads remain confined to a fully initialized local 64-element array.

Implementation: `crates/rs-jxl/src/jpeg/coefficient_writer.rs` and
`crates/rs-jxl/src/jpeg/refinement.rs`.

## End-to-End Results

The new `tools/benchmark_jpeg_reconstruction.py` loads both native libraries
into one process, with a reusable public `rs_jxl.Decoder` context for each.
Calls alternate between libraries, including alternating which runs first.
Only library selection at context construction is patched: all codec calls,
result handling and releases execute through the actual matching native DLL.

Each measurement includes the full public `Decoder.reconstruct_jpeg` call,
FFI, native reconstruction, allocation and the returned Python bytes copy.
It excludes input I/O, library/context creation, and the subsequent comparison
against the original JPEG. There is no image output to disk. Each library has
3 warmups and 20 measured calls per image/thread count.

Two rounds used the same seven inputs at 1 and 8 threads. The second round
swapped library loading order and the before/after arguments. The table below
normalizes that second report back to **old -> new**; its raw JSON labels are
reversed. These are medians, not best-case samples.

| Progressive original | MPix | JXL bpp | 1T old/new ms | Time reduction | 1T old/new MPix/s | 8T old/new ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wallhaven-9oo8k1.jpg | 9.437 | 0.9593 | 179.095 / 164.584 | 8.10% | 52.694 / 57.340 | 61.600 / 61.660 |
| wallhaven-k81776.jpg | 8.294 | 1.8118 | 284.212 / 253.709 | 10.73% | 29.184 / 32.693 | 78.099 / 74.127 |
| wallhaven-yqqvv7.jpg | 7.849 | 1.4627 | 214.434 / 198.045 | 7.64% | 36.603 / 39.632 | 57.788 / 58.450 |

The sum of the three one-thread medians fell from 677.741 to 616.339 ms,
**9.06% less time / 1.100x throughput**. The first round also favored the new
version on all three, with throughput gains of 1.067x, 1.079x and 1.105x.
The eight-thread aggregate was only 1.017x in round two; individual results
were inconsistent across rounds. This is not a demonstrated general
multithread speedup.

Sequential controls in round two:

| Original | 1T old/new ms | 8T old/new ms |
| --- | ---: | ---: |
| wallhaven-yqxzqx.jpg | 498.056 / 496.478 | 122.094 / 126.839 |
| wallhaven-vpyekp.jpg | 15.425 / 15.853 | 8.223 / 8.177 |
| wallhaven-yq5e9x.jpg | 38.825 / 39.189 | 16.484 / 16.803 |
| wallhaven-jexkwm.jpg | 168.719 / 171.883 | 55.193 / 53.062 |

Sequential reconstruction is not intentionally changed. Measurements include
small slowdowns and speedups, reaching approximately 5% across the two rounds.
They establish no sequential improvement or strict no-regression guarantee.
The machine had unrelated workloads; another project's compilation affected
earlier measurements and was not stopped. Alternation and reversed loading
reduce timing bias, but do not make this an isolated benchmark machine. The
repeatable progressive single-thread gain is the reason to retain the change.

The bpp values describe the pre-existing compressed inputs, not a compression
improvement. No old/new JXL byte identity is used as an optimization target.
SHA256 records in the benchmark identify immutable inputs and binary versions.

## Correctness

- JPEG unit suite: 360 passed, 9 pre-existing opt-in tests ignored.
- Decoder integration suites: 37 passed, 1 pre-existing opt-in test ignored.
- Python bindings, reusable contexts and decode-recovery suites against the
  final Release DLL: 46 passed, 3 optional tests skipped.
- Both real-input benchmark rounds: all 1,288 original-JPEG byte comparisons
  passed, including warmups, both libraries and both thread counts.
- Both newly added progressive fixtures also reconstructed byte-for-byte
  through the old/new Rust CLIs and official `djxl` 0.12 (six checks). These
  single-repetition CLI runs are compatibility checks, not timing evidence.
- New differential tests cover every spectral band and point-transform plane,
  signed coefficient boundaries, all i16 values at rotating SIMD lane edges,
  out-of-band invalid values, incomplete Huffman tables, pending corrections,
  output budgets, state poisoning and error precedence.
- Final Release CLI/FFI build, `cargo fmt --all -- --check`,
  `cargo clippy --offline --locked -p rs-jxl --lib --tests -- -D warnings`,
  and `git diff --check` passed.

Portable scalar and runtime AVX2 paths are checked against independent scalar
references, not just against one another. This is focused Windows acceptance,
not a new full-corpus or Linux runtime certification.

## Rejected Pixel Experiment

An ANS alias-table selection rewrite replaced boolean multiplication with
bit masks. Ten PNG inputs passed all 40 before/after original-pixel checks,
but timing was not consistently better: the sum of one-thread medians rose
1.39%, while eight-thread medians fell only 0.91%, under substantial load
variation. The rewrite was reverted. No vendor decoder change remains.

Fresh pixel profiles still put most time in the combined weighted-predictor,
MA-tree traversal and entropy-reading loops. That does not establish that
weighted prediction arithmetic alone is the bottleneck. Further pixel work
needs a more useful change than equivalent arithmetic spelling.

## Reproduction and Artifacts

Freeze before/after native libraries from the same build profile. Use a new
JSON output path; the tool refuses to replace an existing report.

```powershell
python -B tools/benchmark_jpeg_reconstruction.py `
  --before out/backend-speed-20261108/native-before.dll `
  --after out/backend-speed-20261108/native-after.dll `
  --case out/decoder-perf-20261024/medium.jxl ../jxl_python/test_img/wallhaven-9oo8k1.jpg `
  --threads 1 8 --warmups 3 --repeats 20 --out out/jpeg-reconstruction-new.json
```

Repeat `--case JXL ORIGINAL_JPEG` for more inputs. Swap the library arguments
for an order-control run. The tool also accepts shared-library paths, but this
round exercised DLLs on Windows only. It uses the checked-out Python binding
and requires Pillow and CFFI; it must run in a dedicated benchmark process.

Local evidence under `out/backend-speed-20261108/` (ignored build artifacts):

- `profile-before.json`: diagnostic stage profiles.
- `jpeg-comparison.json`, `jpeg-comparison-reversed.json`: both timed rounds,
  every sample, original-byte check counts, dimensions, bpp and provenance.
- `jpeg-self-control.json`: short same-library harness check, not speed evidence.
- `additional-inputs.json`: provenance for the two added progressive fixtures.
- `new-fixture-oracle/report.json`: independent CLI/original-JPEG checks for
  those fixtures.
- `v1-pixels/report.json`: rejected ANS experiment, including noisy timings.
- `rs-jxl-before.exe`, `rs-jxl-after.exe`, `native-before.dll`, `native-after.dll`:
  frozen Release artifacts. `rs-jxl-v1.exe` is rejected, not the final build.

