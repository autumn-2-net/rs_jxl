# Multiframe Stability Checkpoint, 2026-10-27

This checkpoint covers the native retained-reference/lookahead implementation,
optional multiscale rectangle candidates, candidate-local Modular payload reuse,
analysis reports and the matching Rust, CLI, C ABI and Python interfaces. It is
not a new compression benchmark or a claim that standard JXL has reached its
compression limit.

## Scope and Defaults

- Differences, candidate construction, encoding and reference decisions remain
  native Rust. Python validates and forwards options; it does not build residuals.
- Retained references, reference lookahead and additional tile sizes remain
  opt-in. No expensive search was added to the default path in this checkpoint.
- Reuse caches only complete regular Modular sections for identical samples,
  dimensions and encoding options within one logical input. Headers and TOCs
  are rebuilt; reference state and patch atlases are not reused as payloads.
- Pixel correctness compares decoded samples, including hidden RGB and extra
  channels. Float tests compare sample bits. JPEG byte-transcoding remains a
  separate contract. Identical compressed JXL bytes/hashes are not the goal.

## Stale Assertions Corrected

The initial workspace run exposed a stale expectation in
`zero_duration_slot_zero_is_saved_but_timed_slot_zero_is_not`. It expected a timed
frame followed by a crop referencing unsaved slot zero to fail input validation.
The earlier heterogeneous-layer implementation already permits an empty slot as
the standard all-zero background.

Both the local official libjxl blending implementation and maintained Rust
decoder implement that zero-background behavior. The corrected test now checks:

- A zero-duration frame saves slot zero, so pixels outside the following crop
  come from that saved frame.
- A timed frame with the default save slot does not save slot zero; outside the
  following crop the color, alpha and extra-channel samples are zero.
- Frame counts, hidden RGB, interleaved alpha and separate extra output planes
  match the independently constructed expected images.

An additional independent test decodes both cases with official djxl 0.12 and
0.13, comparing the resulting samples with explicit full-frame encodings. No
encoder/decoder runtime logic was changed to make the old assertion pass.

The complete rerun also found the same obsolete rejection expectation in
`zero_duration_crop_validates_each_extra_reference_and_metadata`. That test now
checks each of the four extra-channel source slots: color comes from saved slot
1, while depth comes from that saved slot or a zero background. Source slot 4,
mismatched metadata and invalid alpha blending still must fail before output is
written. The whole `pixel_acceptance` target passed after this test-only change.

## Additional Cache Coverage

Five new tests explicitly populate and hit the cache, rather than merely
checking an automatic encoding that might not reuse anything:

- U16 samples across signed-word boundaries and both endpoints.
- An independent full-resolution 16-bit extra plane alongside 8-bit color.
- Exact F32 words, including signed zero, subnormals, infinity and a NaN payload,
  through native raw-frame decoding.
- Multiple groups and progressive passes, with rebuilt final-frame header,
  timing, name and TOC.
- Populated entries interleaved with skipped candidates, released at the final
  actual or skipped use, followed by complete-sequence pixel reconstruction.

Existing tests also cover mixed zero-duration layers, same-slot reads/writes,
retained backgrounds, lookahead ties, hidden RGB, unequal alpha depths,
subsampled extras, pruning versus complete candidate search, output/work limits,
patch-atlas exclusions and single/multiple thread execution.

## Validation

Environment: Windows x86-64, Rust MSVC toolchain, Python 3.10, local CUDA-enabled
Torch. Official djxl 0.12 and 0.13 are test-only independent decoders, not runtime
dependencies.

| Check | Result |
| --- | --- |
| Complete workspace run, `--no-fail-fast` | 2,499 passed, 1 stale assertion failed, 109 ignored; no other target failed |
| Final `pixel_acceptance` target after correcting that assertion | All 14 passed; only test code changed after the workspace run |
| Metadata/frame target with external-oracle tests explicitly enabled | 11 passed, including 6 tests normally ignored by the workspace run |
| Python tests plus multiframe benchmark-harness tests | 260 passed; official decoder and CUDA checks enabled |
| Isolated installed Windows wheel API tests | 174 passed using the bundled DLL |
| Strict workspace Clippy, all targets, `-D warnings` | Passed |
| Release workspace build, including CLI and FFI | Passed |
| Rustfmt and Git whitespace checks | Passed |
| Wheel and source distribution contents | Required modules, native DLL in wheel, source/C header in sdist, tests, guides and license notices checked |

Across the workspace run and targeted correction, all 2,500 ordinary test cases
were verified. This is not a claim that one final full-workspace command exited
successfully: the last test-only correction was verified in its complete target,
without repeating the unrelated expensive candidate-search matrix. Counts
exclude nested child-process test summaries and do not add reruns as new cases.
The 109 ignored workspace cases include the 6 independently executed oracle
cases above; the other ignored cases were not executed in this checkpoint.

The installed-wheel check ran outside the source root with
`RS_JXL_NATIVE_LIB` unset. It asserted that both the imported package and selected
DLL were inside the isolated installation. The 174 tests overlap the larger
Python test set; they are not 174 additional independent cases. Existing
`pkg_resources` and Python `Py_DEBUG` tooling warnings were non-fatal.

Logs and build artifacts are local under `out/stability-20261027/` and are not
tracked. The source distribution was inspected, not separately rebuilt on a
clean machine. Core reproduction commands, from the repository root:

```powershell
cargo test --locked --offline --workspace -j 2 --no-fail-fast -- --test-threads=4
cargo clippy --locked --offline --workspace --all-targets -j 2 -- -D warnings
cargo build --release --locked --offline --workspace -j 2
cargo fmt --all -- --check
git diff --check

# Set RS_JXL_DJXL to a semicolon-separated list of trusted djxl executables.
cargo test --locked --offline -p rs-jxl --test metadata_frame_reference -- --include-ignored --test-threads=2

$env:PYTHONPATH = "$PWD/python;$PWD/python/tests"
$env:RS_JXL_NATIVE_LIB = "$PWD/target/release/rs_jxl_ffi.dll"
$env:RS_JXL_TEST_CUDA = '1'
python -B -m pytest python/tests scripts/test_benchmark_multiframe_native.py -q -p no:cacheprovider
```

## Boundaries

This is not a new full-image-corpus run, a fuzzing campaign, or Linux/macOS/ARM
runtime validation. Ignored tests are not passes unless separately executed.
Special float-word native tests do not establish independent official-decoder
support for every NaN payload or arbitrary floating metadata combination.

The small gains and high costs in the
[multiscale experiment](MULTIFRAME_MULTISCALE_BENCHMARK.md) justify keeping those
options off by default, not a statement that the format offers no further reuse.
See the [analysis guide](MULTIFRAME_ANALYSIS_GUIDE.md) and
[candidate-order report](MULTIFRAME_ORDER_BENCHMARK.md) for measured tradeoffs.
