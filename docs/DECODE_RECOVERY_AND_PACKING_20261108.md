# Decode Recovery and FFI Packing

## Scope and Result

The 2026-11-08 follow-up to the [lossless audit](LOSSLESS_CORRECTNESS_AUDIT_20261108.md)
checked malformed/truncated input, streaming EOF, metadata budgets, reusable
decoder state, and Python output ownership. No new product correctness failure
was found in this bounded Windows x86_64 check. The official low-effort F32
encoding rejection was deliberately left outside this follow-up.

A small output-copy optimization was retained in
[`pixels::append`](../crates/rs-jxl-ffi/src/pixels.rs). No encoder, scheduler,
MA/LZ77 tool, JPEG reconstruction algorithm, ABI schema, or CI workflow changed.
This is not a compressed-size optimization.

## Correctness Checks

| Check | Result |
| --- | --- |
| Native decode, streaming, hostile-input and reusable-context targets | 72 passed |
| FFI package, Release, including integration tests | 57 passed; manual packing benchmark ignored in ordinary run |
| Python functional suite against the new DLL, with both official oracles enabled | 195 passed; CUDA transfer test skipped |
| FFI Clippy with warnings denied | Passed |
| Workspace formatting | Passed |

The native hostile-input tests run their batches in killable child processes;
their four worker invocations are not additional independent parent-test counts.
The optional real-JPEG structural archive test was not enabled. Packaging,
Linux CI, and big-endian runtime execution were not repeated.

The new [Python recovery tests](../python/tests/test_decode_recovery.py) check:

- A truncated still or U16 multiframe stream fails without poisoning the reused
  decoder. Valid follow-up decodes recover original color and depth planes.
- Failed `decode_into` leaves the destination unchanged. Valid calls then work
  with the same destination and decoder.
- Unavailable display-frame and raw-layer indices do not leak state into later
  inputs. Previously returned arrays remain valid after more calls and close.
- Full inspection/decode rejects a truncated trailing XMP box; selected-frame
  decoding retains its documented early-stop behavior. Successful selection is
  not a whole-file integrity certificate.
- Compressed metadata obeys the requested decompressed-byte limit, and the
  context remains usable after that failure.

Fixtures retain their original container/profile declaration when testing
valid recovery. Stripping a required Level 10 declaration is not a valid
substitute for the complete source file.

## Output Copy Change

Previously each U16/U32/F16/F32 sample called `extend_from_slice` separately.
On little-endian hosts the implementation now copies a single byte view of the
typed plane. `bytemuck` version 1.25.2 was already used by the core crate and is
now a direct FFI dependency as well; no dependency version was upgraded.

The view changes neither ownership nor the wire representation. Float values
are not numerically converted. Existing dimension, overflow, reservation and
descriptor checks remain in place, as does the per-word little-endian conversion
for big-endian hosts. U8 retains its existing bulk-copy path.

The new primitive regression covers all 65,536 U16/F16 word patterns, selected
U32/F32 boundary and non-finite words, odd output offsets, descriptors, and
unchanged output on invalid geometry. Comparing serialized sample bytes here
checks the ABI primitive; it does not make compressed JXL byte identity a goal.

## Measurements

All builds finished before the paired measurements. Inputs and outputs were
verified exactly outside timing. Local evidence is under the ignored
`out/lossless-correctness-20261108/` directory.

### Packing Only

The manual Release unit benchmark reuses an output allocation and packs
1,048,576 samples. Each process uses four warmups and five batches of 64
iterations. Five before/after process pairs alternate execution order. Values
below are medians of the per-process medians, not complete decoding throughput.

| Type | Before, us | After, us | Before/after time |
| --- | ---: | ---: | ---: |
| U8 control | 19.575 | 20.694 | 0.95x |
| U16 | 520.814 | 77.309 | 6.74x |
| U32 | 541.911 | 153.905 | 3.52x |
| F32 | 538.517 | 146.952 | 3.66x |

Evidence: `packing-paired.json`. `packing-before.log` is the initial exploratory
run, not the source of the paired numbers above.

### Complete Python Calls

The [API benchmark](../tools/benchmark_ffi_decode.py) uses four deterministic
1024x1024 tiled inputs encoded once with the before DLL at e3. Both libraries
decode the identical files. One thread and a reusable decoder are used; five
alternating process pairs each measure ten calls after three warmups. Timing
includes native decoding, FFI/JSON and NumPy output, but excludes process
startup, disk I/O, fixture encoding and correctness comparisons.

| Input | API | Before, ms | After, ms | After, MPix/s | Time change |
| --- | --- | ---: | ---: | ---: | ---: |
| RGB8 | decode | 69.230 | 65.792 | 15.938 | -4.97% |
| RGB8 | decode_into | 69.470 | 65.532 | 16.001 | -5.67% |
| RGB16 | decode | 112.551 | 106.904 | 9.809 | -5.02% |
| RGB16 | decode_into | 111.383 | 105.926 | 9.899 | -4.90% |
| RGBA16 | decode | 158.006 | 150.098 | 6.986 | -5.01% |
| RGBA16 | decode_into | 191.092 | 184.963 | 5.669 | -3.21% |
| RGB32F | decode | 154.886 | 151.022 | 6.943 | -2.50% |
| RGB32F | decode_into | 151.907 | 156.118 | 6.717 | +2.77% |

Evidence: `ffi-decode-benchmark.json`, including raw rounds and DLL identity.
These synthetic inputs expose adapter overhead, not corpus performance. The
unchanged U8 control also moved by about 5%, while F32 `decode_into` moved in the
wrong direction. Therefore this run does **not** establish a stable whole-API
speedup attributable to packing, much less a 3x or 6x decoder improvement.
The reliable local result is removal of the per-sample packing loop. Most
decode time remains elsewhere; deeper work needs fresh backend profiles.

## Reproduction

Run from the Rust repository with NumPy, the Python package and two retained
native libraries available. Use a fresh report path; the benchmark refuses to
replace existing evidence.

```powershell
$env:PYTHONPATH = "$PWD/python"
$env:RS_JXL_NATIVE_LIB = "$PWD/target/release/rs_jxl_ffi.dll"
python -B -m pytest python/tests/test_decode_recovery.py -q
cargo test --release --locked -p rs-jxl-ffi -- --test-threads=2
cargo test --release --locked -p rs-jxl-ffi --lib pixels::tests::benchmark_output_packing -- --ignored --nocapture
python -B tools/benchmark_ffi_decode.py --before path/to/before.dll --after target/release/rs_jxl_ffi.dll --out out/ffi-comparison.json
```

The benchmark verifies shape, dtype and sample bytes both before and after each
timed batch. Its subprocess timeout is a test failure, never a codec fallback.
No timing threshold is imposed by CI, and no new platform run is claimed here.
