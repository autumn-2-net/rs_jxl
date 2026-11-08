# Lossless Correctness Audit

## Result and Scope

The 2026-11-08 Windows x86_64 audit found no new Rust pixel-lossless,
JPEG reconstruction, or multiframe correctness failure in the cases below.
It tested the native library from commit
`8ef990f73ce54ff9b54bf1068da78bc2ee394a39`; this round adds tests and
development tooling only, with no codec, scheduler, ABI, or CI policy changes.

Two retained official tool pairs were used independently:

- libjxl `0.12.0 a1ebdf2`.
- libjxl `0.13.0 1e57a79`, a development snapshot, not a claim about every 0.13 build.

The main matrix attempted **515 encode cases**. **503 produced valid streams**
and passed all **2,012 decoder checks**. Rust encoded all **173/173** of its
cases successfully. Each official encoder succeeded on **165/171** cases; the
remaining **12 attempts failed inside the official encoders**, before Rust
decoding. These are not passes and must not be counted as reverse-path coverage.

| Check | Result |
| --- | --- |
| Main bidirectional matrix | 503/515 encoded and verified; 12 official encoding rejections |
| Supplementary U16 cross-group matrix | 6/6 encoded; 24/24 decoder checks passed |
| Native multiframe reference target, including opt-in official tests | 14 tests passed |
| Official metadata field target | 2 tests passed |
| Finite float reference selection | 10 tests passed |
| Python functional suite, excluding packaging | 184 passed, 9 opt-in skips |
| Subsequently enabled Python official-oracle tests | 8 passed; CUDA remains untested |

The supplementary matrix overlaps three default e9 cases from the main run;
do not add the two case counts and describe them as unique samples. Test counts
also differ from encoded-file and per-sample comparison counts.

Local machine-readable evidence is under the ignored directory
`out/lossless-correctness-20261108/`: `matrix.json`, `groups.json`, and
`matrix-failures/`. The earlier `smoke.json` is an overlapping harness check,
not extra acceptance coverage. Reports record tool paths, version strings,
binary fingerprints, requested options, native effective parameters, individual
checks, and failure stage. Binary fingerprints identify executables only;
compressed JXL hash identity is never an acceptance condition.

## What Is Compared

The [bidirectional runner](../tools/verify_lossless_interop.py) exercises both
Rust encode -> official decode and official encode -> Rust decode. Every
successfully encoded file is checked with both official decoders and Rust
decoding with one and four threads.

- U8/U16 pixels: exact original integer samples, including RGB at alpha zero.
  Official float32 NPY output is converted back to the original integer domain;
  arbitrary U16 low bits are exercised, not only U8 values multiplied by 257.
- Finite F32 pixels: exact original word patterns, including signed zero,
  negative values, small magnitudes, and values outside 0..1. No epsilon or
  visually similar comparison is used.
- JPEG transcode: byte-for-byte original JPEG reconstruction, including COM and
  bytes following EOI. JPEG input is never silently decoded to pixels instead.
- Frames and extra channels: original displayed states and full-resolution
  depth/selection planes, plus selected-frame/raw-layer agreement, metadata,
  and independent official decoding where the oracle supports it.

The checker deliberately refuses to certify arbitrary U31 samples using a
float32 NPY oracle. Wide integer projection checks elsewhere in the repository
must not be described as full integer recovery.

## Coverage

The deterministic matrix includes 16 fixtures: grayscale, gray+alpha, RGB,
RGBA, U8/U16, three F32 patterns, and baseline RGB/grayscale plus progressive
4:2:0 JPEG. Shapes include single pixels, one-row/one-column inputs, odd
dimensions, repeated tiles, and 129x131 cross-group inputs. PNG fixtures use
full-depth unfiltered rows, including GA16, so the input writer does not narrow
16-bit channels through Pillow. Pillow is used to generate JPEG fixtures only.

Default effort requests cover **e1 through e10** in all three encoders. The
native JPEG facade's existing effort mapping remains unchanged; the test does
not claim ten distinct JPEG implementations. A separate e9 tool matrix covers
explicit zero predictor/RCT-off, weighted prediction with small groups,
palette requests, patches, and native source search. A requested tool need not
be selected internally; these are parameter-path checks, not proof that every
possible transform was emitted.

The U16 cross-group supplement forces 128-pixel groups in both dimensions. Its
three new cases are now included in the runner's default matrix, which therefore
attempts **518** cases with two official pairs. The recorded 515-case run plus
the 6-case supplement cover those additions without rerunning the entire matrix.

The new [Rust combination matrix](../crates/rs-jxl/tests/metadata_frame_reference/combinations.rs)
uses 24 small encodes: U8/U16 RGBA+Depth, fractional/zero alpha, hidden RGB
changes, extra-only changes, repeated states, zero/mixed durations, and
previous/first/smallest-crop references. It exercises explicit prefix/zero and
ANS/gradient/RCT paths. All displayed frames and raw layers are checked through
both full and selected native decode, with one/four threads. The independent
oracle variant performs 48 original-color/depth checks with the two djxl tools.

The new [Python combination tests](../python/tests/test_lossless_combinations.py)
exercise the real FFI at e3/e9, including differing-size REPLACE layers, saved
references, exact extra planes, raw headers, reusable decoders, and path/bytes
input parity. Missing or incompatible native libraries fail the tests.

Existing opt-in targets were also executed for automatic crop/mask/multiple
rectangles, retained references, subsampled extra channels, mixed float depths,
finite half precision, streaming regions, and the repaired gradient contexts.
Subsampled plane tests retain their documented projected/upsampled comparison
scope; they are not raw-buffer identity tests across different sampling grids.

## Official Encoding Rejections

Both retained official encoders returned `JxlEncoderProcessOutput failed` for
`rgb32_gradient` and `gray32_edges` at e1, e2, and e3. The failure occurs during
PFM -> JXL encoding, before any Rust decoder is called. The generated source
PFM/NPY, any partial output, command and error are retained per failed attempt.

The same inputs encoded by Rust at those efforts were decoded exactly by both
official versions. Official e4..e10 outputs and the ordinary-range
`rgb32_unit` fixture at e1..e10 also passed. This establishes the observed
boundary, not the upstream failure's internal cause. No compatibility flags,
fallback effort, or relaxed comparator was used to turn these failures green.
The runner returns a nonzero exit status when any encode or decode attempt
fails, including this known official rejection.

## Reproduction

Use an already built native library and independent official CLI binaries.
The following PowerShell commands run from the Rust repository; replace the
tool directories with the retained binaries being audited. No compilation,
download, WSL, or C++ runtime fallback is performed by the runner.

```powershell
$env:PYTHONPATH = "$PWD/python"
$env:RS_JXL_NATIVE_LIB = "$PWD/target/release/rs_jxl_ffi.dll"
$official012 = 'C:/tools/libjxl-0.12/bin'
$official013 = 'C:/tools/libjxl-0.13/bin'

python -B tools/verify_lossless_interop.py `
  --cjxl "$official012/cjxl.exe" --djxl "$official012/djxl.exe" `
  --cjxl "$official013/cjxl.exe" --djxl "$official013/djxl.exe" `
  --workers 2 --timeout 120 --report out/lossless-audit/rerun.json

python -B -m unittest discover -s tools -p test_lossless_interop.py -v
python -B -m unittest discover -s python/tests -p test_lossless_combinations.py -v

$env:RS_JXL_DJXL = "$official012/djxl.exe;$official013/djxl.exe"
$env:RS_JXL_JXLINFO = "$official012/jxlinfo.exe;$official013/jxlinfo.exe"
cargo test -p rs-jxl --test metadata_frame_reference -- --include-ignored --test-threads=2
cargo test -p rs-jxl --test metadata_fields_reference -- --ignored
cargo test -p rs-jxl --test float_reference finite_ -- --ignored --test-threads=2
python -B -m pytest python/tests --ignore=python/tests/test_packaging.py -k Official -q
```

The runner requires NumPy and Pillow. Select fixtures with repeated `--case`,
efforts with `--efforts 1 2 3 9`, and disable the additional e9 tool matrix with
`--skip-tool-variants`. `--skip-jpeg` excludes the archival path. A new report
path is required for each run to avoid replacing existing failure evidence.
Native calls run in disposable child processes; timeouts terminate that test
process and count as failures, not as codec limits or successful fallbacks.

## Limits

This is a bounded correctness audit, not exhaustive conformance, a large corpus
rerun, or a performance benchmark. Codec code and compressed-size policy were
not changed. e11, arbitrary non-finite float payloads, all ICC profiles, all
special JPEG variants, fuzzing, and full wide-integer interoperability remain
outside this round. Linux/CI, installed wheels, and CUDA were not revalidated;
local packaging tests were excluded because `tomlkit` is absent. The ordinary
native/Python additions run under the existing test discovery; external-oracle
targets remain explicitly opt-in.
