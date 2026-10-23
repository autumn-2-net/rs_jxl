# Minimal Bitstream Bring-Up (Not Phase Acceptance)

Correction: this document records the initial fixed-path smoke test only. The
user's first acceptance phase is the **complete pixel-lossless encoder port**.
None of the results below authorize advancing to JPEG transcoding. Current
feature coverage is tracked in `PIXEL_LOSSLESS_PORT_STATUS.md`.

Date: 2026-10-22. Repository: `.`, branch `main`.
Source baseline: optimized C++ libjxl commit
`b866b3defec13494bfbaf705db93581cda118e37`.

## Acceptance

The first complete Rust path can encode integer samples into legal single-frame
JXL, independently decoded to exactly the original samples by official djxl.
This is a **correctness/integration milestone**, not completion of the optimized
C++ encoder or a compression-ratio milestone. No C++ encoder is built, linked or
executed by the implementation. No third-party Rust JXL encoder is used.

Implemented pipeline:

1. Typed Gray / GrayAlpha / RGB / RGBA samples, unsigned 8 or 16 bits.
2. Reversible YCoCg for RGB; alpha and hidden color values are preserved.
3. Fixed gradient predictor, signed-to-unsigned residuals and HybridUint000.
4. First pass collects exact per-channel histograms; second pass writes prefix
   codes and extra bits. This differs from the C++ fast path's sampled priors.
5. Four-context fixed MA tree, prefix histogram serialization and 256x256 groups.
6. Complete metadata, frame header, TOC and group sections. A standard container
   with `jxll=10` is emitted for 16-bit Modular; ordinary 8-bit input uses a bare
   codestream. Group processing uses an owned Rayon pool when requested.

Main LZ77/RLE is explicitly disabled using the standard entropy header. It is
not approximated with a private format. Full adaptive MA learning, weighted
prediction, rANS, palette/patch search and the optimized effort scheduler are
not active. Previously ported SIMD prefix-filter and MA entropy helpers remain
tested but separate. Do not claim their acceleration for this fixed path.

## Input Semantics

- `png 0.18.1`: gray, gray-alpha, RGB, RGBA, 8/16-bit. Low-bit gray and palette
  images are expanded; tRNS produces alpha without zeroing hidden color values.
  Preservation means decoded integer samples, not original PNG file bytes.
- `zune-jpeg 0.5.15`: 8-bit gray/RGB JPEG input. The lossless reference is this
  decoder's output samples, not another decoder's IDCT/upsampling result.
- **JPEG file-byte reconstruction is not implemented yet.** Pixel encoding
  cannot reconstruct the original JPEG, and it may be much larger than it.
- Samples are tagged sRGB, orientation 1, with unassociated alpha. There is no
  ICC conversion, EXIF rotation or premultiplication. Untagged input is assumed
  to match this contract.
- Unsupported metadata is rejected by default. `--strip-metadata` explicitly
  permits discarding it, including color/orientation information, without sample
  conversion. Equal sample values do not guarantee unchanged color appearance
  after stripping a non-sRGB profile. Metadata after image data is also checked.
- APNG, CMYK/YCCK JPEG, non-8-bit JPEG, malformed/truncated inputs and multiple
  concatenated input images are rejected; no implicit first-frame extraction.

## Build and CLI

```powershell
cd .
cargo build --locked --release -p rs-jxl-cli
.\target\release\rs-jxl.exe encode input.png output.jxl --threads 8
.\target\release\rs-jxl.exe encode tagged.png output.jxl --strip-metadata
.\target\release\rs-jxl.exe encode input.jpg pixels.jxl --strip-metadata
.\target\release\rs-jxl.exe encode --help
```

On Unix use `./target/release/rs-jxl`. No CMake, system PNG/JPEG library, external
JXL encoder, or Python installation is required for the executable.

Input/output paths accept `-` for binary stdin/stdout. All diagnostics go to
stderr. Use a binary-preserving pipeline: legacy Windows PowerShell text pipes
can corrupt binary data. `tools/check_png_cli.py` demonstrates Python subprocess
binary I/O safely. A destination is not overwritten without `--force`; regular
file output is staged and published only after successful encoding. Codec
failure does not replace an existing output.

CLI limits: 512 MiB input, 16,384 per dimension, 64,000,000 pixels and 512 MiB
decoded samples. The encoder default output limit is also 512 MiB. These are
individual budgets, **not a total process RSS bound**. Input samples, group
buffers and final output coexist. PNG decoder internal allocation is separately
bounded. Encodings are buffered, not streaming.

## Rust API

```rust
use rs_jxl::{encode, encode_to, EncodeOptions, Image, PixelData, PixelLayout};

let image = Image::new(
    2, 1, PixelLayout::Rgba,
    PixelData::U8(vec![31, 97, 221, 0, 0, 255, 0, 255]),
)?;
let options = EncodeOptions {
    threads: 8,
    max_pixels: 64_000_000,
    max_output_bytes: 512 * 1024 * 1024,
    ..Default::default()
};
let bytes: Vec<u8> = encode(&image, &options)?;
let mut output = std::fs::File::create("output.jxl")?;
encode_to(&image, &options, &mut output)?;
# Ok::<(), rs_jxl::Error>(())
```

`Image` owns tightly packed, row-major, interleaved samples; width/height/sample
count are validated. `U16` is host-native typed data, not packed byte input.
PNG's big-endian 16-bit samples are converted without narrowing by the CLI.
There is no stride, arbitrary extra-channel, float, multiframe, or borrowed FFI
API in this milestone. The core permits dimensions up to 262,144 per axis,
subject to the configured pixel budget. Threads must be 1..=256; the owned pool
is capped to the actual group count and does not modify Rayon's global pool.

`encode` returns bytes; `encode_to` completes encoding before calling `Write`.
It does not provide transactional I/O for the caller's writer: a writer can
partially write and fail, and a caller-created file may already be truncated.
Use the CLI's staged output behavior when that guarantee is required.

## Reproducible Verification

Windows x86_64, Ryzen 9 7945HX, Rust 1.97.1 MSVC, release profile with thin LTO.
The declared Rust 1.89 MSRV, Linux/macOS/ARM and sanitizers remain unverified.
No CI was added. Run normal checks:

```powershell
cargo test --locked --workspace
cargo test --locked --workspace --release
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Final normal suites: **93 passed** in both debug and release (60 core unit,
9 core integration, 23 CLI unit, 1 Rust API doctest), 0 failed. The two opt-in
oracle entry points are listed as ignored in those normal runs and were then
executed explicitly as described below. Strict Clippy and formatting passed.

The two oracle test entry points are ignored by ordinary Cargo runs because
they require external programs/data. Invoking them explicitly without required
configuration fails rather than silently passing.

Core synthetic suite (OS path list: semicolons on Windows, colons on Unix):

```powershell
$env:RS_JXL_DJXL = '..\jxl_python\out\build\windows-clang-cl-cli\libjxl\tools\djxl.exe;..\jxl_python\out\encoder_opt_baseline\djxl.exe'
cargo test --locked --release -p rs-jxl --test reference -- --ignored --nocapture
```

Passed: 103 image cases x threads 1/2 x two decoder versions = **412 exact
sample comparisons**. Covers G/GA/RGB/RGBA, 8/16-bit, singleton prefix codes,
all-zero/constant images, mixed and random samples, alpha=0 hidden RGB, 1-pixel
dimensions, exact 256 boundaries, partial groups and both axes of a second DC
group (>2048). Separate tests inspect Level 10 boxes and resource-limit failures.
The oracles report versions `0.12.0 a1ebdf2` and `0.13.0 1e57a79`.

Real JPEG input suite (one decoder executable per invocation):

```powershell
$env:RS_JXL_TEST_JPEG = '..\jxl_python\test_img\wallhaven-3q3z9d.jpg'
$env:RS_JXL_DJXL = '..\jxl_python\out\build\windows-clang-cl-cli\libjxl\tools\djxl.exe'
cargo test --locked --release -p rs-jxl-cli real_jpeg_pixels_match_official_djxl -- --ignored --nocapture
```

Passed with each decoder version separately, threads 1/2. The reference is the
actual zune-jpeg input sample buffer, not Pillow/libjpeg output. Decoder children
have deadlines and are killed/reaped on timeout; generated files are cleaned up.

Real PNG CLI suite (test-only Pillow dependency):

```powershell
python -B tools\check_png_cli.py --encoder target\release\rs-jxl.exe --decoder ..\jxl_python\out\build\windows-clang-cl-cli\libjxl\tools\djxl.exe --decoder ..\jxl_python\out\encoder_opt_baseline\djxl.exe ..\jxl_python\9bde6aaf390109f97622d35ff1e900ee.png ..\jxl_python\test_img\wallhaven-mlzoy1.png ..\jxl_python\test_img\mt_lay\t1\146845733_p0.png
```

Passed all samples against both decoders at threads 1/8, plus binary stdin on
the first image. No committed image fixtures or permanent benchmark outputs.
The Python helper is intentionally 8-bit-only; Rust tests compare 16-bit PNG
samples directly to avoid Pillow's possible multichannel narrowing.

## First-Route Measurements

Single-run observations, not stable performance comparisons. PNG times include
CLI startup, input reading/decoding, JXL encoding and stdout capture, but exclude
the independent decoder. MiB uses 2^20 bytes; bpp uses all encoded file bits
divided by width x height. Thread counts are encoder pool requests.

| Input | Dimensions | Original PNG bytes | JXL bytes | bpp | 1-thread CLI s | 8-thread CLI s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `9bde6aaf390109f97622d35ff1e900ee.png` | 1264x1152 | 323776 | 1401202 | 7.698235 | 0.0791 | 0.0249 |
| `wallhaven-mlzoy1.png` | 3840x2160 | 14570359 | 15302741 | 14.759588 | 0.4863 | 0.1443 |
| `146845733_p0.png` | 4096x4096 | 77341 | 8390698 | 4.000997 | 0.5722 | 0.1123 |

The sparse RGBA image illustrates the current weakness: without RLE/LZ77, a
dominant zero residual still often costs one prefix bit per channel/sample.
These results **must not be presented as completing our optimized compressor**.
No effort flag exists until the relevant tools and scheduling are actually ported.

The real JPEG (`wallhaven-3q3z9d.jpg`, 3840x1635) yields 9,760,627 bytes,
12.437088 bpp. Pure encoding took 0.3173 s with one thread and 0.1704 s with two;
JPEG decoding/file I/O were excluded from those two measurements. It is not a
JPEG-coefficient transcode comparison.

## Deferred JPEG Node

Only after completing pixel-lossless migration, port JPEG byte-reversible conversion:
marker/table/scan parsing, coefficient
entropy decoding (including progressive refinement), exact padding/extra data,
JXL coefficient coding and `jbrd` reconstruction metadata. Reuse appropriate
ported entropy primitives, but do not route this through decoded RGB pixels.
Acceptance is original JPEG bytes reconstructed exactly by official djxl.

Subsequent nodes add multiframe/reference handling, the remaining optimized
Modular/LZ77/MA tools and scheduler, and the versioned Python-facing C ABI.
Keep these separate so an unfinished subsystem is never advertised as supported.
