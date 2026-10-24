# rs_jxl

Pure Rust migration of our optimized libjxl lossless encoder, in an independent
repository on `main`. **The complete pixel-lossless phase is still in progress.**
A legal bitstream or a collection of standalone helpers is not phase acceptance.

Source: the optimized libjxl fork, branch `jxlpy/experiment-tool-learning`,
commit `b866b3defec13494bfbaf705db93581cda118e37`.

No C++ library is linked or invoked. No existing Rust JXL encoder is substituted.
Rust utility crates are allowed: `png` and `zune-jpeg` decode input pixels,
`rayon` schedules Rust work, and pure-Rust `brotli` compresses JPEG reconstruction
metadata. The optional suffix-index route uses a vendored pure-Rust `libsais-rs`
utility with a serial-threading fix, not a codec or native binding; see
[its provenance](vendor/libsais-rs/PROVENANCE.md). SIMD uses `std::arch` with runtime detection.
Runtime decoding uses maintained official Rust source under
`vendor/jxl-rs`, not from C++ or an external CLI. Its first acceptance target is
djxl-equivalent functionality; see [decoder status](docs/DECODER_STATUS.md).
Current development targets x86_64 and the portable fallback; further ARM-specific
kernel work is paused at the user's request. Pixel decoding, visible frames and
original-JPEG reconstruction are connected end to end; this is not yet a claim
of complete djxl coverage. Decoder checks and remaining limits are recorded
separately from encoder migration.
Acceptance prioritizes exact lossless samples, original JPEG bytes and composed
frames. Small finite differences in lossy floating rendering are recorded
diagnostics, not a requirement to reproduce C++ arithmetic bit for bit.

## Active Paths

- Gray, GrayAlpha, RGB and RGBA, preserving invisible RGB; U8/U16/U32 and
  F16/F32 sample paths with explicit depth validation.
- Modular predictors and weighted presets, all 42 RCTs, MA learning/fixed trees,
  source sampling, shared/local models, group sizes 128/256/512/1024.
- Prefix/rANS entropy models, context clustering/maps and HybridUint search.
- Separate FJXL fast lossless pipeline, with scalar and native x86 kernels;
  source eligibility is distinct from the cumulative Modular effort policy.
- RLE/LZ77 greedy, lazy and optimal parsing, including indexed/block/suffix
  source routes and cooperative multi-stream execution.
- Absolute/delta/global/per-channel palettes, reversible squeeze and spatial
  progressive DC/AC grouping.
- Lossless repeated-component patches, atlas references and dictionary emission.
- Explicit color/ICC/orientation and full-resolution or subsampled extra planes.
- Frame/layer API with crop, reference slots, real zero durations and animation.
- Frame-aware source parameter search retaining actual winning frame bytes;
  ordered `jxlp` output can write frames without accumulating the whole sequence.
- Optional checked `jxli` indexes for independent animation intervals, using
  actual codestream offsets and the animation timebase.
- Region-provider input and one-pass seekable output with standard TOC
  backpatching; only current-area data is retained, with caller-thread callbacks.
- Push `streaming::FrameEncoder` accepts successive region providers, including
  crops, reference frames, zero-duration layers and bounded optional indexes.
  Ineligible region frames require an explicit current-frame buffering policy;
  no whole-animation input buffer is retained. See
  [streaming API](docs/STREAMING_OUTPUT.md).
- Full-frame difference planning against the previous or first frame, including
  hidden RGB and extra planes. Crop area is a heuristic, not an encoded-size claim.
- Separate byte-selected automatic multiframe encoding compares full/crop,
  eligible binary masks and multiple REPLACE layers, retaining actual winners.
  See [automatic frames](docs/MULTIFRAME_AUTO.md) and
  [encode-frames CLI](docs/MULTIFRAME_AUTO_CLI.md).
- Owned thread pools, byte-vector/writer APIs, file/stdin/stdout CLI.
- Cooperative source task queues share that pool across search roots, MA
  partitions/subtrees and LZ77 work. Scoped borrowing and failure draining are
  documented in [parallel execution](docs/PARALLEL_EXECUTION.md).
- Separate byte-reversible JPEG coefficient transcoding with shared auxiliary
  trees/entropy, adaptive coefficient order, CfL, source candidate search and
  exact-source ICC/Exif/XMP reconstruction deduplication.
- Maintained pure-Rust decoder source with typed pixel/extra-plane output,
  visible frames, raw crop layers, previews, ICC/color conversion, progressive
  and partial output, metadata inspection, and IDCT-free JPEG restoration.
  Decoder CLI formats include PNG/APNG, NPY, PNM/PAM/PGX/PFM, EXR and metadata
  sidecars. Lossy JPEG output is deliberately excluded.
  See [decoder status](docs/DECODER_STATUS.md) for its distinct acceptance gates.

Source coverage, active versus pending integration, and remaining optimization
work are tracked in [PIXEL_LOSSLESS_PORT_STATUS.md](docs/PIXEL_LOSSLESS_PORT_STATUS.md).
The parallel-execution, x86-fast-kernel, scheduler-detail and push-streaming
follow-up is implemented and independently verified; the latest checkpoint has
1,527 ordinary tests passing in both debug and optimized validation. See
[verification](docs/PORT_VERIFICATION.md) for the separate official decoder
matrices and actual runtime/portability scope.
Automatic scheduling, full source-policy parity, unsupported sample forms and
platform checks remain acceptance gates. Do not infer production readiness.

## Scope and Order

Finish the complete pixel-lossless encoder first, including optimized tools,
scheduling, multiframe contracts and independent verification. JPEG archival
tools are being ported in parallel; this does not finish either acceptance
phase. See [JPEG tool status](docs/JPEG_TOOL_PORT_STATUS.md).
The separate `jpeg::coefficient_reader` and `jpeg::scan_writer` modules can read
quantized coefficients and rebuild original JPEG bytes. Seventeen real JPEGs
passed byte-exact reconstruction after their archived entropy payloads were
removed. Rust reconstruction metadata (`jbrd`) and container assembly passed
136 original-byte checks with official coefficient frames as development fixtures.
The Rust coefficient-frame composer emits complete JPEG-to-JXL files and now
has separate `jpeg::transcode` and `jpeg::transcode_to` entry points. The shared
model passed 456 synthetic original-byte checks through both official decoders;
all-tier, execution, metadata and byte/writer adapter checks add 138, and the
shared-model real-corpus check adds 102. The separate release CLI passed 44
original-byte checks including file/stdin/stdout and output protection. Original
entropy bytes are removed before coefficient encoding. Historical real-file
checks and the latest reruns are recorded in the verification document.
See [JPEG API](docs/JPEG_API.md) for controls, budgets and remaining restrictions.
A future versioned C ABI will
support Python CFFI. Lossy encoding and neural scheduling are out of scope.

**The `encode` command's JPEG input decodes to pixels. Original JPEG bytes cannot
be reconstructed from that output.** Use `transcode-jpeg` or the separate archival API instead of this
route. Python packaging is not implemented. Decoder integration is tracked
separately; official `djxl` remains an independent test oracle, not an
implementation dependency.

## Build and Run

```powershell
cargo build --locked --release -p rs-jxl-cli
.\target\release\rs-jxl.exe encode input.png output.jxl --threads 8
.\target\release\rs-jxl.exe encode input.png output.jxl --shared-tree --rct-candidates 7 --weighted-candidates 2
.\target\release\rs-jxl.exe encode input.png output.jxl --source-effort 9 --threads 8
.\target\release\rs-jxl.exe encode input.png output.jxl --source-effort 9 --source-search --threads 8
.\target\release\rs-jxl.exe transcode-jpeg input.jpg archive.jxl --effort 9 --threads 4
.\target\release\rs-jxl.exe transcode-jpeg input.jpg archive.jxl --effort 9 --source-search --threads 4
.\target\release\rs-jxl.exe decode output.jxl restored.png --threads 4
.\target\release\rs-jxl.exe decode archive.jxl original.jpg
.\target\release\rs-jxl.exe decode animation.jxl frames.npy --bits 32
.\target\release\rs-jxl.exe encode-frames first.png second.png -o animation.jxl --durations 100,150 --threads 4
```

Linux/macOS use `./target/release/rs-jxl`; those platforms still require actual
verification. No CMake or external encoder is needed. `--help` lists explicit
tools. Experimental `--source-effort 1..11` executes one native source-policy
candidate; adding `--source-search` runs the migrated scheduler and retains its
winning output. These modes do not claim complete source/performance parity.
Do not combine them with manual tool flags. `--source-sample-limit N` controls
the hard MA sample resource ceiling, not the source sampling fraction; exceeding
it produces an error rather than silently weakening training. The default is
1,048,576 samples. Effort 11 alone selects the explicit signed-tier backend;
with `--source-search` it executes the separate source frame tournament (23 or
27 complete encodes including its final encode). It can be substantially more
expensive than lower efforts. Optional preflight/frontier/worker policies are
available through `scheduler::DriverOptions`, not automatically enabled.

PNG sRGB/gAMA/cHRM/iCCP and JPEG ICC are retained. Unsupported ancillary metadata
requires explicit `--strip-metadata`; supported color metadata is still retained,
and no color conversion is performed. Untagged input defaults to sRGB. CLI input
currently rejects APNG,
CMYK/YCCK and non-8-bit JPEG. PNG palettes/tRNS and packed gray expand losslessly
to decoded samples. Use `-` for binary stdin/stdout and `--force` to replace an
output file. `encode-frames` accepts ordered still-image inputs; the Rust frame
API additionally exposes explicit layer headers and extra planes.

`transcode-jpeg` never decodes JPEG to pixels and never strips metadata or falls
back to pixel encoding. It preserves the original JPEG byte stream through
standard JXL reconstruction data. Its separate controls and explicit budgets
are documented in [JPEG_API.md](docs/JPEG_API.md).

## Verification

```text
cargo test --locked --workspace
cargo test --locked --workspace --release
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

These are verification commands, not a statement that the current editing
checkpoint passes all of them. The documented full migration remains incomplete.

Independent checks run official `djxl` 0.12/0.13. Configure an OS path list:

```powershell
$env:RS_JXL_DJXL='path\to\0.12\djxl.exe;path\to\0.13\djxl.exe'
cargo test --locked --release -p rs-jxl --test reference --test float_reference --test patch_add_reference --test float_patch_reference --test metadata_frame_reference --test scheduler_frontier_reference --test palette_tools_reference -- --ignored --nocapture --test-threads=1
```

Tests compare decoded samples, including hidden RGB, extra planes, orientation
and composed frames. ICC bytes are compared separately. Compressed-file hash
equality is not an encoding goal. Ordinary Cargo runs do not execute ignored
external-oracle tests.

Development host: Rust 1.97.1, Windows x86_64 MSVC. An intermediate checkpoint
passed `cargo +1.89.0 check --locked --workspace --all-targets` and cross-target
checks for x86_64 Linux, aarch64 Linux and aarch64 macOS. These are compile/type
checks, not Linux/macOS/ARM runtime acceptance. Sanitizers and wheel builds are
not verified. Later changes must repeat the checks; see the checkpoint document.

## Documentation

- [Coverage and acceptance gates](docs/PIXEL_LOSSLESS_PORT_STATUS.md)
- [Verification checkpoints and profiling](docs/PORT_VERIFICATION.md)
- [Source map and migration order](docs/MIGRATION.md)
- [Rust pixel API](docs/PIXEL_API.md)
- [Incremental output and frame indexes](docs/STREAMING_OUTPUT.md)
- [Fast lossless source port](docs/FJXL_PORT_STATUS.md)
- [Byte-reversible JPEG API](docs/JPEG_API.md)
- [Maintained Rust decoder and API](docs/DECODER_STATUS.md)
- [Integer, floating-point and SIMD contracts](docs/INTEGER_AND_FLOAT_PATHS.md)
- [Historical bitstream bring-up](docs/PIXEL_LOSSLESS_MILESTONE.md)
- [Kernel provenance and upstream BSD notice](docs/PORTING_KERNELS.md)
- [Python interface boundary](docs/PYTHON_INTERFACE.md)
- [Contributor rules](AGENTS.md)

Translated libjxl code retains upstream attribution and BSD-3-Clause conditions.
This does not select a license for unrelated new repository code.
