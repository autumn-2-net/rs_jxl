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
- Native [multiframe analysis](docs/MULTIFRAME_ANALYSIS.md) reports exact changed
  pixels, reference-policy geometry, optional historical matches, mask eligibility,
  rectangle proposals and candidate/repeated-payload work
  without encoding; reusable Rust contexts and Python share the same planner.
- Separate byte-selected automatic multiframe encoding compares full/crop,
  eligible binary masks and multiple REPLACE layers, retaining actual winners.
  Optional retained-reference lookahead and multiscale grids expand reuse;
  extra grids remain disabled by default due to their measured search costs.
  See [automatic frames](docs/MULTIFRAME_AUTO.md) and
  [encode-frames CLI](docs/MULTIFRAME_AUTO_CLI.md).
- Typed `codec::EncodeJob` dispatch and reusable `codec::Encoder` workers,
  alongside `decode::Decoder`, byte-vector/writer APIs and file/stdin/stdout CLI.
- Incremental pixel decoding with push/EOF, early frame delivery and blocking
  readers; CLI opt-in `decode --stream-input`. See
  [streaming input](docs/STREAMING_INPUT.md).
- Exact native 17..31-bit Modular integer output without floating-point
  normalization, including extra planes and integer-compatible frame references.
  See [native integers](docs/NATIVE_INTEGER_DECODE.md).
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
follow-up is implemented and independently verified; its earlier checkpoint recorded
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
A versioned Rust C ABI supports Python CFFI through the separate `rs_jxl`
package; see [Python bindings](docs/PYTHON_INTERFACE.md). Lossy encoding and
neural scheduling are out of scope.

**The `encode` command's JPEG input decodes to pixels. Original JPEG bytes cannot
be reconstructed from that output.** Use `transcode-jpeg` or the separate archival API instead of this
route. Python wheels bundle the Rust shared library, not a C++ codec or a CLI
subprocess. Decoder integration is tracked
separately; official `djxl` remains an independent test oracle, not an
implementation dependency.

## Codec Facade

CLI and FFI still-image adapters map their arguments to
`codec::EncodeJob { options, source }`. `SourceRequest` selects a direct
candidate or explicit Auto/Scout/Pilot search; the core owns effort/search and
effort-11 tournament dispatch. `codec::encode_job` and `codec::transcode_jpeg`
provide one-shot calls. `codec::Encoder` reuses workers across still, frame,
automatic multiframe and JPEG calls; `decode::Decoder` separately reuses decoding
workers. Each input has fresh image/parser/reference state. Threads are fixed per
context, and one thread uses no pool.

The version-1 C ABI has optional additive context functions, typed error
categories and still/JPEG reports without changing its request layout or removing
one-shot calls. Decoder metadata types still include upstream reexports, and
Python incremental streaming is not implemented. This is not closure of every
interface or portability gap. See [architecture](docs/ARCHITECTURE.md) and the
[binding contract](docs/PYTHON_INTERFACE.md).

## Python

```powershell
cargo build --locked --offline --release -p rs-jxl-ffi
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
python -m pip install dist/rs_jxl-0.1.0-py3-none-win_amd64.whl
```

The filename above is for Windows x64; wheel tags follow the native build
platform. Source builds need a Rust toolchain and cached locked dependencies;
`cargo fetch --locked` prepares those dependencies on an online machine.
Set `RS_JXL_NATIVE_LIB` to an already built matching library to skip compilation
during packaging. Installing a wheel does not need Rust. No CI workflow changed.

```python
import rs_jxl

jxl = rs_jxl.encode("input.png", effort=9, threads=8)
pixels = rs_jxl.decode(jxl)  # owned NumPy HWC array
archived = rs_jxl.transcode_jpeg("input.jpg", effort=9, threads=8)
original_jpeg = rs_jxl.reconstruct_jpeg(archived)
```

For repeated calls, `Encoder(threads=...)` and `Decoder(threads=...)` support
context-manager use and `close()`. Threads stay fixed; all other options are
per-call and do not become stored defaults. Native calls and close are serialized,
but Python result conversion/copying runs after the context lock is released:

```python
with rs_jxl.Encoder(threads=8) as encoder, rs_jxl.Decoder(threads=8) as decoder:
    jxl, meta = encoder.encode("input.png", effort=9, return_info=True)
    pixels = decoder.decode(jxl)
    print(meta["report"]["codec_seconds"])
```

`encode` and `transcode_jpeg` accept `return_info=True`, returning output plus
metadata with codec statistics under `meta["report"]`. `codec_seconds` excludes
pool creation, external pixel-input decoding and transport/I/O. JPEG transcoding
includes JPEG parsing and coefficient preparation; there is no exact analysis/
backend timing split. `effective_parameters` projects scheduler `CandidateParams`,
not a full backend recipe; FJXL uses its own default effort. Multiframe strategy
metadata retains its separate shape. Mutable encoding inputs are copied by default; explicit
`borrow_inputs=True` permits eligible buffers to be borrowed without concurrent
mutation through any alias, but normalization/consolidation can still copy.
Files and streams remain fully buffered. Decoding returns owned arrays or uses
`decode_into` to copy a selected frame into an exact-shape/dtype writable,
C-contiguous NumPy destination after validation. Public outputs do not borrow
Rust storage; internal derived views retain their native owner even when held
by an exception traceback beyond helper scope exit.

The package also accepts arrays/tensors, encoded bytes and binary streams,
supports extra channels and automatic multiframe strategies, and keeps raw
zero-duration layers separate from display frames. It does not alias `jxlpy`
or silently accept unsupported lossy/libjxl options. See the
[API and build guide](docs/PYTHON_API.md), [FFI boundary](docs/PYTHON_INTERFACE.md),
[code architecture](docs/ARCHITECTURE.md) and [documentation index](docs/README.md).

`rs_jxl.encode_layers(images)` stores different-size images as original-size
layers on a shared canvas; `rs_jxl.decode_layer(data, layer=i)` retrieves one
without canvas padding or compositing. Layout, depth, color and extra-channel
metadata must agree. See [heterogeneous layers](docs/HETEROGENEOUS_LAYERS.md) for
placement, timing, reference semantics and the distinction from automatic diff
encoding. [Stronger temporal reuse](docs/MULTIFRAME_REUSE_ROADMAP.md) tracks
implemented retained-reference policies and future displaced patch matching.
See [multiscale measurements](docs/MULTIFRAME_MULTISCALE_BENCHMARK.md) for the
current compression gains and substantial search-cost tradeoffs.

## Build and Run

The decoder's AVX-512 kernels are optional: add `--features decoder-avx512`
to the build command below. This retains runtime CPU detection and AVX2/SSE/
scalar fallback; the default build stays AVX2-capable. It does not change the
encoder. See [decoder SIMD measurements](docs/DECODER_SINGLE_20261024.md)
and the [paired-AVX2 and targeted-AVX512 experiments](docs/DECODER_SIMD_20261025.md).
The [entropy follow-up](docs/DECODER_ENTROPY_20261025.md) records five measured
candidates that were not retained and the added ANS regression coverage.
The [multithreaded pixel follow-up](docs/DECODER_THREADS_20261025.md) describes
direct output-buffer ownership, 1/8/16-thread measurements, and rejected
scheduling/lookup experiments.

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
it produces an error rather than silently weakening training. Omission imposes
no extra sample-count ceiling and does not change the sampling fraction.
The library uses `ModularOptions::sample_limit: Option<usize>` (`None` by default).
See the [resource-limit audit](docs/RESOURCE_LIMIT_AUDIT.md) for removed default
caps, retained format checks and opt-in limits for untrusted inputs.
Effort 11 alone selects the explicit signed-tier backend;
with `--source-search` it executes the separate source frame tournament (23 or
27 complete encodes including its final encode). It can be substantially more
expensive than lower efforts. Optional preflight/frontier/worker policies are
available through `scheduler::DriverOptions`, not automatically enabled.

`--source-scout` is an explicit experimental frontier for source search at
effort 9/10; it can change compression size and is not the default scheduler.
`--source-pilot` is a separate opt-in content-aware pilot-first experiment at
effort 9/10 (requires `--source-search`, conflicts with `--source-scout`). It
keeps full candidate outputs, reports the winning parameters, and uses sampled
evidence only to decide which candidates to run. It does not promise the result
of exhaustive search or a throughput floor on arbitrary images. The Rust entry
points are `scheduler::pilot::encode` and `encode_with_options`; `PilotOptions`
exposes source MA leaf resource limits and representation receipts. The pilot
preset uses bounded initial candidate waves, content-gated overlap of a reusable
automatic result, and continuous-tone near-tie challenger screening. Flat and
low-color images retain the original candidate checks. These are individually
controlled by `parallel_initial`, `overlap_automatic` and `near_tie_consensus`.
Ordinary source search is unchanged. Probe quarter-sampling and experimental
SIMD/cache alternatives remain disabled. Encoder
thread/pixel/output/sample limits remain explicit; the source MA leaf limit is
not the direct encoder's `base.modular.max_tree_leaves`. For example:

```powershell
.\target\release\rs-jxl.exe encode input.png output.jxl --source-effort 9 --source-search --source-pilot --threads 16
```

`--source-progressive-groups` optionally replaces eligible effort-9 pilot
searches with cheap-front admission and resumable original-group racing.
Decisive fronts encode directly; ambiguous fronts reuse completed group data.
Unsupported dependent layouts and wide integers keep the ordinary pilot. See the
[progressive scheduler](docs/PROGRESSIVE_GROUP_SCHEDULER.md) for scope,
diagnostics, resource limits, and measured rollout status.

`--ma-budget-mode strict|bounded` explicitly selects whether learned MA leaf
exhaustion errors or stops optional splitting while retaining lossless pixels.
`--ma-max-leaves N` adds a cap that survives source/pilot projection. Omission
preserves the existing defaults. Required partitions and decoder limits still
apply; see [MA tree budgets](docs/MA_TREE_BUDGET.md).

Development measurements in `docs/ENCODER_SCOUT_20261025.md` distinguish complete
encoder-call time (including the pilot) from CLI wall time. The call excludes
input decoding and output I/O; report milliseconds alongside MPix/s for small
images. `tools/benchmark_encoder_tools.py --repeats 3 --warmups 1` verifies every
output with official `djxl` and reports medians separately from process timing.

The ordinary JPEG source search now reuses matching AC payloads between
candidates without removing trials. See the
[encoder scout experiment](docs/ENCODER_SCOUT_20261025.md) for measurements,
rejected variants, limits, and current MA parallel/SIMD port coverage.

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
The bounded `jpeg::compatibility::probe` diagnoses special JPEG headers without
decoding pixels. See [JPEG compatibility](docs/JPEG_COMPATIBILITY.md) for the
distinction between format limits and implementation gaps; a successful probe
does not certify transcode support.

## Verification

```text
cargo test --locked --workspace
cargo test --locked --workspace --release
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

These are verification commands, not a statement that the current editing
checkpoint passes all of them. The documented full migration remains incomplete.
Validation of the unified codec facade, contexts, FFI errors/reports and Python
buffers is recorded under [current verification](docs/PYTHON_API.md#current-verification).
That checkpoint separates workspace tests, focused review-fix tests, release-DLL
checks and installed-wheel tests. It does not claim new performance measurements
or cross-platform runtime coverage.

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
checks, not Linux/macOS/ARM runtime acceptance. Sanitizers are not verified.
Current installed-wheel checks are recorded separately under
[current verification](docs/PYTHON_API.md#current-verification). Later changes
must repeat the relevant checks; historical binding validation remains separate.

## Documentation

- [Encoding parameters: defaults, search, JPEG and multiframe (Chinese)](docs/ENCODING_PARAMETERS.md)
- [Multiframe analysis: interpreting reports and search tradeoffs (Chinese)](docs/MULTIFRAME_ANALYSIS_GUIDE.md)
- [Coverage and acceptance gates](docs/PIXEL_LOSSLESS_PORT_STATUS.md)
- [Verification checkpoints and profiling](docs/PORT_VERIFICATION.md)
- [Source map and migration order](docs/MIGRATION.md)
- [Rust pixel API](docs/PIXEL_API.md)
- [MA tree budget modes and CLI controls](docs/MA_TREE_BUDGET.md)
- [Exact MA partition reuse and dominance pruning](docs/MA_SEARCH_PRUNING.md)
- [Progressive original-group scheduling](docs/PROGRESSIVE_GROUP_SCHEDULER.md)
- [Incremental output and frame indexes](docs/STREAMING_OUTPUT.md)
- [Fast lossless source port](docs/FJXL_PORT_STATUS.md)
- [Byte-reversible JPEG API](docs/JPEG_API.md)
- [Maintained Rust decoder and API](docs/DECODER_STATUS.md)
- [Lossless codec performance baseline](docs/PERFORMANCE_BASELINE_20261024.md)
- [JPEG reconstruction optimization](docs/DECODER_PERFORMANCE_20261024.md)
- [Parallel JPEG reconstruction](docs/DECODER_PARALLEL_20261024.md)
- [Single-thread and pixel decoder follow-up](docs/DECODER_SINGLE_20261024.md)
- [Integer, floating-point and SIMD contracts](docs/INTEGER_AND_FLOAT_PATHS.md)
- [Historical bitstream bring-up](docs/PIXEL_LOSSLESS_MILESTONE.md)
- [Kernel provenance and upstream BSD notice](docs/PORTING_KERNELS.md)
- [Python CFFI bindings](docs/PYTHON_INTERFACE.md)
- [Contributor rules](AGENTS.md)

Translated libjxl code retains upstream attribution and BSD-3-Clause conditions.
This does not select a license for unrelated new repository code.
