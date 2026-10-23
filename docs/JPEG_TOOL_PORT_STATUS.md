# JPEG Archival Tool Port

Frozen source: optimized libjxl `b866b3defec13494bfbaf705db93581cda118e37`,
principally `lib/jxl/jpeg/jpeg_data.h`, `enc_jpeg_data_reader.cc`,
`enc_jpeg_huffman_decode.{h,cc}`, `dec_jpeg_data_writer.cc`,
`dec_jpeg_serialization_state.h`, `jpeg_data.cc`, and `enc_jpeg_data.cc`.
BSD attribution is retained in each module.

The user authorized these prerequisites in parallel with the incomplete pixel
port. A Rust coefficient composer now emits complete JPEG-to-JXL files with
separate `jpeg::transcode` and `jpeg::transcode_to` byte/writer APIs. **The full
optimized source-tool port is still in progress.** The existing `encode` CLI
command still decodes JPEG pixels using zune-jpeg; it is not archival transcode.
The separate `transcode-jpeg` CLI now calls the byte-reversible API directly.
See [JPEG_API.md](JPEG_API.md) for public contracts and unsupported structures.

## Implemented Modules

| Module under `crates/rs-jxl/src/jpeg` | Responsibility |
| --- | --- |
| `data.rs`, `sof.rs` | Components, quantized coefficient storage, SOF/MCU geometry, scan and reconstruction metadata |
| `reader.rs` | Bounded SOF0/1/2 structural parsing, DQT/DHT/SOS/DRI, APP/COM, per-scan active table snapshots, ordered archival records |
| `bitstream.rs` | MSB-first entropy reads, FF/00 stuffing, explicit restart boundaries, preserved or strict padding, signed magnitude extension |
| `huffman.rs` | Canonical JPEG codes, eight-bit root/second-level lookup, incomplete-tree validation, symbol emission codes |
| `scan_progression.rs` | Per-component spectral bit-plane coverage, overlap/order checking with atomic errors |
| `coefficients.rs` | Initial DC/AC block scans and progressive refinement, signed coefficients, EOB runs, reset hints and extra trailing zero-run hints |
| `coefficient_reader.rs` | Bounded full scan/MCU traversal, SOS table snapshots, padded component storage, predictors, progression, restart cadence and exact padding/hints |
| `entropy_writer.rs` | Bounded MSB-first output, FF/00 stuffing, preserved padding, restart fill and Huffman symbol/payload emission |
| `coefficient_writer.rs` | Sequential/initial/refinement block emission, bounded EOB/correction buffering, preserved trailing ZRLs and explicit flush boundaries |
| `scan_writer.rs`, `reconstruction_validation.rs` | Coefficient-based JPEG regeneration, marker-only structural preflight, bounded scan traversal, restart/fill/padding and reconstruction hints without reading archived entropy bytes |
| `reconstruction_metadata.rs` | Source VisitFields serialization and validated archival-record adapter; separate bounded byte-aligned header and raw metadata payload for caller-owned Brotli compression |
| `reconstruction_brotli.rs`, `container.rs` | Bounded pure-Rust reconstruction compression and JXL signature/ftyp/jxll/jbrd/jxlc boxes |
| `transcode_layout.rs`, `frame_header.rs` | Source channel/quantization mapping, MCU/group geometry, original-color VarDCT frame header and TOC |
| `quantization.rs`, `dc_streams.rs` | Source F16-rounded DC factors, integer quantization tables, DC/qdc and AC control planes |
| `cfl.rs`, `coefficient_order.rs` | Source JPEG AC chroma-from-luma fitting, reversible integer residuals and DCT8 adaptive coefficient orders |
| `ac_tokens.rs`, `entropy.rs` | DCT8 nonzero/zero-density contexts, block-context map fields, grouped coefficient entropy and order streams |
| `modular_policy.rs`, `color_metadata.rs`, `exif.rs` | JPEG-specific domain/global policies, bounded ICC/XMP/Exif extraction and source display orientation |
| `transcode.rs`, `modular/auxiliary.rs` | Complete coefficient-frame assembly; independent DC/metadata/quant tree domains merged by stream ID, shared auxiliary and AC entropy models; unaligned small-frame writing |
| `archival.rs`, `search.rs` | Public byte-input/writer APIs, source effort mapping, owned threads, explicit budgets, cumulative whole-frame search and no pixel fallback |

These are Rust tools, not C++ wrappers, IDCT/pixel reconstruction, or a substitute
JXL encoder. Mutable public structures are not a stable serialized format or C ABI.
Huffman constructors accept real DHT entries, not libjxl's synthetic invalid
sentinel. Raw archived segments retain their original bytes.

## Frame Acceptance

2026-10-23: the private composer passed 456 official-decoder reconstruction
checks: 19 synthetic JPEGs x 12 configurations x djxl 0.12 and 0.13. This includes
gray, RGB, YCbCr 444/420/422/440, odd MCU edges, 256/257-pixel group boundaries,
six progressive fixtures (including signed refinement) and 11 restart fixtures.
Configurations cover Falcon/Tortoise, source/default/adaptive orders and AC CfL
enabled/disabled. Every input's archived entropy bytes are erased first.

The test calls no external encoder. An earlier, distinct jbrd-only test did use
official coefficient-frame fixtures; its 136 checks must not be conflated with
the fully Rust-generated frames. Tests are in `transcode_reference_tests.rs`.
All 17 real JPEGs under `test_img`, `test_img/mt_lay/t2` and `t3` also passed
three configurations through both official decoders: 102 byte-exact checks,
including three progressive inputs and one restart input. Together these are
558 full-Rust-frame reconstruction checks, distinct from the jbrd-only checks.
The real-corpus run used an optimized validation build with LTO disabled and
16 codegen units, not the normal release profile. Its 51 encodes took 54.753390s
in the coefficient-encoding stage and produced 83,459,228 bytes in total across
all configurations; total test runtime including extraction/decoding was 73.23s.
This is not a warmed production benchmark or a C++ comparison. Synthetic timing
came from a debug build and is not a production speed benchmark either.

Reproduce from the independent repository in PowerShell:

```powershell
$env:RS_JXL_DJXL = '../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe;../jxl_python/out/encoder_opt_baseline/djxl.exe'
$env:RS_JXL_JPEG_INPUTS = '../jxl_python/test_img;../jxl_python/test_img/mt_lay/t2;../jxl_python/test_img/mt_lay/t3'
$env:RS_JXL_JPEG_MATRIX = '0'
cargo test --locked --release -p rs-jxl --lib jpeg::transcode_reference_tests::rust_coefficient_transcode_restores_real -- --include-ignored --nocapture
$env:RS_JXL_JPEG_MATRIX = '1'
cargo test --locked -p rs-jxl --lib jpeg::transcode_reference_tests::rust_coefficient_transcode_restores_synthetic -- --include-ignored --nocapture
```

The above release command uses the normal manifest profile. For the reported
validation build only, `CARGO_PROFILE_RELEASE_LTO=false` and
`CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16` were set for that process. Do not compare
its timings as if they were normal-profile CLI end-to-end measurements.

The above historical real-corpus measurement used local auxiliary models. The
new shared-domain composer has independently passed 456 synthetic checks and
138 focused checks: 44 across all 11 tiers, 16 execution-setting combinations,
30 metadata checks, and 48 checks of the actual byte/writer adapter. The adapter
was promoted only after these passed. The shared-model real-corpus rerun passed
102 checks, bringing complete-frame API coverage to 696 original-byte checks.
A separate normal-release CLI run passed another 44 checks, including actual
file/stdin/stdout paths and output-protection boundaries. `PORT_VERIFICATION.md`
records the build profiles, resource fix and later workspace reruns; historical
evidence is not a claim that all later edits are tested.

Matching ICC/Exif/XMP records now use reconstruction deduplication against the
actual emitted ICC profile and metadata boxes. Unknown, duplicate and mismatched
APP records remain archived. The changed path passed 30 new original-byte
checks. The frozen source has no JPEG JUMBF extraction to port. The cumulative
whole-frame search is now wired through `jpeg::transcode_search` and CLI
`transcode-jpeg --source-search`; 176 independent original-byte checks covered
all effort prefixes and single/multiple threads. Performance parity, the Rust
runtime decoder and future C ABI remain separate work. Unsupported JPEG forms fail;
no pixel-recompression fallback masquerades as byte-reversible conversion.

## API Boundaries

`reader::read_jpeg(&[u8])` returns structured `JpegData` without allocating
pixel/MCU-sized buffers. `read_jpeg_with_limits` exposes input, record, scan,
table and restart-count resource bounds. Structural parsing supports 8-bit
Huffman sequential/extended sequential/progressive DCT JPEG. Arithmetic,
lossless-JPEG, hierarchical/DNL/extension modes are explicit errors.

`JpegData::reconstruct()` replays original ordered marker, fill, scan and tail
records. **It is not coefficient-based reconstruction and not a compression
operation.** Modifying parsed descriptive fields does not rewrite the archived
segments. Original entropy scan bytes are still retained, not re-encoded.

The separate `scan_writer::reconstruct_from_coefficients(&JpegData, &WriteLimits)`
returns a regenerated JPEG from quantized coefficients and reconstruction hints.
It uses original marker/metadata records, but never reads `entropy.bytes`; those
payloads may be cleared. Preflight checks public geometry, tables and scans
against a reparsed marker-only template, validates hints and padding, and bounds
output, traversal work and refinement buffering. Errors return no partial JPEG.
This is JPEG-to-JPEG reconstruction, not a JXL encoder or compressed container.

`reconstruction_metadata::serialize_reconstruction_metadata(&JpegData, max_bytes)`
returns `EncodedMetadata { header: Vec<u8>, payload: Vec<u8> }` or `MetadataError`.
The header uses the frozen `JPEGData::VisitFields` field order, LSB-first bits,
shortest source U32 selectors and zero byte padding. The uncompressed payload
concatenates all APP data, COM data, inter-marker data and tail data, in that
order. This compatibility entry keeps **all APP markers Unknown**, including
ICC/Exif/XMP. The composer instead calls
`serialize_reconstruction_metadata_with_sources` with actual emitted metadata;
only matching records receive known reconstruction types. The
serializer itself performs no Brotli compression, I/O, coefficient coding or
container assembly. `max_bytes` caps the combined header and raw payload length;
failure exposes neither partial output buffer.

The adapter reuses reconstruction header validation and default `WriteLimits`,
without reading coefficients or archived entropy payloads. It excludes SOI from
the wire marker list, synthesizes fake FF entries for consecutive fill and
inter-marker records, and enforces their 65,535-byte combined run limit. It adds
sentinel symbol 256 at the highest populated DHT code length and synthesizes a
zero-table entry for each raw empty DHT marker. Scan quantization selectors map
to vector indices through active table snapshots. One to three quantization
tables are supported; four are rejected by the source wire format. Unused
tables must copy their predecessor. Quantization changes that cannot share a
global component table, differing DRI redefinitions, nonzero restart fill,
unrepresentable Huffman fields and unsupported marker layouts are explicit
errors. Identical table/DRI repetitions remain representable when these checks
hold. Marker count, hint deltas/counts, pass fields, padding and tail sizes are
bounded by source fields plus the default reconstruction work limits.

`decode_initial_block` and `refine_block` operate on one quantized 8x8 block,
preserve signed i16 values and return reconstruction hints. They do not traverse
MCUs or select per-scan tables. The caller must maintain component DC predictors,
scan-wide EOB state, restart intervals, progression and block-indexed hints.
Refinement treats negative DC bit updates differently from signed AC magnitudes.
Errors leave input position, coefficients and block state unchanged.

`coefficient_reader::read_coefficients(&[u8])` returns `JpegData` with original
quantized i16 coefficients in block-major natural order. The separately named
`read_coefficients_with_limits` accepts `CoefficientReadLimits`: structural
reader bounds plus total padded coefficient bytes, aggregate block visits across
all scans, reconstruction-hint entries, padding-bit count and padding policy.
Defaults are 256 MiB coefficient storage, 64 Mi block visits, 1,048,576 hint
entries, 8 Mi padding bits and preservation of nonstandard padding. Storage and
aggregate work/progression/restart-count checks precede every coefficient-buffer
allocation; hint and padding budgets are checked before appending those records.
Coefficient allocations use checked lengths and fallible reservation. Errors
return no partial image; entropy failures identify the scan, scan-order block
index, input position and underlying block/Huffman/entropy error.

Traversal uses the frozen reader's component order and padded row stride.
Interleaved scans read actual dummy edge blocks; noninterleaved scans visit only
the component's unpadded extent and leave untouched padding zero (or retain what
an earlier interleaved scan transmitted). DC predictors are per component; EOB
state is shared within each scan. Both reset at scan starts and expected restart
boundaries. Restarts count MCUs, wrap RST0..7 and start at RST0 for each scan.
DHT and DRI changes affect only scans whose SOS snapshots reference them; DQT
snapshots are retained without dequantizing or remapping on-wire selectors.
Reset/ZRL hints use scan traversal block indices, not padded storage indices.
All alignment bits, including all-one padding, are stored MSB-first in input
boundary order; zero/mixed padding sets `has_zero_padding_bit`. Strict all-one
padding is optional. The original entropy/fill/metadata records remain intact.

`coefficient_writer::{encode_sequential_block, encode_initial_block,
encode_refinement_block}` consume final natural-order coefficients, selected
Huffman tables and the shared `ScanBand`. The caller owns component predictors
and flushes `DctCodingState` before reset-point hints, restarts and scan ends.
`DctCodingState::new(max_refinement_bits)` allocates nothing; its packed correction
buffer grows fallibly within the configured limit and the 32,767-block EOB cap.
Pending runs snapshot only their fifteen possible EOB codewords. Buffer limits
or missing Huffman symbols produce errors, never an early flush that changes
the original representation. Initial negative DC shifts arithmetically; negative
AC shifts its magnitude. The frozen reader's final ZRL overshoot is preserved,
including four trailing ZRLs for a full zero AC band.

Block writer errors can partially advance output and predictors; they poison the
coding state against retry. The enclosing reconstruction call discards that
local state and returns no output on failure, without cloning accumulated output.

The structural parser validates SOS header semantics but does not decode entropy,
validate restart spacing or prove that a complete JPEG is decodable. The source's
progression checker tolerates some skipped refinement levels; it is not a strict
JPEG conformance validator. Coefficient tools currently bound Al to 10, matching
the frozen source's reconstruction path, even though structural SOS accepts 13.
The composed reader additionally rejects AC-before-DC, refinement before an
initial band and components with no DC scan. It allows progressive files that
omit AC bands or low bit planes, and retains the progression primitive's skipped
refinement-level tolerance. It rejects extra entropy bytes after the expected
MCUs, early/late/missing/extra/wrong-order restarts and overlong EOB runs. These
checks do not turn it into a general JPEG conformance validator.

## Verification

The 2026-10-23 central debug and release workspace runs passed 578 core tests, including these
modules. Coverage includes every truncated prefix of a valid structural fixture,
table redefinition, FF fill/stuffing, restarts, arbitrary APP/COM/tail bytes,
16-bit quantization entries, invalid canonical trees, negative DC/AC values,
progressive EOB/refinement and failed-operation atomicity.

The real archive test read all 17 JPEGs in the original project's `test_img`,
including three progressive files (9 or 10 scans): 33,132,526 original bytes
were reconstructed exactly. No JXL encoding or output files were involved.
This validates structural archive preservation, not JPEG-to-JXL reconstruction.

Traversal tests in `jpeg/coefficient_reader_tests.rs` specify known signed
coefficients independently of the decoder and generate small synthetic baseline,
extended sequential, progressive and restarted entropy streams. They cover
reordered 4:2:0 components, interleaved dummy blocks, noninterleaved padded strides,
alternating progressive geometry, DHT/DQT/DRI redefinitions, signed refinement,
EOB/ZRL hints, restart wraparound, mixed padding, truncated/malformed entropy and
tiny resource budgets. A maximal-dimension header exercises limit rejection
without allocating its coefficient image. The block primitive also has a
regression for a full refinement band separating two EOB runs: the frozen source
decrements EOB state even when no EOB symbol was read.

The later 2026-10-23 parent-run central debug checkpoint passed **95 JPEG unit
tests**, including coefficient traversal, block emission, scan reconstruction and
entropy writing. `jpeg/coefficient_writer_tests.rs` checks independently packed
expected bytes, signed DC/AC, progressive initial and refinement passes, correction
bit ordering across packed words, EOB grouping/reset/maximum run, explicit ZRL
overshoot, table snapshots, tiny budgets, missing symbols and poisoned-state
errors. These edge tests do not allocate giant coefficient images. The final
local checkpoint only formats the owned files and makes equivalent test-only
lint cleanups; Cargo and independent oracle execution remain parent-owned.

`tests/jpeg_coefficients_reference.rs` provides an ignored independent
libjpeg-turbo coefficient oracle test. `RS_JXL_JPEG_COEFFICIENT_ORACLE` names the
development-only executable and `RS_JXL_JPEG_INPUTS` is an OS path list of files
or nonrecursive directories. The fixture selects at most the first 17 sorted
JPEGs, reports truncation, and rejects input/coefficient/output cap failures.
Bounds are 16 MiB per input, 64 million pixels and 256 MiB coefficient/output
storage. It validates the complete `JCO1` binary protocol and EOF, compares every
real coefficient using the Rust padded row stride and libjpeg's actual unpadded
block dimensions, and also checks unchanged archival replay. Child processes
have a timeout with kill/reap and checked temporary-output cleanup.

The parent reports **17/17 independent coefficient checks passed**, comparing
**261,138,944 real coefficients** exactly, including **three progressive files
and one file with restart markers**. The oracle executable is isolated outside
the Rust repository at
`../jxl_python/out/rs-jxl-oracle/jpeg_coeff_oracle.exe`; it is neither a
codec implementation dependency nor linked into Rust production code.

The parent also reports **17/17 coefficient-based reconstructions passed**:
after clearing **all archived `entropy.bytes`**, the writer rebuilt
**33,132,526 original JPEG bytes exactly**, including the same progressive and
restart coverage. Unlike archival replay, this exercises regenerated entropy.
That checkpoint alone did not validate metadata serialization or reconstruction
from JXL; the separate metadata-only checkpoint follows.

The parent-run metadata oracle replaced only the `jbrd` box of official JXL
fixtures with the Rust serializer's header plus its payload compressed using
pure-Rust Brotli at qualities 3 and 9. Official djxl versions 0.12 and 0.13
reconstructed the original JPEG bytes exactly: **17/17 files, 68 successful
checks**, including three progressive files and one with restarts. Archived
entropy payloads were cleared before serialization. This independently validates
the metadata prerequisite, **not a full Rust JPEG-to-JXL transcoder**: the tested
coefficient frames were still produced by the official encoder.

`jpeg/reconstruction_metadata_tests.rs` adds 13 focused tests: an independently
packed minimal header; U32 selector boundaries; Unknown APP/COM/payload ordering;
empty DHT entries and sentinels; component IDs and quantization mappings; DRI and
restart-fill rejection; progressive hints, extra ZRLs and mixed padding; exact
output caps; consecutive fill bounds; and mutable-field/layout errors. The
parent's 68-check oracle used the production serializer before these tests were
attached. The new unit file is formatted and ready for central compilation;
this agent has not run Cargo.

```powershell
$env:RS_JXL_JPEG_INPUTS='..\jxl_python\test_img'
cargo test --locked -p rs-jxl --test jpeg_archive -- --ignored --nocapture
```

## Remaining Work

1. Extend central release, malformed-input and independent fixture coverage for
   coefficient reading/writing. The debug and 17-file checkpoints above are
   bounded prerequisite evidence, not complete JPEG or pixel-phase acceptance.
2. Broaden metadata/profile and real-image validation after deduplication;
   preserve opaque records when prerequisites are not satisfied.
3. Measure the integrated optimized JPEG search against frozen C++. Complete
   coefficient/frame coding, cumulative search and shared auxiliary
   models now work, but legal output alone is not full source-tool acceptance.
4. Implement the runtime Rust decoder and C ABI/Python layer. The public
   byte/writer API and separate `transcode-jpeg` CLI already work; do not imply
   unrelated layers are complete or that `encode` preserves JPEG files.
