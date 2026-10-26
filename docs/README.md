# Documentation Index

Start with the [project README](../README.md) for entry points and
[architecture and maintenance](ARCHITECTURE.md) for module ownership, invariants
and review/test selection. This index separates API guidance from source-port
ledgers and dated evidence; it does not certify feature completeness,
portability or performance.

## Reading Order

1. Choose the API/behavioral contract below, then inspect its current public
   definitions and tests when changing code.
2. Consult the relevant status ledger for gaps, deliberate differences and
   source provenance. These documents often accumulate historical checkpoints;
   an old statement that a feature is pending is not proof about today's code.
3. Use dated reports to understand a particular experiment, environment and
   corpus. Do not promote their timing, coverage or source-parity conclusions
   into a current guarantee without new evidence.

Pixel losslessness means recovery of decoded samples, including invisible RGB
and the specified extra-channel/frame semantics. JPEG decoded as pixel input is
not byte-reversible JPEG transcoding. That separate route must reconstruct the
original JPEG bytes. Equality of compressed JXL files or their hashes is not an
optimization goal; primitive/wire regression checks remain valid.

## API and Behavioral References

These are the starting points for using or changing current interfaces. Some
also contain historical verification sections; those sections are not API or
platform certifications.

| Topic | References |
| --- | --- |
| Rust pixel encoding, images and tools | [Pixel API](PIXEL_API.md), [integer and float paths](INTEGER_AND_FLOAT_PATHS.md), [intrinsic dimensions](INTRINSIC_DIMENSIONS.md) |
| Shared PNG/JPEG pixel input | [rs-jxl-image](../crates/rs-jxl-image/src/lib.rs), [CLI encode adapter](../crates/rs-jxl-cli/src/encode.rs); use the JPEG API below when original file bytes must be recoverable |
| Runtime JXL decoding and output semantics | [Maintained decoder API/status](DECODER_STATUS.md), [exact native integer decoding](NATIVE_INTEGER_DECODE.md) |
| Incremental decoding | [Streaming input](STREAMING_INPUT.md) |
| Byte-reversible JPEG | [JPEG API](JPEG_API.md), [compatibility and supported-domain distinctions](JPEG_COMPATIBILITY.md) |
| Frame planning, layers and animation | [Automatic frames](MULTIFRAME_AUTO.md), [multiframe CLI](MULTIFRAME_AUTO_CLI.md), [frame-aware search](MULTIFRAME_SEARCH.md) |
| Incremental encoding and indexes | [Streaming output](STREAMING_OUTPUT.md), [frame index lifecycle](FRAME_INDEX.md) |
| Python and the versioned C boundary | [Python API and packaging](PYTHON_API.md), [Python interface](PYTHON_INTERFACE.md), [C ABI protocol](../crates/rs-jxl-ffi/protocol.md), [C header](../crates/rs-jxl-ffi/include/rs_jxl.h) |
| MA limits and exact search pruning | [Tree budgets](MA_TREE_BUDGET.md), [search pruning](MA_SEARCH_PRUNING.md) |
| Search/probe integration and execution | [Probe API](SCHEDULER_PROBE_API.md), [cooperative execution](PARALLEL_EXECUTION.md), [progressive group scheduler](PROGRESSIVE_GROUP_SCHEDULER.md), [scheduler default routes](../crates/rs-jxl/src/scheduler/DEFAULT_ROUTES.md) |

## Acceptance and Validation

- [Complete pixel-lossless port status](PIXEL_LOSSLESS_PORT_STATUS.md): coverage
  and open acceptance gates, not a completion claim implied by its title.
- [Pixel acceptance audit](PIXEL_ACCEPTANCE_AUDIT.md): sample/metadata contracts,
  regressions and limitations of independent decoder evidence.
- [Resource-limit audit](RESOURCE_LIMIT_AUDIT.md): removed private ceilings,
  retained format/arithmetic checks and explicit caller budgets. Historical
  examples may use older limits; do not restore them as silent defaults.
- [Platform and hostile-input validation](PLATFORM_VALIDATION.md): evidence
  levels, host/toolchain requirements and stopped/out-of-scope work. Preparation
  or cross-compilation is not a platform runtime pass.
- [Port verification and profiling](PORT_VERIFICATION.md): cumulative historical
  checkpoints and reproduction details, not a test result for the current tree.

See the [architecture testing guide](ARCHITECTURE.md#testing-and-review-guide)
for selecting focused tests, independent oracles and adapter checks. Ordinary
workspace runs do not automatically execute ignored external-oracle tests.

## Source Maps and Provenance

- [Direct source migration](MIGRATION.md): frozen encoder source and module
  mapping; includes historical integration and acceptance planning.
- [Ported encoder kernels](PORTING_KERNELS.md): arithmetic/source mapping and
  retained upstream BSD license notice.
- [Frozen C++ reference build](CPP_REFERENCE_BUILD.md): development reference
  setup, not a runtime dependency or permission to modify the old project.
- [Maintained Rust decoder provenance](../vendor/jxl-rs/PROVENANCE.md): pinned
  upstream import, retained notices and local patch ledger.
- [Suffix-index utility provenance](../vendor/libsais-rs/PROVENANCE.md): the
  separately licensed vendored `libsais-rs` utility and documented integration.

Preserve source attribution and vendor licenses/patch histories during moves.
An encoder port notice is not a blanket license for all new repository code.

## Port and Design Ledgers

These explain source mappings, implementation decisions and checkpoint gaps.
Read their dates and integration notes alongside current code; neither a
prerequisite helper nor a successful checkpoint proves full port completion.

| Area | Ledgers and Design Notes |
| --- | --- |
| Early pixel bring-up | [Minimal bitstream milestone](PIXEL_LOSSLESS_MILESTONE.md), explicitly historical and not phase acceptance; start with the current Pixel API for usage |
| Fast path and reversible tools | [FJXL](FJXL_PORT_STATUS.md), [palette](PALETTE_PORT_STATUS.md), [entropy](ENTROPY_PORT_STATUS.md), [LZ77](LZ77_PORT_STATUS.md) |
| MA preparation and arithmetic | [Gather](MA_GATHER_PORT_STATUS.md), [learner source audit](MA_PORT_GAPS.md), [default scoring alignment](MA_SCORING_ALIGNMENT.md), [AVX2+FMA](MA_FMA_PORT.md) |
| Search and transcripts | [Scheduler port status](../crates/rs-jxl/src/scheduler/PORT_STATUS.md), [asynchronous prefix screen](ASYNC_PROBE_PORT.md), [FinalPlan transcript checkpoint](FINAL_PLAN_PORT_STATUS.md) |
| Architecture-specific work | [NEON checkpoint](NEON_PORT_STATUS.md); retained code and source mapping do not establish ARM runtime acceptance |
| JPEG archival integration | [JPEG tool port](JPEG_TOOL_PORT_STATUS.md), [transcode layout](JPEG_TRANSCODE_LAYOUT.md), [cooperative MA handoff](JPEG_COOPERATIVE_MA.md), [auxiliary stream policy](JPEG_MODULAR_POLICY.md) |
| JPEG coefficient pipeline | [Quantization](JPEG_QUANTIZATION.md), [DC/control streams](JPEG_DC_STREAMS.md), [AC tokens](JPEG_AC_TOKENS.md), [coefficient order](JPEG_COEFFICIENT_ORDER.md), [chroma from luma](JPEG_CFL.md) |
| JPEG metadata and decoder adoption | [JPEG color metadata](JPEG_COLOR_METADATA.md), [pre-integration decoder backend audit](DECODER_BACKEND_AUDIT.md) |

## Dated Experiments and Reports

These are historical evidence, including reports whose titles say "current".
Results belong to their recorded revisions, fixtures, hardware, options and
measurement methods. A retained experimental switch is not a universal speedup;
rejected experiments and remaining limitations remain part of the record.

### Encoding and Search

- [Performance baseline, 2026-10-24](PERFORMANCE_BASELINE_20261024.md)
- [Codec optimization experiments, 2026-10-25](CODEC_EXPERIMENTS_20261025.md)
- [Encoder corpus run, 2026-10-25](ENCODER_CORPUS_20261025.md)
- [Encoder scout experiments, 2026-10-25](ENCODER_SCOUT_20261025.md)
- [Encoder kernel work, 2026-10-26](ENCODER_KERNELS_20261026.md)
- [Archived random-500 replay, 2026-10-26](ENCODER_RANDOM500_20261026.md)
- [Group-race checkpoint/sample, 2026-10-26](GROUP_RACE_SAMPLE_20261026.md)
- [Scheduler palette audit, 2026-10-26](SCHEDULER_PALETTE_AUDIT_20261026.md)
- [MA small-histogram sweep kernels, 2026-10-26](MA_SWEEP_KERNELS_20261026.md)
- [MA tree execution experiments, 2026-10-26](MA_TREE_EXPERIMENTS_20261026.md)

### Decoding and Reconstruction

- [JPEG reconstruction optimization, 2026-10-24](DECODER_PERFORMANCE_20261024.md)
- [Parallel JPEG reconstruction, 2026-10-24](DECODER_PARALLEL_20261024.md)
- [Single-thread/pixel decoder follow-up, 2026-10-24](DECODER_SINGLE_20261024.md)
- [Multithreaded pixel decode, 2026-10-25](DECODER_THREADS_20261025.md)
- [Decoder SIMD experiments, 2026-10-25](DECODER_SIMD_20261025.md)
- [Decoder entropy experiments, 2026-10-25](DECODER_ENTROPY_20261025.md)
- [JPEG-transcoded JXL pixel decoding, 2026-10-25](JPEG_PIXEL_DECODE_20261025.md)

Keep rendered pixel comparisons distinct from byte-exact JPEG reconstruction
and native lossless sample checks when interpreting these reports. None of this
index's links establishes complete upstream conformance, cross-platform runtime
coverage or a performance guarantee.
