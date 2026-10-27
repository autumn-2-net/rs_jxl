# Architecture and Maintenance

This is a responsibility map and review guide, not a claim of complete source
parity, format coverage, portability, or performance. Start with the
[documentation index](README.md) for API contracts and separate status reports.
The repository's [contribution rules](../AGENTS.md) still apply.

## Entry Points and Dependencies

The encoder is a direct Rust port of the optimized libjxl source frozen at
`b866b3defec13494bfbaf705db93581cda118e37`. Runtime JXL decoding is a separate
integration of maintained official Rust decoder sources. Neither the CLI nor
the Python bridge should implement a second codec or invoke an external codec
as its runtime implementation.

| Boundary | Responsibility | Keep Outside This Boundary |
| --- | --- | --- |
| [rs-jxl](../crates/rs-jxl/src/lib.rs) | Typed images, encoding, decoding, JPEG reconstruction, framing and codec policy. Public reexports live at the crate root and in the relevant public modules. | CLI argument parsing, Python objects and filesystem policy. |
| [rs-jxl-image](../crates/rs-jxl-image/src/lib.rs) | Shared PNG/JPEG pixel input, signature detection, checked image sizes and input metadata handling. `png.rs` and `jpeg.rs` own the format-specific readers. | JXL encoding and byte-reversible JPEG transcoding. |
| [rs-jxl-cli](../crates/rs-jxl-cli/src/main.rs) | Command selection and diagnostics. `encode.rs`, `decode/`, `multiframe.rs` and `transcode.rs` adapt commands to library calls. `validators.rs` contains shared value parsers; `options.rs` resolves codec controls; `io.rs` owns file/stdio publication. | Duplicated PNG/JPEG readers or codec algorithms. |
| [rs-jxl-ffi](../crates/rs-jxl-ffi/src/lib.rs) | Versioned C request/result boundary over the Rust libraries. `abi.rs` owns result lifetimes, `request.rs` validates operation-specific request views, `error.rs` preserves typed error categories, and `contexts.rs` owns reusable handles. | Python interpreter state and Python path/stream handling. |
| [python/rs_jxl](../python/rs_jxl/) | Python-facing adapters. `_arrays.py` prepares typed sample buffers and layouts, `_io.py` handles paths/binary streams, `_ffi.py` checks the ABI and releases results, and `contexts.py`/`_context.py` expose synchronized worker reuse. | Reimplementing codec decisions or silently narrowing samples. |

The CLI and FFI depend on `rs-jxl` and `rs-jxl-image`; the image-input crate
depends on `rs-jxl` for its image representation, not the reverse. Python calls
the FFI library. This map does not imply that every Rust option has a Python or
C projection, or that wheels and runtime validation exist for every platform.
Consult [Python interface](PYTHON_INTERFACE.md), the
[C header](../crates/rs-jxl-ffi/include/rs_jxl.h) and
[request protocol](../crates/rs-jxl-ffi/protocol.md) for that boundary's contract.

## Codec Module Map

Paths in this table are under `crates/rs-jxl/src`.

| Area | Start Here | Responsibility |
| --- | --- | --- |
| Application encoding facade | [codec.rs](../crates/rs-jxl/src/codec.rs) | Typed jobs, source-policy dispatch, reports and a reusable `Encoder`. CLI and FFI map their inputs here instead of duplicating still-image effort/search selection. |
| Public pixel encoding | [image.rs](../crates/rs-jxl/src/image.rs), [encode.rs](../crates/rs-jxl/src/encode.rs), `encode/` | Typed sample/metadata validation, options, budgets, sessions and byte/writer entry points. |
| Metadata and framing | [metadata.rs](../crates/rs-jxl/src/metadata.rs), `color.rs`, `icc.rs`, `frame.rs`, `codestream.rs`, [container.rs](../crates/rs-jxl/src/container.rs) | Image metadata, color descriptions, frame headers, codestream/container serialization and frame indexes. |
| Modular encoding | [modular/mod.rs](../crates/rs-jxl/src/modular/mod.rs), `modular/encode.rs`, `layout.rs`, `orchestration.rs`, `group*.rs` | Reversible transforms, global/group layout, predictors, palettes, token generation and entropy handoff. |
| MA preparation and learning | [modular/training.rs](../crates/rs-jxl/src/modular/training.rs), `sample.rs`, `gather_samples.rs`, [ma_train.rs](../crates/rs-jxl/src/modular/ma_train.rs), `ma_*.rs`, `tree.rs` | Sample collection and quantization, constrained/adaptive trees, histogram reuse, split scoring, tree budgets and serialization. |
| Fast lossless path | [fast_lossless/mod.rs](../crates/rs-jxl/src/fast_lossless/mod.rs), `pixels.rs`, `regions.rs`, `entropy.rs` and SIMD siblings | Source-derived fast encoding routes with their own eligibility and representation checks. Not a replacement for unsupported general Modular tools. |
| Entropy and matching | [entropy/mod.rs](../crates/rs-jxl/src/entropy/mod.rs), [tools/mod.rs](../crates/rs-jxl/src/tools/mod.rs) | Hybrid integers, histograms/clustering, prefix/rANS streams, LZ77 matching/parsing, MA kernels and patch tools. |
| Search policy | [scheduler/mod.rs](../crates/rs-jxl/src/scheduler/mod.rs), `params.rs`, `routes.rs`, `driver.rs`, `native*.rs` | Candidate identities, probes, search budgets, feedback and native encoding of selected candidates. Explicit tool execution and candidate search are distinct operations. |
| Parallel execution | [execution.rs](../crates/rs-jxl/src/execution.rs), [execution/arena.rs](../crates/rs-jxl/src/execution/arena.rs), `scheduler/worker_pool.rs` | Caller-owned worker resources, serial/null execution and bounded cooperative nested work. Codec code uses these helpers instead of creating independent nested pools. |
| Frames and incremental output | [multiframe.rs](../crates/rs-jxl/src/multiframe.rs), `multiframe/`, [layers.rs](../crates/rs-jxl/src/layers.rs), [streaming.rs](../crates/rs-jxl/src/streaming.rs), [region_input.rs](../crates/rs-jxl/src/region_input.rs) | Difference/crop planning, explicit heterogeneous-size layers, frame references and timing, region loans, sequence/seekable output and TOC/index lifecycles. |
| Byte-reversible JPEG | [jpeg/mod.rs](../crates/rs-jxl/src/jpeg/mod.rs), `reader.rs`, `coefficient_reader.rs`, `transcode.rs`, `archival.rs`, `reconstruction*.rs`, `scan_writer*.rs` | JPEG structure and coefficients, reconstruction metadata, JXL archival frames and recovery of original JPEG bytes. Shares Modular/entropy tools without using a pixel fallback. |
| Runtime JXL decoding | [decode/mod.rs](../crates/rs-jxl/src/decode/mod.rs), `engine.rs`, `native.rs`, `stream.rs`, `jpeg.rs`, `color*.rs` | Backend adaptation, limits, native samples, streaming state, JPEG coefficient capture/reconstruction and explicit rendered conversions. Backend implementation is in `vendor/jxl-rs`. |
| Experimental transcripts | [final_plan.rs](../crates/rs-jxl/src/final_plan.rs), `final_plan/` | Transcript schemas and wire helpers. Their presence is not evidence that ordinary codec capture/replay is connected. |

### Unified Jobs and Reusable Resources

`codec::EncodeJob` combines `EncodeOptions` with an optional `SourceRequest`.
With `source: None`, it executes the explicit manual Modular options. A source
request carries `CandidateParams`, `NativeBackendOptions` and a typed
`SourceSearch`: `Disabled`, `Auto`, `Scout` or `Pilot(PilotOptions)`. `Disabled`
executes one candidate; `Auto` selects the pixel search or the separate effort-11
Tectonic tournament in the core. Effort 11 without search remains the explicit
tier -1 backend. No search is implicitly enabled by creating a context.
Adapters still own their argument compatibility gates and diagnostics.

`codec::encode_job` and `codec::transcode_jpeg` are one-shot facade functions.
`codec::Encoder::new(threads)` retains worker resources for repeated `encode`,
`encode_frames`, `encode_layers`, `encode_auto`, `analyze_multiframe` and `transcode_jpeg` calls. JPEG search is an
explicit `Option<SearchLimits>`, separate from pixel search. The CLI's automatic
multiframe route also uses `Encoder::encode_auto`; its strategy reports and
output publication remain adapter responsibilities.

`multiframe::analyze` and the context method share the packed difference maps,
reference choice and candidate geometry builders with multiframe encoding.
They report geometry, not estimated encoded bytes; no crop buffers or encoding
trials are created. The FFI serializer is a separate `analysis.rs` module, and
Python remains transport-only. See [analysis contract](MULTIFRAME_ANALYSIS.md).

`decode::Decoder::new(DecodeOptions)` separately retains decoding workers for
complete independent inputs, selected frames and original-JPEG reconstruction.
`set_options` validates per-input settings without rebuilding the pool; a thread
count change requires a new context. Core contexts accept 1..=256 threads; one
thread uses the caller without a pool. They require exclusive mutable access.
Input, parser, frame/reference and candidate-cache state is invocation-local,
including after an ordinary error; limits are per input, not cumulative.
Independent contexts can run concurrently, subject to the caller's total worker
budget. Reuse is not incremental parsing or a persistent frame-reference stream.

The decoder boundary is not fully independent of upstream types:
`decode::JxlBasicInfo`, `JxlFrameHeader`, color/extra-channel types and other
public names remain reexports from the maintained `jxl_decoder::api`.
`StreamDecoder` is a separate Rust interface; Python does not yet expose the
complete incremental streaming contract.

### Reports and Binding Lifetimes

`EncodeResult` contains bytes and an `EncodeReport`: route, implementation,
pixel/byte counts, full/probe encode counts, optional winning/effective parameters,
preparation reuse and typed `SearchDetails` (`None`, `Pixel`, `Tectonic`). The
facade also returns JPEG trial/winner information. `codec_seconds` measures codec
work, excluding context/pool creation, external pixel-input decoding and
transport/I/O. JPEG parsing and coefficient preparation are included in JPEG
transcode timing. It is not an exact analysis-versus-backend timing split.
Still-image `bpp` and
`mpix_per_second` derive from the final bytes, original pixel count and that timer;
they are not throughput summed over all candidate pixels. The FFI serializes a
subset under `meta["report"]` for still/JPEG encoding; it does not serialize the
entire Rust search-detail enum. Multiframe strategy metadata remains separate.
JSON still-image `implementation` uses stable `modular`/`fast_lossless` values;
`effective_parameters.patches` uses `auto`/`on`/`off`, not Rust Debug spellings.
`effective_parameters` projects scheduler `CandidateParams`, not a full backend
recipe or exact tool list; FJXL uses its own default effort.

Rust source routes preserve `EncodeReport.artifact: Option<NativeArtifact>`,
`adaptations: Vec<NativeAdaptation>` and `progressive: Option<ProgressiveReport>`.
The artifact contains the winning source's resolved options, gaps and resource
limits; manual jobs have no source artifact. Adaptations retain the search's
diagnostics, including probes and losing trials; direct candidates expose their
diagnostics through the artifact instead. Progressive diagnostics are present
when the pilot completes a real-group race. Representation diagnostics requested
through `record_representations` are retained in the Rust diagnostics. The FFI
JSON subset is unchanged; these are not additional Python report fields.

The C ABI remains version 1 with the existing request layout and one-shot entry
point. Optional additive `rs_jxl_context_create`, `rs_jxl_context_execute` and
`rs_jxl_context_free` symbols expose reusable handles. Callers must serialize
execute/free for each handle; result ownership is independent of context lifetime.
The JSON v1 request remains compatible. `ValidatedRequest` gives operation-specific
views after wire validation, but still borrows the shared option structure;
buffer and codec-specific validation also occurs in the execution adapters.
Typed FFI errors preserve `error` and add `error_kind` and optional `error_stage`;
known Rust error sources retain their category rather than relying on message
matching. Unclassified/string-only errors still default to `invalid_input`.

Python `Encoder(threads=...)` and `Decoder(threads=...)` serialize same-context
native execution against `close()` and support context-manager use. The lock
protects the closed check and native call, not Python result conversion or
copying: it is released before the result is yielded to conversion code.
Independent result ownership permits conversion callbacks to close or reenter
the context without holding that lock. Context-bound PNG/JPEG `read_image` calls
also use the closed/lifecycle guard, while remaining stateless and not reusing
JXL decoder workers. Threads are fixed; other options use per-call defaults.
Mutable encoding inputs are snapshotted by
default. Explicit `borrow_inputs=True` is best-effort and forbids concurrent
mutation through any alias during the call; layout/endian normalization and
multi-buffer consolidation can still copy. Files and streams remain fully
buffered. Decode results are owned arrays/tensors/bytes copied while the Rust
result is alive, or copied into a validated preallocated NumPy destination by
`decode_into`. Public decode return values do not borrow Rust storage. For
nonempty data results, `_ffi._result_buffer` attaches an `ffi.gc` owner to the
data pointer; `ffi.buffer` retains it. Temporary derived views, including ones
retained by an exception traceback, keep the result alive beyond helper scope
exit. The owner's finalizer releases the result after those references are gone;
scope exit releases the helper view, not necessarily the native allocation.
Native error responses and empty-data results still free directly in `finally`.
This avoids intermediate Python copies where possible, not all Rust allocation
or codec scratch.
See [Python API](PYTHON_API.md) and [boundary contract](PYTHON_INTERFACE.md).

### Extracted Implementation Modules

These are physical file boundaries, not new public API namespaces:

- [ma_train_options.rs](../crates/rs-jxl/src/modular/ma_train_options.rs)
  holds `TrainOptions`, `HistogramOptions`, `MaBudgetMode`, defaults and
  validation. `ma_train.rs` reexports the public types and retains the learner
  and histogram-cache orchestration. Internal methods retain their original
  visibility scope; constraint validation stays private to the options module.
- [ma_train_tests.rs](../crates/rs-jxl/src/modular/ma_train_tests.rs)
  remains the `modular::ma_train::tests` module through `#[path]`. Other focused
  MA test modules retain their own names.
- [ma_simd.rs](../crates/rs-jxl/src/tools/ma_simd.rs) retains safe interfaces,
  scalar behavior and dispatch. [ma_simd_x86.rs](../crates/rs-jxl/src/tools/ma_simd_x86.rs)
  and [ma_simd_count_x86.rs](../crates/rs-jxl/src/tools/ma_simd_count_x86.rs)
  isolate architecture-specific kernels. [ma_simd_tests.rs](../crates/rs-jxl/src/tools/ma_simd_tests.rs)
  remains `tools::ma_simd::tests`; optional AVX512/reciprocal tests are separate.
- [lz77_source_tests.rs](../crates/rs-jxl/src/tools/lz77_source_tests.rs)
  remains `tools::lz77_source::tests`. Matching, indexing, parsing and execution
  helpers stay in the existing `lz77_source_*` implementation modules.
- CLI still-image arguments and dispatch live in
  [encode.rs](../crates/rs-jxl-cli/src/encode.rs), shared parsers in
  [validators.rs](../crates/rs-jxl-cli/src/validators.rs), and the extracted
  root tests in [cli_tests.rs](../crates/rs-jxl-cli/src/cli_tests.rs).

Follow the declaring `mod`/`#[path]` and reexports when changing visibility or
test filters. A sibling filename alone does not establish a Rust module path.

## Invariants to Preserve

1. **Keep the two lossless contracts separate.** Pixel encoding must recover
   original decoded samples, including RGB under zero alpha, extra planes and
   frame timing/reference semantics. JPEG passed to `rs-jxl-image` is decoded
   pixel input: losslessness is relative to that decoder's samples. Only the
   JPEG transcode/reconstruction route promises original JPEG bytes within its
   supported domain. Re-encoding pixels or archiving a private payload is not
   a substitute for that route; unsupported modes must remain explicit.
2. **Preserve sample representation.** Native 17..31-bit integer recovery must
   not pass through floating-point storage. Raw float-word checks differ from
   rendered/color-converted output checks. Preserve layout, bit depth, extra
   channel order, alpha association and metadata policy across image, CLI and
   FFI boundaries. Do not silently color-convert, orient, strip metadata or
   narrow a sample type to make an adapter accept an input.
3. **Do not change algorithms during extraction.** Preserve predictor/property
   order, ties, split comparisons, arithmetic widths, signed conversions,
   wrapping/checked operations, histogram mass, clipping, lane/reduction order
   and failure order. FMA, clipped costs and Shannon/count-log experiments are
   not interchangeable. Keep experimental flags, defaults and documented
   limitations even when they are inconvenient to the new module boundary.
4. **Separate limits from policy.** Keep wire/decoder-compatibility bounds,
   representability checks, allocation failures and explicit caller budgets.
   Do not restore hidden migration-era caps or silently reduce sampling/effort
   on exhaustion. MA strict/bounded modes remain distinct; required partitions
   must fit in either mode. Cache capacity is not a leaf-search budget. See
   [resource limits](RESOURCE_LIMIT_AUDIT.md) and [MA budgets](MA_TREE_BUDGET.md).
5. **Preserve ownership and deterministic decisions.** Parallel completion
   order must not choose codec policy. Borrowed sample columns and scratch
   leases stay disjoint; active work must finish before owners release storage,
   including error/panic paths. Retain serial/null-runner behavior, runtime SIMD
   feature checks and scalar fallbacks. A configured switch alone does not prove
   that an experimental execution path actually ran.
6. **Respect transport lifetimes.** Streaming input distinguishes need-more-data
   from EOF/truncation; an early frame result is not validation of the entire
   file. Writer APIs have different preflight and partial-output contracts;
   do not invent a blanket transactional-write guarantee. At the C boundary,
   validate ABI version/size and buffer ranges; inputs are borrowed for the
   synchronous call and result buffers live until exactly one free. Python
   returns owned copies; internal data views retain the result owner, including
   across exception tracebacks, until those views can safely be released.
   Panic containment cannot validate dangling C
   pointers or recover abort/OOM failures. Keep header, protocol and CFFI
   declarations synchronized.
7. **Keep acceptance separate from compressed identity.** Whole-file JXL hash
   equality against another encoder or a previous output is not an optimization
   goal. Compare recovered samples/frames or original JPEG bytes, then measure
   size and time separately. Primitive bit vectors, token order and controlled
   deterministic-route checks remain useful regression tests. A pure extraction
   must not use the freedom to emit different valid JXL bytes as permission to
   change an algorithm.

## Testing and Review Guide

Scale verification to the changed contract. In coordinated work, the parent
runs builds and tests centrally; file owners report focused targets, necessary
fixtures and checks they did not run. The following is a selection guide, not a
record that any suite passed.

The facade/context validation checkpoint is maintained under
[current verification](PYTHON_API.md#current-verification). It distinguishes the
full workspace run, focused review-fix tests, native Python checks, the pure
derived-view lifetime check and tests against an independently installed wheel.
These results do not establish new performance or cross-platform runtime claims;
historical checkpoints remain separate.

| Change Surface | Coverage to Select | Review Focus |
| --- | --- | --- |
| File/module extraction | Original inline tests through their retained qualified module names; relevant focused sibling tests. | Compare moved bodies and defaults, public paths, minimal visibility, `cfg`/target gates, macros, imports and source notices. Format only owned Rust files with `skip_children=true` during shared work. |
| MA/SIMD/LZ77 | `ma_train_tests.rs`, cache/budget/pruning/handoff tests, `ma_simd*_tests.rs`, `lz77_source*_tests.rs`. | Boundary populations, tails, saturation, error ordering, scalar/native arithmetic and enabled/disabled experiments. Do not equate approximate scoring profiles merely because both encode losslessly. |
| Pixel API and sample types | [pixel_acceptance.rs](../crates/rs-jxl/tests/pixel_acceptance.rs), [decode_lossless_edges.rs](../crates/rs-jxl/tests/decode_lossless_edges.rs), [decode_wide_integer.rs](../crates/rs-jxl/tests/decode_wide_integer.rs), explicit float/oracle targets. | Invisible RGB, exact integer values, float words where supported, extra channels, metadata and writer failures. An f32 NPY projection cannot prove arbitrary U31 equality. |
| Frames and streams | [multiframe_auto.rs](../crates/rs-jxl/tests/multiframe_auto.rs), [decode_streaming.rs](../crates/rs-jxl/tests/decode_streaming.rs), sequence/metadata reference tests. | Zero-duration layers, visible-frame indexing, crop/blend/reference state, timing, fragmented input, EOF, trailing metadata and partial outputs. |
| JPEG archival | [transcode_reference_tests.rs](../crates/rs-jxl/src/jpeg/transcode_reference_tests.rs), [decode_jpeg.rs](../crates/rs-jxl/tests/decode_jpeg.rs), coefficient/reconstruction tests. | Compare every reconstructed JPEG byte. Parser or coefficient-to-JPEG helper tests alone do not cover an entire JPEG-to-JXL round trip. |
| Budgets and execution | [default_limits.rs](../crates/rs-jxl/tests/default_limits.rs), [decode_hostile.rs](../crates/rs-jxl/tests/decode_hostile.rs), arena/worker tests, [parallel_execution_reference.rs](../crates/rs-jxl/tests/parallel_execution_reference.rs). | Explicit tiny limits, checked sizes, no hidden truncation, cancellation and borrowed-work cleanup; verify actual dispatch as well as output. |
| Adapters and bindings | Shared image-reader tests, CLI parser/acceptance/decode-writer tests, FFI and Python boundary coverage for the changed operations. | Metadata rejection/stripping, consistent options, invalid requests, ABI mismatch, buffer ownership, endian/layout conversion and short stream writes. Missing boundary coverage must be reported, not inferred from core tests. |

Independent official `djxl` is a development oracle only. Many reference tests
are ignored by default and require explicit executable/fixture paths; read each
target's header and ignore reason before selecting it. Do not enable every
ignored test indiscriminately: some are internal subprocess entry points.
Missing fixtures, skipped tests, timeouts and interrupted runs are not passes.
Keep native round trips distinct from independent-decoder evidence.

Review handoff should identify changed paths, preserved public and behavioral
contracts, any intentional differences, exact checks executed, and remaining
risks. Record build/runtime hosts, toolchain, feature configuration and fixture
scope with validation evidence. A compile check is not runtime validation;
x86_64 Windows evidence does not certify Linux, macOS, ARM or the declared MSRV.
[Platform validation](PLATFORM_VALIDATION.md) records the separate evidence
levels and restrictions. For performance work, retain corpus, options, worker
counts, repetitions and correctness results; a kernel measurement does not
establish an end-to-end improvement.

## Source Provenance

- Preserve translated-code copyright headers and frozen-source mappings when
  moving files. [Migration](MIGRATION.md) maps source families;
  [ported kernels](PORTING_KERNELS.md) retains the upstream BSD notice. That
  attribution does not implicitly license unrelated new project code.
- [vendor/jxl-rs/PROVENANCE.md](../vendor/jxl-rs/PROVENANCE.md) records the
  pinned official decoder revision and local patch ledger. Keep upstream
  AUTHORS, LICENSE and PATENTS notices. Maintained decoder behavior must not be
  presented as an unmodified-upstream result.
- [vendor/libsais-rs/PROVENANCE.md](../vendor/libsais-rs/PROVENANCE.md) records
  the separately licensed pure-Rust suffix-index utility and local changes.
  It is not a codec replacement and must not be relicensed under the caller's
  libjxl attribution.

Keep vendored changes separately owned and auditable. A general module cleanup
is not authorization to rewrite vendored source or erase its patch history.
