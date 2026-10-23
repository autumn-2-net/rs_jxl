# Complete Pixel-Lossless Port Status

The first acceptance phase is the complete pixel-lossless encoder, including
our optimized tools and scheduling. **It is still in progress.** Legal JXL
output alone is not acceptance. JPEG byte-transcode prerequisite tools may now
be ported in parallel by user request; neither acceptance phase is complete.

Source: optimized libjxl `b866b3defec13494bfbaf705db93581cda118e37`.
Official djxl is used only for independent testing. No C++ linkage, runtime
encoder subprocess or substitute Rust JXL encoder is present.

Current execution priority is x86_64 plus portable Rust. At the user's request,
no further ARM-specific kernels, optimization or validation are being developed.
Existing ARM code is retained, without claiming hardware execution coverage.

## Active Coverage

| Area | Active Code | Remaining Work / Evidence |
| --- | --- | --- |
| Framing | Modular groups, Level 5/10, explicit animation/layers/references | Full combinations and resource audit |
| Samples | U8/U16/U32, F16/F32, explicit depths; typed integer/float working planes | Custom floats, special float words and adaptive tools verified; arbitrary U31 oracle precision remains limited; see INTEGER_AND_FLOAT_PATHS.md |
| Entropy | Prefix/RLE and rANS, normalized histograms, clustering, context-map RLE, native cost SIMD, bounded config workers, retained winners and fixed/incremental model lifecycle | Broader hostile-input coverage; non-x86-specific work deferred; see ENTROPY_PORT_STATUS.md |
| LZ77 | RLE/greedy/lazy/optimal, source b1/b3/Optc8/Optc256, indexed/block/suffix routes, frontier/cache/price controls and seed reuse; bounded serial pure-Rust SA-IS; cooperative stream roots sharing nested workers | Diagnostic output is not a codec algorithm; see LZ77_PORT_STATUS.md and the vendored utility provenance |
| Prediction | All 14 predictors, five weighted presets, automatic costs, native row SIMD | More high-depth and non-x86 execution evidence |
| MA | Direct compact-column handoff, sampling/quantiles, channel/wave gather, batching, adaptive search, histogram reuse/subtraction, retained worker scratch, physical column partitioning, paired gather and fused f32 SIMD scoring; bounded cooperative channel/row roots and whole-subtree jobs | Source gates, disjoint sample loans and actual activation have focused tests; broader performance evidence remains separate; see MA_PORT_GAPS.md |
| RCT | All 42 transforms and source automatic costs; in-place row-parallel scalar/SSE2/AVX2 kernels and fused wrapping-i32 cost scans | Non-x86 execution evidence; isolated cost-kernel speedup is not whole-encoder speedup |
| Palettes | Global/per-channel eligibility/costs, absolute/exact-delta/zero-delta, ordering/geometry experiments and bounded direct LUT | Optional ordering and hidden-RGB cases passed both decoders; larger real cases remain |
| Squeeze | Explicit local passes, global default schedule and progressive DC/AC spatial passes; separate source responsive candidate | Responsive source matrix passed for 8/16-bit Gray/RGB/RGBA, 1/4 threads and tiers 0/1/4/7/9; local scripts are not source responsive |
| Patches | Exact Replace atlas and signed source Add; source dictionary and deferred atlas policy; exact decoder-arithmetic guard | Mixed-depth/subsampled extras and <=16-bit float Add passed both official decoders; large real-image performance pending |
| Shared models | Global/local streams, group models and independent frame-area models | Integer progressive/frame-area checks passed; further high-depth combinations |
| Scheduler | Source planning/probes/refinement/racing, retained output, frame-area/atlas routing, batched probes/confirmation, preflight/context/calibration/scope routes, optional frontier tournaments, complete effort-11 tournament, cooperative/LPT dispatch, live prefix-cutoff public helper | Native and new Fn+Sync helpers use live cutoff; legacy FnMut batch callback keeps completed-wave semantics. Environment parsing/diagnostic files are explicit API differences, not missing search routes |
| Metadata | Color/ICC, orientation, extra types/names/depth and planar data; subsampled extras and CLI color preservation | Nonconstant subsampled multiframe oracle checks passed; broader real-profile corpus pending |
| ICC | Prediction and four-way entropy competition with LZ | Larger real-profile corpus |
| API | Memory/Write output, multiframe API, previous/first/smallest-crop planning, declared-word comparison, ordered streaming and independent-frame indexes; push FrameEncoder accepts successive region providers, crops/references/layers and bounded indexes | Frame-area eligibility follows source; ineligible frames require explicit current-frame buffering or fail before acquisition. See STREAMING_OUTPUT.md. Limits are not a process-wide memory sandbox |

See [scheduler status](../crates/rs-jxl/src/scheduler/PORT_STATUS.md) for its
specific remaining routes. A proxy score must not be reused as an exact-source
candidate score. Unsupported requested features must fail explicitly.

## Verified Checkpoints

2026-10-23, Windows x86_64. Checkpoints were run centrally, not inferred from
agent implementation reports. Changes after a checkpoint require revalidation.

The requested parallel execution, x86 fast kernels, scheduler-detail and push
streaming work blocks are now implemented, connected and verified. The final
combined checkpoint passed **1,527 ordinary tests in both debug and optimized
validation**, strict Clippy, formatting, Rust 1.89 compilation and x86_64 Linux
cross-compilation. Official djxl 0.12/0.13 passed the complete 31-suite main
matrix, 28 cooperative-execution comparisons, 108 FJXL pixel checks, 32 typed
F16 checks and 20 new push-sequence comparisons. Eight diverse real e9 full-search
images and a real JPEG archival CLI case passed independently as well. Exact
scope, bpp/timing and remaining performance/platform acceptance limits are in
PORT_VERIFICATION.md. These four blocks are not left as unconnected helpers.

The preceding x86 follow-up passed 1,452 ordinary tests in debug and optimized
validation (1,450 workspace tests plus two new oracle-helper tests). Official
djxl 0.12/0.13 passed 108 FJXL pixel checks, 32 typed F16-word checks, 172 LZ
checks, 16 actively traced channel/row-tree pixel checks and 12 cooperative
JPEG original-byte checks. See PORT_VERIFICATION.md for exact scope and remaining
execution gaps. This is still not complete-phase acceptance.

Latest MA/LZ addition: 180 affected-path official pixel comparisons passed
(80 source tiers, 68 LZ routes, 16 predictor/property dedup, 16 parallel gather).
Shared JPEG auxiliary modeling and its byte/writer API passed 696 original-byte
checks; the separate release CLI passed another 44. These are distinct from
pixel acceptance. The initial integration workspace passed 1,137 core tests;
see PORT_VERIFICATION.md for subsequent API error-classification validation.

The 869-core checkpoint passed debug and release core tests, 10 API tests,
14 explicit pixel/resource acceptance tests and 35 CLI tests. All 56 release
oracle suites passed both official djxl versions: 17 main, 16 float, three
metadata/frame, two integer Add, one floating Add, 11 scheduler, five palette
and one metadata-fields suite. This checkpoint includes channel-token
parallelism, the added index/block/suffix LZ routes, palette order/geometry,
MA scratch/physical partitioning and the JPEG MCU reader. Later trial-budget,
MA cache-fusion and JPEG-writer work requires its own validation.

Eight diverse original images also passed full e9 scheduling and exact sample
comparison through both official decoders. A separate warmed two-image Rust
before/after run measured 1.119x and 1.146x throughput with essentially unchanged
sizes. See PORT_VERIFICATION.md for measurements, restrictions and artifact IDs.
These results do not close remaining source-tool or performance acceptance gaps.

The earlier integration checkpoint passed 741 debug and release core tests (the opt-in
microbenchmark is separate), 10 API tests, 35 CLI tests, ordinary integration
helpers and the doctest. Strict all-target Clippy and formatting checks passed.
This includes the index/SIMD and direct entropy-writer integration. The 11 new
scheduler oracle suites passed both official decoders, including compact
cooperative dispatch and the complete effort-11 tournament. Mixed-extra and
floating Add-patch decoding passed separately. After integration, all 48
official-decoder suites were rerun in release mode with both official versions:
15 main, 16 float, three metadata/frame, two integer Add, one floating Add and
11 scheduler suites. All passed. The comparison harness's 27 unit tests passed.
Complete acceptance is not being inferred from these counts.

Two real-image comparisons against an isolated frozen-source C++ build passed
both official decoders. Full-scheduler sizes are close on these inputs, but
Rust CLI latency remains about 1.7--2.1 times C++. This is one cold measurement
per case, not a representative benchmark or an integer/SIMD speedup claim.
See PORT_VERIFICATION.md and CPP_REFERENCE_BUILD.md. A later warmed Rust
before/after checkpoint measured 1.004x--1.063x throughput on these two images;
the smallest differences may be noise. This does not close the C++ performance
gap or establish a general speedup.

The preceding batching/null-runner checkpoint passed 594 debug/release core tests,
10 API and 34 CLI tests, helpers and doctest, plus strict all-target Clippy.
Both official djxl versions passed all 35 oracle suites: 15 main (including
120 responsive source-candidate comparisons), 16 float, two metadata/frame,
and two Add-patch suites. See PORT_VERIFICATION.md for the separate release
checkpoint and the explicitly Rust-before/after timing comparison.

Preceding default-MA correction: 578 debug/release core tests, 10 API and 34 CLI tests,
helpers and doctest passed; strict all-target Clippy passed. All 34 official
oracle suites below were rerun after that change and passed with both decoders.
The earlier 520-core checkpoint remains historical. See the real-image MA
comparison in [MA_SCORING_ALIGNMENT.md](MA_SCORING_ALIGNMENT.md).

- Debug and release core unit tests: 520 passed; encoding API: 10 passed;
  CLI: 34 passed. Strict all-target Clippy passed.
- Official djxl 0.12 and 0.13: all 14 main suites, 16 float suites, two Add-patch
  suites and two metadata/frame suites passed after the routing and local
  learner-ID fixes. Ignored tests were explicitly enabled for this run.
- Basic layout/depth matrix: 412 independent pixel comparisons.
- Forced-tool matrix: 1,240 comparisons, including source palette/LZ policies,
  all RCTs and predictors, weighted presets, shared models and local/global tools.
- Replace patch plans: 64 comparisons, including hidden RGB and 16-bit samples.
- Source tiers: 80 comparisons; progressive layouts: 64; grouped MA: 24.
- Channel/wave source gather: 16 comparisons. Native default routing boundary:
  24 comparisons, covering 8/9 groups, forced patches, 8/16-bit RGBA and 1/4 threads.
- Source Add patches: manual 96 color and 96 extra-plane checks, plus native
  policy projection and actual-atlas reference configuration checks.
- ICC bytes, extra planes, eight orientations, animation, zero-duration layers,
  reference-only frames and automatic first/previous/crop planning passed.
- Float checks passed for F16/F32, custom formats, NaN payloads, infinities,
  signed zero, all predictors, weighted presets, adaptive MA and custom trees
  with odd multipliers/wide residuals. Official NPY's f32 projection cannot prove
  arbitrary U31 recovery; those checks explicitly verify source projection only.
- Metadata/frame regression: all seven tests passed, including 1,024 official
  comparisons of nonconstant subsampled extras, 8/16-bit and float samples,
  all four automatic reference policies and mixed zero/nonzero durations.
- Pre-transform references are rejected as blend backgrounds; replacement of
  reference slots updates their stage. CMYK/Black forces Level 10 framing.

Subsampled extras currently force full frames in automatic difference planning.
Decoder upsampling precedes compositing, so an aligned bbox alone is not enough
to preserve interpolation at the crop boundary. Explicit sample crops remain
available but are not advertised as equivalent full-frame compositing.

These are checkpoint records, not a permanent guarantee for later edits.
Newly connected features are not marked verified until their oracle runs pass.
Real PNG/JPEG pixel-input checks from the earlier bring-up remain useful but do
not certify the latest scheduler or performance. See
[PORT_VERIFICATION.md](PORT_VERIFICATION.md) for real-image timing, the corrected
default-route mismatch, and the remaining MA performance gap. The previously
timed-out 4K e9 single-candidate encode now completes, but this is not full-phase
acceptance or a guarantee about all large inputs.

## Open Acceptance Gates

1. Continue the wider source/performance audit; the four work blocks named above
   are closed. Explicit typed controls replace environment parsing, diagnostic
   dumps/final-plan capture are separate tooling, and wider SIMD targets are
   not part of the current x86/scalar scope.
2. Extend hostile-input and sample/frame/channel combination coverage; current
   independent oracle matrices pass, but arbitrary U31 recovery is not proved
   by an f32-only decoder output interface.
3. Establish representative performance parity separately from functional
   migration. Diverse real-image bytes, bpp, time and throughput are recorded;
   no claim that every input is now as fast as optimized C++ follows from them.
4. Retain the passing resource/API/lint/MSRV checks; execute non-Windows targets
   before claiming their runtime support. Cross-compilation is not execution.
5. Continue JPEG byte transcoding as a separately verified track. Shared source
   auxiliary models and the public byte/writer API now work and passed 696
   official original-byte checks. Optimized search and metadata deduplication
   now have separate passing checkpoints; runtime decoding and other documented
   gaps remain. This does not complete
   the pixel phase.

Decoded pixels, not unchanged compressed bytes, determine pixel correctness.
