# Multiframe Reuse Roadmap

Source inspection: 2026-10-27, `rs_jxl` base HEAD
`59c66f8338d711730c181e31cfcf0ad30f27b4ce`; read-only optimized C++ reference:
`b866b3defec13494bfbaf705db93581cda118e37`.
This is a future-reuse investigation, not an implementation of temporal patch
search. The implemented [heterogeneous-size layer API](HETEROGENEOUS_LAYERS.md)
is a separate feature.

**Yes, stronger reuse is possible without extending the JXL bitstream.**
The useful next steps are bounded selection of older saved states, then exact
local rectangle matching expressed with standard patches. Neither means free
reuse, arbitrary access to past frames, or general video motion compensation.

## What Exists Today

| Area | Current source behavior |
| --- | --- |
| Automatic references | `Reference` and `plan` in [multiframe.rs](../crates/rs-jxl/src/multiframe.rs), and `reference_indices` in [auto.rs](../crates/rs-jxl/src/multiframe/auto.rs), choose previous, first, both, or full-only. No older-anchor selection or displaced cross-frame search. |
| Difference geometry | [diff.rs](../crates/rs-jxl/src/multiframe/diff.rs) `DiffMap::between`, `regions.rs::tiles_from_map` and `partition.rs::propose` compare the same canvas coordinates. Tiled/tightened and adaptive merged rectangles transmit changed current samples; they do not copy a displaced old rectangle. |
| Selection | `auto::encode_inner` measures complete candidate layer bytes, including generated patch-reference frames. Full REPLACE remains a candidate; `plan` instead uses bounding-box area. Neither proves a global optimum. |
| Reference allocation | Automatic encoding saves the first reconstructed canvas in slot 1 and subsequent/intermediate reconstructed canvases in slot 2. Slot 3 is reserved when current patch tools are enabled. These are encoder policies, not fixed meanings of the standard slots. |
| Explicit layers | [frame.rs](../crates/rs-jxl/src/frame.rs) `FrameOptions` and [encode.rs](../crates/rs-jxl/src/encode.rs) `encode_frames` already expose origins, blending, reference-only frames and save stage. Explicit legal slot selection is broader than automatic first/previous selection, but does not discover matches. |
| Existing patches | `encode_logical_frame_impl` calls [patch_frame.rs](../crates/rs-jxl/src/patch_frame.rs) `prepare`/`prepare_source` on the current image and emits a new pre-transform atlas into slot 3. This is intra-frame repetition reuse, not a persistent temporal dictionary. |

Automatic state APIs currently require identical geometry, layout and metadata.
General layer placement must not be mistaken for automatic heterogeneous-image
matching. Multi-rectangle intermediates have duration zero; only the last layer
carries the logical duration. An identical state still uses a legal 1x1 REPLACE
layer. See [MULTIFRAME_AUTO.md](MULTIFRAME_AUTO.md) and its source links above.

The old C++ encoder has the same important distinction: in
`lib/jxl/enc_patch_dictionary.cc`, `FindBestPatchDictionary` packs components of
the supplied `Image3F`; `RoundtripPatchFrame` writes a reference-only atlas.
Its emitted patches use ADD for color and NONE for extras. This is not evidence
that it searches earlier animation frames.

## Standard References And Their Limits

There are **four image-reference slots, IDs 0..3**, not four total stream frames
and not an arbitrary history indexed by input-frame number. Saving replaces a
slot; older content is usable only while retained or after being encoded again.
Slot 0 is real: for a nonzero-duration frame, save ID 0 means no future save;
zero-duration/reference-only frames can save there. Last frames are not saved.
Local evidence: C++ `lib/jxl/common.h::kMaxNumReferenceFrames`,
`frame_header.h::CanBeReferenced`, `frame_header.cc::FrameHeader::VisitFields`,
and Rust `encode::validate_frame_references`. The official
[JxlLayerInfo definition](https://raw.githubusercontent.com/libjxl/libjxl/main/lib/include/jxl/codestream_header.h)
documents the save-ID/duration distinction. A slot-count limit is not a promise
of unlimited frame count, memory, or decoding time.

Two different reuse stages matter:

- **Frame blending** uses a post-color-transform canvas. A crop's origin places
  newly decoded samples; its background comes from matching canvas coordinates.
  Changing the origin does not translate the saved background.
- **Patches** specify a source rectangle in a saved pre-color-transform frame
  and destination positions in the current frame, with color/extra-channel
  blend rules. REPLACE can therefore express exact translated rectangle copies.
  Source and destination rectangles must fit; missing or post-transform patch
  sources are rejected. A regular frame may save before the color transform
  only under the header's eligible full-coverage/REPLACE conditions;
  reference-only frames provide another route. A partial blended canvas cannot
  simply be relabeled as a patch source, even when no color transform is used.

Local C++ evidence: `lib/jxl/dec_patch_dictionary.cc::PatchDictionary::Decode`
and `lib/jxl/frame_header.cc::FrameHeader::VisitFields`. Official primary sources:
[patch decoding](https://raw.githubusercontent.com/libjxl/libjxl/main/lib/jxl/dec_patch_dictionary.cc)
and [frame-header conditions](https://raw.githubusercontent.com/libjxl/libjxl/main/lib/jxl/frame_header.cc).
These implementation references establish existing standard mechanisms, not
that every proposed encoder policy has been verified against the specification.

**An uninitialized slot is a valid all-zero frame-blending background.**
C++ `lib/jxl/render_pipeline/stage_blending.cc::BlendingStage` allocates zero
rows for an empty background; the vendor's
[BlendingStage::new/process_row_chunk](../vendor/jxl-rs/jxl/src/render/stages/blending.rs)
also accepts absent slots and reads zero rows. Its [canvas-extension stage](../vendor/jxl-rs/jxl/src/render/stages/extend.rs)
and native integer `coalesce` preserve this behavior. This does **not** permit a
patch to read an absent source. Official evidence:
[C++ blending stage](https://raw.githubusercontent.com/libjxl/libjxl/main/lib/jxl/render_pipeline/stage_blending.cc).
The base revision's `encode::validate_frame_references` rejected these unsaved
sources: an encoder overconstraint, not a JXL rule. The heterogeneous-layer
implementation removes this restriction while still rejecting pre-transform
blend backgrounds; its tests cover the zero-background behavior.
An overwritten nonempty slot is not automatically zero again.

## Decoder Support Is Not Encoder Support

The vendor decoder already parses patch reference IDs, source rectangles and
independent destination positions in
[PatchesDictionary::read_with_reference_info/add_one_row](../vendor/jxl-rs/jxl/src/features/patches.rs).
[build_render_pipeline](../vendor/jxl-rs/jxl/src/frame/render.rs) places patches
before output color conversion and frame blending. The decoder can already
express translated copies; temporal search/planning is missing in our encoder.

Our [tools/patches.rs](../crates/rs-jxl/src/tools/patches.rs)
`PatchDictionary::validate`/`tokenize` can describe IDs below four, but its
discovery/composition helpers and frame integration assume a private slot-3
atlas. A tokenizable dictionary alone is not a public temporal-patch encoder.
Persistent references need explicit lifetime, save-stage and slot-conflict
handling; the existing atlas writer must not overwrite them accidentally.

The exact native integer decoder is narrower than general rendering:
[native_integer.rs](../vendor/jxl-rs/jxl/src/frame/native_integer.rs)
`NativeIntegerFrameBuffer::new`, `apply_patches`, `combine` and `coalesce` enforce
format/transform gates, support integer copies/addition, and reject fractional
alpha, multiply and alpha-weighted-add cases. General f32 rendering cannot
establish arbitrary 17..31-bit integer equality.

## Exactness Constraints

- Match every declared sample, including invisible RGB, alpha and extras.
  `DiffMap::between` compares float words, not epsilon distances; signed zeros
  and NaN payloads matter. Hashes may propose matches but need exact comparison.
- Existing binary masks require both inputs to be opaque integer GA/RGBA in
  U8/U16 storage, one unassociated full-resolution alpha channel, and no other
  extras (`diff::mask_alpha_max`, `scan_main`). Partial/premultiplied alpha and
  hidden RGB must not be silently normalized. REPLACE does not have that opacity
  requirement, provided every required channel is reconstructed exactly.
- Automatic crops currently fall back to full frames for subsampled extras.
  Grid alignment alone is insufficient: upsampling before compositing changes
  boundary samples. Temporal patches must separately validate sampling/stage
  behavior, or leave unsupported extras as NONE and encode them normally.
- Do not repeat the previously failed float-ADD reuse experiment blindly. Its
  root cause was not re-established here. Current intra-frame
  [discover_source_float/compose_float_add](../crates/rs-jxl/src/tools/patch_source_float.rs)
  supports eligible <=16-bit float color only with strict atlas/residual
  representability and exact ordered replay; failures are errors. This is not
  proof that `current - reference + reference` preserves F16/F32/custom-float
  words. Even integer ADD needs the official decoder-arithmetic replay guard in
  [patch_source_replay.rs](../crates/rs-jxl/src/tools/patch_source_replay.rs).
  Start temporal work with REPLACE on supported integers; float-copy paths and
  special values still need independent raw-word verification.

## Practical Priority

1. **Retain one older anchor for aligned reuse.** Extend the existing exact diff
   candidates to an explicitly retained post-transform keyframe, initially by
   optionally rotating slot 1 instead of permanently retaining frame zero.
   Keep slot 2 for previous/intermediate state and slot 3 for current patch tools.
   This offers recurrence reuse with little new pixel machinery. Define the
   save/eviction schedule first; scanning old input images that are not saved in
   the codestream is not sufficient. This is proposed, not implemented.
2. **Bounded local crop matching with standard REPLACE patches.** For scrolling
   content or translated repeated regions, search a limited displacement window
   against one retained pre-transform source; verify proposed matches exactly.
   Copy `(sx, sy, width, height)` to `(dx, dy)` and encode unmatched/newly exposed
   samples normally, accounting for cropped-layer coordinates. Start with
   nonoverlapping integer rectangles. Use an eligible full reference save or
   explicitly transmitted reference-only image; retaining both pre- and
   post-transform versions costs slots and may cost an extra encoded frame.
   A persistent slot-3 source must be mutually exclusive with, or coordinated
   with, the current per-layer atlas. No arbitrary translation of a post-transform
   canvas, fractional motion, rotation or scaling is implied.
3. **Make latency/keyframes an explicit policy, not a hidden size heuristic.**
   Optionally cap layers per displayed frame and reference dependency span;
   periodically encode independent full-canvas REPLACE states. A retained anchor
   and a seekable restart point are different concepts. Later frames must not
   reach behind a restart through any color/extra/patch reference. Current
   [IndexPlan::new](../crates/rs-jxl/src/streaming/index.rs) checks interval
   independence, and `encode_indexed_frames_streaming_to` rejects generated
   patch-reference tools. Temporal patches are not automatically index-compatible.

Keep full/crop alternatives and compare real output bytes including dictionaries,
residuals, reference initialization/refresh and headers. A persistent atlas changes
future costs, so current per-frame selection is not a global optimization proof.
Measure encode time, coalesced decode time, seek/replay latency and peak retained
memory as well as size. Many tiny layers can save bytes and still lose on latency.
Local search windows, candidate budgets and rectangle caps are policy controls,
not new bitstream limits or guaranteed speedups.

## Verification Before Adoption

These are future acceptance cases, not tests run for this document:

- Older-state recurrence, scene changes, no-match inputs, source overwrite and
  slot exhaustion; prove actual reuse and exact fallback, not just smaller files.
- Integer translations in both axes, odd dimensions, frame edges, vacated/new
  regions, cropped destinations and repeat placements; reject invalid rectangles
  and wrong-stage/absent patch sources.
- Valid initial zero-background crops and per-extra sources; distinguish them
  from an old nonzero slot. Exercise reference-only saves and slot-0 duration rules.
- Gray/RGB/GA/RGBA, invisible RGB, partial/premultiplied alpha, mixed depths,
  extra-only changes, subsampled extras and U17..31 native output. Separately test
  F16/F32/custom words, signed zero, subnormals, infinities and NaN payloads before
  admitting those formats. A normalized f32 export is not a wide-integer oracle.
- Coalesced, raw-layer and selected-frame decode; zero/nonzero duration mixtures,
  final-frame handling and restart independence. Compare original samples with
  the vendor decoder and independent official `djxl` where its output is exact.
- Include initialization costs in short and long sequences; report actual
  dictionary activation, bytes, layer count, encode/decode/seek time and memory.

Existing starting points are `tests/multiframe_auto.rs`,
`tests/metadata_frame_reference.rs`, `tests/patch_add_reference.rs` and
`tests/float_patch_reference.rs` under `crates/rs-jxl`, plus the vendor's
`features/patches_validation_tests.rs`. Their existing coverage is not evidence
for the new temporal candidates.
