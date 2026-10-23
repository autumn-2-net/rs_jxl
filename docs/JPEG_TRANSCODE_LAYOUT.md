# JPEG Transcode Layout Prerequisites

Frozen, read-only source: `../jxl_python/libjxl`, commit
`b866b3defec13494bfbaf705db93581cda118e37` (verified). Audit date: 2026-10-23.
Owned implementation: `crates/rs-jxl/src/jpeg/transcode_layout.rs`, including
inline focused tests. No entropy, JPEG reader/writer, module registration,
dependency, or C++ source files were changed by this worker.

This is a bounded coefficient-layout prerequisite, NOT a working JPEG-to-JXL
transcoder or completion of the pixel-lossless phase. The parent-reported
17 JPEG coefficient-reader oracles (261 million coefficients) and 17 byte-exact
coefficient-regeneration oracles (33 MB, without stored entropy) validate those
existing paths; they do not validate this new layout or a JXL coefficient frame.

## Source Mapping

| Frozen source | Rust implementation |
| --- | --- |
| `jpeg/enc_jpeg_data.cc:216`, `SetChromaSubsamplingFromJpegData` | One or three components; grayscale repeats its sampling into all three channels. |
| `jpeg/enc_jpeg_data.cc:242`, `SetColorTransformFromJpegData` | APP0 precedence, first exact Adobe APP14, RGB component-ID fallback, grayscale forced YCbCr. |
| `frame_header.h:67`, `JpegOrder` | Gray `[0,0,0]`, YCbCr `[1,0,2]`, RGB `[0,1,2]`; direction is JXL channel to JPEG component. |
| `frame_header.h`, `YCbCrChromaSubsampling::Set`; `frame_header.cc:30-31` | Four raw mode codes, unconditional Y/Cb mode swap, maximum raw shifts, relative shifts and 444/420/422/440 classification. |
| `enc_frame.cc:400`, `MakeFrameHeader` | Reject non-YCbCr with nonzero RAW sampling maxima, including nominally 444 all-2x2 RGB. |
| `frame_dimensions.h::FrameDimensions::Set` | MCU-padded DCT block geometry; checked against reader-provided strides and coefficient lengths. |
| `enc_frame.cc:800`, `ComputeJPEGTranscodingData` | Transposed quant matrices, DC scale inputs and inverse, RGB integer DC offset, natural-order CfL fixed quant ratios, borrowed transposed block access, synthetic zero grayscale chroma planes. |
| `enc_frame.cc:1008-1168` | Padded luma DC histogram, threshold-count heuristic, threshold cuts, DCT8 block-context map with source 13-order layout. |
| `chroma_from_luma.h:43-45` | Eleven fractional bits and maximum fixed quant ratio `(256 << 11) - 1`. |

## Representation And Validation

`TranscodeLayout::new(&JpegData)` produces privately stored, validated metadata
and borrowed coefficient planes. It requires decoded coefficient storage, 8-bit
precision, one or three uniquely identified components, supported sampling,
nonzero bounded dimensions, exact MCU-padded block lengths, and a usable scan
quantization snapshot for every component. It is not an independent validator
of JPEG scan progression, archival reconstruction, or frame coding syntax.

No whole coefficient array or pixel image is allocated or cloned. Each plane
retains a slice into the original `JpegData`, so safe Rust prevents mutation of
the borrowed coefficients while the layout is live. `block(x,y)` checks bounds;
`BlockView::write_transposed` writes only caller-owned 64-entry i32 scratch.
`transposed_coefficient` can inspect the layout without copying a block.
Grayscale channels 0 and 2 return synthetic zero blocks, not duplicate luma
coefficients, while their quantization metadata follows source `JpegOrder`.

The source C++ component `quant_idx` is a quant-vector index. Rust
`Component::quant_index` is an ON-WIRE destination ID. It must never directly
index `JpegData::quant`. This module resolves it through
`ScanInfo::quant_table_indices` for every participating scan, checks the selected
table's destination ID, precision and positive value range, and records the
first participating vector index per component.

Repeated definitions with identical values are compatible even if their
precision or marker-grouping metadata differs. A used destination ID with
different values across participating scans is explicitly rejected, including
disjoint component scans sharing that ID. This is a conservative single-global-
representation policy, not a claim that such a JPEG is invalid or unreadable.
Unused slots and definitions that never participate in a component scan do not
determine the selected matrix. No fallback to the first/last quant-vector entry,
final active table, or numeric wire ID is used when a snapshot is absent.

APP data uses the reader's representation: marker byte, two length bytes,
payload. Public-model inconsistencies are rejected before color inference.
For valid metadata, source precedence is preserved literally: ANY APP0 counts,
not merely a recognized `JFIF` payload; the first exact 15-byte `Adobe` APP14
stops the Adobe search; only transform byte zero means RGB; component IDs are
consulted only if no matching Adobe marker exists.

## Arithmetic And Ordering

- Quant matrices use `out[8*x+y] = jpeg[8*y+x]`. Coefficients use the same
  transpose, widening i16 to i32 without changing quantization or integer DC.
- DC preparation preserves `2040.0f32 / q0`, then a separate `1.0f32 / scale`.
  `scaled_dc` multiplies the adjusted integer DC by that inverse. It does not
  algebraically substitute `q0 / 2040` or use f64 in production.
- RGB adds the integer quotient `1024 / q0` to the DC image input, but does not
  modify the borrowed quantized block. YCbCr and grayscale add zero.
- CfL ratios are in NATURAL JPEG order. The source reads transposed matrices
  and transposes the ratios back. Preparation uses positive integer
  `(2048 * luma_q) / channel_q`, including DC, and rejects ratios above 524287.
  Valid 16-bit quant entries keep the intermediate product within signed i32.
  Source checks the ratio even if subsequent optional CfL fitting is disabled.
- DC histogram traversal includes every MCU-padded luma block. The source's
  `min<uint32_t>(idc+1024, 2047)` converts before clamping; negative sums map to
  the last bin rather than receiving an invented lower clamp.
- The threshold-count heuristic uses transposed CHANNEL ZERO `qt[1..6]`, not
  channel one's quant table. It clamps the requested count to 1..7, uses strict
  `cumsum > cut`, and adds at most one threshold per histogram bin. Empty bins
  after a large jump may receive thresholds; a jump at the last bin may yield
  fewer thresholds than requested. Those source behaviors are retained.
- Only channel one has DC thresholds. The block-context map's plane order is
  Y, X, B, with 13 order slots per plane; only DCT8 entries are populated.
  Actual context counts remain at most 16 (at most 9 for grayscale).

## Parent Integration

Parent registers the completed file with `pub mod transcode_layout;` in
`jpeg/mod.rs`; this worker does not edit registration. The module has no new
dependencies and no hardware assumptions, feature detection or unsafe code.

```rust
let layout = jpeg::transcode_layout::TranscodeLayout::new(&decoded_jpeg)?;
let order = layout.jpeg_order();
let subsampling = layout.subsampling();
let contexts = layout.dc_contexts();
let plane = &layout.planes()[1];
let quant = plane.quantization();
let mut block_scratch = [0i32; 64];
plane.block(0, 0).unwrap().write_transposed(&mut block_scratch);
```

Important getters: `Quantization::{table_index,destination_id,transposed,
cfl_ratios,dc_quantization,inverse_dc_quantization,dc_offset}`;
`PlaneLayout::{jpeg_component,dimensions_in_blocks,is_synthetic_zero,block,
adjusted_dc,scaled_dc}`; `DcContexts::{luma_thresholds,requested_thresholds,
num_dc_contexts,context_map,num_contexts}`. The color/subsampling functions can
also inspect parsed metadata before coefficient decoding. Layout construction
itself requires decoded storage; `can_use_ac_cfl` reports only source geometry/
channel eligibility and does not enable the caller's force-CfL option.

## Remaining Work

- The parent-owned VarDCT/JXL frame writer and its production integration do not
  exist in this module. No public JPEG-to-JXL encoding API is supplied here.
- Prepared DC factors are INPUTS to `DequantMatricesSetCustomDC`. Its custom DC
  F16 serialization/decoder round-trip, RAW matrix coding, custom-matrix setup,
  quantizer setup (`quant_dc=1`, `global_scale=65536`, raw quant field 1), and
  frame x/b matrix scales 2 still belong to the future frame writer. The raw DC
  inverse used for source DC-image construction must not be confused with the
  round-tripped matrix inverse.
- CfL tile search, `RatioJPEG`, two-stage fixed-point AC residual formation,
  and correlation-map signaling are not implemented here. Only fixed quant
  ratios and applicability metadata are prepared.
- AC strategy/progressive splitting, DC/AC Modular metadata, coefficient token
  emission, entropy models, frame sections and JPEG reconstruction-box wiring
  remain outside this bounded slice. No requantization or pixel decoding was
  substituted for those paths.
- Source final-plan record/replay is not implemented. This module computes the
  source default threshold plan; it does not accept replayed threshold edits.
- Incompatible used-slot quant redefinitions need a broader representation
  decision before any future frame writer can accept them. This module rejects
  them explicitly instead of promising byte-reversible output with a wrong table.

## Focused Verification

Nine inline tests are ready for parent-controlled compilation/execution:

1. Color marker precedence, Adobe/APP shape validation, grayscale and channel maps.
2. All 64 supported three-component raw sampling combinations against independent
   integer shift math, invalid factors, and common subsampling classifications.
3. Non-identity quant ID/vector ordering, transpose and fixed-ratio goldens,
   staged f32 DC math, borrowed block pointer identity and bounds.
4. RGB integer DC offsets, untouched coefficient DC, synthetic grayscale planes,
   and source rejection of all-2x2 RGB despite relative 444 classification.
5. Identical and conflicting quant redefinitions, including disjoint component scans.
6. Missing/bad snapshots, quant values/precision, coefficient geometry and input bounds.
7. 16-bit quant matrices and exact fixed-ratio acceptance/rejection boundaries.
8. DC thresholds and context-map goldens, strict cuts, empty-bin thresholds and
   last-bin concentration reducing the actual bucket count.
9. Actual layout DC traversal, channel-zero transposed quality heuristic,
   MCU padding and signed-to-unsigned histogram clamping.

No Cargo build/test command was run by this worker. Scoped rustfmt completed
for the new module only; central checks and new decoder/corpus evidence remain the parent's
responsibility. No commits were made.
