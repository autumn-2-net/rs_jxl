# JPEG DC And Control Preparation

Frozen source: libjxl `b866b3defec13494bfbaf705db93581cda118e37`, principally
`enc_frame.cc::ComputeJPEGTranscodingData`,
`enc_modular.cc::{AddVarDCTDC, AddACMetadata}`, and
`compressed_dc.cc::DequantDC`. `dec_modular.cc`, `dec_frame.cc` and `dec_group.cc`
provide the corresponding decoder shapes and JPEG-output range constraints.
BSD attribution is retained in `jpeg/dc_streams.rs`.

This is bounded preparation of owned `modular::group::Channel` buffers. It does
not encode Modular streams, choose entropy models, write frame headers, modify
JPEG coefficients, perform an IDCT, or implement JPEG-to-JXL transcoding.

## API

`prepare_dc_group(layout, dc_quantization, contexts, geometry, dc_group, limits)`
returns `DcGroup { rect, channels, quant_dc, extra_dc_precision }`.

- Inputs are borrowed `TranscodeLayout`, F16-roundtripped `DcQuantization`, and
  frame-global `DcContexts`, plus validated `FrameGeometry` and a DC-group index.
- The three owned channels are in Modular Y/X/B order, internal JXL 1/0/2. Their
  dimensions use the source's shifted block rectangles, including MCU padding.
  Channel shift metadata stays zero even for subsampled dimensions, matching the
  source encoder and decoder; it is not a full-resolution image layout.
- The DC input uses the original f32 scale from the JPEG quantization value.
  This is multiplied by the separate F16-roundtripped inverse DC factor and
  rounded using ties-away-from-zero, exactly the `nl_dc=false` path. RGB applies
  integer `1024 / quant[0]` before scaling. Grayscale chroma remains zero.
  The grayscale source copies its single sampling mode to all three channels,
  yielding zero effective H/V shifts; both synthetic chroma planes therefore
  have the full MCU-padded luma dimensions, including raw 2x2 sampling.
- `quant_dc` is a full-resolution, row-major group-local u8 context map. Strict
  threshold comparisons use the encoded integer luma channel, replicated by its
  sampling shifts, never the original float DC. JPEG X/B threshold lists are
  empty, so the source mixed-radix computation reduces to the luma bucket.
- `extra_dc_precision` is zero. All DC CfL factors are fixed at zero; the AC
  correlation maps do not affect DC preparation. `JPEG_STREAM_BITS` is the
  source's nominal depth 8, not a restriction to unsigned 8-bit sample values.

`prepare_ac_metadata_group(geometry, dc_group, cfl_maps, limits)` returns
`AcMetadataGroup { rect, channels, ac_metadata_size }`.

- `CflMaps { width, height, ytox, ytob }` borrows full-frame row-major i8 maps at
  `ceil(frame_blocks / 8)` dimensions. The caller owns correlation computation
  and must supply zero maps when AC CfL is disabled or ineligible.
- Channels 0/1 copy the selected YToX/YToB tile rectangle, widening signed i8
  samples to i32, with source shifts `(3, 3)`.
- Channel 2 has width equal to the block count and height two. Its first row is
  DCT8 strategy zero, and its second is unit raw quantization minus one, zero.
- Channel 3 is a group-shaped zero EPF plane. Channels 2/3 have shifts `(0, 0)`.
  Every DCT8 block is a first block, so `ac_metadata_size` equals the block count.

## Bounds And Errors

`PrepareLimits` caps the combined output sample count and logical owned sample
bytes per call, including qdc. Defaults are 1,048,576 samples and 4 MiB. These
are per-group bounds; a caller retaining multiple groups must also bound their
aggregate storage. Shape, checked arithmetic and both budgets are validated
before any output allocation. Reservations are fallible. Errors return no group.

Block geometry and subsampling must agree with the layout, the group must exist, supplied DC
factors must match this layout's F16 roundtrip, and full-frame CfL map dimensions
and lengths must match. Shifted group rectangles must preserve sampling alignment.
`FrameGeometry::subsampling()` exposes the validated sampling configuration.
Preparation rejects mismatched modes even when padded block dimensions match,
including different raw grayscale modes with identical effective shifts.

The reader can retain the full i16 DC range, but the frozen JPEG output path
cannot reproduce every such value: it disables DC dequantization and clamps
reconstructed JPEG DCs to -2047..2047. Preparation explicitly rejects original
DCs outside that range and any float-rounding result that differs from the
adjusted source integer. It does not replace the required arithmetic with an
integer shortcut or silently clamp/drop a coefficient. For example, synthetic
DC 1023 with quantization 37 becomes 1024 under the source float path and is
rejected as not byte-reversible. Smaller physical 8-bit JPEG DCs can still pass
with 16-bit quantization entries.

## Verification

`jpeg/dc_streams_tests.rs` contains 12 focused tests with small fixtures for scalar f32
arithmetic, RGB offsets, Y/X channel order, gray zero chroma, 4:4:4/4:2:0/4:2:2/
4:4:0 and asymmetric sampling, horizontal/vertical DC-group boundaries, padded edges, 16-bit
quantization, strict qdc thresholds, CfL tile extraction, strategy/quant/EPF
shapes, exact budgets, overflow and explicit range/reversibility errors.

No Cargo or official decoder tests were run by this agent. Parent-owned module
registration and central tests are required. Frame/stream composition and an
independent end-to-end JPEG coefficient-frame oracle remain outside this module.
