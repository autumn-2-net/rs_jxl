# Intrinsic Dimensions Port

Source: read-only libjxl commit `b866b3defec13494bfbaf705db93581cda118e37`.
Implementation: `crates/rs-jxl/src/metadata.rs`, with inline focused tests.

## Caller Contract

`ImageMetadata` now has `pub intrinsic_size: Option<IntrinsicSize>`. Existing
struct literals must add `intrinsic_size: None`; `ImageMetadata::new` already
does so. No default wire bytes change.

`IntrinsicSize::new(width, height)` validates both dimensions in `1..=2^30`.
Its private dimensions are available through `dimensions() -> (u32, u32)`.
`validate()` and `write_header(&mut BitWriter)` are also available. Header
serialization neither allocates pixels nor adds byte alignment.

For the source basic-info omission policy, call:

```rust
metadata.set_intrinsic_size((coded_width, coded_height), (width, height))?;
```

The coded canvas must already be validated. `(0, 0)` and a pair equal to the
coded size clear the field; partial-zero pairs and dimensions above `2^30`
fail without modifying metadata. This applies the policy to the new value,
not the source encoder's incidental retained state after repeated basic-info
calls. Directly assigning an explicit equal-to-coded `Some` remains legal.

Intrinsic size is display/resampling advice. It does not resize or rotate
stored pixels, alter frame/canvas dimensions, select a resampler, or increase
pixel allocation. Orientation does not swap the pair when serializing it.
`Image::with_metadata` and `codestream::image_header_fields` already propagate
the field through `ImageMetadata::write_header`; no new frame caller is needed.

## Frozen Source Mapping

- `lib/jxl/image_metadata.h:348`: `have_intrinsic_size` and the nested
  `SizeHeader`, separate from the first/coded `SizeHeader`.
- `lib/jxl/image_metadata.cc:296`: intrinsic presence enables `extra_fields`.
  The wire order is orientation, intrinsic-presence/nested-size, preview,
  animation, then the existing depth/channel/color/tone/extension fields.
- `lib/jxl/headers.cc:39`: aspect ratios are, in order, `1/1`, `12/10`, `4/3`,
  `3/2`, `16/9`, `5/4`, `2/1`. The first matching truncated integer product
  wins. Multiplication uses `u64`, not floating-point approximations.
- `lib/jxl/headers.cc:67`: the small encoding requires height at most 256 and
  divisible by 8. An implicit aspect ratio removes the corresponding width
  restriction: `9x8` and `512x256` both use the small encoding. Explicit small
  dimensions store `dimension / 8 - 1` in five bits.
- `lib/jxl/headers.cc:129`: general dimensions use offset-one U32 choices with
  payload widths 9, 13, 18, and 30. Height precedes the three-bit ratio;
  width is present only for ratio zero.
- `lib/jxl/encode.cc:1474,1539`: API defaults are zero/zero; positive intrinsic
  dimensions different from the canvas are retained, with each capped at
  `2^30`. The lower-level C++ `SizeHeader::Set` has a wider input cap; this
  Rust type deliberately follows the encoder API's supported range.
- `lib/jxl/decode.cc:2285`: the decoder exposes stored intrinsic dimensions
  when present, otherwise its reported image dimensions. This encoder change
  does not add a decoder/display-size fallback API.

## Verification and Remaining Scope

Five new tests cover literal wire goldens, all seven aspect ratios and first
match ties, every U32 selector boundary, source omission/default behavior,
atomic semantic errors, all orientations combined with animation/tone fields,
unaligned embedding, output limits, and unchanged coded dimensions/pixel data.
Existing metadata tests retain default RGB/alpha wire goldens.

No builds or tests were run by this subagent; central compilation, test runs,
and official decoder oracles remain parent-owned. Scoped rustfmt is permitted.
There is no known remaining intrinsic-size encoding gap within the source
encoder API range. CLI options, decoder fallback/oriented display APIs, and
actual display resampling are not implemented by this slice. Preview and
metadata extensions remain separate unported features. This is not a claim
that the full encoder port is complete.
