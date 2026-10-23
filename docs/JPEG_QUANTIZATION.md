# JPEG Quantization Prerequisites

## Scope

`jpeg/quantization.rs` is a bounded pure-Rust port of the JPEG-needed DC and RAW
DCT8 quantization headers and table preparation. It is **not a complete JPEG
transcoder**, complete VarDCT encoder, lossy weight generator, or full-frame API.
This worker owns only that new module, its inline tests, and this document.
Module registration and frame/global integration remain parent-owned.

Frozen, read-only source: libjxl `b866b3defec13494bfbaf705db93581cda118e37` at
`../jxl_python/libjxl`. Translated code retains the JPEG XL Project Authors'
BSD-3-Clause attribution; see `PORTING_KERNELS.md` for the notice.

Source map:

- `enc_quant_weights.cc::{DequantMatricesEncodeDC, DequantMatricesSetCustomDC,
  DequantMatricesEncode, EncodeQuant}`: DC round-trip and table header ordering.
- `quant_weights.h::{SetDCQuant, QuantEncoding::RAW, QuantTable}`: reciprocal
  input convention, raw denominator, 17 slots, and mode numbers.
- `quant_weights.cc::DecodeDC` and `fields.cc::F16Coder::Read`: decoder-visible
  DC values and finite F16 conversion; `enc_fields.cc::F16Coder::Write` truncates.
- `quantizer.cc::QuantizerParams::VisitFields`: the two U32 distributions.
- `enc_modular.cc::{AddQuantTable, EncodeQuantTable}` and
  `dec_modular.cc::DecodeQuantTable`: 8x8x3 integer planes, nominal depth eight,
  denominator header, positive-symbol checks, and embedded Modular boundary.
- `enc_frame.cc::ComputeJPEGTranscodingData`: JPEG channel order, one transpose,
  `2040.0f / quant[0]`, global scale 65536, DC quant 1, and raw block quant 1.

## Parent Interfaces

All entry points return the crate's `Result`; input dimensions are fixed by the
8x8x3 array type. No dependencies or module registration are added here.

- `DcQuantization::from_inverse([f32; 3])` ports SetCustomDC's inverse input
  convention, including its F16 round-trip. `from_layout([&Quantization; 3])`
  and `from_jpeg_tables([&QuantTable; 3])` adapt the prepared/layout inputs.
- `dc_quants()` and `inverse_dc_quants()` expose decoder-rounded values.
  `DcQuantization::write` emits the DC header. The lower-level
  `write_dc_quantization(writer, dc_quants)` accepts DCQuants, not inverse inputs.
- `QuantizerParams { global_scale, quant_dc }::write` emits VisitFields with
  no extra default flag. `jpeg()` is `(65536, 1)`; `default()` is the source
  field default `(1, 1)`. These are intentionally different.
- `RawDct8Quantization::new([[i32; 64]; 3], denominator)` validates a positive
  raw integer table and decoder-valid denominator. `from_source_shift` uses
  the source RAW formula; the JPEG factories `from_layout`/`from_jpeg_tables`
  select shift zero. Samples and denominator getters are read-only.
- `prepare_modular_channels()` returns three 8x8 `Channel`s with zero shifts.
  The source image bit depth is `QUANT_TABLE_BITS = 8`, **not** a sample clamp.
- `write_header(writer, callback)` emits RAW DCT8 and the remaining Library(0)
  slots. The callback is
  `FnOnce(&mut BitWriter, &RawDct8Quantization) -> Result<()>`, invoked immediately
  after the denominator and exactly once. No byte padding is added.

The callback can call parent-owned
`modular::encode::write_prepared_stream(&channels, &transforms, options, writer,
stream_id, QUANT_TABLE_BITS)`. The caller owns transforms and the correct
frame-dependent `ModularStreamId::QuantTable(0)` mapping. Standalone source
round-trips without a shared frame encoder use stream ID zero. Do not substitute
the byte-padded group encoder at this boundary. Tests exercise the unpadded
callback using the prepared-stream writer with both Prefix and ANS coding.

Both input factories expect **internal JXL channel order**. They do not resolve
JPEG destination IDs or reorder color planes. `QuantTable` contains natural
order entries and is transposed once; `transcode_layout::Quantization` is already
transposed and is copied unchanged. JPEG precision validation is retained:
precision 0 allows 1..255 and precision 1 allows 1..65535.

## Numerical And Wire Semantics

SetCustomDC first takes the reciprocal of each supplied inverse factor. If those
three DC values exactly equal `[1/4096, 1/512, 1/256]`, EncodeDC writes one `1`
bit and DecodeDC leaves the existing state unchanged. Otherwise it writes `0`
followed by three F16 values of `dc * 128`, in channel order. Decoding divides
by 128, rejects values below `1e-8`, and recomputes the inverse. This threshold
is applied **after truncation**, not merely to the original float.

The layout's original DC factors are not overwritten: the source separately
uses the pre-round-trip reciprocal for JPEG DC image preparation. Rounded matrix
factors and original layout factors must not be silently interchanged.

F16 writes reuse parent-owned `metadata::f16_bits_truncating`, imported locally
as `f16_bits`; the read conversion is a direct local port. F16 subnormals and
finite signed values are modeled, while NaN/infinity are rejected. Signed
zero/tiny values write positive zero, as in the
source. DC/raw denominators must also satisfy their positive decoder gates.

Important boundary: source `F16Coder::CanEncode` (`fields.cc:576`) accepts finite
magnitude <=65504. The existing `metadata::f16_bits` correctly retains that
strict metadata validation. The directly called `F16Coder::Write` instead accepts
finite magnitude **<65536** and truncates the upper tail to 65504. Parent factored
the separate `pub(crate) f16_bits_truncating` direct-write entry and changed this
module's import to use it, without weakening metadata validation. The earlier
request to widen the strict helper was superseded by this separation.

RAW uses `(1 << shift) * (1.f / (8 * 255))`; JPEG uses shift zero. The encoded
denominator is F16 `0x1004`, decoded as `0.0004901885986328125`, rather than the
original f32 reciprocal of 2040. It scales the raw weight calculation, not the
stored integer table. Source RAW weights use `1.f / (decoded_denominator * q)`.
This helper does not implement the subsequent complete DCT weight generator or
its aggregate bounds. The low-level RAW constructor preserves every positive
i32 symbol, including values above 65535 and 2^24; JPEG factories enforce the
narrower JPEG limits. Shifts >30 are outside the defined source signed-int shift
domain; shifts 27..30 additionally exceed finite F16 denominator range.

The custom table header is: `all_default = 0` (1 bit), DCT8 `RAW = 7` (3 bits),
F16 denominator (16 bits), Modular payload, then sixteen `Library = 0` modes
(3 bits each). There is one predefined library, so no library index bits follow.
Fixed overhead is 68 bits, excluding the caller's payload. Other quantization
modes and full-table library data are not implemented.

Quantizer fields use source shortest-selector/first-tie rules:

- global_scale: `BitsOffset(11,1)`, `(11,2049)`, `(12,4097)`, `(16,8193)`;
  encodable range 1..73728.
- quant_dc: `Val(16)`, `BitsOffset(5,1)`, `(8,1)`, `(16,1)`;
  encodable range 1..65536, with 16 using the direct selector.

Known LSB-first vectors, padded only by the test harness:

| Header | Logical Bits | Hex Bytes |
| --- | --- | --- |
| Default DC | 1 | `01` |
| Custom DC halves 2c04/4c00/6c04 | 49 | `08 58 00 98 08 d8 00` |
| JPEG QuantizerParams | 25 | `ff 7f 07 00` |
| Default QuantizerParams | 20 | `00 20 00` |
| RAW + test 5-bit payload 21 + 16 library modes | 73 | `4e 00 51 01 00 00 00 00 00 00` |

Invalid scalar input is rejected before header writes where possible. Writer
allocation errors and callback failures propagate without rollback; callers must
discard a partial header. No input table is mutated.

## Verification Boundary

Nine inline tests cover fixed vectors, all finite F16 encodings, the source
upper-tail write limit and post-round-trip lower gates, U32 selector edges,
unaligned callback positions, exact raw integers, denominator shifts, JPEG
precision/transposition/layout adapters, error propagation, and actual prepared
Modular payload insertion. These are primitive/header checks, not an independent
official-decoder test of a complete JPEG-transcoded frame.

No Cargo builds/tests, dependency changes, commits, source edits, or parent-owned
file edits are performed by this worker. Parent owns registration, compilation,
Clippy, test execution, and later full-transcoder verification. Scoped standalone
`rustfmt --check --edition 2024 --config skip_children=true` passes for the new
module; this is not a compile check. Parent has registered the module and resolved
the F16 dependency with the separate truncating helper. Parent reports all nine
quantization tests passing in central execution. This worker's code is frozen.
No JPEG archive or byte-reconstruction completion is claimed here.
