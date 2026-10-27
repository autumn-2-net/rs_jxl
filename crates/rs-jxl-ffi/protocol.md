# C ABI protocol v1

C structs use native ABI/endian; pixel buffers are packed LITTLE-ENDIAN.
JSON is UTF-8 without NUL. Status 0=success, 1=request/codec error, 2=caught
panic. Errors contain an unchanged `error` message, an `error_kind` category,
and empty data; see Structured errors below. Null result getters return status
-1 or null with length 0. Results own both buffers until `rs_jxl_result_free`.
The ABI version remains 1; context functions are optional additive symbols.

## Structured errors

An error result has this shape; `error_stage` is optional:

```json
{
  "error": "resource limit: JXL input bytes",
  "error_kind": "resource_limit",
  "error_stage": "context_execute"
}
```

`error` preserves the existing human-readable message. `error_kind` is one of
`invalid_input`, `resource_limit`, `encoding`, `decoding`, `io`, `internal`, or
`unsupported`. Categories do not replace the status code: ordinary failures use
status 1 regardless of category. A caught panic uses status 2, `internal`, and
the message `Rust panic during execution`. Abort and unrecoverable allocation
failures are not converted to results.

Categories come from typed errors, not message text. Core `InvalidInput`,
`Limit`, `Encoding`, `Decoding`, and `Io` variants retain their corresponding
categories. Known boxed reader errors and their sources are inspected without
changing the outer message. String-only errors and unclassified reader errors
default to `invalid_input`; words such as "unsupported" or "limit" in a message
alone do not establish a category. Unsupported ABI versions and context-kind
mismatches are explicitly categorized as `unsupported`.

`error_stage` adds diagnostic context without prefixing or otherwise changing
`error`. Current boundary labels include `context_create`, `context_execute`,
and `serialize_response`; a more specific existing stage is retained. Callers
must not require a stage, including for panic results. Older v1 libraries may
omit `error_kind` as well; compatibility clients should retain their existing
message/status fallback.

## Reusable contexts

The existing `rs_jxl_execute` remains the stateless entry point. Capability-detect
the following symbols rather than assuming ABI version 1 guarantees contexts:

```c
RsJxlResult *rs_jxl_context_create(const RsJxlRequest *request,
                                  RsJxlContext **out_context);
RsJxlResult *rs_jxl_context_execute(RsJxlContext *context,
                                   const RsJxlRequest *request);
void rs_jxl_context_free(RsJxlContext *context);
```

### Creation

Creation uses the same `RsJxlRequest` ABI envelope, but a separate JSON schema:

```json
{"kind":"encoder","threads":2}
```

`kind` is required and must be `encoder` or `decoder`. `threads` defaults to 1
when omitted, must be an integer in 1..256, and cannot be null. Unknown and
duplicate fields fail. Operation fields, an `options` object, and a nonempty
data payload are not accepted.

`out_context` must be writable for one pointer, must not overlap request/input
storage, and may be unaligned. A null output pointer returns an error. A valid
output slot is set to null before request validation and remains null on failure.
Free an existing context before reusing its output slot; creation does not free
the old handle. Success returns status 0, JSON such as
`{"kind":"encoder","threads":2}`, empty data, and a new caller-owned handle in
`*out_context`. Always free the creation result separately, on success or failure.

### Execution

Context execution accepts the unchanged operation-request JSON described below.
Only the thread count is fixed by the context. Where an operation accepts
`options.threads`, omission or null inherits the context count; an explicitly
different count fails with `request threads must match the context`. All other
settings use each request's values or ordinary operation defaults. Decoder
options, including limits and output format, do not carry over to the next call.

| Context kind | Operations using reusable resources | Stateless operations also accepted |
| --- | --- | --- |
| `encoder` | `encode`, `encode_multiframe`, `transcode_jpeg` | `analyze_multiframe` |
| `decoder` | `decode`, `decode_to_png`, `reconstruct_jpeg` | `info`, `read_image` |

Other kind/operation combinations fail. The stateless operations above do not
reuse the context's workers or parser state. Their normal field restrictions
remain in force: `analyze_multiframe` and `read_image` still reject an explicitly
supplied nonnull `threads` option. `info` inherits the thread count as usual.
Per-input parser/frame state and input buffers are not retained between calls.

### Ownership and failure

All calls are synchronous. Request storage follows the same contract as
`rs_jxl_execute`: a nonnull request exposes the readable native version/size
prefix, and a matching prefix requires the full struct plus readable, immutable
storage for each nonempty buffer. Request/input storage must not alias the
context. Invalid or dangling pointers cannot be detected reliably.

A nonnull context must be a live handle returned by creation. The caller must
serialize all execute/free access to one context; there is no internal lock.
Independent contexts may execute concurrently. Null execution handles return an
error. `rs_jxl_context_free(NULL)` is a no-op; otherwise free exactly once with
no concurrent use. A context remains live until explicitly freed, even after an
error.

Ordinary request/codec failures permit reuse of a healthy context. After any
status-2 result, treat that context as poisoned: free it and create a replacement,
never attempt further execution. A handle whose internal poison flag is set
rejects execution with status 1, `error_kind: internal`, and
`context is unusable after a panic`; this does not restore its usability.

Results and their buffers are independent of context lifetime. Further context
calls or freeing the context do not invalidate earlier results. Result getters
return immutable borrowed bytes, valid only until that result is freed. Getter
length outputs may be null; otherwise they must be writable and must not alias
result storage. Do not free a result concurrently with its getters/readers.
Freeing a result does not free its context, and freeing a context does not free
any results.

## Requests

Unknown/inapplicable fields fail. Optional fields may be omitted or null.
Top-level keys: operation (required), options (default {}), images (default []),
durations, reference, timebase, loops, auto. Sequence keys only for encode_multiframe.
Operations: encode, encode_multiframe, analyze_multiframe, transcode_jpeg,
reconstruct_jpeg, decode, decode_to_png, info, read_image.
encode requires one image; multiframe requires nonempty images. Other operations
consume entire data buffer and reject images. info is JXL header inspection;
read_image is shared PNG/JPEG pixel input decoding.

durations defaults to one tick per image; supplied length must equal image count,
zeros remain zeros. reference=previous (full/previous/first/smallest_crop),
timebase=[1000,1] (ticks per second numerator/denominator), loops=0, auto=false.
encode_multiframe always sets animation metadata. auto uses existing search.

ImageSpec: {format:"pixels",offset:0,length,width,height,channels,dtype,
bits_per_sample?,exponent_bits_per_sample?,extra_channels:[],icc?,
color_encoding?,orientation?}. length required; pixels require dimensions,
channels and dtype. channels 1..4 = gray/gray-alpha/RGB/RGBA. dtype is
u8/u16/u32/f16/f32; default bits 8/16/31/16/32, exponent 0/0/0/5/8.
color_encoding=srgb or linear_srgb, mutually exclusive with ICC byte array.
orientation=1..8. format=encoded accepts only format/offset/length; shared image
library detects PNG/JPEG bytes. No metadata override on encoded images.

ExtraSpec: {offset:0,length,dtype,type,name:"",bits_per_sample?,
exponent_bits_per_sample?,dim_shift:0,alpha_premultiplied:false,
spot_color?,cfa_channel?}. Planes append after any interleaved alpha;
dimensions ceil(width/2^dim_shift),ceil(height/2^dim_shift). Types:
alpha/depth/spot_color/selection_mask/black/cfa/thermal/optional. Unknown type
fails if unsupported by encoder. spot_color is four floats, cfa_channel integer.

## Options

- encode/encode_multiframe: threads=1, effort=7, source_search=false,
  strip_metadata=false; optional group_size_shift,rct,predictor,palette_colors,
  patches(bool),learning_fraction,ma_max_leaves. Effort/overrides map to native
  candidates. Unsupported projections fail. source_search only for still images,
  cannot combine with explicit tool overrides (ma_max_leaves is allowed).
- analyze_multiframe: strip_metadata only. Result {frames:[{index,previous,first}]}
  where each bbox is null or {x,y,width,height}.
- decode/decode_to_png: threads=1, frame_index=null, coalescing=true,
  format=native (native/u8/u16/f32), include_extra_channels=true,
  adjust_orientation=true,dither=jxl-rs (jxl-rs/libjxl),limits={}.
  decode_to_png defaults frame_index=0; emits PNG, not packed pixels; refuses
  float/wide integer output, non-alpha extras, or inexact alpha/depth conversion.
- info: threads=1,coalescing=true,adjust_orientation=true,limits={}.
- read_image: strip_metadata=false only. Returns packed native decoded samples.
- reconstruct_jpeg: threads=1,limits={}; byte-exact only, no pixel fallback.
- transcode_jpeg: threads=1,effort=7,source_search=false,brotli_quality=null,
  no_cfl=false,order=source (source/default/adaptive),limits={}.

Limits are NESTED options.limits, never flattened. Decode/info/reconstruct:
max_input_bytes,max_output_bytes,max_metadata_bytes,max_pixels,max_samples,
max_frames. Transcode: max_input_bytes,max_output_bytes,
max_coefficients,max_tokens,sample_limit,max_coefficient_bytes,max_block_visits.
Missing limits use library defaults. Unsupported limits fail.
Transcode rejects max_metadata_bytes: the native JPEG API has no independent
metadata byte budget; use max_input_bytes instead.

## Results

Encode/JPEG/PNG: data holds file bytes, JSON {operation,bytes,...}. Encoded JPEG
pixel input reports jpeg_input:true and is NOT byte-reversible transcoding.
Decode/read_image: top metadata plus frames:[{index,width,height,channels,dtype,
offset,length,duration,name,complete,extra_channels:[...],header:{...}}].
Main data is interleaved; extras are packed planes with offset,length,width,
height,channels=1,dtype,type,name,bits_per_sample,exponent_bits_per_sample,
dim_shift,alpha_premultiplied,spot_color,cfa_channel. Decoded JXL extra planes
are full-size even when source dim_shift is nonzero. Duration is raw ticks.

Info/decode top metadata: width,height,bits_per_sample,exponent_bits_per_sample,
orientation,extra_channels,animation (null or {tps_numerator,tps_denominator,
num_loops,have_timecodes}),icc (byte array or null),output_icc,boxes
({type,data} metadata arrays),bytes_read,complete. Info frames and decode
frame.header share {index,name,width,height,duration,is_last,is_preview,timecode,
layer_info:{have_crop,crop_origin:[x,y],width,height,blend_info:{mode,source,
alpha_channel,clamp},extra_channel_blend_info:[...],save_as_reference,
save_before_color_transform}}. Blend mode strings use backend Debug names.

color_encoding/output_color_encoding are readable objects: kind=icc, or
kind=structured with color_space,white_point,primaries,transfer_function,
rendering_intent. JXL standard profiles also expose name=srgb/linear_srgb and
numeric chromaticities. No color conversion is implied. Spot colors remain
separate planes and are not painted into the returned RGB values.

PNG export keeps the output ICC and unapplied orientation, rejects premultiplied
or mismatched-depth alpha, and rejects ancillary JXL boxes instead of dropping
metadata. Native PNG export requires exactly 8- or 16-bit integer depth. Explicit
decode format conversion is caller opt-in; no native depth is implicitly narrowed.
