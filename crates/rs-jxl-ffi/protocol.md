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
| `encoder` | `encode`, `encode_multiframe`, `encode_layers`, `transcode_jpeg`, `analyze_multiframe` | none |
| `decoder` | `decode`, `decode_to_png`, `reconstruct_jpeg` | `info`, `read_image` |

Other kind/operation combinations fail. The stateless operations above do not
reuse the context's workers or parser state. Their normal field restrictions
remain in force: `read_image` rejects an explicitly supplied nonnull `threads`
option. `info` inherits the thread count as usual. `analyze_multiframe` inherits
the encoder context's threads, reuses its worker pool, and rejects an explicit
thread-count mismatch just like encoding.
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

Unknown, duplicate and inapplicable fields fail. Optional fields may be omitted
or null, except `options`, `analysis_options`, and `layer_options` must be objects
when supplied. Supplied members of `auto_options`/`analysis_options` must have
their declared types (not null); layer field types are specified below.
Top-level keys: operation (required), options (default {}), images (default []),
durations, reference, timebase, loops, auto, auto_options, analysis_options,
layer_options.
Sequence keys only for encode_multiframe; analysis_options only for
analyze_multiframe; layer_options only for encode_layers.
Operations: encode, encode_multiframe, encode_layers, analyze_multiframe, transcode_jpeg,
reconstruct_jpeg, decode, decode_to_png, info, read_image.
encode requires one image; multiframe and encode_layers require nonempty images. Other operations
consume entire data buffer and reject images. info is JXL header inspection;
read_image is shared PNG/JPEG pixel input decoding.

durations defaults to one tick per image; supplied length must equal image count,
zeros remain zeros. reference=previous (full/previous/first/smallest_crop),
timebase=[1000,1] (ticks per second numerator/denominator), loops=0, auto=false.
encode_multiframe always sets animation metadata. auto uses the native automatic
multiframe search, not still-image source-search.

### Heterogeneous Layers

`encode_layers` encodes original-size pixel images on a shared canvas. Inputs
may have different dimensions, but native validation requires compatible image
metadata, sample formats, color encodings and extra-channel layouts. Input
decoding and native pixel words follow the ordinary ImageSpec contract below;
there is no padding, resizing, color conversion, or synthetic canvas frame.

`layer_options` is a top-level object, not a member of `options`, and is valid
only for this operation. Omission or `{}` uses native defaults; null is invalid.
The legacy top-level keys `durations`, `reference`, `timebase`, `loops`, `auto`,
`auto_options`, and `analysis_options` are rejected even when explicitly null.

| Member | Type | Default |
| --- | --- | --- |
| `canvas` | `[u32, u32]` or null | inferred |
| `frames` | array of frame objects or null | defaults for every input |
| `animation` | animation object or null | native inference |

An explicit `frames` array must contain exactly one object per input image; an
empty array is invalid for the required nonempty image sequence. Omitted/null
frames map to an empty native vector, meaning defaults for every input. Canvas
inference uses maximum positive layer extents, including origins. Negative
origins require an explicit canvas. The canvas clips composited output only;
non-coalesced decoding retains original layer dimensions and sample words.
Canvas validity, metadata compatibility, frame/reference semantics and timing
are checked by native code before encoding.

Each frame object accepts only:

| Member | Type | Default |
| --- | --- | --- |
| `origin` | `[i32, i32]` | `[0, 0]` |
| `duration` | u32 | 0 |
| `timecode` | u32 | 0 |
| `name` | string | `""` |
| `blend` | blend object | defaults below |
| `extra_blend` | array of blend objects | `[]` |
| `save_as_reference` | u8 | 0 |

A blend object accepts only `mode` (string: `replace`, `add`, `blend`,
`alpha_weighted_add`, or `multiply`; default `replace`), `source` (u8, default 1),
`alpha_channel` (u8, default 0), and `clamp` (boolean, default false). Every blend
object, including entries of `extra_blend`, uses these defaults. An empty
`extra_blend` applies the color blend rule to every extra channel. Native checks
reference slots, alpha-channel applicability, extra-blend count, and other
format constraints; the u8 wire type does not make every value a valid slot.
Neither `reference_only` nor `save_before_color_transform` is exposed here.

Default source 1 preserves a zero background independently for each layer.
Nonfinal zero-duration layers implicitly save slot 0 when `save_as_reference`
is 0, leaving slot 1 untouched. An explicit source 0 reuses that composited
state; explicit saves may overwrite other slots. Uninitialized reference slots
have the standard zero background. Zero duration is never replaced with one.

An animation object accepts only `ticks_per_second_numerator` (u32, default
1000), `ticks_per_second_denominator` (u32, default 1), `loops` (u32, default 0),
and `timecodes` (boolean, default false). Numerator/denominator must satisfy
native animation limits. `{}` explicitly enables default animation metadata;
omission/null leaves inference to native code, which adds default timing when
durations/timecodes require it and enables timecodes when needed. Nonzero
timecodes conflict with an explicit animation whose `timecodes` is false.

All layer/frame/blend/animation objects reject unknown and duplicate keys;
positional arrays are not objects. Frame, blend, and animation members cannot
be null. Integer fields reject booleans, fractions, strings, and out-of-range
values. Only the three optional `layer_options` members allow null.

```json
{
  "operation": "encode_layers",
  "options": {"effort": 2},
  "layer_options": {
    "canvas": [3, 2],
    "frames": [{"origin": [-1, 0]}, {"origin": [2, 0]}]
  },
  "images": [
    {"width": 2, "height": 1, "channels": 1, "dtype": "u8", "length": 2},
    {"width": 1, "height": 2, "channels": 1, "dtype": "u8", "offset": 2, "length": 2}
  ]
}
```

### Automatic Multiframe Options

`auto_options` is a top-level object, not a member of `options`. It is accepted
only for `operation="encode_multiframe"` with `auto=true`. Omission/null retains
native defaults; an empty object explicitly requests those defaults and still
requires `auto=true`. The outer `reference` remains authoritative; a nested
`auto_options.reference` is rejected even if it agrees with the outer value.

| Member | Type | Native Default |
| --- | --- | --- |
| `allow_mask` | boolean | true |
| `allow_multi_rect` | boolean | true |
| `tile_size` | positive u32 | 32 |
| `max_rectangles` | positive usize | 32 |
| `prune` | boolean | true |
| `adaptive_rectangles` | boolean | true |
| `max_frames` | nonnegative usize | usize::MAX |
| `max_total_trials` | nonnegative usize | usize::MAX |
| `max_trial_pixels` | nonnegative u64 | u64::MAX |

Missing members are filled from Rust `multiframe::AutoOptions::default()`, not
Python defaults. Serde rejects unknown/duplicate members, null member values,
numeric booleans, fractional/string integers and values outside native ranges.
Geometry must be positive even if rectangle candidates are disabled. Zero work
budgets are valid requests that explicitly fail when work exceeds them; they do
not mean unlimited. These controls add no new limits to the codec.

`prune` stops a candidate only when a proven byte lower bound cannot win under
the canonical tie rule; it does not remove existing eligible candidates using
an estimate. `adaptive_rectangles` permits additional `MergedPrevious` and
`MergedFirst` proposals when multi-rectangle candidates are enabled. Their cost
estimate proposes geometry only; selection still uses actual completed encodes.
All difference analysis, candidate generation, pruning and selection run in Rust.

### Multiframe Analysis Options

`analysis_options` is a top-level object accepted only for
`operation="analyze_multiframe"`. Omission or `{}` uses native
`multiframe::AnalysisOptions::default()`; null is invalid. This operation analyzes
matching complete image states, not encoded layers, and performs no trial encodes.

```json
{
  "operation": "analyze_multiframe",
  "options": {"threads": 2, "strip_metadata": false},
  "analysis_options": {"reference": "smallest_crop", "tile_size": 32},
  "images": [
    {"width": 2, "height": 1, "channels": 1, "dtype": "u8", "length": 2},
    {"width": 2, "height": 1, "channels": 1, "dtype": "u8", "offset": 2, "length": 2}
  ]
}
```

| Member | Type | Native Default |
| --- | --- | --- |
| `reference` | `full` / `previous` / `first` / `smallest_crop` | `smallest_crop` |
| `allow_mask` | boolean | true |
| `allow_multi_rect` | boolean | true |
| `adaptive_rectangles` | boolean | true |
| `tile_size` | positive u32 | 32 |
| `max_rectangles` | positive usize | 32 |
| `rectangle_bytes_per_pixel` | finite nonnegative number | 1.0 |

Unknown/duplicate members, null members, numeric booleans, fractional/string
integers, and values outside native ranges fail. Geometry remains positive even
when optional candidates are disabled. Zero `rectangle_bytes_per_pixel` is valid;
the value controls native rectangle-merge proposals only and is not a measured or
predicted compression ratio. The only accepted `options` are `threads` (1..256,
default 1 stateless, inherited for contexts) and `strip_metadata` (default false).
`analysis_options.threads`, nested `options.analysis_options`, encoding effort,
tool overrides, `auto_options`, and encoding-only sequence fields are rejected.
The analysis reference is nested here, not in the outer encoding `reference`.

### Image Inputs

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

- encode/encode_multiframe/encode_layers: threads=1, effort=7, source_search=false,
  strip_metadata=false; optional group_size_shift,rct,predictor,palette_colors,
  patches(bool),learning_fraction,ma_max_leaves. Effort/overrides map to native
  candidates. Unsupported projections fail. source_search only for still images,
  cannot combine with explicit tool overrides (ma_max_leaves is allowed).
  encode_layers rejects source_search=true; it has no automatic strategy search.
- analyze_multiframe: threads=1, strip_metadata=false; geometric controls are in
  top-level analysis_options. Returns a native geometric report and empty data;
  see Multiframe Analysis Results below.
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

### Layer Results

`encode_layers` returns JXL bytes in data and this JSON shape:

```json
{
  "operation": "encode_layers",
  "bytes": 123,
  "jpeg_input": false,
  "canvas": {"width": 3, "height": 2},
  "codec_seconds": 0.001,
  "layers": [
    {
      "index": 0, "width": 2, "height": 1, "origin": [-1, 0],
      "duration": 0, "timecode": 0, "name": "",
      "blend": {"mode": "replace", "source": 1, "alpha_channel": 0, "clamp": false},
      "extra_blend": [], "save_as_reference": 0
    }
  ]
}
```

The example shows one layer entry; actual results contain exactly one per input,
in input order, using native result dimensions and frame settings. `bytes` is the
data length, not a per-layer estimate. `codec_seconds` is native planning/coding
time, excluding pool construction, input decoding and JSON transport. `jpeg_input`
is true if any input was decoded from JPEG pixels, not a byte-transcode claim.
Blend mode strings here are snake_case, unlike legacy decode/info Debug names.
Encoder contexts reuse their workers and enforce the ordinary thread-count
inheritance/mismatch rules; no layer settings persist between calls. Decoder
contexts reject this operation. The C ABI layout and ownership rules are unchanged.

### Multiframe Analysis Results

Analysis returns empty data and a JSON object containing:

- `operation`: `analyze_multiframe`.
- `width`, `height`, `channels`, `frame_count`: native canvas and input counts;
  channels include extra channels.
- `reference`: selected analysis reference, using the option strings above.
- `crop_supported`: whether native cropped replacement is supported; subsampled
  extra channels require complete-frame replacement.
- `rectangle_bytes_per_pixel`: the rectangle-proposal parameter used, not an
  encoded-size result.
- `frames`: one entry per input, in input order, with the fields below.
- `reference_modes`: one summary each for `full`, `previous`, `first`, and
  `smallest_crop`, regardless of the selected analysis reference.
- `best_geometric_reference`: the native minimum-total-bbox-area reference, not
  a compressed-size winner or encoding recommendation.
- `analysis_seconds`: native `analysis_elapsed.as_secs_f64()`, excluding input
  decoding, context creation and JSON transport work.

Every `frames[]` entry contains:

- `index`: zero-based input index.
- `previous`, `first`: backward-compatible `null` or `{x,y,width,height}` bounds.
  Both are the full canvas for frame zero; null later means identical pixels.
- `previous_stats`, `first_stats`: null for frame zero, otherwise objects with
  `bounds`, `changed_pixels`, `changed_ratio`, `bbox_pixels`, `bbox_ratio`,
  `bbox_fill_ratio`, and `mask_eligible`. Counts are geometric pixel areas, not
  sample/byte counts. Changed and bbox ratios divide by the canvas area; fill
  ratio divides changed area by bbox area and is zero for identical frames.
  Comparisons include invisible RGB, exact float words, and extra channels.
  `mask_eligible` reports native binary-alpha eligibility, not candidate selection.
- `plan`: `{reference_index,source,region,changed,layer_pixels,layer_area_ratio}`
  for the chosen reference. `reference_index` is a logical input index or null;
  `source` is the native JXL reference slot. `region` is always a rectangle;
  `changed` is a rectangle or null. Identical cropped frames retain the native
  1x1 placeholder layer, so zero changed pixels need not mean zero layer area.
- `candidates`: geometric alternatives with
  `{strategy,reference_index,source,rectangles,layer_pixels,layer_area_ratio,layer_count}`.
  `rectangles` is an array of `{x,y,width,height}`. Strategies are snake_case:
  `full`, `crop_previous`, `crop_first`, `mask_previous`, `mask_first`,
  `tiles_previous`, `tiles_first`, `merged_previous`, `merged_first`.
  Candidate area is the sum of submitted rectangle areas; mask candidates still
  submit the full bounding rectangle, not only changed pixels. Per-frame layer
  area ratios divide by the canvas area. No candidate is trial-encoded.

Each `reference_modes[]` entry has `reference`, `layer_pixels`, `layer_area_ratio`,
`average_changed_ratio`, `crop_frames`, `identical_frames`, and `suitability`.
Total area includes the initial full frame and each subsequent planned bbox or
placeholder. Its ratio divides by `width * height * frame_count`.
`average_changed_ratio` excludes the initial frame (zero for single-frame input);
full-frame replacement treats subsequent pixels as new. Crop/identical counts
also exclude the initial frame. Suitability is `single_frame`, `strong`,
`moderate`, `limited`, or `full_frames_required`, as classified by the native
analyzer. These are geometric reuse labels, not speed or compression guarantees.
No estimated encoded sizes, savings, or measured encoding throughput are returned.

### Automatic Multiframe Results

Automatic multiframe JSON includes `frames`, one entry per logical input:

- `index`: zero-based logical input index.
- `winner_index`: zero-based index into `candidates`, never into `pruned` or an
  execution-order list. Completed candidates retain canonical order, even when
  cheap candidates execute first. The new Merged candidates follow all legacy
  candidates, preserving existing equal-byte preferences.
- `candidates`: completed trials only, each with `strategy`, `encoded_bytes`,
  `encoded_frames`, `trial_pixels`, `elapsed_seconds`. Byte counts are actual
  layer sizes including headers/private patch frames, excluding shared
  image/container headers. Layer counts exclude private reference-only frames.
- `pruned`: separate records with `strategy`, `lower_bound`, `encoded_frames`,
  `planned_frames`, `trial_pixels`, `elapsed_seconds`. `lower_bound` is a proven
  byte bound, never an actual size or an `encoded_bytes` field. `encoded_frames`
  counts only fully finished regular layers, not a partially encoded layer;
  `planned_frames` is the candidate's total planned regular-layer count.
- `analysis_seconds`: native elapsed difference/planning time for this input.
- `differences`: native `{reference,changed_pixels,bounds}` entries for analyzed
  references. `reference` is a zero-based logical input index, not a JXL reference
  slot. `bounds` is null for no differences, otherwise `{x,y,width,height}`.

All time fields above are native `Duration::as_secs_f64()` seconds, not Python
wall-clock measurements. `trial_pixels` counts pixel areas submitted to trial
encoding, including work for an interrupted layer, not analysis scans or just
the winner. An empty difference list means no reference analysis was needed;
it does not assert that the frame equals another frame. Strategies use Rust
Debug names, including `MergedPrevious`/`MergedFirst`. With `prune=false`, all
eligible trials complete or the operation fails; `pruned` is empty. No throughput
or compressed-size improvement is claimed by these report additions.

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
