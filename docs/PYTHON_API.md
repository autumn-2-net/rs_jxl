# Python API and Packaging

Distribution `rs-jxl`, import `rs_jxl`, Python 3.10+. The C++ `jxlpy` package
can coexist. There is no C++ linkage, CLI subprocess codec or Pillow dependency.
Only supported lossless Rust controls are accepted; this is not full libjxl API
parity. Unknown and inapplicable settings fail instead of being ignored.

## Installation

```powershell
cargo fetch --locked
cargo build --locked --offline --release -p rs-jxl-ffi
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
```

Fetch needs network only for uncached locked dependencies. Packaging defaults
to a locked/offline release build. Install the resulting platform wheel with
pip: users need cffi/numpy, not Rust. Build requirements are in pyproject.toml.
Install Torch separately using the desired CPU/CUDA build.

Set `RS_JXL_NATIVE_LIB` to a prebuilt `rs_jxl_ffi.dll` (Windows),
`librs_jxl_ffi.so` (Linux), or `librs_jxl_ffi.dylib` (macOS) to skip compilation.
The packager checks native format/architecture, not by executing the library.
Wheels are py3-none-platform, never pure-any. Foreign cross-builds are rejected;
Linux manylinux repair remains release work. macOS runtime is out of scope.
Development/editable imports use PYTHONPATH=python and target/release, honoring
CARGO_TARGET_DIR/CARGO_BUILD_TARGET. Wheels use their bundled library; explicit
RS_JXL_NATIVE_LIB overrides either lookup and fails on an invalid path/ABI.

## Data Contract

- Inputs: paths, encoded bytes/bytearray/memoryview, binary readable streams,
  NumPy arrays or Torch tensors. Streams are read at their current position.
- Auto layout: HWC for NumPy, CHW for Torch; HW gray accepted. Explicit hwc/chw
  overrides this. Noncontiguous input is copied without narrowing.
- Storage: uint8/uint16/uint32 (declared 1..31 bits), float16/float32. Signed
  integers/float64 fail. Endianness is normalized without changing samples.
- Outputs: owned writable HWC arrays/tensors, including C=1 gray. out="bytes"
  is packed little-endian samples. out="torch" returns CPU unless device="cuda".
- GPU input requires allow_device_copy=True. Codec execution is CPU; tensors
  detach from autograd. No output points into freed Rust memory.
- output=None returns bytes; a path writes and returns Path; a binary writable
  stream receives all bytes and returns None. Encoding finishes before writing;
  I/O failure can leave partial output. Encoded inputs are fully buffered.

### Input Copies and Borrowing

`encode`, `encode_layers`, `encode_multiframe` and `analyze_multiframe` default to
`borrow_inputs=False`: mutable pixel buffers, including CPU tensors and read-only
views of mutable storage, are
snapshotted before the native call. Files/streams are read completely into memory;
this is not a file mapping or incremental input API. Immutable bytes may be
retained directly. Mutable encoded input passed through `read_bytes` is copied.

`borrow_inputs=True` permits borrowing eligible contiguous input buffers for
one synchronous call. The caller must not mutate the source or any alias until
the call returns, including from other threads while CFFI releases the GIL.
Marking one view read-only does not prevent mutation through another alias.
Exporters are retained through the call. This is best-effort transport borrowing,
not end-to-end zero-copy encoding: layout/byte-order normalization, GPU-to-CPU
transfer, multiple frames/extra planes consolidated into one payload, and Rust
image preparation can still allocate and copy. Borrowing does not bypass dtype,
layout, metadata or resource checks.

## Still Images

```python
import numpy as np
import rs_jxl

rgba = np.zeros((64, 96, 4), dtype=np.uint16)
rgba[..., 0] = 12345  # retained under zero alpha
data = rs_jxl.encode(rgba, effort=9, threads=8)
assert np.array_equal(rs_jxl.decode(data), rgba)
archived = rs_jxl.encode("input.jpg", effort=9)
original = rs_jxl.reconstruct_jpeg(archived)
pixel_jxl = rs_jxl.encode("input.jpg", lossless_jpeg=False, strip_metadata=True)
```

encode defaults: effort=7, threads=1, lossless=True, distance=0,
lossless_jpeg=True, source_search=False. JPEG bytes default to coefficient
transcoding; unsupported structures fail without pixel fallback.
transcode_jpeg is explicit, defaults effort=9, and exposes source_search,
brotli_quality, no_cfl, order=source/default/adaptive, limits. decode_to_jpeg
aliases reconstruct_jpeg: original bytes only, not lossy JPEG encoding.

PNG/JPEG pixel input shares the CLI implementation. PNG retains 8/16-bit
gray/GA/RGB/RGBA, including invisible RGB. JPEG pixel losslessness is relative
to its decoder's samples. Unsupported metadata requires strip_metadata=True;
supported color metadata remains. No EXIF orientation transform is applied.
Arrays accept bits_per_sample, color_encoding=srgb/linear_srgb, icc_profile
(bytes/path), orientation=1..8; explicit color and ICC are mutually exclusive.
Encoded input cannot apply array metadata overrides. Lossy requests fail.

## Encoder Parameters

Effort 1..11 uses the Rust source backend; source_search=True enables the still
scheduler (e11 tournament). Same effort does not promise C++ size/speed parity.
threads=0 explicitly selects CPU count capped at 256; threads=1 is serial.

encoder_options={...} or keyword arguments accept group_size_shift, rct,
predictor, palette_colors, patches, learning_fraction and ma_max_leaves.
Aliases modular_group_size/modular_colorspace/modular_predictor/
modular_palette_colors map to the first four respectively. iterations and
modular_ma_tree_learning_percent map to learning_fraction=percent/100.

```python
data = rs_jxl.encode(
    "content.png", effort=9, modular_group_size=3, iterations=100,
    modular_predictor=0, modular_palette_colors=10000,
    patches=False, modular_colorspace=0,
)
```

Duplicate controls fail. source_search rejects explicit tool overrides because
its candidates cannot guarantee them. ma_max_leaves is an explicit resource
ceiling, not an implicit quality reduction. Raw cjxl flags and numeric libjxl
frame-setting IDs are not forwarded to another encoder.

## Reports

`encode(..., return_info=True)` and `transcode_jpeg(..., return_info=True)`
return `(encoded_output, meta)`. The first element keeps the ordinary bytes,
Path or None behavior of `output`; codec statistics are under `meta["report"]`.
This also applies when `encode` selects JPEG byte transcoding.

```python
data, meta = rs_jxl.encode(rgba, effort=9, source_search=True, return_info=True)
report = meta["report"]
print(report["route"], report["bytes"], report["codec_seconds"])
```

Still-image reports expose `route`, `implementation`, `pixels`, `bytes`, `bpp`,
`codec_seconds`, `mpix_per_second`, `full_encodes`, `probe_encodes`,
`winner_index`, `effective_parameters` and `preparation_reuse`. Optional values
can be None; a direct candidate has no search winner index. The Python projection
does not include the complete Rust search transcript. `implementation` is the
stable lowercase string `modular` or `fast_lossless`. When effective parameters
are present, their `patches` value is `auto`, `on` or `off`, not a Rust Debug name.
`effective_parameters` is a scheduler `CandidateParams` projection, not a full
backend recipe or exact tool list; FJXL uses its own default effort.
JPEG reports instead provide `route` (`jpeg_transcode` or `jpeg_search`), `codec_seconds`,
`backend_effort`, `winner_index`, `full_encodes` and `trials` with each trial's
stage, backend effort and bytes. Output size also remains in top-level
`meta["bytes"]`; do not assume identical report fields for both operations.

`codec_seconds` excludes worker-pool creation, external pixel-input decoding,
Python/FFI transport and external I/O. For JPEG transcoding it includes JPEG
parsing and coefficient preparation. It covers codec work for the call, including
search when requested, not an exact analysis/backend breakdown. Still-image bpp is
final output bits per original pixel; MPix/s uses the original pixel count and
codec time, not the sum of candidate/probe pixels. These are diagnostics, not
performance-parity claims. Multiframe `return_info` retains its separate
strategy metadata described below; no universal timing report is implied.

## Reusable Contexts

```python
with rs_jxl.Encoder(threads=8) as encoder, rs_jxl.Decoder(threads=8) as decoder:
    data, meta = encoder.encode(rgba, effort=9, return_info=True)
    pixels = decoder.decode(data)
    archived = encoder.transcode_jpeg("input.jpg", effort=7)
    original_jpeg = decoder.reconstruct_jpeg(archived)
```

Constructors accept `threads=1`; `threads=0` resolves the CPU count capped at
256, as in the one-shot API. `Encoder` supplies `encode`, `transcode_jpeg`,
`encode_layers`, `encode_multiframe` and `analyze_multiframe`. `Decoder` supplies `decode`,
`decode_into`, `decode_layer`, `decode_extra_channel`, `info`, `reconstruct_jpeg`
and `decode_to_png`. Options
and return values match the corresponding module-level functions.

Thread counts are fixed for a context. Methods inherit that count, and an
explicit `threads` value must normalize to the same count. All other options
are per-call; effort, limits, frame selection and output format from one call do
not become defaults for the next. Contexts reuse workers, not input buffers,
parser state, decoded frames, reference slots or candidate caches. One thread
does not create a pool. `info` and PNG/JPEG pixel reading are stateless and do
not reuse JXL decoder workers. Context-bound `decode`/`decode_into` still send
PNG/JPEG `read_image` requests through the context's closed/lifecycle guard.

`close()` is idempotent; `.threads` and `.closed` expose the context's settings
and lifetime. Context-manager exit closes it, and operations after close raise
RuntimeError. The context lock serializes the closed check and native execution
against close; it is released before Python result conversion or copying.
Conversion callbacks can close or reenter the context without holding that lock,
and the current result retains independent ownership. This is not whole-method
serialization or protection against concurrent mutation of caller buffers.
Independent contexts can run concurrently; avoid multiplying outer workers by large inner
thread counts. Ordinary request/codec failures permit reuse. A native panic
poisons the handle; close it rather than attempting more work. Context creation
requires the optional native context symbols and raises ImportError when an
older ABI-v1 library lacks them; one-shot ABI compatibility is unchanged.

## Extra Channels

```python
depth = np.zeros((64, 96), dtype=np.uint16)
data = rs_jxl.encode(rgba, effort=3, extra_channels=[
    {"data": depth, "type": "depth", "name": "distance"},
])
pixels, details = rs_jxl.decode(data, return_extra_channels=True)
depth_back = details["extra_channels"][0]["data"]  # HxWx1
```

Explicit planes append after interleaved alpha. Specs expose type/name/depth,
dim_shift, alpha_premultiplied, spot_color, cfa_channel where applicable.
Input subsampled size is ceil(width/2**dim_shift) x ceil(height/2**dim_shift).
Rendered decoded planes are full resolution; metadata keeps original dim_shift.
return_extra_channels=True returns (pixels, metadata); alpha is excluded unless
include_alpha_extra=True. decode_extra_channel(..., channel=N) uses the complete
native index including alpha. Mixed plane dtypes remain distinct.

## Frames and Layers

```python
changed = rgba.copy()
changed[12:20, 30:40, 0] = 7
data = rs_jxl.encode_multiframe(
    [rgba, changed], durations=[10, 10], reference="previous", effort=3,
)
second = rs_jxl.decode(data, frame=1)
all_frames = rs_jxl.decode(data, frame=None)
analysis = rs_jxl.analyze_multiframe([rgba, changed])
bounds = analysis["frames"][1]["previous"]
```

reference=previous/first/full/auto (none aliases full). Default strategy="crop"
uses geometry; reference=auto chooses the smaller first/previous bbox, not
measured bytes. strategy="auto" additionally encodes eligible full/crop/mask/
multiple-rectangle candidates and retains byte winners. Proven-bound pruning
and additional adaptive rectangle proposals are enabled by native defaults.
return_info=True returns (encoded_output, strategy_report).
The strategy report is the metadata dictionary itself, with `frames` candidate
details for strategy="auto"; it is not the still/JPEG `meta["report"]` schema.
`borrow_inputs` has the same snapshot/borrowing contract as still-image encode,
though consolidation of several buffers can require copying.

```python
data, report = rs_jxl.encode_multiframe(
    [rgba, changed], strategy="auto", reference="previous", effort=3,
    auto_options={"prune": True, "adaptive_rectangles": True, "tile_size": 16},
    return_info=True,
)
frame_report = report["frames"][1]
winner = frame_report["candidates"][frame_report["winner_index"]]
```

`auto_options=None` is the default. A mapping, including `{}`, is legal only
with `strategy="auto"` on `encode_multiframe` (including the `Encoder` method).
It is forwarded as a structured top-level native request field, separate from
`encoder_options`. Omitted entries use Rust defaults; Python does not fill in
search defaults or perform difference analysis, planning or selection.

| Auto Option | Accepted Value | Native Default |
| --- | --- | --- |
| `allow_mask` | bool | true |
| `allow_multi_rect` | bool | true |
| `tile_size` | positive u32 integer | 32 |
| `extra_tile_sizes` | list/tuple of positive u32 integers | [] |
| `max_rectangles` | positive native usize integer | 32 |
| `prune` | bool | true |
| `adaptive_rectangles` | bool | true |
| `retained_reference` | bool; true requires `reference="auto"` | false |
| `reference_lookahead` | bool; true requires `retained_reference=True` | false |
| `max_frames` | nonnegative native usize integer | usize::MAX |
| `max_total_trials` | nonnegative native usize integer | usize::MAX |
| `max_trial_pixels` | nonnegative u64 integer | u64::MAX |

Unknown options and a nested `reference` are rejected; only the explicit outer
`reference` argument controls references. Boolean options require Python booleans,
not integers or strings. Integer options accept integer-index values (including
NumPy integers), not booleans, floats or strings; explicit `None` member values
are invalid. Geometry remains positive even if multi-rectangle candidates are
disabled. Zero budgets are forwarded unchanged and cause explicit native resource
failures when work exceeds them. Omitting budgets uses native unlimited defaults,
not new Python limits. Direct ABI requests also enforce these types with serde
and reject duplicate members.

Automatic frame reports distinguish completed work from pruning:

- `candidates` contains only completed trials in canonical order. Entries have
  `strategy`, actual `encoded_bytes`, `encoded_frames`, `trial_pixels` and
  `elapsed_seconds`, plus `reference_index`, `source` and `save_as_reference`.
  `winner_index` indexes this list, not execution order.
- `pruned` contains `strategy`, proven `lower_bound`, fully finished
  `encoded_frames`, `planned_frames`, `trial_pixels` and `elapsed_seconds`, plus
  `reference_index`, `source` and `save_as_reference` for the proposed candidate.
  These entries do not have `encoded_bytes`; a lower bound is not an actual size.
- `analysis_seconds` is native difference/planning time. `differences` contains
  native `{reference, source, changed_pixels, bounds}` records, with zero-based
  input reference indices, JXL source slot IDs and null bounds for no changes.
  An empty list means no
  reference analysis was needed.
- `retained_reference` reports whether the opt-in policy is active for this
  sequence. `references_before` and `references_after` are four-element arrays
  mapping JXL slots 0..3 to complete logical input frame indices. JSON null means
  blank/unknown, partial intermediate state, or scratch data, not a complete
  input frame. Slot 3 is reserved for patch scratch data and reported as null.
- `reference_lookahead` is a boolean indicating the opt-in next-input policy.
  `lookahead_comparisons` is a nonnegative native `usize` count of newly computed
  exact future `DiffMap`s during this logical frame, at most 3. It is zero when
  disabled or there is no next input. Ordinary current-input/static analysis and
  reads of previously cached maps do not count as lookahead computations.
- `reused_payloads` (native `usize`) and `reused_payload_pixels` (u64) count cached
  unmasked crop payload hits and the sum of their rectangle areas. Reuse is
  automatic for equivalent frame payloads for repeated identical rectangles
  within the same logical input; there is no reuse option. It does not cross
  logical inputs or calls, reuse masked payloads, or require unsafe patch-mode
  reuse. Patch mode may conservatively bypass the cache. `trial_pixels` still
  counts every submitted rectangle's area, including cache hits; reuse counters
  do not measure skipped analysis scans or encoded-byte savings.

Timing values are native elapsed seconds, not Python measurements. Trial pixels
are areas submitted to the encoder, including interrupted layers, not analysis
scans. Regular-layer counts exclude private reference-only patch frames; in a
pruned record they also exclude a partially encoded layer. Candidate bytes include
layer headers/private patch frames but exclude shared image/container overhead.

Pruning can reorder execution, but only proven byte bounds can stop losing trials;
it preserves selection and stable ties for the same eligible set with sufficient
budgets. `prune=False` completes all eligible candidates or fails. Adaptive
`MergedPrevious`/`MergedFirst` proposals follow all legacy candidates and are
selected only by actual encoded bytes. Their cost estimate uses the initial full
frame's measured bytes per pixel only to propose geometry. Disable them with
`adaptive_rectangles=False`; `allow_multi_rect=False` disables both tile and merged
proposals. No performance or compression improvement is claimed yet.

### Retained Reference Mode

```python
data, report = rs_jxl.encode_multiframe(
    [scene_x, scene_a, scene_b, scene_a_updated],
    strategy="auto", reference="auto", effort=9,
    auto_options={"retained_reference": True}, return_info=True,
)
frame = report["frames"][3]
winner = frame["candidates"][frame["winner_index"]]
reference_index = winner["reference_index"]  # logical input index, or None for Full
source_slot = winner["source"]             # JXL slot, not an input index
```

This policy is opt-in and operates entirely in native Rust. It requires both
`strategy="auto"` and `reference="auto"`; conflicting choices are rejected before
Python consumes the image iterable. Explicit false keeps the existing policy.
It also works through `Encoder.encode_multiframe`; reference images are retained
only within one call, never between context calls.

Two saved canvas slots (1 and 2) rotate. With `reference_lookahead=False` (the
default), when a delta wins, its source reference
is retained and the current complete state is stored in the other slot. When a
full frame wins, the previous input is retained and the older slot is evicted.
Thus an older recurring state can remain useful beyond the immediately preceding
frame. The policy can evict the original first frame; it is not an unlimited
history or a guarantee of a smaller whole file. Default
`retained_reference=False` continues comparing the original first/previous
references as before.

To opt into one-input lookahead, use
`auto_options={"retained_reference": True, "reference_lookahead": True}`.
`reference_lookahead` requires a Python `bool`; true without retained references
is rejected before consuming frames. Retained references still require
`strategy="auto", reference="auto"`. Omission or false leaves the prior
retention policy unchanged.

Native Rust compares the currently saved complete states against the **next**
input using exact full-pixel `DiffMap`s, ranking smaller bounding-box area first,
then fewer changed pixels. It changes which old state survives only when
strictly better than the old policy's choice; ties keep that choice. The current
input is also compared with the next input so its map is ready after saving.
At most one future input is examined, with at most three newly computed future
maps. Maps for the states actually retained are cached for the next iteration,
which reuses at most two retained-map reads. No future input is trial-encoded.
All channels, invisible RGB, extra planes and exact floating-point sample words
participate; metadata compatibility rules are unchanged.

This can retain A across `A, B, B2, A2` even when B2 is encoded from B: B's slot
may be overwritten only after blending. A multi-rectangle candidate reads the
original source for its first rectangle, then its destination's intermediate
composite for subsequent rectangles. The heuristic is opt-in because better
next-input geometry does not guarantee a smaller next encode or whole sequence.
Reports expose the policy and computation counters above, not predicted savings.

Older-reference trials report `CropRetained`, `MaskRetained`, `TilesRetained` or
`MergedRetained`; strategy alone does not identify the source. Use
`reference_index` and `source`, and consult `references_before` to see committed
state. Completed losing candidates and pruned candidates do not change the live
slots. `save_as_reference` is the candidate's final layer save slot, zero for the
last stream frame. Multi-part candidates can save intermediate layers even when
that final value is zero. Such a partial canvas is null in `references_after`,
not advertised as the completed input. These arrays report actual committed
complete states, not hypothetical trial saves.

The mode reuses existing standard JXL reference and blending mechanisms. It
does not introduce a private bitstream extension or displaced-block/temporal
patch search. Crop/mask/tile/merged trials retain their existing eligibility
rules, including exact hidden RGB handling and mask restrictions.
`analyze_multiframe` remains a static first/previous geometry analyzer: it does
not run encodes or know which candidates will win and which slots will survive,
so it accepts neither `retained_reference` nor `reference_lookahead` and does not
predict this dynamic policy.

### Multiframe Sample Semantics

All channels including hidden RGB participate in differences. durations is a
tick count or list, default 1; tps=(1000,1), loops=0. Zero is a true layer, not
a fabricated display duration; visible-frame count can differ from layer count.
`decode_layer(data, layer=N)` defaults to `return_info=True` and returns
`(raw_layer_pixels, metadata)`, with crop, blend and references under `header`.
Use `adjust_orientation=False` for original stored coordinates and sample order
when orientation is not 1. This is not signed arithmetic residuals.
Multiframe extra data accepts a shared plane or list/tuple of one plane per
input. `analyze_multiframe` returns exact previous/first bounds and geometry-only
statistics/plans, not byte estimates (see below).
JPEG frame input remains pixel-lossless relative to the input decoder, not
byte-reversible; use `transcode_jpeg`/`reconstruct_jpeg` for original JPEG bytes.

### Geometry-Only Analysis

Multiscale partitions are opt-in with
`analysis_options={"extra_tile_sizes": [8, 16, 64]}` (and the same key in
`auto_options` for encoding). The base grid remains eligible. Duplicate grids
and equivalent ordered rectangle plans per reference are skipped in native
code. Additional candidates may improve size but increase encoding work.
The `tile_size` candidate field records the proposing grid or None for
full/crop/mask. Geometry minima do not imply minimum encoded size.

`history_window=8` inspects preceding input positions within distance 8, excluding
already compared first/previous inputs. It defaults to 0 (disabled). The report's
`frames[i]["history"]` entries describe potential older references; they do not
claim those inputs occupy available JXL reference slots. `improves_static_bbox`
only indicates a strictly smaller bbox than both static references, and is false
when cropping is unsupported. Large windows add exact pixel comparison work.

`frames[i]["geometry"]` includes `best_area_candidate_index`, `candidate_count`,
`trial_pixels`, `trial_layers`, and `unmasked_rectangles`. These are prospective
unpruned work counts, not performed encodes. The latter reports total/unique/
repeated rectangle work; `unique_pixels` sums distinct rectangle areas, not their
geometric union. Repeated payload work is an opportunity, not guaranteed cache
hits (patches and pruning can bypass reuse). The top-level `geometry` aggregates
each frame's best-area candidate, total layers and workload. Existing
`reference_modes` remains single-bbox-only for backward compatibility.

```python
analysis = rs_jxl.analyze_multiframe(
    images, threads=4,
    analysis_options={"extra_tile_sizes": [8, 16, 64], "history_window": 8},
)
frame = analysis["frames"][1]
best_geometry = frame["candidates"][frame["geometry"]["best_area_candidate_index"]]
```

```python
analysis = rs_jxl.analyze_multiframe(
    [rgba, changed, changed], reference="auto", threads=8,
    analysis_options={"tile_size": 16, "max_rectangles": 32,
                      "allow_mask": True, "rectangle_bytes_per_pixel": 1.0},
)
change = analysis["frames"][1]["previous_stats"]
identical = analysis["frames"][2]["previous_stats"]["changed_pixels"] == 0
geometric_reference = analysis["best_geometric_reference"]

with rs_jxl.Encoder(threads=8) as encoder:
    analysis = encoder.analyze_multiframe([rgba, changed])  # inherits eight workers
    data, measured = encoder.encode_multiframe(
        [rgba, changed], strategy="auto", effort=3, return_info=True,
    )
```

`analyze_multiframe` compares full input states in Rust without encoding, cropping
pixel buffers or running NumPy difference analysis. It accepts the same paths,
encoded bytes, streams, arrays and tensors as multiframe encoding, with `layout`,
`extra_channels`, `allow_device_copy`, `strip_metadata`, `bits_per_sample`,
`color_encoding`, `icc_profile` and `borrow_inputs`. Metadata overrides and
separate extra planes require array/tensor inputs, as for encode. Inputs must
match in canvas size, channel layout, sample depth and metadata; analysis never
pads, resizes, narrows samples or silently converts colors to make them match.
All channels participate, including RGB under zero alpha and exact float sample
words (signed zero and NaN payloads). Extra-channel changes count their covered
canvas pixels, taking the union with color changes rather than counting twice.

The default `reference="auto"` maps to native `smallest_crop`, like encoding.
`previous`, `first`, `full` and the `none` alias for `full` are also accepted.
`threads=1` is the one-shot default; zero resolves CPU count capped at 256.
`Encoder.analyze_multiframe` inherits the context's thread count and rejects
explicit mismatches and calls after close. Calls retain no reference-image state.

`analysis_options=None` or `{}` leaves all remaining controls at Rust defaults.
The adapter sends this as a top-level native `analysis_options` object, separate
from `options.threads`/`options.strip_metadata`, and returns native JSON unchanged.
Like `auto_options`, the mapping rejects unknown/duplicate keys and a nested
`reference`; use only the explicit Python `reference` argument. Boolean and
integer validation follows `auto_options`, including rejection of `None` values.
Encoding/search controls such as effort, prune and trial budgets are not analysis
options. These controls affect candidate proposals, not exact pair statistics.

| Analysis Option | Accepted Value | Native Default |
| --- | --- | --- |
| `allow_mask` | bool | true |
| `allow_multi_rect` | bool | true |
| `adaptive_rectangles` | bool | true |
| `tile_size` | positive u32 integer | 32 |
| `extra_tile_sizes` | list/tuple of positive u32 integers | [] |
| `history_window` | nonnegative native usize integer | 0 |
| `max_rectangles` | positive native usize integer | 32 |
| `rectangle_bytes_per_pixel` | finite nonnegative real number, not bool | 1.0 |

`rectangle_bytes_per_pixel` is only a cost estimate for adaptive rectangle
planning. It is not measured compression, an output-size prediction or a bit
depth. Zero is accepted. Analysis does not obtain this value by encoding; actual
`strategy="auto"` encoding measures an initial full-frame payload and selects
among real candidate encodes by bytes. Minimum geometric area does not imply
minimum compressed bytes. Do not treat `best_geometric_reference` or a suitability
label as an encoder winner or speed prediction.

The report contains `width`, `height`, `channels` (including extra channels),
`frame_count`, native `reference`, `crop_supported`, `rectangle_bytes_per_pixel`,
`analysis_seconds`, `frames`, `reference_modes` and `best_geometric_reference`.
`analysis_seconds` is native analysis/planning time, excluding input decoding,
context creation and Python/FFI transport. Ignore it when comparing equivalent
reports. Rectangles are `{x, y, width, height}` in canvas coordinates; JSON null
becomes Python `None`.

Each `frames` entry contains:

- `index` and the existing `previous`/`first` bbox-or-`None` fields. Frame zero
  retains full-canvas bounds; later identical comparisons have `None` bounds.
- `previous_stats`/`first_stats`, both `None` for frame zero. Otherwise each has
  `changed_pixels`, `changed_ratio`, `bbox_pixels`, `bbox_ratio`, `bbox_fill_ratio`,
  `bounds` and `mask_eligible`. Ratios use canvas area except `bbox_fill_ratio`,
  which is changed pixels divided by bbox area (zero for identical frames).
  Bounds may expand to align subsampled extra planes. Mask eligibility describes
  the exact opaque integer GA/RGBA requirements, even when `allow_mask=False`.
- `plan`: `reference_index`, `source`, `region`, `changed`, `layer_pixels` and
  `layer_area_ratio` for the selected single-crop reference mode. `reference_index`
  indexes input frames, or is `None` for a full replacement; `source` is the JXL
  reference slot (0 full, 1 first, 2 previous). An identical crop has `changed=None`
  but a 1x1 placeholder region and one submitted pixel, not a zero-area layer.
- `candidates`: geometry proposals with `strategy`, `tile_size`, `reference_index`, `source`,
  `rectangles`, `layer_pixels`, `layer_area_ratio` and `layer_count`. These are not
  encode trials: there are no encoded byte sizes or winner indices. Layer pixels
  sum submitted rectangle areas, including the entire bbox for a mask proposal.
  Strategy names are `full`, `crop_previous`, `crop_first`, `mask_previous`,
  `mask_first`, `tiles_previous`, `tiles_first`, `merged_previous`, `merged_first`.

`reference_modes` summarizes `full`, `previous`, `first` and `smallest_crop`.
Each entry has `reference`, `layer_pixels`, `layer_area_ratio`,
`average_changed_ratio`, `crop_frames`, `identical_frames` and `suitability`.
Total layer area includes the initial full frame; its ratio divides by canvas
area times frame count. Average change and frame counts exclude the initial
frame. `best_geometric_reference` minimizes total single-bbox submitted area,
not multi-rectangle candidates or bytes.

Suitability is `single_frame` for one input. Otherwise it is `strong` for a mean
submitted bbox-area ratio below 0.3 over subsequent frames, `moderate` below 0.7,
and `limited` otherwise. This uses subsequent frames, not the report's total
`layer_area_ratio`. Subsampled extra channels disable crop plans and produce
`full_frames_required` for multi-input sequences. Full replacements count as
change ratio 1 and zero identical frames in summaries, including explicit `full`
mode and the subsampled fallback. Per-pair `previous_stats`/`first_stats` still
report actual differences regardless of selected mode or fallback.

See [Native Multiframe Analysis](MULTIFRAME_ANALYSIS.md) for the core contract.

## Explicit Layers

`encode_layers` encodes supplied raw layer images without automatic differences,
crops, resizing or padding. Unlike `encode_multiframe` and `analyze_multiframe`,
layer widths and heights may differ. Channel layout, sample depth and color/extra
metadata must still match; native code validates compatibility. The existing
automatic multiframe and analysis APIs continue to require same-size input states.

```python
background = np.full((6, 8, 4), 65535, dtype=np.uint16)
overlay = np.zeros((2, 3, 4), dtype=np.uint16)
overlay[..., 0] = 12345  # exact even under zero alpha

data, report = rs_jxl.encode_layers([background, overlay], effort=3, return_info=True)
raw, layer_info = rs_jxl.decode_layer(data, layer=1, adjust_orientation=False)
assert np.array_equal(raw, overlay)
displayed = rs_jxl.decode(data)  # 6x8 canvas, only overlay over an empty background
assert displayed.shape == (6, 8, 4)
assert not displayed[2:, :].any()

with rs_jxl.Encoder(threads=8) as encoder:
    composed = encoder.encode_layers(
        [background, overlay], canvas=(8, 6), effort=3,
        layer_options=[{"name": "background"},
                       {"origin": (2, 1), "name": "overlay", "blend": {"source": 0}}],
    )
```

The full signature is:

```python
encode_layers(images, output=None, *, canvas=None, layer_options=None,
              tps=None, loops=0, layout="auto", effort=7, threads=1,
              lossless=True, distance=0, extra_channels=None, bits_per_sample=0,
              color_encoding=None, icc_profile=None, allow_device_copy=False,
              strip_metadata=False, return_info=False, encoder_options=None,
              borrow_inputs=False, _context=None, **tool_options)
```

`images` is a nonempty iterable using the shared multiframe transport: paths,
encoded bytes, readable streams, arrays or tensors. Extra-channel data accepts a
shared plane or one plane per input, each matching that layer's dimensions and
subsampling. Array metadata overrides are unavailable for encoded inputs.
`output`, layout, borrowing, GPU transfer and lossless/tool controls have the
existing encode contracts. `source_search=True` is rejected by native code;
there is no layer search scheduler. `Encoder.encode_layers` inherits context
threads, rejects mismatches and closed handles, and retains no reference state
between calls. `_context` is internal, not an override for a bound encoder.

`canvas=(width, height)` is optional. Native code infers the maximum right/bottom
extent of nonnegative origins when omitted. Any negative origin requires an
explicit canvas. Raw samples outside an explicit canvas remain encoded and are
returned by `decode_layer`; normal `decode` composites and clips to the canvas.
`decode_layer` defaults to `return_info=True`, returning `(pixels, metadata)`;
use `return_info=False` for pixels alone. Raw decoding disables coalescing, not
orientation adjustment: pass `adjust_orientation=False` to preserve original
stored coordinates and sample order when image orientation is not 1.
Raw layer selection and coalesced display-frame selection use different indices:
zero-duration layers are not separate display frames. These are standard JPEG XL
frame headers and pixels, not a new format, private archive or Python canvas.
Python performs no pixel padding, difference analysis or blend computation.

`layer_options=None` retains native defaults. Otherwise supply one mapping per
image; empty mappings retain defaults. Only these frame fields are accepted:

| Field | Value | Native Default |
| --- | --- | --- |
| `origin` | `(x, y)` signed integer coordinates | `(0, 0)` |
| `duration` | nonnegative u32 tick count | `0` |
| `timecode` | nonnegative u32 timecode | `0` |
| `name` | string, native UTF-8 header-length validation | `""` |
| `blend` | blend mapping described below | replace, source 1 |
| `extra_blend` | list of blend mappings, one per extra channel including alpha | `[]`, inherits color blend |
| `save_as_reference` | reference slot `0..3` | `0` |

Blend mappings accept `mode` (`replace`, `add`, `blend`, `alpha_weighted_add` or
`multiply`), `source` (slot `0..3`, default `1`), `alpha_channel` (extra-channel
index, default `0`) and `clamp` (bool, default `False`). Native code validates
which fields are meaningful for the selected blend mode, alpha-channel existence,
extra-blend counts and reference constraints. The last layer cannot save a
nonzero reference slot. `reference_only` and `save_before_color_transform` are
not exposed by this Python API.

Default source slot 1 remains blank because preceding zero-duration layers save
to slot 0 by default. Thus default layers independently replace their supplied
region over an empty canvas, not over the preceding image. Explicit `source=0`
can reuse the preceding zero-duration composite. Timed frames do not implicitly
save their composite to slot 0: save to a nonzero slot and select that slot when
later frames need it. Explicitly saving slot 1 also changes later default-source
backgrounds. Blend source numbers are reference slots, not input indices.

```python
animation = rs_jxl.encode_layers(
    [background, overlay], canvas=(8, 6), tps=(24, 1), loops=2, effort=3,
    layer_options=[{"duration": 12, "save_as_reference": 2, "timecode": 1},
                   {"origin": (2, 1), "duration": 12, "timecode": 2,
                    "blend": {"source": 2}}],
)
```

All-zero durations are genuine layers and do not add an animation header by
default. Any positive duration makes native code infer `1000/1` ticks per second
unless a timebase is supplied. Explicit `tps`, nonzero `loops`, or any nonzero
timecode sends animation metadata even for all-zero durations. A nonzero timecode
also makes Python enable `animation.timecodes`. For direct ABI callers, omitted
or null animation metadata lets native code infer timing and timecode support.
When animation metadata is explicitly supplied, nonzero frame timecodes require
`animation.timecodes=true`; an explicit `timecodes=false` is rejected.
`loops=0` means unlimited repetition when animation is present.

Unknown and duplicate frame/blend fields, nonmapping entries and mismatched
frame counts fail. Numeric fields accept integer-index values including NumPy
integers, but not booleans, floats or numeric strings; `clamp` requires a Python
bool. Pairs and frame/blend lists accept lists or tuples. Python normalizes JSON
integer types/ranges; native code owns wire limits, canvas inference, sample/color
compatibility, reference semantics and final frame validation. It does not
silently convert inputs to satisfy those constraints.

`return_info=True` returns `(encoded_output, report)`, preserving the normal
bytes/Path/None output behavior. The native JSON report is returned unchanged:
`operation="encode_layers"`, `bytes`, `jpeg_input`, `canvas={width,height}`,
`codec_seconds`, and `layers`. Each layer reports `index`, `width`, `height`,
`origin=[x,y]`, `duration`, `timecode`, `name`, `blend={mode,source,alpha_channel,clamp}`,
`extra_blend`, and `save_as_reference`. Timing is native codec time, not a Python
measurement. `jpeg_input` indicates JPEG pixel input, not byte transcoding:
`encode_layers` preserves the input decoder's samples, never original JPEG bytes.
Use `transcode_jpeg` and `reconstruct_jpeg` for byte-reversible JPEG storage.

## Decoding and Safety

decode defaults to frame=0, coalesced native samples, orientation adjustment,
dither="jxl-rs". frame=None retains all frames; targeted selection uses native
target-frame decoding. dither="libjxl" affects floating-to-U8 only, not exact
integers/JPEG bytes. format=u8/u16/f32 explicitly requests sample conversion.
info reads JXL headers/metadata. decode_to_png exports representable native
8/16-bit pixels; unsupported float/wide-integer/extra-plane output is rejected.
Selected-frame decoding does not validate later data. Use frame=None when
whole-input decoding is required; a selected frame's `complete` flag is not
whole-file validation.

### Preallocated Output

```python
destination = np.empty(rgba.shape, dtype=rgba.dtype)
returned, metadata = rs_jxl.decode_into(data, destination, return_info=True)
assert returned is destination
```

`decode_into(src, destination, ...)` decodes one nonnegative frame index (not
frame=None). The destination must be a writable, C-contiguous NumPy ndarray
with the exact returned HWC shape, including C=1 for gray, and exact
native-endian dtype. Torch destinations, reshaping, narrowing and implicit
destination conversions are not supported. Explicit `format` conversion, when
requested, occurs in the decoder and the destination must match that result.
This entry point does not fill separate extra-channel destinations.

Decoding and descriptor/buffer/destination checks complete before any copy;
decode or validation failures leave the destination untouched. Do not mutate
the destination or its aliases concurrently with the call. The copy reads the
live Rust result directly, avoiding intermediate Python bytes and a newly
allocated returned array. It still uses Rust output storage and codec scratch,
and is not direct decoding into caller memory or a zero-copy output view.
Without `return_info`, the destination itself is returned; with it the result
is `(destination, metadata)`. `Decoder.decode_into` has the same contract.

### Limits and Errors

Decode/info/reconstruction limits={...}: max_input_bytes/max_output_bytes/
max_metadata_bytes/max_pixels/max_samples/max_frames. decode also accepts
max_pixels/max_output_bytes directly. Inclusive budgets are not RSS bounds;
omission uses core defaults. PNG/JPEG pixel read_image does not expose budgets
yet. JPEG transcode limits are in the native protocol; unsupported limits fail.

Input pointers live only through the call. Rust owns result handles; Python
copies into owned output or the validated destination while the result is live,
without returning a borrowed Rust view. For nonempty data results, the internal
`_ffi._result_buffer` attaches an `ffi.gc` owner to the data pointer and
`ffi.buffer` retains that owner. Temporary derived views retained by an exception
traceback can therefore safely keep the native result alive beyond helper scope
exit. Cleanup releases the helper view; the owner's finalizer frees the native
result once no retaining references remain. Native error responses and empty-data
results still free directly in `finally`. This lifetime protection does not
change the public owned-output contract or expose a zero-copy output API.

CFFI checks ABI version/size and releases the GIL during native calls. One-shot
calls create their own resources; explicit contexts reuse workers, with
independently owned results. Invalid C pointers cannot be made safe by an ABI
check; see the header's caller contract.

Native failures raise `CodecError` with the original message and a `.kind`
category: `invalid_input`, `resource_limit`, `encoding`, `decoding`, `io`,
`internal` or `unsupported`. The ABI preserves its `error` string and adds
`error_kind` and optional `error_stage`. Python currently exposes `.kind`, not
a separate stage attribute; older libraries without a category yield `unknown`.
Classification follows known typed Rust errors; unclassified/string-only errors
can still be `invalid_input`. Python argument validation and path/stream failures
can raise ordinary TypeError, ValueError, IndexError or OSError before/after the
native call. Do not classify failures by matching message substrings.

[Protocol](../crates/rs-jxl-ffi/protocol.md),
[C header](../crates/rs-jxl-ffi/include/rs_jxl.h).
Not implemented: Python incremental streaming, public zero-copy output views,
direct native decode into user buffers, reference-only/pre-color-transform layer
controls or full C++ parameter parity. Reusable contexts and `decode_into` do not close those
gaps. Native Rust decoder metadata types still include maintained-upstream
reexports; this is not a complete backend-independent public type layer.

## Tests

Install pytest separately for development. The full command below includes
the pytest-parameterized packaging tests; unittest alone only runs the adapter
and native binding tests.

```powershell
$env:PYTHONPATH = "python"
python -B -m pytest python/tests -q
cargo test --locked --offline --workspace
```

RS_JXL_DJXL optionally supplies official decoder paths for independent tests.
Set RS_JXL_TEST_CUDA=1 to require the GPU transfer checks on a CUDA host.
Acceptance compares original samples including hidden RGB, or original JPEG
bytes, not compressed JXL identity. Cross-platform runtime claims need actual
tests on those systems, not just a matching wheel platform tag.

### Current Verification

After the 2026-10-27 module cleanup, the rebuilt Release DLL again passed all
260 Python/harness tests and the freshly built, isolated wheel passed all 174
API tests, with CUDA and official decoder checks enabled. No Python API or
protocol changes were made. See the
[maintenance checkpoint](ARCHITECTURE.md#multiframe-maintenance-checkpoint)
for the complete Rust regression and the scope of this verification.

The final multiframe stability check on 2026-10-27 passed 260 tests across
`python/tests` and the multiframe benchmark harness, with official djxl 0.12/0.13
and CUDA checks enabled. The built Windows wheel was installed into an isolated
directory; all 174 API tests passed with `RS_JXL_NATIVE_LIB` unset, after asserting
that both the Python package and native DLL came from that installation.
These are overlapping test sets, not 434 independent cases. Two existing
packaging/toolchain warnings remained non-fatal. The source distribution was
also checked for the new Rust modules, tests, documentation and C header.
See the [stability checkpoint](MULTIFRAME_STABILITY_20261027.md) for native
regressions, the corrected reference-slot test and validation limitations.

The earlier native automatic-option/report update on 2026-10-27 passed 173 Python
tests against the rebuilt release DLL, with official djxl interoperability and
CUDA transfer checks enabled. Two existing packaging warnings remained non-fatal.
Difference analysis, candidate generation, pruning and timing are native Rust;
Python validates and forwards controls and returns the native report unchanged.
See [MULTIFRAME_AUTO.md](MULTIFRAME_AUTO.md) for native acceptance and
[MULTIFRAME_NATIVE_BENCHMARK.md](MULTIFRAME_NATIVE_BENCHMARK.md) for size/speed
tradeoffs. No cross-platform runtime claim is implied.

The following architecture checkpoint predates the automatic-option additions:

Validated on Windows x86-64 with Python 3.10 on 2026-10-27:

- Full Rust workspace regression: 2,377 passed, zero failed, 107 ignored before
  the final diagnostic-retention review fix. Ignored tests are not passes.
- After that fix, the complete `codec_context` target passed all 8 tests,
  including two new artifact/receipt-preservation tests. This focused rerun is
  separate from the earlier full-workspace count.
- Final strict workspace Clippy (`--all-targets -- -D warnings`), formatting
  and whitespace checks passed. Release CLI and FFI builds completed.
- Full `pytest python/tests` against the final release DLL: 165 passed with
  official `djxl` interoperability and CUDA transfers enabled. Two preexisting
  tooling warnings (`pkg_resources` and `Py_DEBUG`) remained non-fatal.
- The final wheel was installed into `out/python-architecture-final-installed`.
  From the `out` working directory, with `RS_JXL_NATIVE_LIB` unset and the imported
  package path asserted to be installed, all 92 binding, context, adapter and
  buffer tests passed using the bundled DLL. This includes 45 native binding
  and context tests; packaging tests are included in the 165-test source run.

The pure derived-view test in [test_adapters.py](../python/tests/test_adapters.py)
uses a mocked result to check owner retention and exactly-once release after a
derived NumPy view is dropped. Native tests additionally cover closing a context
from an output-conversion callback, closed PNG/JPEG input callbacks, independent
concurrent handles, error recovery, multiframe state and byte-exact JPEG recovery.

These are functional and installed-wheel checks, not a performance comparison,
a full-corpus rerun, a release-profile Rust test run or cross-platform runtime
certification. The historical counts below describe a different checkpoint.

### Initial Binding Validation (Historical)

The earlier binding checkpoint recorded the following on Windows x86-64 with
Python 3.10 and the then-current release DLL, before this context/buffer refactor:

- Python suite: 116 passed, including packaging, native bindings, official
  djxl interoperability and CUDA transfers. Two environment/tooling warnings
  (pkg_resources deprecation and Py_DEBUG configuration) remained non-fatal.
- The real platform wheel was installed outside the source package path with
  RS_JXL_NATIVE_LIB unset; all 32 native integration tests passed using its
  bundled DLL. Wheel and sdist checks retained native sources, runtime assets
  and third-party license/provenance notices.
- Rust workspace: 2,352 passed, 107 ignored, zero failures. Ignored external
  corpus/oracle tests are not counted as passed. Strict workspace Clippy and
  rustfmt checks passed; existing Python tooling tests also passed (181).

This is a functional and packaging checkpoint, not a performance comparison
or Linux/macOS runtime certification. No full dataset benchmark was rerun.
