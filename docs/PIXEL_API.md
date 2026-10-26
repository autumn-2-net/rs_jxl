# Rust Pixel API

This describes active interfaces during migration, not a stable released ABI.
No JPEG byte-transcode or runtime decoder is implied by these functions.

## Still Images

```rust
use rs_jxl::{Image, PixelData, PixelLayout, EncodeOptions, encode};
let image = Image::new(2, 1, PixelLayout::Rgba,
    PixelData::U8(vec![31, 97, 221, 0, 0, 255, 0, 255]))?;
let options = EncodeOptions { threads: 4, shared_tree: true, ..Default::default() };
let bytes = encode(&image, &options)?;
# Ok::<(), rs_jxl::Error>(())
```

`PixelData` stores interleaved native Rust samples, not packed bytes:

| Variant | Meaning |
| --- | --- |
| `U8`, `U16` | Unsigned integer samples; metadata may declare a smaller depth |
| `U32` | Unsigned samples with a declared depth of at most 31 bits |
| `F16` | Raw binary16 words in `Vec<u16>`, including signed zero/NaN payloads |
| `F32` | Binary32 samples in `Vec<f32>`; bit patterns are significant |

`BitDepth::floating(bits, exponent_bits)` also permits narrower custom floating
formats. F32 input must survive the source conversion and inverse exactly;
nonrepresentable values are rejected, not rounded under a lossless label.
Integer samples outside their declared depth are rejected as well.
The frozen C++ public integer input API accepts 1..24 bits. Rust's 25..31-bit
storage/encoding extension is not independently certified for full-bit recovery;
the official floating output oracle loses integer precision there. Do not treat
successful construction or encoding as archival acceptance for that extension.

`Image::with_metadata` takes explicit color/depth/orientation and planar additional
extra channels. Interleaved alpha remains the first extra-channel metadata entry.
Each extra plane may have its own sample depth/storage. `dim_shift` 0..3 gives
plane dimensions `ceil(width / 2^shift)` by `ceil(height / 2^shift)`; interleaved
alpha must remain full-resolution. Extra-channel metadata order is significant.
All samples, including invisible RGB, are preserved. No implicit color conversion,
alpha premultiplication or EXIF orientation rewrite is performed.

`encode_to` buffers a complete valid output before calling a Rust `Write` sink.
Codec errors do not partially write the sink; an I/O error can still be partial.
Default metadata conservatively declares 32-bit Modular scratch, requiring a
Level 10 container even for 8-bit input. A Level 5 declaration must satisfy the
actual buffer, dimensions, ICC and extra-channel constraints.

## Limits And Errors

Limits are local to the stated operation, not a process memory/time sandbox:

| Setting or field | Accepted range and scope |
| --- | --- |
| `threads` | 1..256, default 1; validated before creating the owned pool |
| Dimensions | Nonzero, <=2^30 per axis and <=2^40 pixels (Level 10); the lower explicit max_pixels resource limit still applies; construction separately checks the supplied buffer length |
| `max_pixels` | Default `u64::MAX` (no extra resource cap); explicit values apply per canvas/frame, including reference-only images, not summed over frames or extra planes |
| Extra channels | Metadata construction <=4096; Level 10 encoding <=256 beyond the color channels, including interleaved alpha |
| `max_output_bytes` | Default `usize::MAX` (no extra resource cap); explicit values bound total successful output including container, not peak scratch usage |
| ICC | Encoding <=2^28 bytes; Level 5 additionally <=2^22; structural checks do not replace CMS validation |
| `tree_samples` | Positive legacy sampling target, not a cap on source sampling; `sample_limit: Option<usize>` defaults to `None`. `Some(N)` requires positive N and is a hard selected-sample ceiling |
| `max_tree_leaves` | 1..2097152 (decoder tree range); explicit shared sample allowance is not multiplied per worker. Source projection additionally applies the decoder's geometry-dependent tree bound |
| Group shift / previous channels | 0..3 / <=60 |
| Palette limit / explicit squeeze requests | 1..70911 / no extra list-length cap; emitted transforms must fit the wire fields |

If a sample ceiling is explicitly requested, legacy sampling reserves the
remaining allowance before collection. Local groups need not share one
image-wide sample allowance. Defaults no longer impose the former resource
ceilings; see [the audit](RESOURCE_LIMIT_AUDIT.md).
Caller-owned pixels, intermediate transforms, frame lists and output buffers
can consume memory beyond these local ceilings. No recoverable-OOM, aggregate
sequence budget or cancellation guarantee is provided.

`Error::InvalidInput` reports malformed configuration/metadata; `Error::Limit`
reports resource or level limits; `Error::Encoding` reports coding failures;
`Error::Io` reports sink failures. Encoding completes before the first `Write`:
codec errors leave the sink untouched, but a sink failure can leave partial data.
`Interrupted` writes are retried; `WriteZero` is an I/O error.

`modular_16_bit_buffer_sufficient` is a caller declaration, not an analysis of
every selected transform. The encoder rejects floating or >12-bit channels
with this declaration; satisfying that prerequisite alone does not prove all
intermediates fit. The default 32-bit declaration is conservative.

## Tool Selection

`EncodeOptions` controls threads, pixel/output limits, group size, shared models,
patches and global palettes. `responsive` applies the source global squeeze
schedule; `passes` controls DC/AC spatial pass layout independently, with
`Passes::progressive()` selecting three passes. These are not lossy downsampling.
`threads=1` executes core work directly on the calling thread without a Rayon
pool. Native scheduler batches own outer parallelism: independent jobs use this
same serial execution policy internally, without changing their tool settings.
`modular::groupwise_decision` reports whether an explicit source groupwise
request uses independently coded frame areas or the full-frame route, with a
typed fallback reason. Patches, responsive/custom passes, local squeeze and at
most eight groups use the full-frame route. This is not incremental output or
an input callback API: the complete result is still collected in memory.
`modular` is a separate `ModularOptions` value:

- `predictor`, `weighted_preset`, `weighted_candidates`: predictor and preset search.
- `rct`, `rct_candidates`: fixed transform or source candidate cost selection.
- `palette_colors`, `palette_predictor`: explicit palette experiments.
- `palette_policy`: source pre/post-transform eligibility and cost rejection.
- `palette_policy.tools`: optional source palette ordering/geometry and exact
  flat-map lookup; default tool selection is unchanged. Native candidate and
  patch-atlas projections preserve an explicit selection.
- `squeeze`: explicit reversible horizontal/vertical passes.
- `sampling`: source MA sampling/quantile policy; absent uses explicit legacy sampling.
- `source_constant_channels`: explicit constant-tail bypass for source sampling,
  off by default like C++ `JXL_MA_CONSTANT_CHANNELS`. The legacy sampling path
  retains its existing bypass. Detection checks full channel populations.
- `source_predictor_dedup`, `source_property_dedup`: default-off source verified
  equivalence reuse during MA search. Prediction candidates or property splits
  are reused only after exact node-local proofs, not approximate sampling or a
  reduced search space. Property masks are inherited only where proven safe.
- `source_channel_tree`: default `ma_train::ChannelTreeMode::Disabled`; optional
  `Channels`, `ChannelsAndRows`, or `ChannelsAndRowsTwice` ports source
  `JXL_MA_CHANNEL_TREE=1/2/3`. It partitions already-quantized records by channel,
  then optionally by row, before normal adaptive learning. Quantizers and split
  thresholds remain shared. It is not the independent `tree_groups` algorithm.
  Pixel/frame/region entry points disable it unless the main image and every
  extra channel are integer and at most eight bits. The learner additionally
  requires at least 8192 stored records, channel ID as its first property, at
  most 32 channel bins, multiple occupied bins and no multiplier constraints.
  Row partitions need 16384 records and at least 4096 on each side. These are
  stored-record counts, not accumulated duplicate weights. Forced partitions
  can change compressed size in either direction; no improvement is promised.
- `source_channel_tree_final_only`: false by default. Source scheduler
  projections exclude tiers faster than Tortoise and disposable probes when
  enabled. Low-level Modular calls have no effort/probe context and consume
  resolved tool settings; the MA learner itself does not infer frame metadata.
  With a caller pool, channel partitions borrow disjoint sample columns and
  share the cooperative executor; serial/null calls remain serial.
- `source_ma_parallel`: `ma_train::ParallelOptions` controls default-off
  whole-subtree dispatch (`subtrees`), adaptive grain (`balanced`) and the
  8192-record entry gate (`small_subtrees`, otherwise 65536). `workers` bounds
  root concurrency using a source 32 MiB-per-root estimate; it is not an RSS
  ceiling. The same limits apply to channel partitions. `quiet` selects source
  completion-only notifications. Completed splits and source scoring remain
  unchanged; these flags do not replace adaptive learning with fixed trees.
- `parallel_tokens`: source opt-in channel-parallel token generation for shared
  frames. It activates only with an installed multiworker pool and a dominant
  stream of at least 65536 samples holding at least 75% of the total. Previous
  channels remain read-only; destination slices and predictor histories are
  private. Independent local streams and null-runner probes retain serial
  channel dispatch. The default is false, as in the frozen source.
- `source_gather`: optional whole-channel or bounded-wave source sample collection
  in the caller's thread pool. It requires `sampling`, does not change RNG or
  sample populations, and is off by default like the source's explicit switch.
  Its byte budget is a compact-record work estimate, not an RSS guarantee.
- `tree_samples`, `sample_limit`, `max_tree_leaves`, `split_threshold`: distinct
  sampling targets, hard resource ceilings and split costs, not effort aliases.
- `fixed_tree`: source predefined tree, independent of adaptive leaf budget.
- `custom_tree`: an explicit validated `Tree`, mutually exclusive with `fixed_tree`.
- `tree_groups`, `tree_group_splits`: optional independent MA learning domains,
  source work partition and property-1 merge. Actual stream IDs are preserved.
  The sample ceiling is shared across shards, not multiplied by the worker count.
- `previous_channels`: previous-channel properties; must agree with sampling policy.
- `entropy`, `entropy_options`: prefix/rANS and histogram/configuration policies.
- `source_tree_entropy`: use source entropy/LZ policy for the MA tree itself.
- `lz77`, `lz77_mode`, `source_lz77`: disable, explicit parser or source parser policy.

Source parser wire/cost settings must agree with entropy configuration. A source
scheduler projection reports any remaining policy gaps. Do not treat proxy
scores as exact source-candidate scores. A returned lossless stream is not proof
of compression-ratio or speed parity with the complete C++ scheduler.

`source_lz77.stream_arena` enables source cooperative indexed multi-stream
dispatch when its normal index/seed/stream-count gates permit it. Set
`parallelism=Some(SourceParallelism { ... })` for explicit worker/scratch limits;
`stream_arena_quiet` controls notifications. The defaults remain off. Index
builds and block parsing share the existing caller pool. Native scheduler
projections retain explicit `source_lz77` tool switches from the base options,
while replacing candidate-owned parser/bucket/seed parameters and wire coding
settings. A candidate that disables LZ77 still disables it.

`DriverOptions::arena_quiet` applies to cooperative probes and finalists.
See [PARALLEL_EXECUTION.md](PARALLEL_EXECUTION.md) for nested-work, scratch and
error-draining contracts. The source tools use explicit options, not automatic
process-environment overrides.

`scheduler::CandidateParams::buffering` uses `SourceBuffering`:
`Auto` (default), `FullFrame`, `StreamingAbove2048`, or `Streaming`.
For pixel-lossless source e9 and at least nine groups, Auto uses independently
coded frame areas unless a source gate (for example forced patches) blocks it.
E10 Auto retains the full-frame path. Explicit streaming still obeys the group
count and feature gates; it does not imply incremental API output. Local MA and
streaming entropy are selected together, not by merely disabling shared trees.
Patch reference atlases resolve their own geometry with the original buffering
policy; they do not inherit the main frame's resolved group count or route.

`scheduler::CandidateParams::responsive` is a separate boolean (default false).
True requests the source whole-frame default squeeze script through
`encode_native_candidate`: it disables streaming, defaults to Zero prediction
and the source responsive RCT policy, and keeps a single spatial pass. Explicit
predictor/RCT overrides remain available. It does not implicitly enable
`Passes::progressive()`. Automatic source search accepts only its default false
value, so an explicit responsive request is not silently overwritten by search.
The older `CandidateParams::squeeze` direction list is a local Rust-tool option,
not this source switch: nonempty lists are rejected by the source projection.
Direct `EncodeOptions.modular.squeeze` remains available independently.

`patches` selects the earlier exact Replace dictionary. `source_patches` selects
the source background-relative Add dictionary. Integer color uses signed
internal residuals. Floating color <=16 bits uses declared-format words with
representability and exact Add-replay checks; it never subtracts integer bit
patterns. Mixed-depth and subsampled extras retain their native samples and
use None blending. Integer Add also checks final integer recovery under the
official decoder's floating arithmetic, including overlapping patches.
They are mutually exclusive. Both reserve reference slot 3 and may decline a
fixture when no profitable supported component is found. They are intra-frame
tools, not automatic inter-frame prediction. Their overhead can exceed savings.
Malformed patch limits are rejected before applicability checks, including on
depths that cannot use the selected tool. A valid but ineligible heuristic
request may produce no patch dictionary; that is not acceptance of bad options.
`source_patch_reference` stores an unresolved source policy for Add atlases;
sampling and automatic group geometry are resolved only after the actual atlas
dimensions are known. This keeps reference-frame Gradient/LZ choices separate
from the main frame's resolved choices.
The frozen source reference recipe does not support subsampled atlas geometry;
its projection rejects that combination explicitly, even though the direct
native Add path supports it. Unsupported arithmetic or unrepresentable float
deltas are errors, not silently rounded results. See
[integer/float contracts](INTEGER_AND_FLOAT_PATHS.md).

## Frames and Layers

`encode_frames(canvas, frames, options)` uses `canvas` for size and metadata.
Canvas samples are encoded only when included as a frame. Each `Frame` borrows
an `Image` and owns `FrameOptions`. Frame options expose origin, duration, name,
timecode, color/extra blending, reference ID and reference-only status.

- Duration zero is preserved as a layer. It is never replaced with one tick.
- Nonzero durations require `canvas.metadata().animation` with a valid timebase.
- Reference IDs are 0..3. Referencing an unsaved frame is rejected.
- Pre-transform references are for patches, not ordinary frame blending backgrounds.
- Reference-only frames cannot be last or have a display duration.
- The final visible frame cannot save another reference.
- Color/depth/extra metadata must match the canvas; animation metadata is global.
- Patch-enabled sequences reserve reference slot 3 for the internal atlas.
- Explicit arithmetic/alpha blending has its format-defined semantics. Use
  REPLACE crops for exact restoration of arbitrary RGBA and invisible RGB.

Metadata matching includes orientation, color/ICC, tone mapping, extra names,
order, depth and association flags. `encode_frames` uses the canvas animation
globally, overriding each frame's local animation metadata for this comparison;
`multiframe::plan` requires identical full metadata, including animation.
Names are at most 1071 UTF-8 bytes, not characters. Duration, timecode and loop
count are `u32`; the animation timebase numerator is 1..2^30 and denominator
1..1024. Nonzero timecodes require `Animation::timecodes=true`.
Origin coordinates are -536880256..536880255; out-of-range origins are rejected
before pixel encoding. Alpha blend modes require an existing alpha-channel
index <=10; explicit extra blending must supply exactly one entry per extra,
while an empty list inherits the color blend rule. Slot 0 is retained by a
zero-duration layer, but is not saved by an ordinary timed frame.

Spot-color and tone-mapping metadata use the format's source-compatible
half-precision serialization. They do not promise arbitrary f32 metadata bit
identity. This does not quantize image samples or change pixel-lossless rules.

`encode_frames_to` has the same buffering/I/O contract as `encode_to`.

`streaming::encode_frames_streaming_to` instead emits frame-buffered ordered
`jxlp` output. `encode_indexed_frames_streaming_to` additionally validates and
writes a `jxli` index. They may leave partial output after a later error; their
limits include the container. See [streaming contracts](STREAMING_OUTPUT.md).

`streaming::FrameEncoder<W: Write + Seek>` accepts one `RegionSource` per push,
without retaining the animation's input frames. Its metadata-only canvas,
`FrameEncoderOptions`, `write_frame`/`write_indexed_frame` and `finish` separate
stream resources from each frame's codec settings. Eligible frames are emitted
by area; `StrictAreas` rejects other frames, while explicit `CurrentFrame`
permits bounded current-frame input staging. `FrameProgress.mode` identifies
the route, and `finalized_bytes()` reports an immutable output prefix. Encoding
or I/O failures poison the stream; a validation rejection before acquisition
can be corrected and retried. Index and metadata limits include box overhead.

`scheduler::run_native_sequence_search` evaluates each logical frame in its
real canvas/crop/blend/reference context and retains its winning bytes. It is
not concatenation of independently headed JXL files. See
[MULTIFRAME_SEARCH.md](MULTIFRAME_SEARCH.md).

## Fast Lossless and MA Tools

`fast_lossless` is the separate source FJXL encoder. Its effort 0..127 is not
the main effort scale; the source native adapter uses fixed FJXL effort 2.
Borrowed interleaved integer buffers support 1..16 bits and four channel layouts.
The standalone entry writes integer metadata; the typed native-candidate
adapter preserves outer metadata and supports eligible F16 word buffers too.
`NativeArtifact::implementation` reports `FastLossless` or `Modular` explicitly.
Ordinary default integer cumulative search bypasses FJXL, including effort 1,
as in the frozen fork. Animation/extras/progressive requests that FJXL cannot
represent stay on Modular. See [FJXL_PORT_STATUS.md](FJXL_PORT_STATUS.md).

`ModularOptions::source_ma_histograms` exposes the optional source shard,
padded/dense/property-bin/sparse/paired histogram tools. `source_ma_scoring`
exposes count-log, clipped/sliding entropy, FastEntropy and description costs.
Their enums live in `modular::ma_train`; defaults preserve source default
scoring. They change learning cost or model choice, not pixel fidelity.
`source_gather` controls channel/wave collection, sparse work and stripes;
stripe eligibility is height >=512 and area >=2^18, with 1..16 requested
stripes. A disabled/ineligible optimization does not lower the sample fraction.
`source_ma_search` additionally exposes statistical property racing, 32/64
quantile proposals and local refinement. All are off by default. Statistical
racing requires at least 262144 distinct stored records in a node, more than
four dynamic properties and bounded proposal storage; it is not enabled merely
because the input has that many pixels. Quantiles require at least 512 records
and enough occupied bins. FastEntropy or clipped entropy disables quantile
proposals, following source precedence.

## Difference Planning

`multiframe::plan(images, durations, reference)` compares original samples.
`Reference` is `Full`, `Previous`, `First`, or `SmallestCrop`. Every color/alpha/
extra sample is compared. Returned `PlannedFrame` uses `Cow<Image>`: unchanged
full images are borrowed, cropped images own only their crop samples.
`difference_bounds` reports changed source sample cells and aligned geometry.
For subsampled extras, `plan` conservatively retains complete frames, including
identical frames: upsampling runs before compositing, and cropping changes its
boundary conditions. Explicit `crop` extracts aligned source samples; it does
not promise identical upsampled edges. Comparisons use declared-format sample
words, not storage-container words: equivalent F16 and exact F32 storage of a
declared binary16 image are equal, but signed zero and distinct NaN payloads
are not. Integer comparison never converts pixels to floating point.

`SmallestCrop` compares bbox areas for previous versus first references. It does
not trial-encode candidates or promise the smallest JXL. First reference uses
slot 1; rolling references use slot 2. Identical frames encode a one-pixel REPLACE
layer because zero-size JXL layers are invalid; their duration stays unchanged.

```rust
use rs_jxl::{encode_frames, multiframe::{plan, Reference}};
# fn example(images: &[rs_jxl::Image], options: &rs_jxl::EncodeOptions)
# -> rs_jxl::Result<Vec<u8>> {
let durations = vec![0; images.len()];
let planned = plan(images, &durations, Reference::SmallestCrop)?;
let frames: Vec<_> = planned.iter().map(|f| f.as_frame()).collect();
encode_frames(&images[0], &frames, options)
# }
```

Zero-duration layers compose into a final display image; they are not independent
animation pages. Use nonzero animation durations when every full image should be
a displayed frame. A decoder retrieving a target composed frame must reconstruct
its referenced frames; crop bytes alone are not the original full image.

## Verification Limits

The ignored official-oracle tests require `RS_JXL_DJXL`. PNG output verifies
8/16-bit integer samples. Official NPY output is binary32: it can verify floating
words, but cannot distinguish all high-depth integer values. U31 projection
checks do not establish exact U31 integer recovery. Tests state which comparison
they perform; no tolerance is substituted for bit-exact losslessness.
