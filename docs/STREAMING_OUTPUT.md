# Incremental Container Output

`streaming::encode_frames_streaming_to(canvas, frames, options, sink)` emits an
ordered version-0 `jxlp` container. It reuses one bounded thread pool and holds
one encoded logical frame at a time, not the entire animation. Inline patch
reference frames stay with the frame that consumes them. All sink calls stay
on the calling thread; `Write + Send` is not required.

This API is **frame-buffered output**. `frames` still borrows caller-resident
images; the separate region/seekable API below handles within-frame emission.
Use `encode_frames_to` for codec-error-before-output behavior; streaming may
leave partial output on a later codec, allocation, output-limit or I/O error.
No API flushes or closes this caller-owned sink. The total output limit includes
container headers, chunk indexes and the optional frame index.

## Region Input and Seekable Output

`streaming::encode_regions_seekable_to(source, options, sink)` accepts a
`region_input::RegionSource` and a `Write + Seek` sink. It emits one full-canvas
frame in a normal version-0 container compatible with official 0.12 and 0.13.
This is the frozen source's mode-1 strategy: process DC areas independently,
write their sections, then seek backward to fill the ordinary TOC and container
length. No full input image, full encoded frame, out-of-order transport or
temporary-file spool is required by this path.

`ImageDescriptor` supplies dimensions, layout and metadata without pixel storage.
The provider returns a `RegionLease` with a typed `SampleSlice` and row stride
in sample elements. Color requests include interleaved alpha when present;
`RegionPlane::Extra(i)` addresses a metadata extra-channel index in that plane's
native subsampled coordinates. All original declared samples, including RGB
under zero alpha, are validated and retained. A lease's optional release
callback runs on normal return, failure or unwind.

`ImageRegionSource::new(&image)` adapts existing typed image buffers without
copying input pixels. A custom provider can instead fill/reuse a region buffer.
Provider acquisition/release and sink calls remain on the calling thread;
neither provider nor sink requires `Send`. Only owned area codec work enters
the explicitly bounded pool. The source must remain stable during the call.

Pass root-frame-resolved options with `palette_policy.groupwise=true` and source
streaming eligibility. At most eight AC groups, patches, responsive/custom
passes and incompatible squeeze settings are rejected before input/output
callbacks; this API does not silently buffer the full frame or reduce effort.
Global source parameter search is not run for callback input, matching source
eligibility. Lower-level `FrameAreaPlan` exposes checked topology and section
emission but does not serialize a complete JXL by itself.

Working storage is one DC area's writable planes/codec data, its encoded
sections, a small retained DCGlobal prefix and O(section-count) topology/sizes.
This is not a process-wide RSS limit. Input requests are tiled to at most 2048
native samples per axis; one callback is not necessarily one entire DC area.

The returned length counts the complete container, padding included. Backpatch
writes do not count twice against `max_output_bytes`. The sink starts at its
current position and finishes at the encoded end; the API never truncates,
flushes or closes it. Use an empty destination or retain only the returned
range. Once output begins, any provider/codec/I/O failure can leave an invalid
partial file. Success is only reported after all backpatches complete.

Official validation currently covers 8/16-bit RGBA and full/subsampled extra
channels (12 complete-file decodes across both official versions), plus 16
F16/F32 GrayAlpha/RGBA word checks across horizontal and vertical DC boundaries.
Unit checks cover non-Send callbacks, exact budgets and failure behavior. Other
sample types have loader tests but are not implicitly included in these counts.

FJXL has its separate prepared-state, region and output-processor lifecycle;
see [FJXL_PORT_STATUS.md](FJXL_PORT_STATUS.md). Plain `Write` alone cannot support
this one-pass bounded mode because the TOC precedes group data. A replay/two-pass
or spool implementation would be a separate strategy, not silently emulated seek.

## Push Frame Sequences

`streaming::FrameEncoder<W>` accepts a metadata-only canvas descriptor and a
`Write + Seek` sink. Each push borrows only the current `RegionSource`; it returns
after releasing its regions, finishing codec work and backpatching that frame.
It stores no future providers, full canvas samples or animation pixel history.

```rust,ignore
use rs_jxl::streaming::{FrameBuffering, FrameEncoder, FrameEncoderOptions};

let config = FrameEncoderOptions {
    threads: 4,
    max_output_bytes: 512 * 1024 * 1024,
    buffering: FrameBuffering::CurrentFrame {
        max_sample_bytes: 64 * 1024 * 1024,
    },
    ..Default::default()
};
let mut encoder = FrameEncoder::new(canvas_descriptor, config, sink)?;
let progress = encoder.write_frame(&mut source, &frame_options, &encode_options, is_last)?;
// Submit further providers synchronously; the last push must set is_last=true.
let (sink, encoded_length) = encoder.finish()?;
```

`FrameEncoderOptions::default()` uses `StrictAreas`, one thread and no index.
Strict mode retains exactly the existing source mode-1 eligibility gates; it
does not convert small crops into an approximate area algorithm. The explicit
`CurrentFrame` policy allows source-ineligible frames to use the ordinary
one-shot encoder. Eligible frames still use areas. `FrameProgress.mode` reports
`FrameAreas` or `CurrentFrameBuffered`; fallback is never hidden and never
changes source sampling, quality, tree policy or caller-selected tools.

Fallback stages the **current** frame's declared-format planes and typed sample
buffers, followed by its ordinary codec scratch and compressed logical frame.
`max_sample_bytes` is checked before acquisition and conservatively accounts for
eight bytes per color/extra sample, using each extra plane's native dimensions.
It bounds input staging, not codec scratch, provider storage or process RSS.
F16/custom floats use the existing exact declared-word inverse; there is no
floating-point normalization, integer rescaling or NaN special-case exemption.

Canvas/frame metadata must agree, except animation settings come from the canvas
and packing may vary between interleaved and separate alpha. Local frame sizes,
signed crop origins, frame names, durations (including zero), timecodes,
reference-only frames and per-extra blending use the existing frame validation
and header machinery. Extra requests are in current-frame native coordinates,
not canvas coordinates. There is no inferred difference crop or decoder-side
compositing in this API. Reference state commits only after successful output;
unsaved or pre-transform references cannot be used for visible color/extra
blending. Generated patch references are supported only by explicit fallback,
and their actual creation updates the reserved slot's state.

The new writer declares Level 10 before the first codestream fragment. This is
deliberate: later frame geometry is unknown and cannot retroactively upgrade a
Level 5 prefix. Every frame is checked against the fixed Level 10 limits and
its `EncodeOptions` limits before input/output. Per-frame thread counts must
match the stream configuration; one explicit pool serves the entire stream.
Source acquisition/release and sink operations stay on the caller thread,
without `Send` bounds. Serial/null execution never uses a global Rayon pool.

Output is ordered version-0 `jxlp`, one extended-size box per logical frame,
including generated patch frames when applicable. The first fragment includes
the image header. Backpatches never precede the most recently finalized prefix.
Progress and `finalized_bytes()` count bytes relative to the sink's initial
position, excluding any preexisting bytes. The stream-wide output cap counts
container overhead, padding, metadata and index bytes exactly once; per-frame
output caps additionally bound that frame. No API flushes, truncates or closes
the sink. An unfinished prefix is not a complete standalone image even when its
already finalized bytes are immutable.

Construction validates metadata/resources and creates the pool without writing
bytes; it only queries the sink position. Submission validation failures are
retryable and acquire no samples or output. Once encoding starts, provider,
codec or I/O errors and caught unwinds permanently poison the encoder. A later
failure can leave bytes beyond the finalized boundary, never overwrite it.
`finish()` requires a successfully finalized explicit last frame and never
patches a previous frame's last bit, inserts a terminal frame or retries output.

### Incremental Index and Metadata

Enable `FrameEncoderOptions.index = Some(FrameIndexOptions { max_entries,
max_payload_bytes })`, then call `write_indexed_frame` for the first and each
later selected keyframe. It takes the same arguments as `write_frame`. Other
pushes use `write_frame`. The writer retains only bounded selected-entry
metadata, not one record or input buffer per animation frame. Keyframes must
independently replace and display the complete canvas, including every extra.
Cross-interval references, generated patch tools, excess entries and payload
limits are rejected before acquiring the submitted frame.

Offsets count concatenated bare codestream bytes, excluding metadata, box
headers and jxlp counters. Durations include every interval's true ticks;
presented-frame counts omit reference-only and zero-duration intermediate
layers, but include a zero-duration terminal display. The timebase is seconds
per animation tick. The writer reserves the exact current index payload and
box size in the total output budget, writes one `jxli` after the final chunk,
and includes it in the final push's progress. No frozen lagged-duration bug is
copied into this higher-level timeline contract.

`write_metadata(kind, payload)` writes a complete borrowed payload synchronously,
before, between or after frames. It supports the existing `Exif`, `xml `,
`jumb`, `jxli` allowlist; payload semantics remain the caller's responsibility.
Raw `jxli` is rejected with automatic indexing and duplicate `jxli` is rejected.
Structural boxes and unsupported types such as `brob` return an explicit error,
not a no-op. This explicit metadata call can write/finalize the container prefix
before any frame eligibility is known; it never retains the payload.

### Stage-4 Verification

The parent reports central release sequence units **16/16 passed in 0.26 s**,
plus **4/4 sequence-oracle helper tests passed**.
Coverage includes shared old/new frame encoding, non-Send callbacks with
serial/parallel workers, full/crop/reference/terminal-zero frames, all 65,536
binary16 declared words, F32 specials, custom floats with mixed typed extras,
integer extra widths/subsampling, alpha packing changes, index intervals and
varint-boundary limits, metadata boxes, exact total budgets and error/unwind
poisoning. Late TOC/box-length seek failures preserve the previous finalized
prefix; final jxli write failures preserve the legitimately advanced last-frame
prefix and poison further operations.

All four independent suites in `tests/stream_sequence_reference.rs` passed:
**20 official comparisons**, covering 8 encodes with 1- and 4-thread execution
across both official djxl versions. The comparison breakdown is extras **4**,
indexed whole-stream and suffix decoding **8**, F16 special values **4**, and
mixed sequences **4**. No separate sequence-oracle cases remain pending in this
checkpoint.

Separately, the parent reports all 31 main-reference oracle suites passed across
both official djxl versions in 106.93 s. Those broader regression results and
the strict single-frame counts above are not additional sequence-oracle counts.
All results are from central runs; no local Cargo, test, benchmark or ARM run
was performed by this scoped worker.

## Indexed Animations

```rust,ignore
rs_jxl::streaming::encode_indexed_frames_streaming_to(
    &canvas, &frames, &[0, 10, 20], &options, &mut sink,
)?;
```

The selected indexes address the input `frames` slice, must start at zero and
strictly increase. Each selected frame must display an independent full-canvas
REPLACE image, including all extra channels. Within an interval, crop/blend
frames may only use references saved since that indexed frame. An interval
cannot secretly depend on earlier frames. These conditions are checked before
writing. A zero-duration composition layer cannot itself be selected, except
the displayed terminal frame; unselected composition/reference frames are
supported and excluded from the presented-frame count.
Generated patch atlases are rejected for indexed output: an offset to their
reference-only frame is not an offset to an independently displayed keyframe.
Ordinary, non-indexed streaming still supports those patch tools.

The writer emits exactly one `jxli` after the final codestream chunk. Offsets
count concatenated bare codestream bytes, excluding box headers and `jxlp`
sequence fields. Durations sum the real frame ticks, using the inverse of the
animation's ticks-per-second fraction. The last interval includes the last
frame. This deliberately uses correct intervals, not the frozen source's
lagged duration accumulator; see [FRAME_INDEX.md](FRAME_INDEX.md).

This API writes indexes; it does not provide a runtime seeking decoder.
Independent tests parse offsets, reconstruct a stream starting at a selected
frame, and compare its official-decoder output to the corresponding originals.

## Low-Level Writer

`container::ChunkedWriter<W>` writes one signature/file-type prefix and ordered
`jxlp` chunks with their final marker. `finish` requires a complete codestream.
Successful metadata writes may follow the final chunk. Duplicate `jxli` and
structural boxes submitted as arbitrary metadata are rejected. The low-level
metadata payload semantics remain the caller's responsibility.

Every complete box is size-checked before writing. An I/O failure poisons the
writer because `write_all` may have partially advanced the underlying sink;
retrying would corrupt the container. No out-of-order version-1 mode is enabled.

Source references: frozen `b866b3d` `encode_internal.h`, `encode.cc`, and frame
and container field definitions. Code contains no native codec dependency.
