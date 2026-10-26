# Incremental Pixel Input

The pixel decoder supports push/finish and blocking `std::io::Read` input.
One maintained `JxlDecoder` and one thread runner survive for the entire input.
`NeedMoreInput` suspends that instance; supplying a new chunk never restarts the
decoder, reparses an accumulated file, or waits for EOF before decoding pixels.
Python bindings are a separate, deferred task.

## Reader API

```rust
use std::fs::File;
use rs_jxl::decode::{DecodeOptions, decode_reader, decode_reader_with_chunk_size};

# fn main() -> rs_jxl::Result<()> {
let options = DecodeOptions::default();
let image = decode_reader(File::open("input.jxl")?, &options)?;
let image_in_small_chunks = decode_reader_with_chunk_size(
    File::open("input.jxl")?, &options, 4096,
)?;
# Ok(())
# }
```

`decode_reader(reader, options)` uses a reusable 64 KiB input buffer.
`decode_reader_with_chunk_size(reader, options, chunk_bytes)` uses a positive
maximum read size, capped by the input limit plus one EOF-probe byte. Both return
the existing `DecodedImage` and retain its decoded frames, not a complete
compressed-file copy. They read to EOF to validate trailing container data.

Reader APIs are blocking. Short reads are normal, `Interrupted` retries, and a
zero-byte read is EOF. `WouldBlock` and other I/O errors are returned as I/O
errors, never interpreted as partial-input EOF. Use the push API with an async
transport or a nonblocking reader. The API does not close or rewind the caller's
reader; pass `&mut reader` to retain access to it.

## Deliver Frames Before EOF

```rust
use rs_jxl::decode::{DecodeOptions, decode_reader_with};

# fn consume(reader: impl std::io::Read) -> rs_jxl::Result<()> {
let summary = decode_reader_with(reader, &DecodeOptions::default(), 4096, |frame| {
    // Consume or transfer the owned pixels now. A later input error is possible.
    let is_preview = frame.header.is_preview;
    let index = frame.index;
    let pixels = frame.pixels;
    Ok(())
})?;
assert!(summary.complete);
# Ok(())
# }
```

The callback runs after each chunk is processed and before the next reader call.
With small chunks, earlier animation frames can be delivered while later frame
bytes have not arrived. A large chunk can complete several frames before the
callback runs. Callbacks receive both preview and regular frames; the preview
flag is authoritative, and preview does not consume display-frame index 0.
A callback error stops reading and propagates unchanged.

`StreamSummary` carries:

- `info`, `embedded_profile`, `output_profile`, `output_intensity_target`;
- decompressed `boxes`, grouped using the existing decoder's box policy;
- `frame_count` excluding preview, and `has_preview`;
- `complete` for the entire input and requested output quality;
- `bytes_read` including parser read-ahead and requested open-ended metadata.

For a bare codestream with an ignored suffix, `bytes_read` can be less than the
number of bytes read from the reader. Read-ahead can also depend on chunk size.
Neither value is a frame seeking index.

## Push API

```rust
use rs_jxl::decode::{DecodeOptions, StreamDecoder};

# fn consume(chunks: impl IntoIterator<Item = Vec<u8>>) -> rs_jxl::Result<()> {
let mut decoder = StreamDecoder::new(&DecodeOptions::default())?;
for chunk in chunks {
    let status = decoder.push(&chunk);
    for frame in decoder.take_frames() {
        // Valid completed frames remain available even if this push failed later.
        let pixels = frame.pixels;
    }
    status?;
}
let status = decoder.finish();
for frame in decoder.take_frames() {
    // EOF can flush an incomplete frame only with allow_partial=true.
    let pixels = frame.pixels;
}
let summary = status?;
# Ok(())
# }
```

- `new(&DecodeOptions)` validates the same options as one-shot decoding.
- `push(&[u8]) -> Result<()>` processes borrowed bytes immediately. Empty chunks
  do not signal EOF. Only unread suffixes are retained by this adapter.
- `basic_info()` returns parsed source information as soon as it is available.
- `take_frames() -> Vec<DecodedFrame>` transfers pixels without copying and is
  legal after errors and after EOF.
- `finish() -> Result<StreamSummary>` signals EOF exactly once. Call it even if
  all expected frames have already arrived, then drain frames once more.
- A decode, allocation, limit or conversion error poisons the decoder. Further
  push/finish calls fail. There is no implicit reset, concatenated-file support,
  error recovery, or successful finish after a failed push.

Already delivered frames cannot be retracted if a later frame, metadata box or
read fails. Consumers needing atomic whole-file publication must stage their own
output until `finish` succeeds. The collecting reader APIs do not return a
successful `DecodedImage` after such an error.

## EOF and Container Metadata

The requested boxes are the same as for `decode`: Exif, XMP, JUMBF, JPEG
reconstruction data, and HDR gain maps. Metadata can precede, separate, or follow
codestream chunks. `ftyp` minor version 0 requires ordered `jxlp` chunks;
out-of-order boxes with this declaration are rejected, even in partial mode.
The maintained backend also supports the explicit version 1 out-of-order
extension. This is container chunk ordering, not a frame TOC permutation.
The ordinary writer continues to emit ordered version 0 files. Chunks are
processed by the persistent container parser, not flattened into a temporary
codestream file. See the upstream implementation contracts in
[libjxl PR #4741](https://github.com/libjxl/libjxl/pull/4741) and
[jxl-rs PR #752](https://github.com/libjxl/jxl-rs/pull/752); those establish backend
support, not a claim that a particular final ISO edition has adopted version 1.

Seek-index availability is separate from sequential decoding validity. A
checkpoint cannot restore future chunks already held in the out-of-order
buffer. Once the backend encounters out-of-order input, it returns no new seek
checkpoints for the remainder of that decode, even after the buffer drains.
Previously published valid index entries remain a usable prefix; no later
entries with missing frame/reference history are published. If reordering
precedes the first frame header, the seek index is empty, but sequential frame
events and pixels are still returned. Physically ordered version 1 input keeps
its normal seek index. This avoids treating an unavailable checkpoint as an
invalid container and does not relax version 0 ordering checks.

A box whose encoded length is zero extends to EOF. The backend can emit
`Complete` after its header, exposing an already-read prefix through
`trailing_box().raw_data()`. That is **not** public stream completion. The adapter
retains the remaining requested box data across later pushes, joins it with the
backend prefix exactly once at EOF, and uses the ordinary `boxes::extract`
decompression and aggregate metadata checks. An unrequested open-ended box is
ignored under the same policy as one-shot decoding.

Incomplete bounded box headers/payloads and malformed Brotli data remain errors,
including when `allow_partial=true`. `allow_partial` permits only a backend
`NeedMoreInput` at EOF: completed earlier frames are retained, and current pixels
are flushed if the backend can render them. No fake successful empty image is
returned. Partial output has `complete=false` and no extracted container boxes.
In the exact-wide-integer capture path, only completed captured frames are
published; incomplete native integer frames are not synthesized from float data.

## Pixel Semantics and Budgets

Default coalesced frames, raw zero-duration layers, crop/header orientation,
extra planes, alpha association, preview, requested sample formats, progressive
quality, color conversion and partial-output flags use the ordinary decoding
helpers. Conversion occurs once per emitted frame, using the backend output
profile as source; previously emitted pixels are not converted again. Native
wide integers use the shared pre-render integer capture and `decode::native`
conversion, not float round-tripping.

Budgets are not reset when a frame is taken or a new chunk arrives:

- `max_input_bytes` bounds total accepted bytes, including ignored suffixes and
  metadata. At the exact reader limit one extra byte may be read solely to
  distinguish EOF from an oversized file; it is not fed to the decoder.
- `max_output_bytes` bounds both cumulative reserved pre-conversion frame bytes
  and cumulative emitted pixel bytes, including preview and retained extras.
  Per-frame color conversion scratch is also checked by the existing helper.
- `max_frames` counts regular frames or raw layers over the entire input.
- `max_metadata_bytes` bounds total decompressed requested metadata at EOF.
  An uncompressed open-ended requested box also has an early individual bound.
- Pixel, sample and backend complexity limits remain active for the lifetime of
  the backend. `max_samples` retains its backend/per-frame meaning; it is not a
  second cumulative animation-output limit.

Color staging is sequential rather than staging all frames together at EOF.
Its temporary-memory budget can therefore accept an input that batch color
processing rejects, while cumulative input/output/frame limits remain enforced.

This is **not a constant-memory whole-file decoder**. Backend section buffers,
reference frames, requested metadata, and out-of-order `jxlp` data can occupy a
substantial fraction of the input. Open-ended requested metadata is retained
until EOF; compressed metadata inflation is limited at extraction time, not by
an incremental Brotli sink. Keeping unread push output also keeps those pixels
alive. The collecting reader functions intentionally retain all returned pixel
frames. None of the public budgets promises a strict process-RSS ceiling.

## Integration and Validation

`decode_streaming.rs` contains deterministic in-memory regressions for bytewise
and irregular chunking, all two-chunk split positions, callbacks before EOF,
truncation, malformed trailing boxes, compressed and open-ended metadata,
version 0 ordering rejection, explicit version 1 out-of-order chunks, extended
box sizes, cumulative budgets, I/O/callback errors, color conversion, raw layers,
previews and native wide integers. Animation fixtures use 60 ticks/s so one-tick
display frames remain within Level 5 rather than bypassing profile validation.
The version 1 regression checks single-frame and multiframe pixel equality,
absence of invalid seek targets, and usable targets for ordered version 1 input.
Run the integrated regression target with:

```text
cargo test --locked -p rs-jxl --test decode_streaming
```

The push/reader functions decode pixels only. They do not provide incremental
JPEG-byte reconstruction, metadata-only scanning, or selected-frame early-stop
input. Existing one-slice APIs remain available for those separate workflows.
Cross-platform runtime and malformed-input campaign results must be reported by
their actual runners; these source-level tests alone do not establish them.
