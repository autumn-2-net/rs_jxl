# Frame Index Payload Primitives

Source: frozen read-only libjxl
`b866b3defec13494bfbaf705db93581cda118e37`, `lib/jxl/encode_internal.h:43-119`
and `lib/jxl/encode.cc:653-745,1213,1332`.
Payload implementation: `crates/rs-jxl/src/container/frame_index.rs`.
Integrated output: `crates/rs-jxl/src/streaming.rs`, `streaming/index.rs`, and
`container.rs`. The payload sidecar itself does not emit boxes or select
container versions.

## Parent Integration

The parent has registered `pub mod frame_index;` in the top-level container
module and integrated the payload API below into indexed streaming output.

```rust
pub const BOX_TYPE: [u8; 4] = *b"jxli";
pub struct TickDuration { pub numerator: u32, pub denominator: u32 }
pub struct FrameIndexEntry {
    pub offset_delta: u64,
    pub duration: u64,
    pub frame_count: u64,
}
pub fn encode_payload(
    ticks: TickDuration,
    entries: &[FrameIndexEntry],
    max_payload_bytes: usize,
) -> crate::Result<Vec<u8>>;
```

The wire is NF unsigned base-128 varint, TNUM/TDEN as big-endian u32, then
OFFi/Ti/Fi unsigned base-128 varints for every indexed frame. The first frame
must be included and all indexed frames must be independently decodable
keyframes. The serializer cannot infer those semantic properties from offsets.

Ticks are **seconds per tick**, inverse to Animation's ticks-per-second pair.
`TickDuration::default()` is the source's `1/1000`. The raw wire permits zero
numerator, but zero denominator is rejected. The integrated writer reverses
Animation's numerator/denominator pair to obtain seconds per tick, using the
default when animation metadata is absent. The raw payload API performs no
implicit conversion.

OFFi is the first frame's absolute offset in the bare codestream, then deltas
between indexed frames. Count neither box headers nor the four-byte jxlp
sequence field, regardless of how the codestream is split. Ti is the interval
duration through the next indexed frame or stream end. Fi counts presented
frames in that interval, including the indexed frame, but not nondisplayed
composition frames. The terminal zero-duration displayed frame still counts.
Emit at most one jxli box. Box positioning and ordered format-version-0 policy
remain entirely parent-owned; these primitives add no out-of-order-v1 route.

All varint fields and cumulative offsets are capped at `2^63-1`. A nonempty
index, positive Fi, and positive subsequent OFFi are required. Output size is
checked before allocation; the bound excludes the enclosing box header.

## Completed Indexed Streaming Integration

`encode_indexed_frames_streaming_to` emits ordered format-version-0 jxlp
fragments followed by one jxli box. It calls `encode_payload` with correctly
aggregated intervals, never `SourceFrameIndex::encode_source_payload`.

- Selected input-frame indices must be nonempty, strictly increasing, in
  range, and start at zero.
- Every selected frame must be displayed, not reference-only, and must
  independently replace the complete canvas for color and all extra channels.
  Selected nondisplayed composition starts are rejected, including index zero.
- Reference availability resets at each selected boundary. Later blend/crop
  sources must have been saved within the current interval; references from
  an earlier interval are rejected before writing output.
- Both generated-patch options are rejected. The current API does not claim
  that an offset before an automatically inserted atlas identifies its selected
  displayed keyframe. General composite/atlas index starts remain unsupported.
- Ti sums actual frame durations. Fi counts only non-reference-only frames
  with nonzero duration or the last-frame flag, including a zero-duration last
  display. This makes each accepted interval's Fi positive.
- Offsets are captured before actual logical-frame bytes, starting after the
  codestream header, and exclude container headers and jxlp sequence fields.
- The writer enforces the total output budget, including the jxli box. Sequence
  and index eligibility are validated before output; later codec, resource or
  I/O failures can leave partial output, as with ordinary streaming output.

These are intentionally conservative eligibility rules. The format's broader
composite-keyframe possibilities are not advertised by this integrated API.

## Frozen Lifecycle and Source Defects

`SourceFrameIndex::new(ticks, max_frames)` plus
`record_frame(offset, duration, requested)` ports source AddFrame. Re-recording
the sole first offset replaces it, including duration and request flag. All
other offsets must strictly increase. Every recorded frame is retained, with
a caller limit and the source signed frame-count range checked.
`is_requested()` matches StoreFrameIndexBox.

`encode_source_payload(max_payload_bytes) -> Result<Option<Vec<u8>>>` returns
None when no frame requested an index. Otherwise it implicitly includes the
first frame, even when only a later frame was requested, as the source helper
does. **This explicitly named method preserves flawed frozen timing; do not
use it as a correct animation interval aggregator.**

The actual `EncodeFrameIndexBox` body at lines 704-742 initializes `T_prev`
and `T` to zero, writes T_prev, then sets T_prev=T and adds only the next
selected frame's duration to T. First and unselected durations never enter
T. For frames with durations `[11,22,33,44,55]` and selected indices `[0,2,4]`,
it writes Ti `[0,0,33]`, not the documented interval values `[33,77,55]`.
Even two selected frames produce zero Ti for both. It also uses differences
in recorded-frame indices for Fi, with no filtering of nondisplayed frames.
Signed duration accumulation beyond INT32_MAX is rejected in Rust instead of
depending on source overflow or signed-to-unsigned wrap behavior.

Additional source integration findings: the TNUM/TDEN defaults have no
assignment from Animation in this frozen encoder; the option setter at
`encode.cc:2070` sets the index flag true for both accepted option values 0
and 1. The API header promises first-frame-request/keyframe checks, but this
helper does not implement them. This sidecar neither reproduces that option
setter bug nor claims to validate a frame's dependencies.

The integrated writer constructs correctly aggregated raw entries and calls
`encode_payload`. The source lifecycle method remains an explicitly flawed
compatibility/oracle utility, separate from usable indexed output. Resolving
those source defects does not require changing the varint/field format.

## Verification and Remaining Work

Seven focused inline tests cover literal wire goldens and byte budgets,
base-128 boundaries through u64, bounded raw fields and absolute-offset
overflow, first-entry replacement and request gating, exact frozen timing,
record-count/signed-duration limits, and separately supplied correct intervals.
Parent verification reported on 2026-10-23: **eight official decoder checks
passed**, including actual independent decoding of indexed suffixes. This
verifies the integrated path, not merely index-field parsing or full-stream
decoding. No builds or tests were run by this subagent; central verification
and decoder oracles remain parent-owned.

Actual frame offsets, conservative keyframe/dependency validation, first-frame
selection, displayed-frame counting, timeline/timebase handling, and bounded
box emission are now integrated. Remaining unsupported scope includes selected
nondisplayed composite starts, generated patch atlases in indexed output,
region-input callbacks and within-frame incremental emission. No index parser,
seeking-decoder API or out-of-order container mode is added. This checkpoint
does not claim complete pixel-lossless port acceptance.
