# Multiframe Speed, 2026-11-17

Baseline: `0ce4cd740d6aea61299003126e6215c9b1503df7`, pushed to `origin/main`
before this round. That commit contains the compression improvement measured
in [the rate-feedback report](MULTIFRAME_RATE_FEEDBACK_20261117.md). This round
keeps that search enabled and targets the extra cost of its rectangle layers.

## Search control

`AutoOptions.adaptive_rate_feedback` is a boolean alongside `allow_mask`,
`allow_multi_rect` and `adaptive_rectangles`, defaulting to **true**. False
restores the original first-frame-rate proposals. Python uses the existing
`auto_options` map; CLI `encode-frames` accepts `--no-adaptive-rate-feedback`.
Reports include the configured boolean. There is no new format or decoder mode.

```python
data = rs_jxl.encode_multiframe(
    frames, strategy="auto", threads=4,
    auto_options={"adaptive_rate_feedback": True},
)
```

`strategy="crop"` remains the Python default. The new field affects automatic
search only; reference choice and mask eligibility remain independent controls.
Disabling multi-rectangle or adaptive-rectangle proposals also disables feedback
proposals. All original candidates still compete using actual encoded bytes.

## Implementation

The encoder batches independent unmasked Replace rectangles only when there is
no incumbent byte cutoff and none of their payloads participates in the shared
cache. It uses the installed worker pool, at most four jobs at once. Crops are
materialized per batch, reference validation stays ordered, and complete layer
bytes are appended in their original order. Generated patch-reference layers
remain inside their logical-frame payload. Work is charged before dispatch;
the existing output limit applies to each payload and assembled candidate.
Bounded trials, masks, cached rectangles and serial execution retain their
original route. The bounded batch can retain more scratch/payload memory than
one-at-a-time encoding; it does not allocate crops for the whole candidate.

The decoder has two optimizations:

1. An Extend suffix containing only identity, native F32, single-channel Save
   stages can copy background rows directly into BufferSplitter output views.
   It avoids the intermediate row copy and per-row stage dispatch, and uses
   whole-width strips. Missing references still supply positive zero. Converting,
   interleaved, oriented or otherwise incompatible suffixes use the old stages.
2. A hidden coalesced Replace layer can take ownership of its own destination's
   reference canvas and update only the crop. This avoids allocation and copying
   of the entire canvas for each intermediate rectangle. It requires unique
   ownership, a complete post-transform canvas, and the same source/destination
   slot for every color/extra channel. The first implementation is restricted to
   ordinary Modular RGB integer frames up to 16 bits, with at most one full-size
   integer alpha channel, no patches/LF dependency/noise/splines/upsampling,
   no XYB/YCbCr and no capture path. All other cases keep the allocation/copy route.

The ownership path retains Extend for coordinates and clipping but skips its
outside-crop render. Hidden coalesced frames have no API Save stage; reference
publication still occurs at normal frame finalization. Raw layers, visible
outputs, F32, wide-integer capture and other unsupported combinations fall back.
No frame is reordered, and no new unsafe code or CPU requirement is introduced.

## Measurement method

The nine inputs and encoded streams are the frozen development/control set from
`out/multiframe-rate-20261117/`, including the original photograph ICC profiles.
The photographic sequence with a flat intro and edits is a **derived development
fixture**, not a real animation; the t3 pair is a real control. This is not a new
holdout corpus. Expected original arrays, inputs and DLLs are hashed.

Both builds use `cargo build --offline --release -p rs-jxl-cli -p rs-jxl-ffi`,
rustc 1.97.1, Windows x86_64 on Ryzen 9 7945HX. The baseline DLL is the preceding
report's frozen after build. V1 contains parallel encoding and direct background
copies; the final candidate additionally reuses eligible reference canvases.

Decode-only comparisons use identical frozen feedback-enabled streams. Every
call checks original dimensions, types and sample words outside the timer,
including hidden RGB. MPix/s counts logical full-canvas pixels across all returned
frames. Before/after calls alternate order; timings include the public reusable
Python Decoder and returned output, excluding file IO and correctness checks.
Encode/decode comparisons additionally check original frame durations and ICC.
No build or test runs concurrently with performance measurements.

## Results

Encoding uses one warmup and three measured alternating calls, effort 9 and
four threads. Decode-only measurements use two warmups and nine measured
alternating calls at each thread count. Values are medians. Every output stream
from the final enabled encoder is **byte-identical** to the frozen feedback
baseline, not merely equal in size. The total remains **2,393,900 bytes**, retaining
the preceding 4.628% reduction against the original 2,510,064-byte search result.

| Sequence | Bytes, unchanged | Encode MPix/s, 4T before -> after | Decode MPix/s, 1T before -> after | Decode MPix/s, 4T before -> after |
| --- | ---: | ---: | ---: | ---: |
| Synthetic: revisit first | 450,868 | 0.362 -> 0.375 | 11.398 -> 11.479 | 11.229 -> 11.338 |
| Synthetic: revisit middle | 600,822 | 0.323 -> 0.237 | 8.151 -> 8.075 | 8.339 -> 8.223 |
| Synthetic: repeated crop | 376,207 | 0.224 -> 0.237 | 16.009 -> 16.255 | 15.381 -> 14.889 |
| Synthetic: subtile changes | 164,206 | 0.237 -> 0.244 | 31.270 -> 32.493 | 27.421 -> 28.374 |
| Synthetic: flat to textured | 69,601 | 0.218 -> 0.247 | 15.739 -> 18.325 | 10.289 -> 13.147 |
| Synthetic: textured to flat | 50,594 | 0.557 -> 0.555 | 23.708 -> 23.293 | 19.225 -> 19.513 |
| Synthetic: hidden RGB transition | 69,562 | 0.181 -> 0.207 | 10.794 -> 13.219 | 7.907 -> 9.295 |
| Real control: t3 pair | 371,056 | 0.585 -> 0.802 | 15.509 -> 15.340 | 50.259 -> 50.103 |
| Derived: photo intro + edits | 240,984 | 0.758 -> 0.861 | 16.292 -> 28.010 | 16.891 -> 36.381 |

Across the fixed set's 3,804,888 logical pixels, total pixels divided by the sum
of per-case median times gives:

- Encoding: **0.496 -> 0.548 MPix/s**, about **10.5%** higher throughput.
- Decoding, one thread: **15.252 -> 18.606 MPix/s**, about **22.0%** higher.
- Decoding, four threads: **19.080 -> 26.571 MPix/s**, about **39.3%** higher.

The derived photographic case improves **2.154x** in four-thread decode. V1,
which only removed the intermediate background copy, measured 17.349 -> 23.880
MPix/s on that same stream in its separate paired run. Canvas ownership reuse
provides the larger remaining reduction in whole-canvas work. The new stream
still has 16 rectangle layers in its final logical frame; the original single
crop was faster in the preceding report. This recovers part of the speed cost,
without trading away its smaller size. Full layer batching remains future work.

The revisit-middle encoding regression in the three-repeat table was investigated
rather than omitted. Every multipart trial in this case has an incumbent cutoff
(later frames also hit the payload cache), so the new parallel route is excluded.
The measured encode times had uneven fast/slow runs: before 0.609/0.601/0.874 s,
after 0.590/0.831/0.833 s. A separate two-warmup, seven-repeat alternating check
gave **0.303 -> 0.310 MPix/s**; the regression did not reproduce. Both reports
remain in the artifacts. The table and aggregate above retain the original
three-repeat measurements, without substituting the more favorable repeat.

Small differences in unchanged decode controls are not established improvements.
These fixed development results are not a claim of universal speedup or a
representative corpus estimate.

## Verification

- 38 multiframe unit tests, 78 initial integration tests, 31 FFI tests and 19
  CLI multiframe tests passed. After adding canvas reuse, 44 affected integration
  tests passed; the final five reference-copy tests plus 12 decoder-context
  tests also passed. External-oracle tests remain explicitly ignored in these
  Cargo runs (seven in the metadata matrix, one in FFI).
- The same-slot reference regression checks RGB/RGBA8/16, F32/raw fallbacks,
  clipping, hidden RGB, threads 1/4 and seven-byte incremental input against an
  independent original-word compositor. Direct backend flush testing verifies
  that hidden layers leave the sentinel API buffer untouched and preserve the
  visible BasicInfo/FrameHeader/FrameComplete/Complete event sequence.
- Ten Python option/native tests passed against the final DLL. Defaults,
  explicit false, strict bool validation and native reports are checked. Rust
  1.89 checks, release CLI/FFI builds, changed-file Rustfmt, Python syntax and
  normal repository `git diff --check` passed.
- All 396 final decode-only calls, 72 enabled encode/decode pairs and 18 disabled
  feedback checks passed exact sample validation. Encoding checks also verify
  durations and original ICC bytes. Source, stream and DLL fingerprints remained
  unchanged throughout measurements.
- Every enabled stream matches `0ce4cd7` bytes exactly; every disabled stream
  matches the frozen pre-feedback `53f6715` bytes exactly. Those same 18 byte
  streams were independently decoded by official djxl in the preceding report.
  This round did not rerun that external oracle. Decoder correctness here is
  checked against the original sample arrays, not another native build's output.

Artifacts include `decode-final.json`, `paired-final.json`, `legacy-check.json`,
`revisit-check.json`, `summary.json`, build/source fingerprints and central test
logs under the local output directory.

## Further layer batching

True inter-layer decode batching is still **not implemented**. Existing seek
analysis records slot/LF dependencies, but the parser currently owns one active
Frame and finishes it before parsing the next layer. A batch implementation would
need bounded lookahead for headers/TOCs/payloads, independent crop decode jobs,
and ordered reference/event publication. Patch and LF dependencies require
additional handling. A visible FrameHeader must still precede its payload decode
so callers can allocate output and select capture behavior.

Independent Modular payloads could be decoded in parallel and composited in
order even when their rectangles overlap. Parallel writes to one canvas need
the stronger proof that actual clipped writes, including extras, do not overlap
and require no background arithmetic. Incremental input, flush, raw layers and
seek must preserve their API contracts. The current ownership optimization
removes repeated whole-canvas work without that scheduling change and does not
require rectangles to be disjoint.

## Reproduction

Local manifests, frozen binaries, reports and stream snapshots are under
`out/multiframe-speed-20261117/`; they are development artifacts and are not
committed. The decode manifest points to the preceding report's frozen after
streams and original `.npy` samples; the encode manifest uses the same nine
original inputs. Use new report paths for every run.

```powershell
python tools/benchmark_decoder_outputs.py --before out/multiframe-speed-20261117/before.dll --after out/multiframe-speed-20261117/after.dll --manifest out/multiframe-speed-20261117/decode-manifest.json --threads 1 4 --repeats 9 --warmups 2 --out out/multiframe-speed-20261117/decode-final.json
python tools/benchmark_multiframe_feedback.py --before out/multiframe-speed-20261117/before.dll --after out/multiframe-speed-20261117/after.dll --manifest out/multiframe-speed-20261117/encode-manifest.json --threads 4 --effort 9 --out out/multiframe-speed-20261117/paired-final.json --streams-dir out/multiframe-speed-20261117/streams
python tools/benchmark_multiframe_feedback.py --before out/multiframe-rate-20261117/before.dll --after out/multiframe-speed-20261117/after.dll --manifest out/multiframe-speed-20261117/encode-manifest.json --threads 4 --effort 9 --repeats 1 --warmups 0 --after-rate-feedback off --out out/multiframe-speed-20261117/legacy-check.json
```

The last command compares against the frozen pre-feedback encoder and validates
the disabled field through the real Python/FFI/native path; it is a correctness
check, not the main performance comparison.
