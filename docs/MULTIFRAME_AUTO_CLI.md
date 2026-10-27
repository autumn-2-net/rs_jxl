# Automatic Multiframe CLI

`rs-jxl encode-frames` accepts ordered still PNG/JPEG paths and invokes
`rs_jxl::multiframe::encode_auto`. It uses the existing CLI input decoders and
transactional output writer. It does not run source-search or map effort levels.

```powershell
cargo run -p rs-jxl-cli -- encode-frames frame000.png frame001.png frame002.png -o animation.jxl --duration 100 --tps 1000 --loops 0
```

For an already-built binary:

```powershell
rs-jxl encode-frames frame000.png frame001.png frame002.png -o animation.jxl --durations 1,2,1 --tps 30000/1001 --loops 2 --threads 4
rs-jxl encode-frames first.png second.png -o animation.jxl --reference previous --no-mask --no-multi-rect --rct 0 --predictor 0
rs-jxl encode-frames first.png second.png -o animation.jxl --tile-size 16 --max-rectangles 32
rs-jxl encode-frames first.png second.png -o animation.jxl --no-prune --no-adaptive-rectangles
```

## Ordering And Timing

- Paths are used in argument order, with no sorting or implicit file discovery.
  A single input is also accepted. Formats are detected by signature.
- `-o/--output` is required. Use `-o -` for binary stdout. At most one input may
  be `-` for stdin. All diagnostics go to stderr.
- `--duration TICKS` applies one duration to every input; its default is 100.
- `--durations TICKS,...` requires exactly one unsigned 32-bit value per input.
  It can be repeated. It conflicts with an explicitly supplied `--duration`;
  a one-element list is not broadcast to multiple inputs.
- `--tps NUM[/DEN]` defaults to `1000` (1000/1 ticks per second). Duration in
  seconds is `TICKS * DEN / NUM`. The numerator must be in `1..=1073741824`
  and denominator in `1..=1024`. No rounding or fraction normalization occurs.
- `--loops N` defaults to 0, meaning unlimited repetition; 1 means one play.
- A zero duration remains zero: it is a non-display layer, not a one-tick
  display frame. Coalesced decoder output need not contain one displayed frame
  per input when zero durations are supplied.

## Automatic Strategies

The core always considers an exact full frame. `--reference auto` (the default)
maps to `Reference::SmallestCrop` in the automatic API and permits both previous
and first-frame references, measured by actual encoded bytes. This is not the
older geometric bounding-box heuristic used by `multiframe::plan`.

`--reference previous` and `first` restrict reference candidates; `full` uses
full frames only. Mask and multi-rectangle strategies are enabled by default
where the core can preserve samples and metadata exactly:

- `--no-mask` disables mask candidates. The current mask route requires an
  existing opaque integer gray-alpha/RGBA image with exactly one alpha channel;
  it does not insert alpha, reinterpret partial transparency or discard hidden
  color samples. Eligibility remains the core encoder's decision.
- `--no-multi-rect` disables both tile and adaptive multi-rectangle candidates.
  Intermediate rectangles have zero duration; the final rectangle carries the
  input duration.
- `--no-prune` disables proven-bound pruning. All eligible trials then complete
  or the command fails explicitly on an error. By default, inexpensive candidates
  may execute first; a candidate stops only when a provable byte lower bound
  cannot win under the canonical tie rule. Estimates cannot remove old candidates.
- `--no-adaptive-rectangles` disables the additional `MergedPrevious`/`MergedFirst`
  proposals, retaining legacy tile candidates. The estimate uses the initial full
  frame's measured bytes per pixel only to propose rectangles; actual encoded
  bytes select the winner. This control is independent of pruning.
- `--tile-size PIXELS` defaults to 32 and must be positive.
- `--max-rectangles N` defaults to 32 and accepts positive native `usize` values,
  with no extra 4096 cap. Geometry is validated even if candidates are disabled.

Candidate reports identify each logical input frame (one-based), the winner
index, completed candidate count, pruned count and native `analysis_elapsed`.
The zero-based winner index addresses completed candidates in canonical order,
not execution order. New Merged candidates follow all legacy candidates to keep
existing equal-byte preferences. `candidate=N` lines report strategy via Rust
`Debug`, actual `encoded_bytes`, `encoded_frames`, `trial_pixels` and measured
`elapsed`. Bytes include layer headers and private patch atlas bytes; frame counts
refer to regular layers, excluding private reference-only patch frames.

Separate `pruned=N` lines report strategy, proven `lower_bound`, `encoded_frames`,
`planned_frames`, `trial_pixels` and `elapsed`. They never report `encoded_bytes`
or claim to be winners. `encoded_frames` counts fully finished regular layers,
excluding a partially encoded layer; `planned_frames` is the total planned layer
count. Trial pixels include areas submitted for interrupted layers, not analysis
scans. `difference` lines expose native zero-based input `reference`, exact
`changed_pixels` and `bounds` (`None` for no changes); reference indices are not
JXL reference-slot numbers. No reference analysis gives no difference lines.
All timings and difference statistics come from Rust, without adapter analysis.

The final line gives the complete output byte count, including shared headers
and any container overhead. Candidate byte counts are not standalone JXL sizes.

Selection minimizes encoded bytes, not time; ties and candidate eligibility
follow the core API. This is a per-input local minimum among eligible trials,
not a proof of globally optimal compression. No synthetic effort or timing
estimate is reported. With sufficient budgets, pruning preserves the winner for
the same eligible set. No performance or compression improvement is claimed yet.

The existing manual `Tools` flags (for example `--rct`, `--predictor`,
`--entropy`, `--lz77`, palette and squeeze options) and `--threads 1..=256` are
forwarded through `EncodeOptions`. Incompatible tool combinations fail in the
core; they are not silently disabled. `--effort`, `--source-effort` and
`--source-search` are not accepted by this command.

## Exact Input Semantics

All inputs must have equal dimensions, decoded layout and sample storage,
declared color/alpha sample types and full `ImageMetadata`. Metadata is compared
before the common animation header is attached. The CLI does not resize,
rescale depth, add/drop alpha, premultiply, convert color or apply orientation.
RGB under zero alpha remains significant. Declared alpha depth and association
must also match. Equivalent-looking but unequal color metadata is rejected.

The existing input policy is unchanged:

- PNG gray, gray-alpha, RGB and RGBA at 8/16 bits are accepted. Palette, packed
  gray and tRNS expand through the existing lossless PNG decoder. APNG is rejected.
- JPEG input is 8-bit gray/RGB decoded with zune-jpeg. Losslessness is relative
  to those decoded samples, not the original JPEG bytes. CMYK/YCCK and non-8-bit
  JPEG are unsupported. Use `transcode-jpeg` separately for byte reconstruction.
- Supported PNG sRGB/gAMA/cHRM/iCCP and JPEG ICC information is retained.
  Untagged inputs assume sRGB gray/RGB. Unsupported metadata such as EXIF/XMP
  requires `--strip-metadata`; malformed metadata is not automatically repaired.
- `--strip-metadata` applies to every input and explicitly discards unsupported
  metadata while retaining supported color metadata and decoded samples, matching
  the existing input implementation. It may change displayed orientation and
  emits a warning. It does not make unequal color metadata or sample/alpha types
  compatible. Explicit CLI animation timing is still attached afterward.

## Limits And Publication

No additional CLI input-byte, decoded-sample or retained-sequence byte cap is
imposed. Format bounds, checked size arithmetic, allocation failures and existing
codec limits still apply. Work budgets are not total process-memory limits:
input decoding, metadata attachment, analysis, encoder scratch and retained
candidate/output storage also consume memory.

Core defaults are used directly for:

| Option | Current Default | Scope |
| --- | ---: | --- |
| `--max-frames` | usize::MAX (unlimited) | Logical inputs, not generated layers |
| `--max-total-trials` | usize::MAX (unlimited) | Candidate trials across the sequence |
| `--max-trial-pixels` | u64::MAX (unlimited) | Sum of pixel areas submitted across trials |

These overrides accept zero, explicitly refusing work rather than meaning
unlimited. Geometry controls remain positive. Resource exhaustion is an error; there is no
silent frame dropping, strategy downgrade or partial output. Core output and
pixel limits still apply. Check `encode-frames --help` for defaults if the core
configuration changes. These changes add no new codec limits.

The output is checked against every input before decoding and again before
publication, using the existing path/alias checks. Existing files require
`--force`, which never authorizes overwriting an input. Encoding must succeed
before the shared transactional file writer is invoked; failed input decoding,
metadata checks or encoding do not create or replace the output. Binary stdout
is written only after encoding succeeds, but stdout itself cannot be rolled back
after an I/O failure.

## Verification

On 2026-10-27 the release CLI built successfully, and all 79 non-ignored CLI
tests passed, including the new pruning/adaptive controls and separate
completed/pruned reporting. Two unrelated opt-in CLI tests were not run.
The core also passed independent official djxl 0.12/0.13 checks, including
actual merged-rectangle winners. Native logic and acceptance details are in
[MULTIFRAME_AUTO.md](MULTIFRAME_AUTO.md); real-image tradeoffs are in
[MULTIFRAME_NATIVE_BENCHMARK.md](MULTIFRAME_NATIVE_BENCHMARK.md).

Unit tests live in `crates/rs-jxl-cli/src/multiframe.rs`. They cover CLI/default
mapping, timing and limits, absent effort/search modes, metadata/storage/alpha
compatibility, explicit stripping, sample preservation, output safety, measured
reporting, and 8/16-bit RGBA PNG sequence round trips through the maintained
decoder, including hidden RGB and per-frame durations.

The centralized 2026-10-24 release run passed all twelve CLI tests. Independent
official-decoder acceptance for the core strategies is recorded in
[MULTIFRAME_AUTO.md](MULTIFRAME_AUTO.md). This does not claim complete decoder
conformance or complete pixel-lossless performance acceptance.
