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
- `--no-multi-rect` disables multi-rectangle candidates. Intermediate rectangles
  have zero duration; the final rectangle carries the input duration.
- `--tile-size PIXELS` defaults to 32 and must be positive.
- `--max-rectangles N` defaults to 32 and accepts `1..=4096`.

Candidate reports identify each logical input frame (one-based), the winner
index and all evaluated candidates (zero-based). Each candidate reports its
strategy via Rust `Debug`, actual `encoded_bytes`, `encoded_frames` and measured
`elapsed`. Bytes include layer headers and private patch atlas bytes; frame
counts refer to regular layers, excluding private reference-only patch frames.
The final line gives the complete output byte count, including shared headers
and any container overhead. Candidate byte counts are not standalone JXL sizes.

Selection minimizes encoded bytes, not time; ties and candidate eligibility
follow the core API. This is a per-input local minimum among eligible trials,
not a proof of globally optimal compression. No synthetic effort or timing
estimate is reported.

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

Existing per-input limits remain: 512 MiB encoded bytes, 16,384 pixels per
dimension, 64,000,000 pixels and 512 MiB decoded samples. Retained sequence
sample buffers plus ICC profiles are additionally capped at 512 MiB. This is
not a total process-memory limit: input decoding, metadata attachment, encoder
scratch space and candidate/output storage also consume memory.

Core defaults are used directly for:

| Option | Current Default | Scope |
| --- | ---: | --- |
| `--max-frames` | 10000 | Logical inputs, not generated layers |
| `--max-total-trials` | 100000 | Candidate trials across the sequence |
| `--max-trial-pixels` | 4000000000 | Sum of pixel areas submitted across trials |

These overrides must be positive. Resource exhaustion is an error; there is no
silent frame dropping, strategy downgrade or partial output. Core output and
pixel limits still apply. Check `encode-frames --help` for defaults if the core
configuration changes.

The output is checked against every input before decoding and again before
publication, using the existing path/alias checks. Existing files require
`--force`, which never authorizes overwriting an input. Encoding must succeed
before the shared transactional file writer is invoked; failed input decoding,
metadata checks or encoding do not create or replace the output. Binary stdout
is written only after encoding succeeds, but stdout itself cannot be rolled back
after an I/O failure.

## Verification

Unit tests live in `crates/rs-jxl-cli/src/multiframe.rs`. They cover CLI/default
mapping, timing and limits, absent effort/search modes, metadata/storage/alpha
compatibility, explicit stripping, sample preservation, output safety, measured
reporting, and 8/16-bit RGBA PNG sequence round trips through the maintained
decoder, including hidden RGB and per-frame durations.

The centralized 2026-10-24 release run passed all twelve CLI tests. Independent
official-decoder acceptance for the core strategies is recorded in
[MULTIFRAME_AUTO.md](MULTIFRAME_AUTO.md). This does not claim complete decoder
conformance or complete pixel-lossless performance acceptance.
