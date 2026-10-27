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
rs-jxl encode-frames x.png a.png b.png a2.png -o animation.jxl --retained-reference --report-json
rs-jxl encode-frames a.png b.png b2.png a2.png -o animation.jxl --retained-reference --reference-lookahead --report-json
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
  proposals (and `MergedRetained` when enabled), retaining tile candidates. The estimate uses the initial full
  frame's measured bytes per pixel only to propose rectangles; actual encoded
  bytes select the winner. This control is independent of pruning.
- `--tile-size PIXELS` defaults to 32 and must be positive.
- `--max-rectangles N` defaults to 32 and accepts positive native `usize` values,
  with no extra 4096 cap. Geometry is validated even if candidates are disabled.

### Retained References (Opt-In)

`--retained-reference` requires `--reference auto` (also the default). Combining
it with `previous`, `first` or `full` fails before opening inputs or publishing
output. Omitting the flag preserves the previous/first reference policy.

With this flag, native Rust manages two complete canvas states in standard JXL
reference slots 1 and 2. Without `--reference-lookahead`, a selected delta retains
its source state and stores the new state in the other slot. A full replacement
retains the previous state and
overwrites the other slot. Frame zero is therefore no longer permanently pinned.
For example, `X, A, B, A2` can reuse A for A2 instead of encoding against unrelated
B or X. This is a finite retained set, not a search over every past input.

Candidates referencing a retained frame other than frame zero or the immediate
predecessor use `CropRetained`, `MaskRetained`, `TilesRetained` and
`MergedRetained`. Existing first/previous labels retain their meanings. The
first comparison (input 1 against input 0) remains `CropPrevious` etc., just as
in the default automatic mode.

All candidates for a logical input are evaluated against the same committed
reference states. Failed or pruned trials do not change these states. Multi-rect
trials use the destination slot for intermediate composites; only their final
complete canvas is eligible as a later logical reference. Slot 0 is not used as
a retained canvas and slot 3 remains reserved for generated patch atlases.

The output uses existing JXL references, cropping and blending. This option
does not add temporal patch matching, motion estimation, new decoder features,
resizing or hidden-color normalization. Alpha-mask safety rules remain unchanged.
It can reduce bytes for recurring content, but can lose to the default policy
when an evicted frame recurs. It is not a guaranteed sequence-wide improvement;
compare total bytes and encode/decode cost on the intended workload.

### Reference Lookahead (Opt-In)

`--reference-lookahead` defaults to false and requires `--retained-reference`,
which still requires `--reference auto`. Invalid combinations fail before input
I/O or output publication. Omitting lookahead keeps the existing retention
policy, including when retained references are enabled.

The native heuristic compares current saved states against only the **next**
input using exact full-pixel `DiffMap`s. It prefers smaller bounding-box area,
then fewer changed pixels, and changes the retained old state only if strictly
better than the previous policy's choice. Equal scores keep the previous choice.
The current input is also compared with the next input to prepare its map after
saving. At most three new future maps are computed for a logical input; maps for
the states actually retained are cached, with at most two retained-map reads
reused on the next iteration. No future input is trial-encoded, and no input
beyond the immediate next one is examined.

For `A, B, B2, A2`, this can preserve A while encoding B2 from B and replacing B's
slot after blending. Multi-rectangle candidates read the original source for the
first rectangle, then the destination's intermediate composite for subsequent
rectangles, including when source and destination are the same slot. Comparisons
remain exact for every channel, hidden color, extra plane and float sample word;
metadata compatibility is unchanged. Better next-input geometry is not a promise
of fewer bytes for either the next frame or the whole sequence.

### Reports

Candidate reports identify each logical input frame (one-based), the winner
index, completed candidate count, pruned count and native `analysis_elapsed`.
The zero-based winner index addresses completed candidates in canonical order,
not execution order. New Merged candidates follow all legacy candidates to keep
existing equal-byte preferences. `candidate=N` lines report strategy via Rust
`Debug`, `reference_index`, `source`, `save_as_reference`, actual `encoded_bytes`, `encoded_frames`, `trial_pixels` and measured
`elapsed`. Bytes include layer headers and private patch atlas bytes; frame counts
refer to regular layers, excluding private reference-only patch frames.

Separate `pruned=N` lines report strategy, reference fields, proven `lower_bound`, `encoded_frames`,
`planned_frames`, `trial_pixels` and `elapsed`. They never report `encoded_bytes`
or claim to be winners. `encoded_frames` counts fully finished regular layers,
excluding a partially encoded layer; `planned_frames` is the total planned layer
count. Trial pixels include areas submitted for interrupted layers, not analysis
scans. `difference` lines expose native zero-based input `reference`, exact
`source`, `changed_pixels` and `bounds` (`None` for no changes); reference indices
are not JXL reference-slot numbers. No reference analysis gives no difference lines.
All timings and difference statistics come from Rust, without adapter analysis.

- Candidate `reference_index` is the zero-based logical input index, or `None`
  for a full replacement. `source` is the JXL reference slot for the first layer
  of this candidate; full replacements report 0 without reading a prior canvas.
- Candidate `save_as_reference` is the final layer's destination slot, or 0 for
  the final stream frame, which is not saved. For a pruned trial it is only the
  intended destination, not a committed save.
- Each logical frame reports `retained_reference`, `references_before` and
  `references_after`. These four-entry arrays map JXL slots to zero-based input
  indices only when a complete logical canvas is available. `None` means there
  is no complete input state, which also covers an intermediate partial composite;
  it must not be interpreted as a guarantee of an empty decoder reference buffer.
- `reference_lookahead` is the boolean next-input policy indicator.
  `lookahead_comparisons` (native `usize`) counts new future exact `DiffMap`
  computations for this logical frame, at most 3. Disabled lookahead and the last
  input report zero. Current-input/static analysis and cached-map reads do not
  count toward this field.
- `reused_payloads` (native `usize`) counts cached unmasked crop payload hits;
  `reused_payload_pixels` (u64) sums their rectangle areas. This cache is automatic,
  with no new CLI option, and only reuses equivalent frame payloads for repeated
  identical rectangles within one logical input. Masked payloads and reuse across
  inputs/calls are excluded; patch mode may conservatively bypass reuse.
  `trial_pixels` still counts all submitted rectangle areas, including hits.
  These counters are neither lookahead-map reads nor byte-saving estimates.
- A final multi-rectangle candidate may leave its scratch destination as `None`
  in `references_after`: its preceding rectangles saved partial composites but
  the final stream frame cannot save the complete result. Slot 3 is always
  `None` in these logical maps, even if private patch storage was used.

`--extra-tile-sizes 8,16,64` optionally expands rectangle candidates beyond
`--tile-size` (default 32). The empty default preserves the previous grids.
Identical partitions are skipped; repeated rectangle payloads reuse the native
cache where eligible. Each JSON candidate includes `tile_size` (null for
full/crop/mask). More scales can cost more search time; they do not raise the
per-candidate `--max-rectangles` or bypass explicit trial budgets. See
[Native Automatic Frames](MULTIFRAME_AUTO.md#optional-multiscale-partitions).

`--report-json` replaces the textual candidate report with one compact JSON
line on stderr; it never changes binary stdout or writes a separate report file.
Warnings and the existing informational note can precede that line, so the whole
stderr stream is not a JSON document. The line has `operation: "encode_frames"`,
`inputs`, `output_bytes` and `frames`. Each frame contains zero-based `index`,
`winner_index`, the reference maps, `candidates`, `pruned`, `differences` and
`analysis_elapsed_seconds`. Candidate timings use `elapsed_seconds`; missing
references/bounds are JSON `null`. Bounds are objects with `x`, `y`, `width` and
`height`. Strategy strings match the textual Rust names. Pruned records omit
`encoded_bytes`. The report describes the encoded result; an ensuing output I/O
failure is still an error, not successful publication.

The final line gives the complete output byte count, including shared headers
and any container overhead. Candidate byte counts are not standalone JXL sizes.

Selection minimizes encoded bytes, not time; ties and candidate eligibility
follow the core API. This is a per-input local minimum among eligible trials,
not a proof of globally optimal compression. No synthetic effort or timing
estimate is reported. With sufficient budgets, pruning preserves the winner for
the same eligible set. Workload-specific measurements, not a general improvement
guarantee, are recorded in [MULTIFRAME_RETAINED_BENCHMARK.md](MULTIFRAME_RETAINED_BENCHMARK.md).

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

Lookahead adds adapter validation, CLI flag/report fixtures and a mixed-duration
CLI round trip, plus Python runtime tests for source-slot overwrite,
multi-rectangle composition, default/false equivalence, exact extra planes and
float words, payload reuse accounting, and context recovery. Official-decoder
Python coverage uses the existing `RS_JXL_DJXL` helper for both uint8/uint16 and
positive/mixed-zero durations; configure both official djxl 0.12 and 0.13 paths
with the platform path separator. All 18 focused CLI tests and 169 Python tests
passed in the centralized run, with official djxl 0.12/0.13 enabled. Native
correctness and e9 results are in
[MULTIFRAME_LOOKAHEAD_BENCHMARK.md](MULTIFRAME_LOOKAHEAD_BENCHMARK.md).
The following counts describe earlier checkpoints.

Retained-reference and JSON-report changes add focused tests in
`crates/rs-jxl/tests/multiframe_retained.rs` and the CLI module. They cover older
reference selection and measured size benefit, 8/16-bit hidden color, extra-only
changes, exact floating words, zero/mixed durations, multi-rectangle reference
bookkeeping, pruning equivalence, opt-in/default semantics and context recovery.
The retained-reference update passed all 16 focused CLI tests, 10 new native
reference/patch tests, 36 existing native multiframe/context regressions and
45 FFI tests. The release CLI and FFI build, strict all-target Clippy and the
159-test Python suite also passed. Python independent-decoder checks used
official djxl 0.12 and 0.13. See
[MULTIFRAME_RETAINED_BENCHMARK.md](MULTIFRAME_RETAINED_BENCHMARK.md) for the
e9 measurements and their limits. The following results describe earlier runs,
not a repeat of every CLI test in this update.

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
