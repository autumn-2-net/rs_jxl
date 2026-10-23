# FinalPlan Transcript Checkpoint

Experimental utilities, now registered by the parent as `pub mod final_plan`.
This is an encoder decision
transcript, not a neural model, encoded-output hash objective, codestream,
pixel cache, or replacement codec. Codec capture/replay hooks are NOT wired.
Parent reports central module compilation and all FinalPlan unit tests passed
in the workspace snapshot of 1,285 core + 91 other = 1,376 passing tests.
Release and strict checks remain with the main agent. This scoped implementation
agent has run no Cargo command, build or tests and made no root `lib.rs` edit,
dependency change or commit. There is no environment lookup or filesystem autorun.

## Sources and Ownership

Frozen read-only C++ root: `../jxl_python/libjxl`, commit
`b866b3defec13494bfbaf705db93581cda118e37`.

- `enc_final_plan.h/.cc`: records, Exchange, Read, Write, Finished, scope,
  parameters; source path lookup functions are deliberately not ported.
- `enc_frame.cc`: input eligibility/shape/tree budget, JPEG CFL/DC thresholds,
  coefficient order and replay entry/finish.
- `enc_modular.cc`: transforms, global state, WP and global MA tree records.
- `enc_ans.cc`: LZ header/spans, prefix, clusters, uint configuration, ANS method.
- `enc_context_map.cc`: context-map representation and nested entropy calls.
- `enc_patch_dictionary.cc`: empty-patch decision and unsupported marker.
- `enc_lossless_search.cc`: capture the completed winning candidate's actual
  transcript, discard losing/probe transcripts, reject unsupported winners.
- `dec_ans.h`, `ans_params.h`, `modular/options.h`, `modular/transform/transform.h`:
  supporting distance, alphabet, predictor and transform constants.

Owned new files: `crates/rs-jxl/src/final_plan.rs`, local
`final_plan/{wire,schema,tests}.rs`, and this document. BSD attribution is
retained in translated files; the full notice remains in `PORTING_KERNELS.md`.
Existing MA/LZ/sample/metadata/codec modules are not modified.

## Parent Interface

Parent registration is complete: the crate root declares `pub mod final_plan`
with explicit experimental documentation stating that codec capture/replay
hooks are not connected. Nothing runs on module load or ordinary encodes.

- `FinalPlan::capture(Limits)`, `FinalPlan::read(impl Read, Limits)` create an
  explicit capture/replay plan. `read` returns a new plan, not partial mutation
  of an existing plan. `write(impl Write)` preflights size then writes/flushes
  only the supplied sink. Files/paths remain entirely caller-owned.
- `exchange(name, &mut Vec<i64>)` copies on capture, replaces on replay, and
  advances the replay cursor only after a matching tag and successful copy.
  Record order and repeated names matter; there is no tag-indexed lookup.
  A schema error AFTER Exchange consumes the record, as in C++.
- `finished()` exactly implements source Finished: replay must consume all
  records; capture succeeds. `ensure_supported()` is a separate outer check.
  The source winning-candidate wrapper performs it before exporting. The core
  writer intentionally does not silently change Finished/unsupported semantics.
- `Handle::from(plan)` uses `Rc<RefCell<FinalPlan>>`, is same-thread-only, and
  exposes fallible exchange/parameters/input/write/finish operations, never a
  mutable borrow. Reentrant I/O callbacks attempting mutation get an error.
- `scope(Some(&handle), closure)` installs/restores TLS and the existing
  `execution::serial` policy. `scope(None, closure)` suspends recording for
  speculative work. Restoration covers nesting, Result errors and unwinding.
  No exposed movable/forgettable guard or raw pointer is used. Scopes must be
  synchronous: manually spawned tasks/async migration are not propagated.
- `current()`, `replaying()`, `exchange(...)`, `parameters(...)` support local
  hooks. Exchange with no current plan is a no-op, even for an otherwise
  invalid tag. Parameters still validates, matching the source wrapper.
- `parameters(&mut FrameParameters)` exchanges/validates twelve fields and
  only overwrites parameters during replay. `input(InputShape)` checks exact
  shape equality before setting the derived tree limit.
- `schema` contains all 17 known tags, typed frame/input/LZ/map/coeff records,
  borrowed transform/uint/match views, and validators for scalar/array records
  and BFS MA trees. Read itself deliberately does not guess hook context or
  validate records against unrelated placeholder encoder state.

Typical parent capture lifecycle, with hook integration still to be supplied:

```rust,ignore
let plan = Handle::from(FinalPlan::capture(Limits::default())?);
scope(Some(&plan), || -> Result<()> {
    parameters(&mut frame_parameters)?;
    plan.input(input_shape)?;
    // Parent calls the encoder with source-ordered final-decision hooks here.
    Ok(())
})?;
plan.ensure_supported()?;
plan.finished()?;
plan.write(&mut caller_sink)?;
```

Replay uses `Handle::from(FinalPlan::read(&mut caller_source, limits)?)`, the
same synchronous hook sequence, and `finished()` after successful encoding.
Do not serialize a partially bounded/aborted search candidate as the winner.

## V1 Format and Budgets

Canonical output is exactly the tagged V1 layout, with decimal integers:

```text
JXL_FINAL_PLAN_V1 <record_count>
<name> <value_count> <i64> ...
...
```

The writer uses single ASCII spaces and LF after each line, with no extra
fields, JSON, hashes, comments or dependencies. C++ text-mode CRLF is accepted
on read. Classic-locale ASCII whitespace includes space, tab, LF, CR, VT, FF.
Names preserve arbitrary non-whitespace bytes (including non-UTF-8), are
nonempty and at most 64 bytes. Unknown names are preserved, not silently
skipped; a consuming hook will reject an unexpected tag. Values accept signed
decimal, optional plus/minus, leading zeros and the full i64 range.

Source ceilings: 1..1,000,000 records, and at most 16,000,000 values summed
across records. `Limits` can lower these but cannot raise them. The default
serialized-byte budget is 512 MiB, sufficient for the maximal canonical V1
representation under those ceilings; callers can choose their explicit byte
budget. It bounds both actual input bytes (including whitespace/leading zeros)
and canonical output bytes. Reading consumes at most one additional probe byte
to distinguish exact-limit EOF from over-budget/trailing data. Numeric tokens
are accumulated incrementally without unbounded string allocation; record
names stop at 64 bytes. Declared counts and minimum possible input size are
checked before value allocation; allocation errors are returned.

Deliberate boundary hardening relative to C++ stream extraction: negative
sizes (including `-0`), numeric suffixes/concatenated fields and overflow are
rejected; the source writer never emits them. Empty capture is allowed and
Finished succeeds, but Write rejects zero records because source Read rejects
that output. Capture and Write enforce budgets too, whereas source primarily
bounds Read. Canonical valid source output is unchanged. I/O failures can leave
partial sink output; no atomic filesystem write is implied.

Cursor, replay mode, tree limit and unsupported reason are runtime state, never
V1 fields. Input sets `min(2^22, 1024 + width * height * channels / 16)` as the
tree-node budget. Rust computes the cap without unsigned multiplication wrap.

## Frame Parameters

The `frame` payload has exactly 12 values in this order. Numeric enum values
are retained without translating effort levels or inventing policy defaults.

| Index | Source field | Validation |
| --- | --- | --- |
| 0 | speed_tier | 0..9 |
| 1 | modular_group_size_shift | -1..3 |
| 2 | colorspace | -1..41 |
| 3 | responsive | -1..1 |
| 4 | patches override | -1..1 |
| 5 | palette_colors | -65536..70000 |
| 6 | options.predictor | -1..15 |
| 7 | options.max_properties | -1..7 |
| 8 | force_cfl_jpeg_recompression | 0/1 |
| 9 | force_lfs_jpeg_recompression | -1/0 |
| 10 | lz77_careful_first_pass | 0/1 |
| 11 | keep_invisible override | Exactly equals caller's current value |

In particular, replay cannot change invisible-sample retention. The last
field has source equality validation, not a newly invented independent range.

## Hook Order

This is a call-order transcript, NOT a fixed global list of one record per
tag. `Tag::ALL` is only a registry. Parent must preserve conditional calls,
stream iteration order and recursive entropy nesting.

1. Capture entry (`enc_lossless_search.cc::RunPending`) installs a distinct
   plan for each candidate, reduces capture concurrency to one, exchanges
   `frame`, then calls the encode with a null pool. Replay entry
   (`enc_frame.cc::EncodeFrameImpl`) reads the plan, exchanges `frame`, then
   recursively encodes with search disabled and a null pool. `Finished`
   follows successful replay; winner capture checks `unsupported` before Write.
2. `EncodeFrameImpl` exchanges `input` before the ordinary frame path.
   Shape is `[jpeg ? 2 : 1, width, height, channels]` and must equal the current
   input exactly. Eligibility is null pool, non-streaming input, JPEG or
   lossless pixels, non-floating samples and bits_per_sample <= 16. Parent
   must enforce these, plus incompatible search/policy-mode exclusions; the
   generic transcript module has no metadata/encoder flags to check them.
3. JPEG `ComputeJPEGTranscodingData` exchanges `cfl` for channels 0 then 2,
   only for 4:4:4, enabled CFL and three JPEG components. Each map is row-major.
   Then `dc_thresholds` records luma thresholds. After JPEG preparation,
   `ComputeAllCoeffOrders` exchanges `coeff_order`. These tags are conditional
   on the JPEG path, not required for Modular pixel input.
4. Modular `ComputeEncodingData` can first call `FindBestPatchDictionary`,
   exchanging `patches` only for an empty result. Global transform search is
   scoped out of capture; after it, `FinalTransforms(gi, 0)` exchanges
   `transforms`, immediately followed by `global_state`.
5. Each reached `PrepareStreamParams` exchanges its appended `transforms`,
   then `wp`. Preserve `stream_params_` iteration order, not sorted tag names
   or stream IDs. Early empty-stream returns do not invent placeholder records.
   WP/RCT searches are skipped during replay.
6. Global `ComputeTree` reads `tree` instead of learning on replay, or captures
   the finalized tree after selection. Both happen before TokenizeTree.
   Empty/early-return paths remain conditional. There is no `FinalTree` hook
   added here to source `ModularGenericCompress` local-tree learning.
7. Final writing invokes entropy transcripts wherever source calls them.
   `EncodeGlobalInfo` emits the MA-tree entropy call before the residual-token
   histogram call. JPEG global DC context maps can emit earlier nested entropy
   calls. Group/stream and later AC calls stay in their source invocation order.

For each ordinary final `BuildAndEncodeHistogramsImpl` call, the local order is:

```text
lz
matches  (one record per original stream, including empty spans, iff enabled)
prefix
clusters
map_mode (iff a context map reaches the multi-histogram decision branch)
    nested complete entropy call (iff that map is not simple)
uint
ans      (one per resulting histogram iff ANS, none for prefix coding)
```

Source requires initialize_global_state and forbids streaming, fixed-histogram
addition and missing-symbol addition while a plan is active. Cluster replay
requires zero previous histograms/context offset. A one-histogram context map
returns without `map_mode`; direct-audit/fuzzer branches are not silently
treated as the ordinary transcript path. `writer == nullptr` suspends capture
for the ENTIRE entropy cost-only call, including recursive map costs. Capture
records chosen clusters before map encoding, chosen uint configs after map
encoding, and each ANS method around normalization. Replay consumes the same
order without re-running those searches. Source context-map raw-vs-MTF dry
encodes are skipped on replay.

## Exact Source Hook Index

All locations below are relative to the frozen root's `lib/jxl/` directory
at the commit above. Exchange lines identify the record boundary; validation
immediately follows unless separate surrounding checks are noted. Line order
within a file is NOT execution order: use the conditional Hook Order above.

| Record or boundary | Frozen source locations | Integration detail |
| --- | --- | --- |
| Scope / Exchange / Finished | `enc_final_plan.cc:24`, `:27`, `:29`, `:40`, `:43` | Install/restore, member/wrapper exchange, completion check |
| Read / Write | `enc_final_plan.cc:47`, `:72` | Explicit caller-owned Rust streams replace source path I/O |
| frame | `enc_final_plan.cc:87`, exchange `:102` | Twelve fields, source validation and replay assignment |
| Candidate capture entry | `enc_lossless_search.cc:221`, `:263`, `:264` | Capture concurrency, candidate scope, frame parameters |
| Winner export | `enc_lossless_search.cc:416`, `:420`, `:424` | Retain actual winning plan, reject unsupported, then Write |
| Replay entry / finish | `enc_frame.cc:2674`, `:2675`, `:2684` | Scope, frame parameters, recursive encode then Finished |
| input | `enc_frame.cc:2689`, exchange `:2700`, budget `:2702` | Eligibility, exact shape equality, tree-node cap |
| cfl | `enc_frame.cc:988` | Conditional JPEG maps, channel 0 then 2 |
| dc_thresholds | `enc_frame.cc:1124` | JPEG luma thresholds and context-count rebuild |
| coeff_order | `enc_frame.cc:1239` | One-pass JPEG coefficient permutations |
| patches | replay `enc_patch_dictionary.cc:759`, capture `:790` | Empty dictionary only |
| Patch unsupported / suspension | `enc_patch_dictionary.cc:787`, `:793` | Mark nonempty capture unsupported; suspend patch-reference work |
| transforms schema | `enc_modular.cc:379`, unsupported `:386`, exchange `:397` | Validate/apply each transform against evolving channel state |
| Global transform suspension | `enc_modular.cc:1133` | Exclude speculative transform work |
| Global transforms / global_state | `enc_modular.cc:1227`, `:1229` | Final global transforms followed immediately by state |
| Per-stream transforms / wp | `enc_modular.cc:1951`, `:2018` | Reached stream only, transforms before WP |
| tree schema | `enc_modular.cc:424`, exchange `:439` | BFS reconstruction, connected topology, interval/depth checks |
| tree call sites | replay `enc_modular.cc:1465`, capture `:1615` | Before learning on replay; after final selection on capture |
| Global entropy call order | `enc_modular.cc:1717`, `:1737` | Tree entropy before residual histograms |
| Entropy cost-only suspension | `enc_ans.cc:1363` | Null writer suspends the whole call, including nesting |
| lz / matches | `enc_ans.cc:1303`, `:1328` | Header first; one span record per original stream if enabled |
| prefix | `enc_ans.cc:1497` | Select prefix or ANS before clustering |
| clusters | replay `enc_ans.cc:1072`, capture `:1144`, `:1160` | Multi/single capture branches precede map serialization |
| map_mode | `enc_context_map.cc:147` | Conditional multi-histogram map; nested entropy if not simple |
| uint | replay `enc_ans.cc:784`, capture `:1177` | After context map, before histogram serialization |
| ans | replay `enc_ans.cc:708`, capture `:718` | One normalization method per ANS histogram |

These are source integration points, not installed Rust encoder hooks. In
particular, the schema helpers do not execute LZ token reconstruction, MA
installation, transforms, histogram rebuilding, JPEG mutation or winner search.

## Record Validation

The following mirrors source checks. Helpers validate schema/context arguments;
the final column remains a parent codec obligation, not a claimed hook port.

| Tag | Implemented layout/checks | Remaining live-state work |
| --- | --- | --- |
| input | Four values exactly equal expected kind/width/height/channels; derived tree cap | Input eligibility above |
| transforms | 0..512 rows of seven: id 0/1, nonnegative begin, RCT 0..41, nonnegative num_c, colors 0..65536, predictor 0..13, ordered 0/1 | Validate each row against evolving channel count; begin < channels, num_c <= channels, begin + (RCT ? 3 : num_c) <= channels, palette num_c > 0; require actual do_transform success, lossy_palette=false |
| global_state | Three values: max_bitdepth 0..32, maxval i32, colorspace -1..41 | Restore those exact values |
| wp | Two values: exact stream ID, mode 0..4 | Apply mode to that stream |
| tree | Four-value BFS rows, nonempty odd node count <= input tree limit; property -1..255; offset/split i32; leaves predictor 0..13 and multiplier 1..2^31-1; splits require placeholders 0,1; connected queue topology | Translate validated nodes to existing MA type; capture must traverse source tree BFS, not dump storage order |
| lz | Eight values: enabled 0/1; contexts, stream count, min_symbol, min_length, length-config split/MSB/LSB exactly match caller | Apply captured enabled decision; no matcher rerun |
| matches | Triples position/length/distance-symbol; count <= input tokens; positions >= previous end and < input length; length >= min_length and <= remaining input; symbol u32 | Decode distance, clamp and compare tokens as detailed below; rebuild output tokens |
| prefix | One boolean | Select existing prefix/ANS machinery |
| clusters | Exactly builder.size nonempty entries; IDs 0..255; previous histograms and context offset both zero; returns max+1, gaps allowed | Accumulate existing builder histograms into exactly recorded IDs |
| map_mode | Two booleans simple/use_mtf; simple implies ceil_log2(histograms) < 4 | Use chosen representation, preserve nested call ordering |
| uint | Log alpha then 3 values per histogram; log 5..8 for ANS or exactly 15 for prefix; split 0..log, nonnegative MSB/LSB sum <= split; split==log requires both zero | Clear/rebuild histograms from current tokens; context index and cluster index bounds; non-LZ symbol < min_symbol when LZ active; final symbol < 2^log |
| ans | One method 0..12 (ANS_LOG_TAB_SIZE) | Use recorded ComputeBest normalization method, not a new search |
| patches | Exactly [0] | Empty dictionary only; positive patch references unsupported |
| cfl | Exactly map width*height values, each -128..127 | Assign row-major into the selected channel map |
| dc_thresholds | At most seven strictly increasing integers in -1025..1023 | Replace luma thresholds, recompute num_dc_ctxs |
| coeff_order | One pass; 193 values: used_order 0/1 plus three 64-entry permutations, each starts at 0; flag 0 requires supplied source natural order | Capture requires used_acs==1 and one used_orders entry; replay sets used_acs=1 and installs orders at source offsets |

Tree validation additionally walks DFS with per-property intervals initialized
to `[INT32_MIN, INT32_MAX]`. Each split requires `lo <= split < hi`; its left
child sees `[split+1, hi]`, right sees `[lo, split]`, and siblings restore the
previous interval. Stack size must remain strictly less than 2048, including
leaf visits. No recursive untrusted traversal is used.

LZ replay must use `dec_ans.h::SpecialDistance(symbol, width)` for nonzero width
and symbol < 120; otherwise distance is `symbol + 1 - (width ? 120 : 0)`.
Clamp to `min(distance, position, 1<<20)` and require distance > 0. Compare
`input[position+k].value == input[position+k-distance].value` for every k;
overlapping matches are allowed and contexts are NOT compared. Copy literals,
emit length-minus-min_length using the first matched token's context with its
LZ-length flag, emit the distance symbol at context num_contexts, then copy
remaining literals. This scope does not implement that codec mutation or
distance table. Schema acceptance alone therefore does not authorize replay.

Capture unsupported markers are exactly `squeeze or lossy palette` and
`patch reference capture`; latest reason replaces the previous one. Squeeze
or lossy palette returns before its transforms record; nonempty patch capture
suspends nested patch-reference encoding. Parent must reject these incomplete
winning transcripts, not substitute a different decision to make replay pass.

## Verification and Remaining Work

The four new Rust files have passed scoped rustfmt formatting. Static review
is complete and parent registration is confirmed. Parent reports central
module compilation and all FinalPlan unit tests passed in the workspace
snapshot of 1,285 core + 91 other = 1,376 passing tests. These are parent-run
results, not independently rerun by this agent. Release and strict checks are
being prepared by the main agent; no result is claimed for those checks yet.
This checkpoint is complete. No new codec hooks or hook helpers are included
in this batch, and no ARM work or ARM-specific tests are being performed.

Generic Rust tests passed centrally, as reported by parent: source-format golden capture/replay,
i64 extrema, classic whitespace and byte-string tags, malformed/truncated and
trailing input, cumulative records/values/byte budgets, bounded reader probe,
I/O/flush failures, mismatch atomicity, serial nested/suspended/error/panic TLS
restoration, thread isolation, reentrant borrow errors, unsupported semantics,
all frame field boundaries, input equality/tree cap, all remaining schemas,
tree BFS/range/depth checks, and structural LZ/coeff-order checks.

Parent has completed registration and central compilation/tests and still owns
release and strict checks, subsequent normal generic/x86 validation;
all actual codec hooks and runtime state applications; source eligibility and
search suspension; retaining/exporting only the completed winning transcript;
and decoder-oracle acceptance. No end-to-end capture/replay or full local-tree
coverage is claimed. Reusing MA/LZ/entropy decisions is never optimization
against whole-JXL identity, and transcript equality is not a substitute for
decoded sample/JPEG-byte correctness.
