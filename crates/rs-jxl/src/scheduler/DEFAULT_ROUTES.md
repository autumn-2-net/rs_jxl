# Source Default Route Audit

Baseline: `b866b3defec13494bfbaf705db93581cda118e37`. This distinguishes
candidate decisions from execution scheduling; successful decoding does not
prove both are complete. Source locations below are relative to `lib/jxl/`.

## Default-Enabled Execution

| Source | Default | Rust status |
| --- | --- | --- |
| `enc_lossless_probe.cc:90` RunProbeTasks; `enc_lossless_workers.h:31` | Dynamic task queue, 8 slots, caller runner | `batch::run_ordered`; native cached-screen batches use the single caller pool |
| `enc_lossless_probe.cc:358` BATCH_PROBES | On when unset | Existing sorted/deduplicated cells now reach the ordered batch callback |
| `enc_lossless_search.cc:219` RunPending; `:840` initial concurrency | Up to 4 small-frame trials; 256 MiB divided by pixels * (5 + extras) * 32 | Initial batches dispatch and commit in source plan order; native callback wired |
| `enc_lossless_probe.cc:1021`, `:1188` | Control and validation task batches | Independent ordered batch callback on the caller pool |
| `enc_lossless_probe.cc:1062`, `:1124` | Whole candidate screen and shortlist tasks | Native `BatchEncoders::prefix` uses bounded independent roots, ordered tiles and per-attempt budgets; no tile-wave barriers |
| Batch inner encoder runner | Null runner for independent jobs; caller runner for a single trial | Native batches use `encode_serial`; inner tools run on the calling worker without creating/installing another pool |

Native search creates exactly one pool after validating `threads` in 1..=256.
Single full/probe trials use `encode::encode_with_pool`. Independent batch jobs
use `encode::encode_serial`, with scoped serial dispatch across group encoding,
sample gathering, MA properties/children and grouped trees. The serial scope
restores its prior state after normal return, errors and unwinding. Public
single-thread encoding uses that path too, without allocating a worker pool.
The generic FnMut API remains serial unless BatchEncoders is supplied.

Native small-probe screening runs whole candidates independently. A top-six
mutex publishes a monotonically tightening atomic cutoff after every complete
candidate. The optional feedback bound uses strict `>`; Variable exploratory
scores always complete. Shortlist tasks start from completed screen totals and
visit remaining tiles hardest-first with the fixed `control_bytes * 1.10`
bound. Unlike the screen, that bound applies to Variable candidates too. A
bounded prefix is `None`, never a completed total. All phases share the driver's
cumulative work ceiling and charge actual tile attempts, including failures.
The original FnMut/batched APIs remain valid; without the optional prefix hook,
their tile-wave execution remains a compatibility fallback. Native artifacts
and attempted adaptation records are ordered by candidate/tile, not completion
time. Timing and unnecessary feedback-work counts are not deterministic.

## Optional Decisions

The preflight and frontier decisions below now have explicit DriverOptions
and source-aligned control flow. They remain opt-in. This is an implementation
record; consult the central verification checkpoint for actual execution.

- `enc_lossless_search.cc:529..563`: PREFLIGHT_YIELD, ANCHOR_FIRST, scope
  routing, multiscale, calibration reuse/order, early family, plan-first and
  plan-buckets all require explicit enabling. GEOMETRY_FIRST and AUTOMATIC_YIELD
  are locally default-on but require the default-off plan-first outer route.
- `enc_lossless_search.cc:571..789`: direct/compact/dense/extended frontiers,
  global/strip tournaments and parallel finalists require explicit enabling.
- `enc_lossless_search.cc:857..875`: CONTEXT_FEEDBACK is locally default-on
  only inside PREFLIGHT_YIELD. Statistical race and sibling transfer are opt-in.
  Statistical feedback, sibling transfer and sibling race are now ported.
- `enc_lossless_probe.cc:340..361`, `:602`, `:717`, `:764`: data-parallel,
  cooperative, training caps, LPT, cheap control and statistical race are not
  defaults. Training defaults to no cap. Basic batching is the exception above.
- `enc_lossless_probe.cc:429..475`: context-side fallback 256 and band-pixel
  fallback apply to explicitly selected context/atlas routes, not default crops.
- `enc_lossless_search.cc:496..505`: SEARCH_CONFIRM and PROBE_FEEDBACK are
  default-off experiments. Asynchronous candidate execution itself is default;
  it does not turn on feedback pruning. Confirmation has up to six witnesses
  on at most four source workers. Rust retains batches of at most four, including
  a between-batch barrier, exact witness-cache totals and resident crop limits.
- Token-channel, WP, LZ and MA parallel experiments require their own explicit
  switches. Parent's `ModularOptions::parallel_tokens` represents TOKEN_PARALLEL;
  cloning native base options preserves it. It is not a new source default.

No missing default-enabled candidate-selection branch was identified in this
search/probe switch audit. This is narrower than full encoder parity: the
optional workflows require their own independent decoder/performance validation.
No source-default option was disabled to obtain output identity.

## Remaining Execution Differences

- C++ `RunPending` supplies an incumbent byte bound for nonbaseline,
  noncalibration full-frame challengers. The native bridge now passes it to
  encoder section budgets and may stop once finalized bytes prove the candidate
  cannot win. Baselines remain complete. This is distinct from hard
  `max_output_bytes`; a generic resource failure is never a proven bound.
- Cooperative arenas now port descendant priority, own-group helping, quantum
  and optional quiet wakeups onto caller-owned workers. These remain opt-in
  routes; see the root `docs/PARALLEL_EXECUTION.md` for safety and validation.
- Environment parsing and final-plan diagnostic capture remain separate from
  explicit Rust options. No new defaults are inferred from process environment.

Native prefix screening already uses completed-candidate asynchronous feedback.
The new `probe_lossless_candidates_parallel` convenience API exposes that path
without requiring a custom prefix-batch adapter. Legacy `FnMut` batch callbacks
remain compatible and necessarily publish only completed batches; this is an
explicit API limitation, not the native scheduler's execution route.

Explicit LZ77 tool switches in base options now survive candidate and patch
reference projection. Candidate-owned parser/bucket/seed choices and the
entropy wire configuration still take precedence; LZ-disabled candidates stay
disabled. This prevents a configured parallel/index route from disappearing
during native scheduling.

The parent reported 812 core + 14 acceptance tests passing for the asynchronous
screen checkpoint. The later shortlist-task extension needs its own central
checkpoint; no cargo build/test was run by this worker.

## Coverage Scan

`enc_lossless_search.cc:791` requires SCAN=1, effort >=2 and >=2^19 pixels.
Scanning is not a newly enabled default. `enc_lossless_scan.cc:29..64` assigns
whole tile rows to workers; that path is now `scan_lossless_input_with_pool`.
Native search uses its existing pool when coverage_scan is requested. Planar
passes finish in source order, preserving strict peak-row comparisons and ties.
Literal evidence and bounded RGB palette counting remain serial. Scratch is
still bounded; every input sample contributes to the same tile as before.

Tests in `batch_tests.rs`, `prefix_screen_tests.rs`, `prefix_shortlist_tests.rs`
and `scan_tests.rs` cover stable commit, attempted
work limits, cache parity, small-probe finalists against exhaustive scores,
confirmation totals, all scan fields/regions, hidden RGB, mixed-depth extra
planes, plane-order ties, and float rejection. Central execution checkpoints
are recorded in `docs/PORT_VERIFICATION.md` at the repository root.

## Native Frame Routing

`enc_frame.cc:1903` CanDoStreamingEncoding is now applied before source patch
discovery. In the supported nonresponsive, one-pass pixel-lossless domain:

- Auto buffering (-1) streams at tiers >=1 when the root frame has >8 groups.
  Tier 0 remains whole-frame. FullFrame (0) always remains whole-frame.
- StreamingAbove2048 (1) additionally requires a dimension >2048. Streaming
  (2) skips that size/speed heuristic but still needs >8 groups.
- Explicit patches On blocks all streaming requests, even when the detector's
  separate tier/depth gate would subsequently reject patch discovery.
- Streaming selects existing DC-frame-area orchestration, local palette/RCT,
  independent stream trees/models/LZ and streaming tree/residual entropy. It
  does not start a patch detector. Shared-tree/global-palette options remain
  enabled as the full-frame fallback, not as the effective streaming policy.
- Sampling uses ROOT frame NumStreams/group size and raw fraction exactly
  once. For 3840x2160 auto geometry: dim256, 135 groups, 4 DC groups, 165 streams;
  e9 fraction remains `0.5f32 * 1.3f32`, not divided by groups or threads.
- RoundtripPatchFrame retains original buffering and auto-group inputs. Its
  Gradient/patches-Off atlas resolves routing and sampling on its own geometry
  using metadata only, including private signed atlases. There is no new
  reference-only streaming exclusion in this adapter.
- Candidate equality, frontier families and search eligibility include
  buffering. Complete search accepts only Auto/FullFrame, matching
  `enc_lossless_search.cc:44`; explicit streaming remains a single-candidate API.
- Native projection clears inherited `source_constant_channels`: the source
  `JXL_MA_CONSTANT_CHANNELS=1` extension is not default-enabled.

This closes a material execution-domain mismatch, not numerical parity proof.
Remaining audits include native MA scoring/rounding versus the default C++
float/Highway path and worker scheduling. Local source learners now use group 0,
as C++ LearnTree does; encoding retains absolute root stream IDs. The parent
ran the routing/reference units and the official frame-area/reference suites.
See the root verification document for checkpoint counts and measurements.

## Streaming Entropy Boundary

For the missing-symbol integration audit, `streaming_mode` is NOT by itself an
instruction to inject unseen symbols:

- `enc_frame.cc:1786` skips global `ComputeTree`/`ComputeTokens` entirely in
  streaming mode. `enc_modular.cc:1701` writes the absent-global-tree flag and
  returns before the shared residual model when `tree_tokens_` is empty.
- `enc_modular.cc:1755` then uses `ModularGenericCompress` for each independent
  stream. `modular/encoding/enc_encoding.cc:1133..1163` creates a fresh tree code,
  resets `code = {}`, and creates a fresh residual code with that stream's width.
  `ForModular(..., true)` sets streaming policy but leaves missing-symbol
  injection off. Local training uses group 0; encoding uses the absolute group.
- `enc_modular.cc:1725` explicitly enables missing-symbol injection only for
  the shared global residual model, after its tree has been encoded. It is not
  reached by the ordinary no-global-tree pixel streaming route. Do not enable
  it on every local streaming tree/residual or invent cross-area model reuse.
- Where injection is explicitly requested, `enc_ans.cc:1465` adds one count to
  EACH of 256 symbols in EACH context, including existing symbols and the LZ
  distance context. This occurs after token histogram construction and before
  prefix/ANS selection and clustering; it does not increase `total_tokens`.
- Streaming ANS uses log alphabet 8; prefix uses 15. Streaming fast/best uint
  config search is skipped. These policies are independent of symbol injection.

`CanDoStreamingEncoding` requires strictly more than eight groups. Auto
buffering also excludes tier 0; explicit buffering 1 requires a dimension
greater than 2048. Patches/noise forced On, progressive/DC modes, resampling,
lossy palette, max-error mode, responsive Modular, and a non-None Modular color
transform prevent streaming. Complete source search accepts buffering -1/0
and output_mode <=0 only, although auto buffering may still select internal
DC-frame-area streaming. Those internal areas are not a streaming-output API.
