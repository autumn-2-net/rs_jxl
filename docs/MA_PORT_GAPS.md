# MA Learner Source Audit

Audited 2026-10-23 against local frozen libjxl HEAD
`b866b3defec13494bfbaf705db93581cda118e37`. Source references below are under
`../jxl_python/libjxl/lib/jxl/`. "Default" means lossless speedups
enabled, FastEntropy disabled, and the corresponding experimental environment
variable unset. This is not a claim that every source experiment is enabled by
default or that the complete pixel-lossless migration is accepted.

## Current Validation Checkpoint

Parent-reported status, 2026-10-23:

- Production source gathering transfers prepared columns directly to
  `ma_train::train_prepared`; the temporary production compatibility adapter
  and raw static-ID column are removed. Public legacy row input remains supported.
- `ModularOptions::source_ma_histograms`, `source_ma_scoring` and
  `source_ma_search` are forwarded through their grouped `apply` methods.
  Prepared-storage forwarding and all scoring/histogram fields are active.
- The prior central 1376-core workspace checkpoint, strict Clippy and MSRV
  checks passed. These are parent-run results, not local validation commands.
- All 48 official scoring pixel checks passed across three profiles, including
  description cost, FastEntropy and sliding clipped entropy. Search oracle
  coverage now passes all three profiles: quantile-32, quantile-64 and refinement,
  totaling 48 checks. Quantile-64 official coverage is no longer pending.
- A subsequent 1024x512 Gray16 race case reached 524287 stored records and
  reported `race_nodes=1`, `race_draws=3584`, `race_pruned=0`. Both official
  decoder pixel checks passed. This proves race execution, not pruning;
  pruning/evidence/cache-rebuild behavior has separate component tests.
- An earlier central release run passed all 11 `source_` suites: search 48,
  scoring 48, histogram 16, dedup 16, race 2, parallel gather 16, stripes 12,
  source tiers 80 and LZ 172 checks. The parent reported 5.69 seconds for this
  validation run; this is not an encoder throughput benchmark or speedup claim.
- The cooperative-root/whole-subtree checkpoint passed all 126 release
  `modular::ma_train` tests with zero failures in 0.52 seconds, including all
  ten new range, subtree, root-dispatch and absolute-race tests. This is a
  centralized test result, not an encoder performance measurement.
- The parent also reports the release channel/row oracle passed through both
  official djxl 0.12 and 0.13: two cases with one/four-thread coverage,
  including actual activation of 16 channel/row partition jobs. Quiet mode is
  covered by the executor tests and separate LZ/scheduler oracle cases.
- The subsequent `parallel_execution_reference` rerun passed all five tests
  through both official djxl 0.12 and 0.13, including whole-subtree pixel checks
  with actual execution confirmed by statistics/profile assertions. The
  whole-subtree official oracle is no longer pending. The parent reported
  6.52 seconds for this validation rerun, not whole-pipeline throughput or a
  speedup measurement.

The sections below retain implementation history; their earlier validation
counts are historical checkpoints. These results do not establish full migration
completion, whole-pipeline performance, or close the remaining gaps below.
Those results precede the channel-tree slice below. The parent subsequently ran
all eight focused component tests in a passing 1,450-test debug workspace run.
The new independent pixel oracle then passed 16 comparisons through djxl 0.12
and 0.13. Actual profiles proved 4/8/16 partition jobs for modes 1/2/3, not merely
requested options; see PORT_VERIFICATION.md.

## First Checkpoint

- `modular/encoding/enc_ma.cc::FindBestSplitImpl`, lines 680-717: test the current
  leaf predictor's cost before collecting the other predictors. A terminal
  uncached node now builds only one parent histogram. Existing inherited parent
  histograms are reused. Forced multiplier splits still precede adaptive search.
- `modular/encoding/enc_ma_histograms.h::CollectMaParentHistogram`: collect a
  fixed 256-token histogram in one sample pass, then trim its zero tail. This
  replaces Rust's separate maximum-token and count scans. Nonterminal nodes
  reuse the leaf histogram; other predictors use the existing execution API at
  the source fused-parent gates (>2 predictors, >=8192 records, >=65536 remaining
  predictor-record visits). The fused-parent source implementation is opt-in;
  Rust already enabled parent reuse and now completes that implementation.
- `enc_ma_histograms.h::CompactMaHistograms` and `enc_ma.cc:791`: activate the
  **default lane-preserving** compaction on uncached nodes with <=256 distinct
  records, padded token stride >=32, and at least 25% stride savings. The union
  of active parent symbols defines the remapping. SIMD lane identity and order
  within each lane are unchanged. Scalar uses one lane. Per-predictor zero-tail
  trimming remains active. Raw parents/tokens are retained for child subtraction;
  compacted indices never enter a raw-token cache or the encoded stream.
- `enc_ma_histograms.h::CollectPairedMaHistogram` and `enc_ma.cc:1271`: collect
  uncached adjacent predictors together at >=256 records, with an unpaired final
  predictor when needed. Property values and weights are loaded once per pair.
  Different predictor strides and weighted extra bits remain exact. As in the
  source, compaction takes precedence over pairing. No predictors are dropped.
- `enc_ma.cc:880,974`: cache creation now has the source >=8192-record gate;
  inherited larger-child caches still have the existing >=4096 gate. Populate
  independent property caches through the existing execution API under the
  source property-work gate (>=1024 records and >=65536 record/predictor/property
  work). Previously Rust populated all of them serially before parallel scoring.
  The entire conservative byte lease is reserved before dispatch. Constant
  properties retain their counts/weights but allocate no token histograms.
- Corrected stale `tools/ma_entropy.rs` module documentation: its incremental
  f64/Fenwick scorer is a standalone tool, **not** the default learner backend.
  The tool and its tests are retained.

Changed implementation files: `crates/rs-jxl/src/modular/ma_train.rs`,
`crates/rs-jxl/src/tools/ma_simd.rs`, and documentation in
`crates/rs-jxl/src/tools/ma_entropy.rs`. No training/scheduler/Cargo/entropy/LZ
implementation outside these files was edited.

## Second Checkpoint And PA-03

- Ported the cross-node lifetime of `enc_ma.cc::PropertyWorkspace` into an
  invocation-owned `ScratchPool`. The parent's existing `map_with_state` worker
  loops acquire exclusive leases and return buffers on normal/error/unwind exits.
  No pool lock is held while collecting/scoring and no scratch is indexed by
  thread ID or shared globally. This also permits nested ready-child evaluation
  without aliasing suspended worker state. Retention is capped at one worker set,
  or two sets when ready siblings are enabled, using the existing worker count
  and property count. A serial invocation creates/reuses just one active buffer.
- Ported deferred per-property cache subtraction (`enc_ma.cc:1033` and
  `enc_ma_histograms.h::SubtractProperty/SubtractAndCopy`). `PendingNode` now
  carries the removed sibling's index range. Each property task subtracts its
  exact record counts, weighted counts, token counts and extra bits immediately
  before scoring; the parent split no longer serially subtracts all properties.
  Temporary per-property mutexes provide safe disjoint mutable borrows through
  the existing shared execution callback. There is no cache-wide scoring lock,
  extra histogram copy or scheduler replacement. Range membership stays valid
  when DFS reorders the sibling. A completed decision cannot subtract twice.
- Early terminal nodes release inherited caches without unnecessary subtraction.
  A failed property subtraction invalidates the entire partially updated cache
  after all dispatched tasks return, so it cannot be inherited or retried stale.
  The scalar/serial route uses the same operations and stable ordered reduction.
- Added scratch-created/reused and per-property cache-subtraction profile counts.
  These identify active work, not timing/performance claims.
- PA-03: `training.rs::gather_legacy` calls the parent-owned
  `encode::legacy_sample_count` before invoking the collector. It compares the
  target with the invocation's remaining selected-sample limit, then consumes
  the shared budget, then reserves storage and collects. Removed the old
  post-collection debit. A collector failure aborts without refunding previously
  reserved allowance, consistent with the existing source collection policy.
  Callback tests verify first-stream and cumulative-stream rejection happens
  before collection, and an accepted target consumes shared allowance once.

Second-checkpoint edits are confined to `modular/ma_train.rs`,
`modular/training.rs`, and this audit. The parent-owned `execution.rs`,
`modular/encode.rs`, scheduler and Cargo files were not modified here.

## Third Checkpoint: Gather Dispatch And Column Locality

- `enc_encoding.cc::GatherTreeData<kPredictorPair>` and
  `enc_predict.h::PredictLearnBest`: dispatch once on the exact ordered candidate
  pair `[Weighted, Gradient]`, for both serial and channel/wave gathering.
  The specialized sampled-pixel path uses the existing WP prediction plus
  ClampedGradient directly, avoiding the candidate iterator and predictor switch.
  Other candidate sets retain the generic implementation. Rust already computed
  only requested predictors; this is not a claim of eliminating fourteen
  predictions from the former Rust path. RNG draws, selected-pixel budgets,
  reference properties, causal WP updates and previous-gradient state are
  unchanged. Wrapping i32 residuals match the generic wide-then-narrow operation.
- `enc_ma.cc:322::SplitTreeSamples<..., true>` and
  `enc_ma.cc:2485::TreeSamples::SwapBatch`: collect 128 disjoint swap pairs,
  then apply them column by column to every residual, property and multiplicity
  column. Flush the partial final batch. Mandatory static splits and adaptive
  splits both use this path; quantizer thresholds never move. Production
  preparation passes mutable buffers directly, without a whole-sample clone or
  a second reordered copy. The original index-only path is instantiated only as
  a test oracle, not exposed as an alternative search policy.
- Existing histogram APIs still consume a sequential identity-index buffer;
  arbitrary gather loads are removed by physically reordering the columns, but
  eliminating that now-redundant index load is not claimed. All worker searches
  finish before any column mutation. Deferred sibling cache ranges preserve
  their populations when descendants reorder those columns.
- Added two training tests and four learner tests, and extended the existing
  channel-gather test to compare the specialized pair against forced generic
  gathering. Coverage includes signed extrema/wrapping, exact dispatch gates,
  zero/full/fractional sampling, narrow/empty channels, reference channels,
  swap counts around 128 and 256, unchanged buffers and untouched outer ranges,
  wide bins, cached/uncached serial/parallel full-tree agreement, leaf-limit
  errors, forced static/empty-domain splits and deferred subtraction after
  physical column moves. Existing scratch/cache tests now use independent
  prepared-buffer clones in test code only.

This bounded checkpoint changes only `modular/ma_train.rs`,
`modular/training.rs`, and this audit. This checkpoint was held for parent
validation before beginning the next bounded slice. Parent-owned token-channel extraction,
`modular/encode.rs`, `modular/tokens.rs`, `modular/frame.rs`, registration,
execution and Cargo were not touched. The previously reported 813 core plus
14 acceptance passes predate later agent edits; they do not validate this slice.

## Fourth Checkpoint: Fused Initial Cache Population

- `enc_ma.cc:880,974` reserves a cache before the property dispatch, then each
  property worker fills and scores its entry. Rust now follows that lifetime:
  `HistogramCache::reserve` takes the complete conservative byte lease and
  creates empty property slots, without scanning samples or allocating bin/token
  arrays. `PropertyHistogram::collect` runs immediately before scoring in the
  same existing `map_with_state` callback. There is no initial-population
  dispatch or all-properties fill/scoring barrier in the production learner.
- The checked byte calculation, 8192-record creation gate, property-work gate,
  constant-property token omission, sample weights, token strides, extra bits,
  scratch resets and ordered split reduction are unchanged. A denied reservation
  falls back to the complete uncached sweep; it does not narrow the search.
  The eager population helper remains under `cfg(test)` as an oracle and for
  constructing inherited-cache fixtures.
- Cache ownership is local until the complete property sweep succeeds.
  Successful initial fills and inherited deferred subtractions are published
  together with the decision. A task error or unwind discards the whole partial
  cache after borrowed workers finish, releases its byte lease and returns
  private scratch. Retrying cannot inherit partially filled/subtracted state.
  No global or thread-indexed state, scheduler changes or storage rewrite.
- Four new tests cover reservation-before-population and exact byte accounting;
  zero/one-byte-short denied budgets; same-worker fill/score ordering in serial,
  parallel and null-pool modes; eager/uncached decision parity; constant columns;
  larger-child deferred subtraction; injected failures after fill/score; unwind
  cleanup; and successful retries with retained scratch. Fault/order hooks are
  invocation-local and compiled only in tests.

Fourth-checkpoint edits are confined to `modular/ma_train.rs` and this audit.
Only that Rust file was formatted, using edition 2024 and `skip_children=true`.
The parent reports 869 core tests and eight real-image full-scheduler re9 cases
passing two official decoders before this slice. The reported aggregate
MA-plus-entropy median speedups (1.119x PDF, 1.146x photo) are not attribution to
this change and did not validate it. These changes awaited their own central
build/test/oracle run at handoff; no broader MA work was included in that checkpoint.
Subsequently, the parent reported 903 core, 14 acceptance and 35 CLI tests plus
strict Clippy passing with fusion included. The affected official cases
`native_scheduler_retained`, `frame_area_streaming`, `native_streaming_boundary`,
`parallel_source_gather` and `source_tiers` also passed both decoders after
fusion/budget changes. Those results predate the fifth checkpoint below.

## Fifth Checkpoint: No-Edge Gather Traversal

- `enc_encoding.cc::GatherTreeData`, lines 194-240, splits each eligible row
  into two generic left pixels, a no-edge interior, and two generic right
  pixels. `training.rs::visit_source_row` now implements that segmentation
  using safe current/top/toptop row slices and all seven source neighbor loads.
  The exact gate is `y > 1 && width > 8 && previous_channels == 0`; configured
  reference slots disable it even when no compatible earlier channel exists.
  No unchecked indexing, additional padding or approximate predictor state.
- Default serial gathering now uses the same owned kernel as channel/wave
  gathering, rather than leaving serial work on the general encoder visitor.
  The parent-owned `encode::visit` interface/implementation is untouched and
  remains the independent generic traversal oracle in tests. A pre-collection
  hook preserves serial local/shared budget charges before residual-vector and
  property-copy allocations; callback failures retain the same debit policy.
- One shared pixel callback advances RNG, WP prediction/error state and the
  previous-gradient property across all three row segments and discarded
  samples. WP use, exact candidate order, narrowing, references, channel/stream
  IDs, selected multiplicities and population statistics are unchanged.
  Channel dimensions and compatible reference buffers are checked before row
  traversal, in addition to the existing learning-domain validation.
- Seven new tests cover exact source gates and all neighbor fields; empty,
  width-1/2/8/9 and first-two-row borders; deep signed values through i32 extrema;
  all predictor candidates; all five WP presets and WP-property-only use;
  zero/full/fractional RNG selection through row/segment transitions; missing
  and dimension/shift-mismatched references; cumulative local/shared budgets
  and callback failures; parallel/null-pool channel/wave population parity;
  and invalid-plane rejection before sampling/collection. Existing pair/channel
  parity tests now use the independent generic visitor rather than the shared
  optimized kernel as their reference.

Fifth-checkpoint edits are confined to `modular/training.rs` and this audit.
Only `training.rs` received scoped edition-2024 formatting. No compact-column
storage changes, scheduler/encoder/token/frame edits, Cargo changes, builds,
tests, Clippy, benchmarks or decoder commands were run by this agent. This
checkpoint required its own central validation at handoff; no new performance attribution
or complete-migration claim is made.
Subsequently, the parent reported 938 core passes (940 total, two ignored),
including the no-edge files present at compile time. The no-edge implementation
is frozen; it is not changed by the sixth checkpoint below.

## Sixth Checkpoint: Packed Quantized Property Columns

- `enc_ma.h::TreeSamples::PropertyData`, lines 81-96 and storage at 185-188:
  dynamic quantized columns now use `Vec<u8>` when the complete quantizer has
  at most 256 values. Static property IDs 0/1 always use `Vec<u32>`, regardless
  of their position in the configured property list. Wider dynamic quantizers
  have an explicit `Vec<u32>` fallback, chosen from threshold cardinality, not
  the observed sample maximum. Public option validation is unchanged; the
  internal legacy preparation helper can retain quantizers above 256 values
  without truncation. Quantization still uses strict `value > threshold`.
- The new local `ma_columns.rs` module is registered with `#[path]` only inside
  `ma_train.rs`. One dispatch at the start of each property task specializes
  initial cache fill, deferred subtraction and scoring for a typed byte or
  word slice. Uncached marginals, paired/unpaired token collection and compact
  token mapping all read that typed slice. No per-sample enum access, temporary
  wide vector, allocation or reference conversion in those search loops.
  Sample preparation retains its existing row-wise key/dedup insertion and
  selects the destination column when pushing each newly distinct record.
- `SplitTreeSamples` borrows the selected column separately from the others,
  dispatches its scan once and keeps the source 128-pair swap batches. Other
  columns dispatch once per batch, outside their swap loops. All property,
  residual and count buffers stay allocated and aligned; identity indices,
  sibling-range membership and deferred-cache lifetimes remain unchanged.
- Histogram arrays and multiplicities remain u32. The conservative cache
  lease calculation, publication/partial-error rollback, inherited-cache
  gates, scratch resets, scoring arithmetic and ordered reduction are unchanged.
  There are no SIMD API/tool changes, counts or deduplication algorithm changes,
  new caps, search shortcuts, or parent-file edits.
- Five new tests cover storage widths and 255/256/257-value boundaries; static
  IDs outside a prefix; strict quantizer boundaries, clipping and duplicate
  weights above u16; wider legacy quantizers; packed/all-wide cached, paired,
  compact and uncached costs; repeated subtraction with reversed indices;
  serial/parallel complete searches and fused/inherited cache lifetimes.
  The existing 128-pair boundary test now covers both packed and wide split
  columns, untouched outer ranges and stable allocations. Existing mixed-column
  cache-failure, forced-static-split and deferred-sibling tests remain active.

Sixth-checkpoint edits are confined to `modular/ma_train.rs`, new local
`modular/ma_columns.rs`, and this audit. Both Rust files received scoped
edition-2024 formatting with `skip_children=true`. No Cargo/build/test/Clippy,
benchmark or decoder command was run by this agent; this checkpoint required
central validation separately from the reported no-edge result at handoff.
Subsequently, the parent reported 973 core workspace passes, strict Clippy,
and 47 release official-decoder suites passing both decoders with packing and
the separately owned FMA specialization included. Those results predate the
seventh checkpoint below.

## Seventh Checkpoint: Saturated Counts And Bounded Deduplication

### Frozen Source Policy

- `enc_ma.h:184` stores occurrence counts as `uint16_t`; a saturated count is
  a complete record, not a limit on total multiplicity. `enc_ma.cc:2007` enables
  `FastDedup` by default when its environment variable is unset and lossless
  speedups are active.
- `PrepareForSamples` (`enc_ma.cc:2316`) reserves existing records plus the
  expected additions, using 1.5x vector growth when necessary. Its table has
  `2^ceil_log2(floor(3*N/2))` slots. There is no source one-million-slot ceiling.
  `InitTable` (`2073`) keeps an equal-sized table; a changed size rebuilds into
  empty slots and registers unsaturated rows in record order without merging.
- `Hash1` (`2341`) uses wrapping u64 multiply/add with `0x1e35a7bd`, visiting
  token/extra-bit-count pairs then static and dynamic properties. `Hash2`
  (`2356`) uses multiply/XOR with `0x1e35a7bd1e35a7bd`, properties first, then
  residual pairs. Both select `(hash >> 16) & (table_size - 1)`.
- `AddToTable` (`2308`) tries only the first and second slots. It never evicts,
  probes further or discards an unregistered sample. Weighted merging (`2133`)
  fills matching records in probe order, removes a slot when its count reaches
  65535, and retains the remainder as a new record. Token, extra-bit COUNT and
  quantized-property equality define identity, not residual payload bits.

### Implemented Scope

- Prepared `TreeSamples::counts` is now `Vec<u16>`. Every histogram/cache load
  explicitly widens to u32; histogram arrays, f32 scoring, cache byte leases and
  u64 extra-bit accumulators remain unchanged. Batched swaps move u16 counts
  alongside all existing columns. Record-count-based cache/search gates now see
  source-style saturated/collision records, so their triggering can differ from
  the former global dedup representation without changing the weighted objective.
- New local `ma_samples.rs` implements the above two-probe table over prepared
  column indices. `TreeSamples::new` no longer retains a global HashMap and a
  heap-allocated key per unique record. One reusable quantization key serves
  the invocation; table comparisons use the existing residual/property columns.
  The default source hash order also handles Rust's interleaved static-property
  configurations without changing their quantization or identity.
- External u32 weights are split into exact chunks of at most 65535 before
  source weighted merging. This preserves full multiplicity, including when
  both probe slots already contain matching partial records. No clamp, dropped
  collision, new weight ceiling or new search approximation. The learner's
  existing i32::MAX total-weight limit is still checked before preparation;
  the low-level splitter itself preserves even u32::MAX without unit expansion.
- The preparation iterator is cloned to compute a checked split-record upper
  bound. Columns and the table use checked sizing and fallible reservations.
  An empty population creates no count/table allocation. This preparation API
  receives the complete borrowed input, so it performs one source-style prepare
  for that input; the helper also implements and tests subsequent resize/rehash
  behavior. The invocation-local table drops before learning begins.
- Six helper tests cover capacity rounding, zero/overflow limits, growth,
  source hash goldens, exact identity, static ordering, 65534/65535/65536 and
  full-u32 splitting, repeated saturation across growth/shrink rebuilds,
  unchanged-size retention, collision misses without eviction/extra probes,
  and filling two matching records in order. Three integration tests compare
  full-range weighted key populations and parent/property histograms against
  raw inputs, retain the existing total-weight ceiling, and verify cached/fresh
  costs after physical partitioning and subtraction with saturated counts.
  Existing count assertions now reflect split records or compare weighted mass.

Seventh-checkpoint edits are confined to `modular/ma_train.rs`, new local
`modular/ma_samples.rs` registered with `#[path]`, and this audit. No edits to
`ma_columns.rs`, scorer/FMA tools, gather interfaces, registration, parent files,
Cargo or execution. Only the two owned Rust files receive scoped formatting;
no compile/build/test/Clippy/benchmark/decoder commands are run by this agent.
The parent subsequently reports 1,051 normal-release core tests and the later
1,078-core debug workspace checkpoint passing, plus strict Clippy, Rust 1.89
compilation and 80 official source-tier pixel checks with this storage change.
These checks do not establish complete MA source-tool or performance parity.

## Eighth Checkpoint: Optional Verified Predictor Reuse

- `enc_ma.cc:433-436,892-921,1180-1192`: `TrainOptions::predictor_dedup`
  is a typed opt-in, default `false`, and activates only above two predictors.
  Each node recomputes the earliest representative after checking exactly equal
  f32 penalties, weighted parent counts including zero padding, extra-bit totals,
  and every live residual token/extra-bit-count pair. A matching marginal alone
  never proves reuse. Representatives are canonical, node-local, immutable
  during property dispatch, and never inherited through scratch or child jobs.
- Active representatives alone collect and score property token histograms.
  Adjacent-predictor collection requires both members to remain representatives,
  including when an even member was skipped; an active odd predictor cannot
  consume a stale paired buffer. Raw parents still prove equivalence before
  optional lane-preserving compaction. The strict threshold sweep, source
  penalties, property order and stable predictor tie-breaking remain unchanged.
- `enc_ma_histograms.h::MaHistogramCache::CopyPredictor`: duplicate cache
  entries receive the representative's completed counts and extra bits, so
  descendants can invalidate the proof when their current leaf changes penalties.
  Deferred subtraction updates representatives first, then copies their live
  population; removed sibling records need not be equivalent. Rust's existing
  per-predictor strides are retained without reallocation or changing the cache
  lease. Short destinations require zero truncated source tails; wide destinations
  have their tails cleared. Copy errors use the existing all-properties cache
  invalidation/lease release path. Constant-property caches remain token-free.
- Added the `equivalent_predictors` profile count and five focused tests:
  `predictor_dedup_requires_exact_node_local_proofs`,
  `predictor_dedup_matches_all_costs_across_pair_and_compaction_boundaries`,
  `predictor_dedup_copies_fixed_strides_and_rejects_nonzero_truncation`,
  `predictor_dedup_reproves_inherited_cache_after_subtraction_and_leaf_change`,
  and `predictor_dedup_preserves_complete_search_serial_parallel_and_null_pool`.
  They cover 63/64 and 255/256/257/8192 record boundaries, saturated weights,
  packed/static columns, cached/uncached/compact exact costs, all split winners,
  unequal retained strides, leaf penalty invalidation, and unchanged allocation
  identities. An injected nonzero copy tail checks whole-cache rollback and
  retry with retained scratch. Existing `initial_cache_errors_discard_partial_work_and_allow_retry`
  and `initial_cache_unwind_releases_budget_and_private_scratch` also exercise
  enabled dedup and populated duplicate entries.
- At this checkpoint, `JXL_MA_PROPERTY_DEDUP` was not yet ported; the ninth
  checkpoint below implements it. Source `enc_ma.cc:981-1117`
  first skips inherited masks, records constants, and only in serial nodes with
  >=64 records uses a used-bin/count FNV signature as a filter. It then requires
  matching static/WP classes and exact per-record property values before skipping
  an earlier-equivalent property and invalidating its cache. Masks propagate to
  both children (`1644,1653`). This requires pending-node mask propagation and
  per-property cache validity beyond the predictor-local proof; no placeholder
  option, marginal-only shortcut or approximate property elimination was added.

This slice edits only `modular/ma_train.rs` and this audit. Local sample/column
helpers, scorer tools, `training.rs`, `encode.rs`, execution and Cargo are untouched.
The parent reports `ModularOptions::source_predictor_dedup` added and forwarded
to `encode::train_options(...).predictor_dedup`; direct `TrainOptions` callers can
already opt in. Existing default-constructed options remain off. There is no
learner environment lookup and no implicit enabling for encoder tiers.
Only scoped edition-2024 rustfmt/check was run locally; compilation, tests,
strict Clippy and affected release pixel oracles were delegated to the parent.
The parent reports 1,114 core passes including predictor-dedup tests during this
slice. Later central validation supersedes that mid-edit result; see the current
checkpoint above and the predictor/property pixel checks recorded below.

## Ninth Checkpoint: Optional Verified Property Reuse

- `enc_ma.cc:514-516,981-1119` and `enc_ma.h::SameProperty`:
  `TrainOptions::property_dedup` is a typed opt-in, default `false`. A serial
  sweep with >=64 distinct records computes the source wrapping-u64 FNV signature
  over nonzero `(bin, record count)` pairs. Weighted counts and quantizer tails
  do not enter that signature. Matching signatures only filter candidates:
  static/nonstatic and WP/non-WP classes must match, then every live quantized
  value is compared. No equal-marginal shortcut, bin permutation, shortlist,
  predictor/threshold restriction or approximate search is introduced. Earlier
  equivalent properties have already been scored in stable configured order.
- `enc_ma.cc:972-977`: cross-property reuse is disabled for actual parallel
  sweeps (>=1024 records, >=65536 predictor/property/record work, >1 worker).
  Null-pool scopes and one-worker pools use the serial proof path. Parallel
  tasks still detect constants; inherited masks are honored in both paths.
  Static classification uses property IDs, not a static-prefix assumption;
  comparisons dispatch once to packed/wide slices, including mixed widths and
  the existing legacy wide-column fallback.
- `enc_ma.cc:553,647,1644,1653`: each `PendingNode` owns a u8 redundant-property
  mask, copied/moved to both children before ready-child evaluation. A constant
  or verified equal partition remains redundant in every descendant subset,
  even after physical 128-pair column swaps. Inherited entries return before
  any column scan, cache access or sibling subtraction. Seen-property lists
  remain node-local and serial; no global state or shared mutable worker masks.
- `enc_ma_histograms.h::SubtractProperty/Invalidate`: opt-in tasks collect or
  subtract marginals first, prove redundancy, then update token histograms only
  for active properties. Duplicate cache entries become invalid while retaining
  their original allocation and conservative lease; stale parent token counts
  cannot be passed as live child data. Inherited constants may likewise retain
  unused token storage behind their permanent mask. Unmasked invalid entries
  rebuild from the live node. Default-off subtraction remains a fused scan;
  predictor representatives, copied cache strides, paired collection, source
  scores and strict ordered reduction are unchanged. The cache header's validity
  flag is included automatically in checked `size_of` lease accounting.
- New masks and the working cache publish only after all property tasks succeed.
  Error/unwind paths drop partial caches and release leases while leaving the
  caller's inherited proof mask unchanged. No fill/scoring barrier is added.
  Profile counters identify constant, duplicate and inherited properties, without
  asserting any performance attribution.
- Five focused tests (`property_dedup_` filter) cover source signature goldens;
  equal marginals with unequal per-record partitions; static/WP classifications;
  unused quantizer tails and byte/word comparison; 63/64 and 1023/1024 boundaries;
  single-worker, parallel and null-pool gates; repeated deferred subtraction;
  child-only equivalence and skipped stale tokens; retained allocation identities;
  unchanged proof masks on errors/unwinds; complete search with both dedup
  options, physical/indexed partitioning, cache-on/off and exact leaf-limit errors.
  Existing predictor-dedup tests remain present and unchanged by this slice.

Only `modular/ma_train.rs` and this audit are edited in this checkpoint. The parent
reports `ModularOptions::source_property_dedup` and its forwarding to
`TrainOptions::property_dedup` added separately; encoder/training/frame/scheduler
files and local column/sample helpers are untouched here. Scoped rustfmt/check
was the only local validation. The tests and parent-wired pixel oracle cases
subsequently passed central validation as recorded below; no
Cargo/build/test/Clippy/decoder command was run here.

## Tenth Checkpoint: Optional Histogram Kernels

- Added default-off `TrainOptions` fields `histogram_shards`,
  `padded_histogram`, `dense_histogram`, `property_bins`, `sparse_entropy`, and
  `paired_cache`. The public `ma_train::HistogramOptions` aggregate exposes
  `shards`, `padded`, `dense`, `property_bins`, `sparse`, and `paired`, with
  `apply(&self, &mut TrainOptions)` forwarding only these tools. It derives
  Clone/Copy/Debug/Default/Eq/PartialEq. The parent reports forwarding through
  `ModularOptions::source_ma_histograms`; no encoder file is edited here.
- `enc_ma.cc`/`enc_ma_histograms.h` four-shard collection keeps four independent
  u32 count/u64 extra-bit streams, merges in source order, and clears retained
  scratch completely. Source dispatch priority is preserved: packed property
  bins or token compaction, cached histograms, adjacent active predictor pairs,
  then four shards for eligible unpaired predictors at >=4096 distinct records
  and >=4*property_bins*common_stride work. Initial cache population also uses
  this dispatch inside its existing fill-and-score worker task.
- Padding rounds the common token stride to native SIMD lanes, then adds one
  lane block only for power-of-two strides in [32,256). The exact conservative
  cache lease uses that stride and descendants retain it. DenseHistogram has
  Disabled/Small/Extended variants for source unset/1/2, with sequential-symbol
  compaction through 256/4096 records, still requiring the source stride savings
  and no cache. Default lane-preserving compaction is unchanged.
- PropertyBins has Disabled/SmallOrCompacted/All variants for source unset/1/2.
  Only uncached nodes with strictly less than half their bins occupied pack
  property rows. Mode 1 additionally requires <256 records or token compaction.
  Histogram row mapping is separate from original quantized threshold order;
  weighted marginals, split positions, costs and tie order are not narrowed.
- Sparse entropy builds increasing active parent-block indices once per
  predictor and moves/scores only those blocks, retaining SIMD lane identity
  and partial-tail handling. Scalar, SSE2, AVX2 and AVX2/FMA implementations
  use the existing source f32 arithmetic. The safe wrapper validates indices,
  ordering, alignment and equal slice lengths before native mutation.
- Paired inherited-cache subtraction shares property/weight loads for adjacent
  active predictor pairs at >=256 live records. Predictor representatives,
  unequal fixed root strides, property masks, deferred sibling membership and
  partial-error rollback are preserved. C++ subtracts and copies into scoring
  scratch; Rust scores its live immutable cache directly, avoiding that copy.
- Added five `optional_` helper/learner tests, the aggregate forwarding test
  `histogram_options_forward_only_explicit_tools`, and native-profile test
  `sparse_blocks_match_dense_native_profiles_and_reject_invalid_addresses`.
  They cover exact costs/multiplicity, dispatch boundaries, dense/property-bin
  gates, padded byte leases, initially sharded cache fill, paired subtraction,
  reused scratch and invalid native block indices.

Changes are limited to `ma_train.rs`, new local `ma_histograms.rs`,
`tools/ma_simd.rs` and this audit. The parent initially reported cargo check
succeeded with the histogram types. The prepared handoff and histogram tests
have since been included in the central workspace checks; current forwarding
and oracle status is recorded above. No Cargo/build/test/oracle command is run locally.

## Eleventh Checkpoint: Direct Prepared-Column Handoff

- Added crate-private `ma_train::train_prepared(samples:
  training::gather_samples::PreparedSamples, options: &TrainOptions) -> Result<Tree>`
  in local `ma_prepared.rs`. Source gather passes `master.finish()` after
  `policy.apply_to_train`; legacy callers keep public `train(&[Sample], ...)`.
  Hume completed that production caller switch and removed the compatibility
  adapter and its raw static-ID column; row conversion remains test-only.
- The entry requires exact predictor order and matching supplied quantization,
  validates every column length/type/bin, nonzero u16 multiplicities, exact
  total <=i32::MAX, and HybridUint(4,1,2) token/nbits shape through full u32.
  Dynamic property Vec<u8>, static Vec<u32>, count Vec<u16> and threshold
  buffers transfer without copying. Residual [u8;2] columns undergo an explicit
  fallible per-column conversion to ResidualToken; no unsafe layout cast, raw
  Sample construction, second quantization or second deduplication is used.
  Collision duplicates and saturated records retain their exact order/weights.
- Both entries share constant-suffix assembly and variable-domain leaf-budget
  reservation. The prepared caller must already exclude proven constant
  channels; nonempty all-constant populations are rejected. Supplied static
  bins must intersect the variable domain. A bin spanning its edge, or an
  absent static split property, cannot prove individual raw IDs, so those
  remain gather-domain preconditions rather than fabricated sample evidence.
  Every multiplier constraint, even multiplier 1, is rejected by the prepared
  entry. The raw API's multiplier/constant checks are unchanged.
- Complete-search errors, cache/scratch lifetime, physical swaps, empty seed
  and source predictor/threshold order all use the same learner. The obsolete
  SampleBatchCollector and its allocating quantized-key wrapper are now
  test-only; their older regression tests remain available, not production.
- Five `prepared_` tests cover pointer-preserving ownership transfer,
  interleaved static columns, collision/saturation histograms, raw/prepared
  tree agreement, empty/all-constant/partial-suffix trees, required leaf-budget
  errors, schema/token/shape/count failures, static-boundary uncertainty,
  rejected multiplier constraints and i32::MAX exact total versus overflow.

This handoff edits only `ma_train.rs`, new local `ma_prepared.rs` and this audit.
The gather contract remains owned by Hume in `MA_GATHER_PORT_STATUS.md`.
The prior 1194-core snapshot predated this entry. Parent subsequently reported
1211 core passes including prepared-consumer tests, followed by wired production
gather validation. The current 1376-core workspace/Clippy/MSRV checkpoint and
official pixel results are recorded above. Scoped rustfmt/check remains the
only local Rust validation; prepared/histogram tests and caller wiring are no
longer awaiting their first central run.

## Twelfth Checkpoint: Source Scoring And Search Experiments

Two public grouped options are reexported from `modular::ma_train`. Both derive
Clone/Copy/Debug/Default/Eq/PartialEq, keep every experiment off by default, and
provide `apply(&self, &mut TrainOptions)`. HistogramOptions remains unchanged.

| Group/field | Frozen source mapping |
| --- | --- |
| ScoringOptions.count_log | JXL_MA_COUNT_LOG_TABLE=1 |
| ScoringOptions.fast_entropy | TreeSamples::FastEntropy |
| ScoringOptions.clipped | ClippedEntropyMode::Disabled/Always/AtLeast2048/AtLeast16384/Sliding/SlidingAtLeast2048, source unset/1/2/3/4/5 |
| ScoringOptions.description | DescriptionCost::Disabled/Quarter/Half/Full, source unset/1/2/3 |
| SearchOptions.statistical_race | JXL_MA_STATISTICAL_RACE=1 |
| SearchOptions.quantile_splits | QuantileSplits::Disabled/SixtyFour/ThirtyTwo, source unset/1/2 |
| SearchOptions.refine_quantiles | JXL_MA_REFINE_QUANTILES=1 |

The corresponding TrainOptions fields are `scoring: ScoringOptions` and
`search: SearchOptions`. Parent completed encoder forwarding through
`ModularOptions::source_ma_scoring` and `ModularOptions::source_ma_search`,
alongside `source_ma_histograms`; all groups retain their default-off options.
No parent encoder/training/scheduler/Cargo file is edited here.

- `enc_ma.cc:707,789,1342-1445` scoring now follows source priority: eligible
  clipped state, FastEntropy/Shannon, count-log when both post-move totals are
  <=65536 and their sum >256, then sparse/default. Clipped activation gates use
  distinct records, not weighted population. FastEntropy disables compaction,
  sparse scoring and clipped selection. Count-log and clipped modes retain the
  source default leaf-baseline estimate; Shannon changes that baseline too.
- New local `ma_scoring.rs` owns mode dispatch, sequential f64 Shannon parent
  sums and retained per-worker incremental state. `SourceClippedEntropy` ports
  the source count-frequency Fenwick and sliding-cursor implementations without
  another count-column copy. Reset clears sums, mode, cursor, tail and rare
  buckets; four/five update the tail when changed frequencies cross the cursor.
  Source strict rare thresholds `(total-1)/4096` and f32 final costs remain.
  Existing standalone ClippedEntropy/LogTable public tools are not deleted.
- Count-log has a shared bounded immutable table and a read-only-delta
  SSE2/AVX2 kernel with scalar fallback. It preserves the source lane reduction,
  including partial tails, without clearing a retained property cache or
  rescanning validated totals. Native gather addresses are independently
  bounded. Source description penalties use the selected FastLog2f profile
  and source f32 multiplication order, added to both leaf baseline and split
  sides before the existing predictor penalty/tie selection.
- New local `ma_search.rs` ports `enc_ma_racing.h` and the source dispatch.
  Quantile splits need >=512 records, a dynamic property, and a used-bin span
  greater than the selected 64/32 count. Weighted quantum selection retains
  both edge candidates and large jumps. Skipped thresholds still move all
  integer counts and extra-bit mass, using a native integer-only kernel.
  The mask never changes sample membership or physical split positions.
- Refinement forces 32 coarse quantiles, preserves every coarse point, then
  expands the two best separated finite-cost valleys with stable ties. It
  saves uncached histograms once and reuses them in pass two; existing immutable
  cache rows need no copy. Source's `cells*predictors*4 <= 8 MiB` gate is exact;
  failed eligibility returns to complete threshold search, not coarse-only
  approximation. FastEntropy or any enabled clipped mode disables both
  quantile/refinement globally, even below a clipped node's activation gate.
- Property racing requires >=262144 records, more than four dynamic properties,
  no valid inherited histogram, <=256 bins for each dynamic property and source
  32 MiB proposal-histogram sizing. Four deterministic panels extend from 64 to
  256 weighted draws with source wrapping-u64 hashes including absolute node
  range. Proposal scoring, effective-sample-size >=32 checks, +1 paired costs,
  0.02 evidence margin and strict 0.01 separation match source. Stable order,
  the first four alternatives, static properties and WP-specific property are
  retained. Unchanged first-round survivors stop proposals immediately.
- Race only chooses which properties receive the node's complete sweep; it
  never fabricates a split or sample subset for that sweep. Its mask is not
  inherited as a property-dedup proof. Pruned cache entries become invalid;
  descendants with inherited valid entries skip racing and rebuild invalid
  properties from their live population. Existing transactional publication,
  lease release, deferred subtraction and required leaf-budget errors remain.
  Interleaved static IDs are classified by ID, not a static-prefix assumption.
- Thirteen focused tests were added: four `scoring_` helper tests, two learner
  scoring tests, `source_clipped_`, `count_log_readonly_`, and five `search_`
  tests. Coverage includes option forwarding; source activation/priority and
  256/65536 boundaries; exact Shannon updates; clipping/cursor resets; f32
  description order; weighted cached/uncached costs and leaf-limit failures;
  mask/8-MiB boundaries; separated refinement valleys; race hash goldens,
  evidence/reliability/WP protection, rounds, serial/parallel parity, and stale
  cache rebuilding; refinement's reused histograms and exact scored costs.

Parent's earlier 1262-core-total run (11 ignored) passed with scoring additions.
The later 1376-core workspace/Clippy/MSRV checkpoint and focused official pixel
checks now supersede that partial-snapshot status. Actual race execution and
all three quantile-32/quantile-64/refinement profiles have passed official
coverage in the latest 11-suite central release run, as detailed above.
The reported count-log test scope error is fixed by placing the test inside its
existing tests module, preserving private oracle visibility. Only scoped
rustfmt/check and diff-check are run locally, never Cargo/tests/oracles. No ARM
specialization is added; existing non-x86 scalar fallback remains supported.

## Thirteenth Checkpoint: Optional Channel/Row Model Partitioning

- `enc_ma.cc:1860-1966::FindBestSplit` is now ported as local `ma_channel.rs`.
  Public `ma_train::ChannelTreeMode` has Disabled (default), Channels (source 1),
  ChannelsAndRows (2), and ChannelsAndRowsTwice (3). `TrainOptions::channel_tree`
  is the resolved mode, with no separate eligibility boolean or environment lookup.
  Parent owns `ModularOptions::source_channel_tree`, final-only metadata/probe/tier
  resolution and forwarding. Ineligible routes must resolve to Disabled before
  training; JPEG does not infer eligibility from integer auxiliary planes.
- Internal gates exactly retain no multipliers, >=8192 stored records, first
  configured property 0, <=32 channel quantizer bins, and >1 occupied bin.
  Channel partitions balance occupied bins, not sample weights or population.
  The split is an unconditional model decision using the existing unquantized
  threshold and source greater-first traversal, not a scored channel proposal.
- Modes 2/3 additionally permit one/two levels of property-2 (row/y) partitions.
  Each requires >=16384 records and the first cumulative-record median with
  >=4096 records on both sides, before the final quantizer bin. Weights never
  determine these gates or cuts. Missing property 2 skips spatial subdivision;
  one occupied channel disables the whole wrapper, including spatial work.
- Physical 128-pair swaps partition the original columns. A range entry into the
  existing learner retains absolute identity indices, original quantizers and
  the already-scaled split threshold; it does not regather, requantize, dedup or
  copy sample columns. Absolute range seeds remain valid for statistical race.
  Every partition starts with predictor 0 and owns a separate histogram-cache
  lifetime; invocation-private scratch can be reused safely between serial jobs.
- Source-order grafting relocates child indices and rebuilds final contexts via
  the existing Tree constructor. Skeleton and merged result both respect the
  existing leaf ceiling, including constant-suffix reservations. Partition
  searches retain that ceiling, not equal per-partition quotas; a required
  skeleton or combined learned tree that cannot fit returns a resource error.
- Independent partition roots were explicitly SERIAL in checkpoint thirteen;
  checkpoint fourteen below replaces that limitation.
  Existing property/ready-child parallelism inside each learner remains usable.
  `ma.channel_trees` trace reports channels/jobs/samples/row_depth and
  `concurrent_roots=1`; this is activation evidence, not a performance claim.
  Bounded cooperative partition roots were still unported then. JPEG cooperative
  forwarding was added separately; see JPEG_COOPERATIVE_MA.md. Existing
  stream/property-1 tree groups are not substitutes
  for this channel/property-0 and row/property-2 model algorithm.
- Eight `channel_tree_` tests cover 8191/8192 and 32/33-bin gates, missing or
  reordered properties, multiplier exclusion, 16383/16384 and 4095/4096 spatial
  boundaries, weighted-versus-record medians, depth limits, sparse channel IDs,
  saturated multiplicities, aligned column/buffer/quantizer preservation, graft
  reachability, unequal leaf needs and combined-budget errors, absolute race
  ranges, cached/uncached property-parallel agreement, scratch reuse after an
  injected cache failure, and constant-suffix reservations.

Changed files are only `ma_train.rs`, new local `ma_channel.rs`, and this audit.
No borrowed-view storage refactor, scorer/SIMD changes, parent-file edits, ARM
work or Cargo/build/test/Clippy/benchmark commands are included. Scoped rustfmt
and whitespace checks are local validation only; the coherent first slice is
frozen for parent central tests. Chandra owns the separate channel-tree pixel
oracle file; those results are not inferred from earlier source-suite passes.

## Fourteenth Checkpoint: Cooperative Roots And Whole Subtrees

Source: frozen `enc_ma.cc::FindBestSplit` channel job arena and
`FindBestSplitImpl` subtree queue, `enc_lossless_workers.h` scratch/task limits.
Parent owns `execution`/the cooperative arena and public encoder forwarding.

- `ma_train::ParallelOptions` is public, `Clone + Copy + Debug + Default`, with
  `subtrees`, `balanced`, `small_subtrees`, `workers: WorkerLimits`, and `quiet`.
  `apply` assigns `TrainOptions.parallel_tools`. The first three flags map to
  `JXL_MA_SUBTREE_PARALLEL`, `JXL_MA_SUBTREE_BALANCED`, and
  `JXL_MA_SMALL_SUBTREES`; all default false. Worker limits retain the source
  task/scratch defaults; quiet maps to the existing arena setting. Parent reports
  `ModularOptions.source_ma_parallel` forwarding and validation added.
- Channel/row roots now use bounded `execution::cooperative` on the installed
  pool. Each task has private scratch/cache state and can issue nested property
  work into the same arena. Existing `parallel=false` and null/serial scopes
  retain sequential execution. There is no new pool or global executor.
- New `ma_ranges.rs` makes sorted disjoint `split_at_mut` loans of every packed
  or wide property column, residual column, and count column. Quantizers remain
  shared and immutable. Storage is statically parameterized, not erased behind
  per-record conversions. Only column descriptors and local identity indices
  are allocated; no full sample dataset is cloned. Nested loans retain the
  absolute base used in statistical-race hashes. The old absolute-range race
  test's expected ranges are unchanged.
- Whole-subtree eligibility uses stored records: >=65536, or >=8192 with the
  small switch. No node is detached until this learner actually dispatches
  parent/property work and observes more than one worker. The initialization
  marker resets per learner even when its scratch pool is reused.
- Ordinary grain is 65536; balanced grain is
  `clamp(root_records / (4 * worker_slots), 1024, 65536)`. At node pop, eligible
  nodes are removed from the parent's traversal before scoring, preserving the
  current leaf/predictor/multiplier, static range and redundant-property proof
  mask. Parent histograms and caches that can reference siblings are discarded.
  Parent reuse is suppressed when both children will be deferred; ready-child
  scoring does not pre-evaluate a child scheduled for detachment.
- Detached jobs use stable descending record-count order, source-order grafts,
  and at most `min(worker_slots, workers.concurrency(jobs, 32 MiB))` tasks. Each
  entire subtree runs with the null runner, a private cache budget and its own
  scratch; it neither spawns more subtrees nor rebuilds a multiplier skeleton.
  Local ceilings and combined graft ceilings remain enforced; jobs are not
  assigned equal leaf quotas and failed work joins before loans are released.
- Traces: `ma.channel_trees` reports `root_limit` instead of the old fixed
  `concurrent_roots=1`. `ma.subtrees` reports `jobs`, `worker_tasks`,
  `max_samples`, `balanced`, `cache_budget_bytes`, then `added_nodes` after a
  successful graft. `ready_nodes` alone is not whole-subtree activation proof.

Focused tests: `ma_ranges_`, `ma_roots_cooperative_`, `ma_subtrees_`,
`channel_tree_cooperative_`, and
`race_borrowed_columns_preserve_absolute_hashes_through_nested_ranges`.
They cover no-copy column/gap preservation, invalid range rejection, nested
absolute bases, cooperative bounds/child work/order/null scopes/error joins,
65535/65536 and 8191/8192 gates, record-versus-weight gating, grain boundaries,
stable LPT ties, reset of worker initialization, whole subtrees that grow beyond
their seed leaves, cached/uncached agreement, inherited masks and leaf state,
static range propagation, combined budget errors and channel row-depth parity.

Owned changes: `ma_train.rs`, `ma_channel.rs`, `ma_columns.rs`,
`ma_histograms.rs`, `ma_search.rs`, new local `ma_ranges.rs`/`ma_parallel.rs`,
and this audit. No scorer, gather, parent execution, public encoder, Cargo or
C++ file was edited. Local validation is scoped formatting/source review only;
the parent now reports all 126 release MA learner tests passing, including the
ten tests added here, plus the channel/row official pixel oracle passing through
djxl 0.12 and 0.13 with one/four-thread and quiet coverage. The subsequent
`parallel_execution_reference` rerun passed all five tests through both
decoders, including whole-subtree pixel checks with statistics/profile
activation assertions; that official oracle is no longer pending.
No performance or complete-migration claim is made.

## Existing Default Coverage

The default learner already used source f32 probability-floored FastLog2f,
SSE2/AVX2 fused histogram movement/scoring with scalar fallback, the triangular
small-count table, predictor zero-tail trimming, complete property/predictor
sweeps, ordered split reduction, weighted sample counts, bounded histogram
reuse/subtraction, and property parallelism. Source sampling/quantiles,
channel/wave collection, and weighted batch merging were present before this
pass. Those components were not replaced with a new learner or approximation.

No default sample cap, predictor subset, threshold subset, heuristic early stop,
or leaf limit was introduced. Explicit source search experiments can select
property/threshold proposals as documented above. Tests compare internal counts/costs/decisions,
not JXL hashes. Encoding residuals, multipliers, syntax and pixel reconstruction
are unchanged. Decoder-oracle regression remains the parent's responsibility.

## Remaining Default-Path Differences

Parent verification, 2026-10-23: the predictor/property reuse changes passed
the full debug workspace (1,137 core tests) and 16 dedicated 8/16-bit
Gray/RGBA pixel comparisons through official djxl 0.12/0.13, covering local
single-thread and shared four-thread encoding. The 80 source-tier and 16
parallel-gather pixel checks also passed. The optional tools remain off by
default; this does not close the missing paths listed below.

- Sample columns now use source batched physical partitioning. Histogram loops
  still load their positions through a contiguous identity-index buffer rather
  than iterating a begin/end range directly.
- Prepared byte/static property columns, saturated u16 counts and the default
  two-choice source index table are implemented. Hume replaced production
  gather's old capped row-key collector with direct source-sized columns; see
  `MA_GATHER_PORT_STATUS.md`. The learner's consuming entry now accepts those
  columns without requantization/deduplication, and production source gathering
  calls that entry directly. The temporary compatibility adapter and its raw
  static-ID column have been removed; those are no longer migration gaps. An
  explicit residual-pair column conversion remains; no unsafe cast is used.
- The parent reports the separately owned FMA specialization selected and
  validated in the packed checkpoint. Wider-Highway-target and non-x86 SIMD coverage
  remain separate from these property-storage changes. Native target-specific
  numerical differences remain permitted; exact pixels, not tree/JXL identity,
  are the compatibility criterion.

These are source/code audit findings, not measured attribution of the remaining
wall-clock gap. There were no benchmark runs in this pass.

## Optional Source Paths Still Missing Or Partial

The following switches are **opt-in**, not missing default work:

| Source switch/path | Rust status |
| --- | --- |
| `JXL_MA_PREDICTOR_DEDUP=1` | Implemented as `TrainOptions::predictor_dedup`, default false; verified per-node reuse and populated inherited predictor caches. Parent reports public encoder option forwarding added. |
| `JXL_MA_PROPERTY_DEDUP=1` | Implemented as `TrainOptions::property_dedup`, default false; exact serial partition verification, inherited redundant/constant masks and cache invalidation. Parent reports public encoder option forwarding added. |
| `JXL_MA_HISTOGRAM_SHARDS=4` | Implemented as `histogram_shards` / HistogramOptions.shards, including unpaired initial-cache population. |
| `JXL_MA_SPARSE_ENTROPY=1` | Implemented as `sparse_entropy` / HistogramOptions.sparse with native/scalar active-block sweeps. |
| `JXL_MA_PADDED_HISTOGRAM=1` | Implemented as `padded_histogram` / HistogramOptions.padded with exact cache lease/root stride. |
| `JXL_MA_DENSE_HISTOGRAM=1/2` | Implemented as DenseHistogram::Small/Extended, default Disabled. |
| `JXL_MA_PROPERTY_BINS=1/2` | Implemented as PropertyBins::SmallOrCompacted/All, default Disabled. |
| `JXL_MA_PAIRED_CACHE=1` | Implemented paired deferred subtraction; Rust's immutable cache scoring needs no copied histogram. |
| `JXL_MA_COUNT_LOG_TABLE=1` | Implemented as ScoringOptions.count_log, source priority/gates and immutable-cache native scoring. |
| `JXL_MA_CLIPPED_ENTROPY=1..5` | Implemented as ScoringOptions.clipped, all five Fenwick/sliding policies and source record gates. |
| `TreeSamples::FastEntropy` | Implemented as ScoringOptions.fast_entropy, including Shannon leaf baseline and incremental split sums. |
| `JXL_MA_DESCRIPTION_COST=1..3` | Implemented as ScoringOptions.description on source leaf/split costs. |
| `JXL_MA_STATISTICAL_RACE=1` | Implemented as SearchOptions.statistical_race, with source proposals, evidence and cache invalidation. |
| `JXL_MA_QUANTILE_SPLITS=1/2`, `JXL_MA_REFINE_QUANTILES=1` | Implemented as SearchOptions.quantile_splits/refine_quantiles, default off, full-search fallback at source refinement memory gate. |
| `JXL_MA_CHANNEL_TREE=1..3` | Model partitioning implemented as ChannelTreeMode, default Disabled; bounded cooperative disjoint roots implemented in checkpoint fourteen. Central release learner tests and channel/row official djxl 0.12/0.13 pixel oracle passed, including one/four-thread coverage. |
| `JXL_MA_CHANNEL_FINAL_ONLY=1` | Parent-owned resolved-mode gate; parent reports metadata/tier/probe forwarding added. Channel learner validation is recorded above; gate-specific coverage remains parent-owned. |
| JPEG `JXL_MA_CHUNK_PARALLEL` with split parallelism | Explicit default-off cooperative_ma shares the borrowed pool across domains and nested splits; gathering/budgets unchanged. See JPEG_COOPERATIVE_MA.md; distinct from pixel channel-tree decisions. |
| `JXL_MA_SUBTREE_PARALLEL=1`, balanced/small-subtree variants | Implemented as ParallelOptions with source activation/grain/LPT/seed/mask/graft policies and null-runner whole jobs. Central release component tests passed; the five-test parallel_execution_reference rerun also passed both official decoders, including actual whole-subtree activation and pixel checks. Distinct from ready-sibling evaluation. |
| `JXL_MA_GATHER_STRIPES`, sparse gather variants | Separately implemented by Hume; source gates and validation are tracked in MA_GATHER_PORT_STATUS.md. |

The small entropy table's **default** limit of 256 is implemented. The source's
`JXL_MA_ENTROPY_TABLE_LIMIT` environment override (up to 4096) is not exposed.
Existing Rust parent reuse and batch merging intentionally enable some source
opt-in optimizations; this does not make every other source experiment default.

## Scratch And Verification Handoff

`evaluate_property_values` may be called repeatedly with the same private scratch
(the `evaluate_property_cached` dispatch wrapper is now test-only):

- Uncached `used`/`weights` are cleared and resized before scanning. Cached calls
  read the cache instead; stale scratch marginals are not consulted.
- Constant properties return before accessing any predictor scratch.
- `below`/`above` costs are reset for each nonconstant property.
- Unpaired/compact collection resets `increase`/`extra` completely. Even members
  of a pair reset both current and paired buffers; odd members swap in only the
  buffers just populated by their even predecessor in the same invocation.
  Predictor dedup requires both adjacent members to be active before pairing;
  an active odd member with a skipped predecessor uses ordinary collection.
- `counts_above` is replaced by the current parent; `counts_below` is cleared and
  resized for every predictor. The compact mapping is immutable per node, not
  worker state. No cached/constant early return can expose an old paired buffer.
- Property proofs collect/subtract marginals before token work; uncached scoring
  consumes those same prepared scratch marginals without scanning them twice.
  Inherited masks return no candidates and never consult prior scratch costs.
  Only a successful complete sweep publishes newly discovered proof bits.

Added learner tests cover lane order/union/savings gates (1/4/8 lanes), every
threshold cost with reused scratch across different sizes/predictor counts,
paired odd tails and unequal strides, leaf termination, raw-parent retention,
parallel cache population, constant-property subtraction, and the 8192-record
cache gate. Existing scorer tests now check the exposed backend lane count for
each available native profile and scalar fallback.

Second-checkpoint tests additionally cover exclusive/nested/unwinding scratch
leases and bounded retention; complete-tree agreement between fresh and retained
scratch in serial/parallel/null-pool execution; repeated deferred subtraction
versus fresh histograms; sibling-range reordering and once-only consumption;
empty/single-record/threshold-terminated cache disposal; partial-cache error
invalidation; and the PA-03 pre-collection budget checks described above.

Checkpoint validation is recorded above. Prepared-entry/caller and histogram
tests are included in central validation, with option forwarding active.
Quantile-32, quantile-64 and refinement official coverage has passed. No
low-level helper or internal-tree equality assertion substitutes
for complete release/decoder-oracle/corpus pixel-losslessness validation.

No cargo, build, benchmark, or decoder-oracle command was run by this agent.
Formatting was scoped to `modular/ma_train.rs` in checkpoint four and
`modular/training.rs` in checkpoint five, and to `modular/ma_train.rs` plus its
new local `modular/ma_columns.rs` in checkpoint six.
Checkpoint seven formats only `modular/ma_train.rs` and local `modular/ma_samples.rs`.
Checkpoint eight formats only `modular/ma_train.rs` with `skip_children=true`.
Checkpoint nine also formats only `modular/ma_train.rs` with `skip_children=true`.
Central regression coverage includes `modular::ma_train`, `tools::ma_simd`,
`tools::ma_entropy`, and `modular::training`; focused end-to-end pixel results
are recorded above. Future changes still require fresh central validation.
No whole-pipeline performance result or complete-migration acceptance is claimed.
