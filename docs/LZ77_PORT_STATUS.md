# Pixel-Lossless LZ77 Port Status

Source-file references below follow the 2026-10-29 `tools/lz77/` layout (paths
are relative to `crates/rs-jxl/src/tools/`). This path update does not rerun or
change the dated experimental results.


Reference: read-only `../jxl_python/libjxl`, HEAD verified as
`b866b3defec13494bfbaf705db93581cda118e37`. Audit date: 2026-10-23.
Sources below are relative to its `lib/jxl/` directory. Rust files are in
`crates/rs-jxl/src/tools/`; entry points are reexported by `tools::lz77`.

**Not complete.** The source-default pixel-lossless search/parser routes are
present, and the optional routes below now have implementations and focused
tests. This is not a claim that all optimized source tools, integration,
performance, or codec acceptance have been verified. Indexed stream arena
dispatch is connected to the parent-owned cooperative executor. Parent reports
five debug arena tests, one null-seed test and six official-decoder pixel
checks passed for this slice. Diagnostic tools remain missing.
The vendored serial SA-IS replacement now
has an integrated bounded adapter and parent dependency override; its focused
tests compile and pass on the current default Rust toolchain. Parent also
reports 172 release pixel comparisons, strict Clippy and Rust 1.89 all-target
check passed. No missing lossless experiment is reclassified as
lossy-only.

## Contracts And Defaults

- `parse_source[_streams]_with_state` preserves the outer-vector acceptance
  decision independently of match count. Rejected calls return the original
  literals with `enabled=false`; accepted literal-only/empty member streams
  retain the shared enabled state. There is no new post-DP savings gate.
- Distances remain physical, one-based; match contexts come from the original
  start token. Dictionary windows and maximum match lengths remain independent.
- Existing default selection is unchanged. New options default off; no
  process environment, global Rayon pool, native library, or subprocess codec
  is introduced. Caller-installed pools and explicit scratch limits bound work.
- The source's Modular selection in `enc_ans.cc:1714` chooses disabled,
  b1w3t, b3w3t, or Optc256, with Optc8 available to the lossless probe. RLE is
  also used for ancillary entropy streams. The generic enum is larger than
  that pixel-default set.
- The legacy `SourceBucketMode` remains an explicit override. Optional
  `bucket_selection` implements the source aliases and parameter-first
  precedence without changing existing callers; no environment is read.

## Source-To-Rust Coverage

"Implemented" means code and focused tests exist, not that central verification
or an independent decoder acceptance run has passed for this revision.

| Source tool / gate | Rust route | Coverage and applicability |
| --- | --- | --- |
| `enc_lz77.cc` RLE, source f32 histogram/cumulative model, savings thresholds | `lz77/source.rs`, RLE state entry points | Existing implementation; applicable default/ancillary lossless path. |
| `LZ77HashMap`, b1/b3 and fast bucket7 seed, lazy chain256 small/careful seed | `lz77/source/matchers.rs`, `fast_stream`, `careful_stream` | Existing exact capacities, insertion order, zero-run chains and 65536 combined-token gate. |
| Generic b1/b3/b7/b15/b31 `w3f` and `w3t`, Optc1/3 | `SourceLz77Method`, `lz77/policy.rs`, source dispatcher | Implemented all five exact fast ring capacities, 24/120 spatial probe counts, local cost-veto choice and shared global acceptance. Optc1/3 use actual chain-visit limits with the unchanged careful/fast seed policy and refit/range support. Existing default methods and legacy projection gap reporting are retained. |
| Default-on `FAST_LZ77_DISTANCES` | `range_search`, `Fast::find` | Sorted spatial bounds, canonical symbols and source probe limits. |
| Fast `LZ77HashMap::special_dist_table_` | `lz77/source/distances.rs::FastCodes` | Default-on i8 symbols, source width*8+8 extent trimmed only beyond the maximum queried dictionary distance. Reverse filling preserves aliases; -1 holes and distances beyond the table use generic codes. Width zero allocates nothing; at most window+1 bytes, with saturating width arithmetic. |
| `FAST_RANGES=0..3`, fast `PREFIX_FILTER=0/1/2` | `fast_ranges: SourceFastRanges`, `fast_prefix: Option<SourceFastPrefix>` | Implemented independent repeated insertion and eight-value extension, default off. Extension implies SIMD prefix mode only when no prefix override is supplied. Explicit source scalar/SIMD modes use lazy candidate-work activation with range search, eager without; explicit Disabled suppresses the implicit filter. Existing FilterPolicy overrides remain compatible. |
| Optc8/256, default match-cost 0, default-on cost spans and gated chain cache | `Optimal`, `optimal_stream_replay`, `lz77/dp.rs` | Existing search, strict endpoint ties, interval-cost equivalence and `PreferLZ77MatchCaching` seed gate. Default f32 operation order retained. |
| `COMPACT=1`, `ApplyLZ77_Optimal::CompactStep` | `compact_steps`, `lz77/dp/compact_steps.rs::Steps` | Implemented 12-byte repr(C) u32 length/u32 distance-symbol-plus-one/f32 cost nodes, source f32::MAX initialization and input-origin context recovery. Explicit chain Optc256 switch defaults off; parameter-selected short32 and eligible indexed endpoint routes select compact nodes automatically. Existing endpoint relaxation is unchanged; precise-ring and block/range heaps bypass this storage. Full bucket alias/parameter precedence remains separate. |
| `MATCH_COST=1..7`, cost-seed refit and shared-budget mode4 replay | `lz77/source.rs`, `lz77/source/range.rs`, `lz77/source/replay.rs` | Existing optional lossless tools; mode3 reuses the accepted parse as the next cost seed, modes4..7 remain chain range paths. |
| `MATCH_CACHE=0/1`, `CHAIN_LIMIT` | `chain_match_cache: SourceChainCache`, `chain_limit` | Implemented independent ordinary-optimal chain overrides. Auto retains the source seed gate; Enabled forces a bounded window cache, Disabled forbids it. Valid chain limits 1..256 clamp to the method maximum; invalid values retain the default. Seeds, buckets, indexed parsing and range modes ignore these overrides. |
| Mutable bucket `DOMINANCE=1` | `bucket_dominance`, `Buckets::find_cost_matches` | Implemented source-order short/long/history/anchor nominations, minimum-prefix checks, 16-entry required-length certificates, strict cost/symbol ordering and zero-prefix extension. Only match-cost modes 2/3 take this optional route. The index's existing dominance remains separate. |
| Non-block indexed `INTERVAL_DP=1/2` | `index_interval_dp: SourceIntervalDp`, `visit_endpoint_ranges` | Implemented explicit all-endpoint and boundary-only choices with uint000. Mode 2 selects only the first/last eight endpoints for constant-price spans wider than sixteen, using the existing endpoint relaxer for those destinations. Only bounded batch frontiers qualify; overflow re-queries use the full route. This approximation may change model cost; it is not an alias for exact DP. |
| `BUCKETS=32/parallel32`, history, wide-key, wide-ring/parallel-wide | `Buckets`, `bucket_selection: Option<SourceBucketSelection>` | Exact aliases now distinguish stored versus compact DP nodes and serial versus parallel/index eligibility. The bucket_matcher parameter wins over all selected aliases. None preserves the older SourceBucketMode candidate-override API. |
| `BUCKETS=128` | `ShortBuckets::Large`, `SourceBucketMode::Short128` | Implemented short128/long8, current-origin insertion and newest-first wrap. As in source, short128 ignores fingerprints; long8 can still filter. Serial optimal dispatch. |
| `BUCKETS=precise-ring` | `SourceBucketMode::PreciseRing`, `Costs::optimal_cumulative`, `Parser` | Implemented wide32 candidates with double cumulative and endpoint costs. Histogram prices and length-plus-distance additions remain f32 before promotion. Range modes still use source f32 chains. Serial optimal dispatch. |
| `PREFIX_FILTER`, `FINGERPRINT` | `lz77_prefix.rs`, native prefix SIMD helpers | Existing eager/lazy mandatory-prefix filtering with runtime SIMD/scalar fallback. Not an exact recreation of every environment alias or profiling switch. |
| `INDEXED=1/2`, `SMALL_INDEX`, baseline counting-sort index | `StaticIndex`, `SourceIndexMode` | Existing eligibility, stable current-inclusive short32/long8-or32 slots, query batches and overflow requery. |
| `RUN_KEYS=1/2`, `DEEP_PREFIX=1/2` | `lz77/source/index.rs`, `lz77/source/index/deep.rs` | Existing run sharing/alignment and bounded 16/64-token chains. |
| `PARALLEL_INDEX=1`, index match batches | `lz77/source/index/parallel.rs`, `lz77/source/index/batch.rs` | Existing bounded stable build and task-local match caches. Shared-run/deep restrictions remain. |
| `INDEX_TAGS=1/2` | `SourceIndexTags`, `visit_tagged_slots` | Implemented precomputed slot-order mandatory-three-token tags, scalar or native SIMD masks. Current slot masked out; descending masks preserve candidates. Tags are also usable by non-block batches. |
| `ANCHORS_FIRST=1` | `index_anchors_first` | Implemented source-order spatial anchors before deep/short/long nominations. Default remains anchors-last. |
| `BLOCK_DP=1/2` | `SourceBlockMode`, `lz77/source/blocks.rs` | Implemented explicit-endpoint and 65536-offer heap parsers. Normal dispatch requires eligible index + uint000. Full-stream dictionary, destination-only caps, f32 direct literal pricing, source repetition skip and bounded-frontier overflow requery. |
| `BLOCK_GRAIN=1..8`, `WORK_CUTS=1` | `block_grain`, `block_work_cuts`, `cuts` | Implemented first-pass command-boundary cuts, crossed-target coalescing and source work weights. Caller-pool scheduling also accounts for Rust scratch sizes. |
| `SEED_BLOCKS=1/2`, `SEED_HISTORY`, `TrimLZ77SeedPrefix` | `SourceSeedBlocks`, `seed_parts`, `merge_seed`, `trim_prefix` | Implemented dominant >=262144-token fast-seed splitting; warm overlap uses the duplicated block histogram/gate before trimming. Prefix-crossing copies retain physical distance and rebase context; residual lengths below 3 become literals. Careful seeds and serial scopes do not split. |
| `SHARED_INDEX=1`, `enc_lz77_seed.h::LZ77IndexedSeed` | `shared_index`, `indexed_seed`, retained `StaticIndex` | Implemented largest eligible >=65536-token fast seed with INDEXED=2/BLOCK_DP=2 and short32 non-history modes. Build once, scalar tags/anchors-first, double prefix/gain reduction, then reuse for first block pass. Stream parallel seed dispatch is disabled, as in source. Logical seed partitions now use source task/scratch limits even with a null runner; absent overrides use the source defaults. Refit does not rerun the greedy seed. Auto-packs all-u16 values independently of COMPACT_VALUES; word matching and coordinates stay off. Rust additionally requires uint000. |
| `SUFFIX=1..4`, `LZ77ActiveRanks` | `SourceSuffixMode`, `lz77/source/suffix.rs` | Implemented earlier-position/sliding rank neighbors, strict distance < window, source eight-link recent chain and 24 anchors. Eligible block dispatch uses intervals even for BLOCK_DP=1. |
| `enc_lz77_lcp.h::LZ77SuffixLcp` | `Lcp` in `lz77/source/suffix.rs` | Implemented Kasai construction, 32-rank prefix/suffix minima and block sparse-table RMQ. Precomputed neighbor lengths preserve candidate order. |
| `enc_lz77_suffix.cc::libsais_int`, `JXL_EXPERIMENTAL_LIBSAIS` | `lz77/source/sais.rs`, `vendor/libsais-rs` =0.2.2 | Implemented bounded integer serial adapter to the approved Rust translation of libsais 2.10.4. Five vendored helpers now bypass Rayon for effective one-worker calls. Source caps, copied input, 6*k scratch tail and status/static-index fallback are retained; radix-doubling is removed. Parent reports all focused tests pass, including global-pool isolation; three release source-LZ suites/172 pixel comparisons include suffix routes. Strict Clippy and Rust 1.89 all-target check passed. |
| `COMPACT_VALUES=1`, `LZ77StaticIndex<Value>` | `index_compact_values`, `lz77/source/index/values.rs::Values` | Implemented actual u16 value buffer only when every value <=65535; otherwise retain full u32 values without truncation. Applies to ordinary static block indexes after suffix fallback. Hashes promote values to u32 before wrapping arithmetic; deep chains retain their source u32 construction. No sentinel or context narrowing. Shared seed auto-selection is separate, as above. |
| `WORD_MATCH=1`, `LZ77WordMatchLength<uint32_t/uint16_t>` | `index_word_match`, `word_length`, `word_length_u16` | Implemented source little-endian 64-bit word XOR/first-mismatch localization, two u32 or four u16 values per word, safe unaligned slices and scalar tails. Only ordinary static block-index construction receives the option; non-block batches, shared seed indexes and suffix queries keep the source non-word route. The non-word eight-value comparison also supports packed u16. |
| `INDEX_COORDINATES=1`, `counts_` | `index_coordinates`, `StaticIndex::counts` | Implemented one u16 per origin: current-inclusive saturated short count in low byte, long count in high byte; shared-run slots reuse the long count. Stable serial/parallel slots are retained. Queries skip hash/offset lookup and apply deep-eight's nine-slot cap after loading the count. Applies to ordinary static blocks and non-block batches, not shared seed or suffix indexes. |
| `INDEX_CACHE=1/2`, `LZ77MatchCache` | `index_match_cache`, `block_sparse_cache`, `lz77/source/search.rs::MatchCache` | Direct source u32 endpoint cache now used by static/suffix/block queries and index batches. Sparse blocks use min(window,4096) slots plus distance tags, overwrite endpoints even when shorter, saturate at u32::MAX and reject alias hits. Sparse overrides dense in blocks; non-block batches ignore INDEX_CACHE=2, as in source. Legacy fast/chain/bucket caches are unchanged. |
| `SHARED_FRONTIER=1` | `block_shared_frontier`, common `Query` result | Implemented caller-owned 16-entry frontier, reset even for out-of-range/tail queries. Block overflow requery omits the result and bound, preserving all emitted proposals. |
| `FRONTIER_CARRY=1` | `block_frontier_carry`, `FrontierCarry` | Implemented source-order verified endpoints carried before fresh nominations, clipped to each destination cap, expired on progress and reset after every queried frontier (including empty). State is local to a block. Both static and suffix queries consume it. |
| `DISTANCE_BOUND=1` | `block_distance_bound`, `DistanceBound` | Implemented uint420-class minimum price over the full source symbol bound, first-symbol ties and exact static-index early-exit points. Suffix queries deliberately ignore this bound, as in source. |
| `DISTANCE_TABLE=1`, `DISTANCE_CLASSES=1` | `block_distance_table`, `block_distance_classes`, `DistancePrices` | Implemented shared per-stream table and 128 classes, source class-span filling and classes-first lookup when both requested. The same lookup prices matcher pruning and block relaxation. This is separate from mode4..7 class reduction. |
| `JXL_LOSSLESS_BUCKET_DISTANCES=1`; static/suffix default dense symbols | `bucket_distances`, `lz77/source/distances.rs::Codes` | Implemented bounded i16 special-symbol table iff width is nonzero and largest special distance is positive and strictly <65536. Mutable buckets opt in; static and suffix constructors enable automatically. Reverse insertion retains canonical narrow-width aliases, -1 holes use generic codes, and oversized tables retain sparse lookup. At most 128 KiB per matcher/index. |
| Block `CALENDAR=1` | `block_calendar`, `HeapParser` | Implemented destination heads, linked event lists and recycled slots. Queued count includes pending events and active heap offers; activation, exclusive expiration and heap compaction retain source gates. Only interval blocks enable the calendar. |
| Block `VECTOR_DP=1`, `enc_lz77_bucket.cc::RelaxCosts` | `block_vector_dp`, `lz77/source/range/vector.rs` | Implemented four source f32/u32 SoA arrays and strict masked updates of cost/length/distance/context. Runtime SSE2/AVX2 with portable scalar fallback; native non-x86 targets are not claimed. Both block DP modes accept the option; non-block/LAZY routes bypass it. Physical distances retain the core command contract. |
| Block `RANGE_DOMINANCE=1` | `block_range_dominance`, `HeapParser::interval_context` | Implemented 16-endpoint full-block cost certificates, <4 block-index-span query gate, <= comparison semantics and at most 32 full-block updates per interval. Partial coverage never certifies a block. Both block DP modes support it. |
| Block `DP_JUMP=1` | `block_dp_jump`, `jump_literal_only` | Implemented only for source repetition-skip origins in interval blocks. Stops before pending starts, active exclusive ends and explicit writes; vector/dyadic/calendar disable jumps, as do absent active offers. Landing states retain source cost/context/distance, and the landing origin is reprocessed. |
| Block `DYADIC_DP=1..3` | `block_dyadic: SourceBlockDyadic` | Implemented endpoint, 64-endpoint and 16-endpoint leaves; strict-greater tree base, future-only interval decomposition, explicit partial boundaries and cached changed root-path suffix. Strict cost ties are distinct from the ordinary optimal parser's earliest-origin tree. Only interval blocks enable it. |
| Block `BLOCK_ENVELOPE=1..3` | `block_envelope: SourceBlockEnvelope` | Implemented 64/16/256-endpoint full-block cheapest offers plus explicit partial boundaries. Only interval blocks enable it. Combined source precedence is dyadic > envelope > calendar/heap, with vector storage and dominance orthogonal. |
| `SEED_GUIDED=1`, `LZ77SeedSpans` | `SourceSeedGuided::Hints`, `block_seed_guided`, `seed_spans` | Implemented command-boundary copy spans retaining physical distances, per-block lower-bound lookup and verified containing-copy nomination. Hints follow carried matches and precede fresh index/suffix nominations. Overflow requery retains the hint. No span endpoint cap is imposed on a newly verified match. |
| `SEED_GUIDED=2`, `LZ77SeedSpan::Search` | `SourceSeedGuided::SparseOrigins` | Implemented source repair-origin sampling: all origins for copies shorter than 64; otherwise first 8, last 8 and every 32nd origin from the copy start. Literal relaxation and existing repetition skip precede sampling. Skipped origins do not replace the carried frontier. |
| Block `LAZY=1` | `block_lazy`, `parse_lazy` | Implemented per-block double literal prefixes, increasing constant-cost length endpoints, strict positive gains, longer-length equal-gain ties, first candidate on equal length/gain and strict one-token lookahead with pending-choice reuse. Physical copies may cross cuts backward, never forward. Uses existing index caps and dense/sparse verified cache. Source ignores guidance, carry, shared frontier, bound and class-price lookup in this branch. |
| `ZERO_TAIL=1` | `block_zero_tail`, `zero_tail_end` | Implemented block-local backward scan requiring both zero extra bits and exactly zero model Bits. Nonzero literal values can qualify; tiny nonzero prices cannot. DP still relaxes every literal, then skips queries at/after search_end before repetition/guidance handling. Earlier copies retain the original block destination limit. Source lazy computes but does not use this query cutoff. |
| `LITERAL_BOUND=1` | `block_literal_bound`, `literal_bound_prefix`, `literal_bound_exceeded` | Implemented double prefix of literal Bits then extra bits, source left-associative upper bound and mixed f32/f64 rounding margin. Only strict cost > upper+margin prunes a constant-cost interval. Longest-match, repetition-skip and carried-frontier updates still see pruned nominations. Applies to both DP block modes, including overflow replay; source lazy bypasses it. |
| `enc_lossless_workers.h`, stream/fast/index/block workers | `lz77/source/workers.rs` | Installed-pool scheduling extends to bounded block/seed jobs. Logical partitions retain source job/task/scratch limits independently of physical pool width; parser execution slots are capped separately after cuts. Parent reports the logical-jobs, two-thread/eight-block and seed-parts tests pass. No new/global/nested pools. |
| `STREAM_ARENA=1`, `ARENA_QUIET=1` | `stream_arena`, `stream_arena_quiet`, `workers::StreamArena` | Implemented source indexed/non-reuse/multiple-nonempty-stream gates, stable longest-first order, source scratch-limited roots and logical worker slots. Calls the parent-owned `execution::cooperative` on the current caller pool; child index builds/batches/blocks use arena-aware execution primitives. Input-order results, failure propagation and shared entropy acceptance are retained. Both switches default false. Parent reports five debug arena tests and six official-decoder pixel checks passed. |

## Remaining Applicable Tools

These are lossless-applicable opt-in tools, not lossy-only exclusions:

| Source tool | Remaining gap |
| --- | --- |
| `DETAIL`, `BUCKET_AUDIT`, profile output | Missing source diagnostic/audit/profiling emission. |

The previous null-runner indexed-seed partition discrepancy is fixed, with its
focused test passing centrally. The arena slice below has five passing debug
tests and six passing official-decoder pixel checks reported by parent. These
results validate the tested routes, not every executor policy or performance
case; no independent tests were run by this worker.

SA-IS code and parent dependency wiring are integrated; all focused tests pass
on the current default Rust toolchain. Release source-LZ pixel comparisons,
strict Clippy and Rust 1.89 all-target compatibility check passed. This is no
longer an unimplemented suffix algorithm.

## Lossy-Only And Ownership Boundaries

No suffix, bucket, block, or seed tool requested in this task is lossy-only.
VarDCT quantization/near-lossless DC policy and the non-lossless caller settings
in `HistogramParams::ForModular` are outside the pixel-lossless objective, not
excuses to omit the generic lossless-capable LZ matchers above. Entropy wire
configuration/serialization, scheduling policy selection and decoder acceptance
belong to their parent-owned modules; this task does not edit them or
`tools/mod.rs`.

## Block Parser And Distance Batch

New fields, all source-default off:

- `bucket_distances: bool` (mutable optimal buckets only).
- `block_calendar: bool` (`CALENDAR=1`).
- `block_vector_dp: bool` (`VECTOR_DP=1`).
- `block_range_dominance: bool` (`RANGE_DOMINANCE=1`).
- `block_dp_jump: bool` (`DP_JUMP=1`).
- `block_dyadic: SourceBlockDyadic::{Disabled, Endpoints, Blocks64, Blocks16}`
  (source values 0, 1, 2, 3 respectively).
- `block_envelope: SourceBlockEnvelope::{Disabled, Blocks64, Blocks16, Blocks256}`
  (source values 0, 1, 2, 3 respectively).

Both enums are reexported by the owned `tools::lz77` module. Parent confirms
SourceConfig already flows through the encoder without extra forwarding.
No generic registration, executor, entropy or Cargo file was changed. Static and suffix
dense distance maps are source-default on and do not require a parent flag.
The dense symbol table is independent of DISTANCE_TABLE's f32 model prices.

Existing Optc256/index/Frontier/size/uint000 dispatch gates are unchanged.
Static blocks, successful suffix blocks and retained shared-seed blocks all
receive the new options. Suffix dispatch still forces interval blocks even
if BLOCK_DP=1 was requested. LAZY returns before DP allocation and ignores
these DP options. Non-block modes 4..7 retain their previous default heap.

Sources are frozen `enc_lz77.cc` bucket construction, `enc_lz77_index.h`,
`enc_lz77_suffix.h`, `enc_lz77_block.h::ParseLZ77Blocks`,
`enc_lz77_parse.h::LZ77RangeParser`, and
`enc_lz77_bucket.cc::RelaxCosts`. No alternate endpoint tree was substituted
for the source dyadic cache or block-envelope algorithms. In particular
strict ties, endpoint alignment, event-list insertion, queue limits, and
DP_JUMP's source gating are retained. The public physical-distance contract
is unchanged, including copies into history before a block's start.

AoS steps and dyadic offers occupy 16 bytes. Vector mode replaces the AoS
array with four 4-byte arrays; it does not allocate both. Calendar heads add
4*(n+1) bytes, certificates add 4*(floor(n/16)+1), and envelopes add one offer
per selected block. Dyadic nodes are twice the smallest power of two strictly
greater than floor(n/leaf_width). These arrays use checked sizes and fallible
reservations. The existing 2^24 index input bound remains; the generic range
constructor also rejects indices outside the signed source domain. Event
slots and each heap retain at most 65536 entries for blocks. Worker scratch
estimates include all requested arrays, even ones bypassed by a higher-priority
variant; the existing 8 MiB allowance covers retained heap/event buffers.
The existing worker budget limits concurrent owners and still permits one
oversized task, not a hard total-memory ceiling. That block/distance batch did
not add a cooperative arena; the later integration is documented below.

Fourteen new focused tests cover dense/sparse canonical symbols and candidate
order, table cutoffs/holes, all block variants against an unpruned scalar cost
oracle, dyadic/envelope alignment and source ties, calendar recycling and
expiration, dominance certificate limits, literal-jump gates and event bounds,
default/mode/scratch gates, SSE2/AVX2 versus scalar on every SoA field with
misalignment/tails/ties, independent command replay across non-aligned cuts
using static and suffix search, and LAZY bypass. Tests do not compare whole
JXL hashes. **Parent reports the 1194-core central workspace checkpoint passed,
including this block/dense batch. Official pixel cases for the new fields are
being added by parent; no Cargo/build/test/benchmark was run by this worker.**

The remaining-tools table above remains open. The following independent batch
adds interval/dominance/cache/chain controls, followed by the generic methods
below. Fast-range overrides and the SA-IS adapter are covered in later sections;
cooperative arena dispatch is covered in the later integration section.

## Indexed And Chain Controls

New fields, retaining source defaults:

- `chain_limit: Option<u32>`, default None. Source 1..256 values clamp to
  the method's maximum; 0 and >256 fall back to that maximum.
- `chain_match_cache: SourceChainCache::{Auto, Disabled, Enabled}`, default
  Auto. Only ordinary optimal chains consume it, not range modes or seeds.
- `bucket_dominance: bool`, default false. Source mutable bucket cost-aware
  nomination is selected only in match-cost Frontier/Refit (2/3).
- `index_interval_dp: SourceIntervalDp::{Disabled, AllEndpoints, Boundaries}`,
  default Disabled. Source values are 0/1/2. Only successful non-overflowed
  non-block indexed batches with uint000 use the boundary-only approximation.

No generic scheduler/encoder change is required: parent confirms the entire
SourceConfig is already carried through. All original return/acceptance,
physical-distance, literal/context and maximum-match contracts are unchanged.
Source references are `enc_lz77_parse.h::LosslessLZ77ChainLimit`,
`PreferLZ77MatchCaching`, `enc_lz77.cc::ApplyLZ77_Optimal` option gates and
`LZ77BucketMatcher::FindCostMatches`, and
`enc_lz77_cost.h::VisitLZ77CostRange`. The existing bounded certificate and
endpoint relaxer are reused, not a new search objective for modes 0/1.

Six new tests cover the cache/depth gates and invalid/clamped limits, scalar
best-distance envelopes versus pruned bucket nominations in all bucket layouts,
independent scalar boundary masks across uint000 classes, a decoded example
where boundary-only DP intentionally has a different cost, independent replay
of all indexed choices and chain/bucket controls, and unchanged fast/default
routes. The existing 22-candidate overflow test now checks interval eligibility
before/after overflow and at an empty tail frontier. This second batch was
added after the historical 1194-core result; later parent verification is
recorded in the current checkpoint below. No Cargo was run here.

## Generic Source Methods

`SourceLz77Method` now additionally exposes `Lz77b1w3f`, `Lz77b3w3f`,
`Lz77b7w3f`, `Lz77b15w3f`, `Lz77b31w3f`, `Lz77b7w3t`, `Lz77b15w3t`,
`Lz77b31w3t`, `Optc1` and `Optc3`. Existing variants retain their original
ordering and the default remains b3w3t. No new SourceConfig field is needed.
The source dispatcher at `enc_lz77.cc:2423-2448` selects these exact capacities
and runtime-cost flags. Rings still use physical slot insertion/traversal;
capacities are not rounded to powers of two. Optc1/3 still use the 256-link
careful seed below 65536 combined tokens and bucket7 above it, then their own
1/3-visit optimal chains. Cache/depth overrides clamp to those maxima; bucket
and compact overrides remain Optc256-only. No scheduler choice was changed.

The `w3f` methods bypass only the local `lz77_cost > literal_cost` veto. They
still calculate signed savings, consume source matches, and apply the same
global savings threshold. This is not a force-accept or no-cost-model mode.
Three new tests check all method metadata/seed boundaries, a zero-price input
where w3f and w3t differ locally but the normal global gate rejects both,
independent physical-command replay on wide values and contexts, and Optc1/3
range/refit dispatch. Existing scalar matcher, physical ring rotation,
repeated-range/tail and exact visit-count tests now cover capacities 15/31
and chain limits 1/3. **Parent subsequently reported all 68 generic-method
official pixel comparisons passed (17 routes x two bit depths x official
djxl0.12/0.13).** This does not verify the later alias/SA-IS slice. No build/test
command was run here.

The separate legacy `source_options` projection remains explicitly inexact.
It records the additional missing per-match-veto behavior for w3f, and its
Optc3 proxy uses a valid four-entry native bucket while retaining source's
exact limit three in metadata and its mismatch report. These projections are
not the executable `parse_source` routes.

## Fast Range And Prefix Controls

New SourceConfig fields are `fast_ranges: SourceFastRanges::{Disabled,
RepeatedInsert, MatchLength, Both}` (source 0/1/2/3, default Disabled), and
`fast_prefix: Option<SourceFastPrefix::{Disabled, Scalar, Simd}>` (source
explicit 0/1/2, default None). The source is `enc_lz77.cc:733-981` and
`ApplyLZ77_LZ77` at 1022-1041. These controls affect generic fast methods and
the fast bucket7 optimal seed, not careful chain seeds or optimal matchers.

Previously the fast matcher always used the bulk insertion and eight-value
extension kernels. Now each source bit selects its own kernel, with scalar
defaults and unchanged candidate order, physical ring rotation, match limits,
cost veto and acceptance contracts. Mode 2/3 implicitly enables prefix mode 2
unless `fast_prefix` or an existing non-Disabled `prefix_filter` overrides it.
Explicit source prefix choices use the source range-search activation policy:
max(4096, n/2) candidate entries and position strictly before n/2, or eager
construction without range search. Width zero never builds a prefix cache.
Scalar tags preserve duplicate-distance traversal; SIMD masks visit canonical
distances in ascending order. Both retain exact value verification. The cache
remains one byte per input value, with bounded slice loads instead of padding.

Three new tests cover scalar candidate equivalence for all fast capacities,
four range modes and scalar/SIMD/off prefix overrides; implicit/explicit,
early/late activation and width-zero gates; and independently decoded commands
including signed-bit-pattern values, maximum-match clipping and trailing
prefixes. Existing ring/tail tests now exercise all range modes and split
updates inside runs. **This slice awaits parent compilation and tests; no
Cargo command was run here.** SourceConfig forwarding is already complete.

The same audit found the fast matcher's source-default dense distance layout.
`FastCodes` now uses the source i8 reverse-filled table, clipping only the
unqueryable suffix above the dictionary window. The separate bucket/static/
suffix i16 table retains its strict 65536 gate. A fourth new test checks the
fast table's widths, aliases, holes, upper-window distances and overflow-safe
allocation bound against scalar lookup. Fast worker scratch now allows 8 MiB
fixed storage, covering 31-entry usize rings and the <=window+1-byte table,
plus the unchanged 16 bytes per input value. A ring-size assertion checks that
fixed bound. The unrelated block scratch estimator's endpoint-tree arithmetic
now saturates even for usize::MAX; its existing bound test covers that input.

Source `enc_lz77_parse.h:78-103` has eight-value `memcmp` and LE64 word
extension, not a separate full-match Highway kernel. Rust's corresponding
eight-value comparison/scalar-tail and u32/u16 word functions are present.
The stale header entry claiming a missing source full-match SIMD algorithm was
removed. This does not claim explicit native intrinsics for these comparisons
or matching compiler code generation; the source Highway LZ kernels are
fingerprint masks, spatial masks and vector DP relaxation.

## Exact Bucket Alias Slice

The new field is `SourceConfig::bucket_selection:
Option<SourceBucketSelection>`, default None. Some enables the frozen
`enc_lz77.cc:2450-2498` lossless dispatcher choices: `Unset`, `Short32`,
`Parallel32`, `History`, `WideKey`, `WideRing`, `ParallelWide`, `PreciseRing`,
and `Short128`. This models cost-spans-enabled, non-audit source dispatch;
diagnostic environment/plan-audit emulation is still not provided.

With Some, `bucket_matcher=true` always selects compact short32/long8 before
examining the alias. Otherwise Short32 and WideRing retain stored-context
nodes, stay stream-serial and cannot enable static/shared indexes. Parallel32
and ParallelWide use compact nodes even without INDEXED and permit the source
parallel/index gates. History/WideKey/PreciseRing/Short128 keep their original
noncompact layouts. Unset selects the chain and honors its separate COMPACT
flag. The obsolete candidate-only `bucket_mode` is ignored only with Some;
None keeps every earlier route and override behavior, including the legacy
indexed-wide shorthand. Other methods and range parsers retain their gates.

Four focused tests cover every alias and the parameter override, independent
decoded wide-value commands and actual node/index counters, installed-pool
stream gates and shared-index seed eligibility, and unchanged fast/Optc3/range
routes. Integration is within the existing matchers, index builder, shared seed,
node selector and owned stream scheduler. Scoped formatting/checks passed;
later parent verification is recorded in the current checkpoint below.
No Cargo was run by this worker.

## Logical Partitions And Pool Width

The bounded follow-up removes physical pool width from `workers::index_tasks`.
With an installed caller pool, its result is now the source job bound capped
only by `SourceParallelism::max_workers` and the supplied per-task scratch
estimate. Defaults and validation are unchanged: task cap 8 by default, at
most 64, and one oversized job is still allowed. Null pool, explicit serial
scope and parallelism=None retain their existing one-part gates for ordinary
block/index work. Shared indexed seeds now have the distinct source null-runner
rule described in the next section.

`workers::index_workers` is the separate execution-only helper. Block parsing
first selects the source logical count and grain-expanded seed-aligned cuts,
then limits live parser owners by actual Rust parser scratch and physical
caller-pool width. It never rebuilds cuts after reducing owners. Indexed seeds
likewise select logical destination partitions before limiting execution slots.
For 524288 tokens, task limit 8 and a two-thread caller pool, BLOCK_GRAIN=1 now
requests eight parts, not two; grain 3 requests 24 before seed-boundary
coalescing. Source refs: `enc_lossless_workers.h::Tasks`,
`enc_lz77_block.h:132-142`, `enc_lz77_seed.h:33-35` and `enc_lz77.cc:1656-1660`.

All shared-helper callers were inspected:

| Caller | Retained bound and ownership |
| --- | --- |
| Block cuts | Logical jobs clamp input/65536 to 1..32, then task/scratch caps. Grain is 1..8, so there are at most 256 cut intervals. Execution uses a bounded set of physical parser owners. |
| Shared indexed seed | max(1,input/65536), capped by the configured task/scratch limit; only execution slots are pool-capped. The later null-runner fix preserves this logical count without a runner, using default source limits if parallelism=None. |
| Cold/warm seed splitting | Dominant-stream and installed-pool gates remain; at most 16 source parts. Warm history and later merge/trim behavior are unchanged. |
| Stable parallel index build | max(1,input/16384), at most 64 configured logical chunks. Scratch estimate includes every retained count array and Rust disjoint-scatter slice array. Shared-run/deep-prefix exclusions remain. Stable slot ordering is independent of chunk count. |
| Indexed match batches | At most 16 logical cache partitions. The task/scratch cap charges all retained caches, not merely currently executing callbacks. Cache verification and frontier-overflow requery are unchanged. |

The logical-count fix added no unbounded job queue, extra pool, nested execution
or option. `each` retains a serial scope for each bounded leaf chunk; ordinary
stream `Schedule::new` remains unchanged. The later arena route is separate.
Physical distances, destination caps, input-order merges and return contracts
are unchanged. Fixing logical seed/block partitions can change command choices
and compressed size on narrow caller pools; no whole-JXL identity is claimed.

Four new focused tests cover two threads executing eight logical jobs exactly
once with at most two active callbacks; source task/scratch limits and null
gates; two-thread/eight-block and grain-3 parsing against a sequential fixed-cut
oracle with independent command decoding; full null/serial/scratch-starved block
replay; and cold/warm/shared-seed partitions on two threads. Existing installed-
pool tests now check four stable index chunks and eight cache partitions against
their serial oracles on the same two-thread pool. Parent reports the logical-
jobs, two-thread/eight-block and seed-parts tests pass. The current arena
checkpoint additionally confirms the null-runner indexed-seed test passes.
No separate full count-suite result is recorded here. No Cargo was run here.

## Cooperative Stream Arena Integration

`enc_lz77.cc:2357-2379` enables STREAM_ARENA only for indexed parsing without
shared seed-index reuse, with a caller pool and more than one nonempty stream.
It stable-sorts roots by decreasing token count, bounds active roots by
LosslessWorkerLimits/scratch, and supplies each root a nested runner on the
same caller invocation. The algorithm in `enc_lossless_probe_pool.h` is not
ordinary nested Rayon dispatch:

- Idle workers take the newest published descendant group before new roots.
- A suspended parent's worker helps only its own child group, not unrelated
  roots or ancestor siblings, protecting retained thread-local scratch.
- Group claim quantum is max(1, count/(4*actual_threads)); completion drains
  every claimed child before removing its stack-backed group.
- Root failure stops new roots but drains active roots/groups. ARENA_QUIET
  directly runs zero/one-item groups and suppresses nonfinal child wakes;
  publication, root completion and final child completion still wake waiters.

New explicit `SourceConfig` fields:

- `stream_arena: bool = false`, corresponding to STREAM_ARENA=1.
- `stream_arena_quiet: bool = false`, corresponding to ARENA_QUIET=1 for this
  arena. An already-active parent arena retains its executor policy.

`workers::StreamArena` replaces only eligible indexed optimal stream dispatch.
It does not wrap the seed, rejected calls, fast methods or range/refit modes.
It requires explicit parallelism, a current non-serial caller pool, source
indexed eligibility and at least two nonempty streams. Retained shared seed
index reuse disables it. SMALL_INDEX retains the source 8192 minimum without
an extra combined-65536 gate. Nonempty stream indices are stable-sorted by
decreasing input size, while result ownership stays in input order.

The indexed compact-float scratch expression is exactly
`28*(largest+1) + 21*(1<<20)` bytes: 12-byte node plus four 4-byte per-token
arrays and the fixed source allowance. Root concurrency uses
`min(nonempty_streams, configured_tasks, max(1, scratch_budget/estimate))`.
This is a source estimate, not a hard total-RSS bound; retained input/output,
shared models and nested Rust child allocations are not a global arena budget.
One oversized root remains permitted. Logical worker slots are passed unchanged;
physical concurrency is the executor's responsibility. Each existing index,
cache and block helper retains its own bounded scratch/job calculations.

The parent API is now `execution::cooperative(count, worker_slots, max_roots,
quiet, callback) -> bool`, on the current installed pool. Root callbacks do
not enter a serial scope: they may publish child work to the same arena.
`workers::each` now calls `execution::for_each_mut`, with a serial scope only
inside each independent leaf partition. Existing block and seed owner loops
already call arena-aware `execution::map`; their leaves stay serial. There are
no LZ direct Rayon parallel iterators, new threads or private pools. No borrowed-
pool public wrapper is needed, and no entropy/options/executor/scheduler file
was edited here. SourceConfig already flows through the parent encoder; only
the two fields above need setting at the call site.

Result cells are fallibly reserved, one per original stream. An actual task
error is returned before diagnosing omitted/cancelled streams, so an unclaimed
earlier input cannot mask a later failing longest-first stream. The executor
owns stop-new-roots/drain-active-work and panic unwinding. Tests explicitly
exercise reuse after cancellation and child panic; this module does not add a
second scheduler or substitute ordinary Rayon nesting for the source arena.

`indexed_seed_tasks` separately ports `enc_lz77_seed.h:33-35` with no runner
gate. Null pool, explicit serial scope and parallelism=None retain the same
logical destination partitions; None supplies the default eight tasks/1 GiB.
Explicit limits still apply, including the 4 MiB estimate and one-oversized-job
rule. Only execution owners fall to one. Per-part double prefix/gain sums,
input-order double reduction and one final f32 conversion are unchanged.
Ordinary block cuts and cold/warm seed splitting keep their previous null gates.
This intended correction may change seed commands/compressed size on null
paths; physical-distance replay remains the acceptance contract.

Five new arena tests define source gates/defaults/stable LPT/scratch thresholds,
same-pool nested cooperation and mutable leaf work, error/panic draining, and
real non-block index batches/static endpoint/static interval/suffix routes
against fixed-model sequential-stream runs with independent command decoding.
They also cover shared-index exclusion and rejected/literal-only acceptance.
One new indexed-seed test checks null/pool/serial and absent-limit equivalence,
explicit task/scratch limits, cut-bounded copies and full command replay.
Parent caught a test-only empty-slice inference error during initial compilation;
the fixture now has an explicit `[[&[Literal]; 2]; 3]` type. Scoped formatting
passed. Parent confirms all five arena tests passed in debug and the logical
null-seed test passed. Integration test
`indexed_lz_stream_arena_pixels_match_official_djxl` also passed: three cases
decoded by two official decoders, for six exact-pixel checks. Parent reports
the four accompanying integration tests passed as well; no additional pixel
count is inferred for those tests. No Cargo/build/test/benchmark was run by
this worker.

The parent-owned native projection regression also passed: explicit LZ tool
flags survive projection from `base.modular.source_lz77`, while candidate-owned
method, bucket/careful policy and wire configurations still take precedence.
The None-LZ route remains disabled. This forwarding is not an edit in this
worker's LZ slice. See `SCHEDULER_PROBE_API.md` for the separately verified
six-test release checkpoint of the parallel small-probe convenience helper.

## SA-IS Dependency Audit

The frozen checkout contains the optional `libsais_int` adapter, not a
vendored libsais engine. `lib/jxl.cmake` expects an external
`JPEGXL_EXPERIMENTAL_LIBSAIS_DIR`; both inspected local CMake caches leave it
empty. Thus b866b3d alone does not identify a frozen libsais implementation
revision. The previous radix-doubling code has been removed, not relabeled SA-IS.

Parent approved and added/fetched
`libsais-rs = { version = "=0.2.2", default-features = false }`. The pinned
registry source was inspected read-only at
`$CARGO_HOME/registry/src/<registry>/libsais-rs-0.2.2`.
Its `src/lib.rs` identifies a Rust translation of libsais 2.10.4; SHA-256 is
`86e314fb2ccdb606484d62463bda55f0bee38bc455f3adf8e476dac3071d9d5a`.

- The registry `Cargo.toml` has edition 2021, default=[], and optional cc only through
  upstream-c. `build.rs:1-6` merely emits rerun-if-changed by default; C
  compilation/linking is entirely cfg(upstream-c). No native feature is enabled
  in the inspected workspace manifests. No registry or root Cargo file was edited.
- `libsais_int` at `src/lib.rs:12360` accepts mutable i32 text, i32 SA, alphabet
  k and extra element count fs. It passes threads=1 through normalization and
  recursion. It checks negative fs and SA extent, but does not validate symbol
  range/k or all input-size conversions. The adapter must enforce source
  n<=2^24, values<2^20, k=largest+1, checked n+6*k, and fs=6*k before calling.
  Text must be copied because errors need not restore it; nonzero status must
  retain source static-index fallback, not try a different suffix algorithm.
- The source-required 6*k i32 suffix tail costs 24*k bytes. The former radix
  estimate's usize alphabet buckets were not a sufficient bound for this tail;
  the integrated wrapper now checks 80*n+24*k+1MiB before allocating, respecting
  `suffix_scratch_bytes`. Recursion reuses the SA tail/local buffer, but the
  translation also has bounded heap buckets and an n/2-element LMS copy at
  lib.rs:10574. Internal Vec allocations are not fallible Result APIs.
- **Registry defect fixed in the vendor copy:** serial integer calls reached unconditional
  `(0..1).into_par_iter()` in five helpers: partial_sorting_shift_markers_32s_6k_omp
  (3903), mark_distinct_lms_suffixes_32s_omp (7186),
  clamp_lms_suffixes_length_32s_omp (7224), reconstruct_lms_suffixes_omp (7456),
  and clear_lms_suffixes_omp (10246). `run_rayon_with_threads(1, f)` calls f
  directly but does not make a Rayon iterator serial. Locked Rayon1.12.0
  iter/plumbing::LengthSplitter constructs Splitter::new/current_num_threads
  before deciding whether to split even a singleton. Locked rayon-core1.13.0
  registry.rs:339 reaches global_registry outside a worker. Thus the unpatched
  utility violates the no-global-pool contract on normal caller-thread use.
- The authorized `vendor/libsais-rs/src/lib.rs` now has effective-one-worker
  branches at all five helpers. Mark/clamp/reconstruct call their existing scalar
  kernels; marker shifting preserves the exact descending four-lane/remainder
  loop and clear preserves the ascending conditional-fill loop. Parallel
  branches are unchanged. No auxiliary pool, environment mutation or unsafe
  replacement was added. These branches also cover small-work serial fallback
  when the caller requests more than one worker.
- Rust1.89 audit: the package declares no upstream MSRV; edition2021/default-feature
  source shows no nightly feature or identified post-1.89 requirement. Locked
  Rayon1.12 declares rust-version1.80. Parent subsequently reports the
  **Rust1.89 all-target check passed**, closing the compatibility-check gap.
  This does not claim tests were executed under Rust1.89.
  Only the integer serial API is intended, not 8-bit/BWT/OMP/ARM work. This
  audit is not a full soundness proof of the utility's internal unsafe code.

The vendor import retains the package's four Rust source files and unchanged
Apache LICENSE/README. `vendor/libsais-rs/PROVENANCE.md` records registry package
checksum, packaged VCS revision, original file hashes, local changes and audit
limits. Its Rust-only manifest retains Rayon as its sole direct dependency,
sets build=false, removes cc/C build/example targets and the selectable
upstream-c feature, and keeps the dependency outside parent workspace targets.
No C compiler or bindings are involved. The other three Rust modules are
unchanged; only the 32-bit integer serial API is integrated/audited here.

`lz77/source/sais.rs` is registered by the owned source module and called by
`SuffixIndex::build`; the radix implementation is gone. The adapter copies
input, validates the source n/alphabet limits, allocates exactly n+6*k i32
suffix slots, and returns None for nonzero backend status or invalid returned
indices so the caller can take its existing static-index fallback. New memory
reservations are fallible; the utility's internal bounded Vec allocations are
not Result-returning APIs. The conservative budget is not a recoverable-OOM
guarantee for that dependency.

Five new tests cover checked layout/budget boundaries, status/mapping failures,
recursive/wide scalar suffix-order comparisons up to 131073 tokens, all five
helper branches against scalar kernels/independent marker order, and an
isolated unit-test process that verifies no global Rayon pool was initialized.
That isolated test calls the helpers and small/large serial suffix builds, then
deliberately initializes one global thread as evidence; this is test-only,
not an encoder workaround. Existing exhaustive three-symbol inputs through
length seven, blocked LCP/neighbor oracles, all four suffix-mode command-decode
tests and zero-budget static fallback now exercise the new adapter.

Parent wiring, with no new SourceConfig fields or forwarding changes:

```toml
[patch.crates-io]
libsais-rs = { path = "vendor/libsais-rs" }
```

The existing exact =0.2.2, default-features=false dependency is retained. Parent
reports the root override is added and cargo update --offline resolved the
lockfile to the vendor copy. Root manifest/lock changes and central tests are
parent-owned, not edits or commands by this worker.
The prior release/corpus checkpoints do not verify this SA-IS change. Local
verification is limited to scoped formatting and read-only source/hash checks;
no Cargo/build/test/benchmark was run by this worker. Parent's first workspace
run with the still-unpatched registry dependency reported 1303 passing core
tests and only the expected isolated-global-pool failure. Parent's subsequent
vendor-backed core run (2026-10-23) compiled and passed all SA-IS-focused tests,
including global-pool isolation, on the current default Rust toolchain. Parent
reports 1316 tests in that run and 1304 passed overall; its only failure was an
unrelated new FJXL test owned by another worker. This was not an entirely
passing core-suite checkpoint. Parent subsequently reports all three
vendor-backed release source-LZ suites passed: 172 independent pixel comparisons
including suffix routes. Strict Clippy and the Rust1.89 all-target check also
passed, so the required MSRV compatibility check is no longer pending. SA-IS
code remains frozen; no new LZ features are part of this checkpoint.

## Compact DP Slice

The new field is `SourceConfig::compact_steps: bool`, default false. It maps
source COMPACT=1 for ordinary chain Optc256. Parent-owned options only need to
forward that field; no generic module registration or executor changes are
required. To exercise this explicit route, select Optc256, leave
`bucket_matcher=false`, `bucket_mode=Default` and `index_mode=Disabled`, and
set `compact_steps=true`. Modes 0..3 retain their existing cost/refit behavior.

The parameter-selected default short32 route (`bucket_matcher=true`) now
automatically uses compact endpoint nodes, as does the eligible indexed
wide-ring route representing source's compact parallel-wide instantiation.
The explicit chain switch does not override history, wide-key, ordinary
wide-ring, precise-ring or short128 choices. Optc8 and the fast methods ignore
it. Modes 4..6, plus Auto when it selects Range, use the existing heap parser;
successful block dispatch uses its existing block parser. Neither consumes CompactStep. This
does not change the existing Rust explicit bucket-override precedence.

The source layout is three fields in order: u32 length, u32 distance symbol
plus one (zero is a literal), and f32 total cost. It occupies 12 bytes with
4-byte alignment, rather than the existing endpoint node's 16 bytes. The
compact vector replaces that vector; both are not allocated together. Costs
are already rounded by the source caller's f32 additions/subtractions before
storage; a debug assertion guards against silently feeding f64 precise costs
into this path. Unreached nodes use source f32::MAX, and node zero costs zero.
Backtracking recovers context from the input origin and converts coded
distances back to the public physical-distance command contract.

Only endpoint representation changed. The existing exact interval tree,
earliest-origin tie handling, query/skip logic, pricing, refit and acceptance
remain unchanged. Tree offers and cumulative arrays retain their existing
representations; this slice does not claim to reproduce every C++ scratch
layout or its environment worker estimates. Compact allocation checks the
signed source input domain before allocation, uses checked n+1 arithmetic and
fallible reservation, and adds only a bounded 120-symbol canonical distance
map. Existing conservative worker bounds and execution policy are unchanged.

Seven focused tests cover node layout/sentinels/limits, biased and aliased
distance codes, wide/compact/independent scalar endpoint comparisons at
16-token tree and f32 rounding boundaries, independent command decoding,
default/method/bucket/range/precision gates, chain refit and acceptance, and
indexed endpoint versus block dispatch. The existing scalar source parser
oracle also now runs the explicit compact chain route. The private-interface
warning and ambiguous integer test range reported during parent compilation
were corrected. Parent subsequently reported the compact slices passed the
1145-core / 180-official-check checkpoint. No Cargo command was run here;
that prior checkpoint does not verify the new block-parser/distance batch.

## Compact Value Slice

The preceding slice added two typed `SourceConfig` booleans, both default false:
`index_compact_values` and `index_coordinates`. Existing `index_word_match`
now selects the source u16 specialization when the index is packed. These
are wired into the owned source parser; no generic registration or executor
changes are needed. Parent-owned option/policy surfaces must forward the two
new fields to expose them externally. No environment variables are read.

Frozen-source constructor boundaries were checked in `enc_lz77.cc`:

| Route | Value representation | WORD_MATCH | INDEX_COORDINATES |
| --- | --- | --- | --- |
| Ordinary indexed blocks, lines 1897-1924 | u16 iff COMPACT_VALUES=1 and **all** values <=65535, else u32 | Requested option | Requested option |
| Non-block indexed batches, lines 1962-1968 | Always u32 | Off | Requested option |
| Shared indexed seed and retained first block pass, lines 1605-1650 | Automatically u16 when all values fit, independently of COMPACT_VALUES | Off | Off |
| Successful suffix route, lines 1877-1889 | Its existing u32 suffix index | Off | Off |

A failed/ineligible suffix construction reaches the ordinary static block
builder, so both new options apply there. Source eligibility (Optc256,
short32/non-history index, Frontier, size/pool/SMALL_INDEX gates) and the
existing Rust uint000 block gate remain unchanged. Output acceptance, match
contexts, candidate order, strict physical window bound and destination caps
are unchanged. No value, context, slot or distance is truncated to fit u16.

`COMPACT` is a different source control. At `enc_lz77.cc:1777`, `CompactStep`
omits stored context and backtracking uses `in[pos - len].context`. In source
dispatch at lines 2450-2492, the parameter-selected bucket matcher and
`parallel32`/`parallel-wide` instantiate compact nodes automatically, while
plain `32`, `wide-ring`, history, short128 and precise-ring return through
their own noncompact instantiations. Only after those returns does explicit
COMPACT=1 select compact **chain** Optc256. The indexed gate itself requires
the compact short32 instantiation. The DP slice above now implements that
node layout and the explicit chain switch. The existing Rust explicit
bucket/index API still does not reproduce every alias/parameter combination.

The packed value buffer has 2*n bytes of elements, and coordinates add 2*n
bytes. Index construction now explicitly rejects n > 2^24, matching source
dispatch bounds; both buffers use checked fallible reservations and each is
at most 32 MiB of elements. They are shared across block workers, not cloned
per query. Slots, positions, zero lengths and deep-chain entries remain u32;
coordinates retain the offset arrays as source does. The existing Rust seed
pipeline still retains its u32 value vectors: packing adds a separate u16
query buffer rather than eliminating that prior storage. Therefore this is
not a claim of whole-pipeline memory reduction or the source's total peak
allocation layout. No new pool, dependency, unsafe code or full-match SIMD
engine was introduced; the word path uses native Rust u64 operations.

Seven focused tests were added in `lz77/source/index/compact_tests.rs`: scalar word
comparison at every bit/lane/alignment/tail and overlap; exact narrowing and
wide values at every position (including 65535, 65536 and u32::MAX); candidate
sequence/cache-prefix equivalence across tags, run keys, short/long rings and
deep caps; exact coordinate bytes and shared-run/tail counts; independently
decoded endpoint/interval/lazy block commands on narrow and wide streams;
non-block/suffix/fallback/default gates; and automatic shared seed packing,
wide-seed comparison and retained-index replay. The existing installed-pool
test also compares packed/coordinate serial and parallel index layouts.
These are scalar/token-command oracles, not an independent JXL decoder run.
Parent subsequently reported the compact slices passed the 1145-core /
180-official-check checkpoint. No Cargo command was run by this worker.

## Earlier Slice Integration

Earlier slices exposed seven `SourceConfig` booleans, all false by default:
`index_word_match`, `block_sparse_cache`, `block_shared_frontier`,
`block_frontier_carry`, `block_distance_bound`, `block_distance_table`, and
`block_distance_classes`. These mirror the source opt-in semantics rather than
silently enabling experiments. Existing eligible INDEXED + BLOCK_DP dispatch
and uint000 restrictions still apply. `shared_index` reuse also passes the
frontier/cache/price controls, but does not retrofit word matching into its
source-default builder. The guided/lazy slice wired the hint hook to actual seed
spans and added the independently selected lazy block route:

- `block_seed_guided: SourceSeedGuided::{Disabled, Hints, SparseOrigins}`
  maps the disabled/SEED_GUIDED=1/2 choices; default is `Disabled`.
- `block_lazy: bool` maps LAZY=1; default is false. It overrides either endpoint
  or interval block parsing after the existing block/index eligibility checks.
  It has no effect on non-block source routes or the outer acceptance contract.
- Static, suffix and retained shared-index block routes receive these settings.
  Guided DP uses ordinary matcher verification, not an assumed seed match.
  Lazy parsing deliberately passes no hint, result, carry or distance bound.
- Source LAZY looks up dense-table/direct prices, ignoring `DISTANCE_CLASSES`
  even when the classes were built. The guided DP branch retains classes-first
  lookup. Neither option enables a new executor or alters scheduler policy.

The previous bounded pruning slice added `block_zero_tail: bool` and
`block_literal_bound: bool`, both false by default. They use the existing
static/suffix/shared-index block dispatch and uint000 gates. Non-block routes,
destination limits, physical distances and outer acceptance remain unchanged.
No calendar/vector/parser variants were added. Lazy retains its own prefix
skip and gain logic, without consuming either new DP pruning control.

Literal-bound numerical details follow `enc_lz77_block.h` exactly: prefix sums
are `previous + double(Bits) + nbits`; the upper bound is
`double(base) + prefix[local + end_length - 1] - prefix[local]`. The factor
`(end_length + 2) * float_epsilon` rounds in f32 before promotion and
multiplication by `max(1.0, upper)` in f64. `end_length` is the exclusive length
endpoint, not the interval width. A float match cost is promoted for the strict
`cost > upper + margin` test; equality and near ties remain in normal parsing.

Numerical source detail: in frozen `enc_lz77_block.h`, lazy parsing passes the
integer literal `0` as `VisitLZ77CostIntervals`'s `prefix_cost`. The template in
`enc_lz77_cost.h` therefore deduces `Cost=int`, truncates its float
length-plus-distance price into that integer, and promotes it to the callback's
float `bits`. `lazy_price` preserves that conversion for the supported finite
model prices. Literal prefixes use two left-associative double additions
(`previous + nbits + Bits`), not a pre-summed f32 price. This intentional direct
port is not silently replaced with a float-cost lazy approximation.

Scratch stays bounded: dense endpoint caches are 4*window bytes, sparse caches
at most 32768 bytes per task (the source worker estimate is always 32768), and
carry/frontier certificates have 16 entries. The shared distance table is at
most 4*(2^20+120) bytes and the classes occupy 128 floats; neither is allocated
per task. Source block worker estimates and the existing additional Rust
parser-scratch bound remain in place. No new executor or dependency is used.
Guidance adds one shared span per seed copy (three usize fields); valid copies
cover at least three tokens, so storage is bounded by the indexed stream size.
Lazy parsing allocates one double prefix per block, within the existing
conservative per-worker block scratch estimate, and does not allocate a DP
parser. Seed spans are built once even when lazy takes precedence, as in source.
Literal-bound DP adds 8*(block_tokens+1) bytes for its optional prefix, within
the existing conservative block worker estimate; the zero-tail scan allocates
nothing. Neither changes worker/scheduler code.

Entropy/API policy modules have not been changed to select the optional flags.
Parent reports that the previous slice's `tests/reference.rs` fixtures passed
32 independent pixel checks: eight modes at 8/16-bit depth through official
0.12 and 0.13 decoders. Parent has extended that fixture to 13 routes and
52 independent pixel checks, including static/suffix zero-tail and
literal-bound routes; parent reports all 52 official pixel comparisons passed
in debug. This worker has not edited the reference fixtures or run decoder
acceptance.

## Verification Handoff

Current parent-reported checkpoint, 2026-10-23: five debug stream-arena tests
and one logical null-seed test passed. The new
`indexed_lz_stream_arena_pixels_match_official_djxl` integration passed all
three cases across two official decoders (six pixel checks); the four other
integration tests also passed. The six parallel small-probe helper tests passed
in release, and the explicit-LZ-option projection regression passed. These
results supersede the initial pending status of the arena/helper slices, but
do not claim full migration completion or whole-JXL byte identity.

Historical compact-storage checkpoint follows:

Parent verification, 2026-10-23: compact values, coordinates, shared seed
indexing and compact DP passed the full debug workspace and the expanded
17-route official-decoder fixture. Two depths (8/16) and two decoders
(0.12/0.13) produced 68 exact-pixel checks. Strict Clippy passed after a
test-only redundant-Vec correction. These results do not cover unported
optional routes and are not a whole-JXL identity requirement.

- Added scalar endpoint-oracle coverage for precise-ring and short128;
  independent checks for f64 preservation and f32 model-price promotion.
- Added short128 wrap/capacity/tag-bypass tests and an evicted-short-copy case.
- Added precomputed SIMD/scalar index candidate-order checks including shared
  zero runs and incomplete tails.
- Added suffix-order exhaustive small-alphabet checks, active-rank word/level
  boundaries, all-pair blocked-LCP comparisons, window-neighbor oracles,
  precomputed/nonprecomputed candidate checks and scratch/alphabet fallback.
- Added block-cut/trim/cross-block-reference, seed split pool/dominance gates,
  indexed seed positive-gain/double-prefix, shared-index dispatch and suffix
  dispatch replay tests. The full existing mode4..7/acceptance tests remain.
- Previous slice added 11 focused tests: word localization at every bit/tail,
  dense/sparse endpoint collisions and saturation, uint420 scalar price oracles,
  price-cache precedence, carry ordering/expiration, shared-query reset and
  bound exits, verified hints and cached candidate order, suffix bound bypass,
  overflow replay without the bound, independent command decoding across block
  cuts for both indexes, and integrated static/suffix dispatch replay. These are
  token-command decoders, not an independent JXL wire decoder.
- The guided/lazy slice added 10 tests: span coverage/physical distances,
  exact short/long repair boundaries, nonzero block-start query traces, an
  evicted seed copy nominated before buckets, lazy pending lookahead and option
  bypasses, fractional integer-price semantics, zero/positive gain and tie
  rules, zero-price suffixes, an independent all-length scalar choice oracle,
  cross-cut command decoding with both index types and all three cache modes,
  plus public source dispatch/default-off/non-block preservation checks.
  Existing overflow and price-precedence tests now also check hint retention
  and lazy's dense/direct lookup respectively. Tests live in owned LZ files.
- The previous pruning slice added 8 tests: exact-zero/extra-bit/block-local gates,
  double prefix accumulation, strict and near-tie inequalities, source margin
  rounding and cancellation order, earlier copies ending inside skipped tails,
  a scalar unpruned endpoint oracle without frontier/interval/heap helpers,
  preservation of repetition skips even when all intervals are pruned, lazy
  query/output invariance, and static/suffix public dispatch against unpruned
  results. Independent command decoding checks contexts, samples, physical
  distances and destination caps. Both endpoint and heap block modes are covered.
- **No Cargo builds or tests were run by this worker.** Parent owns central
  compilation/testing. Parent reported the pruning whole-workspace checkpoint
  has **938 core tests passed, 2 ignored**, including the pruning files, and
  strict Clippy passed at the previous checkpoint. The earlier 32 official
  pixel checks passed; parent also reports the expanded 13-route/52-check
  official pixel fixture passed in debug. These results do not establish
  complete LZ tool coverage. Do not compare whole compressed JXL hashes.
- A targeted `rustfmt --check --edition 2024 --config skip_children=true`
  passed for the seven Rust files changed in the compact-value slice. It checks parsing and
  formatting, not type checking or behavior; no repository-wide formatting was
  performed.
- The compact-DP slice has a separate targeted format check for its five Rust
  files. At that handoff, the prior checkpoint was 1145 core and 180 official
  checks passed; those historical results predated the block/distance batch.
- No commits, dependency edits, old-source writes, entropy edits, or
  `tools/mod.rs` edits were made.
