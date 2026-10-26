# Pixel Scheduler Port

Source baseline: libjxl `b866b3defec13494bfbaf705db93581cda118e37`.
Translated portions retain JPEG XL Project Authors attribution and upstream
BSD-3-Clause licensing. Ownership covers this directory and the explicitly
delegated `../modular/tree.rs`. Crate-root registration, builds, tests,
validation and encoder activation belong to main.

## Integration API

Current cooperative follow-up: native batches now use the source descendant-first
arena, including bounded roots, nested same-pool jobs and quiet notifications.
The public `probe_lossless_candidates_parallel` helper exposes the native live
prefix cutoff to `Fn + Sync` callers; the legacy `FnMut` batched API intentionally
retains completed-wave semantics. Candidate projection preserves explicit base
LZ tool controls while replacing candidate and wire-format policy fields.
See `docs/PARALLEL_EXECUTION.md` and `docs/SCHEDULER_PROBE_API.md` at repository
root. Earlier implementation checkpoints below describe their historical state.

Current resource policy supersedes the old 16384-leaf checkpoint below:
`MAX_LEAVES` is the source-format bound (`MAX_TREE_NODES / 2`), further
intersected with global/local decoder geometry limits. Caller ceilings remain
explicit. See `docs/RESOURCE_LIMIT_AUDIT.md` and `docs/MA_TREE_BUDGET.md` for the
removal of migration-era caps; do not restore an old numeric limit from this log.

- `scan_lossless_input(&Image, ScanOptions) -> LosslessScan`: complete scalar
  coverage, bounded scratch, optional bounded RGB palette lower bound.
- `scan_lossless_input_with_pool(&Image, ScanOptions, &rayon::ThreadPool)`:
  source tile-row dispatch with identical plane ordering and bounded scratch.
- `validate_source_input(&Image) -> Result<()>`: source search rejects float
  and >16-bit color or extra channels. U32 storage with <=16-bit declared depth
  has no source input format mapping in this port and is explicitly rejected.
  Automatic source search checks before any encoder callback. Explicit native
  candidates use separate metadata validation and can encode higher-depth and
  floating samples; search ineligibility is not an encoder input restriction.
  The legacy non-Result scan returns unavailable, and literal evidence false,
  on unsupported input; use this validator to retrieve the reason.
- `scan::literal_evidence(&Image) -> bool`: fixed 1024-observation test without
  full coverage work. Does not include the search's <=65536-pixel exception.
- `lossless_probe_plan(&CandidateParams, PlanOptions) -> Vec<Candidate>`:
  source MakePlan order, 15 families or 18 with explicit extended RCT screening.
- `CandidateScreen::new(image, scan, options, callback)`, `score`, `rank`,
  `smaller_groups`, `analyze`: callback type is
  `FnMut(&CandidateParams, &Image) -> crate::Result<usize>`.
- `small_probe::probe_lossless_candidates`: independent nine-tile screen,
  nonnegative prefix bounds, six-candidate shortlist, larger validation crop.
- `small_probe::confirm_lossless_candidate`: independent 512-side witnesses.
  Cached incumbent size means those exact witnesses, never full-frame bytes.
- Both helpers have `_batched` variants accepting ordered `ProbeTask` slices.
  Their legacy FnMut fallback retains tile waves. Native supplies the additional
  `BatchEncoders::prefix` callback: complete candidate screen/shortlist tasks
  run independently, with an atomic top-six screen cutoff and fixed shortlist
  control bound. Only the screen exempts Variable predictors from bounding.
  Confirmation batches contain at most four cells. Malformed or zero sizes,
  incomplete results and work-limit failures abort without ranking a score.
- `run_pixel_search(image, params, options, encode_full, encode_probe)`:
  executable source stage loop, including the default large-frame path.
  Returns `SearchResult<A> { bytes, params, artifact, report }`. It owns and
  retains the winning whole output and artifact locally; it never re-encodes.
- `run_pixel_search_with_batches(..., BatchEncoders)` preserves that API's
  decisions while adding ordered full/probe callbacks and an optional scan_pool.
  `batch::run_ordered(pool, count, worker_slots, task)` uses only the supplied
  pool, a dynamic queue and input-order results. Native bridge supplies these
  hooks and creates one pool, not one per job. Single trials use encode_with_pool;
  independent batch jobs use the core null-runner encode_serial path. Optional
  cooperative batches reuse the same pool, never a nested private/global pool. Probe
  budgets are cumulative across cached screening, legacy probes and confirmation.
- `search::Search<A>`: reusable completed/bounded trial ledger, family and
  calibration lookup, baseline-only eligibility and stable output retention.
- `policy::PixelSearch`: legacy score-only planning helpers. Eligible large
  frames now return `SearchAction::ExecuteLargeFrameDriver` instead of an
  error. This is NOT permission to skip a stage: run the executable driver
  from stage 1, which owns the required output/cache/feedback state.
- `adapter::project_native(params, image, base) -> Result<NativeProjection>`:
  cloned base EncodeOptions with supported source choices applied.
  `state()` is SourceEquivalent, NativeProxy or Unavailable. The projection
  retains detailed gaps, resolved LZ method, training-policy metadata and
  `resources: NativeResourceLimits { max_tree_leaves, sample_limit }`.
  `exact_options()` refuses every projection with gaps. No automatic activation.
- `adapter::project_native_with_resources(params, image, base, policy)`:
  `NativeResourcePolicy { max_tree_leaves: Option<usize> }` supplies an explicit
  adaptive learning ceiling. None selects native MAX_LEAVES, never base's
  direct-encoder preset. sample_limit remains inherited from base.
- `adapter::source_patch_reference_policy(params, image, base, resources)`:
  returns `modular::source_options::SourcePatchReferencePolicy`. It projects
  ORIGINAL candidate inputs with Gradient and patches Off before choosing LZ/WP,
  retaining the raw fraction and optional group-size sentinel. No callback.
  `policy.resolve(&atlas)` returns `SourcePatchReference { modular,
  group_size_shift }` after recomputing geometry/MA properties for the atlas.
  `adapter::project_source_patch_reference` is the direct actual-atlas entry.
  Both reference paths read only geometry/channel metadata, never external
  PixelData or extra_planes; signed private working atlases are supported.
  The adapter fills EncodeOptions.source_patch_reference only when the source
  Add gate is active. Parent encode_frames resolves it on the actual atlas,
  selects shared/global, nonresponsive one-pass framing and disables recursion.
- `native::encode_native_candidate`: concrete encode of an explicit candidate,
  returning bytes and NativeArtifact (actual options, effective params, gaps,
  resource limits). Each NativeAdaptation also reports its resource limits.
- `CandidateParams::responsive`: explicit source whole-frame squeeze, distinct
  from local `squeeze` directions. Selects source responsive predictor/RCT/tree
  policy and nonstreaming framing, with one spatial pass. Tier 9 falls back to
  tier 8 before Modular, as the source frame path does. Atlas policy clears
  responsive before resolving its own tools. Nonempty local `squeeze` scripts
  are still unavailable as source candidates; direct Modular use is separate.
- `run_native_pixel_search`: concrete bridge around the generic driver. Returns
  `NativeSearchResult { output: SearchResult<NativeArtifact>, adaptations }`.
  Strict acceptance is the default. Full native patches-off tests opt in with:

```rust,ignore
let result = scheduler::run_native_pixel_search(
    image, &params, &DriverOptions::default(), &base,
    NativeBackendOptions {
        acceptance: NativeAcceptance::AllowReportedGaps,
        patches_off: true,
        ..Default::default()
    },
)?;
// result.output.bytes: retained encoded output
// result.output.artifact.options: options of that exact invocation
// result.output.report and result.adaptations: source and native decisions
```

The patches-off bridge records ForcedPatchesOff whenever the requested source
candidate was not already Off. It does not conceal an automatic patch decision
or replace requested parameters in the source ledger. Probe and full callbacks
share this explicit adaptation. `encode_native_candidate` can instead receive
an explicit Off candidate without rewriting it. No default encoder path changes.
NativeBackendOptions.resources carries the explicit resource policy for every
full and probe callback. A hard resource failure aborts, never silently changes
the sampling fraction, truncates a beneficial tree search, or yields a score.
Adapter clears base.modular.custom_tree before applying each candidate's fixed
or adaptive tree policy; source automatic search excludes custom_fixed_tree.
It also clears inherited tree_groups and tree_group_splits together before
validation. MA shard/split experiments are not source-default candidate inputs
or implicit resource overrides; no scheduler extension currently enables them.

Callbacks must not recurse into the scheduler. Source parameters must either
be implemented or rejected. Explicit native-proxy experiments may consume
`NativeProjection.options` only while recording its gaps. Their results are
native-proxy scores, not exact source-candidate scores. Never mix score caches
from different backends, proxy policies, source images or metadata.

The generic callbacks are:

```rust,ignore
encode_full: FnMut(&CandidateParams, &Image, TrialBudget)
    -> Result<TrialOutput<A>>
encode_probe: FnMut(&CandidateParams, &Image) -> Result<usize>
```

`TrialOutput::Complete(EncodedOutput { bytes, artifact })` is a complete encoded
frame. `A` may be the resolved native options and projection gaps, or a final
plan captured by that exact invocation. `TrialOutput::Bounded { lower_bound }`
is allowed only when real output bytes or a proven encoder byte lower bound
reach `TrialBudget.beat_bytes`. Baselines and calibrations have no beat bound
and must complete. Native encode's generic resource-limit error is NOT such a
proof. An encoder without safe incremental byte bounds can always return a
complete output. Probe callbacks return completed independent sample bytes.

Only completed whole-frame outputs replace the winner. Crop sizes and
calibration estimates remain screening evidence, never winners or byte bounds.
Legacy callbacks run serially. Batch callbacks return results in input order;
initial full batches commit in source order, and failed probe batches cache no
cells. Errors abort the driver without returning a retained output;
attempts count toward explicit work ceilings even when a callback fails.
Same-size ties retain plan order, except an earlier source stage wins ties.

`DriverOptions` represents source switches explicitly, without reading process
environment. Defaults select the ordinary source route; this module does not
activate it in the application. Optional supported switches include coverage
scan, fast preflight, family beam, confirmation, probe-feedback bounds,
CALIBRATE/FAST/REFINEMENTS/ROUTES, FAST_FAMILY and PALETTE_RACE modes 1/2.
`sibling_race` maps JXL_LOSSLESS_SIBLING_RACE=1: before the regional gate,
stages 4..9 compare e4 whole-frame witnesses (fraction min .01) for otherwise
identical raw-variable RCT/group siblings. A proposal >control*1.0025 is
deferred, not discarded. Stage 10 replays the original candidate without crop
screening. Events report both measured sizes; they are never output bounds.
`preflight` now represents PREFLIGHT_YIELD, anchor-first, scope routing,
multiscale ranking, calibration reuse/order, early-family, plan-first/buckets,
geometry-first, automatic-yield and context feedback. Nested source gates are
preserved: enabling an inner flag never enables its required outer route.
`statistical_feedback` and `sibling_transfer` feed the refinement decision.
`frontier` selects opt-in direct/compact/calibrated/extended/shrink/dense,
global or strip tournaments. These experiments do not promise effort-prefix
dominance. Full measured results, including calibration witnesses, compete for
the retained output; crop projections cannot replace it. `workers` bounds
independent root tasks to 1..64 and caps them by a scratch estimate (default
8 tasks, 1 GiB). A single oversized task may still run; this is not an RSS cap.
Raw context/full-width ScreenOptions overrides are rejected by the driver:
select the corresponding driver route or use the standalone probe API.
`SearchReport` includes source commit, completed/bounded trials, selected
index, work counters, screen/calibration/replay events and unported routes.
BackendSemantics defaults to Unverified; SourceContract is caller attestation,
not scheduler certification. NativeProxy adapters must be declared as proxies.

## Tree Entropy API

`Tree::write_with_options(&mut BitWriter, &EntropyOptions,
Option<&tools::lz77::SourceConfig>) -> Result<()>` implements the fresh-model
tree branch of EncodeGlobalInfo/ModularGenericCompress. It preserves the six
tree contexts; source LZ adds distance context 6 with width zero. It disables
the residual-only bucket override, retains the source method/careful-first-pass,
checks length/min-symbol/literal cost configuration against entropy, and uses
Coding::Source with the supplied ForModular policy. Fixed(cfg) supplies the
literal cost config verbatim (including 000); Fast/Best use default 420.
Fresh streaming is accepted: Fast/Best skip uint search using 420, Fixed is
retained, and ANS uses log 8 (prefix log 15). Each call creates a new model;
cross-call model reuse and missing-symbol injection have no API here.
The source parser's explicit enabled result controls both the LZ header and
seventh context, including accepted output with no Match commands.

Legacy `Tree::write` remains Prefix without LZ. Parent's `encode::write_tree`
selects the new API when `ModularOptions.source_tree_entropy` is true and passes
None when LZ is off. Local encode_prepared and both modular/frame.rs write sites
now use that helper. Adapter sets the flag; TreeEntropyPolicy is no longer
emitted. This closes the integration gap, not independent numerical validation.

`Tree::nodes() -> &[Node]` and `Tree::into_nodes() -> Vec<Node>` expose source
node order for group-tree merging. Tree::new still validates/rebuilds context
IDs breadth-first; no context semantics were changed by these accessors.

## Ported

- `enc_lossless_scan.h/.cc`: byte-equivalent typed-sample equality including
  invisible RGB and all full-resolution extra planes; FNV evidence; tile peaks;
  64-to-64-axis coverage; original activity-stratum and separation decisions.
  RGB lower bound ignores alpha as upstream does; it never proves RGBA fits.
  Scope counting is enabled only when buffer depth equals declared color depth.
- `enc_lossless_probe.h/.cc`: source candidate identity; explicit experimental
  RCT switch; cache normalization; cheap versus exact keys; hard/median prefix;
  representation retention; all-region final scores; original thresholds;
  smaller-group checks; Promising; original crop/strip/atlas construction;
  small probes and confirmation; calibration agreement, shrinkage, sibling
  estimates and diverse representation frontier. No source downsampling.
- Crops preserve all currently represented ImageMetadata, ICC/color semantics,
  orientation, declared depth, interleaved alpha and full-resolution extra planes.
  U32/F16/F32 are copied without conversion, preserving 31-bit integers, raw
  half bits, signed zero and NaN payloads, even though source search is ineligible.
  Probe eligibility rejects >8 extras and subsampled extras, as source does.
- `enc_lossless_search.cc`: initial e3 anchor, literal e6 gate, tiny-image e9
  and six fixed-predictor trials, literal/regular fallback stage decisions,
  baseline-only competitiveness, literal learning gains, measured-winner
  neighborhoods, ambiguous-family exploration and deep-winner parameters;
  retained-output Search, real-byte challenger bounds, related-family feedback,
  default shared-screen early racing with fixed frontier and measured-winner
  refinement, baseline eligibility, optional full-frame e4/e6/e7 calibrations,
  palette preselection/racing, whole-frame sibling race and deferred
  original-candidate replay at e10.
- `enc_modular.cc`, `enc_ans.cc`, `enc_frame.cc`, `dec_modular.h`: actual RCT
  order/count, predictor defaults, training fractions/property order/thresholds,
  weighted-preset counts, LZ method decisions and automatic group geometry.
  These are source contracts, not a fabricated effort-to-tree-size table.
- `enc_palette.cc`: explicit exact-delta and zero-delta settings 1 through 6.
  Lossless automatic search does NOT introduce squeeze: upstream defaults
  responsive to zero. Source global/local palette policy now receives both
  channel-compaction percentages, tier and exact delta/zero mode. Native source
  orchestration applies global palettes before explicit RCT, then local
  palettes before automatic RCT. Legacy all-channel palette fields are cleared.

## Coverage Boundaries

- The additional workflows above are now connected, with deterministic policy
  tests for gate dependencies, calibration order, context vetoes, paired
  evidence, cumulative work, stable ties and actual output retention. Their
  11 official-decoder suites now pass with djxl 0.12 and 0.13, including compact
  frontier modes, global/strip tournaments, cooperative finalists and the full
  effort-11 tournament on small input. This is not complete performance parity.
- LPT dispatch, cooperative probes/finalists and conditional probe data
  parallelism now share the caller's Rayon pool. Explicit probe-worker/root and
  scratch limits apply. Null jobs suppress inner parallelism, including entropy
  config search and clustering; thread-ID tests exercise that contract.
  C++ arena descendant priority/quantum/quiet-wakeup behavior and automatic
  process-environment parsing are not reproduced. Default small probes,
  cached-screen and initial full batches are wired, as is opt-in confirmation.
  Native screen/shortlist candidates run independently through bounded null
  roots; completed screen totals tighten the shared cutoff asynchronously.
  Legacy callbacks retain waves. Native full-frame challengers now use
  finalized-section byte bounds through encode::budget; resource errors remain
  distinct from proven bounded scores.
  No global pool is created. Limits bound sample residency,
  attempted encodes and total encoded sample pixels. Hard limit errors never
  expose partial scores as completed totals.
- Source coverage uses constant-width memcmp specialization, not a Highway
  SIMD kernel. Rust currently uses typed-slice equality; this remains a
  constant-width specialization/performance difference, not an omitted source
  SIMD algorithm. Source tile-row parallel dispatch is now wired
  when a pool is provided. Full coverage is O(input samples), not
  constant-time; only literal evidence and probe work have fixed ceilings.
- Current native backend source-policy differences are reported per projection:
  source Add patches now receive the actual outer gate (tier<6, <=16 bits,
  decoding tier zero, non-groupwise color); On does not bypass that gate.
  Off clears both patch implementations, and no candidate inherits legacy
  Replace patches. Source detector defaults (fast, peak 2, fingerprint 1) are
  explicit. The detector now uses source f32 normalization, subtraction before
  quantization, ordered weighted accumulation and mixed float/double tolerances;
  PatchDetectorIntegerAdaptation is no longer emitted. This is policy coverage,
  not a cross-compiler bit-identity assertion. Dictionary source RLE/default HistogramParams and the
  deferred reference recipe are integrated; AutomaticPatchPolicy is no longer
  emitted. The reference selects Gradient-dependent LZ/WP before resolving
  auto groups, original buffering and raw sampling against atlas dimensions.
  SourceBuffering Auto/FullFrame/StreamingAbove2048/Streaming preserves source
  -1/0/1/2 semantics; CanDoStreamingEncoding is resolved before patch discovery.
  Eligible roots use the actual DC-frame-area orchestration, not merely
  shared_tree=false. Streaming trees/residuals get streaming entropy and
  independent models; source patches are disabled by the groupwise outer gate.
  Groupwise streaming has no global MA tree: local tree/residual entropy uses
  fresh models without missing-symbol injection. The source's explicit
  add_missing_symbols gate belongs to the shared global residual model, not
  every streaming entropy call. See DEFAULT_ROUTES.md for source locations.
  Forced patches On disables root streaming but does not force the patches-Off
  reference atlas to stay nonstreaming. Root and atlas sampling each use their
  own whole-frame geometry and original fraction. See DEFAULT_ROUTES.md.
  Complete search rejects buffering 1/2; explicit candidate encoding supports
  them. Inherited source_constant_channels is reset to its source default false.
  Central routing checkpoints are recorded in docs/PORT_VERIFICATION.md; zero projection
  gaps do not certify numerical or worker-scheduling equivalence. Explicit
  Add patches preserve mixed-depth and subsampled extra planes using None
  blending; their dedicated official-decoder test passes. Source reference
  policy still rejects subsampled atlas geometry where the frozen C++ recipe
  cannot express it, rather than claiming that native extension is identical.
  Floating color <=16 bits uses declared-format words and exact replay, not
  integer arithmetic on bit patterns. The dedicated floating-patch oracle
  passed both official decoder versions, including custom depths and mixed
  extras; it does not certify all possible float inputs or performance.
  Global/channel palette
  eligibility, palette cost rejection, local palette/RCT ordering and zero-delta
  insertion are now integrated and their former gaps are no longer emitted.
  Source RCT/WP cost searches, source sampling/quantile policy and entropy-tier
  choices are also actively wired. Old gap enum variants remain API-compatible.
- source_lz77 now drives the executable parser, including all shared token
  streams, exact method capacities, careful first pass and the bucket override.
  Both b1w3t and b3w3t are greedy source rings, not a lazy-mode substitution.
  SourceConfig receives the entropy policy's source000 length/min-symbol 224
  fields. The active serializer uses the same fields for histograms, alphabet
  selection, headers and tokens; the wire-alignment gap is closed. Alphabet
  search uses log 5..8 for ANS and log 15 for prefix, and Coding::Source selects
  source prefix/ANS policy rather than native Auto comparison. Unsupported LZ
  experiments are explicit low-level options, not selected implicitly by this
  scheduler's source-default parameter contract. History/wide rings, static
  indexes, bounded stream parallelism and match-cost modes 4..7 are implemented;
  suffix indexing and remaining index/precision optimizations are tracked separately.
- Tree/local/shared serializers now consume parse_source_with_state or its
  shared-stream variant, retaining ApplyLZ77 acceptance separately from Match
  commands. Patch dictionaries use source RLE's state API too. LzEnabledState
  is no longer emitted; its enum value is retained for report compatibility.
- Adapter overrides the direct encoder's 16-leaf preset with native MAX_LEAVES
  (16384), unless NativeResourcePolicy supplies an explicit ceiling. Source
  training::learn now enables require_complete_search; ma_train evaluates
  remaining nodes and errors if a beneficial split cannot fit. Consequently
  NativeTreeLeafLimit is no longer emitted as a capability gap. The resource
  ceiling is reported separately; it is not C++'s kMaxTreeSize=1<<22 assertion.
  sample_limit remains the caller's hard error ceiling, never a reduced fraction.
- Fixed speed tier 7 uses the source Weighted template; tier 8 uses Gradient,
  independently of explicit predictor and adaptive leaf budget. Their fixed
  tree gaps are closed. Tier 9 that reaches Modular falls back to tier 8;
  eligible separate FJXL dispatch is marked in NativeArtifact::implementation.
- Native source palette policy accepts caps through 70911 and percentages
  through 100. An exact-delta ATTEMPT still requires <=65536, 3/4 channels and
  <=16 bits. Source policy computes each attempt's cap and falls back to an
  absolute palette on decline; it does not reject a 70000 policy candidate.
- Native source responsive layout is now integrated by orchestration.rs. Its
  default GLOBAL script is distinct from CandidateParams.squeeze's explicit
  LOCAL passes. CandidateParams.responsive now selects the global contract;
  a nonempty LOCAL list remains explicitly Unavailable
  (ResponsiveTransformOrchestration). It must not silently substitute either
  script for the other. Automatic search emits neither and resets base responsive/passes to
  false/one pass. Channel/transform limits remain enforced by the encoder.
- JPEG coefficient search has its separate jpeg::transcode_search entry.
  Frame-aware animation/layer search is run_native_sequence_search. Ordinary
  pixel search keeps its own input eligibility; explicit native candidates
  have a broader sample/tool contract. Source preview input remains a TODO in
  the frozen public encoder rather than an implemented path to migrate here.
- HasLosslessSearchOptions restricts the ordinary driver to efforts 1..10.
  CandidateParams and the full native source chain now use signed speed tiers;
  explicit effort 11 maps to TectonicPlate=-1, never Glacier=0. Source Init's
  -1 MA switch takes DEFAULT: 3 base properties, 32 bins, fraction*0.3,
  threshold 61; RCT tries 19 and the automatic predictor is Variable. Source
  palette, entropy, LZ and patch-atlas projection preserve this tier. This is
  the explicit backend, not the separate frame-level effort-11 tournament.
  `run_native_tectonic` now implements the latter: two initial measurements,
  the selected 20/24-candidate round, and final re-encoding (23/27 encodes in
  total). Candidates use source tier 0 without recursively invoking search.
  CLI `--source-effort 11 --source-search` selects the tournament explicitly.
  Its tiny hidden-RGBA16 oracle passed both official decoder versions.

## Verification

See `DEFAULT_ROUTES.md` for the source-default versus opt-in audit and the
remaining default execution differences. This audit found no missing
default-enabled candidate-selection route; it does not claim full migration.

Focused tests live in `tests.rs` and `driver_tests.rs`, covering sample bits/hidden RGB, metadata and
extras, source constants, crop geometry, cache fidelity, budget checks, native
adaptation states, fixed-tree policy, retained invocation artifacts, true-byte
bounds, baseline/calibration separation, effort prefixes, large-frame routing,
work-limit failures, raw U32/F16/F32 crops, unsupported-input rejection before
callbacks, explicit resource projection, source Add gate, sibling race thresholds
and unconditional e10 replay. Tree tests cover source policies, raw LZ token
replay, context six/width zero, config mismatch before writing, legacy Prefix,
and context rebuilding through public node accessors. Driver test sizes are
deterministic policy oracles, not proof of codec equivalence. Central test and
official-decoder checkpoints are recorded in docs/PIXEL_LOSSLESS_PORT_STATUS.md;
later changes require revalidation. This directory is not evidence of full
pixel-lossless acceptance.
