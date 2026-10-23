# Asynchronous Candidate Prefix Screen

Source: immutable libjxl `b866b3defec13494bfbaf705db93581cda118e37`,
`lib/jxl/enc_lossless_probe.cc::ProbeLosslessCandidates`, screening task bodies
and `RunProbeTasks`. Translated code retains upstream BSD attribution.

## Implementation

- `scheduler/prefix_screen.rs` dispatches whole screen/shortlist candidates through existing
  `worker_pool::run_plan` independent roots. It creates no pool. Each root runs
  ordered screen tiles with private prefix state and null-runner encodes.
- A mutex protects the best six completed totals. An `AtomicUsize` publishes
  the monotonically tightening sixth-best cutoff while that mutex is held.
  Prefix rejection uses strict `>` before the next tile. Variable-predictor
  candidates always complete. Equal totals retain stable input-order ranking.
- `PrefixPass::Shortlist` starts at each completed screen score and visits the
  remaining tiles hardest-first. Its fixed `control_bytes * 1.10` threshold
  applies to every predictor, including Variable. A last-tile overshoot is a
  completed but rejected score, not an early bound. Empty remainders reuse the
  completed initial score with zero work.
- A bounded score is `None`, never its partial total. Callback errors, zero
  frame sizes, byte-sum overflow, and work exhaustion abort the entire result.
  Active invocations drain; later cells do not reserve work after failure.
- `PrefixWork` reserves encode count and pixels together immediately before
  each callback, including callbacks that fail. Pruned/cancelled cells are not
  charged. The driver narrows the local ceiling to the remaining cumulative
  scheduler budget, then accounts actual attempts even on errors.
- `BatchEncoders::prefix` is optional. Existing sequential and batched `FnMut`
  callers retain their APIs and prior scheduling. The native backend always
  supplies the asynchronous prefix callback; the source feedback flag still
  controls pruning and tile order, without being enabled implicitly.
- Native artifacts are prepared on the caller in candidate/tile order.
  Workers borrow immutable artifacts and atomic attempt flags, not `Rc` state.
  Attempted adaptation records are appended in canonical order after draining.
  With feedback enabled, the set of unnecessary attempts can depend on worker
  completion timing, as in C++; ordering of records is not completion order.

Controls and validation retain the existing batch path. Native shortlist
completion now uses the same bounded candidate-root executor as screening.
This change does not reproduce C++ arena descendant priority or quantum policy,
and does not claim byte-identical output or complete pixel-port acceptance.

## Verification Handoff

New `scheduler/prefix_screen_tests.rs` covers exhaustive finalist oracles,
stable ties, strict cutoff equality, Variable completion, feedback off,
multiple pool/root widths, a synchronized live-cutoff case, null-runner scopes,
encode/pixel ceilings, cancellation, backend errors, zero sizes, overflow,
actual savings against chunks-of-eight, large validation witnesses, and shared
cumulative driver budgets with both legacy callback variants.
`prefix_shortlist_tests.rs` adds fixed-bound/Variable checks, independent
shortlist progress, empty thin-frame remainders, invalid initial scores,
backend failure, and actual encode/pixel ceilings across the phase boundary.

The deterministic cheap-first-six fixture requires four fewer encodes than the
old eight-candidate waves on a one-thread pool. Tests assert actual callbacks
equal reported work and rerun with ceilings fitting only the actual work.

Parent reported 812 core + 14 acceptance tests passing for the screen slice.
The later parent checkpoint passed 903 core + 10 API + 14 acceptance + 35 CLI
tests, including the shortlist and finalized-section budget changes.
No cargo build or tests are run by this worker. Parent owns central unit,
integration/official decoder, and non-x86 verification. No source C++ changes
or commits are made.

## Native Whole-Frame Byte Bounds

Source: `enc_lossless_search.h::LosslessTrialBudget`,
`enc_lossless_search.cc::Search::RunPending`, and `enc_frame.cc::EncodeGroups`
at the same frozen revision. Native full-trial callbacks now pass the existing
`TrialBudget.beat_bytes` to private serial/pool encoder entrypoints. Baselines
and calibrations still have no bound. Generic callbacks and public encoding
APIs retain their existing signatures and `Error::Limit` contract.

`encode/budget.rs` returns an explicit private `TrialEncode::Bounded` outcome,
never an error sentinel. Only complete byte-aligned global/DC/AC payloads are
credited, exactly once at their producing task. These disjoint sections are
retained unchanged by TOC ordering and frame/container assembly, so their sum
is a conservative lower bound. Tree/model cost writers, selected-sample counts,
unused patch searches, memory capacities and timing are never evidence.

The resource ceiling remains `min(base.max_output_bytes, trial.max_output_bytes)`;
writers never receive a reduced competitive ceiling. Cancellation is enabled
only for `0 < beat_bytes < hard_max`. Checked totals and known retained headers
or previously completed reference frames enforce the hard ceiling first.
In-flight sections keep contributing after the cutoff: overflow, allocation,
sample-budget and output-limit errors remain errors even if another task has
already reached the bound. Pool dispatch drains active calls before returning.
No error in intentionally skipped work can be predicted.

The native source adapter selects the instrumented layout route. Shared-frame
encoding can stop after finalized global output, before remaining AC stream
serialization and final assembly; shared preparation, tree learning, tokens,
LZ77 and model construction have already happened. Source-aware groupwise
streaming has no shared global tree: per-stream checks avoid later local
tree/token/model/serialization work, and area checks avoid preparing subsequent
DC areas. The current area's preparation has already happened. Existing pool,
null-runner and token-parallel policies are unchanged.

Non-streaming one-group/one-pass collapsed frames remain complete, matching
the source exclusion. Legacy non-layout tool APIs also remain unbounded in
this slice. Completed patch reference/residual frames contribute once, without
recounting their assembled sections; uninstrumented frame bytes constrain only
resources. There is no post-full-encode conversion of a completed output into
a claimed saved-work bound. Public multiframe/JPEG cancellation is not exposed.
This is the native default trial route, not a new experimental switch. Optional
prefix feedback, confirmation, cooperative scheduling and palette/token tools
retain their existing explicit settings and are not enabled by this change.

New `encode/budget_tests.rs` covers exact cutoff/empty sections, disabled bounds,
checked arithmetic, resource-error provenance, in-flight error precedence,
retained overhead, real shared-tree sample-limit failure, patch-frame accounting,
baseline/calibration completion, ties, exhaustive winner retention and cumulative
whole-frame attempt/pixel ceilings. Task-entry counters assert that a global
bound starts zero AC serializers and an early streaming bound prepares only the
first of two DC areas. Native-adapted Auto/FullFrame fixtures exercise both routes.
Parent's 903-core checkpoint includes all 15 nonignored budget tests. This worker
does not run Cargo; parent owns central tests and independent decoding.

The subsequently added ignored regression
`encode::budget::tests::native_bounded_minimum_and_pixels_match_unbounded_official_djxl`
uses real source-adapted native candidates for FullFrame and Auto routes. It
requires an actual typed bounded challenger, compares retained minimum sizes
with exhaustive unbounded encodes, verifies every bound against the complete
candidate size, and decodes all complete outputs with each `RS_JXL_DJXL` oracle.
Only decoded samples and sizes are compared; JXL byte identity is not required.
This new oracle test is not included in the reported 903-core checkpoint and
awaits parent execution. The 56 previously passing release official tests also
predate the budget slice; they are not evidence of post-budget decoder acceptance.
Production edits are frozen for the affected official reruns.
