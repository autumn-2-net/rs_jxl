# Small-Probe Parallel API

Source: read-only libjxl `b866b3defec13494bfbaf705db93581cda118e37`,
`lib/jxl/enc_lossless_probe.cc::ProbeLosslessCandidates`, especially control
tasks at 1021, candidate prefixes at 1062, shortlist tasks at 1124 and
validation tasks at 1188. This helper supplies execution adapters; it does
not introduce a new candidate-selection algorithm or change source heuristics.

## Entry Point

Public path: `rs_jxl::scheduler::small_probe::probe_lossless_candidates_parallel`.
The existing public small_probe module needs no additional registration.

```rust
pub fn probe_lossless_candidates_parallel<F>(
    image: &Image,
    params: &CandidateParams,
    incumbent: Option<&CandidateParams>,
    options: PlanOptions,
    feedback_bounds: bool,
    limits: ProbeLimits,
    pool: &rayon::ThreadPool,
    worker_slots: usize,
    encode: F,
) -> Result<ProbeResult>
where
    F: Fn(&CandidateParams, &Image) -> Result<usize> + Sync;
```

The pool is borrowed for the call. No threads, private/nested pool, global
Rayon pool or mutex around the encoder callback are created. Worker slots
must be 1..=64 and are validated even for inputs that skip probing. The
caller chooses this limit explicitly; the supplied pool determines physical
concurrency. A callback need not be static or own its inputs.

The callback must encode an independent sample, honor the supplied candidate
settings, return a positive complete encoded byte count, and propagate failure.
It must not recursively invoke the scheduler. Encodes can overlap, so shared
callback state must be concurrency-safe. Each invocation uses the source null
runner, including on a one-thread pool: nested core execution stays serial.
An outer serial execution scope prevents pool entry entirely. This is parallel
candidate/sample work, not cooperative parallelism inside each encoder.

## Existing Algorithm

The helper calls `probe_lossless_candidates_with_prefix` without duplicating
its plan, crop creation, stratification, shortlist, exploration or validation.

- Controls and validation use `WorkerPlan::independent` with `run_plan`, whose
  callback runner is Null and whose output order matches input order.
- Screen and shortlist callbacks call `run_prefix_screen` with the exact
  `PrefixScreen` supplied by the existing adapter, including its remaining
  `PrefixWork` budget. Candidates own their ordered crop prefixes.
- With feedback enabled, completed screen totals immediately update the live
  sixth-best cutoff. Other candidates observe it before their next crop. There
  is no tile/wave barrier. Strict-greater pruning preserves stable ties;
  Variable-predictor exploratory candidates still complete their screen scores.
- Shortlisted candidates retain the fixed control-derived bound and original
  crop order. The optional larger validation crop and source acceptance margins
  are unchanged. Crop winners remain estimates, not whole-frame guarantees.

No candidate defaults, predictor choices, sample thresholds or feedback flags
are inferred or changed by choosing the parallel entry point.

## Budgets And Errors

`ProbeLimits` applies to the whole helper invocation, not separately to each
phase. The existing Work adapter accounts for controls before supplying the
remaining encode/pixel budget to the screen, then accounts for that actual
usage before shortlist and validation. It never creates a fresh full budget
for a prefix pass. Resident sample and candidate ceilings remain unchanged.

Controls and validation preflight/reserve their whole batch. Independent
run_plan jobs drain the batch on ordinary callback errors and return the first
input-order error. Prefix work reserves each cell immediately before encoding,
including failed attempts; bounded or cancelled cells are not charged. A
prefix error prevents later reservations and drains already-active calls before
returning its first observed error. Zero sizes and byte-sum overflow are errors.
No incomplete or failed candidate score escapes as a successful ProbeResult.
The prefix failure lock is not held while invoking the encoder.

With feedback disabled, work and winners match exhaustive serial screening
subject to the same existing shortlist bounds. With feedback enabled,
completion order may change the number of safely skipped encodes, not the
completed-score winners. A hard budget that fits one feedback execution may
not fit another interleaving; callers should not assume deterministic parallel
work counts or compressed-JXL byte identity.

## Legacy Compatibility

`probe_lossless_candidates` retains its `FnMut` per-sample callback and serial
algorithm. `probe_lossless_candidates_batched` retains its `FnMut` ordered
batch callback and up-to-eight-candidate waves. Such a callback returns only
after a whole batch; it cannot stream completion events while that call is in
progress. Feedback therefore updates between waves/rounds and may do extra
work, while preserving possible top-six candidates.

These legacy APIs are not wrapped in a mutex and relabeled asynchronous.
Existing mutable/non-Sync callbacks continue to work unchanged. Callers that
need live completions can provide a genuinely shared Fn + Sync callback to
the new helper, or retain direct control through the existing with_prefix API.
Native prefix screening already uses the live implementation independently;
this convenience helper does not replace or modify native/driver dispatch.

## Verification And Ownership

Six new tests in `small_probe_parallel_tests.rs` define:

- Exhaustive serial winner/score/work equivalence with feedback disabled,
  normal and extended plans, optional validation, one/three physical threads,
  and one/two/eight logical workers; all callbacks verify null-runner behavior.
- Live cutoff reuse with one candidate blocked while six others finish, using
  a completion signal rather than timing-based winner assertions.
- Whole-call encode/pixel/resident/candidate budgets, exact-budget success,
  prefix reservation limits and validation batch preflight.
- Control draining, prefix cancellation, zero sizes, overflow, error identity
  and reuse of the borrowed pool after errors.
- Invalid worker settings, small-input early return and outer serial scope
  without entering the supplied pool.
- Existing stateful FnMut callbacks and ordered wave behavior.

Only `scheduler/small_probe.rs`, the new test file, and this document are
changed in this slice. `prefix_screen`, driver, native, worker_pool, adapter,
module registration, Cargo files and frozen C++ remain outside its edits.
Scoped formatting is complete. Parent reports all six
`scheduler::small_probe::parallel_tests` passed centrally in release (0.03 s).
The parent-owned explicit-LZ-option projection regression also passed:
`base.modular.source_lz77` tool flags survive native projection, while
candidate-owned method, bucket/careful policy and wire configurations override
their corresponding fields; None-LZ remains disabled. This adapter change is
separate from the helper and was not edited here.

Parent also confirms five debug LZ stream-arena tests, one null-seed test, and
`indexed_lz_stream_arena_pixels_match_official_djxl` passed (three cases across
two official decoders, six pixel checks); the four accompanying integration
tests passed as well. LZ details are recorded in `LZ77_PORT_STATUS.md`.

No Cargo, benchmark or independent decoder was run by this worker. The helper
and owned code remain frozen. This closes the convenience parallel-probe API
gap, not all scheduler migration work, and does not make legacy FnMut batch
callbacks capable of streaming completions.
