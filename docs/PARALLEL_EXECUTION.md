# Native Cooperative Execution

Source: `lib/jxl/enc_lossless_probe_pool.h` at
`b866b3defec13494bfbaf705db93581cda118e37`.

The Rust implementation lives in `src/execution/arena.rs`. Cooperative encoder
roots and their nested algorithms execute on the caller's installed Rayon pool.
It does not create OS threads, another pool, or a global Rayon executor.

## Scheduling Contract

- Logical root counts and scratch limits are separate from physical workers.
- Idle workers take the most recently published nonempty child group before
  starting another root. This is the source descendant-first policy.
- A waiting callback only helps its own child group, never ancestor siblings.
  This protects scratch retained by a suspended callback.
- Each queue claim contains `max(1, count / (4 * workers))` tasks.
- Quiet mode skips notifications for incomplete groups, not publication or
  root completion; single-child calls execute inline.
- Nested cooperative clients publish bounded root loops into the current
  arena. They do not recursively enter the caller's thread runner.
- Serial/null scopes remain on the calling thread and never publish children.
- Cooperative callbacks must use the supplied encoder/task entry points, not
  directly schedule, yield or install pools through Rayon. The core wrapper
  rejects cross-pool installation before yielding, preventing a suspended-root
  deadlock. Same-pool inline calls and serial scopes remain valid.
- Results merge in input order. Completion order does not select codec policy.

`DriverOptions::arena_quiet` applies to cooperative search probes and full-frame
finalists. `WorkerPlan::with_quiet` exposes the same control to direct clients.
LZ77 stream cooperation has separate explicit `SourceConfig` controls; enabling
one optional execution route does not silently enable other codec experiments.

Core `map`, `map_mut`, `for_each_mut`, `map_with_state`, `try_map` and `join`
dispatch to these queues while a cooperative arena is active. RCT rows,
Modular channel tokenization, entropy jobs/clustering and coverage scans use
these entry points rather than starting independent Rayon child work.

## Borrowing And Failure

Most integration uses safe, disjoint mutable slice borrowing. The executor's
scoped callback registration uses a small audited type-erasure boundary:

1. A callback is `Fn + Sync` and remains owned by its submitting stack frame.
2. Every claimed task remains in the group's unfinished count until its
   callback has returned, including all nested groups.
3. The owner unregisters the group only after that count reaches zero.
4. Panics are caught inside callbacks. All active borrowed work drains before
   the first payload resumes unwinding on its submitting thread.
5. A root failure stops new root claims but does not abandon active callbacks.
6. Thread-local arena pointers are restored before each worker loop exits;
   all loops join before the arena is dropped.

Additional panic payloads are deliberately forgotten: arbitrary payload Drop
implementations must not unwind the executor while borrowed work is live.
Normal results, errors and the first panic payload are not leaked.

## Verification

The parent checkpoint passed all 16 execution tests, including cross-pool
rejection before yielding and secondary-panic-payload handling, plus the 11
worker-plan tests. Tests cover scoped borrowing, Send-only results, root limits,
nested groups, cancellation, own-root scratch isolation, queue ordering and
null-runner behavior. Independent official djxl 0.12/0.13 suites additionally
passed actual MA channel-root/whole-subtree, LZ stream, scheduler batch and JPEG
cooperative execution. Runtime profiles assert activation, not only requested
options. See PORT_VERIFICATION.md for the final integrated checkpoint; these
tests are not a whole-encoder speedup or exhaustive concurrency proof.
