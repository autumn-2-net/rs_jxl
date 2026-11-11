# Chain Batch Scheduling And Speculation Feedback

## Scope

Follow-up to [the FXZ interleaving experiment](FXZ_CHAIN_INTERLEAVE_20261111.md).
This changes execution of the existing immutable SymbolId Chain Batch, not its
match budget, distance rules, global DP or JXL syntax. Chain Batch remains opt-in;
the production ordinary-chain route is not automatically replaced.

## Stream Scheduling

Previously, one eligible batch stream forced the complete optimal pass through
`Schedule::jobs(inputs.len(), 1)`. This retained inner batch workers but also
serialized all unrelated streams.

The new `SourceConfig.chain_batch_streams` path uses the existing cooperative
executor on the installed caller pool. Nonempty roots are ordered largest-first,
results are returned in input order, and batch child tasks share that executor.
The concurrent-root allowance includes conservative parser scratch and the
additional immutable-index/frontier allocation. One oversized root can still
run; serial scope, absent runner and explicit memory/worker limits remain in
force. No process-global or additional OS thread pool is introduced.

## Speculative Work

`SourceConfig.chain_batch_adaptive` starts with on-demand complete queries. It
initially checks every 128 actual queries for their position density and whether long matches
would force the speculative 128-token frontier to be requeried. It starts
lookahead only with at least 75% query density and at most 1/16 long queries.

The first speculative window is `clamp(workers * 512, 2048, 16384)`. A completed
window can grow only when at least 75% of its positions were consumed and at
most 1/16 consumed frontiers needed full requery. Otherwise it returns to
on-demand sampling. Dense regions can re-enable lookahead later; the decision
is not poisoned permanently by an earlier sparse region. A failed window
multiplies the observation period by four, up to 4,096 actual queries, to avoid
repeated small fills in a persistently sparse region. A successful window
restores the 128-query observation period. This is a speculative retry interval,
not a limit or delay on the complete on-demand matching work. A single execution
worker stays demand-only because there is no matching work to parallelize.

These thresholds govern redundant work, not compression search limits. All
on-demand positions use `Index::find` with the same full visit budget and match
length as the scalar matcher. Overflowed speculative frontiers still receive
complete requeries. Incorrect forecasts can cost time but cannot remove
candidates. Encoded-byte identity is not the correctness acceptance criterion.

Work profiles distinguish speculative `computed` positions, total `consumed`
queries, `demand_queries`, `overflow_requeries`, and `unused` precomputation:

```text
computed + demand_queries = consumed + unused
overflow_requeries <= consumed - demand_queries
```

Unlike the original FXZ lane activation test, an active adaptive Batch with
zero computed positions is valid: it may use its immutable index entirely on
demand. Route receipts and positive consumed queries establish activation.

## Controlled Comparison

Both new fields default to true *within* default-off `chain_batch`. Development
CLI ablation flags retain each old behavior independently:

- `--chain-batch-legacy-schedule`: disable cooperative stream roots.
- `--chain-batch-fixed-window`: disable the new feedback controller.
- `--chain-batch-lanes`: remains 1 by default; FXZ interleaving is not enabled.

`tools/benchmark_chain_batch.py` compares ordinary chain, legacy Batch,
scheduling-only, adaptive-only and both changes. Each mode has a separate
profile/activation run before rotated timings. Every output is decoded using
official djxl and compared with original PNG samples, including hidden RGB.
Native time excludes CLI startup and image I/O; reports retain bytes, bpp,
MPix/s and regressions. This fixed-plan comparison is not an outer scheduler
benchmark or a full-corpus acceptance run.

```powershell
cargo build --release -p rs-jxl-cli --example encoder_lab
python -B tools/benchmark_chain_batch.py --encoder target/release/examples/encoder_lab.exe --official-djxl path/to/djxl.exe --case path/to/input.png --family literal --threads 8 --repeats 3 --profile --out out/chain-batch-feedback-20261111/literal-t8.json
```

The older `benchmark_chain_interleave.py` explicitly requests both legacy flags
so future lane-only comparisons do not silently include this scheduling change.

## First Feedback Iteration

Before retry backoff, five-way measurements on the 3.601920 MPix 1eb screenshot
gave these native medians (e9, eight threads, three rotated repeats, separate
profile run, no explicit warmup):

| Fixed family | Ordinary | Legacy Batch | Scheduling only | Feedback only | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| literal | 3.280027 s | 4.767117 s | 3.368818 s | 4.866642 s | 3.371818 s |
| cross6 | 4.441848 s | 4.869734 s | 4.078915 s | 4.597107 s | 3.757135 s |

All 40 outputs passed official pixel verification. All variants of each family
had the same measured size: literal 417,022 bytes / 0.926222 bpp, cross6 351,304
bytes / 0.780259 bpp. These are observations, not encoded-hash requirements.

This isolated the stream-serialization problem but also exposed feedback
oscillation. On literal, unused precomputation fell from 594,564 to 405,299
positions, yet fill calls rose from 57 to 166. On cross6, the adaptive path
recovered useful parallel work after sparse regions, but unused precomputation
rose from 695,700 to 1,303,113 positions, with 531 fills instead of 63. Therefore
fewer wasted positions on one plan was not sufficient evidence of good control.
The final controller above adds backoff after unsuccessful windows.

Initial reports, retained under `out/chain-batch-feedback-20261111/`:
`literal-t8.json`, `cross6-t8.json`; the initial binary is `encoder-lab-v1.exe`.
Do not compare different builds' absolute times without fresh controls.

## Backoff Revision

Fresh controls and three rotated repetitions per mode, excluding separate
diagnostics, give the following native medians in seconds:

| Input / fixed family / threads | Ordinary | Legacy Batch | Scheduling only | Feedback only | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1eb / literal / 8 | 3.646465 | 4.934891 | 3.363073 | 4.931969 | 3.467009 |
| 1eb / cross6 / 8 | 4.752555 | 5.281308 | 4.446554 | 5.507883 | 4.306434 |
| 9bde / automatic / 8 | 1.546473 | 1.546453 | 1.597046 | 1.383624 | 1.471786 |
| 1eb / literal / 1 | 6.094969 | 6.859592 | 6.831262 | 6.138406 | 6.156829 |

The 9bde screenshot is 1.456128 MPix. Its automatic plan invokes many separate
single-stream entropy passes: all schedule receipts report zero cooperative
dispatches. Therefore differences between legacy/scheduling-only or between
feedback-only/combined on that case are measurement variation, not stream
scheduler gains. Inclusive profile times overlap and are not wall times.

| Eight-thread case | Bytes, all modes | bpp | Combined MPix/s |
| --- | ---: | ---: | ---: |
| 1eb literal | 417,022 | 0.926222 | 1.038913 |
| 1eb cross6 | 351,304 | 0.780259 | 0.836404 |
| 9bde automatic | 412,541 | 2.266510 | 0.989361 |

Separate profile work counts, legacy Batch versus combined:

| Case | Speculative positions | Unused positions | Full overflow requeries | Fill calls |
| --- | ---: | ---: | ---: | ---: |
| 1eb literal | 917,512 -> 8,192 | 594,564 -> 4,015 | 15,064 -> 98 | 57 -> 2 |
| 1eb cross6 | 1,015,816 -> 28,672 | 695,700 -> 17,606 | 12,731 -> 426 | 63 -> 7 |
| 9bde automatic | 2,900,088 -> 339,968 | 1,730,756 -> 206,778 | 38,741 -> 5,254 | 192 -> 77 |

Consumed query counts are unchanged: respectively 322,948, 1,232,233 and
1,591,569. The avoided work is speculation, not a reduction in the DP's matching
requests. Work is summed across active streams/passes, not across timing trials.

On the two mixed-stream plans, combined time falls about 29.7% and 18.5% versus
legacy Batch. Most of that benefit is stream scheduling. Feedback alone does
not improve every timing: literal is essentially unchanged versus legacy and
cross6 is slower. Fewer unused positions is not evidence of universal wall-time
improvement; batching can parallelize useful matching too. The independent
controls remain available, and no global ordinary-chain replacement is enabled.

At one thread, feedback removes all speculative positions and overflow
requeries: all 322,948 queries use the immutable index on demand. Combined time
is 6.156829 s / 0.585028 MPix/s versus legacy Batch's 6.859592 s / 0.525092 MPix/s.
Ordinary chain is 6.094969 s / 0.590966 MPix/s. Thus the wasted-work fix recovers
most of Batch's single-thread penalty; it does not establish a win over ordinary
chain. Size remains 417,022 bytes / 0.926222 bpp.

## Verification And Reproduction

- Native tests: 18 Chain Batch tests, 14 worker/arena tests and 30 development
  CLI tests passed. Benchmark harness tests: 19 passed.
- Release build, full workspace compilation (including FFI), formatting and
  diff checks passed. No C ABI or decoder change is required.
- All 120 generated outputs passed independent original-sample verification
  using official `djxl v0.12.0 a1ebdf2`, including profile runs. This is two
  source images under different plans/thread counts/repetitions, not 120 images.
- The backoff revision accounts for 80 checks; the initial iteration accounts
  for 40. Temporary encoded/decoded outputs were cleaned in all six reports.
- Windows x86_64, Rust 1.97.1. No Linux run or full-corpus performance claim.
- Initial release binary SHA256:
  `87f7f6a41dc0944a4212501b655afed6ecc341a7e236ab530b025d308dca8525`.
- Final release binary SHA256:
  `982c1f3ec5d92598fe3ab5545168141cb9b185acd72136575fc449ba6ff22050`.

Final reports under `out/chain-batch-feedback-20261111/`:

| Report | SHA256 |
| --- | --- |
| `literal-t8-v2.json` | `fce6dad4baf9b3a23f7722dc9ec174fa71044fe726249b96ac9ed12ff6bab88b` |
| `cross6-t8-v2.json` | `05f9452311c245f48f49ad4a0f8adc946236775acfc08739235d187481ab963d` |
| `automatic-t8-v2.json` | `e6b28f76e12e9b61311b5a9408fded7a868bf601f548f0ec946f2e36127d7eaf` |
| `literal-t1-v2.json` | `fc4ba10022a50a916e9deec2a61581c603972685d53244de595f60a3b2874a15` |

Hashes identify measured artifacts, not a required encoded output identity.
Per-trial timings are retained because these short runs have visible variation.
Do not interpret small differences as statistically established improvements.
