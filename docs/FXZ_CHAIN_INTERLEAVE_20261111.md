# Immutable Chain Query Interleaving Experiment

## Scope

This is a native Rust, default-off experiment inspired by FXZ's interleaving of
independent linked-list pulls. It does not port FXZ's radix matcher, LZMA2 syntax,
or compression format. It changes how existing immutable chain queries execute
inside a worker, not which candidates the global JXL parser is allowed to use.

Reference inspected:
[FXZ radix_pull_batch.h](https://github.com/peacherwu969/fxz/blob/8236dd5fdb030c2e8a47419b760c664b6c4cabb5/src/liblzma/radix/radix_pull_batch.h).
The implementation here is independent Rust code, not copied FXZ source.
No speedup measured in FXZ is assumed to transfer to JXL integer tokens.

## Native Controls

This report describes the initial fixed-window, serial-stream experiment. Later
[Chain Batch feedback work](CHAIN_BATCH_FEEDBACK_20261111.md) adds separate
scheduling/lookahead controls. The lane benchmark now explicitly requests
`--chain-batch-legacy-schedule --chain-batch-fixed-window` to preserve this
report's comparison. The new controls are not part of the timings below.

`SourceConfig.chain_batch` remains false by default. With it enabled:

- `chain_batch_lanes = 1`: existing per-position scalar batch fill, unchanged.
- `chain_batch_lanes = 2`, `4`, or `8`: interleave that many independent queries
  inside each existing batch worker. This does not multiply worker threads.
- Other lane counts are invalid; non-one lanes require `chain_batch = true`.
- Existing route/size/seed-density/memory gates remain in force. Unsupported
  routes retain their ordinary matcher; an explicit flag alone is not evidence
  that the experiment actually ran.

The development `encoder_lab` exposes `--chain-batch` and
`--chain-batch-lanes 1|2|4|8`. Production CLI defaults, C/Python numeric modes,
the decoder, and JXL bitstream syntax are unchanged.

## Implementation And Invariants

`tools/lz77/source/chain_batch/interleave.rs` owns the experimental query loop.
It maintains one independent state per lane, loads candidates/successor links
in a preparation phase, then compares and extends those candidates. Completed
lanes refill from the remaining positions. There are no extra heap allocations
inside that loop and no new unsafe code or global thread pool.

Per-query behavior matches scalar `Index::find_limited`:

- Refined chains charge coarse ordinal differences, including skipped collisions.
- Other chains retain their original visit counts, order and budget boundaries.
- Zero counts three and four retain their different prefix behavior. The choice
  to follow a zero chain still waits for the actual current match length.
- The dictionary excludes distance equal to the window size, as before.
- A capped 128-token match terminates that speculative query and marks the
  frontier for complete scalar requery. Dominance overflow has the same fallback.
- `best` updates even if the dominance frontier does not retain that candidate.
- Partial worker partitions, tails, skipped origins and demand-only fallback
  remain supported. The global DP still consumes positions in increasing order.

Ordered candidate/command equivalence tests are primitive regression checks,
not a requirement that optimized complete JXL files have an old output hash.
Codec correctness is original decoded sample equality, including hidden RGB.

## Measurement Contract

`tools/benchmark_chain_interleave.py` compares one prebuilt binary with:

1. `baseline`: ordinary mutable chain, no Chain Batch.
2. `batch1`: existing immutable batch, no query interleaving.
3. `batch2`, `batch4`, `batch8`: the corresponding interleaved batch.

It accepts explicit input paths/manifests, never discovers a directory or builds
an encoder. Each mode uses the same fixed `automatic`, `literal`, or `cross6`
candidate family. This is not the outer production search scheduler.

Mandatory separate profile runs verify ordinary Optc256/SymbolId chain routes,
actual batches and positive computed work before timing repetitions. Requested
flags without activation fail the experiment instead of counting as coverage.
The receipt records tokens, lanes, computed/consumed positions, overflow
requeries and demand-only transitions. `interleaved_positions` counts positions
submitted to the new loop, including no-match/tail positions; it is not measured
hardware concurrency or a cache-miss counter.

Timed runs disable profiling, rotate mode order, and report actual bytes, bpp,
native call time and MPix/s. Native time excludes CLI startup and image I/O.
Original input PNG samples are checked using official `djxl` on every output.
Encoded/decoded temporary files are removed; JSON reports preserve all trials,
failures and regressions. Profile scopes are inclusive and may overlap across
workers, so they cannot be added to native wall time.

```powershell
cargo build --release -p rs-jxl-cli --example encoder_lab
python -B tools/benchmark_chain_interleave.py --encoder target/release/examples/encoder_lab.exe --official-djxl path/to/djxl.exe --case path/to/input.png --family literal --effort 9 --threads 8 --repeats 3 --warmups 1 --profile --out out/chain-interleave-20261111/literal-t8.json
```

Historical profiles identify `1eb7e7b9b634459e198627e61f119f84.png` under literal
and cross6 as active cases. Historical ordinary-chain versus batch timings are
not used as today's baseline. Batch can itself cost more than ordinary chain
because speculative positions are never consumed and index construction is
additional work. Beating `batch1` alone is not sufficient to enable this route.

## First Iteration

`literal-t8-v1.json`: 1eb screenshot, 3.601920 MPix, e9, eight threads,
one warmup and three timed repeats per variant plus one separate profile run.
All 25 outputs passed official pixel verification. Every observed file is
417,022 bytes / 0.926222 bpp; size equality is a result, not a required output
hash target. Native time medians:

| Mode | Seconds | MPix/s |
| --- | ---: | ---: |
| Ordinary chain | 3.519545 | 1.023405 |
| Batch, one lane | 5.136948 | 0.701179 |
| Batch, two lanes | 5.326910 | 0.676174 |
| Batch, four lanes | 5.037854 | 0.714971 |
| Batch, eight lanes | 5.406154 | 0.666263 |

The small complete-call difference for four lanes is not evidence of a faster
matching kernel. Its separate profile reports 0.311835 s in batch fill, versus
0.177485 s for one lane. Two/eight lanes also take about 0.31 s in that scope.
All batch modes compute 917,512 positions, consume 322,948, and perform 15,064
complete overflow requeries: this is real interleaving, not fewer candidates.

### Existing Batch Scheduling Constraint

`source.rs` chooses `Schedule::jobs(inputs.len(), 1)` whenever any stream is
eligible for Chain Batch. This preserves the caller executor for intra-stream
batch workers, but also serializes unrelated streams in that pass. Ordinary
chain keeps the independent-stream schedule. This is pre-existing behavior,
not a change made by the interleaving experiment.

In the separate one-lane diagnostic, LZ parse wall time rises from 2.297983 s
(ordinary) to 3.864899 s (batch), while total batch fill is only 0.177485 s and
index construction is 0.021143 s. Deleting the measured fill cost altogether
would not close that gap. This observation is not a cache-stall diagnosis or a
claim that inclusive worker scopes equal wall time.

The follow-up implementation moves endpoint/prefix token checks out of the
preparation loop, leaving just chain metadata there, and reduces temporary
candidate state. It keeps scalar batch behavior and global scheduling unchanged
so the comparison still isolates interleaving. The initial binary is preserved
as `out/chain-interleave-20261111/encoder_lab-v1.exe`.

## Second Iteration Results

The reduced-state implementation was tested at e9 with three timed repetitions
per mode. The eight-thread literal case also has one warmup per mode. Each
matrix has its own fresh ordinary-chain and scalar-batch controls; do not use
the earlier iteration's absolute timings as its baseline.

Native time medians in seconds:

| Input / fixed family / threads | Ordinary | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1eb / literal / 8 | 3.713194 | 5.147837 | 5.143782 | 5.405449 | 5.392074 |
| 1eb / literal / 1 | 6.246403 | 6.694744 | 7.033910 | 7.216904 | 7.285539 |
| 1eb / cross6 / 8 | 4.692156 | 5.078471 | 5.194923 | 5.178213 | 5.270541 |
| 9bde / automatic / 8 | 1.782650 | 1.525501 | not run | 1.813065 | not run |

`9bde` is `9bde6aaf390109f97622d35ff1e900ee.png`, 1.456128 MPix.
It is a separate activation-checked case, not a different crop of 1eb. The
last matrix deliberately compares only ordinary, scalar batch and four lanes.

Sizes and throughput, with the fastest measured *interleaved* mode in each
matrix shown for comparison (not selected by an automatic encoder policy):

| Matrix | Bytes, all modes | bpp | Ordinary MPix/s | Batch 1 MPix/s | Interleaved MPix/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1eb literal, 8 threads | 417,022 | 0.926222 | 0.970033 | 0.699696 | 0.700247 (2) |
| 1eb literal, 1 thread | 417,022 | 0.926222 | 0.576639 | 0.538022 | 0.512079 (2) |
| 1eb cross6, 8 threads | 351,304 | 0.780259 | 0.767647 | 0.709253 | 0.695591 (4) |
| 9bde automatic, 8 threads | 412,541 | 2.266510 | 0.816834 | 0.954524 | 0.803131 (4) |

The scalar batch helps on the second screenshot, but adding interleaving loses
that benefit. On 1eb literal, the 4 ms difference between Batch 1 and Batch 2
at eight threads is not a useful acceleration. Separate batch-fill profiles
still favor the scalar loop:

| 1eb literal batch fill | 1 lane | 2 lanes | 4 lanes | 8 lanes |
| --- | ---: | ---: | ---: | ---: |
| 8 threads, inclusive seconds | 0.197889 | 0.280134 | 0.277149 | 0.279986 |
| 1 thread, seconds | 0.938259 | 1.378719 | 1.357923 | 1.248375 |

These are separate diagnostic runs, not samples included in the timing
medians. The new staging reduces some overhead relative to the first prototype
but does not make the interleaved loop faster than the original scalar fill.
No hardware cache-stall measurements were collected, so attributing this
solely to memory latency, SIMD throughput or branch prediction is unwarranted.

## Decision

Keep the experiment opt-in; do not change production defaults or dispatch.
Neither implementation demonstrates useful interleaving gains in this focused
matrix. This does not establish that FXZ's own radix matcher is slower, or that
all possible JXL query-interleaving implementations must fail.

The more actionable observations are the existing batch scheduling constraint
and speculative work: on the literal case, 917,512 positions are computed but
only 322,948 consumed, plus 15,064 full overflow requeries. Interleaving retains
that work. A separate experiment could address stream parallelism and unused
queries; neither has been changed here, to avoid confounding the comparison.

## Validation And Provenance

- Base revision: `ccdb779` (the preceding palette experiment, pushed to main).
- Windows x86_64, `rustc 1.97.1 (8bab26f4f 2026-07-14)`, release encoder.
- Official oracle: `djxl v0.12.0 a1ebdf2`, Clang 17.0.6, AVX2.
- Final encoder SHA256:
  `81346b28d84ae058741e661f187d015bf9dd8bef5ed478bb6507e61c52d0e10a`.
- Official decoder SHA256:
  `c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.
- 13 native Chain Batch tests, 30 `encoder_lab` argument tests and 10 benchmark
  harness tests passed. Workspace compilation and formatting are also checked.
- All 102 encoded outputs passed original-pixel verification through official
  djxl, including profile runs and warmups. This is 102 checks on two source
  images and several fixed plans, not 102 different images or a corpus sweep.
- No full-corpus, Linux, or JPEG byte-transcode performance claim is made.

Reports are local ignored artifacts under `out/chain-interleave-20261111/`:

| Report | Official pixel checks | SHA256 |
| --- | ---: | --- |
| `literal-t8-v1.json` | 25 | `72c7ed80dee38c7ad37083cc45243d3df94c5713ba86011b88a75b02eb0d8176` |
| `literal-t8-v2.json` | 25 | `c9a534f34f3b3ca81af0df015298f4f5be4e2c2dcc8a4a360aca9845e5242674` |
| `literal-t1-v2.json` | 20 | `5e421ffdf5ea9a91f34174bb86022529f7e74c6fb50c465a389a8c64dc98cfb8` |
| `cross6-t8-v2.json` | 20 | `df3b959d36a40c76fb0586624e104a9bd364d1e459893ada9fb77514fc3e1baa` |
| `automatic-t8-v2.json` | 12 | `3cb5e443056388d43afb7f2d3ce12854008ac259244fded4d53f95c44b6114bb` |

Hashes identify binaries/reports only. Output bitstream identity is not a
codec correctness or compression acceptance condition.
