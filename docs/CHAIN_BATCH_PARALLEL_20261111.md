# Parallel-Only Chain Batch

## Scope

Follow-up to [cooperative Batch scheduling and speculative-work feedback](CHAIN_BATCH_FEEDBACK_20261111.md).

Later implementation and measurements: [compact frontiers and deferred initialization](CHAIN_BATCH_MEMORY_20261111.md).
This experiment enables that existing combination only when its actual execution
context has at least two workers. It does not enable FXZ query interleaving,
change match budgets, alter global DP, or change the JXL format.

The initial opt-in experiment below was subsequently promoted to the default
pixel source policy; see [Default Integration](#default-integration).

An outer eight-thread image/group schedule does not imply eight workers are
available to each parser. Serial child scopes must keep ordinary Chain. The
gate therefore executes inside native code, not in Python or by inspecting a
top-level thread argument alone.

## Controls

Native `SourceConfig` now has `chain_batch_parallel_only`, default false. It
restricts `chain_batch = true`; it does not enable Batch by itself. Existing
explicit forced-Batch experiments retain their behavior when the restriction
is false. Production defaults were unchanged during the initial timed trial.

```rust
use rs_jxl::tools::lz77::{SourceConfig, SourceParallelism};

let lz = SourceConfig {
    chain_batch: true,
    chain_batch_parallel_only: true,
    parallelism: Some(SourceParallelism {
        max_workers: 8,
        ..Default::default()
    }),
    ..Default::default()
};
```

Pass this as the explicit native source-LZ configuration before projecting a
source candidate. The candidate still determines the parser method. An eligible
Optc256/SymbolId route and an installed caller-owned executor are required;
merely constructing this value neither creates a thread pool nor starts Batch.

Development `encoder_lab` exposes `--chain-batch-parallel`. This sets both fields
above, keeps one query lane, and retains cooperative scheduling plus adaptive
lookahead. It conflicts with forced `--chain-batch`, explicitly supplied
`--chain-batch-lanes` (including 1), legacy schedule/window flags, and unrelated
match-extension experiments. It is not a new production
CLI or Python/C ABI parameter.

## Admission

The existing size, seed-density, method, cost-model, index and memory gates still
apply. The additional test uses exactly the worker calculation used by Batch:

- No installed caller pool or explicit serial scope: ordinary Chain.
- A one-thread pool or logical `max_workers` of one: ordinary Chain.
- A caller/cooperative executor allowing multiple effective workers: Batch may
  run if the remaining eligibility conditions pass.

The direct admission check also rejects zero workers, but public configuration
validation still rejects `max_workers = 0`; it is not a supported API setting.

The gate is checked before stream scheduling and again before immutable-index
allocation, keeping both decisions consistent with their execution context.
Unsupported or resource-limited inputs fall back to ordinary matching, not a
smaller search budget. The existing per-stream index budget and cooperative
root-memory allowance are unchanged. They are not a total-process RSS limit.

Single-thread work must have zero Batch index builds, speculative positions and
cooperative Batch schedules, not merely zero useful precomputed matches.

## Validation Method

`tools/benchmark_chain_parallel.py` compares ordinary and parallel-only modes
within one release binary, with separate activation profiles and alternating
timed trials. It records bytes, bpp and native MPix/s, excludes CLI/image I/O,
and checks every output against original PNG samples using official `djxl`.
No compressed-byte identity is required. Temporary output images are removed.

Use `--expect active` for known multithreaded Batch hotspots, `--expect inactive`
for single-thread controls, and `--expect either` for a mixed input set. A
requested flag alone is not counted as activation. The experiment must verify
actual worker counts, adaptive behavior and one-lane receipts when active.

This is a fixed-candidate comparison rather than an outer-scheduler benchmark.
The separate background compression task on this host is not stopped; timing
noise must remain explicit in any performance conclusions.

## Results

Windows x86_64, Rust 1.97.1, release build, effort 9. Each mode had one warmup,
three measured encodes and a separate profiled encode. Times are medians of
native encoding calls, not process wall times. All sizes below are identical
between modes; bpp is `8 * bytes / (width * height)`.

| Input / family | Threads | Batch active | Ordinary seconds | Parallel-only seconds | Time change | MPix/s before -> after | Bytes | bpp |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1eb...` / cross6 | 8 | yes | 4.593511 | 4.170756 | -9.20% | 0.784 -> 0.864 | 351304 | 0.780259 |
| `1eb...` / cross6 | 2 | yes | 5.927698 | 5.419860 | -8.57% | 0.608 -> 0.665 | 351304 | 0.780259 |
| `1eb...` / cross6 | 1 | no | 9.090726 | 9.336151 | +2.70% | 0.396 -> 0.386 | 351304 | 0.780259 |
| `9bde...` / automatic | 8 | yes | 1.691377 | 1.579681 | -6.60% | 0.861 -> 0.922 | 412541 | 2.266510 |
| `wallhaven-n6qzmx` / automatic | 8 | yes | 2.895898 | 2.494619 | -13.86% | 0.716 -> 0.831 | 459153 | 1.771424 |
| `kodim01` / automatic | 8 | no | 0.794644 | 0.805520 | +1.37% | 0.495 -> 0.488 | 454218 | 9.241089 |
| `2026.6test1` / automatic | 8 | no | 1.859293 | 1.807500 | -2.79% | 1.115 -> 1.147 | 2098189 | 8.094865 |
| `basn2c16` / automatic | 8 | no | 0.005401 | 0.005555 | +2.86% | 0.190 -> 0.184 | 153 | 1.195313 |

The six source images span screenshots, RGBA artwork, a photograph, an
illustration and a 32x32 RGB16 correctness probe. Full paths, input hashes,
commands and individual timings are in the reports. The tiny RGB16 probe is
not a representative throughput result. `automatic` is a fixed candidate family
here, not the full production outer search.

The T1 control had **zero Batch builds and zero Batch schedule receipts**.
Its measured ranges overlap: ordinary 9.062-9.322 s, parallel-only 9.034-9.718 s.
The +2.70% median is retained rather than discarded. Same-route fallback does
not prove zero gate overhead or a statistically stable absence of regression.
Likewise, do not credit any timing changes in the other inactive cases to Batch.

For the active `1eb...` case, the profiler recorded six Batch streams, two
cooperative roots, one cooperative schedule, and exactly eight/two effective
workers at T8/T2 respectively. T8 computed 28672 speculative positions, of
which 17606 were unused; T2 computed 14336, of which 8447 were unused. The
adaptive fallback still leaves substantial matching demand-driven. No FXZ
interleaving was enabled. Active Batch on a single stream does not by itself
prove cross-stream cooperation; these results measure the combined mode, not
an isolated benefit from each component.

### Correctness And Checks

- Six original images, eight matrix rows, **80 encoder/official-decoder pixel
  checks passed**, including warmups and profiles. Original sample comparison
  includes invisible RGB and 16-bit samples. No metadata stripping was used.
- All measured sizes remained unchanged. Compressed-byte identity was not an
  acceptance condition.
- `cargo test -p rs-jxl --lib chain_batch -- --test-threads=1`: 20 passed.
- `cargo test -p rs-jxl --lib chain_arena -- --test-threads=1`: 6 passed.
- `cargo test -p rs-jxl-cli --example encoder_lab`: 31 passed.
- Focused Python Chain benchmark harness suite: 29 passed.
- `cargo check --workspace` and the release example build passed.

### Local Artifacts

Reports, with no failed trials or retained temporary image outputs:

- `out/chain-batch-parallel-20261111/cross6-t8.json`
- `out/chain-batch-parallel-20261111/cross6-t2.json`
- `out/chain-batch-parallel-20261111/cross6-t1.json`
- `out/chain-batch-parallel-20261111/mixed-t8.json`

The exact executable is preserved locally as
`out/chain-batch-parallel-20261111/encoder-lab.exe`. Its SHA256 is
`8024f2ac023cfacb719ea362067a70e8e150c29b4fdfe99fa2f8d04329c796e1`.
The official libjxl 0.12 `djxl.exe` SHA256 is
`c97ec1990e564c4bbf55fbeb550919bdce2f93457817334a7256ce0100026481`.
These hashes identify test binaries, not target compressed image contents.

## Reproduction

From the repository root, using an existing release binary:

```powershell
python -B tools/benchmark_chain_parallel.py `
  --encoder out/chain-batch-parallel-20261111/encoder-lab.exe `
  --official-djxl ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe `
  --case ../jxl_python/1eb7e7b9b634459e198627e61f119f84.png `
  --family cross6 --threads 8 --expect active `
  --repeats 3 --warmups 1 --profile `
  --out out/chain-batch-parallel-20261111/cross6-t8-rerun.json
```

For the single-thread control, use `--threads 1 --expect inactive` and a fresh
report path. For mixed inputs, repeat `--case`, use `--family automatic
--expect either`, and retain the mandatory separate profiles. Existing reports
are not overwritten. The harness does not compile binaries or enable global
production settings.

## Initial Decision

Retain parallel-only admission as an opt-in native/development mode. It avoids
building Batch state in serial contexts and the measured active multithreaded
cases improved by about 6.6-13.9% without a size change. This focused sample,
under concurrent host load, is not sufficient to turn it on globally or promise
the same gains under the full image scheduler. No C/Python ABI or production
default was changed.

## Default Integration

After the user accepted the parallel-only results, the default pixel source
policy now enables cooperative/adaptive Chain Batch with the actual-worker
restriction. This covers source candidate projection used by the CLI, Python
FFI, typed layers and source search, including the Scout/Pilot defaults.
Eligibility still requires Optc256/SymbolId, sufficient stream/seed density,
memory and at least two effective workers. Serial/null scopes never build a
Batch index; bucket/indexed and lower-tier parsers keep their existing routes.
Lower-tier default stream-worker policy is unchanged.

An explicit `base.modular.source_lz77` remains authoritative. The low-level
`SourceConfig::default()` remains ordinary, so existing explicit experiments
(including match-extension kernels) do not accidentally conflict with Batch.
For an explicit production override, set `chain_batch = false` in the source
configuration before projecting the candidate. No new Python/C ABI option is
needed: existing `threads` controls the executor. JPEG coefficient-transcode
policy and manual legacy LZ77 are unchanged.

The development `encoder_lab` intentionally keeps its no-flag ordinary
baseline, with `--chain-batch-parallel` selecting the candidate. Thus historical
benchmark commands do not silently compare Batch with itself after promotion.

Two narrow work reductions accompany promotion:

- Check route/executor eligibility before allocating the admission mask or
  rescanning seed commands for Batch density. Ordinary-parser seed accounting
  is retained; this removes the additional Batch-only scan on rejected routes.
- Reused speculative frontiers reset only their live-entry count and overflow
  flag. All backing entries remain initialized and only `entries[..size]` is
  read. Overflow still triggers the full original query. No query limit,
  speculative window, tie rule or DP search budget changes.

These are reductions in redundant setup/stores, not a claim of an independently
measured end-to-end speedup. Index construction and unused speculative queries
remain potential hotspots; lazy allocation or a different speculation policy
would require a separate measured experiment.

### Promotion Validation

The final release was checked through the actual production CLI, **without a
Batch flag**, as well as the development comparison. Production e9/T8 activated
Batch on `9bde...` and `wallhaven-n6qzmx`; e9/T1, e3/T8 and the 32x32 RGB16
input bypassed it. All five production outputs decoded to the original samples
with official libjxl 0.12. The CLI initially rejected unsupported PNG `pHYs`
metadata before encoding; the rerun explicitly used `--strip-metadata` and is
a pixel check, not a metadata-preservation check. Both receipts are retained.

Final same-binary comparison, ordinary versus parallel-only Batch, e9/T8:

| Input | Ordinary s | Parallel-only s | Time change | MPix/s before -> after | Both bytes | Both bpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `9bde...` | 1.683398 | 1.361409 | -19.13% | 0.865 -> 1.070 | 412541 | 2.266510 |
| `wallhaven-n6qzmx` | 2.611851 | 2.542954 | -2.64% | 0.794 -> 0.815 | 459153 | 1.771424 |

Each mode again had a separate profile, one warmup and three timed calls;
all 20 output checks passed. These rows compare the **whole Batch mode** with
ordinary Chain, not the two new work reductions against the previous Batch
binary. The ongoing host load and varying timings still preclude attributing
the difference from the initial table to those small kernel changes.

Final native checks passed: 21 Chain Batch tests, 6 cooperative Chain arena
tests, 348 scheduler tests, 31 development CLI tests and 29 Python benchmark
harness tests. The complete production CLI and development example release
build succeeded. This is focused validation, not a new corpus-wide or Linux CI
acceptance run.

The Python FFI release library was also rebuilt. Its 39 selected end-to-end
tests passed (still inputs, multiframe, reusable contexts and configured
official-decoder checks). Four retained MA collector tests from the preceding
row-kernel experiment passed as well. No Python transport or ABI change was
required to receive the new source-policy default.

New local reports: `out/chain-batch-parallel-default-20261111/final-t8.json`,
`production-default.json` (the rejected metadata invocation), and
`production-default-v2.json` (five passing production cases). Temporary encoded
and decoded image outputs were removed. Final release SHA256 values:

- Production CLI: `4b5fd123bc71aced2d0fcd26661fb6f016accc8e0c288d776794786bc7e89be3`.
- Development example: `fb755a0761c8d34bc92682e43d36b1ce6018a6f1206fc420b65fb3a70f7ce5e7`.
