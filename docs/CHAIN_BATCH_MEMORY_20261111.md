# Chain Batch Memory Work: 2026-11-11

**Retained: compact frontiers, deferred initialization and exact zero-run skipping.**
The longer-admission experiment was reverted. Targeted Windows measurements below
compare against the already-enabled Batch baseline, not against ordinary Chain.
They are not a new corpus-wide performance or cross-platform acceptance claim.

## Baseline and Method

- Baseline commit: `b989bf3` (Enable parallel-only Chain Batch for pixel encoding).
- Prebuilt baseline: `out/chain-batch-feedback-20261111/encoder-lab-before.exe`.
- Baseline SHA256: `fb755a0761c8d34bc92682e43d36b1ce6018a6f1206fc420b65fb3a70f7ce5e7`.
- Both checkpoints receive identical e9, fixed-family, T8 and
  `--chain-batch-parallel` settings. Explicit inactive controls are separate.
- The checkpoint harness uses serial alternating trials and native encoder
  seconds, bpp and MPix/s. Separate profiles establish actual eligible Batch work
  or the explicitly requested inactive control; profiles are excluded from medians.
- Every output is decoded by official djxl and compared with original samples,
  including depth and invisible RGB. JXL-byte identity is not required. Binary
  hashes establish executable identities and are checked before/after the run;
  they are not pixel-correctness evidence. Failures and regressions remain recorded.

## Retained Native Candidate

1. **Zero-run certificate:** unequal capped zero-run counts prove an exact match
   length when the shorter run ends before the requested limit. Equal runs skip
   proven zeros and extend normally, including saturated runs and overlapping copies.
2. **Lazy initialization:** frontier capacity remains reserved fallibly. Only the
   needed prefix is initialized, growing to the used high-water mark; demand-only
   streams need no initialized frontier entries. This is not lazy index construction.
3. **Compact SymbolId frontier:** two u32 fields replace the general frontier's
   usize length, symbol and redundant float cost. The x64 footprint is
   144 versus 272 bytes per frontier. Integer-symbol ordering preserves float-cost
   ordering with symbol ties. The 16-entry bound and complete demand fallback remain;
   lengths above the speculative cap of 128 defensively trigger overflow, not truncation.

The admission strategy is restored to the original **128 observations**, including
the successful-window reset. Existing backoff, eligibility gates and native search
budgets are retained. The longer-evidence experiment described below is not retained.

## Disk IO Versus Memory Work

In `crates/rs-jxl-cli/examples/encoder_lab.rs`, input reading/decoding occurs at
lines 394-396, the timer starts at 728, the direct native encode stops the timer at
826, and output-file writing occurs at 968-973. Input/output disk IO and independent
djxl verification are therefore outside the reported native encoder seconds.

Allocation, initialization and memory access during encoding can still contribute
to that time. Smaller frontiers and deferred initialization reduce storage and
initialization work; they do **not** establish measured RSS savings, memory bandwidth,
cache misses or the root cause of a slowdown. Existing unrelated host/user load was
observed; do not terminate those processes. These runs are not isolated-host evidence.

## Preliminary Zero-Kernel Trials

Records under `out/chain-batch-feedback-20261111/`: `zero-t8.json` and
`zero-cross6-t8.json`. Each used one warmup and three measured repeats, T8/e9,
with active Batch receipts. Changes below are after/before native-time medians.

| Case | Family | Before (s) | Zero kernel (s) | Change |
| --- | --- | ---: | ---: | ---: |
| `9bde6aaf390109f97622d35ff1e900ee.png` | automatic | 1.424677 | 1.367808 | -3.99% |
| `wallhaven-n6qzmx.png` | automatic | 2.638330 | 2.490153 | -5.62% |
| `1eb7e7b9b634459e198627e61f119f84.png` | cross6 | 5.056469 | 5.192640 | +2.69% |

All three cases passed their harness checks. Results are small/mixed with
overlapping trial ranges, not a stable general speedup or a memory-bottleneck diagnosis.

## Rejected Longer Evidence

`evidence-t8.json` used one warmup and five measured repeats. The trial replaced
128 observations with `initial_window(workers)/8` and reset successful windows to
that base. At eight effective workers, admission required 512 demand observations.

| Case | Before (s) | Evidence trial (s) | Change | Profile unused positions |
| --- | ---: | ---: | ---: | ---: |
| `9bde...` | 1.592193 | 1.648694 | +3.55% | 206778 -> 122518 |
| `n6qzmx` | 2.665888 | 2.620327 | -1.71% | 152213 -> 140129 |

Fewer unused speculative positions did not yield a stable timing benefit: 9bde
regressed despite substantially less unused work. The policy was rejected and
reverted; native full search always remained available through demand fallback.

## Validation and Final Results

**29 chain_batch tests passed**, including six focused compact
frontier tests for equivalence, float-rounding ties, clear cycles, capacity overflow,
oversized lengths and portable layout bounds. All 37 focused Python benchmark
harness tests passed, including single report publication after both binaries'
identity checks and failure-path cleanup.

Final measured example: `out/chain-batch-feedback-20261111/encoder-lab-memory.exe`,
SHA256 `1bd255715493e20d3042334eade9b9ec3e84fc5b3def22a04da53082ab20fe5a`.
The rebuilt production CLI SHA256 is
`7dc8a6a88445f3a7ae71bd9e9e906e327bc8505da7ae751b75e8c6d9ff3ece58`.
The Python native library was rebuilt too: `target/release/rs_jxl_ffi.dll`, SHA256
`97d1b95f6ccec450d5eb834bc17fb7cf5989ea7497f7bbd06fd9102422b42bbc`.
Its 39 selected binding tests passed (still images, multiframe, contexts and
configured official decoder interoperability). No ABI or Python API changed.

### Native Encode Time

T8/e9, one warmup and five measured calls per binary/case, plus a separate profile:

| Input / fixed family | Before s | After s | Time change | MPix/s before -> after | Both bytes | Both bpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `9bde...` / automatic | 1.953789 | 1.736212 | -11.14% | 0.7453 -> 0.8387 | 412541 | 2.266510 |
| `n6qzmx` / automatic | 2.921654 | 2.618091 | -10.39% | 0.7097 -> 0.7920 | 459153 | 1.771424 |
| `1eb...` / cross6 | 4.946733 | 4.806885 | -2.83% | 0.7281 -> 0.7493 | 351304 | 0.780259 |

Reports: `memory-t8.json` and `memory-cross6-t8.json` in the same output directory.
Batch was observed active in both checkpoints. These are whole-combination results;
they do not isolate how much each of the three retained changes contributes.

The T2 check on `9bde...` used one warmup and three measured calls per binary:
5.118416 -> 4.779030 s (-6.63%), 0.2845 -> 0.3047 MPix/s, unchanged
412541 bytes / 2.266510 bpp. Both profiles used two effective workers.
Report: `memory-t2.json`.

### Memory Work

The new profile reports `frontier_slots` (initialized high-water mark) and
`frontier_stride` (144 bytes on this target). The former code initialized
`min(tokens, 16384)` frontiers of 272 bytes for every admitted stream.

| T8 case | Prior initialized structure bytes | New initialized structure bytes | MiB before -> after |
| --- | ---: | ---: | ---: |
| `9bde...` | 102498304 | 16515072 | 97.750 -> 15.750 |
| `n6qzmx` | 66846720 | 27131904 | 63.750 -> 25.875 |
| `1eb...` | 26738688 | 2949120 | 25.500 -> 2.813 |

These are cumulative initialized data-structure sizes, not measured physical
write traffic, simultaneous RSS or cache misses. Per-stream maximum frontier
reservation falls from 4.25 to 2.25 MiB; index and DP allocations on the tested
default-budget routes are unchanged. The memory-bound admission calculation now
uses the actual compact type. A tighter explicit budget may consequently admit
a refined index where the previous, larger frontier required a coarse index.

### Controls and Correctness

T8 inactive controls, one warmup and three measured calls per binary:

| Input | Before s | After s | Time change | MPix/s before -> after | Both bytes | Both bpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Photo `kodim01` | 0.949745 | 0.946154 | -0.38% | 0.4140 -> 0.4156 | 454218 | 9.241089 |
| Illustration `2026.6test1` | 1.745659 | 1.763833 | +1.04% | 1.1879 -> 1.1756 | 2098189 | 8.094865 |
| RGB16 `basn2c16` | 0.007568 | 0.007159 | -5.41% | 0.1353 -> 0.1430 | 153 | 1.195313 |

Report: `memory-controls-t8.json`. These profiles correctly bypass Batch; the
illustration's small slowdown is retained rather than filtered out. The tiny
RGB16 timing difference is less than half a millisecond, not evidence of a new
optimization on that bypassed route.

All **82 final comparison outputs**, including baseline, candidate, warmups and
profiles, passed official libjxl 0.12 decoding against original samples. All
measured codestream sizes were unchanged. An additional **five production CLI**
checks passed: e9/T8 active on the two main images, e9/T1 and e3/T8 inactive on
`9bde...`, and e9/T2 inactive on RGB16. Report: `production-memory.json`.
Production checks explicitly used `--strip-metadata` for unsupported input `pHYs`;
they validate pixels, not preservation of stripped metadata.

Keep the memory changes, but do not identify memory bandwidth as the proven root
cause of the earlier slowdown: the longer-evidence trial also shifted useful work
from speculative parallel queries to demand queries on the parser's path. Work
counts alone do not describe critical-path time. Host-load variation remains a
limitation, and full-corpus/Linux validation was not repeated here.

## Reproduce a Focused Comparison

Use two prebuilt examples and a new output filename; the tool never builds them
and refuses to overwrite an existing report. Both receive the same Batch flags.

```powershell
python -B tools/benchmark_chain_checkpoints.py --before out/chain-batch-feedback-20261111/encoder-lab-before.exe --after out/chain-batch-feedback-20261111/encoder-lab-memory.exe --official-djxl ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe --case ../jxl_python/9bde6aaf390109f97622d35ff1e900ee.png --threads 8 --warmups 1 --repeats 5 --profile --out out/chain-batch-memory-recheck.json
```

Use `--family cross6` for the recorded `1eb...` candidate. Use `--expect inactive`
for serial/bypassed controls. Inactive controls must not be presented as evidence
that the optimized Batch kernel executed.
