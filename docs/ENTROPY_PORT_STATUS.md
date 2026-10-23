# Entropy Port Audit

Baseline: read-only `../jxl_python/libjxl`, commit
`b866b3defec13494bfbaf705db93581cda118e37`. Audit date: 2026-10-23.
Scope: `crates/rs-jxl/src/entropy/*` only. No JPEG conversion, execution-policy
module, source-tree, dependency, or other subsystem changes were made here.

This is a source-path audit, not acceptance of the complete pixel-lossless port.
"Optional" below identifies a source feature gate, not permission to omit it
from a milestone that requires that feature. Compressed image byte equality is
not an acceptance goal; original decoded pixel samples remain the goal.

## Integrated Source Paths

| Frozen source | Rust implementation and production integration |
| --- | --- |
| `enc_ans_simd.cc::EstimateTokenCostImpl` | `simd.rs`: AVX2 integer-to-float exponent extraction, ten-bit repair above `2^22-1`, literal masks, MSB/LSB token construction and extra-bit widths. Used by raw trials, weighted-bin construction/trials, and final length-token counting. Scalar handles tails and unsupported CPUs/configurations. |
| `enc_cluster.cc::HistogramCondition` | `simd.rs::condition`: SIMD sum, trailing-zero extent and checked signed-source weight domain. Used by clustering and adaptive candidates/winners. Public u64 storage stays intact. |
| `enc_cluster.cc::Entropy`, `HistogramEntropy` | Eight-lane AVX2 source rational log and source reduction order, plus non-fused scalar fallback. Used by clustering and the source prefix-selection entropy threshold. |
| `enc_cluster.cc::HistogramDistance`, `HistogramKLDivergence` | Integer addition before float conversion, absent-symbol infinity, zero-actual masks, source subtraction order and KL `NegMulAdd`. Used in seed/assignment and preexisting-center refit. Prefix singleton penalties are retained. |
| `base/fast_math-inl.h`, `base/rational_polynomial-inl.h`, Highway `generic_ops-inl.h` | Frozen f32 coefficients; Horner `MulAdd` fused only on the FMA path; real division; cross-block sum then `Reverse4`/`Reverse2`. Entropy multiply/subtract and running addition remain non-fused. |
| `enc_ans.cc::ChooseUintConfigs` | Original Fast/Best candidate order, capacity and LSB reservation checks, population plus signaling costs, strict winner comparison, separate literal/length transposition, pointer swaps for winning raw counts, and final alphabet canonicalization. The internal model entry point passes these counts straight to table construction. |
| `enc_uint_histogram.h` | Refining `(12,2,5)` bins retain exact representative token/width counts for supported candidates. SIMD is active in both refinement and candidate evaluation; no sampling or candidate pruning. |
| `enc_lossless_workers.h`, `enc_ans.cc`, `enc_cluster.cc` | `jobs.rs`: at most eight independent roots per entropy dispatch, 1 GiB conservative scratch budget, at least one oversized job, stable result/error order, atomic largest-first search queue, caller pool only, no recursive entropy job dispatch. Seed chunks and initial refinement rows honor the local cap. |
| `enc_ans.cc::BuildAndEncodeHistogramsImpl` | Explicit `EntropyOptions::add_missing_symbols` adds one to every bin 0..255 in every raw context, including present symbols and the distance context. Explicit `add_fixed_histograms` installs a flat 256-symbol table before clustering. Both default to false. |
| `enc_ans.cc::HistogramFromSymbolInfo`, `BuildAndStoreEntropyCodes`, `EncodeHistograms` | `Model::append_with_options` reconstructs old coding weights, keeps old tables immutable, appends local context ranges and new tables, and caches exact histogram bit lengths. `write_stream_at` selects a stored batch; deferred `write_header` replays cached bits. |
| `enc_frame.cc::RemoveUnusedHistograms` | First-occurrence compaction moves configurations, coding tables and cached histograms together. Header emission also uses a compact wire view without dropping reusable centers from the model. |

Fixed/streaming uint policies retain their non-transposing scans and reuse the
clustered raw counts, including injected bins. Adaptive model building does not revisit raw literal tokens after
the winning trial. Empty reused winners receive the same final symbol-zero
coding fallback as empty freshly counted tables, including prefix tables.
The public `ConfigPlan` API is unchanged.

## Frame-Area Applicability And Parent Hook

The actual default-eligible source Modular streaming route does NOT reach
missing-symbol injection. This is a call-chain conclusion, not an assumption
that streaming is optional or lossy:

1. `enc_frame.cc:1903` (`CanDoStreamingEncoding`) can admit lossless Modular
   frames with more than eight groups, subject to its other gates. It explicitly
   rejects JPEG input. More than eight groups alone is not sufficient.
2. `EncodeFrameStreaming` creates a fresh `ModularFrameEncoder`; its area loop
   at `enc_frame.cc:2252` sets streaming mode and initializes global state only
   for the first area. `ComputeEncodingData` guards `ComputeTree` and
   `ComputeTokens` with `!streaming_mode` at `enc_frame.cc:1786`.
3. `EncodeGlobalInfo` is called during initial global output
   (`enc_frame.cc:1469`). No shared tree tokens were built, so
   `enc_modular.cc:1701-1709` writes the no-global-tree bit and returns via
   `skip_rest`. The assignment at line 1725 is NOT executed on this route.
4. `EncodeStream` sees empty `tokens_` and calls `ModularGenericCompress`
   (`enc_modular.cc:1755`). Local options still inherit the streaming uint
   policy from `ForModular` (`enc_modular.cc:926`); they do not inherit the
   unreachable `EncodeGlobalInfo` injection assignment.
5. Rust `modular/orchestration.rs::encode_frame_areas` likewise sets only
   `streaming_mode`, omits the shared ComputeTree/ComputeTokens stage, and
   encodes the global section and local streams through
   `frame::encode_prepared_independent`. Its local models are independent.
   There is no shared entropy-model divergence to fix on this route. This
   does not certify the parent-owned eligibility/layout/output policy as a whole.

No orchestration/frame change is requested for current frame areas. Do NOT
enable either injection flag for these independent local models. If a caller
actually supplies a nonempty shared global tree in streaming mode, enable
`add_missing_symbols` only for its leaf-token model after tree-token entropy,
matching `EncodeGlobalInfo`'s placement. Non-streaming shared trees keep it false.

The genuine current source user of incremental/fixed tables is the streaming
VarDCT `EncodeGlobalACInfo` path (`enc_frame.cc:1330-1390`), followed by
`RemoveUnusedHistograms`/`OutputAcGlobal` at lines 2080/2110. These APIs are
explicit reusable entropy tools, not newly enabled Modular frame-area behavior:

```rust
// Initial global state: choose Coding once; retain this Model across batches.
let mut options = EntropyOptions { streaming_mode: true, ..base_options };
options.add_fixed_histograms = true; // VarDCT fixed-center caller, not local Modular.
let mut model = Model::build_with_options(&first, contexts, lz77, coding, &options)?;
model.write_stream_at(&first[0], 0, first_payload)?;

options.add_fixed_histograms = false;
options.max_histograms = new_table_budget; // May be zero; total stays <= 256.
let range = model.append_with_options(&next, next_contexts, lz77, &options)?;
model.write_stream_at(&next[0], range.start, next_payload)?;
model.remove_unused_histograms(); // After final append; earlier payloads stay valid.
model.write_header(global_header)?; // Final combined context map and cached tables.
```

Exact additions: both flags are `bool`; `append_with_options(&mut self,
streams: &[Vec<Token>], contexts: usize, lz77: bool, options: &EntropyOptions)
-> Result<Range<usize>>`; `write_stream_at(&self, tokens: &[Token],
context_offset: usize, writer: &mut BitWriter) -> Result<()>`;
`remove_unused_histograms(&mut self) -> usize`; `histogram_count()` and
`context_count()` return `usize`. Context IDs in each batch remain local;
`contexts` includes that batch's distance context when LZ77 is enabled.

Initial build chooses prefix/ANS once; append preserves it and the uint/length
mapping. Append rejects repeated fixed injection and changed LZ77 enablement,
length config or reserved start. Parsing and consistent LZ77 policy remain
the caller's responsibility, not an automatic incremental ApplyLZ77 port.
Failed appends leave logical model state unchanged. Prefix coding weights
follow source depth semantics, including zero weight for zero-depth singletons;
a zero-new-table refit without symbol support therefore fails explicitly.
Non-streaming adaptive uint trials replace injected counts with trial counts,
as in source; guaranteed unseen-symbol support uses the fixed/streaming policy.

## Arithmetic And Storage

- SIMD selection requires runtime AVX2 detection; fused polynomial/KL loops
  additionally require FMA. Neither instruction set is a build prerequisite.
- No blanket f64-to-f32 conversion: normalization targets, logarithm-table
  construction, merge lower bounds and existing exact wire-cost estimators keep
  their existing source-appropriate arithmetic.
- Source histogram weights are checked against `i32::MAX`. The existing Rust
  extension for a combined distance total above that bound uses safe scalar
  arithmetic instead of source signed overflow.
- Extra-bit sums keep the existing Rust u64 contract rather than introducing
  source uint32 wraparound above `2^32` bits.
- Count vectors expose exact nonzero extent, not target-dependent padded
  lengths. Tail loads never require caller-provided padding or overreads.
- Token conversion uses bounded 256-value batches. MSB widths above twelve
  retain scalar encoding because the source's ten-bit large-value shift would
  discard required mantissa bits; all source search candidates use at most two.
- Weighted search, entropy worker execution and the already-ported merge bound
  remain enabled by local eligibility checks, without C++ experimental env
  gates. This preserves the existing Rust policy, not the source gate defaults.

## Remaining Gaps

### Default Pixel-Lossless Path

1. Final centralized tests and independent decoder suites must be rerun after
   this edit set. The prior 741-test/all48-decoder checkpoint predates these
   changes and is not evidence for the new kernels. No performance gain is
   asserted without a representative benchmark.
2. The entropy-local eight-root/scratch cap is not a process-wide scheduler
   contract. Simultaneous parent-owned encoder roots can each initiate a local
   entropy dispatch. Coordinating independence and shared scratch permits with
   other subsystems requires the parent-owned `execution.rs`/scheduler work;
   this change does not resize or replace the executor.
3. Native entropy SIMD is AVX2/FMA. SSE-only x86 and non-x86 targets execute the
   complete scalar algorithms. Native SSE, NEON/SVE, RVV, WASM SIMD and wider
   AVX-512 kernels are still target-specific optimized-port gaps, not missing
   pixel-decoding functionality. ARM64 compilation is not ARM64 execution or
   SIMD-performance evidence.

The named generic entropy algorithms and adaptive winner/search paths are
implemented. The caller trace above resolves the specific missing-symbol
concern for default-eligible Modular frame areas; it is not a blanket claim
that all default encoder policy or all optimized target paths are complete.

### Optional Source Features

- Direct-audit contiguous bucket selection, direct entropy audit records,
  final-plan record/replay (including prefix/cluster/uint/LZ77 exchanges), and
  `SetANSFuzzerFriendly` are not ported by this change. ANS Direct0/Direct6
  normalization itself was already implemented.
- `JXL_LOSSLESS_WORKER_LIMIT` and `JXL_LOSSLESS_WORKER_MEMORY_MIB` environment
  overrides are not read. Entropy uses the source default eight-root/1-GiB
  limits; overriding them needs an explicit owned policy/API decision.
- `AuxOut` layer accounting and exact source profiling/audit schema remain
  separate from the Rust profile hooks. Context-map representation choice uses
  already-implemented measured serialized candidates, not C++ dry-cost parity.

### Lossy-Only Or Non-Pixel Callers

- VarDCT caller integration for the now-available incremental model/fixed-table
  API remains outside this entropy ownership. Generic table/context lifecycle
  is implemented, not VarDCT per-area histogram budgets or AC global output.
  Future JPEG callers may reuse these tools, but the frozen source explicitly
  disables this streaming frame route for JPEG input.
- The VarDCT `enc_context_map.cc::EncodeBlockCtxMap` wrapper is not implemented
  in this entropy module. Generic context-map encoding is implemented. A future
  JPEG coefficient path may also need that caller-specific wrapper; it must not
  be mistaken for a missing default Modular/pixel-lossless path.
- No lossy quantization or DCT-specific histogram preparation was added.
  Near-lossless-labelled uint candidates already belong to the shared Best
  list and remain supported; they are not silently pruned here.

## Focused Verification

Added nine kernel/search tests and seven lifecycle tests, for centralized execution:

- Token dispatch versus scalar `HybridUintConfig::encode` over every valid
  configuration, power-of-two/large-u32 boundaries, pseudo-random values, tails
  and untouched output sentinels.
- Conditioning extent/total versus scalar, including oversized signed-domain
  weights, `u64::MAX`, high-bit values and tails.
- Both AVX2 FMA/non-FMA entropy and KL against an explicit eight-lane source
  arithmetic/reduction oracle, including absent-support and unused-symbol cases.
- Empty/singleton entropy and safe large combined-distance totals.
- Worker-count/scratch-limit boundaries and ordered nested-call behavior.
- Serial/parallel winning histogram reuse versus an independent recount,
  including shared contexts, empty histograms, weighted thresholds and LZ77.
- Fixed/streaming plans returning no adaptive histogram allocation.
- Source singleton entropy checks and zero-winner prefix fallback.
- Explicit missing-symbol counts, including present bins, unused contexts and
  LZ77 distance contexts; streaming by itself does not inject symbols.
- Fixed-table and multi-batch payload readback from the actual unaligned wire
  header using independent prefix/ANS test decoders and final-state checks.
- Immutable old payloads/tables/caches, unused fixed-center compaction, and
  cached versus direct histogram emission at a non-byte boundary.
- Atomic failed append, missing support, configuration/mode restrictions,
  local context-offset bounds, per-batch LZ77 distance contexts, prefix
  zero-depth coding weights, and initial/injected alphabet option bounds.

No Cargo build, test or formatting command was run by this entropy worker.
The parent reported Windows MSRV 1.89 all-target checking and Linux ARM64
all-target checking passing an intermediate snapshot; the ARM64 unused test
helper warning was corrected with the matching target cfg. Final checks after
the subsequent length-transpose/reuse and bounded lifecycle integration edits
remain the parent's responsibility. No commits were made.
