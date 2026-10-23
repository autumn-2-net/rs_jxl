# AArch64 NEON Port Checkpoint

Source checkpoint: implemented, awaiting parent compilation and execution.
No Cargo command, build, test, benchmark, commit, or C++ modification was run
for this checkpoint. Scoped `rustfmt` and `rustfmt --check` passed on the four
owned Rust files, with child-module traversal disabled.

## Scope and Source

Frozen read-only source: `../jxl_python/libjxl`, commit
`b866b3defec13494bfbaf705db93581cda118e37`. Its Highway submodule is pinned to
`457c891775a7397bdb0376bb1031e6e027af1c48`.

| Rust implementation | Direct source |
| --- | --- |
| `crates/rs-jxl/src/modular/rct_neon.rs` | `lib/jxl/modular/transform/enc_rct.cc`: `RctPermute`, `FwdRctRow`, `FwdRctImpl` |
| `crates/rs-jxl/src/entropy/neon.rs`: tokens | `lib/jxl/enc_ans_simd.cc`: `EstimateTokenCostImpl`; scalar tail uses `dec_ans.h::HybridUintConfig::Encode` |
| entropy conditioning and estimates | `lib/jxl/enc_cluster.cc`: `HistogramCondition`, `Entropy`, `HistogramEntropy`, `HistogramDistance`, `HistogramKLDivergence` |
| log polynomial | `lib/jxl/base/fast_math-inl.h::FastLog2f`, `base/rational_polynomial-inl.h::EvalRationalPolynomial` |
| target arithmetic/reduction | `third_party/highway/hwy/ops/arm_neon-inl.h`, `ops/set_macros-inl.h` |

Only these new local modules, their existing `entropy/simd.rs` and
`modular/rct_simd.rs` dispatch parents, and this document are owned/edited.
The existing BSD attribution remains in each translated file and
`docs/PORTING_KERNELS.md`. No utility crate, SIMD binding, Highway binding,
native codec linkage, subprocess encoder, or build-system change was added.

## Integration Contract

- Public/crate-visible signatures are unchanged. No parent module registration
  is needed: `#[path = "neon.rs"]` and `#[path = "rct_neon.rs"]` resolve local
  children, each guarded by `#[cfg(target_arch = "aarch64")]`.
- The crate already requires `std`; there is no `std` Cargo feature to gate.
  Explicit `std::arch::is_aarch64_feature_detected!("neon")` dispatch precedes
  `#[target_feature(enable = "neon")]` kernels. Baseline NEON targets work
  without new feature flags; other architectures never import AArch64 APIs.
- RCT `forward` and `forward_rows` use NEON for full four-pixel groups;
  `forward_scalar` still bypasses native kernels. The internal
  `neon::run<const T: u32>(&mut [&mut [i32]; 3], [usize; 3]) -> usize` returns
  the processed prefix, leaving 0..3 pixels to the existing scalar loop.
  Validation, in-place storage, channel permutations, row scheduling and the
  caller's serial/borrowed-pool behavior remain unchanged.
- Entropy `tokens`, `add_tokens`, `condition`, `entropy`, `combined_entropy`
  and `coding_cost` use the existing interfaces. The child exposes only
  `pub(super)` native entry points. Distance/KL final subtraction remains with
  the callers; the child returns combined entropy or coding cost, respectively.
- All x86 AVX2/SSE2 bodies, x86 feature dispatch and encoder defaults are
  preserved. No MA, LZ, sample, FJXL or metadata implementation was touched.

## Exactness and Bounds

- All 42 RCT types are covered: six input permutations and seven source
  transforms, specialized outside the pixel loop. Four signed-i32 lanes use
  wrapping add/subtract **before** arithmetic shifts. This preserves negative
  odd rounding and the entire i32 domain, including raw binary32 sample words.
  All three inputs are loaded before any in-place store.
- HybridUint uses the source unsigned comparisons, shift-by-ten large-value
  handling, exact u32-to-f32 conversion of the reduced values, exponent
  extraction and bit assembly. Configurations with `msb_in_token <= 12` use
  NEON; the existing integer scalar path preserves exactness for wider public
  configurations. Token widths are also returned. Extra-bit accumulation
  widens to u64, preserving the Rust interface rather than source u32 wrapping.
- Conditioning validates u64 weights before narrowing and checks the widened
  running total against `i32::MAX` each block. It returns the exact nonzero
  extent, not Highway's padded extent. Entropy/coding-cost callers retain the
  established validated histogram contract. Combined totals above `i32::MAX`
  continue through the existing overflow-safe scalar extension.
- Histogram arithmetic uses four lanes, matching NEON's full/capped source
  descriptors (the cap is eight, but NEON has four f32 lanes). Combined counts
  are added as integers before conversion. The frozen f32 polynomial constants
  are shared with the existing code. `vfmaq_f32` implements Horner MulAdd;
  `vdivq_f32` performs actual division, without a new log approximation.
- Entropy multiply, subtract-from-zero and accumulation remain separate.
  Coding cost uses `vfmsq_f32` for source NegMulAdd. Zero actual counts are
  masked before that operation; missing positive coding support yields
  infinity, not `0 * infinity` NaNs. Entropy retains the source comparison in
  f32 against the total, including integer counts that round to the same f32.
- AArch64 reduction is `(lane0 + lane1) + (lane2 + lane3)` via `vaddvq_f32`,
  not the x86 reverse-shuffle order. Different lane counts/FMA can legitimately
  change estimate bits versus scalar/x86. Existing standalone scalar
  `fast_log2` and large-total fallback behavior are not changed.
- No vector accesses caller padding. RCT/tokens process only complete groups
  and use scalar tails; histogram loads zero-fill short/missing lanes in local
  four-element arrays. Loads require only the element alignment guaranteed by
  Rust slices, not 16-byte alignment. Vector stores never touch scalar tails
  or sentinels.

## Parent Verification Pending

The new oracle tests are compiled only on AArch64 and runtime-guard NEON:

- `modular::rct_simd::tests::neon_all_types_match_integer_oracle_with_unaligned_tails`:
  all 42 types, independent i64/modulo/floor-division oracle plus inverse,
  signed extrema, negative odd values, independent plane offsets and sentinels.
- `entropy::simd::neon::tests`: source-order polynomial/reduction oracle,
  integer token oracle across all native configurations and exponent boundaries,
  unaligned outputs/tails, checked conditioning, mismatched histogram lengths,
  empty/singleton/missing support, and integer-to-f32 rounding boundaries.
- Existing portable dispatch tests additionally cover all public HybridUint
  configurations, scalar tails, large combined totals and RCT wrapper/row-pool
  parity. Their AArch64 run exercises the new dispatch automatically.

Suggested central checks, **not executed by this agent**:

```text
cargo +1.89.0 check -p rs-jxl --lib --tests --target aarch64-pc-windows-msvc
cargo check -p rs-jxl --lib --tests --target aarch64-unknown-linux-gnu
```

Use whichever AArch64 target std libraries are installed centrally. `check`
type-checks the cfg-gated test bodies without requiring an AArch64 linker;
it does not run the oracle tests. On an AArch64 host/configured runner, run
the `entropy::simd::` and `modular::rct_simd::` library tests in debug and
release modes, plus the parent's normal x86 regression and codec oracle suite.

No AArch64 compilation/runtime, guard-page instrumentation, performance or
end-to-end codec acceptance is claimed here. ARMv7, SVE, inverse production
RCT and other agents' kernels are outside this checkpoint. Broader floating
sample format correctness, signaling and exact decoded sample preservation
are independent acceptance work: histogram-estimate parity or raw-word RCT
parity does not establish those properties.
