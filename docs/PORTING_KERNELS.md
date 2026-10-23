# Ported Encoder Kernels

**Direct ports connected to the Rust encoder; full migration is still in progress.**

These Rust kernels migrate bounded pieces of our frozen C++ fork. The rejected
third-party-codec bootstrap was deleted, not retained as an encoder backend.
The active pixel path includes learned MA trees, LZ77, transforms, patches and
multiframe framing. Reversible-JPEG now has its separate coefficient encoder
and original-byte validation; see JPEG_TOOL_PORT_STATUS.md. Native Rust SIMD is
isolated in prefix, MA, predictor, cost and RCT modules with scalar fallbacks.
Helper tests do not establish end-to-end acceleration. See
[the migration map](MIGRATION.md) and [current coverage](PIXEL_LOSSLESS_PORT_STATUS.md).

## Provenance

Read-only source: `../jxl_python/libjxl`, commit
`b866b3defec13494bfbaf705db93581cda118e37` (`b866b3d`). The helper ports used
`git show` / `git grep` against that commit. Later framing/pixel ports inspected
the corresponding unchanged tracked source read-only; unrelated dependency
worktree changes are not source-port inputs.

| Rust module | Frozen source and migrated behavior |
| --- | --- |
| `crates/rs-jxl/src/tools/lz77_prefix.rs` | `lib/jxl/enc_lz77.cc`: `GetHash<3>`, `LZ77HashMap::BuildPrefixFilter`, pending-prefix activation/work feedback in `FindMatch`, and the mandatory-prefix-only rule in `LZ77BucketMatcher::Update`. |
| Same module | `lib/jxl/enc_lz77_bucket.h`: `LZ77RecentBuckets` fingerprint bookkeeping, uniform-tag shortcut, newest-first ring traversal. `lib/jxl/enc_lz77_bucket.cc`: scalar equivalents of `FingerprintMask` and `SpecialPrefixMasks`. |
| `crates/rs-jxl/src/tools/ma_entropy.rs` | `lib/jxl/modular/encoding/enc_ma_entropy.h`: `MaClippedEntropy` Fenwick mode and probability-floor formula. `lib/jxl/modular/encoding/enc_ma.cc`: `CollectExtraBitsIncrease` weighted counts/extra bits and split-sweep count transfer. `lib/jxl/ans_params.h`: 12-bit, 4096-entry ANS probability floor. |

The crate root registers these modules; Cargo is the central test entry point.

## Prefix Filter Contract

- `prefix_hash` uses exactly three token **values**, with explicit wrapping
  `u32` arithmetic matching the source. Contexts and optional fourth through
  eighth tokens do not enter the tag. An incomplete prefix returns `None`.
- Tags are the top eight hash bits. Equal mandatory prefixes necessarily have
  equal tags, so rejecting unequal tags cannot reject a valid match of length
  at least three. Equal tags, including collisions, still need exact token
  comparison. This guarantee is relative to the supplied candidate set; bucket
  eviction can still remove matches, exactly as without filtering.
- `PrefixFilter` borrows one immutable stream, preventing cached tags from
  becoming stale. Minimum match lengths below three and incomplete query-tail
  prefixes bypass filtering. Invalid positions produce errors, not unchecked
  reads. The native SIMD implementation checks complete lanes before each load;
  tails use the scalar path. Unsafe operations are isolated there.
- `Lazy` starts with no tag allocation. Pending eligible-candidate work is
  saturated at `u64::MAX`, independently of profiling. At the next query start,
  the threshold is `max(4096, floor(token_count / 2))`. Build only if the query
  position is strictly below `floor(token_count / 2)`; otherwise permanently
  enter `SkippedLate`. No new rejection heuristic is introduced.
- `visit_special` performs that query-start check and then counts eligible
  distances itself. Callers using `begin_query` / `may_match` directly must call
  `record_candidate_work` for actual candidate work, not successful matches or
  sampled query counts. Do not double-count work when using `visit_special`.
- `group_distances` requires positive, unique, ascending physical distances.
  Groups contain up to 16 distance slots. Descending set-bit traversal visits
  surviving distances in ascending order, including partial first rows. Use
  the groups in their original order. Near-`usize::MAX` group endpoints saturate
  rather than wrap; only explicitly supplied valid slots can be visited.
- Width-dependent distance generation and canonical distance-symbol lookup are
  intentionally outside the module. Canonicalize duplicate physical distances
  to their smallest distance symbol before grouping, especially for widths one
  and two. `safe_distance` must be justified by the caller's retained-bucket
  coverage; the helper cannot prove that coverage on the caller's behalf.
- `RecentBucket<N>` supports power-of-two capacities 1 through 64. It records
  tags from the first insert and preserves newest-first order across ring wrap.
  Pass the mandatory-three-token hash even when the enclosing bucket key comes
  from a longer prefix. Query with `None` when there is no complete prefix or
  when filtering is disabled. Insertion expects a complete-prefix hash; callers
  need not index terminal positions that cannot start a three-token match.
- The one-bucket primitive does not allocate a hash table, generate candidates,
  enforce windows, compute exact match lengths, select symbols, or parse LZ77.
  It does not reproduce the other fast matcher's physical-slot traversal: use
  it only where the existing candidate order is newest-first.

## MA Entropy Contract

`ClippedEntropy` owns nonnegative `u32` symbol counts and their exact total.
Its declared `maximum_total` bounds updates and the frequency-of-frequencies
Fenwick tree. Scratch has at most 2^20 rare-count entries, even at `u32::MAX`.
Signed 64-bit tree masses accommodate count removal without unsigned wrap.

The modeled symbol cost is:

```text
sum over count > 0: count * min(log2(total / count), 12)
```

The maintained sum is `sum(count * log2(count))`. Counts at or below
`(total - 1) / 4096` are the strictly clipped tail. Fenwick prefix sums recover
that tail's mass and entropy without rescanning all symbols. The formula and
`f32` result match the source model; floating-point arithmetic is not a claim
of bitwise identity across platforms or libm implementations. Tests compare to
direct probability-floor recomputation with a stated tolerance.

`replace`, `add_weight`, and `remove_weight` validate symbol bounds, count
arithmetic, and maximum total before mutating state. `move_weight_to` validates
both histograms before transferring a symbol's full sample multiplicity.
Neither histogram changes on error. The caller still owns split order and
non-histogram costs.

`add_weighted_sample` is the single-property-bin form of the source histogram
update: add the entire `u16` multiplicity to the selected count, and
`weight * extra_bits` to the `u64` extra-bit sum. It does not count each distinct
sample as one, cap its contribution, or rescale samples. Each call is atomic on
error; a caller's loop is not a transactional batch. The caller selects the
property bin, residual symbol, and extra-bit count before calling it.

This entropy helper is not the entire MA learner. Property/predictor mapping,
histogram reuse, sample deduplication, parallel split search and tree sharding
live in `modular/ma_train.rs`, `training.rs` and `tree_groups.rs`; tree and entropy
serialization have their own modules. Native count/entropy kernels are in
`tools/ma_simd.rs`. See module source maps for individual arithmetic contracts.

## Verification

The original 19 semantic helper tests covered the following contracts. Current
tests also exercise the integrated encoder; no JXL output-hash equality is an
encoding acceptance condition:

- Wrapping prefix arithmetic against a separate widened-integer reference;
  deliberate tag collisions; empty, short, tail, and invalid positions.
- 0/64/65-byte mask bounds; narrow-width-derived distance sets; partial groups;
  checked maximum distances; minimum-length bypass and ascending survivor order.
- Disabled/eager/lazy policies produce identical exact match sequences after
  token comparison, including activation transitions and short matches.
- Activation threshold, midpoint refusal, saturating work counters, actual
  eligible-work accounting, and no allocation before activation.
- Bucket wrap, eviction, capacities 1/2/8/64, invalid capacities, uniform-tag
  recovery, and long-bucket tags that ignore optional prefix tokens.
- Empty/constant histograms; probability-floor boundaries around multiples of
  4096; arbitrary updates and forward/reverse split sweeps against direct cost.
- Weighted samples against literal expansion, including 65535 multiplicity,
  zero weight, weighted extra bits, overflow/underflow, atomic errors, and a
  maximum-`u32` total.

Run current tests through Cargo (native SIMD prevents forbidding all unsafe):

```powershell
cargo test --locked -p rs-jxl
cargo test --locked --release -p rs-jxl
cargo clippy --locked --all-targets -- -D warnings
```

The original 19 scalar helper tests passed before adding native SIMD. That is
historical helper evidence only, not a result for the current full worktree.
Current consolidated results are recorded in `PIXEL_LOSSLESS_PORT_STATUS.md`.

The parent should run `cargo test -p rs-jxl tools::` after module registration,
then run workspace integration checks centrally. Before activating any kernel,
connect it to our migrated matcher/MA training path, preserve candidate and
distance-symbol tie ordering, verify decoded pixels and reversible-JPEG bytes
and frame semantics, and measure on representative workloads. Do not silently
substitute an existing encoder or mark the native migration complete.

## Retained Upstream License

The derived Rust kernels retain the following frozen-fork license. Include this
notice with source distributions and reproduce it in binary distribution
documentation or other accompanying materials.

```text
Copyright (c) the JPEG XL Project Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
