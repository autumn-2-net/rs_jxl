# JPEG DCT8 Coefficient Order Prerequisite

Frozen source: read-only `../jxl_python/libjxl`, commit
`b866b3defec13494bfbaf705db93581cda118e37` (verified). Date: 2026-10-23.
Owned files: `jpeg/coefficient_order.rs` with inline tests, and this document.
No registration, Cargo, entropy, layout, metadata, reader/writer, or C++ files
were changed by this worker.

This is a bounded JPEG-to-JXL prerequisite, not a VarDCT encoder or completed
transcoder. The pixel-lossless phase remains in progress. The parent's passing
nine layout tests and 107 JPEG tests precede this module and are not evidence
for its new order-selection code.

## Source Mapping

| Frozen source | DCT8 Rust port |
| --- | --- |
| `coeff_order_fwd.h`, `coeff_order.h` | u32 coefficient positions; order bucket zero/bit one, offsets corresponding to three consecutive 64-entry channel orders, eight permutation contexts. Other twelve order buckets are outside scope. |
| `ac_strategy.cc::CoeffOrderAndLut` | DCT8 natural zigzag and inverse rank LUT. The numeric order equals the first 64 existing `jpeg::data::NATURAL_ORDER` entries, reused rather than copied into another unrelated table. |
| `enc_coeff_order.cc::ComputeUsedOrders` | DCT8 presence/customization masks, exact speed-tier and small-image branch order. |
| `enc_coeff_order.cc::ComputeCoeffOrder` | Borrowed i16/i32 AC image groups, source seeded sampling, zero counts, LLF/DC pinning, f32 count quantization, zigzag tie keys and removal of the transmitted bit when all three orders remain default. |
| `enc_coeff_order.cc::EncodeCoeffOrder` / `EncodeCoeffOrders` | Inverse-natural-order lookup, skip one DC rank, channel 0/1/2 token concatenation, and no tokens when used-orders is zero. Entropy emission remains separate. |
| `enc_coeff_order.cc::TokenizePermutation` | Trailing-zero Lehmer trimming, length token, previous-digit contexts and ordinary entropy `Token` construction. |
| `lehmer_code.h::ComputeLehmerCode` | Checked O(n log n) Fenwick prefix sums/updates, with explicit uniqueness/range validation before indexing. |
| `coeff_order.cc::CoeffOrderContext` | Existing `HybridUintConfig(0,0,0)` token, clamped to context seven. |

## Input Contract

`Dct8Image::new(width_blocks, height_blocks, groups)` validates a borrowed view:

- Dimensions are nonzero; checked multiplication and the source Level-10
  `2^40`-pixel / `2^34`-DCT8-block count bound are enforced.
- Groups are raster ordered in 32x32-block regions. The number of groups must
  equal `ceil(width/32) * ceil(height/32)`.
- A `Dct8Group::I16` or `Dct8Group::I32` holds three borrowed channel slices.
  Every group has one common AC image integer type; mixing types is rejected.
- Each channel contains consecutive 64-entry blocks in JXL TRANSPOSED layout,
  in the source ACImage's packed per-group order. Complete extra storage is
  allowed, matching source fixed-capacity edge groups; the required geometry
  prefix must exist and slice length must be a multiple of 64.
- Data must be the actual pass coefficients AFTER applicable JPEG CfL residual
  formation and progressive splitting, with source zero padding. This API does
  not consume original component coefficient arrays as though they were that
  ACImage. In particular, compact subsampled component planes and grayscale
  need the caller's source-compatible AC image packing/zero planes.

No coefficient buffers are allocated or cloned by the production module.
Counting uses three 64-entry signed counters; sorting uses 64 positions per
channel. Permutation scratch is bounded to a 65-entry Fenwick tree plus at most
64 Lehmer digits. Only the returned order/token data is owned.

The default numeric zigzag is `[0,1,8,16,9,2,...]` IN THE TRANSPOSED JXL BLOCK.
Reading original JPEG natural-order storage at those offsets would be wrong.
For an original block containing positions 0..63, correctly transposed input
read in default JXL order starts `[0,8,1,2,9,16,...]`. The order itself must not
be transposed a second time.

## Selection And Arithmetic

`OrderPolicy::Source` uses the literal frozen `ComputeUsedOrders` branches.
For nonempty DCT8 geometry the present mask is always one. Below Falcon,
customization is disabled only when BOTH block dimensions are below five.
Falcon and faster return `(1,1)` before that small-image check. The source comment
says no reordering there, but its executable branch enables the candidate bit;
this module follows the code rather than silently changing that behavior.

`OrderPolicy::Default` corresponds to `current_used_orders=0` (including the
source direct-plan override). It performs no zero counting. A caller needing
only defaults can directly use `Dct8Orders::default()` without an AC image.
`OrderPolicy::Adaptive` sets the candidate bit explicitly, bypassing only the
small-image policy. It retains source sampling and sorting, not a different
approximation or forced-full-sample heuristic.

For DCT8, Squirrel and faster use the frozen Xorshift128+ sequence and
`(bits >> 32) <= 0x7fffffff`; slower tiers accept every draw. Arithmetic wraps
as unsigned u64, and the two fixed source seeds are retained. Sampling state
continues across groups, while the packed coefficient offset restarts at each
group. IMPORTANT: the frozen `ac_offset` increments only after an accepted
sample. Rejected draws therefore do not skip a stored block. The implementation
preserves the resulting accepted-count prefix of each group, not an invented
spatial subset. This is recorded as source behavior, not presented as a sampling
algorithm improvement.

All three channels share a sample decision. Each accepted block counts zeros
at every coefficient, then sets DC count to -1. Sort keys preserve the source:

```text
quantized = u64(f32(zero_count) * 0.125f32 + 0.1f32)
key = (quantized << 16) | natural_zigzag_rank
```

DC's small negative intermediate truncates to zero. Zigzag rank explicitly
breaks ties, so an unstable sorting primitive still yields the source stable
tie order. Production arithmetic does not replace this with integer division,
f64 arithmetic or an explicit fused multiply-add. This matters for counts above
f32's exact integer range. A fresh computed plan clears the used bit only when
ALL three channel orders remain natural.

## Permutation Contract

`lehmer_code` and `tokenize_permutation` accept sizes 1..=64 for this DCT8-only
slice, including non-powers of two. Out-of-range/duplicate values, empty or
oversized permutations, skip past the end, and nonidentity skipped prefixes
are rejected. Rejecting a nonidentity prefix prevents silently omitting a
nonzero Lehmer digit that the decoder would reconstruct as zero.

`tokenize_permutation` consumes permutation RANKS in 0..size. For coefficient
positions, use `Dct8Orders::tokens()`: it first maps each position through the
default inverse zigzag LUT and skips DC rank zero. For each permutation:

1. Trim zero Lehmer suffix digits down to `skip`.
2. Emit `Token(CoeffOrderContext(size), end-skip)`.
3. For each retained digit, emit `Token(CoeffOrderContext(previous_digit), digit)`;
   the initial previous digit is zero.

Context calculation uses HybridUint000 regardless of the later entropy model's
value configuration. All tokens are literals, not LZ77 length tokens. If any
channel is custom, tokens for all three channels are concatenated, including
zero-length permutations for default channels. With used-orders zero, emit no
tokens and no coefficient-order entropy header. With a retained used bit of
one and all default orders, emit three zero-length permutation tokens.

## Parent Hook

Parent registers `pub mod coefficient_order;` in `jpeg/mod.rs`. No registration
was added here. Main API:

```rust
let groups = [Dct8Group::I32([&channel0, &channel1, &channel2])];
let image = Dct8Image::new(width_blocks, height_blocks, &groups)?;
let orders = Dct8Orders::compute(&image, speed_tier, OrderPolicy::Source);
let used_orders = orders.used_orders(); // Only bit zero can be set.
let positions = orders.positions();    // [JXL channel][scan rank] -> coefficient position.
let tokens = orders.tokens();          // One stream with eight raw contexts.
```

`Dct8Orders::from_positions(positions, transmit)` validates externally retained
orders, including unique positions, DC first, and no unsignaled custom order.
Explicit `transmit=true` supports an already-committed source bit even if the
positions happen to be default. The parent must retain per-pass orders when
`prev_used_acs` or an already-committed bucket prevents source recomputation;
this single-call sidecar does not own global/pass lifecycle or mutate used-ACS
flags. For the current source JPEG route, streaming frame encoding is disabled;
the retained-plan API does not enable a new JPEG streaming route.

## Remaining Gaps

- No non-DCT8 AC strategies, other order classes, arbitrary transform dimensions,
  or general format-maximum permutation sizes are ported here. The public
  permutation bound is intentionally 64, not 65536.
- The caller still constructs the actual pass AC image, applies JPEG CfL and
  progressive splitting, signals used-order bits, freezes pass state, and
  supplies the orders to coefficient tokenization.
- `EncodeCoeffOrders`/`EncodePermutation` entropy writing is not implemented
  here. Their source default HistogramParams include RLE selection; raw tokens
  must go through the caller's normal entropy/LZ77 preparation, not assume the
  test's no-LZ77 encoding is the complete source wrapper.
- Final-plan record/replay, profiling/layer accounting, coefficient entropy
  coding, JXL frame output and reconstruction-box integration remain separate.
  No lossy DCT computation or pixel decode was added.
- Exact C++ order/token oracle and end-to-end JPEG-to-JXL decoder evidence are
  still future validation. The existing JPEG coefficient/reconstruction oracles
  do not establish these new contracts automatically.

## Focused Verification

Eleven inline tests are provided for parent-controlled execution:

- Independent diagonal-math default order, inverse LUT and transposed addresses.
- All source speed tiers and small-image policy boundaries.
- Quantized zero-count ties, DC placement and a large-count f32/f64 discriminator.
- Real i16/i32 adaptive counts, per-channel orders, default choices, token
  readback and compatibility with the existing entropy Model token API.
- Seeded sample goldens, wrapping/inclusive threshold and the packed-prefix behavior.
- Continuous PRNG state across 33x33-block geometry and partial edge groups.
- Geometry, group count, integer type, coefficient extent and alignment rejection.
- Fenwick Lehmer output versus independent O(n^2) rank removal, with shuffled
  permutations at every size 1..64.
- Literal token/context goldens, previous-digit chaining and trailing-zero trimming.
- All-three-channel signaling, no-order omission and retained default used bits.
- Invalid permutation/skip/DC/used-flag rejection before token output mutation.

No Cargo build or test command was run by this worker. Scoped rustfmt completed
for the new module only. Central compilation, clippy and execution remain the parent's
responsibility. No commits were made.
