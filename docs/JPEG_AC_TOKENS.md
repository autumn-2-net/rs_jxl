# JPEG DCT8 AC Tokenization Prerequisite

This is a pure-Rust coefficient-tokenization and context-model prerequisite,
not a complete JPEG-to-JXL encoder or byte-reversible JPEG migration claim.
No pixels are decoded, no DCT is computed, and no coefficients are quantized or
transposed here. Full JPEG reconstruction and independent byte comparisons
remain parent integration work.

Frozen read-only source: `../jxl_python/libjxl`, commit
`b866b3defec13494bfbaf705db93581cda118e37` (HEAD checked during this slice).
Translated code retains the upstream BSD notice; see `PORTING_KERNELS.md`.

## Ownership and Registration

Only these new files are owned by this slice:

- `crates/rs-jxl/src/jpeg/ac_tokens.rs`
- `crates/rs-jxl/src/jpeg/ac_tokens_tests.rs`
- `docs/JPEG_AC_TOKENS.md`

No module registration was edited. Parent can register `pub mod ac_tokens;`
in `jpeg/mod.rs`. `ac_tokens.rs` registers its own test file with a private
`#[cfg(test)]` path module. There is no dependency on the concurrently developed
`coefficient_order.rs`, no Cargo change, and no entropy/LZ/scheduler edit.
The parent-owned `frame_header` geometry and registered `coefficient_order`
module are not dependencies of this slice; callers pass prepared group views
and orders. The temporary generic test filename was renamed before handoff.

## Interface

`tokenize_dct8(&Dct8Input, BlockContextModel, TokenizeLimits)` returns
`Result<Vec<entropy::stream::Token>, AcTokenError>`. This existing token has a
`usize` context and a `u32` value, sufficient for all source contexts and signed
coefficient bit patterns. Every emitted token has `is_length=false`. These are
unclustered entropy tokens, not ANS bytes or a JXL codestream.

- Three borrowed planes are in JXL channel order, not JPEG component order.
  Each plane is tightly packed raster block order, 64 transposed `i32` values
  per block. Coefficient index zero is DC and is never emitted.
- Three independent `[u32; 64]` orders map scan rank to transposed storage
  index. Every array must be a permutation of 0..63 and start with zero.
  A caller with natural-order JPEG coefficients must prepare transposed
  storage and matching orders before this call.
- The rectangle is group-local. HShift/VShift are explicit per-channel source
  shifts, each zero or one, not JPEG sampling factors. Each axis needs at least
  one zero shift, as in source subsampling. Plane dimensions must exactly equal
  the ceil-divided rectangle dimensions, including odd right/bottom edges.
- qdc and qf are packed **full-resolution** rectangle grids. qdc is a
  precomputed DC-context index, not a DC coefficient; qf is the raw integer
  quantization field in 1..=256 (`quantizer.h::kQuantMax`). All supplied grid
  entries are validated. The caller handles extraction from larger images.
- Prediction resets at this rectangle's top and left boundaries, as in the
  source per-group tokenization call. There is no cross-group predictor state.
- A canonical 0x0 rectangle with empty planes/grids is accepted. Mixed-zero
  dimensions, overflow, trailing/short coefficient buffers, invalid grids,
  malformed orders and resource-limit violations return errors.

`BlockContextModel::default()` supplies the exact source 39-entry default map,
15 block contexts, one DC context and no thresholds. `new` borrows all 13 order
classes of a caller-supplied map and validates source wire bounds: at most 15
thresholds per list, at most 64 combined DC/QF classes, at most 16 block context
IDs, exact map length and QF thresholds in 1..=300. Duplicate or unsorted
thresholds remain valid: source counts strict comparisons, not binary searches.
Every declared context ID must appear in the map, matching
`dec_context_map.cc::VerifyContextMap`; missing or excess IDs are rejected.

`context` preserves the source channel swap and layout:

```text
qf_index = count(qf > threshold)
channel_index = channel < 2 ? channel ^ 1 : 2
index = ((channel_index * 13 + order) * (qf_thresholds.len + 1)
         + qf_index) * num_dc_contexts + qdc
block_context = context_map[index]
```

Only DCT8 order class zero is used by the tokenizer. The optional `dc_context`
helper takes already prepared quantized DC values in JXL channel order and
implements `compressed_dc.cc::DequantDC`'s strict threshold bucket formula
`(bucket_X * (thresholds_B.len + 1) + bucket_B) *
(thresholds_Y.len + 1) + bucket_Y`. It performs no sampling, dequantization,
DC residual construction or CfL preparation. Those remain caller work.

## Source Coverage

| Frozen source | Rust implementation | Scope/status |
| --- | --- | --- |
| `enc_entropy_coder.cc::TokenizeCoefficients` | `tokenize_dct8` | DCT8-only: block traversal, channel order **1,0,2**, source H/V sampling gates, component offsets, nonzero token followed by scanned AC tokens. |
| `NumNonZero8x8ExceptDC` | `count_nonzeros` | Exact zero comparison and DC masking, scalar plus runtime-dispatched native Rust SSE2/AVX2. Unaligned bounded loads; no native codec or Highway linkage. |
| `entropy_coder.h::PredictFromTopAndLeft` | Rolling prediction rows | Default 32 at top-left; left-only/top-only edges; rounded-up mean in the interior. Stored nonzero counts are in 0..63. |
| `ac_context.h` coefficient tables / `ZeroDensityContext` | Exact two 64-entry tables and `zero_density_context` | DCT8 specialization: covered_blocks=1, log2=0. Uses remaining nonzeros, scan rank, and previous-nonzero flag exactly. |
| `ac_context.h::BlockCtxMap` | `BlockContextModel` | Exact default map, channel/order/QF/DC indexing, clamped nonzero buckets, AC offsets and context totals. |
| `enc_context_map.cc::EncodeBlockCtxMap`, `entropy_coder.cc::DecodeBlockCtxMap`, `dec_context_map.cc::VerifyContextMap` | `write_block_context_map`, model validation and `is_default` | Default shortcut or all three signed DC threshold lists, QF thresholds minus one, then existing generic context-map serialization. Validates threshold/map wire constraints and complete context-ID coverage. |
| `compressed_dc.cc::DequantDC` qdc bucket section | `dc_context` | Scalar bucket nomination only, from caller-prepared quantized DC values. |
| `pack_signed.h::PackSigned` | Signed coefficient emission | Full i32 domain, including i32::MIN -> u32::MAX; no signed overflow or magnitude restriction. |
| General strategies / `NumNonZeroExceptLLF` | Not implemented | Multiple-block transforms, non-DCT8 orders and strategy first-block gating are outside this JPEG DCT8 prerequisite. |
| Coefficient order search/serialization | Not implemented here | Borrowed validated arrays only; no dependency on the concurrent order helper. |
| AC histogram clustering / coefficient entropy emission / AC groups / JPEG reconstruction | Not wired here | Existing Token representation is reused, but no encoder route, frame/group assembly or wire acceptance is claimed. |

### Block Context Wire Header

`write_block_context_map(model, &mut BitWriter) -> crate::Result<()>` is the
global-DC sibling of this tokenizer. It writes no padding and can start at any
bit offset. The exact default predicate emits only a one bit. Other models
emit, in source order:

1. A zero bit (not default).
2. For channels 0, 1, 2, the four-bit DC threshold count and each signed
   threshold packed with PackSigned, then source U32 selectors
   `(4,0), (8,16), (16,272), (32,65808)` as `(bits, offset)`.
3. The four-bit QF threshold count and every threshold minus one, with source
   U32 selectors `(2,0), (3,4), (5,12), (8,44)`.
4. `entropy::context_map::write_context_map` of the full validated map and its
   block-context count, not a DCT8-only projection.

All thresholds, including unsorted/duplicate values and signed i32 extrema,
are preserved. The parent prepares full-frame qdc indices separately and
supplies group-local qdc/qf grids to tokenization. Model defaults or empty
threshold lists are not silently substituted for supplied values.

Nondefault headers are built in a temporary bounded writer and appended
atomically; destination limit failures leave existing bits intact. The
temporary byte bound is `8*map_entries + 8192 + ceil((17 + 34*thresholds)/8)`,
using the existing generic coded-map bound plus the threshold prefix. Validated
map length is at most 2496. Nested generic entropy allocation/error policies
are inherited unchanged. The caller's BitWriter controls final output size.

The existing generic writer compares exact serialized simple/raw/MTF candidate
costs rather than frozen C++'s dry estimates. This wrapper preserves the source
entropy structure but does not claim identical representation selection or
compressed bytes. Optional direct/final-plan histogram overrides, profiling
and audit instrumentation are not exposed by this wrapper. No generic entropy
implementation was edited. Global-DC/group orchestration remains unwired here.

Important exact traversal details:

- Each visited component block samples `qdc[by * width + bx]`, but
  **qf[by * width + (bx >> HShift(c))]**. It does not sample qf at full bx or
  at component-row sy. This preserves the frozen encoder's `row_qf[sbx[c]]`.
- The nonzero context prediction bucket is `min(prediction,64)` below eight,
  otherwise `4 + min(prediction,64)/2`, multiplied by the block-context count.
- Initial previous-nonzero is one for at most four AC nonzeros, zero otherwise.
  Every scanned zero before the last nonzero is emitted. Trailing zeros and DC
  are not emitted. Signed coefficient values use the exact source mapping.
- AC context is `num_contexts*37 + block_context*458 +
  (nonzero_table[remaining] + frequency_table[rank])*2 + previous_nonzero`.
  Valid DCT8 encoder states stay below 458 density contexts. The checked public
  table helper also accepts independent 1..63 indices, matching the source's
  bounded decoder guard, whose broader limit is 474. It does not emit invalid
  encoder states. The default model has 7425 AC contexts; the maximum is 7920,
  below the existing entropy stream's 16385 raw-context bound.

## Bounds and SIMD

Callers must supply `TokenizeLimits`; there is no hidden unbounded default.
Before allocation, the implementation checks all dimensions/mappings/orders,
sum/product overflow, vector byte-size bounds and both caller budgets.

Output reserves at most `64 * sum(component_block_count)` tokens, the exact
DCT8 worst case of one count token plus 63 coefficient tokens per block.
`max_tokens` must accommodate that conservative bound even for sparse input.
Prediction scratch is `sum(component_width_blocks)` bytes in three rolling
rows. No input coefficients are copied, and no full-size nonzero map is needed.
All vector reservations are fallible and occur before emission; output is
returned only on success. SIMD uses fixed stack arrays and allocates nothing.

SSE2/AVX2 functions mask the first lane (DC), compare all 64 i32 lanes to zero,
sum the negative comparison masks, and add 64. Feature checks precede every
unsafe target-feature call. Loads are unaligned and exactly within the
64-element reference. Other architectures use the scalar implementation;
native ARM SIMD and wider source SIMD variants remain future optimizations,
not required for scalar functionality. No source C++ is compiled or linked.

## Verification Handoff

`ac_tokens_tests.rs` contains 17 focused tests, including:

- Exhaustive density-table comparison against an independently expressed
  piecewise oracle, valid-state and independent-index maxima, and invalid
  table coordinates.
- Default map values for every channel/order; nonzero prediction clamp through
  1008 and u32::MAX; custom DC/QF indexing with strict ties, unsorted thresholds,
  qdc channel mixing, and malformed/maximum-bound models.
- Scalar/native nonzero cross-checks at every lane with signed extremes,
  unaligned input and varying dense prefixes; SIMD paths only run if supported.
- DC-only blocks, exact channel order, signed extrema, transposed/permuted
  orders, interior zeros, trailing-zero omission and top/left predictor edges.
- Explicit subsampled qdc/qf address checks plus a scalar token oracle across
  444/420/422/440 and asymmetric source shifts, odd dimensions, custom maps,
  varied quantizers and deterministic signed coefficient data.
- Independent command-free token replay reconstructing every AC coefficient,
  invalid permutations, grid/dimension/shift validation, resource caps and
  canonical empty input.
- Exact default header bits at every alignment; independently decoded DC/QF
  threshold fields covering every U32 selector boundary and signed extrema;
  full one-context header decoding, multi-context generic-writer delegation,
  no added padding and atomic destination-limit failure.

The two coefficient tables (64 entries each) and default context map (39 entries)
were compared directly against the frozen source and match exactly. No Cargo
build, test or Clippy run was performed by this worker. Parent owns registration
and central compilation/testing. No independent JXL/JPEG decoder acceptance is
claimed for this prerequisite, and no compressed JXL hashes are compared.
Scoped `rustfmt --edition 2024 --config skip_children=true --check` passed on
`ac_tokens.rs` and `ac_tokens_tests.rs`; it does not establish type correctness
or test success. No repository-wide formatting was performed.
