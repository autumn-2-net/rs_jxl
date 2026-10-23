# Lossless Palette Port Status

## Scope And Provenance

Direct pure-Rust translation of the immutable supplied libjxl snapshot
`b866b3defec13494bfbaf705db93581cda118e37`, under the retained JPEG XL
BSD-3-Clause attribution in `docs/PORTING_KERNELS.md`:

- `lib/jxl/modular/transform/enc_palette_order.h`: principal-axis permutation,
  maximum-weight degree-two adjacency forest, Hilbert transpose keys, and
  full-width interleaved comparison.
- `lib/jxl/modular/transform/enc_palette_map.h`: exact signed tuple keys,
  source hash constants, linear probing, insertion-order entries, rehashing.
- `lib/jxl/modular/transform/enc_palette.cc::FwdPaletteIteration`: optional
  frequency_order 1..9, geometry 1..2, mapping selection, implicit/explicit
  index assignment, adjacency sampling and the ordering dispatch precedence;
  default single-channel dense lookup and sorted large-range fallback.
- `lib/jxl/modular/transform/palette.h::kMaxPaletteLookupTableSize`: the
  inclusive 65536-value span cap for the default single-channel lookup table.

Owned files: `modular/transforms.rs`, `modular/palette.rs`, new
`modular/palette_order.rs`, new `tests/palette_tools_reference.rs`, and this document.
The new module is a path child
of `transforms`; `modular/mod.rs` is unchanged. The frozen C++ source is unchanged.

## Defaults Versus Experiments

Normal/default behavior is retained: sorted single-channel palettes,
first-appearance candidate collection, both implicit absolute color cubes,
promotion only at frequency >10, and the default f32 luma key with its >4
frequency sign switch and alpha multiplier. Equal luma keys retain the existing
Rust image-order tie behavior; C++ `std::sort` does not specify those ties.
`ordered=false` retains image order and ignores both ordering selectors.

The source single-channel dense lookup-table route is now active by default for
inclusive value spans <=65536; larger spans retain the sorted tree path. This
does not enable an optional selector or claim that every default-enabled source
optimization is implemented. No performance parity or benchmark is claimed.

All tools below are **optional source experiments**, not newly enabled defaults.
They change palette representation only; no sample quantization is performed.
The Rust defaults are `PaletteOrder::Default`, `PaletteGeometry::None`, and
`PaletteMapping::Tree`. There are no process environment reads.

| Source selector | Rust variant | Ordering |
| --- | --- | --- |
| order 1 | FrequencyLuma | Descending exact frequency, then original luma key |
| order 2 | FrequencyBandsLuma | Descending floor(log2(max(1,f)))/2, then luma |
| order 3 | FrequencyBandsMorton | Same bands, then signed full-width Morton |
| order 4 | FrequentMorton | Frequency >4 first, then signed Morton |
| order 5 | Morton | Signed full-width Morton |
| order 6 | FrequencyBandsHilbert | Descending bands, then full 32-bit Hilbert |
| order 7 | Hilbert | Full 32-bit Hilbert |
| order 8 | Adjacency | Maximum-weight degree-two acyclic color chains |
| order 9 | FrequencyBandsAdjacency | Same forest, with cross-band edges omitted |
| geometry 1 | Principal | Unweighted covariance, 12 power iterations |
| geometry 2 | FrequencyBandsPrincipal | Descending bands before principal projection |
| map 1 | Flat | Exact four-word key map, only for <=4 channels |

Geometry takes precedence over order for 3..4 channels. Hilbert/adjacency also
require 3..4 channels. For >4 channels, geometry is ignored, the flat map falls
back to the tree, and order modes 6..9 fall through to unbanded Morton, exactly
as the source dispatch does. One-channel palettes bypass these tools; two-channel
palettes may use the flat map but do not reorder.

Principal ties use projection, complete tuple, then original index; equal
diagonal variances select the first channel. Adjacency edge ties use ascending
endpoint IDs, and equal-frequency chain endpoints retain ascending IDs. Sampling
uses right/down image neighbors at step `max(1, ceil(width*height/262144))`, with
no row wrapping. Morton/interleaved comparisons use the first channel at the
highest differing bit. Stable sorting retains existing order for comparator
equivalences left unspecified by C++. Low-depth implicit aliases and promoted
duplicates retain last-assignment-wins indices.

The flat map compares every channel, including invisible RGB and signed
transformed values. Its repeated-color cache stores entry indices instead of
C++ pointers, so vector growth cannot invalidate the cache. It is used for
counting, implicit classification, adjacency indices, and final sample mapping,
not just as an unused helper. No unsafe code or native dependencies were added.

## Active Entry Points

Types are available through `modular::transforms` and re-exported from
`modular::palette`:

```rust
PaletteTools { order: PaletteOrder, geometry: PaletteGeometry, mapping: PaletteMapping }
```

- `PaletteOptions::tools` flows through `apply_palettes`, every attempted range,
  and its direct branch.
- `palette::apply_palette_range_with_options(channels, meta, bits, begin, count,
  max_colors, ordered, mode, tools)` includes geometry/layout validation,
  transform fields, channel insertion, exact-delta attempts and exact-zero masks.
- `transforms::palette_forward_with_options(planes, begin, num_channels, width,
  height, max_colors, bit_depth, ordered, tools)` builds the reordered palette
  and remaps indices from original samples. The older `palette_forward` uses
  default tools and retains its signature.
- `PaletteOrder::from_source_mode` accepts 0..9;
  `PaletteGeometry::from_source_mode` accepts 0..2. Other values are errors.

Successful exact-delta palettes keep their own residual-frequency order. Ordinary
fallbacks use the selected tools; exact-zero masks are computed from original
samples before ordinary palette ordering and index shifting. Range errors and
declines leave caller channels/meta unchanged. Existing depth limits remain:
ordinary multichannel palettes decline >24 bits, exact delta/zero requires
3..4 channels and <=16 bits, and single-channel palettes support 1..31 bits.

## Parent Integration And Verification

Parent was sent the final types/signatures and owns `ModularOptions`, local/global
prepare, source-option projection, and `group.rs` wrappers. Parent confirmed
`PaletteOptions::tools` is sufficient for direct encoding; no extra
`ModularOptions` field is needed. Parent delegates legacy group wrappers to the
new range entry and exposes `group::validate_transform`, which this entry calls
for shared wire validation. The palette policy path is wired within this ownership.
Parent also owns preserving explicit base policy tools when scheduler projection
rebuilds a policy, analogous to the source's process-global experiment switches.
Parent reported a checkpoint of 812 unit and 14 acceptance tests passing after
palette-core integration and adapter retention of tools. That checkpoint does
not include the subsequently added official-decoder suite described below.

For explicit source projection, preserve requested low-level selectors only when
the source API can represent them; otherwise report an experiment. Silently
resetting tools or labeling nondefault selectors as default source behavior is
not equivalent projection.

Focused tests were added for mode mapping, collision/rehash/cache behavior,
principal and forest ties, Hilbert continuity/bijection, full-width comparators,
all selector combinations, tree/flat equality, implicit aliases and promotion,
signed/invisible sample expansion, channel-count fallbacks, actual image geometry,
the adjacency sampling boundary, delta composition, meta offsets and rollback.
Existing default palette tests remain in place. Tests are **not executed** by this
worker. Owned Rust files were formatted/parsed with standalone rustfmt; no Cargo
build/test, repository-wide formatting, commit or frozen-source edit was performed.
Parent runs central compilation/tests and decoder-oracle checks.

This is not a whole-JXL hash goal, bitstream compatibility proof, compression
benefit claim, or completion claim for the overall pixel-lossless milestone.

## Dense Lookup Checkpoint

The single-channel branch in `palette_forward_with_options` now mirrors the
source's default dense path: compute min/max, allocate zeroed presence slots,
decline on the first distinct count exceeding the requested cap, emit the
dictionary in ascending sample order, replace presence flags with indices, then
remap original samples. The <=65536 cutoff is the inclusive span, not the number
of distinct samples. Larger spans retain the existing sorted BTreeMap fallback.

Range and offset arithmetic uses i64 before conversion to allocation/index
sizes, so negative minima and the full 2^32 signed span cannot wrap on 32-bit
targets. Dense counts are bounded by 65536; palette values use checked i32
conversion. Existing nonempty, depth, color-budget and policy gates are unchanged.
Single-channel palettes still bypass optional ordering/geometry/mapping tools.
The active range wrapper retains shared `group::validate_transform` validation.

Five focused tests cover signed holes and exact dictionary/index order; spans
65535/65536/65537 near both signed endpoints; full signed-range tree fallback;
distinct counts and color caps at 65535/65536/65537; duplicate/constant/empty
inputs and depth gates; and PreRct/PostRct policy metadata plus exact expansion.
They are prepared for central execution, not reported as passed. This checkpoint
does not alter the multichannel 25..31-bit guard or parent-owned integration.

## Official Decoder Fixture Recipe

The policy test `all_palette_tools_policy_matrix_matches_direct_ranges_and_expands_samples`
uses this exact fixture, suitable for the parent's existing official `djxl`
PNG/NPY helper. It uses no out-of-range signed values at the external image API.

- Width 32, height 16; RGB and RGBA variants; depths 8 and 16.
- Base RGBA tuples:
  `[0,0,0,0]`, `[255,255,255,255]`, `[1,2,3,0]`, `[7,11,13,0]`,
  `[7,11,13,255]`, `[90,10,120,127]`, `[32,32,32,0]`, `[254,1,200,255]`.
- For row-major pixel `pos`, first eight pixels use tuple `pos`. Thereafter
  `slot = (pos + (pos / 32) * 5) % 64`; slots 0..31 use tuple 0,
  32..47 use 2, 48..55 use 3, 56..59 use 4, 60..61 use 5, 62 uses 6,
  and 63 uses 1. RGB drops alpha; RGBA retains all hidden RGB.
- Multiply every component by 257 for depth 16. Interleave the selected
  channels for `Image::new`; expected decoded values are the original samples.
- Use palette limit 16, both channel-compaction percentages zero, and speed
  tier 3. Exercise `direct=false/true` with Absolute, ExactZero(Left), and
  ExactDelta(Left). The unit policy matrix covers PreRct/PostRct too.
- Run order 0..9 with geometry 0 and both Tree/Flat mappings. Also run order 9
  with geometry 1 and 2 for precedence, both mappings. Keep palette activation
  assertions, not only output equality. Disable unrelated RCT/squeeze/patch
  experiments for the initial isolated decoder matrix, then add combinations.
- Compare all decoded samples, including invisible RGB. Use caller-supplied
  official decoder paths; do not reinterpret missing oracles as a passed run.

Additional bounded fixture for parent oracle expansion: a 2x2 RGB image with
gray values `[10,20,30,40]` (adjacency palette `[30,10,20,40]`). The existing
262145x1 adjacency-stride fixture is **sample-level only**: the public encoder
rejects axes above 262144. An external stride-boundary fixture would need both
axes within the limit and total area >262144, with a global palette stage.
Signed transformed extrema are also sample-level tests; do not pass them as
unsigned external input.

## Focused Official Decoder Suite

`crates/rs-jxl/tests/palette_tools_reference.rs` follows the existing
`tests/reference.rs` patterns: unique temporary directories, file-backed process
logs, timeout with kill/reap cleanup, strict decoder version/path checks, PNG
identity decoding, and big-endian U16 sample comparison. `RS_JXL_DJXL` is the
OS path list of caller-supplied official decoders (semicolon-separated on
Windows). Missing or invalid decoder paths fail explicitly invoked oracle tests.
No Rust/C++ encoder subprocess is used; Rust encoding calls the real library.

The suite contains one nonignored fixture-activation test and five ignored oracle
tests, totaling **23 encodes per complete oracle suite**, each checked by every
listed decoder. It is deliberately not the full selector Cartesian product:

- Twelve global palette cases: order 0..9, alternating Tree/Flat, plus the other
  mapping for orders 0 and 9; 257x9 RGBA8 spans three 128-wide groups. Includes
  shared/unshared trees and one direct-policy branch.
- Four geometry cases: geometry 1/2 with both mappings, order 9 for precedence,
  RGB16/RGBA16 respectively. Sample preflight also checks geometry-only equality.
- Two local cases: 256x8 RGBA, globally 16 explicit tuples exceeding limit 8,
  locally eight per group. Global YCoCg (RCT 6) creates signed tuples before
  local adjacency/Morton ordering, at depths 8/16.
- Three delta cases: exact-left residual count demonstrably exceeds limit 3,
  while three absolute tuples fit (U8 Tree and U16 Flat); plus U16 exact-zero
  composition with Hilbert ordering.
- Two source native-adapter cases: explicit base tools retained in the actual
  encoded artifact while candidate limit/mode replace other base policy values.
  Covers Hilbert Flat U16 and banded principal Tree with exact-zero U8. Uses
  strict native acceptance and checks the artifact has no projection gaps.

Unlike the original unit-fixture recipe, U16 oracle fixtures perturb nonendpoint
components after scaling by 257 (`+ value % 7`). Black, white and transparent
endpoints remain exact, but an accidental conversion through eight bits cannot
preserve all samples. The activation test asserts genuine U16 low bits and
nonzero invisible RGB are present.

Public-policy preflights assert transform eligibility, local/global declines,
delta counts and signed RCT samples; the oracle independently compares every
decoded component, including hidden RGB. These preflights do not parse the JXL
transform script and are not represented as a new independent bitstream parser.
The integration tests are not yet compiled or executed by this worker.

Parent central invocation:

```text
cargo test -p rs-jxl --test palette_tools_reference
cargo test -p rs-jxl --test palette_tools_reference -- --ignored --nocapture
```

## Default Route Audit

Audit is against the frozen `enc_modular.cc` global/local palette gates,
`enc_palette.cc` lossless branches, `palette.h`, and `palette.cc`, not whole-JXL
byte equivalence. Parent-owned adapter/group/orchestration files were read only.

### Active And Preserved

- Supplying `modular.palette_policy` selects `encode.rs`'s layout route:
  `orchestration::prepare_layout` tries PreRct palettes, performs eligible global
  RCT, then partitions; `finish_stream` tries PostRct palettes on nonzero/local
  streams before local RCT search. Small channels/meta remain global. Source
  speed, responsive, groupwise, depth and equal-shape gates remain in place.
- `PaletteOptions::default` retains source palette limit 1024, channel percentages
  95/80, and speed tier 3. `try_palettes` keeps all/all-minus-one/single attempts,
  source threshold arithmetic, original cost baseline and strict-increase cost
  rejection at slower tiers. Tools do not change these gates or force acceptance.
- Source projected candidates construct an active palette policy and retain
  explicitly supplied base `tools`. Legacy group range wrappers delegate to
  `palette::apply_palette_range_with_options` with default tools. That range entry
  calls `pub(crate) group::validate_transform` for shared wire validation; there
  is no duplicated palette wire-limit validator.
- Default cubes, promotions and luma ordering remain on the active kernel path.
  Equal-key C++ sort order is unspecified; stable Rust ties are not a correctness
  gap. Thirty-two-bit palettes decline in both source and Rust, not an unported
  float palette optimization. Optional order/map/delta settings remain opt-in.
- Default single-channel compaction uses the source dense LUT at inclusive
  spans <=65536, retaining ascending explicit dictionaries and tree fallback
  above the cap. Direct, legacy-wrapper and policy routes reach this same kernel.

### Remaining Differences And Risks

- **Depth coverage gap:** Rust declines multichannel palettes for 25..31 bits.
  Source enters the branch below 32 bits, but builds forward implicit colors at
  input depth while inverse palette expansion caps depth at 24. No explicit-only
  wide-depth palette specialization is implemented. Do not relax this guard
  without resolving implicit-index consistency and an appropriate sample oracle.
- **Public-default contract:** bare `EncodeOptions::default` is the existing
  low-level Rust preset, not the complete source default profile: palette policy
  and legacy palette limit are None, and RCT is fixed to 6. Source automatic RCT
  selection and default palette heuristics require the source projection/policy
  route. This audit does not silently change public defaults or claim parity for
  bare default encoding.
- **Legacy-route distinction:** without a policy, local `encode::planes` applies
  explicit/selected RCT before an explicit all-channel palette. The separate
  `global_palette` route tries an absolute/exact palette before splitting and
  falls back to the local route if it declines. These are explicit compatibility
  controls, not replacements for source global/local heuristics. Nondefault tools
  are configured through the policy API, which selects the active layout route.
- **Stale duplicate route:** `encode.rs` also retains an `else`-branch call to
  `global::encode_policy`, but that branch is unreachable when `palette_policy`
  is Some because the earlier layout condition already includes that condition.
  Its narrower guards should not be mistaken for active-route limitations;
  consolidating/removing it belongs to the parent.
- **Allocation/performance difference:** palette policy keeps transactional clones
  at stage/attempt boundaries and selected sample planes. C++ speculates in place
  and invokes the inverse on cost rejection. This is a memory/performance gap to
  assess on large images, not evidence of pixel failure. No benchmark is claimed.
- **Verification limits:** the new oracle suite covers representative U8/U16
  RGB/RGBA workflows, not a large-image performance corpus, all channel counts,
  >16-bit output, all cost-rejection decisions, subsampled extras, or every
  responsive/groupwise combination. Existing unit/other oracle suites and parent
  milestone tracking retain those responsibilities.
