# JPEG Auxiliary Stream Policy

## Source And Scope

Frozen libjxl `b866b3defec13494bfbaf705db93581cda118e37`:

- `enc_modular.cc:674`: `Init`, including the JPEG MA eligibility gate.
- `enc_modular.cc:1468`: `ComputeTree` domains and fixed-tree population.
- `enc_modular.cc:1698`: `EncodeGlobalInfo` shared tree/residual entropy.
- `enc_modular.cc:2130,2289,2387`: `AddVarDCTDC`, `AddACMetadata`, `AddQuantTable`.
- `modular/encoding/enc_encoding.cc`: `MakeFixedTree`, `TreeSampleFraction`,
  `PredefinedTree`, `LearnTree` and the local `ModularGenericCompress` fallback.
- `enc_ans.cc:1698`, `enc_ans_params.h`: `ForModular` and default histograms.
- `enc_params.h:98,150`, `enc_frame.cc::GetGroupSizeShift`,
  `dec_modular.h::ModularStreamId`: CfL defaults, lossless predicates and layout.

Only actual JPEG input, `nl_dc=false`, non-streaming VarDCT framing, default
predictor selection, no extra channels and zero previous-channel properties are
represented. No lossy quantization, transforms, environment reads, encoder
subprocesses or full-transcoder implementation are added here. Upstream BSD
attribution is retained in the module and tests.

JPEG has `IsLossless()==false` because `modular_mode=false`, but its integer
auxiliary streams have `ModularPartIsLossless()==true`. Conflating these predicates
would select the wrong initial predictor, decoding-speed restrictions and entropy.

## APIs And Ownership

All four top-level functions return `Result<ModularOptions>`:

```rust
options_for_dc(tier: i8, decode_speed: u8)
options_for_ac_metadata(tier: i8, decode_speed: u8)
options_for_quant(tier: i8, decode_speed: u8)
options_for_global_entropy(tier: i8, decode_speed: u8)
```

The first three also exist as methods on `StreamContext`. Its defaults are
`num_streams=22`, `learning_fraction=0.5`,
`force_cfl_jpeg_recompression=true`, `dc_tree_pixels=None`.

**Audit correction:** the previous Rust context incorrectly defaulted forced CfL
to false. `CompressParams` defaults it to true. Default AC metadata now learns at
tiers -1 through 2; explicitly false retains the existing Zero-leaf route. The
flag alone controls this tree-kind override, even for grayscale, subsampled,
constant or zero CfL maps. CfL computation/eligibility is a separate caller job.

`options_for_global_entropy` is an entropy/resource carrier, not another tree
domain. It has no sampling policy, fixed tree, custom tree or tree-group plan.
Its neutral Gradient predictor does not replace the predictors of domain leaves.
Its entropy/LZ settings equal the DC/AC `ForModular` settings, with no dependence
on DC population or CfL. The existing local APIs remain separate and executable.

For local streams, pass prepared planes and transform descriptors to
`modular::encode::write_prepared_stream`, using nominal `bits=8` and the real
stream ID. Nominal depth is not an integer-sample clamp. No RCT, palette, squeeze,
WP mode search or channel reorder is added. The writer does not byte-pad.

For shared streams, give each `auxiliary::Domain` its corresponding context
method's result, and give `auxiliary::Shared::build` the global entropy result.
Use that same global policy for merged-tree tokens and shared residual tokens.
Do not substitute local Quant options: its default RLE is wrong for every shared
JPEG tier. Parent owns the composer, domain preparation and shared writer.

## Stream Domains

For D DC groups, G AC groups and P passes, all slot counts and IDs are source IDs:

| Kind | Stream ID |
| --- | --- |
| Global | 0 |
| VarDCT DC group g | 1 + g |
| Modular DC group g, normally empty | 1 + D + g |
| AC metadata group g | 1 + 2D + g |
| Quantization table i, 0 <= i < 17 | 1 + 3D + i |
| Modular AC group g, pass p, normally empty | 1 + 3D + 17 + Gp + g |
| Total slots | 1 + 3D + 17 + GP |

Empty slots still count; do not replace total slots with three domains or the
number of nonempty streams. `Init` starts domains at 0, 1, 1+D, 1+2D, 1+3D,
1+3D+17 and the final slot count. `ComputeTree` removes empty domains, retains
their original start positions for merging, and supplies real IDs to learners.
It builds each useful domain independently and merges by group property 1.

JPEG always takes this domain branch, even at Falcon/Thunder/Lightning (7/8/9),
because `!modular_mode` is true. The whole-pixel-frame fixed-tree branch and
Lightning Gradient-leaf shortcut must not replace JPEG domain choices. The
special tier-7/8 tree kinds assigned to slot 0 do not overwrite DC/metadata/Quant.

DC planes remain Modular Y/X/B, corresponding to internal JXL channels 1/0/2;
subsampling changes their actual areas, not their source zero shifts. Metadata
planes remain YToX tiles, YToB tiles, flattened strategy/raw_quant-minus-one rows,
and EPF. The first two have shifts 3/3. Quantization planes keep prepared input
order. Property 0 means the channel within that stream, not a JPEG component ID
or a channel index concatenated across domains.

## Tree Boundaries

Valid tiers are source SpeedTier -1..=9, not effort: TectonicPlate=-1, Glacier=0,
Tortoise=1, Kitten=2, Squirrel=3, Wombat=4, Hare=5, Cheetah=6, Falcon=7,
Thunder=8, Lightning=9. Valid decode speeds are 0..=4; other values error.

| Domain | Predictor option | Tree kind |
| --- | --- | --- |
| DC, tier <= 1 | Variable | Learn at decode 0; Gradient fixed at decode >= 1 |
| DC, tier 2 | Best | Learn at decode 0; Gradient fixed at decode >= 1 |
| DC, tier >= 3 | Weighted | Weighted fixed at decode 0; Gradient fixed at decode >= 1 |
| Metadata | Gradient | Learn iff forced CfL and tier < 3; otherwise one Zero leaf |
| Quant | Gradient | Learn at every tier; no inherited DC fixed tree |

DC overrides the Init tree mode after its decode-speed processing: Default below
tier 3, WeightedOnly at/above 3, even when the final tree kind is Gradient fixed.
Metadata overrides it to NoWeighted at every decode speed. Quant retains Init's
Default (decode 0), NoWeighted (1/2), or GradientOnly (3/4). Decode 4 sets sampling
to zero; it does not itself change a Learn tree kind into a fixed template.
JPEG's final `fast_decode_multiplier` is exactly 1.0, not the pixel-path values.

Fixed DC trees reuse `modular::fixed::build`. Shared source construction uses
bitdepth 8 and prevprop 0: Weighted splits use property 15, Gradient splits use
property 9, never previous-channel property 18. Every leaf has offset 0 and
multiplier 1. Strict `property > cutoff` takes the greater child; breadth-first
greater-first node/context ordering is retained.

Population is the sum of actual channel areas across the entire DC domain, not
spatial image area, selected training samples, or a single group. Sum with checked
arithmetic. `dc_tree_pixels=Some(total)` materializes that custom tree; it is
ignored on a Learn branch. `None` defers the sum to the writer: the shared writer
must supply all streams of the domain, the local writer only its local planes.
Zero population is normalized to one. The resulting leaf counts are:

| Total samples | Leaves |
| --- | --- |
| 0..=512 | 1 |
| 513..=4096 | 2, cutoff 0 |
| 4097..=8192 | 4, cutoffs -31, 0, 47 |
| >=8193 | 34, all 33 source cutoffs |

## Sampling And Limits

Unscaled learning fraction must be finite and nonnegative. Values above one are
allowed. Multiply in f32 once by the following scale, then cap at 1.0; decode 4
keeps zero. TectonicPlate (-1) deliberately uses the default case, not Glacier.

| Tier | Scale | Initial property count | Quantile bins |
| --- | --- | --- | --- |
| -1, 6..=9 | 0.3 | 3 | 32 |
| 0, 1 | 1.3 | all 16 | 256 |
| 2 | 1.1 | 10 | 128 |
| 3 | 1.0 | 7 | 96 |
| 4 | 0.7 | 5 | 64 |
| 5 | 0.5 | 4 | 48 |

Initial property order is `[0,1,15,9,10,11,12,13,14,2,3,4,5,6,7,8]`. Remove
property 1 before truncation only when total slots < 30 and tier > 1. Then apply
the final domain tree-mode filter: WeightedOnly replaces with `[15]`,
GradientOnly with `[9]`, NoWeighted removes 15, Default retains the list.
Previous-channel properties remain disabled. Node threshold is the exact integer
`75 + 14*tier + 10*decode_speed`, stored as f32 (range 61..=241).

The per-channel sampling probability is zero for zero repeats; otherwise it is
`max(repeats, min(1, 1024 / channel_area))`, using the source f64 promotion.
Thus all 64 samples of each 8x8 Quant channel are selected whenever repeats > 0,
not merely half of them. Reset the source sampler for each channel/stream; do not
rescale repeats by domain area or by the number of shared groups. Existing sample
and training modules handle RNG, quantiles and sampled split-cost correction.

DC/metadata `max_channel_size` is 0xFFFFFF; Quant inherits 256. VarDCT group size
stays 256 at all decoding speeds, unlike the pixel-Modular group-size override.

`sample_limit` is now an optional hard error ceiling, not a target sampling count.
Returned options default to `None`; no selected-sample resource cap is imposed.
The maximum configurable MA leaves is the decoder tree range, not the former
16,384 private cap. Propagate an explicit caller sample budget to BOTH the global
carrier (shared budget) and every domain policy (per-learner ceiling).
Source-mode learning errors if a beneficial split remains
at the leaf ceiling. No silent truncation or sampling adjustment is authorized.

## Global Versus Local Entropy

For shared JPEG, `modular_mode=false`, `IsLossless=false`, extra DC precision is
zero, streaming is false. The common tree and residual entropy policy is:

| Tier | LZ method | Clustering | ANS histogram | Hybrid uint |
| --- | --- | --- | --- | --- |
| -1..=1 | Optc256 | Best | Precise | Best |
| 2 | b3w3t | Best | Precise | Best |
| 3..=8 | None | Fast | Approximate | fixed (4,2,0) |
| 9 | None | Fast | Fast | fixed (4,2,0) |

Decode speeds 0/1 allow up to 256 histograms in Rust; 2..=4 cap at 12. Source's
unlimited default is bounded by the existing encoder's 256-histogram wire limit.
Length uint is (0,0,0), min-symbol 224; model construction retains existing
source prefix/ANS selection. No fixed/missing-symbol injection, precise-entropy
pixel override, lossless bucket matcher, parallel LZ or disposable probe override
is enabled. Optimal parsing keeps default `careful_first_pass=false`: the source
65536-token first-pass threshold applies to the combined shared input, not each
group. The merged MA tree is a separate entropy stream with no image width.

Local DC/metadata inherit the same ForModular policy. Local Quant instead retains
`HistogramParams()` from Init: Best/Precise/Best, RLE for tree and data, 256
histograms regardless of tier/decode speed. Its length uint is also (0,0,0).
`AddQuantTable` does not copy slot 0's histogram policy. The standalone
`EncodeQuantTable` path with no frame encoder also starts from unscaled Modular
defaults; that distinct fallback is not what `options_for_quant` represents.
Parent has wired the private tree-RLE helper; this change does not touch it.

## Typed MA Speedups Proposal

Source Init enables `lossless_tree_speedups` for lossless pixel input, or for
actual JPEG input when `JXL_LOSSLESS_JPEG_MA_SPEEDUPS` is absent or exactly `"1"`.
Any other supplied string disables the JPEG branch. This is default-enabled
exact cost/count reuse, not an optional incremental-entropy approximation. A
generic VarDCT frame alone is insufficient evidence of JPEG input.

Proposed parent-owned explicit mapping, not a new dead resolver in this module:

```rust
enum JpegMaSpeedups {
    SourceDefault, // enabled for an actual JPEG input
    Disabled,
}
```

Resolve this only with a typed JPEG input/domain, not `modular_mode=false` alone.
For actual JPEG, SourceDefault resolves exact cost/count reuse to true, Disabled
to false. `lossless_fast_entropy` and `lossless_channel_trees` remain false for
JPEG: their source Init gates require lossless pixel input. Constant-channel
bypass, predictor deduplication, sample-batch/sparse gathering and chunk/property
parallelism are separate explicit experiments, not enabled by this enum alone.

Current Rust training already uses its default f32 scorer/SIMD, parent histogram
reuse and bounded histogram cache. `ModularOptions` does not expose a complete
source JPEG-speedups-off switch. In particular, toggling only
`TrainOptions::reuse_parent_histograms` does not disable every source-flag-gated
path. A future Disabled mapping must audit/wire all relevant core gates, or
report unsupported/source-divergent behavior; it must not pretend a partial
toggle is source-equivalent. No environment reading or parent-owned API changes
are implemented by this worker.

## Verification Boundary

The focused tests cover all 11 tiers x 5 decode speeds, corrected default/explicit
CfL behavior, source sampling boundaries, fixed-tree pruning/property IDs,
local policy preservation and representative heterogeneous shared writes through
the parent's `auxiliary::Shared` API. Shared checks include low-tier source LZ,
empty IDs and unaligned insertion; they are not independent JPEG decode oracles.

This worker does not run Cargo, build native code, register modules, edit shared
encoder/composer files, or commit. Parent owns central compilation/tests and
official decoder/JPEG byte-reconstruction verification. No passing result for
this audit's new tests is claimed until parent reports it. Full JPEG transcoder
completion and identical intermediate JXL bytes are not claimed by this module.
