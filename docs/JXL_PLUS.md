# Experimental jxl_plus backend and extension design

This experiment is archived on branch `jxl_plus` at the user's request.
Standard-JXL decoding-speed work resumes on `main` from pre-Plus commit
`d003ae0`. The implementations, results and unfrozen format proposal below are
retained for reference; this branch is not a released format.

`rs-jxl-plus` implements a separate native Rust format named **jxl_plus**, with
extension **`.jxl_plus`**. The current experimental **hashless version 1**
recompresses supported JPEGs through their
original quantized DCT coefficients and restores the original JPEG bytes.
It is neither standard JPEG XL nor a PJA archive. Standard JXL APIs and their
compatibility requirements remain unchanged.

The user explicitly authorized removing the original-file hash and trailing
container hash. Version 1 makes that change and rejects old version-0 inputs;
the shortened layout is not silently read as version 0. It is still the
standalone experimental container, not the planned JXL-derived extended frame
layout. Defining and implementing that final layout remains separate work.

The dependency is one-way: `rs-jxl-plus -> rs-jxl`. The new crate reuses the
existing JPEG parser, coefficient reader and reconstruction metadata/writer;
it does not require moving those modules into a common crate. PJA's stream-v7
coefficient model and arithmetic coder are translated into Rust. No C/C++
codec, native codec linkage or subprocess encoding is used by the runtime.
Source attribution is in [PROVENANCE.md](../crates/rs-jxl-plus/PROVENANCE.md).

## Current interface and scope

Rust entry points are `encode_jpeg(&[u8], &EncodeOptions)` and
`decode_jpeg(&[u8], &DecodeLimits)`, both returning owned bytes. Encoding
decodes its completed output and compares every original JPEG byte before
returning. These are in-memory operations; verification is part of the
current encode call's cost. The focused size comparison below is not a general
compression or speed guarantee.

The dedicated experimental CLI uses:

```powershell
cargo run -p rs-jxl-cli --bin jxl-plus -- encode input.jpg output.jxl_plus
cargo run -p rs-jxl-cli --bin jxl-plus -- decode output.jxl_plus restored.jpg
```

Version 1 accepts the supported `rs-jxl` **8-bit Huffman DCT** JPEG path with
one or three components: baseline, extended sequential and progressive,
subject to parser/reconstruction restrictions. Component sampling factors are
1..4, with integral ratios to the maximum horizontal and vertical factors.
Zero quantizers and coefficient -32768 are rejected. Restart, scan, padding,
metadata and trailing-byte preservation use the existing reconstruction
machinery; success means full original-byte equality, not equal pixels.
Unsupported inputs return errors, without pixel encoding or opaque-original
fallback.

The current public API exposes only this JPEG-container experiment. The expanded
internal port includes arithmetic and lossless JPEG, JPEG-LS, JPEG 2000
(including HT paths), and the original algorithm/model modes. Their
translation, integration and verification are tracked in
[PJA_PORT_STATUS.md](PJA_PORT_STATUS.md). Internal composition now uses typed
model/reconstruction records. Old archive versions, hashes, Pjax, GivenTables,
recursive parts/repair/PATCH, whole-file raw fallback and their protocol-only
tests have been removed. Valid source bytes remain exact; five pinned damaged
JPEGs that depended on repair or raw fallback are now explicitly rejected.

Full/Fast/V7, MED/Blend/symbol and dynamic/static coding tools are retained for
selection and composition with JXL tools. The static experiment does not
authorize deleting static coding. Local raw codeblock exceptions and metadata
byte coding remain necessary tools. The J2K chooser retains all four model
paths and compares their actual output sizes even when larger than the input.
MPO and embedded-JPEG bytes remain preserved; their old recursive independent
recompression wrapper is not retained in the typed internal representation.

The latest instruction removes `.pja` compatibility as a goal. Existing
source comparisons are development checks, not a format contract. The
eventual `jxl_plus` format will use the original JXL codestream plus extra
extensions, after this port. This hashless version 1 remains an experiment; it
does not yet expose the newly ported codec paths or implement that layout.
Pixel-lossless images, multiframe, Python/C ABI integration and standard-JXL
decoding are not provided by this backend's version-1 API. The existing
standard backend retains its own interfaces.

## Target extension format: pixels and reversible source reconstruction

The user explicitly confirms two primary targets: exact lossless image samples
and byte-reversible JPEG recompression. Adaptive probabilities are a supported
format tool, alongside static JXL entropy coding. The current JPEG-only v1
public API is an implementation checkpoint, not a restriction on the target.
This section records the design direction; it does not assign final wire tags
or claim the extended frame/group syntax is already implemented.
The detailed working identification, extension dispatch, model rules and
freeze gates are now in [JXL_PLUS_BITSTREAM_DRAFT.md](JXL_PLUS_BITSTREAM_DRAFT.md).
Its provisional major-2 assignments are distinct from the current JPEG-v1 and
pixel-research formats; the actual JXL frame integration is still required.

Keep the JXL image/frame/group structure and its standard coding branches as
the core. Extend coding units with explicitly identified tool combinations.
An extension-dependent unit requires the Plus decoder; an ordinary JXL
decoder is not promised to render a partially extended frame. Avoid storing a
complete duplicate standard image merely to carry a second compressed image
beside it. Preserve one-way `rs-jxl-plus -> rs-jxl` implementation reuse.

| Layer | Information the decoder needs |
| --- | --- |
| Reconstruction target | Pixel dimensions, channel/sample representation, bit depth and frame information, or original JPEG reconstruction metadata plus its coefficient/sample geometry |
| Reversible preparation | Ordered transform IDs and parameters: eligible pixel RCT/palette/squeeze, coefficient CfL, MA predictor/tree, or other declared tools; each tool declares its input/output domain |
| Entropy model | Versioned JXL ANS/prefix profile or adaptive coefficient/sample/bit-context profile; selected static tables or adaptive initialization rules |
| Coding unit dependencies | Channel/group order, reset boundaries, any model inheritance from an explicitly identified earlier unit, byte lengths and bounded decoded sizes |

An MA tree and adaptive probabilities are compatible. The tree can choose
a predictor and/or a context; the selected context's probability state can
then update after each decoded symbol. A fitted tree is transmitted once,
while both sides reproduce subsequent adaptive updates. This is distinct from
the test below that first transforms coefficients with MA and then gives the
residuals to unchanged v7. Direct MA-leaf-conditioned entropy remains another
candidate, not a result established by that composition experiment.

Decoder-visible adaptive configuration must be self-contained: a fixed,
versioned profile can specify context formulas, bucket precision, active
contexts, mixer selectors, initialization/priors, learning rates and exact
integer rounding/overflow rules. Optional transmitted parameters must have
bounded, defined syntax. Encoder search effort and worker count need not be
transmitted when they do not affect decoding. Runtime environment variables
must never supply missing decoding semantics. Numerical lookup tables must
be deterministic rather than regenerated with platform-dependent floating
point during decoding.

Independent groups start with the profile's initial state. A channel pipeline
may inherit an earlier channel's final state only through a declared causal
dependency; both decoder scheduling and resource accounting must honor it.
Choosing groups or cloning models must account for the entire live model,
trees/tables, descriptors and stream termination, not just compressed payloads
or the main counter array. Keep resource and structural validation without
reintroducing integrity hashes or generic broken-file repair.

Pixel acceptance compares all original decoded samples, including invisible
RGB, bit depth and frames. JPEG acceptance still compares every original JPEG
byte. An RGB pixel codec cannot stand in for JPEG byte reconstruction. The
ported JPEG-LS, lossless JPEG and J2K tools remain available within their
explicit domains; their presence does not establish a finished general
pixel-lossless Plus API.

Default ceilings are 512 MiB each for input/output, 256 MiB for coefficient
storage, 64 MiB for reconstruction metadata, 100 million pixels, 1024 scans,
100 million block visits and one million hint entries. They are stage/buffer
limits, not a process-wide peak-memory guarantee. Coefficient copies, models,
metadata and output can coexist. Reconstruction metadata Brotli quality
defaults to 6 and accepts 0..11; it does not change the coefficient model.

## Model and execution

The v7 loop codes interior counts and coefficients, then the fourteen axis
AC coefficients, then a DC residual predicted from reconstructed AC boundary
information. Interior/edge probability contexts include causal spatial and
cross-component information. Adaptive counters and mixer updates are visible
to both encoder and decoder. Cb and Cr share a chroma model; luma uses a
separate model. Each horizontal band resets both models and its boundary
state.

Band count follows the source rule:
`max(1, min(total_padded_blocks / 65536, 16, MCU_rows))`, using integer
division. Band `i` covers MCU rows
`[MCU_rows*i/bands, MCU_rows*(i+1)/bands)`. Streams are independent, but the
public entry points process them serially. Internal parallel entry points
are implemented and tested; no end-to-end speedup measurement is claimed.

Prediction and mixer dot/weight-update kernels use Rust `std::arch` AVX2 on
x86_64 after runtime feature detection, with a portable scalar fallback.
Unsafe intrinsics are isolated and document bounded loads/stores. Arithmetic
coding and counter updates use scalar Rust. Wrapping
32-bit sums, truncation toward zero, integer prediction constants and frozen
probability tables retain source numerical rules. No new ARM kernel is added.

## comr9 pixel-model research input, 2026-11-17

The user supplied `comr9/r9.md` and `comr9/image_compress2.zip` as unfinished
research code. The archive implements 8-bit RGB sample compression, with
report measurements on three illustrations. It has not been imported into
the runtime or benchmarked here. Its reported timings and ratios are not
comparisons against this repository's JXL or JPEG recompression results.

Useful candidate tools, read from the archived sources:

| Prototype source | Candidate for the Plus pixel path |
| --- | --- |
| `src/dmctx/{contexts,neighbours,frame}.rs` | Up to 44 causal spatial/cross-color contexts derived from neighboring samples, gradients and edge structure |
| `src/algos/r4.rs` | Four additional contexts for left runs, G-guided edge/antialias prediction, two-dimensional neighborhood matches and two-left differences |
| `src/cm4.rs` | Adaptive binary context tables, multiple first-layer mixers and a second mixer, content/bit-dependent selectors, weak predictor-centered priors, configurable active model size |
| `src/algos/r6.rs`, `src/algos/r8.rs` | Choose original color samples or reversible modulo-256 R-G/B-G differences; use content statistics and trial coding as encoder search |
| `src/algos/r8.rs`, `src/algos/r9.rs` | Reuse the state after coding G to initialize B/R models; causal B-to-R row dependencies on decode; independent horizontal stripes with model resets |

These are pixel-domain tools. A future MA combination can use the tree to
select predictors or CM context classes. If the CM backend codes MA residuals
instead of the prototype's unsigned 8-bit samples, signed symbol mapping,
prior centers and the causal reconstructed neighborhood must be defined for
that domain. Pixel modulo-256 arithmetic is not suitable for direct use on
JPEG's signed quantized DCT coefficients. Keep the JPEG coefficient and pixel
tool families distinct while sharing validated primitives and scheduling.

Review caveats are implementation work for a future import, not reasons to
discard the algorithms. The AC18/AC19 containers omit model parameters and
rely on matching environment variables. R9's ceiling-sized stripes can emit
fewer entries than its declared count (`h=17, k=10` emits nine nonempty
stripes); R8's minimum-8-pixel trial crop can underflow on smaller images.
The light model's 1,572,864-byte counter table is only part of its storage:
the still-full-sized first-layer weights occupy 11,239,424 bytes. Declared
arrays total about 12.95 MB per light model or 112.04 MB per full model,
before allocator overhead and pixel buffers; R8 maintains three model copies
and R9 multiplies them by active stripes. These are source-derived allocation
counts, not measured peak RSS. The arithmetic reader also pads exhausted
input and needs a defined bounded termination contract for a real format.

The archive declares `license = "unlicense"` and comments identify deepmix
derivations as public domain, but contains no separate upstream license or
pinned-source ledger. Preserve and resolve that provenance when translating
actual code; this review imports no implementation. The pinned archive SHA256
and independent stripe/allocation calculations are recorded in
`out/pja-port-20261117/comr9-source-review.json`.

## Hashless experimental version-1 bytes

All multibyte integers are little-endian; there is no standard-JXL container.
The independent eight-byte magic is ASCII **`JXLPLUS\n`** (hex
`4a 58 4c 50 4c 55 53 0a`). The extension is not the format discriminator.

| Field, in order | Representation |
| --- | --- |
| Magic | 8 bytes |
| Version, flags | u16 = 1, u16 = 0 |
| Original pixel width, height | u32 each |
| Original JPEG length | u64 |
| Component count, band count, reserved | u8, u8, u16 = 0 |
| Compressed reconstruction metadata length | u64 |
| Component descriptors | Repeated component count times |
| Band stream lengths | u64 repeated band count times |
| Reconstruction metadata | Exactly the declared byte length |
| Arithmetic streams | Band order, exactly their declared lengths |

Each component descriptor is 140 bytes: `h_sampling:u8`, `v_sampling:u8`,
`reserved:u16=0`, padded block `width:u32`, padded block `height:u32`, then
64 quantization entries as u16 in natural frequency order. Coefficients
are carried only by the band streams. The fixed header is 40 bytes; total
framing includes `40 + 140*components + 8*bands` bytes, before metadata
and arithmetic payloads.

The metadata payload is `rs-jxl`'s Brotli-compressed JPEG reconstruction
representation, not a stored copy of the original JPEG. It is coupled to
the current reconstruction implementation; this is an experimental syntax,
not a promise of stable interchange across future revisions. Unknown
versions (including the old version 0), nonzero flags/reserved fields,
inconsistent geometry, truncated declared lengths and trailing container
bytes are rejected. Each arithmetic band checks actual consumed length and
its canonical termination state. Reconstructed JPEG length must equal the
declared original length. These are structural checks, not a replacement
content hash: a valid-looking modification can reconstruct different bytes.
Encoding retains its mandatory decode-and-compare of every original input
byte before returning an output. No claim of detecting every corruption or
authenticating content is made.

## Static probability experiment, 2026-11-17

The test-only experiment retains v7 coefficient prediction, event traversal,
direct bits and band boundaries. Pass one counts observations; pass two and
decoding use transmitted frozen 16-bit probabilities. Adaptive counters and
mixer weights are bypassed. One whole-image table is shared by all bands.
Tables use a versioned dense/sparse representation, whichever is smaller.
Their complete serialized cost and arithmetic termination bytes are included.

Three fixed context projections were checked at 1K, 4K, 16K and 64K buckets:
the full context tuple, the primary counter, and the mixer selector. Every
variant reconstructed original JPEG bytes; all 12 variants per input had the
same modeled/direct event counts. This is 84 checked variants on the same
seven small smoke inputs, with current input SHA256 identities verified.

| Configuration | Aggregate equivalent complete bytes | Change vs adaptive |
| --- | ---: | ---: |
| Actual adaptive hashless v1 | 3,138,437 | baseline |
| Full-tuple static, best fixed capacity (16K) | 4,557,650 | +45.2204% |
| Primary-counter static, best fixed capacity (64K) | 3,729,671 | +18.8385% |
| Selector static, best fixed capacity (64K) | 3,731,178 | +18.8865% |
| Smallest tested static variant chosen per input | 3,687,998 | +17.5107% |

Static totals add actual table/stream bytes to the same v1 framing and
reconstruction information; they are accounting equivalents, not outputs of
a public static container API. Per-file selection is accounted for by the
table's profile/capacity fields. Its tables total 120,800 bytes; coefficient
streams are 428,761 bytes larger than adaptive, adding 549,561 bytes overall.

These results do not isolate online learning alone: the frozen models replace
the multiple-expert mixer with a single conditional Bernoulli estimate, and
finite context hashing introduces collisions. Laplace smoothing, table layout
and whole-image fitting are also choices. This is neither an optimum nor a
lower bound for static coding, and it makes no speed claim. Adaptive coding
remains the public default. Experiment code is excluded from normal builds.

Detailed receipt: `out/pja-port-20261117/static-probabilities-profiles.json`.
Run the opt-in test `static_probability_seven_inputs` in release with
`RS_JXL_PLUS_STATIC_MANIFEST` pointing at the prior seven-input comparison JSON
and `RS_JXL_PLUS_STATIC_RECEIPT` naming a writable JSON receipt destination.

## JXL CfL plus v7 composition experiment, 2026-11-17

The first concrete JXL/PJA combination reuses `rs-jxl`'s reversible AC
chroma-from-luma transform, then encodes the resulting coefficients with the
unchanged native v7 model. Both Source and ActualMaximum factor fitting are
tested. This is test-only code in `src/tool_tests.rs`; it neither changes the
public default nor defines the final extended codestream.

Only two of the existing seven inputs qualify for the current three-component
relative-4:4:4 CfL implementation. The other five retain the original v7
candidate. Residuals are checked for the v7 coefficient range, never clipped.
The inverse uses decoded luma, transmitted factor maps and ratios derived
from the transmitted quantization tables. All four eligible candidates
reconstruct both the original coefficients and every original JPEG byte.
A separate zero-factor control checks coefficient and v7-stream identity.

| Input | Original v7 candidate | CfL Source + v7 | CfL ActualMaximum + v7 | Selected |
| --- | ---: | ---: | ---: | --- |
| rgb-adobe | 836,962 | 829,410 | 829,429 | CfL Source + v7 |
| project-wallhaven-yqqvv7 | 1,348,568 | 1,350,251 | 1,349,903 | Original v7 |
| All seven, including five ineligible inputs | 3,138,444 | — | — | 3,130,892 |

Numbers are equivalent complete bytes: unchanged v1 framing and reconstruction
metadata, actual coefficient streams, and actual descriptor/map bytes. The
original candidate includes a one-byte mode selector; the CfL candidate
includes an 18-byte descriptor and arithmetic-coded maps. Selection saves
7,552 bytes (0.2406%) against the baseline with the same selector cost, or
0.9023% on the RGB input. Compared with the existing selector-free v1 total
of 3,138,437 bytes, the saving is 7,545 bytes. This is a small focused sample,
not a corpus estimate or a speed comparison.

The v7 spatial/boundary predictors now operate on CfL residual coefficients;
their estimates no longer describe the original DCT boundary. That explains
why reversibility alone cannot guarantee a smaller result. Keep both tools
available and choose using their full costs. Receipts are
`out/pja-port-20261117/cfl-v7-tools.json` and `cfl-v7-tools.log` in the same
directory. Run `cfl_then_v7_seven_inputs` in release with
`RS_JXL_PLUS_TOOL_MANIFEST` pointing at the seven-input comparison JSON and
`RS_JXL_PLUS_TOOL_RECEIPT` naming the output receipt.

Future composition separates reversible transforms/predictors from alternative
entropy backends. CfL, RCT and MA/weighted prediction can prepare a declared
coefficient or residual domain; PJA models and JXL token/HybridUint/ANS or
prefix coding are alternative encodings for a segment. Token LZ belongs
before entropy coding. Fitting and model search are encoder decisions; the
decoder needs only the selected parameters. Per-group selection must account
for shared tables/trees, mode descriptors, reset dependencies and offsets.
Summing independent payload minima would omit these costs. The earlier
single-Bernoulli static experiment is not an evaluation of the JXL static
entropy toolchain, and it does not justify removing that toolchain.

## JXL MA-tree plus v7 composition experiment, 2026-11-17

The test-only `src/ma_tool_tests.rs` uses the existing JXL MA learner and
predictors on JPEG's quantized coefficient planes. One learned tree is shared
across component/frequency planes per candidate. The two candidates transform
only DC, or all 64 frequencies, before unchanged adaptive v7 coding. This is
not a static-probability test or a pixel-lossless image experiment.

Training uses five predictors (Zero, Left, Top, Gradient and Weighted), seven
properties (0, 4, 5, 9, 10, 11, 15), at most eight leaves, 64 property bins
and 4,096 deterministic samples across the full coefficient population.
Weighted state advances for every coefficient, including unsampled positions.
Stateful property 8 is explicitly excluded because the property collector
only runs at sampled positions. These finite search choices are not an
exhaustive assessment of MA prediction.

Actual tree nodes and the Weighted header are serialized, compressed with
the raw model and strictly parsed back. The inverse uses this decoded tree
and decoded causal coefficient neighbors, never original neighbors or the
encoder's tree. Residuals must fit the v7 i16 domain; unsupported candidates
are rejected rather than clipped. MA dependencies can cross v7 band
boundaries, so this experiment finishes band decoding before inverse MA.

| Candidate on the same seven JPEGs | Equivalent complete bytes | Change vs baseline |
| --- | ---: | ---: |
| Original v7 plus one-byte mode selector per file | 3,138,444 | baseline |
| MA DC prediction then v7 | 3,250,631 | +3.5746% |
| MA all-frequency prediction then v7 | 3,348,627 | +6.6970% |
| Choose smallest candidate per file | 3,138,444 | unchanged; all select original v7 |

Every one of the 14 MA candidates reconstructed original coefficients and
all original JPEG bytes. Total MA descriptors, including trees, Weighted
parameters and framing, cost 637 bytes for DC or 619 bytes for all frequencies.
After subtracting the baseline's seven selector bytes, these account for
only 630/612 bytes of the respective 112,187/210,183-byte growth. Most of the
growth is in coefficient streams, not the tree description.

The experiment changes the coefficient domain while retaining v7's own
prediction machinery. Its source physical-boundary estimates no longer
describe original coefficients. The result therefore supports keeping this
particular composition optional; it does not establish that MA trees are
ineffective, nor evaluate replacing the v7 predictor or conditioning dynamic
probability models directly on MA leaves. Pixel-domain MA/CM, JXL ANS and
joint CfL+MA candidates remain separate experiments. No tool was removed.

Receipt: `out/pja-port-20261117/ma-v7-tools.json`. Run the ignored release test
`ma_tree_then_v7_seven_inputs` with `RS_JXL_PLUS_MA_MANIFEST` pointing at the
existing seven-input comparison and `RS_JXL_PLUS_MA_RECEIPT` naming the output
receipt. Three focused debug tests passed (nontrivial learned-tree/v7
reconstruction, zero-predictor stream identity, explicit Weighted reconstruction
and overflow rejection); the seven-input release test passed in 21.32 s.
Rust 1.89 checking of this crate's tests and package format checking passed.
Timing fields in the receipt are stage diagnostics, not a formal benchmark.

## MA-selected dynamic probability contexts, 2026-11-17

This follow-up preserves the original coefficient domain, traversal,
predictions, adaptive counters and expert-weight updates. Test-only event
notifications identify luma/chroma namespace, one/avg/mix event kind,
counter/mixer bank, selector/index and referenced counters. A learned MA-style
tree selects an online calibration context immediately before arithmetic
coding. Direct bits bypass it. The original experts still update from their
original probabilities and the actual bit; no coefficient predictor is
stacked onto v7.

Each leaf has 16 base-probability bins, each with an initially zero Q16 log-odds
bias. The frozen PJA squash table converts integer bias into an odds ratio;
integer arithmetic applies it to the original probability. Zero bias is
exactly the original probability. After each modeled bit, the selected bias
updates from the calibrated prediction error, bounded to +/-512 integer
table steps. Rates 6 and 8 were checked, with every bias reset at each v7 band
boundary. This is an additional calibration layer, not replication of the
entire v7 expert model per leaf.

The tree learner uses up to 32,768 deterministic reservoir samples across
the complete original v7 event walk. It fits conditional Bernoulli log loss
with the base probability as an offset, separately in each of the 16 bins.
It tests nine constant fitting biases (-256, -128, -64, -32, 0, 32, 64, 128,
256), at most 32 thresholds per property, at least 128 samples per child,
a 16-bit training-gain gate and at most eight leaves. This is a new bounded
context-partition learner, not the JXL residual-histogram predictor learner.
The fitted biases are discarded; only the partition is transmitted, and
runtime states learn online from zero. Floating-point fitting is encoder
search only. Final selection uses actual encoded bytes, not fitted losses.

Three controls use the same calibration rule: one leaf with 16 states;
six fixed namespace/event-kind leaves with 96 states; and a learned tree
with up to 128 states. All seven learned trees reached eight leaves. The
context properties come from existing decoder-visible event IDs; this
experiment introduces no extra spatial coefficient features. State counts
differ between controls, so their difference is not a pure capacity-matched
causal estimate of learned partition quality.

| Same seven JPEGs; one configuration for every input | Equivalent complete bytes | Change vs original v7 candidate |
| --- | ---: | ---: |
| Original v7, including one mode byte per file | 3,138,444 | baseline |
| One-leaf calibration, rate 6 | 3,138,514 | +0.0022% |
| One-leaf calibration, rate 8 | 3,137,052 | -0.0444% |
| Fixed six-leaf calibration, rate 6 | 3,134,360 | -0.1301% |
| Fixed six-leaf calibration, rate 8 | 3,133,633 | -0.1533% |
| Learned MA contexts, rate 6 | 3,136,200 | -0.0715% |
| Learned MA contexts, rate 8 | 3,134,967 | -0.1108% |

Choosing between original v7 and the two learned-tree rates per file totals
3,134,744 bytes, saving 3,700 bytes (0.1179%). The corresponding choice with
fixed contexts totals 3,133,483 bytes, saving 4,961 bytes (0.1581%). Allowing
all six calibration candidates and original v7 totals **3,133,469 bytes**,
saving **4,975 bytes (0.1585%)**. Only `illustration-icc` selects a learned
tree in this full pool: rate 6 is 14 bytes smaller than its best fixed-context
candidate. Thus calibration helps this sample, while learned partitions do
not consistently outperform the fixed context control. The earlier CfL
candidate is not included in this pool, and savings from separate experiments
must not be added without testing their composition.

Complete costs include unchanged v1 framing/reconstruction information,
the real compressed tree/profile/rate descriptor, a five-byte outer mode and
length header, and all coefficient streams/termination. Rate-8 descriptors
total 154 bytes for one leaf, 294 for fixed leaves, and 417 for learned trees.
Compared to the existing selector-free v1 total of 3,138,437 bytes, the full
pool saves 4,968 bytes. No final public wire layout is implied.

All **42** candidates reconstructed every original coefficient and JPEG byte,
with unchanged modeled/direct event counts. The inputs exercise one, two
and five v7 bands. Nine focused tests passed, including zero/disabled
calibration identity, direct-bit notification consumption, exact budgets,
strict descriptor parsing, genuinely useful synthetic tree splits and
controls preventing splits based only on already-explained probabilities or
probability-bin mixtures. The unchanged-v7 original-C comparison passed after
the hook change. Production checking, Rust 1.89 test checking and formatting
passed. The focused release experiment took 39.20 s; this is a test duration,
not a production speed comparison.

Receipt: `out/pja-port-20261117/dynamic-ma-tools.json`. Reproduce with the
ignored release test `dynamic_ma_context_seven_inputs`, setting
`RS_JXL_PLUS_DYNAMIC_MA_MANIFEST` to the existing seven-input manifest and
`RS_JXL_PLUS_DYNAMIC_MA_RECEIPT` to the output JSON destination. Public defaults
remain unchanged. Pixel-domain MA/CM and richer coefficient context features
remain unmeasured by this JPEG-only calibration experiment.

## Broader compression-tool search, 2026-11-17

The next search expands beyond MA trees while retaining every existing tool.
It evaluates three independent mechanisms and their actual compositions on
the same seven JPEG originals:

- V7 with one, two, four or the original number of bands, after deduplication
  and geometry checks. Full/Fast are also checked at their original band count.
- Finer online probability refinement, retaining original v7 predictions and
  expert updates: a namespace/event-kind/bank layer interpolates 33 log-odds
  knots; optional 8,192-entry selector contexts add a second correction.
- Both existing JXL CfL fits combined with the best raw-domain V7, legacy and
  refinement recipes. Each combination is encoded and independently reversed;
  standalone savings are never added to predict a combined result.

The refinement profiles are fixed, decoder-visible rules: profile 0 is exact
v7 probability pass-through; profile 1 uses the bank layer; profile 2 adds
selector groups (`index >> 4`); profile 3 adds exact selector indices. Both
selector modes use a bounded deterministic hash. Adjacent log-odds knots are
linearly interpolated with integer arithmetic. Coarse/fine update shifts are
8/9, their Q16 bias bounds are 512/256, and the final combined integer bias
is bounded to +/-512. Every band starts at zero bias, so direct bits and
original v7 learning remain intact. Extra calibration arrays occupy 50,688
bytes for profile 1 or 1,132,032 bytes for profiles 2/3, per active band; these
are array sizes, not process-wide memory measurements. No probability-state
snapshot is transmitted.

| Growing candidate pool, choosing by complete size per file | Total bytes | Saving vs original v7 candidate |
| --- | ---: | ---: |
| Original v7, including one mode byte per file | 3,138,444 | baseline |
| Add V7 band counts and Full/Fast | 3,124,608 | 13,836 B / 0.4409% |
| Add fine probability refinement | 3,115,603 | 22,841 B / 0.7278% |
| Add actual CfL combinations | **3,111,998** | **26,446 B / 0.8426%** |

These are nested candidate-pool differences, not isolated causal contributions
that can be added to other experiments. The last total is 21,471 bytes smaller
than the previous dynamic-MA pool, or 16,660 bytes below the best per-file union
of the earlier standalone CfL and dynamic-MA candidates. Compared with the
selector-free public v1 total, the saving is 26,439 bytes.

| Input | Original v7 candidate | Selected complete bytes | Selected recipe |
| --- | ---: | ---: | --- |
| baseline-420 | 3,305 | 3,305 | Original v7 |
| progressive-420 | 3,133 | 3,133 | Original v7 |
| restart-bicycles | 66,691 | 66,616 | Refinement 2, one band |
| rgb-adobe | 836,962 | 824,199 | CfL Source + refinement 2, one band |
| wallhaven-regression | 87,085 | 87,044 | Refinement 2, one band |
| illustration-icc | 792,700 | 790,871 | Refinement 2, one band |
| project-wallhaven-yqqvv7 | 1,348,568 | 1,336,830 | Refinement 3, one band |

The RGB input improves by 1.5249%; project-wallhaven improves by 0.8704%.
Full/Fast do not win these seven comparisons but remain available. One-band
V7 improves the two inputs that originally had two/five bands; the other five
were already single-band. The reduction is predominantly coefficient-stream
size, not the few length/termination bytes: changing bands also changes model
warmup and prediction at band boundaries. Fewer independent bands can reduce
available parallelism; this run does not measure that tradeoff.

Accounting uses actual descriptors and streams, unchanged reconstruction
metadata, and `40 + 140*components + 8*actual_bands` framing. Band/model
descriptors are four bytes; refinement descriptors are fourteen bytes and
explicitly describe the profile/band/precision/update parameters. CfL
compositions include their actual coded maps, inner coding descriptor and a
ten-byte mode/version/two-length header. Decode parses these fields, decodes
the coefficient payload, and reverses CfL using only decoded luma/maps.

All **75 candidates** restored original coefficients and every original JPEG
byte. Seven focused tests passed for source-mode identity, independent band
decoding, all refinement profiles, direct-bit bypass, complete budgets,
truncation and descriptor boundaries. Rust 1.89 test checking and package
format checking passed. The opt-in release test took 79.44 s; this duration
is not a codec throughput benchmark. Original source-band V7 streams also
matched the baseline on all seven real inputs.

The search is finite: refinement tries source and best raw-V7 band counts;
CfL tries the two supported fits with three chosen entropy recipes. It is
not an exhaustive cross-product, a representative corpus, or a pixel-lossless
experiment. The formats here remain internal experiments and the public
defaults are unchanged. Receipt:
`out/pja-port-20261117/compression-tool-search.json`. Reproduce with release
test `compression_tools_seven_inputs` and the variables
`RS_JXL_PLUS_SEARCH_MANIFEST` and `RS_JXL_PLUS_SEARCH_RECEIPT`.

## Speed optimization and larger fixed corpus (2026-11-17)

The user now prioritizes aggregate complete bytes on a larger fixed dataset;
individual files may grow. `tools/build_jxl_plus_corpus.py` records 301 unique
source hashes: the seven development images, 38 other previously audited
images, and 256 previously unused CC12m JPEGs. The new subset is 45,298,243
bytes. From 2,960 local JPEGs, four equal-rank file-size strata each contribute
64 images, ranked by SHA256 of their relative path and excluding previously
used or duplicate contents. Selection does not depend on compression results.
The four population size ranges are 3,395..81,738, 81,746..133,075,
133,185..218,853 and 218,870..904,437 bytes. This is a local stratified sample,
not a claim of representativeness across image domains or other JPEG families.

The fixed evaluation includes source-band V7, one-band V7, one-band refinement
profiles 2/3, and Source-CfL plus each refinement profile where eligible.
Inapplicable CfL inputs retain V7 for a full-coverage policy comparison.
It records fixed-policy totals separately from the measured per-file minimum;
the latter requires candidate search and is not a cheap encoder policy.
All descriptors, maps, reconstruction metadata and band framing are counted.
The JSONL progress file is tied to a runner-supplied build fingerprint; it may
not resume across implementations under the same recipe names. Source hashes
are checked externally, while every encoded candidate must independently
restore coefficients and the original JPEG bytes.

The run completed **301/301 inputs, 1,228 actual candidate encodes/decodes,
zero skips or reconstruction failures**. All 301 source SHA256 values were
rechecked after the run. Complete-size results:

| Split | Original V7 candidate | Fixed refinement 2 | Fixed refinement 3 | Per-file minimum over six tools |
| --- | ---: | ---: | ---: | ---: |
| Development 7 | 3,138,444 | 3,115,639 (-0.7266%) | 3,115,994 (-0.7153%) | 3,111,998 (-0.8426%) |
| Other prior audit 38 | 22,023,323 | 21,847,993 (-0.7961%) | 21,849,038 (-0.7914%) | 21,781,050 (-1.1001%) |
| Previously unused 256 | 29,556,007 | **29,490,840 (-0.22049%)** | 29,495,188 (-0.20578%) | 29,490,819 (-0.22056%) |

The new-set gain is smaller than on development/prior inputs. Both refinement
profiles shrink all 256 inputs in this particular sample; that is an observed
result, not a per-file guarantee. Profile 2 saves 13..1,115 bytes per image;
the four size strata improve 0.2423%, 0.2544%, 0.2568% and 0.1942% respectively.
All 256 were already one-band and none admitted the existing 4:4:4 CfL path,
so this validation does not establish the broader CfL or band-reset gains.
One-band V7 alone adds only the three extra descriptor bytes per file.
The measured minimum saves just 21 more bytes than fixed profile 2 on the
entire new set: searching all six tools is unnecessary for almost all of this
observed gain. With speed now a priority, the added refinement cost should
remain an explicit size/throughput choice; its good size result alone does not
justify making it the default. The public V7 default is unchanged.
Receipts: `corpus256-manifest.json`, `corpus256-fixed-experts.jsonl` and
`corpus256-fixed-experts.json` under `out/pja-port-20261117/`.

Fresh standard-JXL references on the same 256 previously unused images use
the public Rust encoder, fixed effort 3/9, one thread and Brotli quality 6.
All **512 standard outputs** also restore their original JPEG bytes. Effort 3
totals **32,386,114 bytes**, effort 9 **31,398,801 bytes**. Original Plus V7
therefore saves **8.7386% / 5.8690%** respectively; fixed refinement 2 saves
**8.9399% / 6.0765%**. These compare complete files, with the Plus candidate's
mode byte included. No per-input standard-JXL effort search is performed.
Receipts: `standard-corpus256.jsonl` and `standard-corpus256-summary.json`.

The speed changes preserve model arithmetic and stream bytes:

- Fixed expert counts now remain array lengths through `Model::mix` and the
  probability update, removing the per-event slice of mutable references and
  dynamic expert loops.
- Dedicated eight-lane x86_64 AVX2 mixer operations use zero input/weight
  padding, with runtime feature detection and an equivalent scalar fallback.
- Refinement reuses adjacent state pairs and precomputed row bases. Fixed-rate
  learning cancels integer factors before multiplication, preserving signed
  truncation, saturation and all state values. The odds correction is unchanged.

On the local Ryzen 9 7945HX, release builds, one thread, the same seven inputs
(4,425,657 original bytes / 18,778,820 pixels), one warmup and three measured
repetitions give these **coefficient-stage** totals (sum of per-file medians):

| Model | Encode before / after, s | Decode before / after, s | Encode / decode throughput increase |
| --- | ---: | ---: | ---: |
| Source-band V7 | 1.6822 / 1.3694 | 1.8713 / 1.5043 | 22.8% / 24.4% |
| One-band V7 | 1.7182 / 1.3706 | 1.8593 / 1.4948 | 25.4% / 24.4% |
| Refinement 2, one band | 2.6706 / 2.0615 | 2.7017 / 2.2746 | 29.5% / 18.8% |
| Refinement 3, one band | 2.6461 / 2.1466 | 2.7270 / 2.3004 | 23.3% / 18.5% |

An intervening repeat of the original binary measured V7 1.6765/1.8470 s,
supporting a 22.4%/22.8% gain against that control too. Refinement timings vary
more (profile 3 baseline 2.8137/2.8609 s on the repeat); these are local
measurements, not a guaranteed cross-machine speedup. The initial isolated
SIMD/refinement edits alone did not give a clear gain; the final result includes
the fixed-expert implementation. All 28 pre/post descriptor-and-stream
snapshots are identical, and all decoded coefficients/original bytes match.
Timing includes model allocation and zigzag conversion, excludes input reads,
JPEG preparation, destination clearing, assertions and JPEG reconstruction.
Receipts: `speed-before.json`, `speed-before-recheck.json`,
`speed-fixed-experts.json` under `out/pja-port-20261117/`.

Two subsequent attempts on the default path were measured and **reverted**.
Inlining scalar prefix dot/weight updates instead of the fixed-width runtime
kernel took source-V7 1.6236/1.5701 s encode/decode. An exactly equivalent
unsigned-32-bit counter update took 1.4235/1.5257 s. Neither beat the retained
1.3694/1.5043 s fixed-expert implementation; no additional speedup is claimed.
Both retained identical pre/post streams while tested. Receipts:
`speed-inline-experts.json` and `speed-u32-counter.json`. The final code keeps
the signed source counter recurrence and the faster fixed-width mixer path.

A separate public API comparison uses the same seven inputs, one thread,
metadata Brotli quality 6, one warmup and three rounds with rotated order:

| Public API | Complete bytes | Encode, s (MB/s of original JPEG) | JPEG restoration, s (MB/s) |
| --- | ---: | ---: | ---: |
| Standard `rs-jxl`, effort 3 | 3,490,623 | 0.8302 (5.33) | 0.3156 (14.02) |
| Standard `rs-jxl`, effort 9 | 3,478,051 | 6.9012 (0.64) | 0.3446 (12.84) |
| Plus V7 public API | 3,138,437 | 3.3447 (1.32), **includes full verification** | 1.6860 (2.62) |

These are medians of whole-set round totals, with decimal MB/s and all source
bytes checked outside the timed decoder calls. Standard encoding alone does
not include a verification decode. Its encode-plus-restore median is 1.1514 s
at effort 3 and 7.2573 s at effort 9; Plus's comparable verified-encode call is
3.3447 s. Do not add its separately measured decode again. Standard decoding
uses the maintained official Rust decoder's coefficient capture and shared
JPEG reconstruction. This measures the current Rust implementations, not
upstream C++ libjxl. These effort-3/9 options also differ from the earlier
saved tuned/diagnostic JXL comparison. Receipt: `public-api-speed.json`.

Reproduce coefficient timing with ignored `coefficient_speed` and
`RS_JXL_PLUS_SPEED_{MANIFEST,RECEIPT,SNAPSHOTS}`, setting `RS_JXL_PLUS_SPEED_SAVE=1` only
for a fresh reference directory (existing snapshots are never overwritten).
The public API test is `public_jpeg_api_speed`, with `RS_JXL_PLUS_SPEED_MANIFEST` and
`RS_JXL_PLUS_API_SPEED_RECEIPT`. The fixed corpus test is `fixed_tools_corpus`
with `RS_JXL_PLUS_CORPUS_{MANIFEST,JSONL,RECEIPT,RUN_ID}`. Keep performance
timing separate from concurrent compilation or corpus work.

### Parallel and Fast tradeoffs; defaults remain V7

The user explicitly keeps the default on V7 with the original band rule and
one thread. Probability refinement, Fast and additional parallel bands remain
optional internal experiment paths; no public syntax or CLI default selects
them. The fixed-expert implementation improvements apply to ordinary V7.

On the seven development inputs, one warmup and three repetitions at the
fixed-expert checkpoint gave the following sum-of-file-median stage timings:

| Candidate | Complete bytes | Change vs source V7 | Encode / coefficient-decode MB/s |
| --- | ---: | ---: | ---: |
| V7, one band, one thread | 3,124,623 | -0.4404% | 3.14 / 2.90 |
| V7, source bands, up to four threads | 3,138,465 | +21 descriptor bytes only | 5.25 / 4.77 |
| V7, four bands/four threads | 3,162,216 | +0.7574% | 8.68 / 8.14 |
| V7, eight bands/eight threads | 3,196,005 | +1.8341% | 13.48 / 13.52 |
| Fast, source bands, one thread | 3,473,159 | +10.6650% | 4.84 / 4.39 |

These are coefficient stages, not complete JPEG restoration or a comparison
to standard-JXL whole-API throughput. Timing includes each call's pool,
model allocations and coefficient conversion. More cores and simultaneously
live model states are used; the adaptive model has not become asymmetric.
Every parallel stream matched a serial encode with the same band count and
all candidates restored original JPEG bytes. Source-band parallelism preserves
original streams; extra bands alter reset/warmup and boundary prediction.
The four-byte tool descriptor costs three bytes more per file than the
one-byte original-mode candidate, including when source streams are identical.

The independent 256-image set then verified **512 parallel candidates**:

| Fixed parallel recipe | Complete bytes | Change vs source V7 | Smaller than standard E3 / E9 |
| --- | ---: | ---: | ---: |
| Four bands/four threads | 30,213,173 | +2.22346% | 6.70948% / 3.77603% |
| Eight bands/eight threads | 30,769,342 | +4.10521% | 4.99218% / 2.00472% |

Band count is capped at the input's MCU rows. All 512 restored coefficients
and original bytes exactly, with zero exclusions. Source hashes were rechecked
after all corpus runs. This larger-set reset cost is greater than on the seven
development images, so the small-set size tradeoff must not be generalized.
Four bands are a useful optional balance here, while source-band V7 remains the
default per the user's instruction. Fast is retained despite its unfavorable
tradeoff in the seven-image timing run. Receipts: `fast-parallel-speed.json`,
`parallel-corpus256.jsonl`, `parallel-corpus256-summary.json`.

Parallel decoders now check canonical arithmetic termination on every band
before publishing any coefficients. Focused tests cover 1/2/4 workers, serial
stream identity, aggregate output budgets and late-band truncation/trailing
bytes/corrupt endpoints without publishing partial output. Original-source
loose padding is available only to test/oracle code.

Final retained-source checks passed: 100 release library tests, nine normal
integration tests plus the explicitly enabled seven-file integration test,
and the independent original-C v7 stream/cross-decode oracle. Rust 1.89
`check --tests`, package formatting and `git diff --check` passed. The corpus
receipts above cover 1,228 Plus tool candidates, 512 standard-JXL outputs and
512 parallel V7 outputs; all use original-byte verification. The discarded
follow-up speed trials are not left enabled in the source.

### Pixel-lossless route: JXL tools plus adaptive entropy

The first internal experiment is implemented in test-only `pixel_model.rs`.
It is **not yet the final JXL-derived pixel Plus format or a public API**.
The broader tool-composition direction remains:
Retain JXL's channel layout, reversible RCT/palette/squeeze transforms,
prediction, MA tree, weighted-predictor state, leaf offset/multiplier and
group dependency semantics. Add a signalled entropy alternative at the
Modular residual layer; do not transplant the JPEG coefficient model or the
unfinished comr9 format into pixels.

Relevant existing `rs-jxl` boundaries are:

| Source | Reusable work / extension point |
| --- | --- |
| `modular/encode.rs::planes`, `modular/group.rs` | Prepared channels and reversible transform descriptors |
| `modular/encode.rs::{learn,visit_channel_with_history}` | Tree and causal prediction, including weighted and prior-channel history |
| `modular/tokens.rs::channel` | Produces the leaf context and signed-packed residual after offset/multiplier; all fixed/pair/scalar dispatch paths must feed the extension consistently |
| `modular/encode.rs::{group_stream_with_history,parse_group_tokens}` | LZ is an independent selectable tool; avoid conflating its presence with entropy-model changes |
| `entropy/hybrid_uint.rs`, `entropy/stream.rs::Model` | Existing integer token/extra-bit mapping, static model/header/ANS or prefix coding |
| `modular/encode.rs::write_prepared_stream` | Current group/header/tree/model composition boundary |

A low-cost first experiment can adapt only the **zero/nonzero decision** for
each transformed prediction residual. A context such as MA leaf plus its
previous zero decision selects one online counter; nonzero values continue
through the existing HybridUint and static ANS path. This adds one adaptive
update per sample, no mixer/APM and state proportional to the bounded leaf
count. It may skip static-symbol decoding on zeros, but its speed and size
benefit are unproven. Start with independent group resets and explicitly
define channel-history resets and exact integer update rules.

Transmit separate lengths for the gate stream and nonzero entropy stream,
with versioned profile/reset parameters. The decoder must obtain the MA leaf
from already reconstructed neighbors, read the gate, decode a nonzero token
only when required, and apply the original predictor/transform inverses.
Encoder-supplied original samples/context traces are not decoder inputs.

Use three matched controls on identical prepared planes/tree/WP/HybridUint:
full static entropy, the split syntax with transmitted fitted fixed gate
probabilities, and the same syntax with adaptive gate probabilities. Initially
disable LZ on all three controls; also compare with the complete unchanged
default JXL encoder, including its LZ/tools. Count actual group/tree/transform
and probability descriptors, both stream lengths/termination, TOC and padding.
Only real decode and sample equality establish a result. Include invisible
RGB, extra channels and wide integer values without float storage or clipping.

The new bounded `modular::plus_bridge` shares preparation and causal prediction,
while `entropy::experimental_decode` parses serialized entropy models and trees
through the maintained native Rust decoder. Dependencies remain one-way
`rs-jxl-plus -> rs-jxl`. Standard encoder/decoder defaults are unchanged.
A JPEG experiment does not establish pixel compression gains or completion of
the remaining standard pixel port milestone.

The first measurement's profile accepts interleaved Gray/GrayAlpha/RGB/RGBA
unsigned integer samples of one shared 1..31-bit depth. It uses independent
256x256 groups, RCT 6 for RGB, actual MA learning (up to 16 leaves), JXL WP,
and fixed HybridUint(4,2,0) with static ANS. Palette, squeeze, LZ, extra planes,
subsampling, animation, floating-point samples and metadata are outside this
first experiment. All pixel values, including RGB at alpha zero, are preserved.

Profile 1 signals the mode and group geometry in `JXLPPX01` internal framing.
Each group stores its RCT descriptor, actual JXL WP/tree/model bitstreams with
exact bit lengths, symbol count, gate byte length, fitted probability table
when applicable, and residual stream. No integrity hash is stored. This is a
self-contained experimental representation: its framing bytes are counted,
but they are not a finalized standard JXL frame/TOC extension. The profile fixes
the adaptive PJA `AdaptiveBit` recurrence (initial p=32768, n=0), context
`2*MA_leaf + previous_zero_for_that_leaf`, and reset at every group. Channels
share gate state inside a group. The fitted control transmits a rounded/clamped
16-bit zero probability for every context, including unused contexts; it is not
a frozen uniform prior. Nonzero packed residuals retain their original values.

Decoding parses all models from bytes, obtains each MA leaf from reconstructed
neighbors, validates exact symbol consumption and arithmetic/ANS termination,
and inverts RCT without floats or clipping. Caller byte/sample ceilings are
checked before sample allocation. Normal default JXL is also measured end to
end, but has a different tool policy and optimized decoder; only the three
experimental modes isolate the gate's cost and gain.

#### First fixed pixel-corpus result (2026-11-17)

`tools/build_jxl_plus_pixel_corpus.py` fixes 20 complete PNG images by category
and filename, without compression-based selection: 15,830,823 pixels and
20,064,758 original PNG bytes. Seven images (4,140,631 pixels) were present
in the recorded prior runs or conservatively treated as development fixtures;
13 (11,690,192 pixels) were absent from those records. The latter classification
does not prove they were never used elsewhere. Coverage includes photos, UI,
web, text, illustrations, line art, textures, palette, alpha, gray and 16-bit.

Windows x86_64 / Ryzen 9 7945HX, release thin LTO, one thread, one warmup plus
three measured rounds with rotated order. Times below sum the per-image
medians; throughput uses image pixels, not channel samples or compressed bytes.
Encoding includes preparation/training/model writing; decoding parses the
actual complete bytes and reconstructs samples. PNG input decoding, I/O and
sample comparison are outside timing. Standard Native output also includes
an authoritative duplicate alpha plane where present; both it and main pixels
are verified. This is local diagnostic timing, not a general hardware claim.

| Fixed recipe | Complete bytes | Encode s | Decode s | Encode MP/s | Decode MP/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plus experiment: static JXL ANS | 15,218,813 | 3.706441 | 1.950968 | 4.271 | 8.114 |
| Plus experiment: fitted fixed zero gate | 15,103,081 | 3.640236 | 1.939469 | 4.349 | 8.162 |
| Plus experiment: adaptive zero gate | 15,058,580 | 3.735835 | 2.038906 | 4.238 | 7.764 |
| Standard `rs_jxl::encode`, default options | 14,950,598 | 10.307890 | 1.048907 | 1.536 | 15.093 |

The adaptive gate saves **160,233 bytes (1.05286%)** versus matched static
entropy, with 17 smaller / 3 larger images. Encode time rises 0.793%, decode
time 4.507%. Compared with the fitted-gate control, it saves 44,501 bytes
(0.29465%; 19 smaller / 1 larger). The default standard encoder still wins
overall by 107,982 bytes: the adaptive experiment is **0.72226% larger**.
Its encoding throughput is 2.759 times that default call, while decoding
throughput is 0.514 times. These standard comparisons mix tool policy and
decoder implementation differences; they do not isolate adaptive entropy.
The first profile lacks LZ and other selectable tools. The generic causal
bridge also lacks the normal decoder's specialized fast paths: even the
experiment's static decoder is slower than the standard decoder. No causal
attribution of the complete standard size gap to one missing tool is claimed.

| Sample scope | Static experiment bytes | Fitted gate bytes | Adaptive bytes | Standard bytes | Adaptive saving vs matched static |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previously used, 7 | 6,013,058 | 5,972,197 | 5,962,240 | 5,852,801 | 0.84513% |
| Absent from listed prior records, 13 | 9,205,755 | 9,130,884 | 9,096,340 | 9,097,797 | 1.18855% |

The 13-image scope is only 1,457 bytes (0.01601%) below standard; this is
effectively a tie at this dataset scale, not evidence of a broad standard-JXL
win. The full fixed dataset is the primary aggregate result. No per-file
minimum across recipes is used in these totals.

Cost accounting across 281 groups reconciles exactly. Adaptive gating removes
2,008,370 bytes from the static residual payload, adds 1,848,245 gate bytes,
and saves 108 model-header bytes, yielding the net 160,233-byte saving. Trees
(6,534 bytes), WP (1,405) and framing (5,847) are unchanged. Against the fitted
gate, nonzero entropy payload/header are identical: 38,537 saved bytes come
from adaptive gate probabilities and 5,964 from avoiding transmitted fitted
tables. Thus most of that incremental gain is probability adaptation, not
table omission. The larger matched-static improvement also includes the
benefit of explicitly splitting the zero decision into its Markov contexts.

Content results are mixed: adaptive versus matched static saves 1.164% on the
four photos, 28.526% on the one UI image, 12.485% on two web images and 17.123%
on two alpha images; the two texture images grow 0.023%. These tiny categories
are diagnostics, not general claims. Standard defaults remain substantially
smaller for some gray, line-art, text and palette inputs. Preserve the tools
and test composition/selection rather than replacing the static path globally.

All **240 measured outputs**, plus 80 warmup outputs, decoded to exact original
samples, including invisible RGB. Post-run source SHA256, deterministic sizes,
per-image medians and all cost totals were independently checked by
`tools/analyze_jxl_plus_pixel_corpus.py`. Receipts are
`out/pja-port-20261117/pixel-{manifest,corpus,corpus-analysis}.json`.
The first interrupted four-image attempt was discarded; the final receipt is
one complete run after correcting the harness's duplicate-alpha assumption.

Focused tests cover 8/16/31-bit samples, invisible RGB, constant images,
257x259 partial tiles, caller byte/sample limits, malformed/truncated framing,
exact entropy termination and dynamic MA context reconstruction. Six entropy
bridge tests, four prediction bridge tests, 103 release library regressions
and the four final pixel tests passed. Rust 1.89 checks include tests/examples.
The changed files pass formatting; a broader `rs-jxl` format check also found
pre-existing differences outside this work, which were left untouched.

The original four-recipe harness remains available without linking unrelated
codec tests. It now uses the current research framing; the exact historical
profile-1 totals above are preserved in the saved receipt:

```powershell
$env:RS_JXL_PLUS_PIXEL_MANIFEST = 'D:/cpp_pj/rs_jxl/out/pja-port-20261117/pixel-manifest.json'
$env:RS_JXL_PLUS_PIXEL_RECEIPT = 'D:/cpp_pj/rs_jxl/out/pja-port-20261117/pixel-corpus.json'
cargo test --release -p rs-jxl-plus --example pixel_lab -- pixel_tests::pixel_corpus --ignored --exact --nocapture --test-threads=1
python -B tools/analyze_jxl_plus_pixel_corpus.py
```

The following composition experiment advances the LZ and group-selection work.
Integration with the complete Modular tools and real JXL frame syntax remains
separate from these research packets. JPEG V7 defaults remain unchanged.

#### Residual LZ and actual group selection (2026-11-17)

Research framing is now **`JXLPPX02`**, profile 2. It rejects the old research
magic instead of silently reinterpreting it. The 28-byte image header no longer
stores an encoder recipe: its old mode byte is reserved zero. Each bounded
group starts with one flag byte: bits 0..1 select static/fitted/adaptive gate,
bit 2 says its entropy header enables LZ, and all other bits are zero. Mode 3
is invalid. Transforms, WP, tree, counts, gate/table and entropy sections follow.
All flag bytes are included in complete size: the three no-LZ controls are
exactly **281 bytes** larger than the first run, one per group.

The existing native Modular LZ parser is applied either to all residuals or to
the gate's filtered nonzero residuals. Full streams use tile-width spatial
distances; filtered streams use multiplier 0 and direct distance-minus-one.
Stream counts describe expanded values, not compressed command tokens. The
decoder reconstructs each MA context causally even during copies, and gates
still update once per sample. Model headers and group flags must agree about
LZ. Independent group resets include the LZ history. No-match parses retain
the matching no-LZ syntax; LZ is a tool, not a compulsory overhead.

The native entropy bridge's opt-in bounded LZ methods check output counts,
available history, the 2^20-value window, overlap, pending copies, exact bit
consumption and arithmetic/ANS ending. Ordinary standard-decoder methods are
unchanged. A statically specialized single MA leaf now bypasses unused
property/history computation in the causal bridge, while preserving predictors,
WP updates and wide wrapping arithmetic. This change has exactness coverage;
the run does **not** isolate its speed improvement from other changes.

`Selected` prepares each group once, then actually encodes Static, Adaptive,
StaticLz and AdaptiveLz, choosing the shortest complete group. Ties retain the
earlier candidate. The complete encoder timer includes all four trials, and
the emitted flags drive ordinary decoding with no encoder search knowledge.
Fitted modes remain available/tested but are not part of this four-candidate
policy. Selection is not an offline per-file oracle or a wire requirement.

Same fixed 20 PNGs, 15.83 MP, prior-use split and hardware as the first run;
one thread, one warmup, three rotated measured rounds. Per-image medians are
summed below. Standard means the direct default `rs_jxl::encode` call, not a
separate upstream cjxl executable or a tuned CLI effort search. Absolute times
vary from the earlier run; use **same-run comparisons**, not a cross-run speedup.

| Recipe | Complete bytes | Encode s | Decode s | Encode MP/s | Decode MP/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static | 15,219,094 | 4.543115 | 2.509765 | 3.485 | 6.308 |
| Fitted gate | 15,103,362 | 4.512930 | 2.330446 | 3.508 | 6.793 |
| Adaptive gate | 15,058,861 | 4.562254 | 2.514447 | 3.470 | 6.296 |
| Static + LZ | 15,019,940 | 10.567386 | 2.492851 | 1.498 | 6.350 |
| Fitted gate + LZ | 14,866,634 | 7.239361 | 2.377553 | 2.187 | 6.658 |
| Adaptive gate + LZ | 14,822,133 | 7.322254 | 2.479192 | 2.162 | 6.385 |
| Selected, all search included | 14,779,624 | 15.245159 | 2.511805 | 1.038 | 6.303 |
| Standard default | 14,950,598 | 12.570012 | 1.286521 | 1.259 | 12.305 |

Fixed Adaptive+LZ is **0.85926% smaller** than standard default, encoding at
1.717 times its throughput and decoding at 0.519 times. Selected is
**1.14359% smaller**, but encode time is 21.282% higher than standard and
decoding throughput is 0.512 times. Both are smaller on 15 images and larger
on five. The user-approved aggregate criterion is met on this fixed sample;
this is not a universal compression or speed claim.

LZ saves **236,728 bytes (1.57202%)** over the adaptive gate without LZ, with
eight smaller and twelve equal images. Selection saves another **42,509 bytes
(0.28679%)**, at 2.082 times the fixed Adaptive+LZ encoding time. This is the
tradeoff between a fixed fast candidate and more exhaustive encoder search;
both can use one decoder grammar. Adaptive+LZ is 1.31696% smaller than matched
Static+LZ and 0.29933% smaller than Fitted+LZ. Against Fitted+LZ the residual
payload/header are identical: 38,537 saved gate bytes plus 5,964 omitted fitted
table bytes explain the entire 44,501-byte difference.

| Scope | Standard bytes | Adaptive+LZ bytes | Selected bytes | Adaptive+LZ saving | Selected saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previously used, 7 | 5,852,801 | 5,845,837 | 5,816,420 | 0.11899% | 0.62160% |
| Absent from listed prior records, 13 | 9,097,797 | 8,976,296 | 8,963,204 | 1.33550% | 1.47940% |

The actual Selected distribution across 281 groups is 87 Static, 123 Adaptive,
12 Static+LZ and 59 Adaptive+LZ. This mixed result supports keeping tools
composable. Static+LZ alone emits copies in 17 groups; filtering nonzeros permits
the fixed Adaptive+LZ candidate to use them in 71 groups. This is measured
activation, not merely the requested recipe. Complete Selected cost reconciles
as tree 6,534; WP 1,405; gate 1,541,936; entropy header 23,125; entropy data
13,200,496; framing 6,128 bytes. There is no integrity hash.

All **480 measured outputs and 160 warmups** restored every original sample,
including invisible RGB and duplicate native alpha output in the standard
reference. Source hashes, deterministic sizes, medians, group counts, complete
costs and Selected's size bound against its four actual candidates were checked
after the run. Receipts: `pixel-composition.json`,
`pixel-composition-analysis.json` and `pixel-composition-build.json` under
`out/pja-port-20261117/`. Tests passed: 10 entropy-bridge, 6 preparation/inverse,
15 focused release laboratory, and 7 standard wide-integer integration tests.
The focused tests force real Static/Fitted/Adaptive LZ copies on 8/31-bit inputs,
mix group policies, and check limits, unknown flags and old magic. Rust 1.89
checks include tests/examples; changed-source formatting and diff checks pass.

```powershell
$env:RS_JXL_PLUS_PIXEL_MANIFEST = 'D:/cpp_pj/rs_jxl/out/pja-port-20261117/pixel-manifest.json'
# Use a new receipt path; the harness refuses to overwrite an existing run.
$env:RS_JXL_PLUS_PIXEL_COMPOSE_RECEIPT = 'D:/cpp_pj/rs_jxl/out/pja-port-20261117/pixel-composition-rerun.json'
cargo test --release -p rs-jxl-plus --example pixel_lab -- pixel_compose_tests::pixel_composition_corpus --ignored --exact --nocapture --test-threads=1
python -B tools/analyze_jxl_plus_pixel_composition.py --receipt out/pja-port-20261117/pixel-composition-rerun.json --output out/pja-port-20261117/pixel-composition-rerun-analysis.json
```

These results support proceeding with the JXL-derived structure in
[the bitstream draft](JXL_PLUS_BITSTREAM_DRAFT.md). They do not freeze the
research bytes or complete palette/squeeze/extra-plane/multiframe integration.
Before freezing final bytes, implement retained image/frame extension dispatch,
the exact unit/TOC mapping and target reconstruction descriptors, then publish
golden vectors and boundary/incremental tests. Compression choices remain
encoder policy; static, adaptive and LZ alternatives stay in the design.

## Verification checkpoints

The subsequent archive cleanup passed **89 release library tests and 10
integration tests**, including 61 valid upstream originals, five explicit
damaged-input rejections, all source algorithm oracles and the seven real
JPEGs. Old protocol tests were removed with the corresponding implementation;
the unchanged static size experiment was not rerun. The receipt is
`out/pja-port-20261117/cleanup-tests.log`.

The subsequent CfL composition passed its zero-factor control and the focused
release experiment (four reversible combinations across two eligible inputs,
with the other five baseline inputs also verified). Rust 1.89 checking passed
again for the production crate and `jxl-plus` CLI after cleanup; package
format checking passed. These are separate checks, not a rerun of the full
cleanup suite.

The hash-removal change updates version/offset, structure, truncation and
resource-limit checks, while retaining normal original-byte roundtrips.
The central hashless run passed **85 release library tests and 10 integration
tests**, including all 66 upstream originals and seven local JPEGs. Three
historical archive compatibility tests were excluded. The extended static
comparison then passed six targeted tests including all 84 real-input variants;
the exact J2K pixel output-budget regression also passed. Rust 1.89 checking
passed for the production crate and CLI. The following port figures are earlier
historical checkpoints, not additional independent coverage.

The final port checkpoint passed **77 library tests and 10 integration
tests in release**, with three original-archive compatibility tests explicitly
excluded. This includes all 66 pinned upstream files through the unified
internal entry, seven existing local JPEGs, source algorithm comparisons,
malformed/limit cases and scalar/AVX2 equivalence. The independent Python
Decimal verifier checked all 8192 frozen squash/stretch entries. Counts are
not a claim of full-format coverage; see the detailed port status and limits.

The original-C v7 oracle passed 20 generated cases with complete arithmetic
stream equality and cross-decoding in both directions. This establishes the
tested coefficient-model equivalence, subject to the shared prediction-table
boundary below. Rust 1.89 compilation passed for the expanded crate and CLI.
Strict Clippy passed at the earlier v0 checkpoint; it has not been rerun over
the complete expanded port. The existing `rs-jxl` dependency has unrelated
strict-Clippy warnings outside this change.

The historical **version-0** seven-input smoke size comparison uses the saved e9 JXL diagnostic baseline
from `jpeg-entropy-audit-20261117/audit45.json`. All current original SHA256
values match those receipts. Complete file sizes (including the new container,
models and reconstruction data) are:

| Input label | Saved JXL bytes | Historical v0 jxl_plus bytes |
| --- | ---: | ---: |
| baseline-420 | 3,374 | 3,368 |
| progressive-420 | 3,202 | 3,196 |
| restart-bicycles | 75,777 | 66,754 |
| rgb-adobe | 934,879 | 837,025 |
| wallhaven-regression | 92,710 | 87,148 |
| illustration-icc | 934,171 | 792,763 |
| project-wallhaven-yqqvv7 | 1,429,265 | 1,348,631 |
| Total | 3,473,378 | 3,138,885 |

Those historical v0 outputs total 334,493 bytes (9.6302%) below that saved JXL baseline and 29.0753%
below the original JPEG total of 4,425,657 bytes. The sample is seven fixed
smoke inputs, not a representative corpus. The baseline enabled cooperative,
symmetric-MA, CfL-peak and audit diagnostics; this is a size comparison only.
No speed ratio or peak-memory measurement is claimed. The ignored local receipt
is `out/pja-port-20261117/smoke7-size-comparison.json`. This table retains its
original hash-bearing file sizes; version 1 removes 64 framing bytes per file,
but no new size measurement is claimed here.

The development-only original-C oracle is opt-in:

```powershell
python tools/fetch_pja_reference.py
clang -O2 -fwrapv -ffp-contract=off tools/pja_v7_reference.c -o out/pja-port-20261117/pja_v7_reference.exe
$env:RS_JXL_PLUS_PJA_ORACLE = 'D:/cpp_pj/rs_jxl/out/pja-port-20261117/pja_v7_reference.exe'
cargo test --offline -p rs-jxl-plus original_c_v7_equivalence -- --ignored
```

Its source wrapper is `tools/pja_v7_reference.c`, including the pinned,
unmodified source under `out/pja-port-20261117/source/`. It is not part of
runtime encoding. The oracle independently runs source arithmetic and
coefficient decisions; prediction tables are exported from Rust, so that
comparison does not independently validate those prediction constants or
establish full PJA archive conformance. Official `djxl` cannot decode the new
format. Existing standard-JXL timing reports are not measurements of this
backend. The saved byte sizes are used only with source-hash matching above.
