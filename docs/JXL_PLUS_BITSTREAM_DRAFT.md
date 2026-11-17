# JXL Plus bitstream draft

Status: candidate decisions and architecture proposal, 2026-11-17. This document
does not freeze a wire format, allocate an external JPEG XL extension ID, or
describe a standard-conforming JPEG XL extension. `MUST` below states a proposed
Plus requirement, subject to implementation and the freeze gates below.
The implemented JPEG version-1 container and pixel research streams remain
separate experiments. They MUST NOT be accepted under a future final signature
without explicit conversion of their versioned identification and body.

The working decisions for the next integration are:

- Identification tuple: ASCII `JXLPLUS\n`, little-endian u16 major **2**, u16
  minor **0**, then the single JXL-derived body starting with its ordinary
  codestream signature. This 12-byte prelude replaces the experimental v1
  header, not wraps an existing v1 file. Major 1 remains a separate experimental
  decoder entry; `JXLPPX01`/`JXLPPX02` are never aliases for major 2.
- Within Plus major 2 only, image and frame extension selector **bit 0** are
  provisional private assignments for the bounded Plus descriptors. They are
  not reserved/registered JPEG XL IDs. The image descriptor is mandatory;
  a frame without the frame descriptor uses standard layout and semantics.
  Other selector bits are unsupported in the first major-2 implementation.
- Preserve the ordinary Modular section topology for pixel frames. Give source
  reconstruction a separate explicit frame layout with its own unit count and
  TOC mapping; do not alter the one-bit standard frame encoding field.
- First release requires independent adaptive/entropy units: reset mode 0,
  predecessor count 0. Existing explicit JXL frame-reference dependencies
  remain legal. Cross-unit entropy inheritance is reserved and rejected.
- Encoder search is not wire state. Each group identifies the chosen decoder
  profile and parameters; the file does not prescribe an encoder's search,
  effort, thread count or candidate order.

These assignments make the proposal reviewable. They are not published/frozen
wire constants, and the current runtime has not yet implemented this prelude
or retained extension dispatch. Finalization requires that integration.

## Scope and decisions proposed for adoption

The core is one JXL-derived image/frame/group codestream. Keep JXL metadata,
frame geometry, reference/blending machinery, Modular preparation and ordinary
ANS/prefix branches. Add explicit alternative coding and source-reconstruction
branches at the appropriate frame/group boundaries. A coding unit contains one
selected representation, not both a full standard image and another full
compressed image. An ordinary standard branch is decoded by existing JXL
machinery; an extended branch is dispatched by the Plus decoder.

The two primary reconstruction targets are exact native image samples and exact
original JPEG bytes. Pixel equality includes invisible RGB, extra channels,
integer precision and every frame. Reconstructed JPEG equality includes headers,
scans, entropy choices, padding and retained trailing bytes. JPEG pixels are not
a substitute for JPEG bytes. A compressed representation may exceed its source
size and still be valid.

There are no integrity hashes, PJA archive versions, generic repair/patch
programs, recursive archival parts, whole-file opaque fallback, or original-size
admission gates. Invalid or currently unsupported inputs fail explicitly.
Localized original bytes needed to reconstruct a supported coded object remain
legal: marker metadata, unusual valid entropy termination and a local J2K
codeblock exception are different from a whole-file escape route. Lengths and
entropy termination validate structure; they do not authenticate content or
detect every corruption.

All existing tools remain available for composition or future profiles:
Full/Fast/V7 coefficient models, MED/Blend/symbol sample coding, JXL transforms,
trees and static entropy, adaptive probability models, JPEG arithmetic and
lossless paths, JPEG-LS, classic J2K and HT. Their preservation does not reserve a
wire profile for every current implementation combination.

## Source evidence and current implementation boundary

Paths below are repository-relative and describe inspected local code, rather
than an assertion about unreviewed external standard text.

| Source | Relevant existing boundary |
| --- | --- |
| `crates/rs-jxl/src/metadata.rs` | Standard image metadata serialization currently writes zero extensions; bit depth, color and extra-channel information already have their own syntax. |
| `crates/rs-jxl/src/frame.rs`, `codestream.rs` | Existing frame contracts and frame/TOC serialization; the standard path does not emit Plus descriptors. |
| `vendor/jxl-rs/jxl/src/headers/image_metadata.rs::ImageMetadata` | An image-level `extensions: Option<Extensions>` field follows tone mapping. |
| `vendor/jxl-rs/jxl/src/headers/frame_header.rs::FrameHeader` | Frame-level `extensions` follows restoration filtering; encoding is still a one-bit VarDCT/Modular choice. Frame geometry, extra-channel upsampling, animation, reference slots and blending are already represented. |
| `vendor/jxl-rs/jxl/src/headers/encodings.rs::Extensions` | Reads a U64 selector, then U64 bit lengths for selected entries, adds lengths with overflow checks, and skips the total number of bits. It discards extension identity/content. Unknown extensions do not inherently produce an unsupported error here. |
| `vendor/jxl-rs/jxl/src/headers/toc.rs` | TOC carries byte lengths with the existing JXL U32 selector coding, optional permutation and byte-boundary handling. Incremental parsing is separate from group decoding. |
| `vendor/jxl-rs/jxl/src/headers/frame_header.rs::num_toc_entries` | One group/one pass uses one entry; otherwise the count is `2 + num_dc_groups + num_groups * num_passes`. This count cannot simply grow without a declared new layout. |
| `crates/rs-jxl/src/modular/plus_bridge.rs` | Experimental bridge exposes actual transformed planes, tree, weighted predictor, tokens and plane layouts. It currently rejects palette/squeeze/LZ/refinement configurations; this is an implementation restriction, not a target-format prohibition. |
| `crates/rs-jxl-plus/src/pixel_model.rs` | Research `JXLPPX02` stream, 256-square tiles, static/fitted/adaptive zero gates and optional residual LZ; per-group decoder choice with encoder selection kept off wire. Historical `JXLPPX01` is rejected; neither is integrated JXL frame syntax. |
| `crates/rs-jxl-plus/src/container.rs` | Public hashless JPEG experiment uses `JXLPLUS\n`, version 1, geometry, reconstruction metadata and independent model bands. It is not the architecture proposed here. |
| `crates/rs-jxl-plus/src/jpeg_native.rs`, `jls.rs`, `j2k/` | Expanded native source-codec reconstruction tools; their internal records are not a final wire specification. |

The translated standard encoder follows libjxl commit
`b866b3defec13494bfbaf705db93581cda118e37`; the PJA-derived tools follow
`1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd`. Existing attribution remains in
the corresponding source/provenance notices. Reusing code does not imply
compatibility with either PJA archives or all standard-JXL decoders.

## Identification and insertion points

**Proposed initial transport:** use a distinct, short Plus signature followed by
a layout major/minor and a single JXL-derived codestream body. The working
identification tuple is specified above. This is format identification, not an archive containing a
second image. The signature MUST be recognized before calling the standard JXL
parser. Standard JXL signatures still select the unchanged standard decoder.
This avoids depending on a standard decoder to reject a skipped extension.

Inside the Plus body, preserve the existing JXL image-header structure. Attach
a bounded Plus image descriptor at the image extension boundary, and a Plus
frame descriptor at the frame-header extension boundary, before the TOC. The
provisional selector position is bit 0 in each respective Plus-major-2 scope.
Until implemented, notation such as `PlusImageExtension` denotes a field in this
draft, not a legal standard reserved bit. A Plus-mode reader must retain and
parse these records rather than call the current skip-only `Extensions` reader.
No change to the standard reader's treatment of ordinary input is implied.

The image descriptor declares the required feature/profile set early. The frame
descriptor selects a standard section layout or a versioned Plus section layout
and maps logical units to TOC entries. Payloads with the standard coding profile
MUST be handed to their existing parser unchanged. An extended coding profile
MUST be dispatched by Plus even when it retains ordinary Modular TOC topology.
Dispatch precedes the relevant LF/HF/Modular entropy parsing. Do not overload the one-bit
VarDCT/Modular field with an invented third standard value.

For pixel Modular extensions, prefer retaining the existing frame section
topology and replacing only declared group's entropy/preparation records.
If a source-reconstruction frame needs a different topology, its explicit
layout profile defines the entry count; it MUST NOT pretend to follow the
standard `num_toc_entries` formula. A frame cannot mix these interpretations.
An extended reconstruction frame is an explicitly typed Plus frame operation,
not a disguised standard VarDCT frame with arbitrary coefficient precision.

An unknown required image feature, layout major, frame layout, profile or
transform MUST fail before allocating its model/output. Unknown optional
annotations may be skipped only when their schema says they cannot change
samples, original-byte reconstruction, lengths, dependencies or scheduling.
An ordinary JXL decoder is not promised a preview or partial rendering of an
extension-dependent file. Do not claim that a private extension selector makes
the result standard-compatible.

## Minimal structural syntax proposed

This is a concrete candidate encoding for **Plus records**, independent of the
existing JXL bit encodings. Review must decide it before vectors are frozen.
Record boundaries are byte-aligned; required alignment bits are zero. Plus
integers use canonical unsigned base-128 little-endian varints: at most ten
bytes for u64, no redundant final zero group, no overflow. Signed values use
zigzag mapped integers with a profile-defined width. Existing JXL fields keep
their existing encodings and alignment rules.

| Record | Fields in order |
| --- | --- |
| Identification | eight-byte `JXLPLUS\n`; u16 LE major 2; u16 LE minor 0; then the JXL-derived body |
| Image descriptor | required feature count and sorted unique feature IDs; target capability declarations; optional annotation record count and bounded records |
| Frame descriptor | descriptor schema version; reconstruction target; section-layout profile; unit count; unit descriptor table; target-specific reconstruction descriptor length and bytes |
| Unit descriptor | logical unit ID; domain; preparation profile and parameter length/bytes; entropy profile and parameter length/bytes; reset mode; predecessor count and IDs; bounded decoded item count; substream count and lengths |

The image descriptor at the JXL extension boundary is the authoritative feature
declaration; the outer identification carries no duplicate descriptor. Its
existing extension bit length bounds the descriptor. Because Plus records are
byte-aligned, specify and validate zero alignment bits within that extension
budget. Parse the image header and its bounded descriptor before allocating
image planes or entropy models. Header parsing itself remains subject to
metadata/count limits. Reject duplicate semantic declarations.

IDs in these tables are symbolic until the registry is assigned. Descriptor
parameters are normative decoding data, not an encoder search log. Unit count,
IDs and decoded shapes MUST reconcile with frame/channel geometry or the source
codec's block geometry. Empty units are explicit; an absent unit is not an
implicit all-zero channel. No arbitrary original-file chunk kind is defined.

For each TOC entry, its byte length bounds the complete selected payload,
including substream endings and local descriptors. A unit's substream lengths
MUST sum exactly to its remaining entry length. Reuse the existing TOC length
coding/permutation when its representable range and entry-count rules apply.
Reject entries beyond that range; do not silently add an incompatible wide TOC.
All additions, products, offsets, geometry derivations and u64-to-usize
conversions are checked before slicing or allocation. Every entry is consumed
exactly, except explicitly specified terminal zero alignment bits. Trailing
bytes are accepted only as a declared source-reconstruction field with a bounded
length, never as ignored container garbage.

Major versions change interpretation; unsupported majors fail. Minor versions
may add optional annotations but cannot silently change existing profile
semantics. Required new tools use a new required feature/profile. Known numeric
profiles never change initialization, tables, rounding or stream termination.
There is no ambient environment-variable configuration on the decoder.

## Profile registry to specify before assignment

The following names are proposed registry entries, **not allocated wire IDs or
claims that a finished profile already exists**.

| Profile family | Domain and contract |
| --- | --- |
| `JxlStandard` | Existing standard section syntax and entropy, unchanged; all its required global information remains present. |
| `ModularStatic` | JXL reversible preparation, MA tree/WP and HybridUint with explicitly transmitted static ANS/prefix model. Fitted probability table cost is part of the payload. |
| `ModularZeroGateAdaptive` | Same sample/predictor preparation, with a causal adaptive zero decision and static residual entropy. Exact counter recurrence, zero mapping, MA-leaf context, prior and event order must be fixed. Current pixel experiment is input to this specification, not automatically its wire schema. |
| `JpegDctV7` | Quantized coefficient domain, frozen PJA-derived integer model rules, luma/chroma sharing and explicit band geometry. Default encoder policy retains the source band rule; alternate legal geometry is transmitted, never inferred from worker count. |
| `JpegDctFull`, `JpegDctFast` | Separate versioned coefficient model contracts; independent of the JPEG source's Huffman/arithmetic entropy method. |
| `SampleMed`, `SampleBlend`, `SampleSymbols` | Exact sample-domain models with sign/precision, causal neighbors and boundary rules fixed; may serve lossless JPEG/JPEG-LS reconstruction when geometry and coded auxiliaries are declared. |
| `J2kClassicDecision`, `J2kClassicPixels`, `J2kHtDecision`, `J2kHtPixels` | Source codeblock/pass or sample-domain tools with explicit packet/reconstruction data. Preserve local exceptional codeblock bytes only where required for exact source reconstruction. |

A profile specification must include valid precision/channel ranges, transform
order and inverse, arithmetic widths and wrapping/saturation, signed division
rounding, immutable integer tables, event order, initial probabilities/weights,
learning rates, context indexing, permissible parameters, termination and a
maximum-state formula. Predictor trees may select contexts and/or predictors;
adaptive context updates must still occur identically on both sides. Transmitted
static models cannot be replaced by a uniform prior and called fitted.

Independent units reset all entropy, mixer, predictor, boundary and match-history
state specified by their profile. Initial release MUST allow only independent
units; channel-to-channel causal use inside a unit is explicit in its profile.
Later inheritance requires a declared predecessor of the same compatible state
type. Predecessor IDs must be earlier logical IDs and the graph must be acyclic;
TOC physical permutation does not alter event order. Neither completion order
nor thread count may change encoded state. No state leaks across frames without
a separately specified dependency. Decoder reference-frame dependencies remain
the existing JXL frame contract, not implicit entropy inheritance.

Each entropy substream has its own exact end condition. Current adaptive coder's
four-byte canonical finalizer and `finish_check` are a candidate contract that
must be copied into the selected profile: exact byte consumption and terminal
code/low relation, not unlimited synthesized zeros. ANS/prefix ends keep their
specified state/bit checks. A truncated live stream waits for input until EOF,
then fails if the unit's specified ending cannot be met. Cancellation and
resource failure propagate rather than leave dependent jobs waiting.

### Candidate zero-gate numerical contract

The preparation, MA tree, offset/multiplier and weighted prediction retain the
chosen Modular profile's exact integer rules. For each prediction residual,
`u` is JXL's packed signed integer after offset/multiplier. Bit `z=1` means
`u=0`; bit `z=0` means one nonzero `u` follows in the residual entropy stream.
The nonzero mapping does not subtract one. A decoded zero from that nonzero
stream is invalid. All channels within one unit share gate state. No state
crosses a unit boundary. Every sample updates the gate, including samples whose
nonzero value is supplied by an expanded LZ copy.

Gate context is `2 * MA_leaf + previous_zero[MA_leaf]`, with previous_zero false
at reset. Every adaptive context starts with `p=32768, n=0`. For the coded bit
`z`, let `target=65535` for 1 and `target=1` for 0. The integer update is:

```
d = 2*n + 3
reciprocal = floor((131072 + floor(d/2)) / d)
p = p + trunc_toward_zero((target - p) * reciprocal / 65536)
n = min(n + 1, 255)
previous_zero[MA_leaf] = z
```

The product is signed i64; p is stored as u16 and n as u8. The binary arithmetic
state starts at low=0, high=2^32-1. Its split is
`low + floor((high-low)*clamp(p,1,65535)/65536)` using a u64 product. A 1 selects
the lower interval including the split, a 0 selects the upper interval after
the split. A 1 sets high=split; a 0 sets low=split+1. While low and high have
the same top byte, emit high>>24 and set low=low<<8,
high=(high<<8)|255, using wrapping u32 shifts. The decoder initializes code
from four bytes in big-endian order; at each identical renormalization step
it also sets code=(code<<8)|next_byte with wrapping u32 arithmetic.
The encoder terminates with four successive high bytes of low. The decoder
requires exact byte consumption and code==low, without reading beyond the
declared segment. This is structural termination, not a checksum.

The fitted-gate alternative transmits exactly two u16 probabilities per MA
leaf in context order, each in 1..65535, and never updates them. Encoder fitting
is not normative; the experiment rounds observed zero frequency to 16 bits and
uses 32768 for unused contexts. Gate presence is orthogonal to static ANS/prefix
choice and residual LZ. With the gate disabled, LZ history covers all residuals
and the profile provides the usual image-width distance multiplier. With the
gate enabled, LZ history covers only nonzero residuals and the distance
multiplier is zero (direct distance-minus-one). Expanded counts are declared,
not inferred from compressed token counts; history and copy bounds are checked.

This fixes the candidate model math, but profile IDs and final descriptor
grammar remain unassigned until the real frame/group reader and writer pass
the integration gates. Larger context/mixer tools require distinct profiles.

## Reconstruction targets and source information

Targets and entropy tools are separate axes. Arithmetic JPEG, 12-bit JPEG and
lossless JPEG are **source reconstruction kinds**, not new standard JXL image
encoding enum values. JPEG-LS and J2K likewise describe source syntax to restore;
their sample models may also be usable for pixel units. A future wire target
registry may group JPEG variants under one typed JPEG descriptor, but it must
retain their explicit marker/precision/scan semantics.

| Target | Required information beyond coded samples/coefficients |
| --- | --- |
| Native pixels | JXL image and extra-channel metadata, bit depth/type, plane geometry, frame timing/crops/blends/references, reversible preparation and any required metadata. No PNG original-byte claim. |
| JPEG DCT, Huffman or QM arithmetic, 8/12-bit | Component IDs/sampling, quantizers and coefficient order, frame/scan parameters, Huffman or arithmetic conditioning tables, restart behavior, marker ordering and original marker bytes/metadata, padding/termination/refinement auxiliaries where source coding choices are not implied by coefficients. |
| Lossless JPEG | Precision, point transform, predictor/scan/restart definitions, padded and visible sample geometry and exact entropy reconstruction auxiliaries. Pixel conversion alone loses source-byte choices. |
| JPEG-LS | NEAR/interleave, MAXVAL/threshold/reset parameters, component mapping, scan/restart structure and source run/termination choices required by the writer. A source's lossy NEAR does not weaken byte equality. |
| J2K/HT and host formats | Main/tile headers, geometry, progression/packet/tagtree/pass layout, segment ending choices and required local exceptions. JP2/JPX/MJ2 host metadata, codestream identities and relationships stay explicit; do not flatten multiple codestreams into unrelated images. |

For valid inputs, source reconstruction metadata is declarative and bounded; it
is not a patch program applied to a repaired surrogate. First-version support
may reject damaged inputs, including former repair-dependent JPEG fixtures.
The supported-input matrix MUST name such restrictions. Structural source data
whose algorithm has not yet been specified cannot be hidden in a whole-file raw
kind. The encoder returns Unsupported instead. Model failure and resource Limit
must remain distinguishable from a valid but less compact result.

JPEG sample display may need inverse DCT/color handling beyond byte restoration.
The first byte-reconstruction profile must state whether it offers display;
it MUST NOT manufacture standard rendered-frame semantics from quantized data.
Similarly, source-host reconstruction order is separate from JXL animation
display order. MPO/embedded images need typed relationships if recompressed;
retaining valid marker metadata does not authorize recursive archive syntax.

## Resource and native-image requirements

Stream declarations describe dimensions, counts and exact decoded cardinalities;
implementation ceilings remain caller policy. Validate profile recognition,
descriptor sizes, channel/frame counts, geometry, tree/table limits, decoded
sample/coefficient bytes, output length and work bounds before large allocation.
Include model state, predecessor retention, metadata, transform intermediates
and concurrent groups in peak-live accounting. No fixed experiment tile size,
slice count, expansion-ratio heuristic or source-size floor becomes a format
requirement by accident. Geometry-dependent upper bounds remain necessary.

Extra-channel type, bit depth, dimensional shift, alpha association, color
metadata and frame references MUST survive ordinary and extended branches.
Unassociated invisible RGB is never discarded. Native 17..31-bit integers MUST
remain integer-valued through entropy, inversion and output; f32 storage is not
an exact interchange. A profile that cannot handle a requested representation
must refuse it, allowing another declared coding profile where available.
Multiframe uses actual frame metadata, not a concatenation of standalone Plus
files. Random access requires the declared frame and unit predecessor closure;
seeking to an arbitrary dependent group cannot imply reset.

## Freeze gates and open decisions

The following work is required before calling this a frozen format:

- Implement and validate the working identification/selector assignments above.
  Verify no ambiguous identification with JPEG v1 or
  pixel-experiment signatures. No selector is claimed standard-reserved/legal.
- Implement retained image/frame extension parsing and writing in Plus mode;
  verify standard input/output paths remain unchanged. Demonstrate early failure
  for unknown required profiles even though the existing reader skips them.
- Specify the first frame topology and logical-unit/TOC mapping, including the
  single-entry case, LF/global records, permutations and empty groups. Implement
  the working decision of a separate explicit source-reconstruction frame layout.
- Assign profile IDs only after complete parameter grammars and state/memory
  formulas exist. Publish scalar numerical vectors, tables, event order and
  canonical termination vectors for every released adaptive profile.
- Integrate Modular preparation and static/adaptive entropy into the actual
  frame/group path. Test trees/WP, RCT and supported transforms; list palette,
  squeeze, LZ and refinement gaps rather than silently claiming them supported.
- Integrate native wide/extra-channel/multiframe decoding through Plus dispatch;
  test invisible RGB, gray/alpha, differing extra-channel shifts, 16/17/31-bit
  extrema, frame crops/blends/references and incremental input at every boundary.
- Specify and wire exact JPEG reconstruction descriptors for each accepted
  precision/entropy family. Test baseline/progressive, restarts, arithmetic,
  high precision and lossless separately; reject repair-dependent damaged inputs.
  Document JLS/J2K/HT profile readiness separately from translated module coverage.
- Test descriptor/TOC count disagreement, overflowing lengths, unknown IDs,
  duplicate parameters, cycles, trailing bytes, truncated finalizers, exact
  resource limits, allocation failure and worker cancellation. No hash is needed
  to test these structural failures.
- Test bit-identical entropy across scalar/SIMD and scheduler choices where the
  profile promises deterministic output; exact decoded samples/source bytes on
  all routes. Existing source-C oracles remain algorithm checks, not archive
  compatibility requirements.
- Record complete encoded costs on the fixed corpus, including descriptor,
  trees/tables, framing, endings and metadata. Separate fixed-profile aggregate
  results from per-file selection and prior-used from absent-prior-record data.
  Compression results do not replace missing-format-feature tests.

The transport identification, independent Modular units, separate source
reconstruction layout and initial prohibition on cross-unit adaptive inheritance
are working decisions above. Review now verifies their implementation and
decides which fully specified profiles ship first. Keeping the remaining tools in the
implementation is independent of exposing unfinished wire profiles. Final bytes,
IDs and public APIs remain unfrozen until these integration gates pass.
