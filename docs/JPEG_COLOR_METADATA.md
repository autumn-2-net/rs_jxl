# JPEG Color Metadata

`jpeg/color_metadata.rs` provides the public ICC-only adapter:

```rust,ignore
pub fn color_encoding(
    jpeg: &JpegData,
    max_profile_bytes: usize,
) -> crate::Result<ColorEncoding>;
```

The module is centrally registered. The new crate-private composer hook is:

```rust,ignore
pub(crate) fn prepare_metadata(
    jpeg: &JpegData,
    max_profile_bytes: usize,
    initial_orientation: Orientation,
    keep_exif: bool,
    max_exif_box_bytes: usize,
) -> crate::Result<PreparedMetadata>;
```

`PreparedMetadata` contains `color_encoding`, `color_transform`, `orientation`,
and `exif_box_payload: Option<Vec<u8>>`. Source defaults are
`initial_orientation = Orientation::Identity` and `keep_exif = true`.
The preparer and its local `#[path = "exif.rs"]` parser are available to the
production composer; only the unit-test modules retain `cfg(test)` gates.
The existing public ICC adapter is unchanged.

An additive crate-private XMP hook leaves that preparer's signature and fields
unchanged:

```rust,ignore
pub(crate) fn xmp_box_payload(
    jpeg: &JpegData,
    keep_xmp: bool,
    max_xmp_box_bytes: usize,
) -> crate::Result<Option<Vec<u8>>>;
```

Call it once with source-default `keep_xmp = true` and an explicit byte limit;
pass its returned bytes unchanged as the payload of an `xml ` box (trailing space
in the four-byte type). No four-byte offset or JPEG signature is included. ICC,
EXIF and XMP budgets are separate, and the container must also enforce its total
output budget. This hook shares the ICC adapter's APP envelope/order/mirror
validation, but does not reassemble or interpret ICC/EXIF a second time. Even when
`keep_xmp=false`, incoherent archival mirrors are rejected; no payload allocation
or byte-budget check is then needed. It never reclassifies APP markers.

Parent creates `ImageMetadata` with the returned color encoding and assigns
`metadata.orientation = prepared.orientation`. The returned coding transform
must agree with `TranscodeLayout::color_transform()`; both use the same helper.
Pass any returned Exif payload to the container as an `Exif` box (or compressed
`brob` carrying `Exif`), including it in the cumulative output budget. The returned
bytes are payload only, not box framing. No composer, `mod.rs`, Cargo, container,
coefficient writer, or CLI file is modified here.

The reconstruction serializer now also has an opt-in deduplication hook:

```rust,ignore
let encoded = serialize_reconstruction_metadata_with_sources(
    jpeg,
    max_reconstruction_bytes,
    ReconstructionMetadataSources {
        icc_profile: metadata.color_encoding.icc_profile(),
        exif_box_payload: emitted_exif_payload.as_deref(),
        xmp_box_payload: emitted_xmp_payload.as_deref(),
    },
)?;
```

Both the new function and `ReconstructionMetadataSources<'a>` are public in
`jpeg::reconstruction_metadata`. The returned `EncodedMetadata { header, payload }`
is unchanged. The existing two-argument `serialize_reconstruction_metadata`
passes empty sources and preserves its all-Unknown behavior and wire output.
Sources borrow actual emitted bytes, not JPEG metadata inferred in isolation.
The parent must keep the corresponding ICC/profile and Exif/xml boxes in every
cached candidate output that reuses this reconstruction metadata. Compression of
boxes may differ, but their decompressed payloads must match exactly. Output
budgets must include both the jbrd and those external sources.

## Policy And Sources

The immediate implementation reference is the existing read-only
`crates/rs-jxl-cli/src/input/jpeg.rs::preflight` ICC path. Frozen libjxl reference
commit is `b866b3defec13494bfbaf705db93581cda118e37`:

- `lib/jxl/jpeg/enc_jpeg_data.cc::ParseChunkedMarker` and
  `SetColorEncodingFromJpegData`: multipart extraction and absent-profile sRGB.
- `lib/extras/dec/jpg.cc::ReadICCProfile`: numbered chunks, count consistency,
  duplicate/empty/missing rejection, ordered assembly.
- `lib/jxl/jpeg/enc_jpeg_data.cc:242`: `SetColorTransformFromJpegData`.
- `lib/jxl/jpeg/enc_jpeg_data.cc:394`: `SetBlobsFromJpegData`.
- `lib/jxl/jpeg/enc_jpeg_data.cc:44-118,287-380`: `DetectIccProfile`,
  `DetectBlobs`, and `EncodeJPEGData` classification and payload omission.
- `lib/jxl/jpeg/jpeg_data.cc:115-133,476-507`: APP type/length wire fields and
  `SetJPEGDataFromICC`; `jpeg/dec_jpeg_data.cc:71-118` reconstructs APP headers.
- `lib/jxl/base/exif.h:25-87`: `IsExif`, `FindExifTagPosition`, `InterpretExif`.
- `lib/jxl/encode.cc:2421-2505`: `JxlEncoderAddJPEGFrame` profile, orientation,
  Exif box, and reconstruction-retention policy.
- `lib/jxl/enc_frame.cc:385-393,800-1087`: JPEG coding transform and coefficients.
- `lib/jxl/enc_params.h:105-108`: compressed boxes and keep-EXIF default true.
- `lib/jxl/decode_to_jpeg.cc:91-99,143-163` and `decode.cc:1762-1770,2089-2100`:
  Exif reconstruction uses specially classified APP markers, not Unknown APPs.

This deliberately follows the current Rust CLI's strict policy, not the frozen
transcode caller's warning-and-sRGB fallback for malformed ICC. A malformed or
mismatched profile is an error; only an absent profile defaults to sRGB.
The existing `icc::validate_profile` and `ColorEncoding::icc` are reused; no
new ICC parser, CMS, profile normalization or color conversion is introduced.
Validation is structural, not full ICC/CMS conformance validation.

### ICC And Coding Transform Audit

The frozen JPEG transcode ICC caller uses `ParseChunkedMarker` with
`allow_permutations=false`; malformed/out-of-order chunks are cleared and fall
back to sRGB. A nonempty profile goes through `SetICC` with the default CMS.
`JxlEncoderAddJPEGFrame` only derives color encoding when the caller has not
already set it. This Rust adapter intentionally retains its existing strict CLI
contract: out-of-order complete chunks are reassembled, malformed/mismatched
profiles are errors, and original profile bytes remain explicit ICC without CMS
field inference. The new preparer is for deriving metadata from JPEG, not for
overwriting caller-supplied profile overrides. This is a documented policy
difference, not a claim of identical ICC decisions to the frozen transcode API.

The coefficient transform is independent of ICC and EXIF. Any APP0 means YCbCr;
otherwise the first exact Adobe APP14 payload decides (transform byte zero means
RGB/no transform), otherwise exact `R,G,B` component IDs imply RGB/no transform.
Grayscale always uses the source YCbCr coding-transform value. The preparer calls
the existing `transcode_layout::color_transform_from_jpeg`; it does not duplicate
that algorithm. A three-component YCbCr-coded JPEG still has RGB display profile
metadata. No CMS or EXIF tag changes the stored coefficient color transform.

### Orientation And Exif Audit

The source **does interpret EXIF into JXL display orientation**, but **does not
rotate, flip or transpose the JPEG image/coefficient grid for that orientation**.
Coding dimensions stay `jpeg.width`/`jpeg.height`; EXIF orientations 5 through 8
do not swap coding dimensions or sampling factors. DCT transposition used by JXL
coefficient layout is unrelated to display orientation.

`SetBlobsFromJpegData` considers exact APP1 `Exif\0\0` signatures and chooses the
first **nonempty** TIFF blob. A signature-only empty blob does not block later
selection; a nonempty malformed first blob does block later valid blobs. All
APPs are considered, including after SOS, unlike the current strict ICC gate.
Malformed APP envelopes cannot come from the validated reader; this adapter
rejects inconsistent public records instead of the source blob helper's local
skip-on-invalid-envelope behavior. Malformed TIFF itself is not an error.

The new parser follows the small frozen `base/exif.h` implementation exactly:

- Accept `II 2a 00` or `MM 00 2a`, with at least 12 TIFF bytes.
- Read the first IFD offset, requiring offset >= 8 and length >= offset + 14.
- Scan at most the declared u16 entry count, also bounded by the available bytes.
  An entry whose 12 bytes end exactly at the blob boundary is rejected, preserving
  the source's strict comparison. One trailing byte is sufficient; a complete
  next-IFD pointer is not required once a matching tag is found.
- Use only the first matching orientation tag 274 in IFD0, with type SHORT=3,
  count=1 and inline value 1 through 8. An invalid first matching tag blocks later
  duplicates. Next-IFD, sub-IFD and out-of-line value pointers are not followed.
- Invalid headers, offsets, truncation, or tags leave the supplied initial
  orientation unchanged. All indexing/addition is checked and no allocation or
  unsafe code is used by the parser. Checked bounds replace source pointer math.

`JxlEncoderAddJPEGFrame` interprets EXIF **before** checking `jpeg_keep_exif`.
Thus `keep_exif=false` still updates display orientation. With retention enabled,
the source adds an `Exif` box whose payload is four zero TIFF-offset bytes followed
by the original selected TIFF bytes, without JPEG's six-byte signature. Neither
the TIFF orientation tag nor other TIFF bytes are normalized or rewritten.
The preparer returns precisely this payload, including malformed nonempty TIFF.
`max_exif_box_bytes` bounds the whole payload, including the four-byte prefix,
before allocation. It is unused when no box is emitted. Valid APP framing limits
TIFF to 65527 bytes and the returned payload to 65531 bytes.

### XMP And JUMBF Audit

The same frozen `SetBlobsFromJpegData` also selects the first nonempty APP1 XMP
payload after the exact 29-byte `http://ns.adobe.com/xap/1.0/\0` signature. Empty
payloads do not block a later selection; other signatures and extended-XMP APPs
are not assembled. Selection includes APPs after SOS and performs no XML parsing
or rewriting. `JxlEncoderAddJPEGFrame` emits these original bytes as an `xml `
box when `jpeg_keep_xmp` is true (the default). This is implemented by the owned
`xmp_box_payload` adapter, without changing the existing
five-argument `prepare_metadata` contract. The selected bytes are copied once
after the `max_xmp_box_bytes` check and a fallible allocation; absent/empty XMP
permits a zero budget. JPEG's APP size bounds this payload to 65504 bytes. The
limit excludes the 29-byte signature and ignored duplicate blobs. Malformed XML
and arbitrary binary bytes are retained just as in the source; no XML dependency
is added. Source helper invalid-envelope skipping is intentionally replaced with
the adapter's strict validation of reader/public-record correspondence.

Although `Blobs` has a `jumbf` member and `JxlEncoderAddJPEGFrame` has a conditional
`jumb` box branch, this frozen JPEG route never populates that member.
`SetBlobsFromJpegData` extracts only EXIF and XMP. There is no APP11/JUMBF assembly
to port from this route. The separate extras color-hints JUMBF input does not
change that fact. Unknown APP11 records remain archival bytes, and accepting
`jumb` in the container API alone does not justify inventing JPEG extraction.

### Archival Deduplication

The default serializer still stores every APP as Unknown. The opt-in serializer
derives private wire classifications without changing any `JpegData` record,
coefficient, or existing `app_marker_type` annotation. It writes the source APP
type distribution (Unknown=0, ICC=1, Exif=2, XMP=3) and original marker lengths,
then omits only classified markers from the raw payload fed to Brotli. Unknown
APPs, COM, inter-marker/fill and tail data keep their existing payload order.
APP11/JUMBF stays Unknown; no JUMBF input or extraction is invented.

ICC selection follows `DetectIccProfile`, not the stricter color adapter above:

- Only exact APP2 `ICC_PROFILE\0` plus sequence/count bytes is eligible.
- Encounter order must produce ordinal 1, then 2, etc., with the source's common
  total count. Out-of-order, duplicate and mismatched-count candidates are skipped
  and remain Unknown. A later matching candidate may complete the selection.
- The selected count must equal its declared total, and its concatenated chunk
  bytes must exactly equal the supplied nonempty output ICC profile. Chunks are
  compared as borrowed slices without assembling another profile. Their original
  sizes are retained, including empty chunks inside an otherwise matching profile.
- If the selection is incomplete or the profile differs in length/content, all
  selected ICC candidates fall back to Unknown atomically. Unlike the source
  helper's incomplete-selection error, no archival bytes are lost or normalized.
- A zero declared total always prevents deduplication. The source's zero-valued
  count sentinel can accept a zero-total first chunk followed by a nonzero-total
  chunk; decoder regeneration would change the first marker's total byte. This
  explicitly guarded corner case therefore falls back to Unknown.
- No supplied ICC, an empty supplied ICC, or an implicit color encoding means no
  ICC deduplication. A caller must supply the actual emitted explicit profile,
  not a synthesized profile for structured sRGB fields.

Exif uses the first exact APP1 `Exif\0\0` with a nonempty TIFF remainder. Only an
emitted Exif payload consisting of four zero bytes plus that exact TIFF qualifies.
Malformed TIFF can qualify because it is copied, not interpreted or rewritten by
this serializer. A mismatch at the first candidate does not allow selecting a
later duplicate. Empty EXIF signatures are ignored, exactly as `DetectBlobs`.

XMP uses the first exact APP1 XMP signature, **including an empty remainder**.
This differs from `SetBlobsFromJpegData`, which selects the first nonempty blob for
display. If the first XMP is empty but a later nonempty blob is emitted, no XMP
marker is deduplicated. An explicitly emitted empty xml payload can match the
empty first marker. Other duplicates remain Unknown. No XML parsing, extension
assembly, line-ending conversion or normalization occurs.

All detectors inspect actual APP records, including those after SOS; they never
search entropy, inter-marker or tail bytes for signatures. Full existing record,
APP length, marker-order, scan and table validation remains mandatory. Selected
markers cannot bypass malformed-envelope or inconsistent-mirror rejection.
Classification allocates at most one enum per source-bounded APP marker (16384
maximum) and uses no coefficient or entropy copies. `max_bytes` remains the
combined header plus retained raw payload budget; matching external bytes are
not charged a second time here. Allocation failures remain checked errors.

The source requires keep-EXIF/keep-XMP for JPEG reconstruction because it omits
their payloads. Here absent output boxes safely disable the corresponding
deduplication. The frozen decoder reconstructs classified ICC bytes from the
profile and classified Exif/XMP bytes from their boxes; Unknown APPs continue to
come entirely from jbrd. Do not change an emitted source after preparing/caching
jbrd. JPEG byte reconstruction and oriented display remain separate checks.

## Data Contracts

- One component defaults to sRGB gray; three default to sRGB RGB. Other component
  counts are unsupported. Embedded profiles must be exactly GRAY or RGB to match;
  CMYK is rejected even though general `ColorEncoding` supports it with Black.
- Archival `JpegRecord::Marker` records are authoritative and must agree with
  `marker_order` and the corresponding `app_data` bytes. APP segment lengths are
  checked. Missing records with leftover saved metadata are rejected rather than
  being treated as an untagged image. A metadata-only fixture may omit unrelated
  frame/scan records; this helper is not a complete JPEG syntax validator.
- Raw `marker_order` includes exactly `JpegRecord::Marker` codes, including
  SOI/EOI. `reader.rs::Reader::marker` appends it; `next_marker` emits Fill and
  InterMarker records without appending codes, and trailing bytes form a Tail.
  These records do not advance the ICC adapter's marker-order iterator. Synthetic
  `0xff` codes belong only to `reconstruction_metadata.rs::WireModel::intermarker`,
  which groups fill/inter-marker data in a separate jbrd wire map. They must not
  be injected into raw `JpegData::marker_order`.
- Only APP2 payloads beginning with `ICC_PROFILE` are candidates. Candidates
  require the complete NUL-terminated signature, sequence/count bytes and a
  nonempty chunk. Other APP signatures remain opaque to the ICC-only adapter;
  the preparer interprets only the selected EXIF orientation described above, and
  the independent XMP hook returns the first nonempty matching APP1 blob.
- Chunks may appear in any order, interspersed with other records. Sequence IDs
  must be 1 through the shared nonzero count, with no duplicates or omissions.
  ICC after SOS, scan entropy or EOI is rejected, following the CLI's header-only
  ICC policy. Non-ICC metadata is not stripped.
- A fixed 255-slot array borrows chunk slices. Checked cumulative payload size
  must not exceed `max_profile_bytes` before allocating the joined profile.
  The limit excludes APP framing and fixed bookkeeping. Allocation failure
  returns `Error::Limit`; zero budget still permits an absent profile.
- Profile bytes are copied once into the returned owned `ColorEncoding` without
  changing their order or contents. No coefficient or image buffer is cloned.
- `JpegData`, archival records and `app_marker_type` are never modified. The
  default serializer remains all-Unknown. Only the explicit-source serializer
  may classify its private wire model as described above; no canonical rechunking
  or original-record mutation occurs.

## Tests And Boundaries

The original ten focused unit tests cover all six three-part permutations for gray/RGB,
default encodings, malformed signatures and numbering, duplicate/inconsistent/
missing/empty chunks, malformed ICC structure, strict color-space matching,
total byte limits, late ICC, inconsistent archival mirrors, all 255 chunks in
reverse order, and unchanged raw reconstruction bytes/Unknown APP types. A
reader-backed baseline fixture includes Fill, InterMarker, unknown APP15/APP2,
COM, late non-ICC APP2, and an opaque tail containing apparent ICC/SOS markers.
It checks real-marker alignment, byte-exact archival reconstruction, and rejection
of a synthetic jbrd `0xff` injected into raw marker_order. Tail bytes are never
scanned for metadata; reserved/unsupported real marker codes remain the JPEG
reader's validation responsibility.

Six local parser tests cover both endiannesses and all eight orientations, shifted
IFDs, every truncated prefix, exact entry boundaries, invalid/overflowing offsets,
invalid types/counts/values, first-match duplicate behavior, oversized tag counts,
ignored next/sub-IFD pointers and deterministic bounded malformed inputs. Five
composer-helper tests cover unchanged raw records/coefficient grids and ICC bytes,
display-only orientation, retained TIFF bytes, EXIF after SOS, initial orientation,
keep-EXIF independence, APP signature variants, first-nonempty selection, maximum
APP-size payloads and cumulative Exif payload bounds, and independent source
color-transform precedence. Five additional XMP tests cover exact/truncated/wrong
signatures and wrong APP types, extended-XMP exclusion, empty/duplicate/non-XML
first-blob behavior, 65504-byte payload and retention limits, late APP1 selection,
opaque APP11/Fill/InterMarker/Entropy/Tail bytes, invalid archival mirrors even
when disabled or after selection, and unchanged reconstruction/Unknown APPs.
Total in these two files: 26 tests.

Nine new reconstruction-metadata tests cover matching/absent/mismatched emitted
sources, unchanged default output, all source-presence combinations, all six ICC
three-part permutations, selected versus skipped duplicate/count candidates,
incomplete and zero-count fallback, empty chunks and 255 chunks, first-candidate
Exif/XMP behavior, empty-XMP/display-selection mismatch, maximum APP sizes and
exact retained header/payload budgets, malformed mirrors and ignored annotations.
An independent decoder-side helper rebuilds APP signatures, lengths, ICC sequence
and count bytes, and payloads from the parsed wire fields plus supplied sources,
then checks original archival bytes. These tests also preserve COM/fill/inter-
marker/tail payload ordering and verify no dependence on stored entropy bytes.

No Cargo builds/tests are run in this slice. Parent performs registration and
central compilation, tests and corpus/oracle checks. Scoped rustfmt is permitted.

The parent has wired the returned orientation into `ImageMetadata` and the
optional Exif payload through `container::wrap_with_metadata`, whose metadata
box allowlist is `Exif`/`xml `/`jumb`. Parent reports 696 complete-JPEG and 44 CLI
checks passed before this deduplication slice. Those results do not establish
the new serializer's compatibility. Parent owns wiring its prepared-metadata
cache to the new explicit-source hook, central tests, and byte-exact JPEG
reconstruction through official 0.12 and 0.13 decoders. JUMBF extraction is absent
from the frozen JPEG route, not an unimplemented active source caller. Caller
profile override state and CMS profile-to-field inference remain outside scope.
The narrow TIFF port does not claim general EXIF metadata interpretation. No
native codec dependencies or whole-image rotation are introduced, and no complete
JPEG-to-JXL acceptance is claimed by helper tests.
