# JPEG Reconstruction Compatibility

## Contract and Scope

JPEG archival means reconstructing the original JPEG bytes from coefficients
and standard reconstruction metadata. It is not merely decoding to the same
pixels. No private box, whole-JPEG attachment, coefficient-as-pixel convention,
or automatic pixel-lossless fallback is implemented here.

The current Rust transcoder handles its documented subset of 8-bit Huffman DCT
JPEG. The new header probe is a diagnostic tool, **not an implementation of
arithmetic/CMYK/12-bit JPEG archival**. Normal `transcode` validation remains
mandatory. Python integration is outside this work.

## Evidence and Classification

The distinction matters: a field's storage capacity, a conforming reconstruction
mapping, and an implementation accepting that mapping are different things.
The following audit uses the local reference libjxl commit
`b866b3defec13494bfbaf705db93581cda118e37`, not a claim about every future libjxl.

The inspected first-edition standard's Table 1 can signal 1..4 components;
this alone does not establish a four-component coefficient mapping. Its SOF
procedure explicitly includes C9/CA, but emits precision 8. The SOS procedure
refers to JPEG coefficient encoding, with additional Huffman-oriented rules.
There is no dedicated DAC bundle/marker-generation rule. Unrecognized segments
can be retained separately, so absence of a DAC field alone does **not** prove
that all arithmetic reconstruction is forbidden. Source:
[ISO/IEC 18181-2:2021, Tables 1 and 6; 9.11.2.1, .2, .6 and .11 (public copy)](https://www.normsplash.com/FreeDownload/146298163/ISO-IEC-18181-2-2021-en.pdf).

The [2026 ISO foreword](https://www.iso.org/obp/ui?_escaped_fragment_=iso%3Astd%3Aiso-iec%3A18181%3A-2%3Aed-3%3Av1%3Aen) notes revised
reconstruction clarifications. This audit has not obtained the full 2024/2026
normative clauses; the first edition is not being asserted to settle every
current-edition arithmetic edge case.

| Input / structure | Evidence category | Current disposition |
| --- | --- | --- |
| 8-bit Huffman SOF0/1/2, one or three components | Existing implementation, subject to full validation | Candidate |
| CMYK/YCCK four-component JPEG | Coefficient-mapping limitation, not merely a two-bit count issue | Reject archival conversion |
| Non-8-bit JPEG sample precision, including 12-bit | Inspected standard SOF procedure fixes output precision at 8 | Reject; increasing Rust coefficient width alone cannot preserve SOF |
| Arithmetic SOF9/SOF10 | Standard SOF enumeration admits these; complete interoperable mapping remains unresolved | Report unsupported by current implementation, not universally forbidden |
| Arithmetic DAC and selectors 4..15 | No dedicated current reconstruction state for conditioning; scan selector fields are two bits | Diagnose; no silent defaulting or selector remapping |
| Lossless JPEG SOF3/7/11/15; hierarchical/differential modes | Absent from inspected ordinary SOF reconstruction procedure; current coefficient tools are DCT-only | Reject, without asserting arbitrary-container storage is impossible |
| DNL / height deferred from SOF | Current reader and reconstruction procedure do not preserve this structure | Reject |
| Ordinary restart intervals and correctly placed RSTn | Already implemented | Do not reject solely because restart markers exist |
| Repeated DRI with the same interval | Rust supports it; reference C++ input parser rejects duplicate DRI | Candidate, with scan validation still required |
| Progressive DC refinement whose unused DC selector has no table | Raw correction-bit scan; no new representation is required | Rust metadata checks corrected; see regression below |
| Changing DRI interval | Current reconstruction model retains one global interval | Reject current path; not a general impossibility theorem about raw segment mappings |
| Fill bytes before an in-scan restart | Current entropy reconstruction emits the restart directly, without a per-restart fill field | Reject current path |
| 16-bit quantization entries with 8-bit samples | Distinct from 12/16-bit sample precision | Candidate; coefficient/layout limits still apply |
| More than three DQT definitions, incompatible DQT redefinitions | Current model restrictions, not inferred solely from two-bit table-count capacity | Reject / full validation |

CMYK has additional primary evidence: the format authors describe the fixed
three-component VarDCT mapping and Modular-only extra channels as an intentional
limit on lossless CMYK JPEG recompression. Storing K samples discards its original
DCT coefficients. This does not prevent ordinary CMYK pixel-lossless JXL; it
prevents treating that pixel route as original-JPEG archival.
[Authors' paper, section 4.3.5](https://arxiv.org/html/2506.05987v1#S4.SS3.SSS5).

## Exact Reference Implementation Checks

Paths and line numbers in this section are relative to the reference libjxl
checkout at the commit above. They establish implementation behavior; they
are not substitutes for normative text.

- `lib/jxl/jpeg/enc_jpeg_data_reader.cc:65`, `ProcessSOF`: precision is checked
  against 8 at line 80. `ReadJpeg` at line 955 handles SOF0/1/2, not arithmetic,
  lossless, hierarchical or DNL input. `FindNextMarker` at line 904 recognizes
  only its whitelist, so reading a raw marker byte somewhere is not support.
- `lib/jxl/jpeg/enc_jpeg_data.cc:216`, `SetChromaSubsamplingFromJpegData`, and
  `:242`, `SetColorTransformFromJpegData`: only 1/3 components are accepted.
- `lib/jxl/jpeg/jpeg_data.cc:70`, `JPEGData::VisitFields`: component count uses
  an explicit 1/2/3/4 choice at line 183, followed by an explicit 1/3 check at
  line 184. The rejection is **not** evidence that the syntax has no fourth
  value. Quant table count similarly rejects 4 at line 147. Scan selectors at
  lines 295/296 are two-bit fields. Line 307 stores one 16-bit restart interval.
- `lib/jxl/dec_frame.cc:214`, `FrameDecoder::InitFrameOutput`: JPEG output
  rejects Modular, non-1/3 component counts and XYB, then derives coefficient
  buffers and sampling from the three-plane frame.
- `lib/jxl/jpeg/dec_jpeg_data_writer.cc:332`, `EncodeSOF`: line 344 writes
  `kJpegPrecision`, fixed at 8 on line 41. `SerializeSection` at line 919
  accepts SOF9/10 but has no DAC case. Do not mistake those SOF cases for a
  working arithmetic writer: `DoEncodeScan` calls Huffman block writers at
  lines 865/869; the Huffman tables are built by `EncodeDHT`.
- `lib/jxl/jpeg/enc_jpeg_data_reader.cc:334`, `ProcessDRI`: all duplicate DRI
  markers are rejected. `ProcessRestart` at line 725 and the scan MCU loop
  at line 810 validate interval/order. Writer `EncodeDRI` at line 484 always
  emits the global interval; the scan writer at line 814 emits two-byte RSTn.
- `lib/jxl/frame_header.h:103`, `YCbCrChromaSubsampling::Set`: component factors
  must match one of four 1x1/2x1/1x2/2x2 modes. These are raw sampling factors,
  not just ratios. Rust RGB layout additionally requires all raw factors 1.

Matching Rust checks are in `jpeg/reader.rs`, `sof.rs`, `transcode_layout.rs`,
`frame_header.rs`, `coefficient_reader.rs`, `reconstruction_metadata.rs`, and
`reconstruction_decode.rs`. A preflight must not replace their entropy,
progression, metadata and byte-reconstruction validation.

## DC Refinement Check Correction

Progressive Huffman DC refinement emits a raw correction bit per DC coefficient,
not a Huffman code. The JPEG XL SOS reconstruction procedure delegates the
coefficient encoding to JPEG; no extra `jbrd` field is needed for this case.
The existing Rust reader, coefficient reader and scan writer already distinguish
initial DC scans (`Ss == 0 && Ah == 0`) from refinement (`Ss == 0 && Ah != 0`).
The local libjpeg-turbo `jdphuff.c:163` likewise builds DC Huffman tables only
for the initial scan. This is separate from arithmetic JPEG support.

`reconstruction_metadata.rs` and `reconstruction_decode.rs` now apply that
same distinction. An initial DC scan still needs its selected table; sequential
scans still need both tables; AC scans retain their table requirements.
Progression validation is unchanged. Original SOS selectors and DHT markers
are preserved exactly: no replacement selector, added dummy DHT, or pixel path.

The independent integration target `tests/jpeg_dc_refinement.rs` hand-builds
progressive JPEGs defining DC table 1 and AC table 0, with no DC table 0.
Initial DC uses table 1; refinement has selector 0. It covers gray and three
components, positive/negative DCs, ordinary/restarted scans, and threads 1/4.
It checks the known source coefficients and absent DC table 0, then performs
`jpeg::transcode` -> `decode::reconstruct_jpeg`, giving the decoder only JXL
bytes. Acceptance compares the entire original JPEG, including markers and
tail, and rechecks the restored coefficients/tables. Negative tests retain
missing initial-DC/AC-table and refinement-before-initial rejection.

The normative argument is the correction-bit coding rule; tests verify the
implementation and byte-exact roundtrip, not normative validity by themselves.
The pinned C++ `jpeg_data.cc:425` still requires a DC table for refinement,
so official JPEG reconstruction is not claimed to pass this extension. Its
failure must be recorded separately, not reclassified as a format prohibition
or hidden by adding a DHT. No official reconstruction test has been run for
this fixture in the isolated implementation task.

Parent-run verification filters (implementation added; execution pending the
central Cargo run):

```text
cargo test --locked -p rs-jxl --test jpeg_dc_refinement
cargo test --locked -p rs-jxl --lib jpeg::reconstruction_metadata
cargo test --locked -p rs-jxl --lib jpeg::reconstruction_decode
```

## Arithmetic Feasibility: Work Required Before Support

Local libjpeg-turbo has reusable algorithm source, without requiring a runtime
C dependency: `third_party/libjpeg-turbo/jaricom.c` (QM state table),
`jdarith.c` (arithmetic coefficient reader), and `jcarith.c` (writer).
License/attribution review and direct Rust ports would still be required.

An initially bounded experiment could restrict input to sequential 8-bit SOF9,
one/three components, existing supported sampling, selectors 0..3 and default
conditioning. It is **not yet a supported or proven standard-compatible mode**:

1. Establish the applicable edition's arithmetic reconstruction semantics,
   especially whether retained DAC segments update arithmetic encoder state.
   `jdmarker.c:213` initializes DC L=0/U=1 and AC K=5; `:388` parses DAC.
   `jcmarker.c:228` emits DAC, including when the values are defaults. Thus
   "default conditioning" does not mean existing input has no DAC bytes.
2. Prove a legitimate mapping for DAC segments. A small original marker retained
   as an unrecognized segment is not the same as stashing an entire JPEG, but
   a reconstructed coefficient writer must interpret its state correctly.
   The current libjxl `EncodeInterMarkerData` only copies bytes; it does not
   update an arithmetic coder. Standard permission for that interpretation
   needs verification, not invention.
3. Port coefficient decoding and encoding with bounded MCU/restart work and
   exact input comparisons. `jcarith.c:139`, `finish_pass`, implements one
   specific final coding-interval choice, delayed carry, byte stuffing and
   trailing-zero suppression. Successfully decoding coefficients does not
   prove that re-encoding reproduces every encoder's original entropy bytes.
   A narrower canonical-input subset must explicitly reject any mismatch.
4. Account for `jbrd` metadata conventions: its current Huffman-count coding has
   minimum 2 despite arithmetic scans not needing DHT. Unused entries are not
   by themselves proof of invalidity, but a valid mapping must be specified
   and independently checked, not treated as private state storage.
5. Add arithmetic scan snapshots/progression, reconstruction state and matching
   writer support. Changing only `reader.rs` to accept C9/CA would bypass gates
   without adding these missing algorithms and would be incorrect.
6. Independently validate normative output and exact reconstruction. The user's
   updated policy permits a standard-supported feature even if libjxl lacks its
   implementation. Keep three separate acceptance checks: standard mapping,
   JXL coefficient/pixel decoding, and original-JPEG-byte reconstruction.
   The checked official writer cannot serve as a passing arithmetic byte
   reconstruction oracle today. That implementation gap alone is not a veto
   under the updated policy, but neither is it a passing test. Use independent
   JPEG coefficient decoding and original-byte comparisons, and explicitly
   record that the official JPEG reconstruction path does not support the mode.

The probe below only provides the facts needed to identify candidate samples
for that investigation. No arithmetic encoding is enabled by it.

## Bounded Probe API

Implementation: `crates/rs-jxl/src/jpeg/compatibility.rs`, exported as
`rs_jxl::jpeg::compatibility`. This optional diagnostic does not replace
validation inside `archival.rs`.

```rust,ignore
use rs_jxl::jpeg::compatibility::{probe_with_limits, ProbeLimits};

let report = probe_with_limits(input, &ProbeLimits::default())?;
if !report.is_candidate() {
    for finding in &report.findings {
        eprintln!("byte {} marker {:02x}: {:?}",
                  finding.offset, finding.marker, finding.feature);
    }
}
// No findings means only "no detected header obstacle".
// Continue through jpeg::transcode and original-byte verification.
```

`probe` uses defaults; `probe_with_limits` accepts input-byte, marker, scan and
finding ceilings. It walks through the first EOI, skips length-delimited APP/COM
payloads, skips entropy with stuffing awareness, and counts tail bytes without
interpreting them as a second image. Work is linear in input size; retained
memory is a maximum 255-entry frame header and the bounded finding vector.
No allocation is proportional to width, height, MCU count or entropy size.
Allocation failure is a `ProbeError::Limit`, never a partial success.

The report includes first SOF/component facts, scan and restart counts, DRI
changes, DQT definition count, Adobe CMYK/YCCK hints, arithmetic conditioning
definition count/default status, and selectors above 3. Four components without
Adobe metadata are reported as unspecified, not confidently labeled CMYK.
Findings include the actual marker offset and structured feature code.
Malformed lengths/headers, truncated input and exhausted limits are errors.

It intentionally does not validate Huffman contents, missing tables, entropy
codes, restart spacing/order, progressive refinement, ICC, quantizer consistency
between scans, coefficient limits or encoder resources. In particular,
`is_candidate()` may be true for an input whose complete transcoding fails.
Limits are preflight limits, not a substitute for `TranscodeLimits`.

## Verification

The module's isolated synthetic unit tests cover special SOFs, APP14 hints,
DAC defaults/invalid values, DRI change vs duplication, restart fill vs presence,
16-bit DQT vs sample precision, marker-looking metadata/stuffed/tail bytes,
truncation of every prefix, exact resource edges and deterministic malformed
inputs. These are **preflight tests, not new archival feature acceptance**.
Workspace export and the final integrated Cargo run belong to the parent task.
