# Byte-Reversible JPEG API

`rs_jxl::jpeg::transcode` encodes an original JPEG into a standard JPEG XL
container, including its reconstruction data (`jbrd`). It operates on quantized
DCT coefficients. It does not decode JPEG to RGB, invoke an external encoder,
perform IDCT, or re-quantize coefficients. Unsupported input returns an error;
there is no pixel-lossless fallback.

This is a separately verified working path, not a declaration that the entire
optimized C++ encoder, its search scheduler, or the pixel phase is complete.

## CLI

```powershell
cargo build --locked --release -p rs-jxl-cli
.\target\release\rs-jxl.exe transcode-jpeg input.jpg archive.jxl --effort 9 --threads 4
```

On Linux/macOS use `./target/release/rs-jxl`. `transcode-jpeg` checks file
contents, not the extension. It is distinct from `encode input.jpg`, which
decodes JPEG to pixels and cannot preserve the original JPEG byte stream.
Use `-` for binary stdin/stdout. Avoid text-mode shell pipelines for image bytes.
Diagnostics go to stderr. Existing output is refused unless `--force` is set;
input and output still cannot be the same path. Encoding completes before
stdout writes or atomic file publication, so encoding errors leave output
untouched. Destination I/O errors can still fail publication or a stdout write.

| Option | Default / effect |
| --- | --- |
| `--effort 1..11` | 3; source coefficient backend policy |
| `--source-search` | Off; search the optimized C++ JPEG candidate prefix and retain its smallest complete output |
| `--threads 1..256` | 1; owned per-call pool for values above 1 |
| `--decoding-speed 0..4` | 0; decoder complexity policy |
| `--brotli-quality 0..11` | Unspecified uses `min(effort + 1, 11)` |
| `--no-cfl` | Disable coefficient CfL prediction |
| `--order source/default/adaptive` | `source`; `--default-order` aliases `--order default` |
| `--sample-limit N` | No extra ceiling by default; explicit N is a hard shared MA sample ceiling, not a sampling target |
| `--force` | Allow replacing an existing regular output file |

The CLI applies the other library budgets below. It has no metadata-stripping
or pixel-fallback mode. Use the Rust API to customize the remaining budgets.

## Memory and Writer Usage

```rust
use rs_jxl::jpeg::{transcode, transcode_to, TranscodeOptions};

fn archive(jpeg: &[u8]) -> rs_jxl::Result<Vec<u8>> {
    let options = TranscodeOptions {
        effort: 9,
        threads: 8,
        ..Default::default()
    };
    transcode(jpeg, &options)
}

fn archive_to(jpeg: &[u8], sink: &mut impl std::io::Write) -> rs_jxl::Result<()> {
    transcode_to(jpeg, &TranscodeOptions::default(), sink)
}
```

Input is a borrowed byte slice. The output is an owned byte vector, or is
written to a caller-owned `Write` implementation. The writer version completes
encoding before writing: encode/validation errors leave the sink untouched.
An I/O error can leave a partially written sink. It does not flush or close it.
The convenience writer is not a streaming encoder and retains the encoded
container in memory until writing finishes.

## Controls

| Field | Default | Meaning |
| --- | --- | --- |
| `effort` | 3 | 1..=11; direct JPEG frame entry normalizes 1 to 2 and 11 to 10, then maps to tier `10 - effort` |
| `threads` | 1 | 1..=256; 1 stays on the calling thread, other values use a bounded per-call Rayon pool |
| `cooperative_ma` | `None` | Rust API opt-in `Some(scheduler::WorkerLimits)` lets multiple auxiliary MA learners and their split tasks share the same pool |
| `decoding_speed` | 0 | 0..=4, source auxiliary-tree/entropy decode-complexity policy |
| `cfl` | `CflMode::Enabled` | Reversible JPEG AC chroma-from-luma prediction; source eligibility still applies |
| `order` | `OrderPolicy::Source` | Source adaptive-order decision, explicit `Default`, or forced `Adaptive` |
| `reconstruction_quality` | `None` | Brotli quality 0..=11 for reconstruction metadata; default `min(effort + 1, 11)` |
| `limits` | `TranscodeLimits::default()` | Explicit input/work/storage/output ceilings |

The CfL search uses source floating-point scoring, but its coefficient residual
and inverse are integer operations. No JPEG pixel buffer or floating-point IDCT
is involved. The DC path follows source rounded dequantization factors and
rejects values it cannot reconstruct exactly rather than silently clipping.

Higher effort in the direct backend is not a guarantee of a smaller output.
Nor does `threads=N` imply N-way scaling for every stage. Exact reconstructed
JPEG bytes, not identical JXL bytes across strategies, are the correctness
criterion.

The cooperative MA option requires more than one useful auxiliary domain and
an actual caller pool. One-thread requests remain serial. It preserves the
selected-sample ceiling, gathering order, domain boundaries and merge order;
it does not change the search candidate list or create another thread pool.
`WorkerLimits` defaults to eight logical tasks and a 1 GiB scratch estimate.
The per-root estimate is 32 MiB, not a hard process-wide memory guarantee.
The CLI does not currently expose this opt-in. See
[cooperative MA details](JPEG_COOPERATIVE_MA.md) for source gates and executor
differences; no speedup is implied without a measurement.

## Source Candidate Search

`jpeg::transcode_search(input, options, jpeg::search::SearchLimits::default())`
returns `SearchResult` with `data`, the winning `backend_effort`, `winner_index`,
all completed `trials` and total `full_pixels`. CLI: add `--source-search`.

The frozen source JPEG schedule is:

| Requested effort reaches | Candidate backend effort |
| --- | --- |
| 1 | 3 |
| 3 | 5 |
| 5 | 7 |
| 7 | 8 |
| 9 | 9 |
| 10 | 10 |

Effort 11 uses the same JPEG prefix as 10, not the pixel Tectonic tournament.
Each candidate is an actual complete coefficient encoding. Stable ties retain
the earlier result. There is no crop estimate, timeout fallback or input-file
hash target. Input parsing, image header, reconstruction metadata and the pool
are reused; backend analysis is still repeated as required by each candidate.
`decoding_speed != 0` is explicitly rejected by search, matching source search
eligibility; direct `transcode` remains available with that control.

Search limits bound candidate count, total full-image pixel work and output
bytes. Exhausting a limit returns an error, not an incomplete winner. To compare
prefix monotonicity, keep all other controls fixed, including an explicit
`reconstruction_quality`: its automatic effort-dependent default otherwise
changes reconstruction compression between calls.

## Resource Limits

`TranscodeLimits` and nested parser/coefficient/reconstruction defaults now use
the corresponding integer maximum, meaning no extra resource-policy ceiling.
Format and numeric representation limits still apply. See the
[resource-limit audit](RESOURCE_LIMIT_AUDIT.md).

- `input`: explicit limits can bound input bytes, coefficient storage, decoded
  block visits, reconstruction hints, padding, records, scans and tables.
- `max_output_bytes`: bounds the complete returned container, including
  reconstruction and display metadata boxes.
- `max_coefficients`: counts i32 values in the JXL working layout, including
  MCU padding and subsampled zero tails, not bytes or original coefficients.
- `max_tokens`: bounds each AC/auxiliary coding stage. Reservations are checked
  before dispatch and are not multiplied by worker count.
- `sample_limit`: selected MA samples across auxiliary domains. `usize::MAX`
  means no extra ceiling; a finite explicit value errors rather than reducing
  the sampling rate. There is no longer a hidden four-million-sample default.

These are stage/resource ceilings, not a process-wide peak-memory sandbox.
Original coefficient storage, JXL workspaces, token streams, histograms and
output may coexist. Raising one limit does not implicitly raise the others.
No resource error silently lowers effort, truncates training, or changes routes.
Budget/allocation failures in the bounded parser, metadata, DC and AC stages
are propagated as `Error::Limit`, not mislabeled as malformed JPEG input.
Unsupported or malformed input validation returns `Error::InvalidInput`;
writer I/O failures retain the underlying I/O error.

## Input and Preservation

Supported input uses 8-bit Huffman DCT coding: baseline, extended sequential,
or progressive JPEG, with one or three components and supported source chroma
sampling. Restart markers, signed refinement, scan layout, metadata, padding,
and bytes after EOI are retained for byte-exact JPEG reconstruction.

Arithmetic JPEG, lossless-JPEG, hierarchical/DNL modes, CMYK/YCCK, non-8-bit
input and unsupported reconstruction structures are errors. Examples of current
reconstruction restrictions include incompatible DQT redefinitions, changing
DRI intervals, restart fill and more than three reconstructed quant tables.
This is not a universal JPEG conformance validator.

Both 8-bit and 16-bit DQT entries are accepted. Like the frozen encoder,
layout preparation requires `floor(2048 * Qmiddle / Qchannel) <= 524287`
at every coefficient, even with CfL disabled. `Qmiddle` is Y for YCbCr and
G for RGB. Extreme 16-bit table ratios can therefore be rejected; this is a
source encoder restriction, not a claim that the JXL format forbids them.

Original APP/COM contents are preserved for reconstruction. ICC bytes are
placed in the JXL image metadata. ICC parsing is intentionally strict:
malformed or mismatched profiles fail instead of being silently replaced with
sRGB. EXIF orientation is interpreted for JXL display, without rotating stored
coefficients; its original bytes are preserved. A separate Exif box exposes
the TIFF payload. Standard APP1 XMP is also exposed in an `xml ` box; its bytes
are opaque, without XML rewriting or extended-XMP assembly. APP11/JUMBF stays
in the reconstruction archive, matching the frozen source's JPEG extraction.
ICC/Exif/XMP payloads are deduplicated from `jbrd` only when they match the
actual emitted profile or box bytes. Ordered ICC chunk selections, the first
eligible Exif/XMP marker and source decoder reconstruction rules are respected.
Mismatched, incomplete or unselected duplicate APP markers retain their opaque
bytes; no metadata is discarded to save space.

The returned JXL can be reconstructed with an official decoder, for example:

```text
djxl archive.jxl restored.jpg
```

Check `restored.jpg` against the original JPEG bytes. Decoding both JPEGs to RGB
is insufficient to establish archival correctness. The Rust runtime decoder is
available as `rs_jxl::decode::reconstruct_jpeg(&jxl, &DecodeOptions::default())`.
It restores original bytes from captured coefficients and `jbrd`, without pixel
re-encoding or access to the original JPEG. Missing, ambiguous required metadata
sources and incomplete input are errors; pixel partial/progression options never
permit a partial JPEG. Reconstruction budget failures return `Error::Limit`.
The Python CFFI/wheel layer remains separate pending work.

## Verification

Independent acceptance uses official djxl 0.12 and 0.13, never an external
encoder. The shared auxiliary-model composer passed 456 synthetic checks;
all-tier, execution-setting, metadata and byte/writer adapter tests add 138
checks. The new shared model also passed 102 real-image checks across 17 files,
including the 6144x3321 case with the explicit four-million-sample ceiling.
These **696** checks used official djxl 0.12 and 0.13 and fully Rust-produced
frames, not transplanted official coefficient codestreams.
The separate normal-release CLI check added **44** original-byte comparisons
over four real JPEGs, efforts 3/9 and threads 1/4, including binary stdin/stdout,
file output and failure protection. Reproduce that opt-in check with:

```powershell
python -B tools/check_jpeg_cli.py --encoder target/release/rs-jxl.exe `
  --decoder ../jxl_python/out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe `
  --decoder ../jxl_python/out/encoder_opt_baseline/djxl.exe `
  --effort 3 9 --threads 1 4 --timeout 120 -- input.jpg another.jpg
```

The helper requires Pillow only for dimensions/bpp, not for JPEG transcoding.
It uses bounded child-process timeouts, compares complete original JPEG bytes,
and removes only its private temporary outputs. It never writes to input files.
See `PORT_VERIFICATION.md` for the latest actual build/corpus checkpoint.
