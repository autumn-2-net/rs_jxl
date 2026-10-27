# Python Interface Boundary

The `rs_jxl` Python package calls the versioned Rust C ABI in-process.
See [Python API and packaging](PYTHON_API.md) for usage, supported controls,
ownership, validation and remaining differences from the C++ `jxlpy` package.

The encoder core owns Rust image/coefficient/frame data and does not depend on
Python. `rs-jxl-ffi` exposes a C ABI implemented in Rust for CFFI callers;
it does not call or link a C++ codec. `rs-jxl-image` shares PNG/JPEG input
implementations with the CLI, avoiding separate Python input semantics.

## Codec Ownership

Still-image adapters construct `rs_jxl::codec::EncodeJob` and call `encode_job`
or `codec::Encoder::encode`. Optional `SourceRequest` selects explicit candidate,
automatic search, Scout or Pilot policy in the core rather than duplicating
effort-11 selection in the CLI/FFI. Python currently exposes source search, not
every Rust policy preset. JPEG uses `codec::transcode_jpeg` or its context method;
multiframe calls use the encoder's frame/automatic methods. `decode::Decoder`
is the separate reusable decoding resource owner.

This is not a complete backend-independent decoder type layer: public
`JxlBasicInfo`, `JxlFrameHeader`, color and extra-channel types still include
maintained-upstream reexports. Typed FFI `ValidatedRequest` views cover operation
selection after JSON validation, but still borrow shared options; buffer and
codec-specific checks remain in their adapters.

## ABI and Contexts

ABI version **1** and the `RsJxlRequest` layout are unchanged. The optional
additive symbols `rs_jxl_context_create`, `rs_jxl_context_execute` and
`rs_jxl_context_free` do not replace `rs_jxl_execute` or change result ownership.
Creation takes JSON such as `{"kind":"encoder","threads":4}` with no data
payload; `kind` may instead be `"decoder"`. Threads default to one and must be
1..256 at the C boundary.

Execute/free access to a handle must be exclusive; independent handles can run
concurrently. Omitted request threads inherit the handle's count, explicit
mismatches fail, and other options use per-request defaults. Only workers are
reused: no input pointers, parser/frame references or candidate caches survive a
call. Ordinary errors allow reuse; a panic poisons the context, which must then
only be freed. Results have separate ownership and remain valid after context
free. Stateless info/read_image operations do not reuse decoder workers;
context-bound PNG/JPEG reads still pass through the closed/lifecycle guard.

Python `Encoder(threads=...)` and `Decoder(threads=...)` wrap these handles with
serialized native execution, idempotent `close()`, `.closed`/`.threads` properties
and context-manager methods. Their lock guards the closed check and native call
against close, then releases before Python result conversion/copying or yielding
the result. Independent result ownership allows conversion callbacks to close
or reenter the context without holding that lock; entire Python methods are not
serialized. Their methods mirror the one-shot functions and
use per-call options except for fixed threads. Older ABI-v1 libraries remain
usable for one-shot operations; missing context symbols cause ImportError when
a Python context is constructed, not an ABI layout change.

## Buffers, Errors and Reports

Maintenance contracts:

- An explicit ABI version and structure sizes, validated before accessing fields.
- Typed U8/U16/U32 (declared 1..31-bit integer) and F16/F32 samples,
  layout/stride/byte-order validation and resource limits; no silent narrowing.
- Borrowed input buffers for the duration of a call; library-owned output handles
  with matching Rust release functions. Never free Rust memory with Python/C.
- Separate pixel-lossless and original-JPEG transcode entry points. No silent
  fallback from JPEG coefficient transcoding to RGB pixel encoding.
- Multi-frame references/timings and separate raw-layer/coalesced-frame semantics.
- Explicit errors, panic barriers, thread limits and a documented thread-safety
  contract. No borrowed view may outlive its owner.
- Path and bytes interfaces at the Python layer, ndarray/tensor layout checks,
  and CPU transfer policy for tensors. No implicit lossy conversion.

Python encoding snapshots mutable inputs by default. `borrow_inputs=True` on
`encode`/`encode_multiframe` explicitly permits eligible buffers to be borrowed
only through the synchronous call. Sources and aliases must not be mutated
concurrently, including while the GIL is released. Read-only aliases are not an
immutability guarantee. Layout/endian normalization and payload consolidation
may still copy; files/streams are fully buffered. This is not end-to-end zero-copy
encoding or a streaming API.

Decoded arrays/tensors/bytes are owned independently of the Rust result. For
nonempty data, `_ffi._result_buffer` attaches an `ffi.gc` owner to the result data
pointer, retained by `ffi.buffer`. A temporary derived view held in an exception
traceback safely retains the result beyond helper scope exit; the owner's
finalizer releases it after all retaining references are gone. The helper view
is released on exit; native error responses and empty-data results still
free directly in `finally`. Public outputs remain owned copies, not borrowed
views. `decode_into` copies one selected frame into a writable, C-contiguous
NumPy array with exact HWC shape and
native-endian dtype. Decode and descriptor/destination validation precede the
copy, so those failures leave the destination unchanged. It avoids intermediate
Python bytes/new output-array allocation, not Rust output storage or scratch.
Callers must not concurrently mutate the destination or its aliases. No borrowed
output or direct native decode-into-caller-memory contract is exposed.

FFI errors retain the JSON `error` string and add typed `error_kind`:
`invalid_input`, `resource_limit`, `encoding`, `decoding`, `io`, `internal`, or
`unsupported`; `error_stage` is optional. Python `CodecError.kind` exposes the
category. Known source errors retain their category; unclassified/string-only
errors can still default to `invalid_input`. This is not message-substring
classification or a guarantee that every unsupported case has its own variant.

`encode(..., return_info=True)` and `transcode_jpeg(..., return_info=True)`
return `(encoded_output, meta)` with codec statistics under
`meta["report"]`. The schemas differ: still-image reports project route,
implementation, size/pixel and candidate counts, effective parameters and reuse;
`implementation` is `modular` or `fast_lossless`, and effective `patches` is
`auto`, `on` or `off`. These stable JSON values do not use Rust Debug spelling.
`effective_parameters` projects scheduler `CandidateParams`, not a complete
backend recipe or exact tool list; FJXL uses its own default effort.
JPEG reports contain backend effort and trial/winner data. `codec_seconds`
excludes pool creation, external pixel-input decoding and transport/I/O; JPEG
parsing and coefficient preparation are included for JPEG transcoding. It does
not split analysis from backend execution. Multiframe strategy metadata retains its
existing separate shape. See [Python API](PYTHON_API.md#reports) for details.
Rust `EncodeReport` preserves `artifact: Option<NativeArtifact>`,
`adaptations: Vec<NativeAdaptation>` and `progressive: Option<ProgressiveReport>`
on source routes, including requested `record_representations` diagnostics.
These Rust fields do not change the FFI summary or add Python metadata fields.

The ABI header and serialized request contract live under `crates/rs-jxl-ffi`.
Breaking native layout/protocol changes must bump the ABI version and update
the Python declaration and tests together. Python metadata, arrays, encoding,
decoding and multiframe adapters are separate modules, not one API monolith.

Full incremental Python streaming, arbitrary manual blend headers, public
zero-copy outputs and full C++ option parity remain unimplemented. See
[current verification](PYTHON_API.md#current-verification) for the Rust, Python,
release-DLL, installed-wheel and derived-view lifetime checks. A successful build
alone does not establish runtime compatibility or performance.
