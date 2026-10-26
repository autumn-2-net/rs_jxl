# Python Interface Boundary

The `rs_jxl` Python package calls the versioned Rust C ABI in-process.
See [Python API and packaging](PYTHON_API.md) for usage, supported controls,
ownership, validation and remaining differences from the C++ `jxlpy` package.

The encoder core owns Rust image/coefficient/frame data and does not depend on
Python. `rs-jxl-ffi` exposes a C ABI implemented in Rust for CFFI callers;
it does not call or link a C++ codec. `rs-jxl-image` shares PNG/JPEG input
implementations with the CLI, avoiding separate Python input semantics.

Maintenance contracts:

- An explicit ABI version and structure sizes, validated before accessing fields.
- Typed 8/16-bit samples, layout/stride/byte-order validation and resource limits.
- Borrowed input buffers for the duration of a call; library-owned output handles
  with matching Rust release functions. Never free Rust memory with Python/C.
- Separate pixel-lossless and original-JPEG transcode entry points. No silent
  fallback from JPEG coefficient transcoding to RGB pixel encoding.
- Multi-frame references/timings and separate raw-layer/coalesced-frame semantics.
- Explicit errors, panic barriers, thread limits and a documented thread-safety
  contract. No borrowed view may outlive its owner.
- Path and bytes interfaces at the Python layer, ndarray/tensor layout checks,
  and CPU transfer policy for tensors. No implicit lossy conversion.

The ABI header and serialized request contract live under `crates/rs-jxl-ffi`.
Breaking native layout/protocol changes must bump the ABI version and update
the Python declaration and tests together. Python metadata, arrays, encoding,
decoding and multiframe adapters are separate modules, not one API monolith.
