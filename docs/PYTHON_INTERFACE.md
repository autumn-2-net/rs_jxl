# Python Interface Boundary

This is an architectural reservation, not a working ABI or Python package.
Do not reintroduce the deleted bootstrap's CFFI wrapper as an implementation.

The encoder core owns Rust image/coefficient/frame data and does not depend on
Python. A later `rs-jxl-ffi` crate will expose a C ABI implemented in Rust for
CFFI callers; a C ABI is not permission to call or link a C++ codec.

Required contract before publishing:

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

Choose the detailed ABI only after the underlying Rust operations work. General
Rust packages may be used for threading, image I/O and packaging support.
