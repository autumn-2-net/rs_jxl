"""Versioned CFFI boundary. No Python interpreter state enters the Rust codec."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading

from cffi import FFI


ffi = FFI()
ffi.cdef("""
typedef struct {
    uint32_t abi_version;
    size_t struct_size;
    const uint8_t *json;
    size_t json_len;
    const uint8_t *data;
    size_t data_len;
} RsJxlRequest;
typedef struct RsJxlResult RsJxlResult;
uint32_t rs_jxl_abi_version(void);
size_t rs_jxl_request_size(void);
RsJxlResult *rs_jxl_execute(const RsJxlRequest *request);
int32_t rs_jxl_result_status(const RsJxlResult *result);
const uint8_t *rs_jxl_result_json(const RsJxlResult *result, size_t *length);
const uint8_t *rs_jxl_result_data(const RsJxlResult *result, size_t *length);
void rs_jxl_result_free(RsJxlResult *result);
""")

ABI_VERSION = 1
_native = None
_lock = threading.Lock()


class CodecError(RuntimeError):
    """The Rust codec rejected input, options or an exhausted resource budget."""


def native_name() -> str:
    if sys.platform == "win32":
        return "rs_jxl_ffi.dll"
    if sys.platform == "darwin":
        return "librs_jxl_ffi.dylib"
    return "librs_jxl_ffi.so"


def _library_path() -> Path:
    explicit = os.environ.get("RS_JXL_NATIVE_LIB")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ImportError(f"RS_JXL_NATIVE_LIB does not exist: {path}")
        return path
    package = Path(__file__).resolve().parent
    bundled = package / native_name()
    if bundled.is_file():
        return bundled
    root = package.parent.parent
    if (root / "Cargo.toml").is_file() and (root / "crates/rs-jxl-ffi").is_dir():
        target = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
        if not target.is_absolute():
            target = root / target
        triple = os.environ.get("CARGO_BUILD_TARGET")
        if triple:
            target /= triple
        path = target / "release" / native_name()
        if path.is_file():
            return path.resolve()
    raise ImportError(
        "Rust native library is missing. Install a platform wheel, run "
        "cargo build --release -p rs-jxl-ffi, or set RS_JXL_NATIVE_LIB."
    )


def _validate_abi(lib, path: Path) -> None:
    try:
        version = lib.rs_jxl_abi_version()
        size = lib.rs_jxl_request_size()
        for symbol in ("rs_jxl_execute", "rs_jxl_result_status", "rs_jxl_result_json",
                       "rs_jxl_result_data", "rs_jxl_result_free"):
            getattr(lib, symbol)
    except AttributeError as error:
        raise ImportError(f"Missing rs_jxl ABI symbols in {path}") from error
    if version != ABI_VERSION or size != ffi.sizeof("RsJxlRequest"):
        raise ImportError(
            f"Incompatible rs_jxl native ABI in {path}: version={version}, "
            f"request_size={size}; expected {ABI_VERSION}/{ffi.sizeof('RsJxlRequest')}"
        )


def library():
    global _native
    with _lock:
        if _native is None:
            path = _library_path()
            try:
                loaded = ffi.dlopen(str(path))
            except OSError as error:
                raise ImportError(f"Cannot load Rust native library {path}: {error}") from error
            _validate_abi(loaded, path)
            _native = loaded
        return _native


def _copy(getter, result) -> bytes:
    length = ffi.new("size_t *")
    pointer = getter(result, length)
    if not length[0]:
        return b""
    if pointer == ffi.NULL:
        raise CodecError("Native result has a null nonempty buffer")
    return bytes(ffi.buffer(pointer, length[0]))


def execute(request: dict, data: bytes | bytearray | memoryview = b"") -> tuple[bytes, dict]:
    lib = library()
    encoded = json.dumps(request, allow_nan=False, separators=(",", ":")).encode("utf-8")
    payload = memoryview(data)
    if not payload.c_contiguous:
        payload = memoryview(payload.tobytes())
    payload = payload.cast("B")
    c_json = ffi.from_buffer("const uint8_t[]", encoded)
    c_data = ffi.from_buffer("const uint8_t[]", payload) if payload.nbytes else ffi.NULL
    native_request = ffi.new("RsJxlRequest *", {
        "abi_version": ABI_VERSION, "struct_size": ffi.sizeof("RsJxlRequest"),
        "json": c_json, "json_len": len(encoded),
        "data": c_data, "data_len": payload.nbytes,
    })
    # Keep Python-owned input buffers alive throughout the synchronous call.
    result = lib.rs_jxl_execute(native_request)
    if result == ffi.NULL:
        raise CodecError("Rust native call returned no result")
    try:
        meta = json.loads(_copy(lib.rs_jxl_result_json, result))
        if lib.rs_jxl_result_status(result):
            raise CodecError(meta.get("error", "Rust codec operation failed"))
        return _copy(lib.rs_jxl_result_data, result), meta
    finally:
        lib.rs_jxl_result_free(result)
