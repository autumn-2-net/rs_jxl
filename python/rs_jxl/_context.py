"""Optional v1 ABI extension for caller-owned, exclusively accessed workers."""

from contextlib import contextmanager
import threading
import weakref

from . import _ffi
from ._options import threads as normalize_threads


_ffi.ffi.cdef("""
typedef struct RsJxlContext RsJxlContext;
RsJxlResult *rs_jxl_context_create(const RsJxlRequest *request, RsJxlContext **output);
RsJxlResult *rs_jxl_context_execute(RsJxlContext *context, const RsJxlRequest *request);
void rs_jxl_context_free(RsJxlContext *context);
""")


class NativeContext:
    def __init__(self, kind, threads):
        self.threads = normalize_threads(threads)
        self._lock = threading.Lock()
        self._lib = _ffi.library()
        for symbol in ("rs_jxl_context_create", "rs_jxl_context_execute", "rs_jxl_context_free"):
            try:
                getattr(self._lib, symbol)
            except AttributeError as error:
                raise ImportError("Native library lacks reusable contexts; rebuild rs-jxl-ffi") from error
        request, owners = _ffi._request({"kind": kind, "threads": self.threads}, b"")
        output = _ffi.ffi.new("RsJxlContext **")
        result = self._lib.rs_jxl_context_create(request, output)
        try:
            with _ffi._result_buffer(self._lib, result):
                if output[0] == _ffi.ffi.NULL:
                    raise _ffi.CodecError("Native context creation returned null", kind="internal")
        except BaseException:
            if output[0] != _ffi.ffi.NULL:
                self._lib.rs_jxl_context_free(output[0])
            raise
        self._handle = output[0]
        self._finalizer = weakref.finalize(self, self._lib.rs_jxl_context_free, self._handle)
        del owners

    @property
    def closed(self):
        return not self._finalizer.alive

    def close(self):
        with self._lock:
            self._finalizer()

    @contextmanager
    def execute_buffer(self, request, data):
        with self._lock:
            if self.closed:
                raise RuntimeError("codec context is closed")
            native, owners = _ffi._request(request, data)
            result = self._lib.rs_jxl_context_execute(self._handle, native)
        # Results outlive the context. Conversion can invoke user callbacks,
        # which must be able to close/reenter the context without deadlocking.
        with _ffi._result_buffer(self._lib, result) as value:
            yield value
        del owners
