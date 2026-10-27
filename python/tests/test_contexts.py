"""Reusable-context state, lifetime, and public entry-point parity tests."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import gc
import importlib.util
import io
import json
import math
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
import rs_jxl
from rs_jxl import _ffi

try:
    import test_bindings as fixtures
except ModuleNotFoundError as error:
    if error.name != "test_bindings":
        raise
    # pytest importlib mode need not put the tests directory on sys.path.
    spec = importlib.util.spec_from_file_location(
        "_context_fixtures", Path(__file__).with_name("test_bindings.py"))
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)


class ContextTests(fixtures.PixelAssertions, unittest.TestCase):
    def test_output_conversion_can_close_context_without_deadlocking(self):
        source = fixtures._rgba(height=2, width=3)
        data = rs_jxl.encode(source, effort=3)
        destination = np.empty_like(source)
        original = np.copyto
        with rs_jxl.Decoder(threads=2) as decoder:
            def close_during_copy(*args, **kwargs):
                # Assert before calling close, so a locking regression fails
                # immediately instead of leaving this test hanging.
                self.assertFalse(decoder._native._lock.locked())
                decoder.close()
                return original(*args, **kwargs)

            with patch("rs_jxl._arrays.np.copyto", side_effect=close_during_copy):
                self.assertIs(decoder.decode_into(data, destination), destination)
            self.assertTrue(decoder.closed)
        self.assertPixels(destination, source)

    def test_input_callback_cannot_bypass_closed_context_for_pixel_readers(self):
        jpeg = fixtures._tiny_jpeg()
        png = rs_jxl.decode_to_png(rs_jxl.encode(fixtures._rgba(), effort=3))
        for data in (jpeg, png):
            with self.subTest(signature=data[:4]), rs_jxl.Decoder(threads=2) as decoder:
                class ClosingStream:
                    def read(self):
                        decoder.close()
                        return data

                with self.assertRaisesRegex(RuntimeError, "closed"):
                    decoder.decode(ClosingStream())

    def test_pixel_readers_validate_threads_like_jxl_inputs(self):
        jpeg = fixtures._tiny_jpeg()
        for threads in (True, -1, 257, 1.5):
            with self.subTest(threads=threads), self.assertRaises((TypeError, ValueError)):
                rs_jxl.decode(jpeg, threads=threads)

    def assertReport(self, encoded, metadata, source, route="candidate"):
        self.assertIsInstance(encoded, bytes)
        self.assertEqual(metadata["bytes"], len(encoded))
        report = metadata["report"]
        self.assertEqual(report["route"], route)
        self.assertEqual(report["pixels"], source.shape[0] * source.shape[1])
        self.assertEqual(report["bytes"], len(encoded))
        self.assertAlmostEqual(report["bpp"], len(encoded) * 8 / report["pixels"])
        self.assertTrue(math.isfinite(report["codec_seconds"]))
        self.assertGreaterEqual(report["codec_seconds"], 0)
        self.assertGreaterEqual(report["full_encodes"], 1)
        self.assertGreaterEqual(report["probe_encodes"], 0)
        if route == "candidate":
            self.assertEqual(report["full_encodes"], 1)
            self.assertEqual(report["probe_encodes"], 0)
            self.assertIsNone(report["winner_index"])
        else:
            self.assertIsInstance(report["winner_index"], int)
            self.assertGreaterEqual(report["winner_index"], 0)

    def test_serial_and_two_worker_reuse_preserves_original_samples_after_errors(self):
        for threads in (1, 2):
            with self.subTest(threads=threads):
                with rs_jxl.Encoder(threads=threads) as enc, rs_jxl.Decoder(threads=threads) as dec:
                    retained = []
                    for source in (fixtures._rgba(), fixtures._rgba(np.uint16, 3, 4),
                                   np.arange(15, dtype=np.uint8).reshape(3, 5)):
                        with self.assertRaises(rs_jxl.CodecError) as failure:
                            enc.encode(source, effort=3, modular_colorspace=42)
                        self.assertEqual(failure.exception.kind, "invalid_input")
                        encoded, metadata = enc.encode(source, effort=3, return_info=True)
                        self.assertReport(encoded, metadata, source)
                        with self.assertRaises(rs_jxl.CodecError) as failure:
                            dec.decode(encoded, max_pixels=1)
                        self.assertEqual(failure.exception.kind, "resource_limit")
                        with self.assertRaises(rs_jxl.CodecError):
                            dec.decode(b"\xff\x0a\x00")
                        actual = dec.decode(encoded)
                        self.assertPixels(actual, source)
                        self.assertPixels(rs_jxl.decode(encoded, threads=threads), source)
                        fresh, fresh_info = rs_jxl.encode(source, effort=3, threads=threads,
                                                         return_info=True)
                        self.assertReport(fresh, fresh_info, source)
                        self.assertPixels(dec.decode(fresh), source)
                        retained.append((encoded, actual, source.copy()))
                    self.assertFalse(np.shares_memory(retained[0][1], dec.decode(retained[0][0])))
                self.assertTrue(enc.closed)
                self.assertTrue(dec.closed)
                gc.collect()
                for encoded, actual, expected in retained:
                    self.assertPixels(actual, expected)
                    self.assertTrue(actual.flags.writeable)
                    self.assertPixels(rs_jxl.decode(encoded, threads=threads), expected)

    def test_close_is_idempotent_and_every_bound_entry_rejects_closed_context(self):
        source = fixtures._rgba(height=2, width=3)
        jpeg = fixtures._tiny_jpeg()
        with rs_jxl.Encoder(threads=2) as enc, rs_jxl.Decoder(threads=2) as dec:
            self.assertFalse(enc.closed)
            self.assertFalse(dec.closed)
            self.assertEqual((enc.threads, dec.threads), (2, 2))
            encoded = enc.encode(source, effort=3)
            self.assertPixels(dec.decode(encoded), source)
        for context in (enc, dec):
            context.close()
            context.close()
            self.assertTrue(context.closed)
            with self.assertRaises(RuntimeError):
                context.__enter__()
        calls = [
            lambda: enc.encode(source, effort=3),
            lambda: enc.transcode_jpeg(jpeg, effort=3),
            lambda: enc.encode_multiframe([source], effort=3),
            lambda: dec.decode(encoded),
            lambda: dec.decode_into(encoded, np.empty_like(source)),
            lambda: dec.reconstruct_jpeg(encoded),
            lambda: dec.info(encoded),
            lambda: dec.decode_layer(encoded),
        ]
        for index, call in enumerate(calls):
            with self.subTest(entry=index), self.assertRaises(RuntimeError):
                call()
        for factory in (rs_jxl.Encoder, rs_jxl.Decoder):
            context = factory(threads=1)
            with self.assertRaisesRegex(LookupError, "caller failure"):
                with context:
                    raise LookupError("caller failure")
            self.assertTrue(context.closed)

    def test_thread_mismatches_do_not_change_context_or_next_call(self):
        source = fixtures._rgba(height=2, width=3)
        with rs_jxl.Encoder(threads=2) as enc, rs_jxl.Decoder(threads=2) as dec:
            encoded = enc.encode(source, effort=3)
            destination = np.full_like(source, 71)
            calls = [
                lambda: enc.encode(source, effort=3, threads=1),
                lambda: enc.transcode_jpeg(fixtures._tiny_jpeg(), effort=3, threads=1),
                lambda: enc.encode_multiframe([source], effort=3, threads=1),
                lambda: dec.decode(encoded, threads=1),
                lambda: dec.decode_into(encoded, destination, threads=1),
                lambda: dec.info(encoded, threads=1),
                lambda: dec.decode_layer(encoded, threads=1),
                lambda: dec.reconstruct_jpeg(encoded, threads=1),
            ]
            for index, call in enumerate(calls):
                with self.subTest(entry=index), self.assertRaises(ValueError):
                    call()
                self.assertEqual((enc.threads, dec.threads), (2, 2))
                self.assertPixels(dec.decode(enc.encode(source, effort=3)), source)
            np.testing.assert_array_equal(destination, np.full_like(source, 71))
            # Bypassing the Python precheck must still reject mismatches in Rust.
            for context, operation, data in (
                    (enc, "transcode_jpeg", fixtures._tiny_jpeg()),
                    (dec, "decode", encoded)):
                with self.assertRaises(rs_jxl.CodecError) as failure:
                    _ffi.execute({"operation": operation, "options": {"threads": 1}},
                                 data, context=context._native)
                self.assertEqual(failure.exception.kind, "invalid_input")
            self.assertPixels(dec.decode(enc.encode(source, effort=3)), source)

    def test_source_effort_report_is_distinct_from_explicit_auto_search(self):
        source = np.arange(12, dtype=np.uint8).reshape(3, 4)
        with rs_jxl.Encoder(threads=2) as enc:
            for encode in (enc.encode, lambda image, **kw: rs_jxl.encode(image, threads=2, **kw)):
                for search in (False, True, False):
                    encoded, metadata = encode(source, effort=3, source_search=search, return_info=True)
                    self.assertReport(encoded, metadata, source, "search" if search else "candidate")
                    if not search:
                        self.assertEqual(metadata["report"]["effective_parameters"]["effort"], 3)
                    self.assertPixels(rs_jxl.decode(encoded, threads=1), source)

    def test_jpeg_entry_parity_reports_and_reconstruction_outlive_contexts(self):
        original = fixtures._tiny_jpeg()
        with rs_jxl.Encoder(threads=2) as enc, rs_jxl.Decoder(threads=2) as dec:
            retained = []
            for encode in (enc.transcode_jpeg, enc.encode,
                           lambda data, **kw: rs_jxl.transcode_jpeg(data, threads=2, **kw),
                           lambda data, **kw: rs_jxl.encode(data, threads=2, **kw)):
                for search in (False, True):
                    encoded, metadata = encode(original, effort=3, source_search=search, return_info=True)
                    report = metadata["report"]
                    self.assertEqual(metadata["bytes"], len(encoded))
                    self.assertEqual(report["route"], "jpeg_search" if search else "jpeg_transcode")
                    self.assertTrue(math.isfinite(report["codec_seconds"]))
                    self.assertGreaterEqual(report["codec_seconds"], 0)
                    self.assertEqual(report["full_encodes"], len(report["trials"]))
                    winner = report["trials"][report["winner_index"]]
                    self.assertEqual(winner["bytes"], len(encoded))
                    self.assertEqual(winner["backend_effort"], report["backend_effort"])
                    with self.assertRaises(rs_jxl.CodecError):
                        enc.transcode_jpeg(b"not JPEG", effort=3)
                    recovered = dec.reconstruct_jpeg(encoded)
                    self.assertEqual(recovered, original)
                    self.assertEqual(rs_jxl.reconstruct_jpeg(encoded, threads=2), original)
                    retained.append((encoded, recovered))
            for encode, data in ((enc.encode, fixtures._rgba()), (enc.transcode_jpeg, original)):
                stream = io.BytesIO()
                result, metadata = encode(data, stream, effort=3, return_info=True)
                self.assertIsNone(result)
                self.assertFalse(stream.closed)
                self.assertEqual(metadata["bytes"], len(stream.getvalue()))
        for encoded, recovered in retained:
            self.assertEqual(recovered, original)
            self.assertEqual(rs_jxl.reconstruct_jpeg(encoded, threads=1), original)

    def test_multiframe_layers_info_and_still_calls_do_not_share_references(self):
        frames = fixtures._frames()[:3]
        durations = [2, 3, 5]
        still = fixtures._rgba(np.uint16, height=2, width=3)
        with rs_jxl.Encoder(threads=2) as enc, rs_jxl.Decoder(threads=2) as dec:
            for strategy in ("crop", "auto", "crop"):
                with self.assertRaises(ValueError):
                    enc.encode_multiframe(frames, durations=[1], effort=3)
                encoded = enc.encode_multiframe(frames, durations=durations, reference="previous",
                                                strategy=strategy, effort=3)
                decoded, metadata = dec.decode(encoded, frame=None, return_info=True)
                self.assertEqual(len(decoded), len(frames))
                self.assertEqual([entry["duration"] for entry in metadata["frames"]], durations)
                for actual, expected in zip(decoded, frames):
                    self.assertPixels(actual, expected)
                self.assertEqual(dec.info(encoded), rs_jxl.info(encoded, threads=2))
                layer, details = dec.decode_layer(encoded, layer=1)
                fresh_layer, fresh_details = rs_jxl.decode_layer(encoded, layer=1, threads=2)
                self.assertPixels(layer, fresh_layer)
                self.assertEqual(details, fresh_details)
                if strategy == "crop":
                    self.assertPixels(layer, frames[1][0:1, 1:2])
                self.assertPixels(dec.decode(enc.encode(still, effort=3)), still)
                self.assertPixels(dec.decode(encoded, frame=2), frames[2])
                fresh = rs_jxl.encode_multiframe(frames, durations=durations, reference="previous",
                                                 strategy=strategy, effort=3, threads=2)
                for actual, expected in zip(dec.decode(fresh, frame=None), frames):
                    self.assertPixels(actual, expected)
            self.assertFalse(np.shares_memory(decoded[0], decoded[1]))
        for actual, expected in zip(decoded, frames):
            self.assertPixels(actual, expected)

    def test_decode_into_identity_exact_layout_and_failure_atomicity(self):
        with rs_jxl.Encoder(threads=2) as enc, rs_jxl.Decoder(threads=2) as dec:
            for dtype in (np.uint8, np.uint16):
                source = fixtures._rgba(dtype, height=3, width=4)
                encoded = enc.encode(source, effort=3)
                for decode_into in (dec.decode_into,
                                    lambda data, dst, **kw: rs_jxl.decode_into(data, dst, threads=2, **kw)):
                    destination = np.empty_like(source)
                    self.assertIs(decode_into(encoded, destination), destination)
                    self.assertPixels(destination, source)
                    result, metadata = decode_into(encoded, destination, return_info=True)
                    self.assertIs(result, destination)
                    self.assertEqual((metadata["height"], metadata["width"]), source.shape[:2])
                    wrong_dtype = np.uint16 if dtype == np.uint8 else np.uint8
                    for invalid in (np.full(source.shape, 71, dtype=wrong_dtype),
                                    np.full((4, 3, 4), 71, dtype=dtype),
                                    np.full((3, 4, 3), 71, dtype=dtype)):
                        before = invalid.copy()
                        with self.assertRaises((TypeError, ValueError)):
                            decode_into(encoded, invalid)
                        np.testing.assert_array_equal(invalid, before)
                    destination.fill(71)
                    with self.assertRaises(rs_jxl.CodecError):
                        decode_into(encoded, destination, max_pixels=1)
                    np.testing.assert_array_equal(destination, np.full_like(source, 71))
                    with self.assertRaises(rs_jxl.CodecError):
                        decode_into(b"\xff\x0a\x00", destination)
                    np.testing.assert_array_equal(destination, np.full_like(source, 71))
                    self.assertIs(decode_into(encoded, destination), destination)
                    self.assertPixels(destination, source)

    def test_explicit_borrow_inputs_has_entry_parity_and_does_not_retain_arrays(self):
        with rs_jxl.Encoder(threads=2) as enc:
            for encode in (enc.encode, lambda image, **kw: rs_jxl.encode(image, threads=2, **kw)):
                for borrow in (False, True):
                    for dtype in (np.uint8, np.uint16):
                        source = fixtures._rgba(dtype).copy()
                        expected = source.copy()
                        reference = weakref.ref(source)
                        encoded, metadata = encode(source, effort=3, borrow_inputs=borrow, return_info=True)
                        self.assertReport(encoded, metadata, expected)
                        source.fill(0)
                        del source
                        gc.collect()
                        self.assertIsNone(reference(), "a completed context call retained its input")
                        self.assertPixels(rs_jxl.decode(encoded, threads=2), expected)
            original = fixtures._rgba(height=3, width=4)
            png = rs_jxl.decode_to_png(enc.encode(original, effort=3))
            jpeg = fixtures._tiny_jpeg()
            for borrow in (False, True):
                png_buffer = bytearray(png)
                encoded = enc.encode(png_buffer, effort=3, borrow_inputs=borrow)
                png_buffer[:] = b"\0" * len(png_buffer)
                self.assertPixels(rs_jxl.decode(encoded), original)
                jpeg_buffer = bytearray(jpeg)
                archived = enc.encode(jpeg_buffer, effort=3, borrow_inputs=borrow)
                jpeg_buffer[:] = b"\0" * len(jpeg_buffer)
                self.assertEqual(rs_jxl.reconstruct_jpeg(archived), jpeg)

    def test_concurrent_calls_on_shared_and_separate_contexts(self):
        workers = 4
        for shared in (True, False):
            with self.subTest(shared=shared), ExitStack() as stack:
                count = 1 if shared else workers
                encoders = [stack.enter_context(rs_jxl.Encoder(threads=2)) for _ in range(count)]
                decoders = [stack.enter_context(rs_jxl.Decoder(threads=2)) for _ in range(count)]
                start = threading.Barrier(workers)
                # Independent handles must reach the FFI boundary together, not
                # wait on a process-wide lock. Shared handles must never overlap.
                gates = {} if shared else {kind: threading.Barrier(workers)
                                           for kind in ("encoder", "decoder")}
                guard = threading.Lock()
                active = {"encoder": 0, "decoder": 0}
                peak = {"encoder": 0, "decoder": 0}

                class ObservedLibrary:
                    def __init__(self, library, kind):
                        self.library = library
                        self.kind = kind

                    def __getattr__(self, name):
                        return getattr(self.library, name)

                    def rs_jxl_context_execute(self, *args):
                        with guard:
                            active[self.kind] += 1
                            peak[self.kind] = max(peak[self.kind], active[self.kind])
                        try:
                            if not shared:
                                gates[self.kind].wait(timeout=20)
                            else:
                                # Yield the GIL to expose accidental overlap;
                                # elapsed time is not a test acceptance criterion.
                                threading.Event().wait(0.02)
                            return self.library.rs_jxl_context_execute(*args)
                        finally:
                            with guard:
                                active[self.kind] -= 1

                for kind, contexts in (("encoder", encoders), ("decoder", decoders)):
                    for context in contexts:
                        stack.enter_context(patch.object(context._native, "_lib",
                                                         ObservedLibrary(context._native._lib, kind)))

                def worker(index):
                    enc = encoders[0 if shared else index]
                    dec = decoders[0 if shared else index]
                    source = fixtures._rgba(np.uint16, height=3 + index, width=5)
                    source[..., 0] += index
                    start.wait(timeout=20)
                    outputs = []
                    for _ in range(2):
                        encoded = enc.encode(source, effort=3)
                        outputs.append(dec.decode(encoded))
                    return source, outputs

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(worker, index) for index in range(workers)]
                    for future in futures:
                        source, outputs = future.result(timeout=60)
                        for actual in outputs:
                            self.assertPixels(actual, source)
                        self.assertFalse(np.shares_memory(*outputs))
                expected = 1 if shared else workers
                self.assertEqual(peak, {"encoder": expected, "decoder": expected})


class ErrorKindTests(unittest.TestCase):
    def test_all_structured_ffi_error_kinds_survive_transport_and_free_result(self):
        # Encoding/io/internal failures are not reliably inducible with tiny
        # valid inputs, so exercise their native-result transport directly.
        for kind in ("invalid_input", "resource_limit", "encoding", "decoding", "io", "internal"):
            with self.subTest(kind=kind):
                payload = json.dumps({"error": "native failure", "error_kind": kind}).encode("utf-8")
                metadata = _ffi.ffi.new("uint8_t[]", payload)
                lib = Mock()
                result = object()
                lib.rs_jxl_execute.return_value = result
                lib.rs_jxl_result_status.return_value = 1

                def json_buffer(_result, length):
                    length[0] = len(payload)
                    return metadata

                lib.rs_jxl_result_json.side_effect = json_buffer
                with patch.object(_ffi, "library", return_value=lib):
                    with self.assertRaises(rs_jxl.CodecError) as failure:
                        _ffi.execute({"operation": "decode"}, b"invalid")
                self.assertEqual(failure.exception.kind, kind)
                self.assertEqual(str(failure.exception), "native failure")
                lib.rs_jxl_result_free.assert_called_once_with(result)


if __name__ == "__main__":
    unittest.main()
