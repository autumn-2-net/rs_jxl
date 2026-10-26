"""Pixel-only corpus acceptance; official djxl is a verification oracle.

No build/download/discovery, C++ encoder, compressed hashes, resizing, or Pillow
narrowing. Reuses the tested 8/16-bit PNG readers in compare_cpp_port, NOT its
encoder/benchmark entry points. Requires numpy and opencv-python only for runs.
Originals are read-only. Temporary metadata-free copies retain IDAT/PLTE/tRNS
verbatim; compare every original decoded sample, including invisible RGB.

This does not certify metadata preservation, JPEG input, float/U31, or timing
parity. --list and --self-test run no encoder and need no pixel dependencies.
The built-in manifest is a default selection, not an input-size or case-count
limit. Optional --max-* arguments impose caller-selected corpus/file budgets.
Child/total timeouts and 1 MiB per diagnostic stream remain runner safeguards,
not codec input/output limits. The oracle remains scoped to 8/16-bit PNG.
"""

import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from time import monotonic, sleep

from compare_cpp_port import compare_pixels, pixel_libraries, png_payload, png_samples


MAX_LOG_BYTES = 1 << 20
REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "jxl_python"
MANIFEST = {
    "photo": "images/photo_kodak/kodim01.png",
    "icon": "images/icon_64/actions-address-book-new.png",
    "texture": "images/textures_photo/clover.png",
    "game-ui": "images/screenshot_game/20XX_Menu_picture.png",
    "photo16": "testdata-main/external/wesaturate/64px/Nikon-D3-14bit_2020_g1.png",
    "gray": "testdata-main/external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png",
    "rgba": "testdata-main/external/wesaturate/500px/tmshre_riaphotographs_alpha.png",
    "gray-alpha": "testdata-main/jxl/flower/flower_small.ga.png",
}


def png_header(header, max_pixels=None):
    if len(header) < 33 or header[:16] != b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR":
        raise ValueError("expected PNG IHDR")
    width, height, depth, color, compression, filtering, interlace = struct.unpack_from(
        ">IIBBBBB", header, 16
    )
    if not 0 < width <= 0x7fffffff or not 0 < height <= 0x7fffffff:
        raise ValueError(f"invalid PNG dimensions: {width}x{height}")
    if max_pixels is not None and width * height > max_pixels:
        raise ValueError(f"per-image geometry budget exceeded: {width}x{height}")
    if depth not in (8, 16) or color not in (0, 2, 3, 4, 6) or (color == 3 and depth != 8):
        raise ValueError("corpus oracle accepts only 8/16-bit PNG (8-bit palette)")
    if compression or filtering or interlace not in (0, 1):
        raise ValueError("unsupported PNG methods")
    return dict(width=width, height=height, bit_depth=depth, color_type=color,
                interlaced=bool(interlace))


def inside(path, roots):
    return any(path == root or root in path.parents for root in roots)


def check_total(records, max_cases=None, max_total_pixels=None):
    if not records:
        raise ValueError("select at least one image")
    if max_cases is not None and len(records) > max_cases:
        raise ValueError(f"select at most {max_cases} images")
    total = sum(record["width"] * record["height"] for record in records)
    if max_total_pixels is not None and total > max_total_pixels:
        raise ValueError(f"total corpus budget exceeded: {total} pixels")
    return total


def select_sources(args):
    roots = [args.testdata.resolve(strict=True), args.test_img.resolve(strict=True)]
    selected = ([(path.name, path) for path in args.image] if args.image else
                [(key, roots[0] / MANIFEST[key]) for key in (args.case or MANIFEST)])
    if not selected:
        raise ValueError("select at least one image; no recursive discovery")
    if args.max_cases is not None and len(selected) > args.max_cases:
        raise ValueError(f"select at most {args.max_cases} images; no recursive discovery")
    records, seen = [], set()
    for label, path in selected:
        path = path.resolve(strict=True)
        if not inside(path, roots) or not path.is_file():
            raise ValueError(f"source must be a file within an original-image root: {path}")
        if path in seen:
            raise ValueError(f"duplicate input: {path}")
        seen.add(path)
        if args.max_input_bytes is not None and path.stat().st_size > args.max_input_bytes:
            raise ValueError(f"input exceeds {args.max_input_bytes} bytes: {path}")
        with path.open("rb") as stream:
            info = png_header(stream.read(33), args.max_pixels)
        records.append(dict(label=label, source=str(path), **info))
    check_total(records, args.max_cases, args.max_total_pixels)
    return roots, records


def read_bounded(path, limit=None):
    if limit is not None and path.stat().st_size > limit:
        raise ValueError(f"file budget exceeded: {path}")
    with path.open("rb") as stream:
        if limit is None:
            return stream.read()
        # Read actual bytes in chunks; an explicit large cap is not an allocation size.
        data = bytearray()
        while chunk := stream.read(min(1 << 20, limit - len(data) + 1)):
            data.extend(chunk)
            if len(data) > limit:
                raise ValueError(f"file grew beyond budget: {path}")
    return bytes(data)


def run_child(command, environment, directory, timeout, deadline, output=None,
              max_output_bytes=None):
    """Bound time/logs, optionally bound output, and reap each child on failure."""
    start = monotonic()
    end = min(start + timeout, deadline)
    if start >= end:
        raise TimeoutError("total corpus deadline exhausted")
    if output is not None and output.exists():
        raise ValueError(f"stale child output: {output}")
    stdout, stderr = directory / "stdout.log", directory / "stderr.log"
    with stdout.open("wb") as out, stderr.open("wb") as err:
        child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err)
        try:
            while child.poll() is None:
                if monotonic() >= end:
                    raise TimeoutError(f"child/total deadline exhausted: {command}")
                for path, limit in [(stdout, MAX_LOG_BYTES), (stderr, MAX_LOG_BYTES),
                                    (output, max_output_bytes)]:
                    if (path is not None and limit is not None and path.exists()
                            and path.stat().st_size > limit):
                        raise ValueError(f"child file budget exceeded: {path}")
                sleep(0.02)
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
    text = (read_bounded(stdout, MAX_LOG_BYTES) + read_bounded(stderr, MAX_LOG_BYTES)).decode(
        "utf-8", errors="replace"
    )
    if child.returncode:
        raise RuntimeError(f"exit={child.returncode}, command={command!r}\n{text[-4000:]}")
    if output is not None:
        if not output.is_file() or output.stat().st_size == 0:
            raise ValueError(f"missing or empty output: {output}")
        if max_output_bytes is not None and output.stat().st_size > max_output_bytes:
            raise ValueError(f"oversized output: {output}")
    finished = monotonic()
    if finished >= end:
        raise TimeoutError(f"child/total deadline exhausted after exit: {command}")
    return finished - start, text


def sample_statistics(pixels):
    _, np = pixel_libraries()
    hidden, low_bits = 0, 0
    for row in pixels.rgba:
        hidden += int(np.count_nonzero((row[:, 3] == 0) & np.any(row[:, :3] != 0, axis=1)))
        if pixels.depth == 16:
            low_bits += int(np.count_nonzero(row % 257))
    return dict(nonzero_rgb_under_zero_alpha=hidden,
                u16_samples_not_representable_as_expanded_u8=low_bits)


def encoder_command(args, binary, source, output):
    command = [str(binary), "encode", str(source), str(output), "--strip-metadata",
               "--threads", str(args.threads)]
    if args.mode != "legacy":
        command += ["--source-effort", str(args.effort)]
        if args.source_sample_limit is not None:
            if args.source_sample_limit < 1:
                raise ValueError("Explicit source sample limit must be positive")
            command += ["--source-sample-limit", str(args.source_sample_limit)]
        if args.mode == "source-search":
            command.append("--source-search")
    return command


def decoder_command(binary, source, output, depth):
    if depth not in (8, 16):
        raise ValueError("integer PNG oracle requires original 8/16-bit precision")
    return [str(binary), str(source), str(output), f"--bits_per_sample={depth}",
            "--num_threads=1", "--quiet"]


def verify(args, roots, records):
    if not args.encoder or not args.decoder or not 1 <= len(args.decoder) <= 2:
        raise ValueError("runs require --encoder and one or two explicit --decoder paths")
    encoder = args.encoder.resolve(strict=True)
    decoders = [path.resolve(strict=True) for path in args.decoder]
    if len(set(decoders)) != len(decoders) or encoder in decoders:
        raise ValueError("encoder and oracle paths must be distinct; no duplicate oracles")
    for decoder in decoders:
        if decoder.stem.lower() != "djxl":
            raise ValueError(f"expected caller-attested official djxl executable: {decoder}")
    temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
    if inside(temp_root, roots):
        raise ValueError("temporary output directory cannot be within source-image roots")
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(("JXL_", "RS_JXL_"))}
    deadline = monotonic() + args.total_timeout
    with tempfile.TemporaryDirectory(prefix="rs-jxl-pixel-corpus-", dir=temp_root) as temporary:
        directory = Path(temporary)
        versions = {}
        for binary in [encoder, *decoders]:
            _, version = run_child([str(binary), "--version"], environment, directory,
                                   min(10, args.timeout), deadline)
            versions[str(binary)] = version.strip()
        print(json.dumps(dict(event="start", mode=args.mode, effort=args.effort,
                              threads=args.threads, cases=len(records),
                              pixels=check_total(records, args.max_cases, args.max_total_pixels),
                              limits=dict(max_cases=args.max_cases, max_pixels=args.max_pixels,
                                          max_total_pixels=args.max_total_pixels,
                                          max_input_bytes=args.max_input_bytes,
                                          max_output_bytes=args.max_output_bytes),
                              binaries=versions, oracle_origin="caller-attested",
                              metadata_acceptance=False)), flush=True)
        for index, record in enumerate(records):
            original = read_bounded(Path(record["source"]), args.max_input_bytes)
            info = png_header(original[:33], args.max_pixels)
            if any(info[key] != record[key] for key in info):
                raise ValueError("source geometry changed after manifest preflight")
            snapshot, png_info = png_payload(original)
            expected = png_samples(original)
            statistics = sample_statistics(expected)
            # The snapshot is lossless metadata removal, never a decoded rewrite.
            source = directory / f"input-{index}.png"
            source.write_bytes(snapshot)
            jxl = directory / f"encoded-{index}.jxl"
            elapsed, diagnostics = run_child(encoder_command(args, encoder, source, jxl),
                                             environment, directory, args.timeout, deadline, jxl,
                                             args.max_output_bytes)
            if args.mode != "legacy":
                marker = f"search={str(args.mode == 'source-search').lower()} "
                if marker not in diagnostics:
                    raise ValueError("Rust diagnostics did not confirm requested source mode")
                if args.mode == "source-search" and "source search: full_encodes=" not in diagnostics:
                    raise ValueError("Rust search did not report completion")
            encoded_bytes = jxl.stat().st_size
            for oracle, decoder in enumerate(decoders):
                output = directory / f"decoded-{index}-{oracle}.png"
                run_child(decoder_command(decoder, jxl, output, expected.depth),
                          environment, directory, args.timeout, deadline, output,
                          args.max_output_bytes)
                decoded = read_bounded(output, args.max_output_bytes)
                png_header(decoded[:33], args.max_pixels)
                compare_pixels(expected, png_samples(decoded))
                output.unlink()
            if monotonic() > deadline:
                raise TimeoutError("total corpus deadline exhausted after verification")
            pixels = expected.width * expected.height
            print(json.dumps(dict(event="case-pass", **record, **statistics,
                                  removed_metadata=png_info["removed_metadata_chunks"],
                                  original_bytes=len(original), jxl_bytes=encoded_bytes,
                                  bpp=8 * encoded_bytes / pixels, cli_seconds=elapsed,
                                  megapixels_per_second=pixels / elapsed / 1e6,
                                  official_decoders=len(decoders))), flush=True)
            source.unlink()
            jxl.unlink()
    print(json.dumps(dict(event="pass", cases=len(records),
                          comparison="all original decoded integer samples, including invisible RGB")), flush=True)


def self_test():
    """Only new orchestration boundaries; existing PNG helpers have their own tests."""
    import unittest
    from io import BytesIO
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    module = sys.modules[__name__]

    class Boundaries(unittest.TestCase):
        def header(self, width=3, height=1, depth=16, color=6):
            return b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR" + struct.pack(
                ">IIBBBBBI", width, height, depth, color, 0, 0, 0, 0
            )

        def test_no_narrowing_in_header(self):
            self.assertEqual(png_header(self.header())["bit_depth"], 16)
            with self.assertRaises(ValueError):
                png_header(self.header(depth=4, color=0))

        def test_hostile_dimensions_and_palette_depth(self):
            for header in [self.header(width=0), self.header(width=0xffffffff),
                           self.header(height=0), self.header(height=0x80000000),
                           self.header(color=3)]:
                with self.assertRaises(ValueError):
                    png_header(header)

        def test_geometry_limits_are_opt_in_and_format_bounds_remain(self):
            for width, height in [(16385, 1), (16384, 16384), (0x7fffffff, 0x7fffffff)]:
                with self.subTest(width=width, height=height):
                    header = self.header(width=width, height=height)
                    info = png_header(header)
                    self.assertEqual((info["width"], info["height"]), (width, height))
                    self.assertEqual(png_header(header, width * height), info)
                    with self.assertRaises(ValueError):
                        png_header(header, width * height - 1)
            with self.assertRaises(ValueError):
                png_header(self.header(width=16384, height=16384), 2_100_000)

        def test_count_and_total_budget(self):
            item = dict(width=1000, height=1000)
            self.assertEqual(check_total([item] * 9), 9_000_000)
            self.assertEqual(check_total([item] * 5, 8, 5_000_000), 5_000_000)
            with self.assertRaises(ValueError):
                check_total([])
            for records in [[], [item] * 6, [dict(width=1, height=1)] * 9]:
                with self.assertRaises(ValueError):
                    check_total(records, 8, 5_000_000)

        def test_containment_is_not_string_prefix(self):
            root = Path("samples").resolve()
            self.assertTrue(inside(root / "a.png", [root]))
            self.assertFalse(inside(root.with_name(root.name + "-other") / "a.png", [root]))

        def test_encoder_commands_select_only_requested_rust_mode(self):
            args = SimpleNamespace(threads=1, effort=7, mode="source-candidate", source_sample_limit=None)
            command = encoder_command(args, Path("rs-jxl"), Path("a.png"), Path("a.jxl"))
            self.assertEqual(command[0:2], ["rs-jxl", "encode"])
            self.assertNotIn("--source-search", command)
            self.assertNotIn("--source-sample-limit", command)
            self.assertNotIn("--bits_per_sample=8", command)
            args.mode = "source-search"
            self.assertIn("--source-search", encoder_command(args, "rs-jxl", "a", "b"))
            args.mode = "legacy"
            self.assertNotIn("--source-effort", encoder_command(args, "rs-jxl", "a", "b"))

        def test_explicit_sample_budget_is_forwarded_without_reduction(self):
            for mode in ("source-candidate", "source-search"):
                args = SimpleNamespace(threads=1, effort=7, mode=mode, source_sample_limit=123)
                command = encoder_command(args, "rs-jxl", "a", "b")
                self.assertEqual(command[command.index("--source-sample-limit") + 1], "123")
                for limit in (0, -1):
                    args.source_sample_limit = limit
                    with self.assertRaises(ValueError):
                        encoder_command(args, "rs-jxl", "a", "b")

        def test_decoder_command_retains_original_precision(self):
            for depth in (8, 16):
                self.assertEqual(decoder_command("djxl", "a.jxl", "a.png", depth),
                                 ["djxl", "a.jxl", "a.png", f"--bits_per_sample={depth}",
                                  "--num_threads=1", "--quiet"])
            for depth in (0, 12, 31, 32):
                with self.assertRaises(ValueError):
                    decoder_command("djxl", "a.jxl", "a.png", depth)

        def test_source_preflight_rejects_duplicate_escape_and_size(self):
            with tempfile.TemporaryDirectory(prefix="rs-jxl-corpus-selftest-") as tmp:
                root = Path(tmp)
                data, images = root / "data", root / "images"
                data.mkdir()
                images.mkdir()
                source = data / "tiny.png"
                source.write_bytes(self.header())
                outside = root / "outside.png"
                outside.write_bytes(self.header())
                args = argument_parser().parse_args([
                    "--testdata", str(data), "--test-img", str(images), "--image", str(source)
                ])
                _, records = select_sources(args)
                self.assertEqual(records[0]["bit_depth"], 16)
                for selection in ([source, source], [data / ".." / outside.name]):
                    args.image = selection
                    with self.assertRaises(ValueError):
                        select_sources(args)
                args.image = [source]
                args.max_input_bytes = 32
                with self.assertRaises(ValueError):
                    select_sources(args)
                self.assertEqual(source.read_bytes(), self.header())
                self.assertEqual(outside.read_bytes(), self.header())

        def test_explicit_images_do_not_inherit_manifest_budgets(self):
            with tempfile.TemporaryDirectory(prefix="rs-jxl-corpus-selftest-") as tmp:
                root = Path(tmp)
                args = argument_parser().parse_args(["--testdata", tmp, "--test-img", tmp])
                args.image = [root / f"wide-{index}.png" for index in range(9)]
                for source in args.image:
                    source.write_bytes(self.header(width=16385, height=129))
                actual_stat = Path.stat

                def reported_stat(path, *args, **kwargs):
                    info = actual_stat(path, *args, **kwargs)
                    if path in selection:
                        return os.stat_result((*info[:6], 17 << 20, *info[7:]))
                    return info

                selection = set(args.image)
                with patch.object(Path, "stat", reported_stat):
                    _, records = select_sources(args)
                    self.assertEqual(len(records), 9)
                    self.assertEqual(check_total(records), 9 * 16385 * 129)
                    for name, limit in [("max_cases", 8), ("max_pixels", 2_100_000),
                                        ("max_total_pixels", 5_000_000),
                                        ("max_input_bytes", 16 << 20)]:
                        with self.subTest(name=name):
                            setattr(args, name, limit)
                            with self.assertRaises(ValueError):
                                select_sources(args)
                            setattr(args, name, None)

        def test_read_budget_checks_stat_and_actual_read(self):
            for reported_size, contents in [(4, b"abcd"), (1, b"abcd")]:
                path = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=reported_size),
                                       open=lambda _: BytesIO(contents))
                with self.assertRaises(ValueError):
                    read_bounded(path, 3)
            path = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=3),
                                   open=lambda _: BytesIO(b"abc"))
            self.assertEqual(read_bounded(path, 3), b"abc")

        def test_read_has_no_default_budget_or_cap_sized_allocation(self):
            path = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=1 << 40),
                                   open=lambda _: BytesIO(b"abc"))
            self.assertEqual(read_bounded(path), b"abc")
            stream = Mock(wraps=BytesIO(b"abc"))
            stream.__enter__ = Mock(return_value=stream)
            stream.__exit__ = Mock(return_value=False)
            path.open = lambda _: stream
            self.assertEqual(read_bounded(path, 1 << 100), b"abc")
            self.assertTrue(all(0 < call.args[0] <= 1 << 20 for call in stream.read.call_args_list))

        def test_expired_deadline_and_stale_output_never_launch(self):
            with tempfile.TemporaryDirectory(prefix="rs-jxl-corpus-selftest-") as tmp:
                root = Path(tmp)
                output = root / "output.png"
                output.write_bytes(b"stale")
                with patch.object(subprocess, "Popen") as launch, \
                        patch.object(module, "monotonic", return_value=10):
                    with self.assertRaises(TimeoutError):
                        run_child(["unused"], {}, root, 1, 10)
                    with self.assertRaises(ValueError):
                        run_child(["unused"], {}, root, 1, 100, output)
                    launch.assert_not_called()
                self.assertEqual(output.read_bytes(), b"stale")

        def test_timeout_kills_and_reaps_child(self):
            with tempfile.TemporaryDirectory(prefix="rs-jxl-corpus-selftest-") as tmp:
                child = Mock(returncode=None)
                child.poll.return_value = None
                with patch.object(subprocess, "Popen", return_value=child), \
                        patch.object(module, "monotonic", side_effect=[10, 11]), \
                        patch.object(module, "sleep") as pause:
                    with self.assertRaises(TimeoutError):
                        run_child(["unused"], {}, Path(tmp), 1, 100)
                child.kill.assert_called_once_with()
                child.wait.assert_called_once_with()
                pause.assert_not_called()

        def test_running_child_log_and_output_growth_kill_and_reap(self):
            for destination in ("stdout", "stderr", "output"):
                with self.subTest(destination=destination), \
                        tempfile.TemporaryDirectory(prefix="rs-jxl-corpus-selftest-") as tmp:
                    root = Path(tmp)
                    output = root / "output.png"
                    child = Mock(returncode=None)
                    child.poll.return_value = None

                    def launch(*args, **kwargs):
                        if destination == "output":
                            output.write_bytes(b"123456789")
                        else:
                            kwargs[destination].write(b"123456789")
                            kwargs[destination].flush()
                        return child

                    with patch.object(subprocess, "Popen", side_effect=launch), \
                            patch.object(module, "monotonic", return_value=10), \
                            patch.object(module, "MAX_LOG_BYTES", 8), \
                            patch.object(module, "sleep") as pause:
                        with self.assertRaises(ValueError):
                            run_child(["unused"], {}, root, 1, 100, output, max_output_bytes=8)
                    child.kill.assert_called_once_with()
                    child.wait.assert_called_once_with()
                    pause.assert_not_called()

        def test_fast_exit_checks_status_and_output_even_without_poll_iteration(self):
            cases = [(0, None, ValueError), (0, b"", ValueError),
                     (0, b"123456789", ValueError), (2, b"valid", RuntimeError),
                     (0, b"valid", None)]
            for code, contents, error in cases:
                with self.subTest(code=code, contents=contents), \
                        tempfile.TemporaryDirectory(prefix="rs-jxl-corpus-selftest-") as tmp:
                    root = Path(tmp)
                    output = root / "output.png"
                    child = Mock(returncode=code)
                    child.poll.return_value = code

                    def launch(*args, **kwargs):
                        if contents is not None:
                            output.write_bytes(contents)
                        return child

                    with patch.object(subprocess, "Popen", side_effect=launch), \
                            patch.object(module, "monotonic", side_effect=[10, 10.25]):
                        if error:
                            with self.assertRaises(error):
                                run_child(["unused"], {}, root, 1, 100, output, max_output_bytes=8)
                        else:
                            self.assertEqual(run_child(["unused"], {}, root, 1, 100, output,
                                                       max_output_bytes=8), (0.25, ""))
                    child.kill.assert_not_called()
                    child.wait.assert_called_once_with()

        def test_child_output_has_no_default_size_budget(self):
            for running in (False, True):
                with self.subTest(running=running), \
                        tempfile.TemporaryDirectory(prefix="rs-jxl-corpus-selftest-") as tmp:
                    output = Mock()
                    output.exists.side_effect = [False, True]
                    output.is_file.return_value = True
                    output.stat.return_value = SimpleNamespace(st_size=33 << 20)
                    child = Mock(returncode=0)
                    child.poll.side_effect = [None, 0, 0] if running else [0, 0]
                    times = [10, 10, 10.25] if running else [10, 10.25]
                    with patch.object(subprocess, "Popen", return_value=child), \
                            patch.object(module, "monotonic", side_effect=times), \
                            patch.object(module, "sleep"):
                        self.assertEqual(run_child(["unused"], {}, Path(tmp), 1, 100, output),
                                         (0.25, ""))
                    child.kill.assert_not_called()
                    child.wait.assert_called_once_with()

        def test_fast_exit_cannot_bypass_log_or_deadline_budget(self):
            for oversized_log in (False, True):
                with self.subTest(oversized_log=oversized_log), \
                        tempfile.TemporaryDirectory(prefix="rs-jxl-corpus-selftest-") as tmp:
                    child = Mock(returncode=0)
                    child.poll.return_value = 0

                    def launch(*args, **kwargs):
                        if oversized_log:
                            kwargs["stdout"].write(b"123456789")
                            kwargs["stdout"].flush()
                        return child

                    with patch.object(subprocess, "Popen", side_effect=launch), \
                            patch.object(module, "monotonic", side_effect=[10, 12]), \
                            patch.object(module, "MAX_LOG_BYTES", 8):
                        with self.assertRaises(ValueError if oversized_log else TimeoutError):
                            run_child(["unused"], {}, Path(tmp), 1, 100)
                    child.kill.assert_not_called()
                    child.wait.assert_called_once_with()

    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Boundaries))
    return 0 if result.wasSuccessful() else 1


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testdata", type=Path, default=REFERENCE_ROOT / "jxl_testdata")
    parser.add_argument("--test-img", type=Path, default=REFERENCE_ROOT / "test_img")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--case", action="append", choices=MANIFEST)
    select.add_argument("--image", action="append", type=Path, help="Explicit original PNG within either root")
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--decoder", action="append", type=Path, help="Caller-attested official djxl; at most two")
    parser.add_argument("--threads", type=int, choices=range(1, 257), default=1)
    parser.add_argument("--effort", type=int, choices=range(1, 11), default=7)
    parser.add_argument("--mode", choices=("legacy", "source-candidate", "source-search"), default="source-candidate")
    parser.add_argument("--source-sample-limit", type=int, default=None,
                        help="Optional positive source MA sample cap; omitted uses the CLI default")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Optional positive selected-image count cap; default: no cap")
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="Optional positive per-image pixel cap; default: no cap")
    parser.add_argument("--max-total-pixels", type=int, default=None,
                        help="Optional positive total input pixel cap; default: no cap")
    parser.add_argument("--max-input-bytes", type=int, default=None,
                        help="Optional positive per-input byte cap; default: no cap")
    parser.add_argument("--max-output-bytes", type=int, default=None,
                        help="Optional positive per-encoded/decoded-file byte cap; default: no cap")
    parser.add_argument("--timeout", type=float, default=120, help="Positive per-child timeout in seconds")
    parser.add_argument("--total-timeout", type=float, default=600, help="Positive total timeout in seconds")
    parser.add_argument("--list", action="store_true", help="Read at most 33 bytes per file; no sample decode/encoding")
    parser.add_argument("--self-test", action="store_true", help="Test orchestration helpers only")
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.source_sample_limit is not None:
        if args.source_sample_limit < 1:
            parser.error("--source-sample-limit must be positive")
        if args.mode == "legacy":
            parser.error("--source-sample-limit requires a source mode")
    for name in ("max_cases", "max_pixels", "max_total_pixels", "max_input_bytes", "max_output_bytes"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not all(math.isfinite(value) and value > 0
               for value in (args.timeout, args.total_timeout)):
        parser.error("timeouts must be finite and positive")
    try:
        roots, records = select_sources(args)
        if args.list:
            print(json.dumps(dict(cases=records, total_pixels=check_total(
                records, args.max_cases, args.max_total_pixels)), indent=2))
        else:
            verify(args, roots, records)
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps(dict(event="fail", error=str(error))), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
