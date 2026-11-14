import contextlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import benchmark_portfolio as bench


def chunk(kind, body):
    return (struct.pack(">I", len(body)) + kind + body
            + struct.pack(">I", zlib.crc32(kind + body)))


def png(width=2, depth=16, extra=b""):
    header = struct.pack(">IIBBBBB", width, 1, depth, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + extra
            + chunk(b"IDAT", zlib.compress(b"\0" + b"\0" * width * 4 * (depth // 8)))
            + chunk(b"IEND", b""))


class PortfolioTests(unittest.TestCase):
    def fixture(self, directory, inputs=True):
        root = Path(directory)
        for name in ("encoder.exe", "djxl.exe"):
            (root / name).write_bytes(name.encode())
        cases = []
        for index, width in enumerate((2, 8)):
            source = root / f"image{index}.png"
            source.write_bytes(png(width))
            cases.append(dict(path=str(source), sha256=bench.file_sha256(source),
                              dataset="test", bit_depth=16))
        if not inputs:
            cases = [dict(original=row["path"], sha256=row["sha256"]) for row in cases]
        manifest = root / "source.json"
        manifest.write_text(json.dumps({"inputs" if inputs else "cases": cases}), encoding="utf-8")
        return SimpleNamespace(root=root, manifest=manifest, cases=cases, calls=[], artifacts=[],
                               argv=["--manifest", str(manifest), "--out", str(root / "out"),
                                     "--encoder", str(root / "encoder.exe"),
                                     "--djxl", str(root / "djxl.exe")])

    def edit_manifest(self, fixture, rows):
        fixture.manifest.write_text(json.dumps(dict(inputs=rows)), encoding="utf-8")

    def receipt(self, width):
        size = 10 if width == 2 else 20
        return dict(schema=1, effort=9, threads=8, width=width, height=1, bytes=size, bpp=8 * size / width,
                    seconds=2.0, mpix_per_second=width / 1e6 / 2,
                    baseline=dict(bytes=size + 5, seconds=1.0, bpp=8 * (size + 5) / width),
                    scout_seconds=0.25, probe_encodes=1, probe_pixels=2, full_encodes=2,
                    winner="candidate", candidates=[dict(name="baseline", family="Baseline",
                    decision="full-rejected", priority=100, probe_bytes=[None], probe_seconds=0,
                    full_bytes=size + 5, full_seconds=1.0, error=None),
                    dict(name="candidate", family="source",
                    decision="finalist", priority=1, probe_bytes=[5], probe_seconds=0.2,
                    full_bytes=size, full_seconds=0.5, error=None)] + [
                    dict(name=decision, family="source", decision=decision, priority=0,
                         probe_bytes=[None, None], probe_seconds=0,
                         full_bytes=None, full_seconds=None, error=None)
                    for decision in ("queued", "ineligible", "unconfirmed")], regions=[], features={})

    def invoke(self, fixture, extra=(), effect=None):
        args = bench.arguments(fixture.argv + list(extra))
        current = {}
        def prepare(source, directory):
            data = source.read_bytes()
            _, info = bench.png_framing(data)
            snapshot = directory / "snapshot.png"
            snapshot.write_bytes(data)
            fixture.artifacts.append(snapshot)
            current["width"] = info["width"]
            expected = SimpleNamespace(width=info["width"], height=1, depth=16)
            return dict(kind="png", input_sha256=bench.file_sha256(source),
                        payload_sha256=bench.file_sha256(snapshot), width=info["width"],
                        height=1, depth=16), snapshot, expected
        def process(command, environment, timeout):
            fixture.calls.append(command)
            result = dict(returncode=0, stdout="", stderr="", command=command, wall_seconds=0.01)
            if "--output" in command:
                output = Path(command[command.index("--output") + 1])
                measurement = self.receipt(current["width"])
                measurement.update(effort=args.effort, threads=args.threads)
                output.write_bytes(b"x" * measurement["bytes"])
                result["stdout"] = json.dumps(measurement)
                self.assertEqual(command[command.index("--effort") + 1], str(args.effort))
                self.assertEqual(command[command.index("--max-probe-encodes") + 1], "48")
            else:
                output = Path(command[2])
                self.assertIn("--bits_per_sample=16", command)
                output.write_bytes(b"decoded")
            fixture.artifacts.append(output)
            if effect:
                effect(command, result)
            return result
        with patch.object(bench, "prepare", side_effect=prepare), \
                patch.object(bench, "run_process", side_effect=process), \
                patch.object(bench, "png_samples", return_value=object()), \
                patch.object(bench, "compare_pixels") as compare:
            result = bench.benchmark(args)
        fixture.comparisons = compare.call_count
        self.assertTrue(all(not p.exists() for p in fixture.artifacts))
        return result

    def test_atomic_write_retries_sharing_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            replace = bench.os.replace
            calls = []
            def sharing_once(source, target):
                calls.append(target)
                if len(calls) == 1:
                    raise PermissionError("reader holds destination")
                return replace(source, target)
            with patch.object(bench.os, "replace", side_effect=sharing_once), patch.object(bench, "sleep") as wait:
                bench.atomic_write(path, {"ok": True})
            self.assertEqual(bench.loads(path.read_text()), {"ok": True})
            wait.assert_called_once_with(0.01)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_locked_reports_do_not_abort_checkpointing(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            write = bench.atomic_write
            def locked(path, value):
                if path.name in ("progress.json", "summary.json"):
                    raise PermissionError("report reader")
                return write(path, value)
            with patch.object(bench, "atomic_write", side_effect=locked), contextlib.redirect_stderr(io.StringIO()):
                result = self.invoke(f, ("--run",))
            self.assertEqual(result["statuses"], {"passed": 2})
            self.assertFalse((f.root / "out/run.lock").exists())
            self.assertEqual(len((f.root / "out/trials.jsonl").read_text().splitlines()), 2)
            f.calls.clear()
            self.invoke(f, ("--resume", "--run"))
            self.assertEqual(f.calls, [])
            self.assertEqual(bench.loads((f.root / "out/progress.json").read_text())["phase"], "complete")

    def test_critical_manifest_write_failure_stays_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            with patch.object(bench, "atomic_write", side_effect=PermissionError("manifest write")):
                with self.assertRaises(PermissionError):
                    self.invoke(f, ("--run",))
            self.assertFalse((f.root / "out/run.lock").exists())
            self.assertEqual(f.calls, [])

    def test_journal_write_failure_stays_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            open_path = Path.open
            def fail_journal(path, mode="r", *args, **kwargs):
                if path.name == "trials.jsonl" and mode == "a":
                    raise OSError("journal disk failure")
                return open_path(path, mode, *args, **kwargs)
            with patch.object(Path, "open", fail_journal):
                with self.assertRaisesRegex(OSError, "journal disk failure"):
                    self.invoke(f, ("--run",))
            self.assertEqual((f.root / "out/trials.jsonl").read_text(), "")
            self.assertFalse((f.root / "out/run.lock").exists())
            self.assertEqual(bench.loads((f.root / "out/progress.json").read_text())["phase"], "failed")

    def test_report_serialization_failure_stays_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                bench.write_report(Path(directory) / "summary.json", {"bad": float("nan")})

    def test_stop_finishes_current_image_then_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            def stop(command, _):
                if "--output" in command:
                    (f.root / "out/STOP").touch()
            result = self.invoke(f, ("--run",), effect=stop)
            self.assertEqual(result["completed"], 1)
            self.assertEqual(result["statuses"], {"passed": 1})
            self.assertEqual(f.comparisons, 1)
            progress = bench.loads((f.root / "out/progress.json").read_text())
            self.assertEqual(progress["phase"], "stopped")
            self.assertIsNone(progress["active"])
            self.assertEqual(progress["pid"], os.getpid())
            f.calls.clear()
            self.invoke(f, ("--resume", "--run"))
            self.assertEqual(f.calls, [])
            (f.root / "out/STOP").unlink()
            result = self.invoke(f, ("--resume", "--run"))
            self.assertEqual(result["completed"], 2)
            self.assertEqual(f.comparisons, 1)

    def test_prepare_only_cases_and_inputs(self):
        for inputs in (False, True):
            with self.subTest(inputs=inputs), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(directory, inputs)
                with patch.object(bench, "run_process") as process, patch.object(bench, "prepare") as prepare:
                    result = bench.benchmark(bench.arguments(f.argv))
                process.assert_not_called()
                prepare.assert_not_called()
                self.assertEqual(result["mode"], "prepare-only")
                self.assertEqual(result["pending"], 2)
                frozen = bench.loads((f.root / "out/manifest.json").read_text())
                self.assertEqual(frozen["cases"][0]["bit_depth"], 16)
                self.assertEqual((f.root / "out/trials.jsonl").read_text(), "")

    def test_filter_aliases_apng_bad_png_and_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            alias = f.root / "alias.png"
            alias.write_bytes(png())
            bad, animated = f.root / "bad.png", f.root / "animated.png"
            bad.write_bytes(b"broken")
            animated.write_bytes(png(extra=chunk(b"acTL", struct.pack(">II", 1, 0))))
            rows = f.cases + [dict(path=str(p)) for p in
                             (alias, bad, animated, f.root / "photo.jpg", f.root / "float.pfm")]
            self.edit_manifest(f, rows)
            result = self.invoke(f)
            self.assertEqual(result["duplicate_aliases"], 1)
            self.assertEqual(result["preparation_failures"], 2)
            self.assertEqual(result["excluded"], 2)
            population = bench.inventory(f.manifest, 1)
            self.assertEqual(len(population["cases"]), 1)
            self.assertEqual(len(population["excluded"]), len(rows) - 1)
            self.assertIn("JPEG", population["excluded"][-2]["reason"])
            self.assertEqual(f.calls, [])

    def test_run_weighted_bpp_precision_cleanup_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            self.invoke(f)
            result = self.invoke(f, ("--resume", "--run"))
            self.assertEqual(result["statuses"], {"passed": 2})
            totals = result["successful"]
            self.assertEqual(totals["arithmetic_mean_bpp"], 30)
            self.assertEqual(totals["aggregate_weighted_bpp"], 24)
            self.assertEqual(totals["baseline_bytes"], 40)
            self.assertEqual(totals["baseline_arithmetic_mean_bpp"], 42.5)
            self.assertEqual(totals["baseline_aggregate_weighted_bpp"], 32)
            self.assertEqual(totals["byte_change_percent"], -25)
            self.assertAlmostEqual(totals["overall_mpix_per_second"], 10 / 1e6 / 4)
            self.assertEqual(totals["native_seconds"], 4)
            self.assertEqual(totals["scout_seconds"], 0.5)
            self.assertEqual(totals["full_encode_seconds"], 3)
            self.assertEqual(f.comparisons, 2)
            calls = len(f.calls)
            self.invoke(f, ("--resume", "--run"))
            self.assertEqual(len(f.calls), calls)

    def test_resume_rejects_config_manifest_executable_input_and_inventory_changes(self):
        for change in ("config", "candidates", "serial_probes", "preparation_cache_mib",
                       "no_finalist_bound", "manifest", "encoder", "decoder", "input", "inventory"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(directory)
                self.invoke(f)
                extra = ["--resume"]
                if change == "config":
                    extra += ["--threads", "2"]
                elif change == "candidates":
                    extra += ["--candidates", "prev2,source"]
                elif change == "preparation_cache_mib":
                    extra += ["--preparation-cache-mib", "0"]
                elif change in ("serial_probes", "no_finalist_bound"):
                    extra += ["--" + change.replace("_", "-")]
                elif change == "manifest":
                    f.manifest.write_text(f.manifest.read_text() + " ")
                elif change in ("encoder", "decoder"):
                    (f.root / ("encoder.exe" if change == "encoder" else "djxl.exe")).write_bytes(b"changed")
                elif change == "input":
                    Path(f.cases[0]["path"]).write_bytes(png(4))
                else:
                    (f.root / "out/manifest.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "Resume requires"):
                    self.invoke(f, extra)

    def test_resume_rejects_torn_duplicate_tampered_and_semantically_corrupt_records(self):
        for change in ("torn", "duplicate", "checksum", "receipt", "run_id", "status"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(directory)
                self.invoke(f, ("--run",))
                journal = f.root / "out/trials.jsonl"
                lines = journal.read_text().splitlines(keepends=True)
                if change == "torn":
                    journal.write_text("".join(lines).rstrip("\n"))
                elif change == "duplicate":
                    journal.write_text(lines[0] + lines[0])
                else:
                    row = json.loads(lines[0])
                    if change == "checksum":
                        row["case"]["sha256"] = "0" * 64
                    else:
                        if change == "receipt":
                            row["measurement"]["bpp"] = 0
                        elif change == "run_id":
                            row["run_id"] = "wrong"
                        else:
                            row["status"] = "unverified"
                        row.pop("checksum")
                        row["checksum"] = bench.digest(row)
                    journal.write_text(bench.canonical(row) + "\n" + lines[1])
                with self.assertRaises(ValueError):
                    self.invoke(f, ("--run", "--resume"))

    def test_failures_are_checkpointed_without_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            def fail(command, result):
                if "--output" in command:
                    result.update(returncode=1, stderr="native failure")
            result = self.invoke(f, ("--run",), effect=fail)
            self.assertEqual(result["statuses"], {"failed": 2})
            self.assertEqual(result["successful"]["count"], 0)
            calls = len(f.calls)
            self.invoke(f, ("--run", "--resume"))
            self.assertEqual(len(f.calls), calls)

    def test_skip_verification_is_explicit_and_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            f.argv = f.argv[:-2]
            result = self.invoke(f, ("--run", "--skip-verify"))
            self.assertEqual(result["statuses"], {"unverified": 2})
            self.assertEqual(len(f.calls), 2)
            self.assertEqual(f.comparisons, 0)

    def test_invalid_receipt_and_source_mutation_are_failures(self):
        for change in ("receipt", "input"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(directory)
                def mutate(command, result):
                    if "--output" not in command:
                        return
                    if change == "receipt":
                        data = json.loads(result["stdout"])
                        data["bytes"] += 1
                        result["stdout"] = json.dumps(data)
                    else:
                        for case in f.cases:
                            Path(case["path"]).write_bytes(b"mutated")
                result = self.invoke(f, ("--run",), effect=mutate)
                self.assertEqual(result["statuses"], {"failed": 2})
                self.assertEqual(result["successful"]["count"], 0)
                if change == "receipt":
                    self.assertEqual(result["native_time_unavailable_calls"], 2)
                else:
                    self.assertEqual(result["observed_native"]["count"], 1)

    def test_verification_failures_keep_native_elapsed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            def fail(command, result):
                if "--output" not in command:
                    result.update(returncode=1, stderr="decoder failed")
            result = self.invoke(f, ("--run",), effect=fail)
            self.assertEqual(result["statuses"], {"failed": 2})
            self.assertEqual(result["observed_native"]["native_seconds"], 4)
            self.assertEqual(result["successful"]["native_seconds"], 0)

    def test_large_cases_and_missing_inputs_are_accounted_for(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            large = f.root / "large.png"
            # Only framing is inspected in prepare mode; no large raster allocated.
            large.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(
                ">IIBBBBB", 8_000_001, 1, 16, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(b"\0")) + chunk(b"IEND", b""))
            missing = f.root / "missing.png"
            self.edit_manifest(f, [dict(path=str(large)), dict(path=str(missing))])
            result = self.invoke(f)
            self.assertEqual(result["large_cases"]["planned"], 1)
            self.assertEqual(result["preparation_failures"], 1)
            self.assertEqual(f.calls, [])

    def test_interrupted_run_resumes_only_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            original = bench.trial
            def interrupt(index, *args):
                if index == 1:
                    raise KeyboardInterrupt()
                return original(index, *args)
            with patch.object(bench, "trial", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.invoke(f, ("--run",))
            progress = json.loads((f.root / "out/progress.json").read_text())
            self.assertEqual(progress["phase"], "interrupted")
            self.assertEqual(progress["completed"], 1)
            self.assertEqual(progress["active"], 1)
            result = self.invoke(f, ("--run", "--resume"))
            self.assertEqual(result["completed"], 2)
            self.assertEqual(len(f.calls), 4)

    def test_cli_guards_and_strict_json(self):
        base = ["--manifest", "input.json", "--out", "output"]
        self.assertIsNone(bench.arguments(base).limit)
        for extra in (("--run",), ("--limit", "0"), ("--timeout", "nan"),
                      ("--preparation-cache-mib", "-1"),
                      ("--run", "--dry-run"), ("--effort", "7"), ("--probe-side", "31"),
                      ("--probe-side", "513"), ("--candidates", ""),
                      ("--candidates", "prev2,"), ("--candidates", "prev2,prev2")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.arguments(base + list(extra))
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                bench.loads(text)

    def test_candidate_filter_and_native_efforts(self):
        for effort in (8, 9, 10):
            with self.subTest(effort=effort), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(directory)
                self.invoke(f, ("--run", "--effort", str(effort), "--candidates", " prev2, source "))
                for command in f.calls:
                    if "--output" in command:
                        self.assertEqual(command[command.index("--candidates") + 1], "prev2,source")
                run = bench.loads((f.root / "out/run.json").read_text())
                self.assertEqual(run["frozen"]["identity"]["config"]["candidates"], "prev2,source")

    @unittest.skipUnless(os.environ.get("RS_JXL_PORTFOLIO_LAB") and os.environ.get("RS_JXL_PORTFOLIO_DJXL"),
                         "requires explicit native portfolio executable and official djxl")
    def test_real_native_and_official_decoder_smoke(self):
        with tempfile.TemporaryDirectory(prefix="portfolio-oracle-") as directory:
            root = Path(directory)
            sources = []
            for depth in (8, 16):
                width, height = 80, 72
                factor = 1 if depth == 8 else 257
                raw = bytearray()
                for y in range(height):
                    raw.append(0)
                    for x in range(width):
                        values = [(x * 3 % 256) * factor, (y * 3 % 256) * factor,
                                  ((x + y) % 256) * factor, (0 if x % 3 == 0 else 255) * factor]
                        raw.extend(bytes(values) if depth == 8 else struct.pack(">4H", *values))
                header = struct.pack(">IIBBBBB", width, height, depth, 6, 0, 0, 0)
                path = root / f"rgba{depth}.png"
                path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                                 + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
                sources.append(dict(path=str(path)))
            manifest = root / "input.json"
            manifest.write_text(json.dumps(dict(cases=sources)), encoding="utf-8")
            args = bench.arguments([
                "--manifest", str(manifest), "--out", str(root / "result"), "--run",
                "--encoder", os.environ["RS_JXL_PORTFOLIO_LAB"],
                "--djxl", os.environ["RS_JXL_PORTFOLIO_DJXL"],
                "--threads", "2", "--probe-side", "32", "--max-probe-encodes", "12",
                "--candidates", "prev1,offset-distribution,shallow-full",
            ])
            summary = bench.benchmark(args)
            journal = (root / "result/trials.jsonl").read_text(encoding="utf-8")
            self.assertEqual(summary["statuses"], {"passed": 2}, journal)
            self.assertEqual(summary["candidate_errors"], 0, journal)
            self.assertLessEqual(summary["successful"]["bytes"], summary["successful"]["baseline_bytes"])

    def test_null_probe_bytes_are_valid_but_other_invalid_values_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            args = bench.arguments(f.argv)
            case = dict(width=2, height=1, pixels=2)
            receipt = self.receipt(2)
            bench.validate_receipt(receipt, case, args)
            for invalid in (-1, "unknown", False, 1.5):
                receipt["candidates"][-1]["probe_bytes"] = [invalid]
                with self.assertRaises(ValueError):
                    bench.validate_receipt(receipt, case, args)

    def test_receipt_effort_and_threads_must_match_when_present(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            args = bench.arguments(f.argv)
            case = dict(width=2, height=1, pixels=2)
            for key in ("effort", "threads"):
                for invalid in (getattr(args, key) + 1, float(getattr(args, key)), True, None):
                    with self.subTest(key=key, invalid=invalid):
                        receipt = self.receipt(2)
                        receipt[key] = invalid
                        with self.assertRaisesRegex(ValueError, key):
                            bench.validate_receipt(receipt, case, args)
            receipt = self.receipt(2)
            del receipt["effort"], receipt["threads"]
            bench.validate_receipt(receipt, case, args)

    def test_performance_controls_are_frozen_and_forwarded(self):
        for extra in ((), ("--serial-probes", "--preparation-cache-mib", "0", "--no-finalist-bound")):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(directory)
                args = bench.arguments(f.argv + list(extra))
                self.assertFalse(args.run)
                self.invoke(f, ("--run",) + extra)
                config = bench.loads((f.root / "out/run.json").read_text())["frozen"]["identity"]["config"]
                for key in ("serial_probes", "preparation_cache_mib", "no_finalist_bound"):
                    self.assertEqual(config[key], getattr(args, key))
                for command in f.calls:
                    if "--output" not in command:
                        continue
                    self.assertEqual(command[command.index("--preparation-cache-mib") + 1],
                                     str(args.preparation_cache_mib))
                    self.assertEqual("--serial-probes" in command, args.serial_probes)
                    self.assertEqual("--no-finalist-bound" in command, args.no_finalist_bound)

    def test_optional_performance_receipt_fields_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            args = bench.arguments(f.argv)
            case = dict(width=2, height=1, pixels=2)
            receipt = self.receipt(2)
            bench.validate_receipt(receipt, case, args)
            receipt.update(parallel_probes=True, preparation_cache_bytes=128 << 20,
                           bound_finalists=True, probe_workers=1,
                           preparation_reuse=dict(hits=2, misses=3, duplicate_builds=1,
                                                  bypasses=0, entries=2, retained_bytes=16, peak_bytes=32))
            # Concurrent elapsed probe work is not scout wall time.
            receipt["candidates"][1]["probe_seconds"] = 0.5
            bench.validate_receipt(receipt, case, args)
            for key, invalids in {
                "parallel_probes": (False, 1, None),
                "bound_finalists": (False, 1, None),
                "preparation_cache_bytes": (0, float(128 << 20), True, None),
                "probe_workers": (-1, True, 1.5, 2, None),
                "preparation_reuse": (None, [], {}),
            }.items():
                for invalid in invalids:
                    with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                        bench.validate_receipt(dict(receipt, **{key: invalid}), case, args)
            for key in receipt["preparation_reuse"]:
                for invalid in (-1, True, 1.5, None):
                    with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                        reuse = dict(receipt["preparation_reuse"], **{key: invalid})
                        bench.validate_receipt(dict(receipt, preparation_reuse=reuse), case, args)
            reuse = dict(receipt["preparation_reuse"], retained_bytes=33)
            with self.assertRaises(ValueError):
                bench.validate_receipt(dict(receipt, preparation_reuse=reuse), case, args)
            serial = bench.arguments(f.argv + ["--serial-probes"])
            parallel_receipt = dict(receipt, parallel_probes=False, probe_encodes=2, probe_workers=2)
            with self.assertRaisesRegex(ValueError, "probe_workers"):
                bench.validate_receipt(parallel_receipt, case, serial)
            bench.validate_receipt(dict(receipt, probe_encodes=0, probe_workers=0), case, args)

    def test_full_bounded_contract_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            args = bench.arguments(f.argv)
            case = dict(width=2, height=1, pixels=2)
            receipt = self.receipt(2)
            bounded = receipt["candidates"][-1]
            bounded.update(decision="full-bounded", full_lower_bound=receipt["bytes"],
                           full_seconds=0.1)
            receipt["full_encodes"] = 3
            for lower_bound in (receipt["bytes"], receipt["baseline"]["bytes"] + 100):
                bounded["full_lower_bound"] = lower_bound
                bench.validate_receipt(receipt, case, args)
            for change in (dict(full_lower_bound=None), dict(full_lower_bound=-1),
                           dict(full_lower_bound=0), dict(full_lower_bound=receipt["bytes"] - 1),
                           dict(full_lower_bound=True), dict(full_lower_bound=10.0),
                           dict(full_lower_bound="10"), dict(full_bytes=10),
                           dict(full_seconds=None), dict(full_seconds=-1),
                           dict(error="bounded"), dict(decision="full-error"),
                           dict(name=receipt["winner"])):
                with self.subTest(change=change):
                    original = dict(bounded)
                    bounded.update(change)
                    with self.assertRaises(ValueError):
                        bench.validate_receipt(receipt, case, args)
                    bounded.clear()
                    bounded.update(original)
            del bounded["full_lower_bound"]
            with self.assertRaises(ValueError):
                bench.validate_receipt(receipt, case, args)
            bounded["full_lower_bound"] = receipt["bytes"]
            with self.assertRaises(ValueError):
                bench.validate_receipt(receipt, case,
                                       bench.arguments(f.argv + ["--no-finalist-bound"]))

    def test_full_bounded_is_not_failure_and_resume_rechecks_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            def bound(command, result):
                if "--output" not in command:
                    return
                receipt = json.loads(result["stdout"])
                receipt["candidates"][-1].update(decision="full-bounded",
                                                full_lower_bound=receipt["bytes"] + 100,
                                                full_seconds=0.1)
                receipt["full_encodes"] = 3
                result["stdout"] = json.dumps(receipt)
            summary = self.invoke(f, ("--run",), effect=bound)
            self.assertEqual(summary["statuses"], {"passed": 2})
            self.assertEqual(summary["candidate_errors"], 0)
            self.assertEqual(summary["candidate_decision_counts"]["full-bounded"], 2)
            self.assertAlmostEqual(summary["successful"]["full_encode_seconds"], 3.2)
            self.invoke(f, ("--resume", "--run"))
            journal = f.root / "out/trials.jsonl"
            lines = journal.read_text().splitlines(keepends=True)
            row = bench.loads(lines[0])
            row["measurement"]["candidates"][-1]["full_lower_bound"] = row["measurement"]["bytes"] - 1
            row["encoder"]["stdout"] = bench.canonical(row["measurement"])
            row.pop("checksum")
            row["checksum"] = bench.digest(row)
            journal.write_text(bench.canonical(row) + "\n" + "".join(lines[1:]))
            with self.assertRaisesRegex(ValueError, "full-bounded"):
                self.invoke(f, ("--resume", "--run"))

    def test_winner_identity_and_complete_output_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            args = bench.arguments(f.argv)
            case = dict(width=2, height=1, pixels=2)
            for change in ("missing", "duplicate", "wrong-bytes", "error", "full-error",
                           "full-rejected", "wrong-winner-decision"):
                with self.subTest(change=change):
                    receipt = self.receipt(2)
                    winner = receipt["candidates"][1]
                    if change == "missing":
                        receipt["candidates"].remove(winner)
                    elif change == "duplicate":
                        receipt["candidates"].append(dict(winner))
                    elif change == "wrong-bytes":
                        winner["full_bytes"] += 1
                    elif change == "error":
                        winner["error"] = "failed full encode"
                    elif change in ("full-error", "full-rejected"):
                        winner["decision"] = change
                    else:
                        receipt["candidates"][-1].update(decision="winner", full_bytes=12,
                                                         full_seconds=0.1)
                    with self.assertRaises(ValueError):
                        bench.validate_receipt(receipt, case, args)
            for decision in ("winner", "finalist"):
                receipt = self.receipt(2)
                receipt["candidates"][1].update(decision=decision, full_seconds=0)
                bench.validate_receipt(receipt, case, args)

    def test_completed_winner_and_rejected_records_require_bytes_and_time(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            args = bench.arguments(f.argv)
            case = dict(width=2, height=1, pixels=2)
            for index in (0, 1, 2):
                for key, invalids in (("full_bytes", (None, 0, -1, True, 10.0, "10")),
                                      ("full_seconds", (None, -1, True, "0.1", 3.0))):
                    for invalid in invalids:
                        with self.subTest(index=index, key=key, invalid=invalid):
                            receipt = self.receipt(2)
                            receipt["candidates"][2].update(decision="full-rejected", full_bytes=12,
                                                             full_seconds=0.1)
                            receipt["candidates"][index][key] = invalid
                            with self.assertRaises(ValueError):
                                bench.validate_receipt(receipt, case, args)

    def test_baseline_candidate_identity_consistency_and_bounded_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            args = bench.arguments(f.argv)
            case = dict(width=2, height=1, pixels=2)
            for change in ("missing", "duplicate", "bytes", "seconds", "bounded", "lower-bound",
                           "failed-full"):
                with self.subTest(change=change):
                    receipt = self.receipt(2)
                    baseline = receipt["candidates"][0]
                    if change == "missing":
                        receipt["candidates"].remove(baseline)
                    elif change == "duplicate":
                        receipt["candidates"].append(dict(baseline))
                    elif change == "bytes":
                        baseline["full_bytes"] += 1
                    elif change == "seconds":
                        baseline["full_seconds"] += 0.1
                    elif change == "bounded":
                        baseline.update(decision="full-bounded", full_bytes=None,
                                        full_lower_bound=receipt["bytes"])
                    elif change == "lower-bound":
                        baseline["full_lower_bound"] = receipt["bytes"]
                    else:
                        baseline.update(decision="full-error", error="failed baseline")
                    with self.assertRaises(ValueError):
                        bench.validate_receipt(receipt, case, args)

    def test_baseline_probe_failure_preserves_complete_fallback_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            def scout_failure(command, result):
                if "--output" not in command:
                    return
                receipt = json.loads(result["stdout"])
                receipt.update(winner="baseline", bytes=receipt["baseline"]["bytes"],
                               bpp=receipt["baseline"]["bpp"], full_encodes=1)
                receipt["candidates"][0].update(decision="probe-error", error="baseline crop failed")
                receipt["candidates"][1].update(decision="budget-deferred", full_bytes=None,
                                                 full_seconds=None)
                Path(command[command.index("--output") + 1]).write_bytes(b"x" * receipt["bytes"])
                result["stdout"] = json.dumps(receipt)
            summary = self.invoke(f, ("--run",), effect=scout_failure)
            self.assertEqual(summary["statuses"], {"passed": 2})
            self.assertEqual(summary["candidate_error_counts"], {"probe-error": 2})
            journal = f.root / "out/trials.jsonl"
            lines = journal.read_text().splitlines(keepends=True)
            for line in lines:
                receipt = bench.loads(line)["measurement"]
                baseline = receipt["candidates"][0]
                self.assertEqual(baseline["decision"], "probe-error")
                self.assertEqual(baseline["error"], "baseline crop failed")
                self.assertEqual(baseline["full_bytes"], receipt["bytes"])
                self.assertEqual(baseline["full_seconds"], receipt["baseline"]["seconds"])
            resumed = self.invoke(f, ("--resume", "--run"))
            self.assertEqual(resumed["candidate_error_counts"], {"probe-error": 2})
            args = bench.arguments(f.argv)
            for key in ("full_bytes", "full_seconds", "error"):
                with self.subTest(key=key):
                    row = bench.loads(lines[0])
                    row["measurement"]["candidates"][0][key] = None
                    with self.assertRaises(ValueError):
                        bench.validate_receipt(row["measurement"], row["case"], args)

    def test_resume_rechecks_complete_winner_even_with_matching_stdout_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            self.invoke(f, ("--run",))
            journal = f.root / "out/trials.jsonl"
            lines = journal.read_text().splitlines(keepends=True)
            row = bench.loads(lines[0])
            row["measurement"]["candidates"][1]["full_bytes"] += 1
            row["encoder"]["stdout"] = bench.canonical(row["measurement"])
            row.pop("checksum")
            row["checksum"] = bench.digest(row)
            journal.write_text(bench.canonical(row) + "\n" + "".join(lines[1:]))
            with self.assertRaisesRegex(ValueError, "complete winner"):
                self.invoke(f, ("--resume", "--run"))

    def test_empty_summary_metrics_are_defined(self):
        summary = bench.summarize([], dict(cases=[], excluded=[], duplicate_aliases=[]), "prepare-only")
        totals = summary["successful"]
        self.assertEqual(totals["baseline_bytes"], 0)
        for key in ("baseline_arithmetic_mean_bpp", "baseline_aggregate_weighted_bpp",
                    "byte_change_percent", "overall_mpix_per_second"):
            self.assertIsNone(totals[key])
        self.assertEqual(summary["candidate_errors"], 0)
        self.assertEqual(summary["candidate_decision_counts"], {})
        self.assertEqual(summary["candidate_error_counts_by_name"], {})

    def test_baseline_fallback_reports_candidate_failures_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            def fallback(command, result):
                if "--output" not in command:
                    return
                receipt = json.loads(result["stdout"])
                receipt["winner"] = "baseline"
                receipt["bytes"] = receipt["baseline"]["bytes"]
                receipt["bpp"] = receipt["baseline"]["bpp"]
                receipt["candidates"][0]["decision"] = "winner"
                receipt["candidates"][1].update(decision="full-error", error="full encode bug",
                                                 full_bytes=None)
                receipt["candidates"][2].update(decision="probe-error", error="probe encode bug")
                receipt["candidates"][3].update(error="structurally unsupported")
                receipt["candidates"][4].update(decision="budget-deferred")
                rejected = dict(receipt["candidates"][3], name="rejected", decision="probe-rejected", error=None)
                receipt["candidates"].append(rejected)
                output = Path(command[command.index("--output") + 1])
                output.write_bytes(b"x" * receipt["bytes"])
                result["stdout"] = json.dumps(receipt)
            result = self.invoke(f, ("--run",), effect=fallback)
            self.assertEqual(result["statuses"], {"passed": 2})
            self.assertEqual(result["candidate_errors"], 4)
            self.assertEqual(result["cases_with_candidate_errors"], 2)
            self.assertEqual(result["candidate_error_counts"], {"probe-error": 2, "full-error": 2})
            self.assertEqual(result["candidate_error_counts_by_name"], {"queued": 2, "candidate": 2})
            self.assertEqual(result["candidate_decision_counts"]["budget-deferred"], 2)
            self.assertEqual(result["candidate_decision_counts"]["probe-rejected"], 2)
            self.assertEqual(result["candidate_decision_counts"]["ineligible"], 2)
            self.assertEqual(result["candidate_decision_counts_by_name"]["candidate"], {"full-error": 2})
            resumed = self.invoke(f, ("--resume", "--run"))
            self.assertEqual(resumed["candidate_error_counts"], result["candidate_error_counts"])


if __name__ == "__main__":
    unittest.main()
