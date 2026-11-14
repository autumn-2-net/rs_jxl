"""Refresh isolation, frozen baselines and native policy tests; no real encodes."""

import copy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_effort_refresh as bench
from test_benchmark_context_corpus import complete_report


def identity(old, path):
    return dict(schema=bench.SCHEMA, previous_report=bench.corpus.hashed(path),
        previous_fingerprint=old["fingerprint"], inputs=copy.deepcopy(old["identity"]["inputs"]),
        label="new", binaries=dict(encoder=dict(path="new.exe"), decoder=dict(path="decoder.exe")),
        sources=[], config=dict(efforts=[1, 9], threads=8, repeats=1, timeout=3600.,
                                serial=True, patches="auto", search=False))


def row(ident, index=0, effort=9, gray=False):
    case = ident["inputs"][index]
    size, seconds = 100 + index, .02 * (index + 1)
    pixels = case["width"] * case["height"]
    ep = dict(command=bench.command(ident, effort, "snapshot.png", "output.jxl"),
              returncode=0, wall_seconds=1., stdout="", stderr="")
    dp = dict(command=bench.port.decoder_command("decoder.exe", "output.jxl", "oracle.png", 16),
              returncode=0, wall_seconds=1., stdout="", stderr="")
    m = dict(effort=effort, threads=8, encode_calls=1, bytes=size, width=case["width"],
        height=case["height"], direct=True, candidate="baseline", timing=bench.TIMING,
        seconds=seconds, bpp=size * 8 / pixels, mpix_per_second=pixels / 1e6 / seconds,
        direct_ma_profile="half-single-pass" if effort == 9 else "off",
        direct_tree_learning="symmetric" if effort == 9 else "greedy",
        source_tools=dict(profile=None, max_previous_channels=None,
            previous_channels=int(effort == 9 and not gray),
            lz_quick="Sparse" if effort == 9 else "Off",
            palette_order="ScreenedRuns" if effort == 9 else "Default"))
    return bench.corpus.seal_row(dict(fingerprint=bench.corpus.fingerprint(ident),
        case_index=index, effort=effort, repetition=0, source_sha256=case["sha256"],
        pure_gray=gray, verified=True, status="verified", bytes=size, wall_seconds=1.,
        measurement=m, encoder_process=ep, decoder_process=dp,
        verification=dict(original_samples_equal=True, includes_hidden_rgb=True, depth=16)))


def fixture(folder):
    old = complete_report()
    path = folder / "old.json"
    bench.corpus.atomic_json(path, old)
    ident = identity(old, path)
    rows = [row(ident, i, e) for i in range(2) for e in (1, 9)]
    _, totals = bench.old_plot.validate_report(old)
    return dict(schema=bench.SCHEMA, status="complete", integrity_verified=True, errors=[],
        identity=ident, fingerprint=bench.corpus.fingerprint(ident), trials=rows,
        verified_trials=4, expected_trials=4, failed_trials=0,
        series=bench.series(ident, rows, old, totals))


class RefreshTests(unittest.TestCase):
    def test_defaults_only_new_efforts(self):
        args = bench.arguments(["--previous", "old.json", "--encoder", "new.exe",
            "--decoder", "decoder.exe", "--outdir", "run"])
        self.assertEqual(args.efforts, list(range(1, 10)))
        with tempfile.TemporaryDirectory() as tmp:
            report = fixture(Path(tmp))
            cmd = bench.command(report["identity"], 9, "in.png", "out.jxl")
            self.assertEqual(cmd, ["new.exe", "in.png", "--output", "out.jxl", "--effort", "9",
                                  "--threads", "8", "--direct-candidate", "baseline"])

    def test_gray_transparency_matches_native_expansion(self):
        self.assertTrue(bench.pure_gray(dict(color_type=0)))
        self.assertFalse(bench.pure_gray(dict(color_type=0, transparency_key=(0,))))
        for color in (2, 3, 4, 6):
            self.assertFalse(bench.pure_gray(dict(color_type=color)))

    def test_policy_and_encoding_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ident = fixture(Path(tmp))["identity"]
            for effort in (1, 9):
                for gray in (False, True):
                    r = row(ident, effort=effort, gray=gray)
                    bench.validate_row(r, ident, bench.corpus.fingerprint(ident))
            for field, value in (("encode_calls", 2), ("candidate", "prev2"),
                                 ("seconds", -1), ("timing", "wall time"), ("bpp", 1)):
                r = row(ident)
                r["measurement"][field] = value
                r = bench.corpus.seal_row(r)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    bench.validate_row(r, ident, bench.corpus.fingerprint(ident))
            r = row(ident)
            r["measurement"]["source_tools"]["previous_channels"] = 0
            with self.assertRaisesRegex(ValueError, "e9 defaults"):
                bench.validate_native(r["measurement"], ident, r)

    def test_cached_series_and_exact_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = fixture(Path(tmp))
            with patch.object(bench.port, "run_process", side_effect=AssertionError("no subprocess")):
                series = bench.validate_report(report)
            point = series["new"]["points"][0]
            self.assertAlmostEqual(point["mean_bpp"], (100 * 8 / 2 + 101 * 8 / 10) / 2)
            self.assertEqual(point["summed_bytes"], 201)
            self.assertAlmostEqual(point["native_seconds"], .06)
            old, totals = bench.previous_report(report["identity"]["previous_report"])
            self.assertEqual(series["previous"]["points"][0], dict(effort=1, **totals["rust-e1"]))
            self.assertEqual(series["cjxl"]["points"][1], dict(effort=9, **totals["cjxl-e9"]))

    def test_incomplete_duplicate_and_modified_cache_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = fixture(Path(tmp))
            for mutation in ("missing", "duplicate", "failed", "aggregate", "threads"):
                r = copy.deepcopy(good)
                if mutation == "missing":
                    r["trials"].pop()
                elif mutation == "duplicate":
                    r["trials"].append(r["trials"][0])
                elif mutation == "failed":
                    r["trials"][0].update(status="failed", verified=False)
                    r["trials"][0] = bench.corpus.seal_row(r["trials"][0])
                elif mutation == "aggregate":
                    r["series"]["new"]["points"][0]["mean_bpp"] += 1
                else:
                    r["identity"]["config"]["threads"] = 16
                    r["fingerprint"] = bench.corpus.fingerprint(r["identity"])
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    bench.validate_report(r)
            Path(good["identity"]["previous_report"]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Cached report changed"):
                bench.validate_report(good)

    def test_resume_skips_verified_rows_and_never_runs_cached_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = fixture(root)
            ident = report["identity"]
            out = root / "run"
            out.mkdir()
            bench.corpus.atomic_json(out / "identity.json", dict(identity=ident,
                                    fingerprint=report["fingerprint"]))
            (out / "trials.jsonl").write_text("".join(bench.corpus.serialized(r) + "\n"
                for r in report["trials"]), encoding="utf-8")
            args = SimpleNamespace(outdir=out, resume=True, resume_harness_update=False)
            with patch.object(bench, "freeze", return_value=ident), \
                 patch.object(bench, "check_frozen"), patch.object(bench.port, "pixel_libraries"), \
                 patch.object(bench.port, "run_process", side_effect=AssertionError("unexpected encode")):
                result = bench.benchmark(args)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["series"], report["series"])

    def test_plot_failure_is_reported(self):
        import plot_effort_refresh
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            bench.corpus.atomic_json(out / "progress.json", dict(status="complete"))
            args = SimpleNamespace(outdir=out, plot=True)
            with patch.object(bench, "arguments", return_value=args), \
                 patch.object(bench, "benchmark", return_value=dict(status="complete",
                    expected_trials=4, verified_trials=4, failed_trials=0, elapsed_seconds=1.)), \
                 patch.object(plot_effort_refresh, "main", return_value=1), redirect_stdout(StringIO()):
                self.assertEqual(bench.main([]), 1)
            self.assertEqual(bench.corpus.loads((out / "progress.json").read_bytes())["status"], "plot_failed")

    def test_atomic_write_retries_windows_conflict_but_not_other_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            conflict = PermissionError("held by reader")
            conflict.winerror = 5
            original_replace = bench.os.replace
            with patch.object(bench.os, "replace", side_effect=[conflict, conflict, None]) as replace, \
                 patch.object(bench.time, "sleep") as sleep:
                bench.atomic_json(path, {"status": "running"})
            self.assertEqual(replace.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            with patch.object(bench.os, "replace", side_effect=OSError("disk full")) as replace, \
                 patch.object(bench.time, "sleep") as sleep, self.assertRaises(OSError):
                bench.atomic_json(path, {})
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()
            with patch.object(bench.os, "replace", wraps=original_replace):
                bench.atomic_json(path, {"status": "complete"})
            self.assertEqual(bench.corpus.loads(path.read_bytes())["status"], "complete")
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_progress_failure_is_advisory_and_recovers_without_warning_spam(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = bench.ProgressWriter(Path(tmp) / "progress.json")
            stderr = StringIO()
            with patch.object(bench, "atomic_json", side_effect=PermissionError("locked")), \
                 redirect_stderr(stderr):
                self.assertFalse(writer.write({"status": "running"}))
                self.assertFalse(writer.write({"status": "running"}))
            self.assertEqual(stderr.getvalue().count("snapshot unavailable"), 1)
            self.assertEqual(writer.failures, 2)
            self.assertTrue(writer.write({"status": "complete"}))
            self.assertFalse(writer.warning_active)
            self.assertEqual(bench.corpus.loads(writer.path.read_bytes())["progress_write_failures"], 2)

    @unittest.skipUnless(os.name == "nt", "Windows file-sharing regression")
    def test_real_windows_reader_without_delete_sharing(self):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                           wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        create.restype = wintypes.HANDLE
        close = kernel.CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            bench.atomic_json(path, {"status": "old"})
            handle = create(str(path), 0x80000000, 3, None, 3, 0, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value, ctypes.get_last_error())
            writer = bench.ProgressWriter(path)
            try:
                with patch.object(bench.time, "sleep"), redirect_stderr(StringIO()):
                    self.assertFalse(writer.write({"status": "running"}))
                self.assertEqual(bench.corpus.loads(path.read_bytes())["status"], "old")
            finally:
                close(handle)
            self.assertTrue(writer.write({"status": "running"}))
            self.assertEqual(bench.corpus.loads(path.read_bytes())["status"], "running")

    def test_runtime_repair_is_explicit_and_cannot_change_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = fixture(Path(tmp))
            original = report["identity"]
            saved = dict(identity=original, fingerprint=report["fingerprint"])
            current = copy.deepcopy(original)
            current["sources"] = [dict(path="runner.py", sha256="1" * 64)]
            with self.assertRaisesRegex(ValueError, "resume-harness-update"):
                bench.resume_identity(saved, current)
            self.assertEqual(bench.resume_identity(saved, current, True), original)
            for field in ("binaries", "inputs", "config", "previous_report"):
                invalid = copy.deepcopy(current)
                invalid[field] = {}
                with self.subTest(field=field), self.assertRaises(ValueError):
                    bench.resume_identity(saved, invalid, True)
            runtime_digest = bench.corpus.fingerprint(current)
            report["runtime_identities"] = {report["fingerprint"]: original, runtime_digest: current}
            report["trials"][0]["runtime_fingerprint"] = runtime_digest
            report["trials"][0] = bench.corpus.seal_row(report["trials"][0])
            bench.validate_report(report)
            del report["runtime_identities"][runtime_digest]
            with self.assertRaisesRegex(ValueError, "Unknown trial runtime"):
                bench.validate_report(report)

    def test_resume_with_locked_progress_preserves_all_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = fixture(root)
            original = report["identity"]
            current = copy.deepcopy(original)
            current["sources"] = [dict(path="repaired-runner.py", sha256="2" * 64)]
            out = root / "run"
            out.mkdir()
            bench.corpus.atomic_json(out / "identity.json", dict(identity=original,
                                    fingerprint=report["fingerprint"]))
            journal = out / "trials.jsonl"
            journal.write_text("".join(bench.corpus.serialized(r) + "\n"
                for r in report["trials"]), encoding="utf-8")
            before = journal.read_bytes()
            old_identity = (out / "identity.json").read_bytes()
            actual_write = bench.atomic_json

            def locked_progress(path, value):
                if path.name == "progress.json":
                    raise PermissionError("progress locked for entire run")
                actual_write(path, value)

            args = SimpleNamespace(outdir=out, resume=True, resume_harness_update=True)
            with patch.object(bench, "freeze", return_value=current), \
                 patch.object(bench, "check_frozen") as check, \
                 patch.object(bench.port, "pixel_libraries"), \
                 patch.object(bench, "atomic_json", side_effect=locked_progress), \
                 patch.object(bench.port, "run_process", side_effect=AssertionError("unexpected encode")), \
                 redirect_stderr(StringIO()):
                result = bench.benchmark(args)
            self.assertEqual(result["status"], "complete")
            self.assertGreater(result["progress_write_failures"], 0)
            self.assertEqual(journal.read_bytes(), before)
            self.assertEqual((out / "identity.json").read_bytes(), old_identity)
            self.assertEqual(result["series"], bench.validate_report(result))
            check.assert_called_once_with(current)


if __name__ == "__main__":
    unittest.main()
