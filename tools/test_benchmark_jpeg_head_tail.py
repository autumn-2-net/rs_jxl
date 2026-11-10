from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

import benchmark_jpeg_head_tail as harness


class JpegHeadTailTests(unittest.TestCase):
    @staticmethod
    def measurement(mode="head-tail", **changes):
        work = dict(attempted=True, candidate=True, selected=True, histograms=4,
                    changed_histograms=2, configurations=3, baseline_bytes=100,
                    candidate_bytes=90, search_seconds=0.2, write_seconds=0.1)
        size = 100 if mode == "baseline" else 90
        result = dict(bytes=size, seconds=1.0, width=8, height=6, bpp=8 * size / 48,
                      mpix_per_second=48 / 1e6, native_verified=True,
                      head_tail=None if mode == "baseline" else work)
        result.update(changes)
        return result

    @staticmethod
    def validate(m, mode="head-tail"):
        harness.validate_measurement(m, mode, m["bytes"], 8, 6)

    def test_command_only_opt_in_changes_mode(self):
        args = SimpleNamespace(encoder=Path("lab"), effort=9, threads=8)
        baseline = harness.command(args, "in.jpg", "out.jxl", "baseline")
        self.assertEqual(baseline, ["lab", "in.jpg", "--effort", "9", "--threads", "8",
                                    "--output", "out.jxl"])
        self.assertEqual(harness.command(args, "in.jpg", "out.jxl", "head-tail"),
                         baseline + ["--head-tail"])
        with self.assertRaises(ValueError):
            harness.command(args, "in.jpg", "out.jxl", "other")

    def test_argument_defaults_and_guards(self):
        with TemporaryDirectory() as folder:
            argv = ["--encoder", "lab", "--decoder", "djxl", "--case", "input.jpg",
                    "--out", str(Path(folder) / "report.json")]
            args = harness.arguments(argv)
            self.assertEqual((args.effort, args.threads, args.repeats), (9, 8, 3))
            for extra in (["--threads", "0"], ["--threads", "257"], ["--effort", "12"],
                          ["--repeats", "0"], ["--timeout", "nan"], ["--timeout", "inf"],
                          ["--timeout", "0"], ["--manifest", "cases.json"]):
                with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    harness.arguments(argv + extra)
            Path(args.out).write_text("existing", encoding="utf-8")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                harness.arguments(argv)
            self.assertEqual(args.out.read_text(), "existing")

    def test_rotates_both_case_and_repetition(self):
        self.assertEqual(harness.trial_order(harness.MODES, 0, 0), ("baseline", "head-tail"))
        self.assertEqual(harness.trial_order(harness.MODES, 1, 0), ("head-tail", "baseline"))
        self.assertEqual(harness.trial_order(harness.MODES, 0, 1), ("head-tail", "baseline"))

    def test_valid_baseline_winner_and_fallback_receipts(self):
        self.validate(self.measurement("baseline"), "baseline")
        self.validate(self.measurement())
        for attempted, candidate, candidate_bytes in ((False, False, None),
                                                     (True, False, None), (True, True, 110),
                                                     (True, True, 100)):
            m = self.measurement(bytes=100, bpp=800 / 48)
            m["head_tail"].update(attempted=attempted, candidate=candidate, selected=False,
                                  candidate_bytes=candidate_bytes,
                                  changed_histograms=2 if candidate else 0,
                                  configurations=3 if attempted else 0)
            self.validate(m)

    def test_requires_exact_activation_and_native_verification(self):
        for value in (False, 1, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(self.measurement(native_verified=value))
        m = self.measurement()
        for key in ("head_tail", "native_verified"):
            missing = dict(m)
            del missing[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.validate(missing)
        with self.assertRaises(ValueError):
            self.validate(self.measurement(head_tail=None))
        with self.assertRaises(ValueError):
            self.validate(self.measurement(), "baseline")

    def test_log_alpha_only_candidate_can_have_zero_changed_histograms(self):
        m = self.measurement()
        m["head_tail"]["changed_histograms"] = 0
        self.validate(m)

    def test_all_receipt_fields_required_and_strictly_typed(self):
        for key in self.measurement()["head_tail"]:
            m = self.measurement()
            del m["head_tail"][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.validate(m)
        for key in ("attempted", "candidate", "selected"):
            for value in (0, 1, None, "true"):
                m = self.measurement()
                m["head_tail"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.validate(m)
        for key in ("histograms", "changed_histograms", "configurations", "baseline_bytes",
                    "candidate_bytes"):
            for value in (-1, True, 1.5, None):
                m = self.measurement()
                m["head_tail"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.validate(m)

    def test_inconsistent_work_and_selection_rejected(self):
        for changes in (dict(attempted=False), dict(candidate=False), dict(selected=False),
                        dict(changed_histograms=5),
                        dict(configurations=0), dict(candidate_bytes=0), dict(candidate_bytes=110),
                        dict(baseline_bytes=80), dict(search_seconds=2), dict(write_seconds=2)):
            m = self.measurement()
            m["head_tail"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.validate(m)

    def test_invalid_native_metrics_rejected(self):
        for key in ("seconds", "bpp", "mpix_per_second"):
            for value in (0, -1, True, None, float("nan"), float("inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.validate(self.measurement(**{key: value}))
        for key in ("search_seconds", "write_seconds"):
            for value in (-1, True, None, float("nan"), float("inf")):
                m = self.measurement()
                m["head_tail"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.validate(m)
        for key, value in (("bytes", True), ("width", True), ("height", 6.0),
                           ("width", 7), ("bpp", 1), ("mpix_per_second", 1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                harness.validate_measurement(self.measurement(**{key: value}), "head-tail", 90, 8, 6)

    def test_same_trial_guard_including_internal_baseline_identity(self):
        pair = {mode: self.measurement(mode) for mode in harness.MODES}
        harness.validate_pair(pair)
        pair["head-tail"]["head_tail"]["baseline_bytes"] = 101
        with self.assertRaises(ValueError):
            harness.validate_pair(pair)
        pair["head-tail"] = self.measurement(bytes=101)
        with self.assertRaises(ValueError):
            harness.validate_pair(pair)

    def test_medians_and_weighted_aggregate_use_native_seconds(self):
        samples = [{"measurement": dict(bytes=size, seconds=seconds)}
                   for size, seconds in ((90, 1), (100, 3), (110, 2))]
        summary = harness.summarize(samples, 48, 200)
        self.assertEqual((summary["bytes"], summary["seconds"], summary["original_jpeg_ratio"]),
                         (100, 2, 0.5))
        cases = [dict(pixels=48, original_bytes=200,
                      summaries={mode: summary for mode in harness.MODES})] * 2
        total = harness.aggregate(cases)
        self.assertEqual(total["modes"]["baseline"]["bytes"], 200)
        self.assertEqual(total["modes"]["baseline"]["seconds"], 4)
        self.assertEqual(total["modes"]["baseline"]["mpix_per_second"], 96 / 1e6 / 4)

    def run_mock(self, folder, *, failure=None, repeats=2, case_count=1, large_input=False):
        root = Path(folder)
        source = root / "input.jpg"
        buffer = BytesIO()
        Image.new("RGB", (8, 6), (20, 30, 40)).save(buffer, format="JPEG")
        original = buffer.getvalue()
        if large_input:
            original += b"\0" * (16 << 20)
        source.write_bytes(original)
        sources = [source]
        for index in range(1, case_count):
            extra = root / f"input-{index}.jpg"
            extra.write_bytes(original)
            sources.append(extra)
        encoder, decoder = root / "lab.exe", root / "djxl.exe"
        encoder.write_bytes(b"mock encoder")
        decoder.write_bytes(b"mock decoder")
        args = harness.arguments(["--encoder", str(encoder), "--decoder", str(decoder),
                                  "--case", str(source), "--out", str(root / "report.json"),
                                  "--repeats", str(repeats), "--timeout", "7"])
        args.case = sources
        calls = []

        def process(cmd, env, timeout):
            calls.append(cmd)
            self.assertEqual(timeout, 7)
            self.assertNotIn("RS_JXL_UNEXPECTED", env)
            self.assertNotIn("JXL_UNEXPECTED", env)
            record = dict(command=cmd, returncode=0, stdout="", stderr="", wall_seconds=99)
            if cmd[0] == str(encoder.resolve()):
                output = Path(cmd[cmd.index("--output") + 1])
                self.assertFalse(output.exists())
                self.assertEqual(Path(cmd[1]).read_bytes(), original)
                mode = "head-tail" if "--head-tail" in cmd else "baseline"
                m = self.measurement(mode)
                output.write_bytes(b"x" * m["bytes"])
                if failure == "native":
                    m["native_verified"] = False
                if failure == "pair" and mode == "head-tail":
                    m["head_tail"]["baseline_bytes"] = 101
                if failure == "timeout":
                    record.update(returncode=None, error="Process timeout after 7s")
                if failure == "size":
                    output.write_bytes(b"short")
                record["stdout"] = json.dumps(m)
                if failure == "json":
                    record["stdout"] = "not JSON"
                if failure == "nonfinite":
                    record["stdout"] = record["stdout"][:-1] + ', "extra": NaN}'
            else:
                self.assertEqual(cmd[0], str(decoder.resolve()))
                self.assertEqual(cmd[4:], ["--quiet"])
                self.assertEqual(cmd[3], "--num_threads=1")
                self.assertEqual(Path(cmd[2]).suffix, ".jpg")
                Path(cmd[2]).write_bytes(original + b"tail" if failure == "jpeg" else original)
                if failure == "mutation":
                    encoder.write_bytes(b"changed encoder")
                if failure == "decoder":
                    record.update(returncode=1, stderr="decoder failed")
            return record

        with patch.object(harness.port, "run_process", side_effect=process), \
                patch.dict("os.environ", {"RS_JXL_UNEXPECTED": "1", "JXL_UNEXPECTED": "1"}), \
                redirect_stdout(StringIO()):
            report = harness.benchmark(args)
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertEqual(json.loads(args.out.read_text())["status"], report["status"])
        self.assertFalse(list(root.rglob("*.jxl")))
        for cmd in calls:
            if "--output" in cmd:
                self.assertFalse(Path(cmd[cmd.index("--output") + 1]).exists())
        return report, calls

    def test_mock_end_to_end_rotates_verifies_hashes_and_cleans(self):
        with TemporaryDirectory() as folder:
            report, calls = self.run_mock(folder)
            self.assertEqual(report["status"], "passed", report["errors"])
            encodes = [cmd for cmd in calls if "--output" in cmd]
            self.assertEqual(["--head-tail" in cmd for cmd in encodes], [False, True, True, False])
            self.assertEqual(len(calls), 8)
            self.assertEqual(len(report["encoder"]["sha256"]), 64)
            self.assertEqual(len(report["decoder"]["sha256"]), 64)
            case = report["cases"][0]
            self.assertEqual(len(case["sha256"]), 64)
            self.assertEqual(case["summaries"]["baseline"]["seconds"], 1)
            self.assertTrue(all(s["verified"] for rows in case["variants"].values() for s in rows))

    def test_fail_closed_and_cleanup_on_each_failure(self):
        for failure in ("native", "jpeg", "pair", "timeout", "size", "mutation",
                        "json", "nonfinite", "decoder"):
            with self.subTest(failure=failure), TemporaryDirectory() as folder:
                report, _ = self.run_mock(folder, failure=failure, repeats=1)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["errors"])

    def test_manifest_relative_paths_and_duplicates(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "original.jpg"
            source.write_bytes(b"unused")
            manifest = root / "cases.json"
            manifest.write_text(json.dumps({"cases": [{"original": source.name}]}))
            args = SimpleNamespace(case=[], manifest=manifest)
            self.assertEqual(harness.load_cases(args), [source.resolve()])
            manifest.write_text(json.dumps({"cases": [{"original": source.name}] * 2}))
            with self.assertRaises(ValueError):
                harness.load_cases(args)

    def test_more_than_sixteen_cases_are_not_rejected_or_truncated(self):
        with TemporaryDirectory() as folder:
            report, calls = self.run_mock(folder, case_count=17, repeats=1)
            self.assertEqual(report["status"], "passed", report["errors"])
            self.assertEqual(len(report["cases"]), 17)
            self.assertEqual(len(calls), 17 * 4)

    def test_input_above_sixteen_mib_preserves_all_original_bytes(self):
        with TemporaryDirectory() as folder:
            report, calls = self.run_mock(folder, large_input=True, repeats=1)
            self.assertEqual(report["status"], "passed", report["errors"])
            self.assertGreater(report["cases"][0]["original_bytes"], 16 << 20)
            self.assertEqual(len(calls), 4)

    def test_directory_rejected_before_any_process(self):
        with TemporaryDirectory() as folder:
            with patch.object(harness, "load_cases", return_value=[Path(folder)]):
                report, calls = self.run_mock(folder)
            self.assertEqual(report["status"], "failed")
            self.assertIn("not a file", report["errors"][0])
            self.assertEqual(calls, [])

    def test_input_must_be_jpeg_not_merely_an_image(self):
        with TemporaryDirectory() as folder:
            buffer = BytesIO()
            Image.new("RGB", (8, 6)).save(buffer, format="PNG")
            path = Path(folder) / "wrong.jpg"
            path.write_bytes(buffer.getvalue())
            with patch.object(harness, "load_cases", return_value=[path]):
                report, calls = self.run_mock(folder)
            self.assertEqual(report["status"], "failed")
            self.assertIn("Not JPEG", report["errors"][0])
            self.assertEqual(calls, [])

    def test_report_exclusive_creation(self):
        with TemporaryDirectory() as folder:
            out = Path(folder) / "report.json"
            out.write_text("keep")
            with self.assertRaises(FileExistsError):
                harness.benchmark(SimpleNamespace(out=out))
            self.assertEqual(out.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
