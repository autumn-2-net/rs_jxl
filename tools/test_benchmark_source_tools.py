"""No codec, corpus or build required: receipts, summaries and mocked trials."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_source_tools as bench


def options(**changes):
    values = dict(encoder=Path("portfolio_lab.exe"), decoder=Path("officialdjxl"),
                  effort=9, force_source=False, timeout=1,
                  mode=["baseline", "patch"], threads=[1], repeats=3)
    values.update(changes)
    return SimpleNamespace(**values)


def receipt(mode="baseline", **changes):
    value = dict(direct=True, candidate="baseline", effort=9, threads=1,
                 width=2, height=3, bytes=10, seconds=0.5, encode_calls=1,
                 source_tools=dict(profile=mode,
                                   patch_parallel_gather=mode in ("patch", "sparse", "prev1", "screened"),
                                   previous_channels=int(mode in ("prev1", "screened", "prev1-only")),
                                   max_previous_channels=int(mode in ("prev1", "screened", "prev1-only")),
                                   lz_quick="Sparse" if mode in ("sparse", "prev1", "screened", "sparse-only") else "Off",
                                   palette_order="ScreenedRuns" if mode in ("screened", "screened-only") else "Default"))
    value.update(changes)
    return value


class ModeTests(unittest.TestCase):
    def cli(self, *extra):
        return bench.arguments(["--encoder", "portfolio_lab.exe", "--decoder", "officialdjxl",
                                "--manifest", "manifest.json", "--out", "new-report", *extra])

    def test_defaults_and_repeatable_modes(self):
        args = self.cli()
        self.assertEqual(args.mode, list(bench.DEFAULT_MODES))
        self.assertEqual(args.threads, [1, 4, 16])
        self.assertEqual(args.repeats, 3)
        self.assertFalse(args.force_source)
        self.assertEqual(self.cli("--mode", "baseline", "--mode", "sparse-only").mode,
                         ["baseline", "sparse-only"])

    def test_invalid_modes_and_bounds(self):
        for extra in (("--mode", "unknown"), ("--mode", "patch"),
                      ("--mode", "baseline", "--mode", "baseline"),
                      ("--threads", "1", "1"), ("--threads", "0"),
                      ("--repeats", "101"), ("--timeout", "nan"), ("--effort", "8")):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.cli(*extra)

    def test_refuse_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.cli("--out", tmp)

    def test_rotation_and_candidate(self):
        modes = list(bench.DEFAULT_MODES)
        self.assertEqual(bench.trial_order(modes, 1, 1), modes[2:] + modes[:2])
        for forced, candidate in ((False, "baseline"), (True, "source")):
            command = bench.command(options(force_source=forced), Path("in.png"), Path("out.jxl"), 4, "patch")
            self.assertEqual(command[command.index("--direct-candidate") + 1], candidate)
            self.assertEqual(command[-2:], ["--direct-source-tools", "patch"])
        with self.assertRaises(ValueError):
            bench.command(options(), Path("in"), Path("out"), 1, "unknown")

    def test_receipt_modes_and_one_encode(self):
        expected = SimpleNamespace(width=2, height=3)
        for mode in bench.MODES:
            bench.validate_receipt(receipt(mode), options(), mode, 1, 10, expected)
        for changes in (dict(encode_calls=2), dict(encode_calls=True), dict(candidate="source"),
                        dict(seconds=float("nan")), dict(bytes=11), dict(direct=False),
                        dict(source_tools={"profile": "patch"}), dict(width=9)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bench.validate_receipt(receipt(**changes), options(), "baseline", 1, 10, expected)
        wrong = receipt("patch")
        wrong["source_tools"]["patch_parallel_gather"] = False
        with self.assertRaises(ValueError):
            bench.validate_receipt(wrong, options(), "patch", 1, 10, expected)
        gray = receipt("prev1")
        gray["source_tools"]["previous_channels"] = 0
        bench.validate_receipt(gray, options(), "prev1", 1, 10, expected)
        for field in ("max_previous_channels", "lz_quick", "palette_order"):
            wrong = receipt("screened")
            wrong["source_tools"][field] = "ignored"
            with self.assertRaises(ValueError):
                bench.validate_receipt(wrong, options(), "screened", 1, 10, expected)


class SummaryTests(unittest.TestCase):
    def rows(self):
        rows = []
        for source, width in (("small", 10), ("large", 30)):
            for mode, size in (("baseline", 100), ("patch", 80)):
                for repeat, seconds in enumerate((1, 3, 20)):
                    rows.append(dict(source=source, mode=mode, threads=1, repeat=repeat,
                                     width=width, height=10, bytes=size, seconds=seconds,
                                     verified=True, status="passed"))
        return rows

    def summaries(self, rows):
        return bench.summarize(rows, ["small", "large"], ["baseline", "patch"], [1], 3)

    def test_median_sum_weighting_and_size_delta(self):
        baseline, modified = self.summaries(self.rows())
        self.assertEqual(baseline["seconds"], 6)  # Not median of pooled trials.
        self.assertEqual(baseline["bytes"], 200)
        self.assertAlmostEqual(baseline["mean_bpp"], (8 + 8 / 3) / 2)
        self.assertEqual(baseline["weighted_bpp"], 4)
        self.assertAlmostEqual(baseline["mpix_s"], 400 / 1e6 / 6)
        self.assertEqual(modified["bytes"], 160)
        self.assertEqual(modified["per_image"][0]["bytes_delta_vs_baseline"], -20)
        self.assertAlmostEqual(modified["per_image"][0]["size_percent_vs_baseline"], -20)

    def test_failed_missing_unverified_or_duplicate_trials_invalidate_aggregate(self):
        for change in ("failed", "missing", "unverified", "duplicate"):
            rows = self.rows()
            if change == "failed":
                rows[0]["status"] = "failed"
            elif change == "missing":
                rows.pop(0)
            elif change == "unverified":
                rows[0]["verified"] = False
            else:
                rows[0]["repeat"] = 1
            with self.subTest(change=change):
                baseline, modified = self.summaries(rows)
                self.assertEqual(baseline["status"], "failed")
                self.assertNotIn("weighted_bpp", baseline)
                self.assertEqual(modified["status"], "passed")
                self.assertIsNone(modified["per_image"][0]["bytes_delta_vs_baseline"])


class TrialTests(unittest.TestCase):
    def trial(self, native_receipt, decoder_failure=False):
        expected = SimpleNamespace(width=2, height=3, depth=16)
        row = dict(source="image", mode="baseline", threads=1, repeat=0, verified=False)
        folders = []

        def process(command, env, timeout):
            if "--output" in command:
                output = Path(command[command.index("--output") + 1])
                folders.append(output.parent)
                output.write_bytes(b"x" * 10)
                return dict(returncode=0, stdout=json.dumps(native_receipt), stderr="", command=command)
            self.assertIn("--bits_per_sample=16", command)
            Path(command[2]).write_bytes(b"mock decoded PNG")
            return dict(returncode=1 if decoder_failure else 0, stdout="", stderr="decoder failed", command=command)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(bench.port, "run_process", side_effect=process) as run, \
                    patch.object(bench.port, "png_samples", return_value=expected), \
                    patch.object(bench.port, "compare_pixels") as compare:
                bench.measure(options(), Path("snapshot.png"), expected, Path(tmp), {}, row)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(compare.call_count, 0 if decoder_failure else 1)
            self.assertFalse(any(folder.exists() for folder in folders))
        return row

    def test_each_output_decoded_and_cleaned(self):
        row = self.trial(receipt())
        self.assertEqual(row["status"], "passed")
        self.assertTrue(row["verified"])
        self.assertEqual(row["receipt"]["source_tools"]["profile"], "baseline")

    def test_invalid_receipt_still_independently_decoded(self):
        row = self.trial(receipt(encode_calls=2))
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["verified"])
        self.assertIn("encode/receipt", row["errors"][0])

    def test_decoder_failure_is_recorded(self):
        row = self.trial(receipt(), decoder_failure=True)
        self.assertEqual(row["status"], "failed")
        self.assertFalse(row["verified"])
        self.assertIn("decode/pixels", row["errors"][0])


class ManifestTests(unittest.TestCase):
    def run_fixture(self, *, wrong_hash=False, mutate=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, encoder, decoder = (root / name for name in ("input.png", "encoder", "decoder"))
            for path in (source, encoder, decoder):
                path.write_bytes(b"mock fixture")
            digest = bench.port.file_sha256(source)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(cases=[dict(
                source=str(source), source_sha256="wrong" if wrong_hash else digest)])), encoding="utf-8")
            snapshots = []

            def prepare(path, directory):
                snapshot = directory / "snapshot.png"
                snapshot.write_bytes(b"snapshot")
                snapshots.append(snapshot)
                return (dict(kind="png", input_sha256=digest,
                             payload_sha256=bench.port.file_sha256(snapshot)),
                        snapshot, SimpleNamespace(width=2, height=3, depth=16))

            def measure(args, snapshot, expected, directory, env, row):
                row.update(status="passed", verified=True, width=2, height=3,
                           seconds=0.5, bytes=10)
                if mutate:
                    source.write_bytes(b"changed externally")

            args = options(encoder=encoder, decoder=decoder, manifest=manifest,
                           out=root / "report", repeats=1)
            with patch.object(bench, "prepare", side_effect=prepare), \
                    patch.object(bench, "measure", side_effect=measure) as trial, \
                    contextlib.redirect_stdout(io.StringIO()):
                report = bench.benchmark(args)
            self.assertTrue(report["temporary_outputs_cleaned"])
            self.assertFalse(any(path.exists() for path in snapshots))
            self.assertEqual(json.loads((args.out / "summary.json").read_text(encoding="utf-8"))["status"],
                             report["status"])
            self.assertEqual(sorted(path.name for path in args.out.iterdir()), ["summary.json", "trials.jsonl"])
            return report, trial.call_count

    def test_mocked_manifest_success(self):
        report, calls = self.run_fixture()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(calls, 2)

    def test_manifest_hash_mismatch_is_not_skipped(self):
        report, calls = self.run_fixture(wrong_hash=True)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(calls, 0)
        self.assertIn("Manifest hash mismatch", report["errors"][0]["error"])
        self.assertTrue(all(item["status"] == "failed" for item in report["summaries"]))

    def test_source_mutation_invalidates_run(self):
        report, calls = self.run_fixture(mutate=True)
        self.assertEqual(calls, 2)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("Immutable input changed" in error["error"] for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
