"""Lightweight defaults-runner tests: no native builds or benchmark launches."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_source_defaults as bench
from test_compare_cpp_port import make_png


def options(**changes):
    values = dict(baseline=Path("old.exe"), candidate=Path("new.exe"),
                  decoder=Path("officialdjxl"), force_source=False, timeout=1,
                  threads=[1], repeats=2)
    values.update(changes)
    return SimpleNamespace(**values)


def receipt(mode="baseline", **changes):
    row = dict(direct=True, candidate="baseline", effort=9, threads=1,
               width=2, height=3, bytes=10, seconds=0.5, encode_calls=1,
               source_tools=dict(profile=None, previous_channels=int(mode == "candidate"),
                                 lz_quick="Sparse" if mode == "candidate" else "Off",
                                 palette_order="ScreenedRuns" if mode == "candidate" else "Default"))
    row.update(changes)
    return row


class PolicyTests(unittest.TestCase):
    def cli(self, *extra):
        return bench.arguments(["--baseline", "old.exe", "--candidate", "new.exe",
                                "--decoder", "officialdjxl", "--manifest", "cases.json",
                                "--out", "new-report", *extra])

    def test_cli_defaults_and_bounds(self):
        args = self.cli()
        self.assertEqual(args.threads, [1, 4, 16])
        self.assertEqual(args.repeats, 2)
        self.assertFalse(args.force_source)
        for extra in (("--threads", "0"), ("--threads", "1", "1"),
                      ("--threads", "257"), ("--repeats", "0"), ("--repeats", "101"),
                      ("--timeout", "nan"), ("--timeout", "inf"), ("--timeout", "0"),
                      ("--direct-source-tools", "screened"), ("--effort", "8")):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.cli(*extra)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.cli("--out", tmp)

    def test_exact_commands_no_experiment_switches(self):
        for forced in (False, True):
            for mode, exe in (("baseline", "old.exe"), ("candidate", "new.exe")):
                self.assertEqual(bench.command(options(force_source=forced), Path("in.png"),
                                               Path("out.jxl"), 4, mode),
                                 [exe, "in.png", "--output", "out.jxl", "--effort", "9",
                                  "--threads", "4", "--direct-candidate",
                                  "source" if forced else "baseline"])
        with self.assertRaises(ValueError):
            bench.command(options(), Path("in"), Path("out"), 1, "screened")

    def test_old_minimal_and_new_effective_receipts(self):
        expected = SimpleNamespace(width=2, height=3)
        for mode in bench.MODES:
            bench.validate_receipt(receipt(mode), options(), mode, 1, 10, expected)
        new = receipt("candidate")
        new["source_tools"]["previous_channels"] = 0
        with self.assertRaises(ValueError):
            bench.validate_receipt(new, options(), "candidate", 1, 10, expected)
        bench.validate_receipt(new, options(), "candidate", 1, 10, expected, grayscale=True)
        with self.assertRaises(ValueError):
            bench.validate_receipt(receipt("candidate"), options(), "candidate", 1, 10,
                                   expected, grayscale=True)
        for changes in (dict(encode_calls=2), dict(encode_calls=True), dict(effort=True),
                        dict(threads=True), dict(width=9), dict(bytes=11), dict(direct=False),
                        dict(candidate="source"), dict(seconds=True), dict(seconds=0),
                        dict(seconds=float("nan")), dict(source_tools={})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bench.validate_receipt(receipt(**changes), options(), "baseline", 1, 10, expected)
        for mode in bench.MODES:
            for key, value in (("profile", "screened"), ("previous_channels", 2),
                               ("previous_channels", True), ("lz_quick", "Wrong"),
                               ("palette_order", "Wrong"), ("max_previous_channels", 1)):
                wrong = receipt(mode)
                wrong["source_tools"][key] = value
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    bench.validate_receipt(wrong, options(), mode, 1, 10, expected)
        bench.validate_receipt(receipt(candidate="source"), options(force_source=True),
                               "baseline", 1, 10, expected)


class SummaryTests(unittest.TestCase):
    def rows(self):
        return [dict(source=source, mode=mode, threads=1, repeat=repeat,
                     width=width, height=10, bytes=size, seconds=seconds,
                     status="passed", verified=True)
                for source, width in (("small", 10), ("large", 30))
                for mode, size in (("baseline", 100), ("candidate", 80))
                for repeat, seconds in enumerate((1, 3, 20))]

    def test_medians_weighting_and_changes(self):
        old, new = bench.summarize(self.rows(), ["small", "large"], [1], 3)
        self.assertEqual(old["seconds"], 6)
        self.assertEqual(old["weighted_bpp"], 4)
        self.assertAlmostEqual(old["mean_bpp"], (8 + 8 / 3) / 2)
        self.assertAlmostEqual(old["mpix_s"], 400 / 1e6 / 6)
        self.assertEqual(new["changes_vs_baseline"]["bytes"]["delta"], -40)
        self.assertAlmostEqual(new["changes_vs_baseline"]["weighted_bpp"]["percent"], -20)

    def test_no_partial_comparisons(self):
        for change in ("missing", "failed", "unverified", "duplicate"):
            rows = self.rows()
            if change == "missing":
                rows.pop(0)
            elif change == "duplicate":
                rows[0]["repeat"] = 1
            else:
                rows[0]["status" if change == "failed" else "verified"] = "failed" if change == "failed" else False
            old, new = bench.summarize(rows, ["small", "large"], [1], 3)
            self.assertEqual(old["status"], "failed")
            self.assertNotIn("seconds", old)
            self.assertTrue(all(v is None for v in new["changes_vs_baseline"].values()))


class TrialTests(unittest.TestCase):
    def test_real_integer_samples_reject_hidden_rgb_alpha_and_low_bit_changes(self):
        try:
            bench.port.pixel_libraries()
        except RuntimeError as error:
            self.skipTest(str(error))
        for depth in (8, 16):
            top = (1 << depth) - 1
            samples = [19, 81, 253, 0, 7, 11, 13, top]
            if depth == 16:
                samples[:3] = [259, 513, 65534]
            original = make_png(depth, 6, [samples])
            for changed_channel in (None, 0, 3):
                with self.subTest(depth=depth, channel=changed_channel), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    source = root / "source.png"
                    source.write_bytes(original)
                    _, snapshot, expected = bench.prepare(source, root)
                    changed = list(samples)
                    if changed_channel is not None:
                        changed[changed_channel] ^= 1
                    decoded = make_png(depth, 6, [changed])
                    row = dict(mode="candidate", threads=1, verified=False)

                    def process(cmd, env, timeout):
                        if "--output" in cmd:
                            Path(cmd[cmd.index("--output") + 1]).write_bytes(b"x" * 10)
                            return dict(returncode=0, stdout=json.dumps(receipt("candidate", height=1)), stderr="")
                        self.assertIn(f"--bits_per_sample={depth}", cmd)
                        Path(cmd[2]).write_bytes(decoded)
                        return dict(returncode=0, stdout="", stderr="")

                    with patch.object(bench.port, "run_process", side_effect=process):
                        bench.measure(options(), snapshot, expected, root, {}, row)
                    self.assertEqual(row["status"], "passed" if changed_channel is None else "failed")
                    self.assertEqual(row["verified"], changed_channel is None)

    def test_all_outputs_decoded_even_invalid_receipt_and_cleanup(self):
        for invalid, decoder_fails in ((False, False), (True, False), (False, True)):
            with self.subTest(invalid=invalid, decoder_fails=decoder_fails), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                snapshot = root / "input.png"
                snapshot.write_bytes(b"mock PNG")
                expected = SimpleNamespace(width=2, height=3, depth=16)
                row = dict(mode="candidate", threads=1, verified=False)
                folders = []

                def process(cmd, env, timeout):
                    if "--output" in cmd:
                        output = Path(cmd[cmd.index("--output") + 1])
                        folders.append(output.parent)
                        output.write_bytes(b"x" * 10)
                        return dict(returncode=0, stdout=json.dumps(receipt("candidate", encode_calls=2 if invalid else 1)), stderr="")
                    self.assertIn("--bits_per_sample=16", cmd)
                    Path(cmd[2]).write_bytes(b"decoded")
                    return dict(returncode=1 if decoder_fails else 0, stdout="", stderr="failure")

                with patch.object(bench.port, "png_payload", return_value=(b"", {"color_type": 6})), \
                        patch.object(bench.port, "run_process", side_effect=process) as run, \
                        patch.object(bench.port, "png_samples", return_value=expected), \
                        patch.object(bench.port, "compare_pixels") as compare:
                    bench.measure(options(), snapshot, expected, root, {}, row)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(compare.call_count, 0 if decoder_fails else 1)
                self.assertEqual(row["status"], "failed" if invalid or decoder_fails else "passed")
                self.assertEqual(row["verified"], not decoder_fails)
                self.assertFalse(any(p.exists() for p in folders))


class ManifestTests(unittest.TestCase):
    def fixture(self, *, wrong_hash=False, mutate=None, fail=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {name: root / name for name in ("baseline", "candidate", "decoder", "source")}
            for name, path in paths.items():
                path.write_bytes(name.encode())
            digest = bench.port.file_sha256(paths["source"])
            manifest = root / "cases.json"
            manifest.write_text(json.dumps(dict(cases=[dict(source=str(paths["source"]),
                source_sha256="wrong" if wrong_hash else digest)])), encoding="utf-8")
            snapshots, order = [], []

            def prepare(source, directory):
                snapshot = directory / "input.png"
                snapshot.write_bytes(b"snapshot")
                snapshots.append(snapshot)
                return (dict(kind="png", input_sha256=digest, payload_sha256=bench.port.file_sha256(snapshot)),
                        snapshot, SimpleNamespace(width=2, height=3, depth=16))

            def measure(args, snapshot, expected, directory, env, row):
                order.append((row["repeat"], row["mode"]))
                row.update(status="failed" if fail and row["mode"] == "baseline" else "passed",
                           verified=True, width=2, height=3, bytes=10, seconds=0.5)
                if mutate:
                    paths[mutate].write_bytes(b"changed")

            args = options(**{k: paths[k] for k in ("baseline", "candidate", "decoder")},
                           manifest=manifest, out=root / "report")
            with patch.object(bench, "prepare", side_effect=prepare), \
                    patch.object(bench, "measure", side_effect=measure), \
                    contextlib.redirect_stdout(io.StringIO()):
                report = bench.benchmark(args)
            self.assertTrue(report["temporary_outputs_cleaned"])
            self.assertFalse(any(p.exists() for p in snapshots))
            self.assertEqual(sorted(p.name for p in args.out.iterdir()), ["summary.json", "trials.jsonl"])
            self.assertEqual(json.loads((args.out / "summary.json").read_text())["status"], report["status"])
            self.assertEqual(len((args.out / "trials.jsonl").read_text().splitlines()), len(order))
            self.assertEqual({i["role"] for i in report["identities"]},
                             {"baseline", "candidate", "decoder", "manifest", "input"})
            return report, order

    def test_rotation_and_success(self):
        report, order = self.fixture()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(order, [(0, "baseline"), (0, "candidate"), (1, "candidate"), (1, "baseline")])
        self.assertEqual(bench.source_tools.trial_order(list(bench.MODES), 1, 0), ["candidate", "baseline"])

    def test_hash_mismatch_is_not_subset(self):
        report, order = self.fixture(wrong_hash=True)
        self.assertEqual(order, [])
        self.assertEqual(report["status"], "failed")
        self.assertTrue(all(s["status"] == "failed" for s in report["summaries"]))

    def test_mutations_and_failures_recorded_without_stopping_trials(self):
        for key in ("baseline", "candidate", "source"):
            report, order = self.fixture(mutate=key)
            self.assertEqual(len(order), 4)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(any("Immutable input changed" in e["error"] for e in report["errors"]))
        report, order = self.fixture(fail=True)
        self.assertEqual(len(order), 4)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(all(v is None for v in report["summaries"][1]["changes_vs_baseline"].values()))


if __name__ == "__main__":
    unittest.main()
