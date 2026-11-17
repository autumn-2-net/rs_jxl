from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

import benchmark_jpeg_transcode as harness
from benchmark_jpeg_transcode import summary


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.argv = ["--encoder", "lab", "--decoder", "djxl", "--manifest", "cases.json",
                     "--out", str(Path(folder.name) / "report.json")]

    def test_default_variant_flags_are_empty(self):
        args = harness.arguments(self.argv)
        self.assertEqual(args.baseline_arg, [])
        self.assertEqual(args.candidate_arg, [])

    def test_both_variants_accept_all_allowed_flags(self):
        for mode in ("off", "fast", "optimal", "source", "fast-forced", "optimal-forced"):
            flags = ["--cooperative", "--symmetric-ma", "--cfl-peak", "--paired-luma",
                     f"--ac-lz={mode}"]
            with self.subTest(mode=mode):
                args = harness.arguments(self.argv + ["--baseline", "lab"] + [
                    f"--{name}-arg={flag}"
                    for name in ("baseline", "candidate") for flag in flags
                ])
                self.assertEqual(args.baseline_arg, flags)
                self.assertEqual(args.candidate_arg, flags)

    def test_conflicting_ac_lz_flags_rejected_for_each_variant(self):
        for name in ("baseline", "candidate"):
            for first in ("off", "fast", "optimal", "source"):
                for second in ("off", "fast", "optimal", "source"):
                    if first == second:
                        continue
                    with self.subTest(name=name, first=first, second=second), \
                            redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                        harness.arguments(self.argv + [f"--{name}-arg=--ac-lz={first}",
                                                       f"--{name}-arg=--ac-lz={second}"])

    def test_ac_lz_modes_are_independent_and_identical_repeats_allowed(self):
        args = harness.arguments(self.argv + ["--baseline", "lab",
                                 "--baseline-arg=--ac-lz=optimal",
                                 "--baseline-arg=--ac-lz=optimal",
                                 "--candidate-arg=--ac-lz=fast"])
        self.assertEqual(harness.ac_lz_mode(args.baseline_arg), "optimal")
        self.assertEqual(args.baseline_arg, ["--ac-lz=optimal"])
        self.assertEqual(harness.ac_lz_mode(args.candidate_arg), "fast")

    def test_unknown_flags_and_ac_lz_modes_rejected(self):
        for name in ("baseline", "candidate"):
            for flag in ("--unknown", "--ac-lz=other"):
                with self.subTest(name=name, flag=flag), redirect_stderr(StringIO()), \
                        self.assertRaises(SystemExit):
                    harness.arguments(self.argv + [f"--{name}-arg={flag}"])


class ReceiptTests(unittest.TestCase):
    def test_context_and_order_receipts_and_conflicts(self):
        for flag in harness.BLOCK_CONTEXT_FLAGS + harness.AC_ORDER_FLAGS:
            key, value = flag[2:].split("=", 1)
            key = key.replace("-", "_")
            receipt = {key: value, "dc_thresholds": 0, "ac_block_contexts": 2}
            harness.validate_experiments(receipt, [flag], channels=1)
            with self.assertRaises(ValueError):
                harness.validate_experiments({}, [flag])
            with self.assertRaises(ValueError):
                harness.validate_experiments({key: "unknown"}, [flag])
        for flags in [
            ["--paired-luma", "--block-context=channel"],
            ["--block-context=source", "--block-context=channel"],
            ["--ac-order=source", "--ac-order=zigzag"],
        ]:
            with self.assertRaises(ValueError):
                harness.representation_modes(flags)

    def test_actual_context_counts_are_bounded_integers(self):
        harness.validate_experiments(dict(ac_block_contexts=16, dc_thresholds=0), [])
        for key in ["ac_block_contexts", "dc_thresholds"]:
            for invalid in [None, -1, True, 1.0, 17]:
                with self.assertRaises(ValueError):
                    harness.validate_experiments({key: invalid}, [])

    def test_actual_policy_counts_cannot_silently_disagree(self):
        for mode, cuts, gray, color in [
            ("channel", 0, 2, 3), ("quantile4", 3, 5, 12),
            ("activity5", 4, 6, 15), ("shared-chroma", 4, 6, 8),
            ("paired-luma", 4, 4, 9),
        ]:
            flags = [f"--block-context={mode}"]
            for channels, count in [(1, gray), (3, color)]:
                receipt = dict(block_context=mode, dc_thresholds=cuts, ac_block_contexts=count)
                harness.validate_experiments(receipt, flags, channels=channels)
                for key in ("dc_thresholds", "ac_block_contexts"):
                    missing = dict(receipt)
                    del missing[key]
                    with self.assertRaises(ValueError):
                        harness.validate_experiments(missing, flags, channels=channels)
                receipt["ac_block_contexts"] = count + 1
                with self.assertRaises(ValueError):
                    harness.validate_experiments(receipt, flags, channels=channels)
        for mode, cuts in [("channel", 4), ("quantile4", 4), ("activity5", 5)]:
            with self.assertRaises(ValueError):
                harness.validate_experiments(dict(block_context=mode, dc_thresholds=cuts,
                    ac_block_contexts=15), [f"--block-context={mode}"])

    def test_histogram_requests_and_dense_actual_count(self):
        for requested in range(1, 5):
            flags = [f'--ac-histograms={requested}']
            for used in range(1, requested + 1):
                harness.validate_experiments(dict(ac_histograms=requested, ac_histograms_used=used), flags)
            for used in [0, requested + 1, None, True, 1.0]:
                with self.assertRaises(ValueError):
                    harness.validate_experiments(dict(ac_histograms=requested, ac_histograms_used=used), flags)
            with self.assertRaises(ValueError):
                harness.validate_experiments({}, flags)
        with self.assertRaises(ValueError):
            harness.ac_histograms(['--ac-histograms=2', '--ac-histograms=3'])

    def test_old_receipts_and_default_source_receipts_accepted(self):
        harness.validate_experiments({}, [])
        harness.validate_experiments({"ac_lz": "source"}, [])
        harness.validate_size_breakdown({}, 100)

    def test_boolean_experiments_require_exact_receipts(self):
        for flag in harness.EXPERIMENT_FLAGS:
            key = flag[2:].replace("-", "_")
            harness.validate_experiments({key: True}, [flag])
            harness.validate_experiments({key: False}, [])
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                harness.validate_experiments({}, [flag])
            for value in (True, 0, 1, "true", None):
                with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                    harness.validate_experiments({key: value}, [])
            for value in (False, 0, 1, "true", None):
                with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                    harness.validate_experiments({key: value}, [flag])

    def test_ac_lz_overrides_require_exact_mode_receipts(self):
        for mode in ("off", "fast", "optimal", "source"):
            flags = [f"--ac-lz={mode}"]
            harness.validate_experiments({"ac_lz": mode, "ac_lz_enabled": False}, flags)
            with self.subTest(mode=mode, missing=True), self.assertRaises(ValueError):
                harness.validate_experiments({}, flags)
            for value in ("off", "fast", "optimal", "source", "other", True, 0, None):
                if value == mode:
                    continue
                with self.subTest(mode=mode, value=value), self.assertRaises(ValueError):
                    harness.validate_experiments({"ac_lz": value}, flags)

    def test_default_rejects_non_source_ac_lz_receipts(self):
        for value in ("off", "fast", "optimal", "other", True, 0, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                harness.validate_experiments({"ac_lz": value}, [])

    def test_receipt_validation_also_rejects_conflicting_modes(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            harness.validate_experiments({"ac_lz": "fast"}, ["--ac-lz=off", "--ac-lz=fast"])

    @staticmethod
    def breakdown():
        return dict(global_bits=8, aux_model_bits=8, dc_bits=8, ac_metadata_bits=8,
                    quant_bits=8, ac_model_bits=8, ac_data_bits=704, padding_bits=0,
                    header_bits=8, reconstruction_bits=8, metadata_bits=16, container_bits=16)

    def test_valid_breakdown_accounts_for_all_output_bits(self):
        harness.validate_size_breakdown({"size_breakdown": self.breakdown()}, 100)

    def test_breakdown_values_must_be_nonnegative_integers(self):
        for key in self.breakdown():
            for value in (-1, True, False, 8.0, "8", None, float("nan"), float("inf"), {}):
                breakdown = self.breakdown()
                breakdown[key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "bit count"):
                    harness.validate_size_breakdown({"size_breakdown": breakdown}, 100)

    def test_breakdown_requires_complete_named_fields(self):
        for key in self.breakdown():
            breakdown = self.breakdown()
            del breakdown[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                harness.validate_size_breakdown({"size_breakdown": breakdown}, 100)
        for breakdown in (None, [], 800, {}, {**self.breakdown(), "unknown_bits": 0}):
            with self.subTest(breakdown=breakdown), self.assertRaises(ValueError):
                harness.validate_size_breakdown({"size_breakdown": breakdown}, 100)

    def test_breakdown_sum_must_match_exactly(self):
        for delta in (-1, 1):
            breakdown = self.breakdown()
            breakdown["ac_data_bits"] += delta
            with self.subTest(delta=delta), self.assertRaisesRegex(ValueError, "sum"):
                harness.validate_size_breakdown({"size_breakdown": breakdown}, 100)


class MockBenchmarkTests(unittest.TestCase):
    def run_mock(self, root, *, failure=None, old_receipt=False):
        encoder, decoder = root / "lab.exe", root / "djxl.exe"
        encoder.write_bytes(b"mock encoder")
        decoder.write_bytes(b"mock decoder")
        image = BytesIO()
        Image.new("RGB", (8, 6)).save(image, format="JPEG")
        original = image.getvalue()
        source = root / "input.jpg"
        source.write_bytes(original)
        manifest = root / "cases.json"
        manifest.write_text(json.dumps({"cases": [{"original": str(source)}]}), encoding="utf-8")
        baseline_flags = ["--cooperative", "--symmetric-ma", "--cfl-peak"]
        candidate_flags = baseline_flags + ["--paired-luma"]
        if not old_receipt:
            baseline_flags.append("--ac-lz=optimal")
            candidate_flags.append("--ac-lz=fast")
        args = harness.arguments([
            "--encoder", str(encoder), "--baseline", str(encoder), "--decoder", str(decoder),
            "--manifest", str(manifest), "--out", str(root / "report.json"), "--repeats", "2",
            *[f"--baseline-arg={flag}" for flag in baseline_flags],
            *[f"--candidate-arg={flag}" for flag in candidate_flags],
        ])
        calls = []

        def process(cmd, env, timeout):
            calls.append(cmd)
            record = dict(command=cmd, returncode=0, stdout="", stderr="", wall_seconds=99)
            if cmd[0] == str(encoder.resolve()):
                flags = cmd[8:]
                name = "candidate" if "--paired-luma" in flags else "baseline"
                self.assertEqual(flags, candidate_flags if name == "candidate" else baseline_flags)
                Path(cmd[cmd.index("--output") + 1]).write_bytes(b"x" * 100)
                m = dict(bytes=100, jpeg_bytes=len(original), seconds=1.0, width=8, height=6,
                         bpp=800 / 48, mpix_per_second=48 / 1e6,
                         native_verified=True, head_tail=None)
                m.update({flag[2:].replace("-", "_"): flag in flags
                          for flag in harness.EXPERIMENT_FLAGS})
                if not old_receipt:
                    m.update(ac_lz="fast" if name == "candidate" else "optimal",
                             size_breakdown=ReceiptTests.breakdown())
                if failure == (name, "experiment"):
                    m["cooperative"] = False
                if failure == (name, "ac_lz"):
                    m["ac_lz"] = "off"
                if failure == (name, "size_breakdown"):
                    m["size_breakdown"]["padding_bits"] += 1
                record["stdout"] = json.dumps(m)
            else:
                self.assertEqual(cmd[0], str(decoder.resolve()))
                Path(cmd[2]).write_bytes(original)
            return record

        with patch.object(harness.port, "run_process", side_effect=process), redirect_stdout(StringIO()):
            if failure:
                with self.assertRaises(ValueError):
                    harness.benchmark(args)
            else:
                harness.benchmark(args)
        report = json.loads(args.out.read_text(encoding="utf-8"))
        return report, calls

    def test_same_binary_uses_independent_flags_and_receipts_for_both_variants(self):
        with TemporaryDirectory() as folder:
            report, calls = self.run_mock(Path(folder))
        self.assertEqual(report["status"], "passed")
        encodes = [cmd for cmd in calls if "--output" in cmd]
        self.assertEqual(["--paired-luma" in cmd for cmd in encodes], [False, True, True, False])
        self.assertEqual(len(calls), 8)
        for name, mode in (("baseline", "optimal"), ("candidate", "fast")):
            for sample in report["cases"][0]["variants"][name]:
                self.assertTrue(sample["verified"])
                self.assertEqual(sample["measurement"]["ac_lz"], mode)
                self.assertEqual(sample["measurement"]["size_breakdown"], ReceiptTests.breakdown())

    def test_old_binaries_do_not_require_ac_lz_or_size_breakdown(self):
        with TemporaryDirectory() as folder:
            report, _ = self.run_mock(Path(folder), old_receipt=True)
        self.assertEqual(report["status"], "passed")

    def test_each_variant_rejects_bad_receipts_before_decoding(self):
        for name in ("baseline", "candidate"):
            for field in ("experiment", "ac_lz", "size_breakdown"):
                with self.subTest(name=name, field=field), TemporaryDirectory() as folder:
                    report, calls = self.run_mock(Path(folder), failure=(name, field))
                    self.assertEqual(report["status"], "failed")
                    self.assertIn("ValueError", report["error"])
                    self.assertIn("--output", calls[-1])


class SummaryTests(unittest.TestCase):
    def test_disabled_parser_requires_actual_lz_deactivation(self):
        harness.validate_experiments(dict(ac_lz="off", ac_lz_enabled=False), ["--ac-lz=off"])
        for active in [True, None, 0, 1]:
            with self.assertRaises(ValueError):
                harness.validate_experiments(dict(ac_lz="off", ac_lz_enabled=active), ["--ac-lz=off"])

    def test_forced_parser_requires_actual_lz_activation(self):
        for mode in ['fast-forced', 'optimal-forced']:
            measurement = dict(ac_lz=mode, ac_lz_enabled=True)
            harness.validate_experiments(measurement, ['--ac-lz=' + mode])
            for active in [False, None, 1]:
                measurement['ac_lz_enabled'] = active
                with self.assertRaises(ValueError):
                    harness.validate_experiments(measurement, ['--ac-lz=' + mode])

    def test_distinguishes_equal_image_mean_and_pixel_weighted_bpp(self):
        cases = [
            dict(pixels=100, original_bytes=500, summaries=dict(candidate=dict(bytes=100, seconds=1, bpp=8))),
            dict(pixels=900, original_bytes=500, summaries=dict(candidate=dict(bytes=450, seconds=3, bpp=4))),
        ]
        result = summary(cases, 'candidate')
        self.assertEqual(result['bytes'], 550)
        self.assertEqual(result['seconds'], 4)
        self.assertEqual(result['bpp'], 4.4)
        self.assertEqual(result['mean_bpp'], 6)
        self.assertEqual(result['jpeg_ratio'], .55)
        self.assertEqual(result['mpix_per_second'], .00025)


if __name__ == '__main__':
    unittest.main()
