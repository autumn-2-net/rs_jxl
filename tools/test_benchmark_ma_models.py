import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark_ma_models import (
    arguments, benchmark, profile_rows, summarize, trial_order, validate_measurement,
)


class ModelMeasurements(unittest.TestCase):
    def argv(self, output):
        return ["--encoder", "encoder", "--decoder", "decoder", "--case", "input.png",
                "--out", str(output)]

    def test_prefix_parser_defaults_and_explicit_levels(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv = self.argv(Path(temporary) / "report.json")
            self.assertEqual(arguments(argv).prefix_levels, [0])
            self.assertEqual(arguments(argv + ["--prefix-levels", "0", "2", "4", "6", "255"])
                             .prefix_levels, [0, 2, 4, 6, 255])
            for values in [["-1"], ["256"], ["x"], ["2", "2"], []]:
                with self.subTest(values=values), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        arguments(argv + ["--prefix-levels"] + values)

    def test_prefix_confirmation_requires_exact_integer_and_legacy_only_for_zero(self):
        validate_measurement(self.measurement(), "quarter", 100, 0)
        with self.assertRaises(ValueError):
            validate_measurement(self.measurement(), "quarter", 100, 4)
        for levels in [0, 2, 255]:
            data = self.measurement()
            data["ma_prefix_levels"] = levels
            validate_measurement(data, "quarter", 100, levels)
        for value in [None, True, False, "4", 4.0, 0, 2, 256]:
            data = self.measurement()
            data["ma_prefix_levels"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_measurement(data, "quarter", 100, 4)

    def test_harness_cross_product_forwarding_keys_and_alternation(self):
        for levels in [[0], [4], [0, 2, 4, 6]]:
            with self.subTest(levels=levels), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for name in ["encoder", "decoder", "input.png"]:
                    (root / name).touch()
                args = arguments([
                    "--encoder", str(root / "encoder"), "--decoder", str(root / "decoder"),
                    "--case", str(root / "input.png"), "--out", str(root / "report.json"),
                    "--modes", "off", "half", "--prefix-levels", *map(str, levels),
                    "--warmups", "0", "--repeats", "2",
                ])
                calls = []

                def process(command, env, timeout):
                    if command[0] == str(args.encoder):
                        mode = command[command.index("--description-cost") + 1]
                        prefix = (int(command[command.index("--ma-prefix-levels") + 1])
                                  if "--ma-prefix-levels" in command else 0)
                        calls.append((mode, prefix))
                        Path(command[command.index("--output") + 1]).write_bytes(b"x" * 100)
                        data = self.measurement()
                        data.update(description_cost=mode, ma_prefix_levels=prefix)
                        # A frozen encoder may omit the new field for the zero baseline.
                        if prefix == 0:
                            del data["ma_prefix_levels"]
                        return dict(stdout=json.dumps(data), stderr="")
                    Path(command[-1]).touch()
                    return dict(stdout="", stderr="")

                with patch("benchmark_ma_models.run_process", side_effect=process), \
                        patch("benchmark_ma_models.check_process"), \
                        patch("benchmark_ma_models.file_sha256", return_value="digest"), \
                        patch("benchmark_ma_models.png_samples", return_value=SimpleNamespace(
                            width=10, height=10, depth=8)), \
                        patch("benchmark_ma_models.compare_pixels"), \
                        patch("benchmark_ma_models.decoder_command",
                              side_effect=lambda decoder, jxl, png, depth: [str(decoder), str(png)]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    report = benchmark(args)
                self.assertEqual(report["status"], "passed", report["errors"])
                variants = [(mode, prefix) for mode in ["off", "half"] for prefix in levels]
                self.assertEqual(calls, trial_order(variants, 0, 0) + trial_order(variants, 1, 0))
                expected_keys = (["off", "half"] if levels == [0] else
                                 [f"{mode}/prefix-{prefix}" for mode, prefix in variants])
                case = report["cases"][0]
                self.assertEqual(list(case["modes"]), expected_keys)
                self.assertEqual(list(case["summaries"]), expected_keys)
                for key, (mode, prefix) in zip(expected_keys, variants):
                    for sample in case["modes"][key]:
                        self.assertEqual(sample["config"], dict(description_cost=mode,
                                                                ma_prefix_levels=prefix))
                        self.assertTrue(sample["verified"])

    def test_each_variant_leads_and_two_way_runs_alternate(self):
        self.assertEqual([trial_order(["a", "b"], r, 0) for r in range(4)],
                         [["a", "b"], ["b", "a"], ["a", "b"], ["b", "a"]])
        for count in [2, 3, 4, 6]:
            variants = list(range(count))
            for case_index in range(count):
                rounds = [trial_order(variants, r, case_index) for r in range(count)]
                self.assertEqual(sorted(order[0] for order in rounds), variants)
                for order in rounds:
                    self.assertEqual(sorted(order), variants)

    def measurement(self):
        return dict(seconds=0.5, bytes=100, description_cost="Quarter", search=False,
                    model_costs=dict(tree_bits=40, entropy_header_bits=80, total_bits=120))

    def test_validated_actual_costs_and_native_settings(self):
        validate_measurement(self.measurement(), "quarter", 100)

    def test_invalid_measurements_are_not_accepted_as_success(self):
        for key, value in [("seconds", 0), ("seconds", True), ("seconds", float("nan")),
                           ("bytes", 99), ("search", True), ("description_cost", "Off"),
                           ("model_costs", None)]:
            data = self.measurement()
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_measurement(data, "quarter", 100)
        for key, value in [("tree_bits", -1), ("tree_bits", True), ("total_bits", 121)]:
            data = self.measurement()
            data["model_costs"][key] = value
            with self.assertRaises(ValueError):
                validate_measurement(data, "quarter", 100)

    def test_receipt_cannot_exceed_entire_file(self):
        data = self.measurement()
        data["model_costs"] = dict(tree_bits=400, entropy_header_bits=500, total_bits=900)
        with self.assertRaises(ValueError):
            validate_measurement(data, "quarter", 100)

    def test_profiles_are_structured_and_not_wall_time(self):
        rows = profile_rows("note\nma.find_best_split | 23 | 4.5 | 0.2\n"
                            "ma.ready_frontier.records_le256.width_1 | 21 | 0.0 | 0.0\n")
        self.assertEqual(rows["ma.find_best_split"]["calls"], 23)
        self.assertEqual(rows["ma.find_best_split"]["seconds"], 4.5)
        self.assertEqual(len(rows), 2)

    def test_summary_excludes_warmups_and_preserves_size_variation(self):
        samples = []
        for seconds, size, warmup in [(100, 999, True), (1, 100, False), (3, 102, False)]:
            data = copy.deepcopy(self.measurement())
            data.update(seconds=seconds, bytes=size)
            samples.append(dict(measurement=data, warmup=warmup))
        result = summarize(samples, 200)
        self.assertEqual(result["seconds"], 2)
        self.assertEqual(result["sizes"], [100, 102])
        self.assertEqual(result["bpp"], [4, 4.08])
        self.assertEqual(result["mpix_s"], 0.0001)


if __name__ == "__main__":
    unittest.main()
