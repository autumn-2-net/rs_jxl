"""Small synthetic report tests; no encodes, builds, network or real benchmarks."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import statistics
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_patch_ma_followup as summary


def trial(repetition, size, seconds, pixels, counters=None):
    measurement = dict(bytes=size, seconds=seconds, bpp=8 * size / pixels,
                       mpix_per_second=pixels / 1e6 / seconds,
                       refinement=counters)
    return dict(repetition=repetition, warmup=repetition < 0, verified=True,
                included_in_timing=repetition >= 0, measurement=measurement,
                encoder=dict(returncode=0), decoder=dict(returncode=0))


def refresh(case, variant):
    samples = [s["measurement"] for s in case["variants"][variant] if not s["warmup"]]
    seconds = statistics.median(s["seconds"] for s in samples)
    size = statistics.median(s["bytes"] for s in samples)
    case["summaries"][variant] = dict(bytes=size, seconds=seconds,
                                     bpp=8 * size / case["pixels"],
                                     mpix_per_second=case["pixels"] / 1e6 / seconds,
                                     sizes=sorted({s["bytes"] for s in samples}))


def report():
    cases = []
    # Strongly unequal pixels distinguish arithmetic mean bpp from weighted bpp.
    for index, (pixels, base_size, size, base_times, times) in enumerate((
            (100, 100, 80, [1, 2, 30], [2, 4, 80]),
            (900, 200, 250, [3, 4, 90], [2, 3, 100]),
            (1000, 300, 300, [4, 5, 200], [5, 7, 300]))):
        case = dict(source=f"image-{index}.png", pixels=pixels, width=pixels, height=1,
                    variants={}, summaries={})
        for variant, nbytes, seconds in (("off", base_size, base_times),
                                          ("candidate", size, times)):
            samples = [trial(-1, nbytes + 100, 999, pixels,
                             {k: 9000 for k in summary.OFFSET_COUNTERS})]
            for repetition, value in enumerate(seconds):
                counters = ({k: (index + 1) * 10 + repetition for k in summary.OFFSET_COUNTERS}
                            if variant == "candidate" else None)
                samples.append(trial(repetition, nbytes, value, pixels, counters))
            case["variants"][variant] = samples
            refresh(case, variant)
        cases.append(case)
    return dict(status="passed", errors=[], completed=3,
                config=dict(variants=["off", "candidate"], repeats=3, warmups=1), cases=cases)


def with_diagnostics(data):
    for index, case in enumerate(data["cases"]):
        case["diagnostics"] = {}
        for variant in data["config"]["variants"]:
            selected = variant == "candidate" and index != 1
            case["diagnostics"][variant] = [dict(
                verified=True, included_in_timing=False,
                patch_application=dict(source_patch_selected=selected,
                                       reference_seconds=999 if selected else None,
                                       residual_seconds=888 if selected else None))]
    data["config"]["profile"] = True
    return data


def add_aggregate(data):
    result = summary.summarize(data, "off")
    data["aggregate"] = {
        variant: dict(bytes=value["total_bytes"], seconds=value["native_seconds"],
                      bpp=value["weighted_bpp"], mpix_per_second=value["mpix_per_second"])
        for variant, value in result["overall"].items()}
    return data


class MetricTests(unittest.TestCase):
    def test_patch_ma_producer_aggregate_schema(self):
        from benchmark_source_patch_detector import aggregate
        data = report()
        data["aggregate"] = aggregate(data["cases"], data["config"]["variants"])
        result = summary.summarize(data, "off")
        self.assertEqual(result["overall"]["off"]["total_bytes"], 600)
        data["aggregate"]["off"]["mean_bpp"] += 1
        with self.assertRaises(ValueError):
            summary.summarize(data, "off")

    def test_actual_bytes_mean_and_weighted_bpp_native_medians_and_units(self):
        result = summary.summarize(report(), "off")
        base = result["overall"]["off"]
        value = result["overall"]["candidate"]
        self.assertEqual(base["total_bytes"], 600)
        self.assertIs(type(value["total_bytes"]), int)
        self.assertEqual(value["total_bytes"], 630)
        self.assertAlmostEqual(value["mean_image_bpp"], (6.4 + 2000 / 900 + 2.4) / 3)
        self.assertEqual(value["weighted_bpp"], 8 * 630 / 2000)
        self.assertNotEqual(value["weighted_bpp"], value["mean_image_bpp"])
        self.assertEqual(base["native_seconds"], 11)
        self.assertEqual(value["native_seconds"], 14)
        self.assertEqual(value["mpix_per_second"], 2000 / 1e6 / 14)
        self.assertEqual(value["total_mib"], 630 / 1048576)
        self.assertAlmostEqual(value["total_bytes_ratio_to_baseline"], 1.05)
        self.assertEqual(value["native_seconds_ratio_to_baseline"], 14 / 11)
        self.assertEqual((value["byte_increases"], value["byte_decreases"], value["byte_ties"]),
                         (1, 1, 1))
        self.assertEqual(base["byte_ties"], 3)
        self.assertEqual(result["images"], 3)

    def test_largest_regressions_filter_gains_and_sort(self):
        value = summary.summarize(report(), "off")["overall"]["candidate"]
        regressions = value["regressions"]
        self.assertEqual([r["source"] for r in regressions["largest_byte_increases"]],
                         ["image-1.png"])
        self.assertEqual(regressions["largest_byte_increases"][0]["byte_delta"], 50)
        self.assertEqual([r["source"] for r in regressions["largest_time_regressions_seconds"]],
                         ["image-0.png", "image-2.png"])
        self.assertAlmostEqual(regressions["largest_time_regressions_pct"][0]["time_change_pct"], 100)

    def test_offset_first_timed_sample_not_repeat_sum_or_warmup(self):
        data = report()
        for case in data["cases"]:
            case["variants"]["candidate"].reverse()
        value = summary.summarize(data, "off")["overall"]["candidate"]
        self.assertEqual(value["offset_counters"], {k: 60 for k in summary.OFFSET_COUNTERS})
        self.assertEqual(value["offset_counter_images"], 3)
        result = summary.summarize(data, "off")
        self.assertEqual(result["overall"]["off"]["offset_counters"], {})
        self.assertEqual(result["overall"]["off"]["offset_counter_images"], 0)
        self.assertIn("no multiplication by repeats", result["contracts"]["offset"])

    def test_diagnostic_actual_activation_only_and_never_timing(self):
        data = with_diagnostics(report())
        for case in data["cases"]:
            for samples in case["variants"].values():
                for sample in samples:
                    sample["measurement"].update(selected_mode="source", source_patch_selected=True)
        result = summary.summarize(data, "off")["overall"]
        self.assertEqual(result["off"]["source_patch_activation_count"], 0)
        self.assertEqual(result["candidate"]["source_patch_activation_count"], 2)
        self.assertEqual(result["candidate"]["source_patch_diagnostic_images"], 3)
        self.assertEqual(result["candidate"]["native_seconds"], 14)
        self.assertIsNone(summary.summarize(report(), "off")["overall"]["candidate"]
                          ["source_patch_activation_count"])

    def test_refinement_schema_without_diagnostics_or_inclusion_flag(self):
        data = report()
        for case in data["cases"]:
            for samples in case["variants"].values():
                for sample in samples:
                    del sample["included_in_timing"]
        data.pop("completed")
        result = summary.summarize(data, "candidate")
        self.assertEqual(result["baseline"], "candidate")
        self.assertEqual(result["overall"]["off"]["byte_decreases"], 1)

    def test_modes_schema_and_checked_aggregate(self):
        data = add_aggregate(report())
        data["config"]["modes"] = data["config"].pop("variants")
        self.assertEqual(summary.summarize(data, "off")["overall"]["off"]["total_bytes"], 600)
        data["aggregate"]["off"]["bytes"] += 1
        with self.assertRaisesRegex(ValueError, "aggregate bytes"):
            summary.summarize(data, "off")

    def test_strata_totals_and_manifest_does_not_expand_cohort(self):
        data = report()
        manifest = dict(cases=[dict(original=c["source"], stratum="small" if i == 0 else "large")
                               for i, c in enumerate(data["cases"])])
        manifest["cases"].append(dict(original="extra.png", stratum="unused"))
        result = summary.summarize(data, "off", manifest)
        self.assertEqual(set(result["strata"]), {"small", "large"})
        small = result["strata"]["small"]["candidate"]
        large = result["strata"]["large"]["candidate"]
        self.assertEqual(small["total_bytes"], 80)
        self.assertEqual(large["total_bytes"], 550)
        self.assertEqual(large["native_seconds"], 10)
        self.assertEqual(large["offset_counters"]["offset_reused_tokens"], 50)
        self.assertEqual(result["images"], 3)

    def test_windows_manifest_path_slashes_case_and_relative_metadata(self):
        data = report()
        for i, case in enumerate(data["cases"]):
            case["source"] = f"D:\\Corpus\\image-{i}.png"
        manifest = dict(cases=[dict(original=f"d:/corpus/IMAGE-{i}.png", stratum="x")
                               for i in range(3)])
        self.assertEqual(summary.summarize(data, "off", manifest)["strata"]["x"]["off"]["images"], 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = report()
            for case in data["cases"]:
                case["source"] = str(root / case["source"])
            manifest = dict(cases=[dict(original=f"image-{i}.png") for i in range(3)])
            self.assertEqual(summary.summarize(data, "off", manifest, root)["images"], 3)


class RejectionTests(unittest.TestCase):
    def test_failed_incomplete_missing_and_inconsistent_reports(self):
        mutations = {
            "failed": lambda d: d.update(status="failed"),
            "running": lambda d: d.update(status="running"),
            "errors": lambda d: d.update(errors=["failure"]),
            "cleanup": lambda d: d.update(temporary_outputs_cleaned=False),
            "empty": lambda d: d.update(cases=[]),
            "completed": lambda d: d.update(completed=4),
            "lost_case": lambda d: d["cases"].pop(),
            "duplicate_source": lambda d: d["cases"][1].update(source="image-0.png"),
            "missing_summary": lambda d: d["cases"][0]["summaries"].pop("candidate"),
            "missing_variant": lambda d: d["cases"][0]["variants"].pop("candidate"),
            "extra_variant": lambda d: d["cases"][0]["variants"].update(extra=[]),
            "duplicate_variant": lambda d: d["config"]["variants"].append("off"),
            "pixels": lambda d: d["cases"][0].update(pixels=101),
            "case_error": lambda d: d["cases"][0].update(error="failure"),
            "empty_trials": lambda d: d["cases"][0]["variants"].update(candidate=[]),
            "missing_repeat": lambda d: d["cases"][0]["variants"]["candidate"].pop(),
            "duplicate_repeat": lambda d: d["cases"][0]["variants"]["candidate"][2].update(repetition=0),
            "unverified": lambda d: d["cases"][0]["variants"]["candidate"][1].update(verified=False),
            "failed_warmup": lambda d: d["cases"][0]["variants"]["candidate"][0].update(verified=False),
            "process_error": lambda d: d["cases"][0]["variants"]["candidate"][1]["encoder"].update(returncode=1),
            "warmup_flag": lambda d: d["cases"][0]["variants"]["candidate"][1].update(warmup=True),
            "timing_flag": lambda d: d["cases"][0]["variants"]["candidate"][1].update(included_in_timing=False),
            "summary_bytes": lambda d: d["cases"][0]["summaries"]["candidate"].update(bytes=81),
            "summary_time": lambda d: d["cases"][0]["summaries"]["candidate"].update(seconds=5),
            "summary_bpp": lambda d: d["cases"][0]["summaries"]["candidate"].update(bpp=5),
            "summary_sizes": lambda d: d["cases"][0]["summaries"]["candidate"].update(sizes=[80, 81]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                data = report()
                mutate(data)
                with self.assertRaises(ValueError):
                    summary.summarize(data, "off")
        with self.assertRaisesRegex(ValueError, "Missing baseline"):
            summary.summarize(report(), "unknown")

    def test_invalid_numeric_values_and_units(self):
        for field in ("seconds", "bytes", "bpp", "mpix_per_second"):
            for value in (None, True, "1", -1, 0, float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    data = report()
                    data["cases"][0]["variants"]["candidate"][1]["measurement"][field] = value
                    with self.assertRaises(ValueError):
                        summary.summarize(data, "off")
        data = report()
        data["cases"][0]["variants"]["candidate"][1]["measurement"]["mpix_per_second"] *= 1e6
        with self.assertRaisesRegex(ValueError, "mpix_per_second"):
            summary.summarize(data, "off")

    def test_varying_sizes_cannot_make_fractional_actual_bytes(self):
        data = report()
        for case in data["cases"]:
            for variant, samples in case["variants"].items():
                samples.pop()
                refresh(case, variant)
        data["config"]["repeats"] = 2
        case = data["cases"][0]
        measurement = case["variants"]["candidate"][2]["measurement"]
        measurement["bytes"] += 1
        measurement["bpp"] = 8 * measurement["bytes"] / case["pixels"]
        refresh(case, "candidate")
        self.assertEqual(case["summaries"]["candidate"]["bytes"], 80.5)
        with self.assertRaisesRegex(ValueError, "varying timed bytes"):
            summary.summarize(data, "off")

    def test_offset_bad_counters_and_missing_case_coverage(self):
        for value in (-1, True, 1.5, None, float("nan")):
            data = report()
            data["cases"][0]["variants"]["candidate"][1]["measurement"]["refinement"]["offset_reused_tokens"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summary.summarize(data, "off")
        data = report()
        data["cases"][1]["variants"]["candidate"][1]["measurement"]["refinement"].pop("offset_reused_tokens")
        with self.assertRaisesRegex(ValueError, "counter coverage"):
            summary.summarize(data, "off")

    def test_inconsistent_trial_schedule_without_config_counts(self):
        data = report()
        del data["config"]["repeats"]
        data["cases"][1]["variants"]["candidate"].pop()
        refresh(data["cases"][1], "candidate")
        with self.assertRaisesRegex(ValueError, "schedules"):
            summary.summarize(data, "off")

    def test_missing_or_failed_diagnostics(self):
        mutations = [
            lambda d: d["cases"][0].pop("diagnostics"),
            lambda d: d["cases"][0]["diagnostics"].pop("candidate"),
            lambda d: d["cases"][0]["diagnostics"].update(candidate=[]),
            lambda d: d["cases"][0]["diagnostics"]["candidate"][0].update(verified=False),
            lambda d: d["cases"][0]["diagnostics"]["candidate"][0].update(included_in_timing=True),
            lambda d: d["cases"][0]["diagnostics"]["candidate"][0].pop("patch_application"),
            lambda d: d["cases"][0]["diagnostics"]["candidate"][0]["patch_application"].update(source_patch_selected=None),
            lambda d: d["cases"][0]["diagnostics"]["candidate"][0]["patch_application"].update(reference_seconds=-1),
            lambda d: d["cases"][0]["diagnostics"]["off"][0]["patch_application"].update(reference_seconds=0),
        ]
        for index, mutate in enumerate(mutations):
            data = with_diagnostics(report())
            mutate(data)
            with self.subTest(index=index), self.assertRaises(ValueError):
                summary.summarize(data, "off")

    def test_invalid_manifest(self):
        good = dict(cases=[dict(original=f"image-{i}.png", stratum="x") for i in range(3)])
        for mutate in (lambda m: m["cases"].pop(),
                       lambda m: m["cases"].append(m["cases"][0]),
                       lambda m: m["cases"][0].update(stratum=""),
                       lambda m: m["cases"][0].update(pixels=999)):
            manifest = deepcopy(good)
            mutate(manifest)
            with self.assertRaises(ValueError):
                summary.summarize(report(), "off", manifest)

    def test_failed_aggregate_and_lost_case_even_without_completed(self):
        for mutate in (lambda d: d["aggregate"].pop("candidate"),
                       lambda d: d["aggregate"]["candidate"].update(status="incomplete"),
                       lambda d: d["cases"].pop()):
            data = add_aggregate(report())
            del data["completed"]
            mutate(data)
            with self.assertRaises(ValueError):
                summary.summarize(data, "off")


class OutputTests(unittest.TestCase):
    def run_main(self, argv):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return summary.main(argv)

    def test_cli_json_markdown_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            source.write_text(json.dumps(report()), encoding="ascii")
            prefix = root / "nested" / "result"
            argv = ["--report", str(source), "--baseline", "off", "--out", str(prefix)]
            self.assertEqual(self.run_main(argv), 0)
            output = summary.load_json(root / "nested" / "result.summary.json")
            self.assertEqual(output["overall"]["candidate"]["total_bytes"], 630)
            text = (root / "nested" / "result.summary.md").read_text(encoding="ascii")
            for phrase in ("Mean image bpp", "Weighted bpp", "Offset Counters",
                           "Largest Regressions", "unavailable", "no multiplication by repeats"):
                self.assertIn(phrase, text)
            self.assertEqual(self.run_main(argv), 1)
            self.assertEqual(json.loads(source.read_text(encoding="ascii")), report())

    def test_multiple_reports_keep_independent_cohorts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.json", root / "second.json"
            first.write_text(json.dumps(report()), encoding="ascii")
            data = report()
            data["cases"] = data["cases"][:1]
            data["completed"] = 1
            second.write_text(json.dumps(data), encoding="ascii")
            argv = ["--report", str(first), "--report", str(second),
                    "--baseline", "off", "--out", str(root / "out")]
            self.assertEqual(self.run_main(argv), 0)
            a = summary.load_json(root / "out.first.summary.json")
            b = summary.load_json(root / "out.second.summary.json")
            self.assertEqual((a["images"], b["images"]), (3, 1))
            self.assertEqual((a["overall"]["off"]["total_bytes"],
                              b["overall"]["off"]["total_bytes"]), (600, 100))

    def test_all_reports_validated_before_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.json", root / "second.json"
            first.write_text(json.dumps(report()), encoding="ascii")
            second.write_text('{"status":"failed"}', encoding="ascii")
            self.assertEqual(self.run_main([
                "--report", str(first), "--report", str(second), "--baseline", "off",
                "--out", str(root / "out")]), 1)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["first.json", "second.json"])

    def test_output_input_and_multi_report_name_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "out.summary.json"
            source.write_text(json.dumps(report()), encoding="ascii")
            argv = ["--report", str(source), "--baseline", "off", "--out", str(root / "out")]
            self.assertEqual(self.run_main(argv), 1)
            self.assertEqual(self.run_main(argv + ["--report", str(source)]), 1)
            nested = root / "nested"
            nested.mkdir()
            other = nested / source.name
            other.write_text(source.read_text(encoding="ascii"), encoding="ascii")
            self.assertEqual(self.run_main(argv + ["--report", str(other)]), 1)

    def test_json_duplicate_members_and_nonfinite_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for text in ('{"status":"passed","status":"failed"}', '{"a":NaN}',
                         '{"a":Infinity}', '{"a":1e999}'):
                path.write_text(text, encoding="ascii")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    summary.load_json(path)

    def test_optional_plot_is_lazy_and_cli_uses_separate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            source.write_text(json.dumps(report()), encoding="ascii")
            argv = ["--report", str(source), "--baseline", "off", "--out", str(root / "out")]
            with patch.object(summary, "plot_summary") as plot:
                self.assertEqual(self.run_main(argv + ["--plot"]), 0)
                paths = plot.call_args.args[1]
                self.assertEqual([p.name for p in paths],
                                 ["out.bpp-v-time.png", "out.totalMiB-v-time.png"])

    def test_plot_points_use_arithmetic_mean_bpp_and_correct_time_units(self):
        calls, saved = [], []

        class Axes:
            def scatter(self, x, y, label):
                calls.append((x, y, label))

            def set(self, **kwargs):
                pass

            def legend(self, **kwargs):
                pass

            def grid(self, **kwargs):
                pass

        class Figure:
            def savefig(self, path, **kwargs):
                saved.append(path)

        matplotlib = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")
        matplotlib.use = lambda backend: None
        pyplot.subplots = lambda **kwargs: (Figure(), Axes())
        pyplot.close = lambda figure: None
        matplotlib.pyplot = pyplot
        result = summary.summarize(report(), "off")
        result["input_report"] = "example.json"
        with patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}):
            summary.plot_summary(result, [Path("bpp.png"), Path("mib.png")])
        self.assertEqual(saved, [Path("bpp.png"), Path("mib.png")])
        self.assertEqual(calls[1], ([14.0], [result["overall"]["candidate"]["mean_image_bpp"]], "candidate"))
        self.assertEqual(calls[3], ([14.0], [630 / 1048576], "candidate"))


if __name__ == "__main__":
    unittest.main()
