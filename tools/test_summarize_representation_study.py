"""Tiny report fixtures only: no codec, build, image access or corpus execution."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_representation_study as study


def work(variant, index, repetition):
    if variant == "off":
        return None
    search = variant != "first"
    formed = index + 2
    emitted = index != 1
    result = dict.fromkeys(study.SOURCE_COUNTERS, 0)
    result.update(retained_groups=formed, occurrences=formed * 3, searched_groups=2 if search else 0,
                  variable_sample_groups=1 if search else 0, changed_groups=int(search and index == 0),
                  emitted_plans=int(emitted), atlas_pixels=100 if emitted else 0,
                  dictionary_positions=4 if emitted else 0)
    result.update(mode=variant, selection_seconds=(0.01, 0.03, 0.02)[repetition % 3] if search else 0)
    return result


def event(mode, pixels, scope="local", fallback=False, used=False):
    delta, zero = mode.startswith("exact-delta"), mode.startswith("exact-zero")
    exact = delta and not fallback
    deltas = 3 if exact else 1 if zero and not fallback else 0
    predictor = mode.rsplit("-", 1)[1].title() if deltas else "Zero"
    return dict(scope=scope, requested_mode=("ExactDelta" if delta else "ExactZero")
                + "(" + mode.rsplit("-", 1)[1].title() + ")",
                width=pixels, height=1, begin=0, channels=1 if fallback else 4, limit=1024,
                exact_delta_attempted=delta, exact_delta_produced=exact, exact_zero_attempted=zero,
                zero_mask_eligible=zero and not fallback, zero_mask_matches=pixels if zero and used else 0,
                absolute_attempted=not exact, absolute_produced=not exact,
                colors=0 if exact else 2, deltas=deltas, predictor=predictor,
                delta_pixels=pixels if exact or zero and used else 0,
                cost_before=10, cost_after=None, cost_bypassed=True, outcome="accepted")


def receipt(mode, index, pixels):
    events = []
    if mode == "exact-delta-left" and index < 2:
        events = [event(mode, pixels, scope="global" if index == 0 else "local", fallback=index == 1)]
    if mode == "exact-zero-left":
        events = [event(mode, pixels, fallback=index == 2, used=index == 1)]
    return dict(scope=study.SCOPE, route="full_image", attempts=events)


def report(kind="source"):
    modes = ["off", "first", "medoid", "entropy"] if kind == "source" else list(study.MODES)
    base = "first" if kind == "source" else "absolute"
    cases = []
    for index, pixels in enumerate((100, 900, 1000)):
        sizes = {base: (100, 200, 300)[index], "off": (120, 180, 400)[index],
                 "medoid": (80, 250, 300)[index], "entropy": (110, 190, 330)[index],
                 "exact-delta-left": (80, 250, 300)[index], "exact-zero-left": (110, 190, 330)[index]}
        times = {base: (2, 4, 5)[index], "off": (3, 5, 6)[index],
                 "medoid": (4, 3, 7)[index], "entropy": (5, 6, 7)[index]}
        case = dict(source=f"image-{index}.png", sha256=str(index) * 64, width=pixels, height=1,
                    pixels=pixels, variants={}, summaries={})
        for mode in modes:
            size = sizes.get(mode, sizes[base] + 20)
            seconds = times.get(mode, times[base] + 1)
            samples = []
            for r in (-1, 0, 1, 2):
                elapsed = 999 if r < 0 else seconds * (0.5, 1, 2)[r]
                m = dict(bytes=size, seconds=elapsed, width=pixels, height=1, bpp=8 * size / pixels,
                         mpix_per_second=pixels / 1e6 / elapsed)
                if kind == "source":
                    m.update(representative_work=work(mode, index, r), selected_mode="source",
                             source_detector_enabled=True, source_patch_selected=None)
                else:
                    m.update(palette_evidence=receipt(mode, index, pixels), invariant_options="same e9 settings")
                samples.append(dict(repetition=r, warmup=r < 0, included_in_timing=r >= 0,
                                    verified=True, encoder=dict(returncode=0), decoder=dict(returncode=0), measurement=m))
            case["variants"][mode] = samples
            case["summaries"][mode] = dict(bytes=size, seconds=seconds, bpp=8 * size / pixels,
                                          mpix_per_second=pixels / 1e6 / seconds)
        cases.append(case)
    return dict(status="passed", errors=[], temporary_outputs_cleaned=True,
                config={"variants" if kind == "source" else "modes": modes, "repeats": 3, "warmups": 1}, cases=cases)


def manifest():
    return dict(cases=[dict(original=f"image-{index}.png", sha256=str(index) * 64,
                            stratum="small" if index == 0 else "large") for index in range(3)])


class Metrics(unittest.TestCase):
    def test_source_metrics_losses_and_oracle_all_variant_cost(self):
        result = study.summarize(report())
        self.assertEqual(result["baseline"], "first")
        self.assertEqual(result["images"], 3)
        base, medoid = result["overall"]["first"], result["overall"]["medoid"]
        self.assertEqual(base["total_bytes"], 600)
        self.assertEqual(medoid["total_bytes"], 630)
        self.assertEqual(medoid["native_seconds"], 14)
        self.assertAlmostEqual(medoid["mean_image_bpp"], (6.4 + 2000 / 900 + 2.4) / 3)
        self.assertEqual(medoid["weighted_bpp"], 8 * 630 / 2000)
        self.assertNotEqual(medoid["weighted_bpp"], medoid["mean_image_bpp"])
        self.assertEqual(medoid["mpix_per_second"], 2000 / 1e6 / 14)
        self.assertEqual(medoid["total_mib"], 630 / 2**20)
        self.assertEqual((medoid["size_wins"], medoid["size_ties"], medoid["size_losses"]), (1, 1, 1))
        self.assertEqual(medoid["bytes_saved_on_wins"], 20)
        self.assertEqual(medoid["bytes_added_on_losses"], 50)
        self.assertEqual(medoid["net_bytes_saved"], -30)
        self.assertEqual(medoid["activation_by_size_result"]["loss"], {"not_emitted": 1})
        self.assertGreater(medoid["total_bytes_ratio_to_baseline"], 1)
        self.assertEqual(result["oracle"]["total_bytes"], 560)
        self.assertEqual(result["oracle"]["native_seconds"], 57)
        self.assertEqual(result["oracle"]["selection_counts"], {"medoid": 1, "off": 1, "first": 1})
        self.assertEqual(result["oracle"]["size_losses"], 0)
        self.assertEqual(result["oracle"]["net_bytes_saved"], 40)

    def test_strata_use_entire_common_cohort_and_ties_prefer_baseline(self):
        result = study.summarize(report(), manifest())
        self.assertEqual(set(result["strata"]), {"small", "large"})
        self.assertEqual(result["strata"]["small"]["variants"]["medoid"]["images"], 1)
        self.assertEqual(result["strata"]["large"]["variants"]["medoid"]["images"], 2)
        self.assertEqual(result["cases"][2]["oracle"]["winner"], "first")
        data = report()
        data["config"]["variants"] = ["medoid", "first", "off", "entropy"]
        self.assertEqual(study.summarize(data)["cases"][2]["oracle"]["winner"], "first")

    def test_source_counts_once_per_image_selection_medians_not_repeat_sum(self):
        data = report()
        for case in data["cases"]:
            case["variants"]["medoid"].reverse()
        result = study.summarize(data)["overall"]["medoid"]["source"]
        self.assertEqual(result["formed_groups"], 9)
        self.assertEqual(result["counters"]["emitted_plans"], 2)
        self.assertEqual(result["counters"]["searched_groups"], 6)
        self.assertEqual(result["counters"]["variable_sample_groups"], 3)
        self.assertEqual(result["counters"]["changed_groups"], 1)
        self.assertAlmostEqual(result["selection_seconds"], 0.06)
        self.assertEqual(result["formed_images"], 3)
        self.assertEqual(result["emitted_images"], 2)
        self.assertEqual(result["formed_not_emitted_images"], 1)
        self.assertEqual(result["varying_evidence_images"], 0)

    def test_delta_acceptance_actual_use_fallback_and_scope_are_separate(self):
        result = study.summarize(report("delta"))
        self.assertEqual(result["baseline"], "absolute")
        left = result["overall"]["exact-delta-left"]["delta"]
        self.assertEqual(left["global_scope"]["exact_delta_accepted"], 1)
        self.assertEqual(left["global_scope"]["delta_pixels"], 100)
        self.assertEqual(left["local_scope"]["exact_delta_attempted"], 1)
        self.assertEqual(left["local_scope"]["absolute_fallback_accepted"], 1)
        self.assertEqual(left["exact_delta_used_images"], 1)
        self.assertEqual(left["fallback_only_images"], 1)
        self.assertEqual(left["inactive_images"], 1)
        self.assertEqual(result["overall"]["exact-delta-left"]["activation_by_size_result"]["loss"], {"fallback_only": 1})
        zero = result["overall"]["exact-zero-left"]["delta"]
        self.assertEqual(zero["exact_zero_accepted_images"], 2)
        self.assertEqual(zero["exact_zero_used_images"], 1)
        self.assertEqual(zero["accepted_unused_images"], 1)
        self.assertEqual(zero["fallback_only_images"], 1)
        self.assertEqual(zero["local_scope"]["delta_pixels"], 900)
        self.assertEqual(result["overall"]["absolute"]["delta"]["delta_used_images"], 0)

    def test_missing_evidence_is_not_inactive_and_does_not_drop_verified_bytes(self):
        for kind, mode, field in (("source", "medoid", "representative_work"),
                                  ("delta", "exact-delta-left", "palette_evidence")):
            data = report(kind)
            for sample in data["cases"][0]["variants"][mode]:
                sample["measurement"].pop(field)
            result = study.summarize(data)
            evidence = result["overall"][mode][kind]
            self.assertEqual(result["images"], 3)
            self.assertEqual(result["overall"][mode]["total_bytes"], 630)
            self.assertEqual(evidence["unavailable_images"], 1)
            self.assertEqual(evidence["evidence_images"], 2)
            self.assertFalse(result["cases"][0]["variants"][mode]["evidence"]["available"])

    def test_counter_variability_exposed_not_summed(self):
        data = report()
        data["cases"][0]["variants"]["medoid"][2]["measurement"]["representative_work"]["occurrences"] += 1
        result = study.summarize(data)
        self.assertEqual(result["overall"]["medoid"]["source"]["varying_evidence_images"], 1)
        e = result["cases"][0]["variants"]["medoid"]["evidence"]
        self.assertEqual(e["repetition"], 0)
        self.assertEqual(len(e["trial_evidence"]), 3)

    def test_fully_unavailable_evidence_reports_null_counts_not_zero(self):
        for kind, mode, field in (("source", "medoid", "representative_work"),
                                  ("delta", "exact-delta-left", "palette_evidence")):
            data = report(kind)
            for case in data["cases"]:
                for sample in case["variants"][mode]:
                    sample["measurement"].pop(field)
            result = study.summarize(data)
            self.assertEqual(result["images"], 3)
            e = result["overall"][mode][kind]
            self.assertEqual(e["unavailable_images"], 3)
            self.assertIsNone(e["formed_groups" if kind == "source" else "delta_used_images"])

    def test_explicit_intersection_failures_and_missing_manifest_cases(self):
        data = report()
        data["status"] = "failed"
        data["errors"] = ["one decoder failed"]
        data["cases"][1]["variants"]["medoid"][1]["verified"] = False
        metadata = manifest()
        metadata["cases"].append(dict(original="absent.png", stratum="large"))
        result = study.summarize(data, metadata)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["images"], 2)
        self.assertEqual(result["coverage"]["report_cases"], 3)
        self.assertEqual(result["coverage"]["manifest_cases"], 4)
        self.assertEqual(len(result["coverage"]["missing_manifest_sources"]), 1)
        self.assertEqual(result["excluded_cases"][0]["source"], "image-1.png")
        self.assertIn("medoid", result["excluded_cases"][0]["reasons"])
        self.assertTrue(all(v["images"] == 2 for v in result["overall"].values()))
        self.assertEqual(result["overall"]["medoid"]["total_bytes"], 380)

    def test_raw_checkpoint_without_saved_summaries_still_uses_all_verified_cases(self):
        data = report()
        for case in data["cases"]:
            case.pop("summaries")
        self.assertEqual(study.summarize(data)["images"], 3)

    def test_interrupted_full_manifest_is_never_promoted_or_pooled_with_rerun(self):
        interrupted = report("delta")
        interrupted.update(status="failed", errors=["KeyboardInterrupt: deliberately stopped"])
        interrupted["cases"] = interrupted["cases"][:2]
        interrupted["config"].update(full_frame=True, palette_colors=1024)
        partial = study.summarize(interrupted, manifest())
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["input_status"], "failed")
        self.assertEqual(partial["images"], 2)
        self.assertEqual(partial["excluded_cases"], [])
        self.assertEqual(len(partial["coverage"]["missing_manifest_sources"]), 1)
        self.assertTrue(all(r["study_status"] == "partial" and r["manifest_cases"] == 3
                            for r in study.csv_rows(partial)))
        with patch.object(study.common, "plot_summary") as plot:
            study.plot_summary(partial, ["one.png", "two.png"])
        title = plot.call_args.args[0]["input_report"]
        self.assertIn("PARTIAL", title)
        self.assertIn("2 of 3 manifest cases", title)
        self.assertIn("full-frame; cap=1024", title)
        self.assertEqual(title.count("\n"), 2)
        native = report("delta")
        native["config"].update(full_frame=False, palette_colors=1024)
        complete = study.summarize(native, manifest())
        self.assertEqual(complete["status"], "passed")
        self.assertEqual(complete["images"], 3)
        self.assertEqual(complete["overall"]["absolute"]["total_bytes"], 600)
        sensitivity = report("delta")
        sensitivity["cases"] = sensitivity["cases"][:1]
        sensitivity["config"].update(full_frame=True, palette_colors=10000)
        declared_small = dict(cases=manifest()["cases"][:1])
        control = study.summarize(sensitivity, declared_small)
        self.assertEqual(control["status"], "passed")
        self.assertEqual(control["coverage"]["manifest_cases"], 1)
        self.assertEqual(control["overall"]["absolute"]["total_bytes"], 100)


class Validation(unittest.TestCase):
    def test_corruption_and_unknown_study_are_rejected(self):
        mutations = [lambda d: d["config"].update(variants=["off"]),
                     lambda d: d["config"].update(variants=["first", "first"]),
                     lambda d: d["cases"].append(deepcopy(d["cases"][0])),
                     lambda d: d.update(temporary_outputs_cleaned=False),
                     lambda d: d.update(errors=["Input/binary changed during measurement"])]
        for mutate in mutations:
            data = report()
            mutate(data)
            with self.assertRaises(ValueError):
                study.summarize(data)
        with self.assertRaises(ValueError):
            study.summarize(report(), kind="delta")
        metadata = manifest()
        metadata["cases"][0]["sha256"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "Manifest sha256"):
            study.summarize(report(), metadata)

    def test_incomplete_or_nondeterministic_case_is_explicitly_excluded(self):
        for mutation in (lambda s: s.pop(), lambda s: s[1]["measurement"].update(bytes=True),
                         lambda s: s[1].update(encoder=dict(returncode=1)),
                         lambda s: s[1].pop("decoder"), lambda s: s[1]["measurement"].update(seconds=float("nan")),
                         lambda s: s[1].update(included_in_timing=False)):
            data = report()
            mutation(data["cases"][0]["variants"]["medoid"])
            result = study.summarize(data)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["images"], 2)
            self.assertEqual(len(result["excluded_cases"]), 1)
        data = report()
        for case in data["cases"]:
            case["variants"]["medoid"] = []
        with self.assertRaisesRegex(ValueError, "No common"):
            study.summarize(data)

    def test_invalid_activation_not_replaced_with_configured_mode_claims(self):
        data = report("delta")
        data["cases"][0]["variants"]["exact-delta-left"][1]["measurement"]["palette_evidence"]["attempts"][0]["delta_pixels"] = 0
        with self.assertRaisesRegex(ValueError, "exact delta"):
            study.summarize(data)
        data = report("delta")
        data["cases"][0]["variants"]["absolute"][1]["measurement"]["invariant_options"] = "different cap"
        with self.assertRaisesRegex(ValueError, "Non-mode"):
            study.summarize(data)
        data = report()
        data["cases"][0]["variants"]["medoid"][1]["measurement"]["representative_work"]["emitted_plans"] = 0
        with self.assertRaisesRegex(ValueError, "emitted"):
            study.summarize(data)


class Outputs(unittest.TestCase):
    def test_csv_per_case_losses_and_plot_oracle_cost(self):
        result = study.summarize(report())
        rows = list(study.csv_rows(result))
        self.assertEqual(len(rows), 12)
        loss = next(r for r in rows if r["source"] == "image-1.png" and r["variant"] == "medoid")
        self.assertEqual(loss["size_result"], "loss")
        self.assertEqual(loss["byte_delta"], 50)
        self.assertEqual(loss["source_formed_groups"], 3)
        self.assertEqual(loss["source_emitted_plans"], 0)
        self.assertEqual(loss["source_searched_groups"], 2)
        self.assertAlmostEqual(loss["source_selection_seconds"], 0.02)
        self.assertEqual(loss["oracle_all_variants_seconds"], 18)
        self.assertEqual(json.loads(loss["evidence_json"])["counters"]["emitted_plans"], 0)
        with patch.object(study.common, "plot_summary") as plot:
            study.plot_summary(result, ["one.png", "two.png"])
        plotted = plot.call_args.args[0]
        self.assertEqual(plotted["overall"]["oracle (all variant cost)"]["native_seconds"], 57)
        self.assertEqual(plotted["overall"]["oracle (all variant cost)"]["total_bytes"], 560)
        delta_rows = list(study.csv_rows(study.summarize(report("delta"))))
        used = next(r for r in delta_rows if r["source"] == "image-0.png" and r["variant"] == "exact-delta-left")
        unused = next(r for r in delta_rows if r["source"] == "image-0.png" and r["variant"] == "exact-zero-left")
        self.assertTrue(used["delta_global_exact_delta_used"])
        self.assertEqual(used["delta_global_delta_pixels"], 100)
        self.assertFalse(unused["delta_local_exact_zero_used"])
        self.assertEqual(unused["delta_local_exact_zero_accepted"], 1)

    def test_cli_tiny_reports_csv_charts_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root / "source.json", root / "delta.json"]
            for kind, path in zip(("source", "delta"), paths):
                path.write_text(json.dumps(report(kind)), encoding="utf-8")
            argv = ["--report", str(paths[0]), "--report", str(paths[1]), "--out", str(root / "result"), "--plot"]
            def fake_plot(summary, streams):
                self.assertEqual(len(streams), 2)
                for stream in streams:
                    self.assertFalse(stream.closed)
                    stream.write(b"mock png")
            with patch.object(study, "plot_summary", side_effect=fake_plot) as plot, \
                    patch("subprocess.run", side_effect=AssertionError("No codec/build allowed")), redirect_stdout(StringIO()):
                self.assertEqual(study.main(argv), 0)
                self.assertEqual(plot.call_count, 2)
            saved = json.loads((root / "result.source.summary.json").read_text())
            self.assertEqual(saved["baseline"], "first")
            self.assertEqual(saved["oracle"]["native_seconds"], 57)
            with (root / "result.source.cases.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            self.assertEqual({r["variant"] for r in rows}, {"off", "first", "medoid", "entropy"})
            before = {p: p.read_bytes() for p in root.iterdir()}
            with redirect_stderr(StringIO()):
                self.assertEqual(study.main(argv), 1)
            self.assertEqual(before, {p: p.read_bytes() for p in root.iterdir()})
            self.assertFalse(list(root.glob("*.md")))

    def test_strict_json_and_output_input_collisions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "input.json"
            path.write_text('{"status":"passed","status":"failed"}', encoding="utf-8")
            argv = ["--report", str(path), "--out", str(root / "result")]
            with redirect_stderr(StringIO()):
                self.assertEqual(study.main(argv), 1)
            self.assertFalse((root / "result.summary.json").exists())
            path.write_text(json.dumps(report()), encoding="utf-8")
            with redirect_stderr(StringIO()):
                self.assertEqual(study.main(argv + ["--report", str(path)]), 1)
            collision = root / "result.summary.json"
            collision.write_text(json.dumps(report()), encoding="utf-8")
            with redirect_stderr(StringIO()):
                self.assertEqual(study.main(["--report", str(collision), "--out", str(root / "result")]), 1)


if __name__ == "__main__":
    unittest.main()
