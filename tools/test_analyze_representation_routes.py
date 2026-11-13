import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import analyze_representation_routes as routes
from test_prepare_representation_corpus import png_bytes


def fixtures(collapsed=False):
    cases, feature_rows, reports = [], [], []
    for index in range(8):
        case = dict(sha256=hashlib.sha256(str(index).encode()).hexdigest(),
                    original=str(Path(tempfile.gettempdir()) / f"representation-{index}.png"),
                    split="calibration" if index < 4 else "holdout",
                    source_group=f"family-{index // 2}", width=4, height=4,
                    bit_depth=8, color_type=6)
        features = {name: float(index % 2) for name in routes.FEATURES}
        feature_rows.append(dict(sha256=case["sha256"], features=features,
                                 semantics=dict(width=4, height=4, source_bit_depth=8,
                                                source_color_type=6, sample_depth=8),
                                 timing=dict(total_seconds=0.02, feature_seconds=0.01,
                                             read_validate_decode_seconds=0.01)))
        delta_bytes = 110 if collapsed else (80 if index % 2 else 120)
        variants = {}
        for name, size, seconds in (("baseline", 100, 1.0), ("delta", delta_bytes, 2.0)):
            variants[name] = [dict(verified=True, warmup=False,
                                   measurement=dict(bytes=size, seconds=seconds, width=4, height=4))]
        reports.append(dict(source=case["original"], sha256=case["sha256"],
                            pixels=16, variants=variants))
        cases.append(case)
    return (dict(cases=cases), dict(schema="representation-features-v1",
                                   feature_names=list(routes.FEATURES), cases=feature_rows),
            [dict(cases=reports)])


class FeatureTests(unittest.TestCase):
    def test_windows_deterministic_unresized_in_bounds(self):
        a = routes.windows(1000, 600)
        self.assertEqual(a, routes.windows(1000, 600))
        self.assertEqual(len(a), 9)
        for x, y, w, h in a:
            self.assertEqual((w, h), (64, 64))
            self.assertTrue(0 <= x <= 1000 - w and 0 <= y <= 600 - h)
        self.assertEqual(routes.windows(1, 1), [(0, 0, 1, 1)])

    def test_flat_and_gradient_features(self):
        a = np.zeros((8, 8, 4), dtype=np.uint8)
        a[:, :, 3] = 255
        flat = routes.content_features([a], 8)
        self.assertEqual(flat["flat_block_rate"], 1)
        self.assertEqual(flat["horizontal_equality"], 1)
        self.assertEqual(flat["left_residual_mean_abs"], 0)
        a[:, :, :3] = np.arange(8, dtype=np.uint8)[None, :, None]
        gradient = routes.content_features([a], 8)
        self.assertEqual(gradient["horizontal_equality"], 0)
        self.assertEqual(gradient["vertical_equality"], 1)
        self.assertEqual(gradient["gradient_residual_mean_abs"], 0)
        self.assertGreater(gradient["directional_contrast"], 0)
        self.assertEqual(set(gradient), set(routes.FEATURES))

    def test_hidden_rgb_is_not_composited_away(self):
        a = np.zeros((4, 4, 4), dtype=np.uint8)
        a[:, 2:, 0] = 255
        observed = routes.content_features([a], 8)
        self.assertGreater(observed["horizontal_change"], 0)
        self.assertGreater(observed["left_residual_mean_abs"], 0)
        decoded, semantics = routes.decode_samples(png_bytes(width=4, height=4,
                    raw=b"".join(b"\0" + row.tobytes() for row in a)))
        np.testing.assert_array_equal(decoded, a)
        self.assertEqual(semantics["sample_depth"], 8)
        self.assertTrue(semantics["alpha_present"])

    def test_gradient_overshoot_clamps_to_neighbor_range(self):
        for depth, dtype, scale in ((8, np.uint8, 1), (16, np.uint16, 257)):
            for corner, expected in ((0, 30), (80, 20)):
                with self.subTest(depth=depth, corner=corner):
                    # L=30, T=20: raw gradient is 50 or -30, but JXL
                    # clamps to [20, 30], not to the full sample range.
                    gray = np.array([[corner, 20], [30, expected]], dtype=dtype) * scale
                    rgba = np.repeat(gray[:, :, None], 4, axis=2)
                    rgba[:, :, 3] = (1 << depth) - 1
                    features = routes.content_features([rgba], depth)
                    self.assertEqual(features["gradient_residual_mean_abs"], 0.0)
                    self.assertEqual(features["gradient_residual_cardinality"], 0.25)

    def test_tuple_cardinality_distinguishes_equal_scalar_histograms(self):
        patterns = (
            np.array([[1, 1, 0, 0], [1, 1, 0, 0], [2, 2, 0, 0], [2, 2, 0, 0]]),
            np.array([[1, 1, 0, 0], [1, 2, 0, 0], [2, 1, 0, 0], [2, 2, 0, 0]]),
        )
        for predictor in ("left", "top", "gradient"):
            results = []
            for deltas in patterns:
                row = np.vstack((np.zeros((1, 4), dtype=np.int64), np.cumsum(deltas, axis=0)))
                row[:, 3] = 255
                if predictor == "left":
                    window = row[None, :, :]
                elif predictor == "top":
                    window = row[:, None, :]
                else:
                    background = np.zeros_like(row)
                    background[:, 3] = 255
                    window = np.stack((background, row))
                results.append(routes.content_features([window.astype(np.uint8)], 8))
            with self.subTest(predictor=predictor):
                self.assertEqual(results[0][predictor + "_residual_cardinality"],
                                 results[1][predictor + "_residual_cardinality"])
                self.assertEqual(results[0][predictor + "_residual_mean_abs"],
                                 results[1][predictor + "_residual_mean_abs"])
                self.assertEqual(results[0][predictor + "_residual_tuple_cardinality"], 0.5)
                self.assertEqual(results[1][predictor + "_residual_tuple_cardinality"], 1.0)

    def test_zero_tuples_preserve_signed_hidden_rgb_and_alpha(self):
        deltas = np.array([[0, 0, 0, 0], [5, -3, 0, 0],
                           [0, 0, 0, 0], [0, 0, 0, 7]], dtype=np.int64)
        base = np.array([20, 20, 20, 0], dtype=np.int64)
        row = np.vstack((base, base + np.cumsum(deltas, axis=0)))
        windows = dict(left=row[None, :, :], top=row[:, None, :],
                       gradient=np.stack((np.tile(base, (len(row), 1)), row)))
        for depth, dtype, scale in ((8, np.uint8, 1), (16, np.uint16, 257)):
            for predictor, window in windows.items():
                with self.subTest(depth=depth, predictor=predictor):
                    features = routes.content_features([(window * scale).astype(dtype)], depth)
                    self.assertEqual(features[predictor + "_residual_zero_tuple_fraction"], 0.5)
                    self.assertEqual(features[predictor + "_residual_tuple_cardinality"], 0.75)

    def test_tuple_features_never_connect_separate_windows(self):
        first = np.zeros((2, 2, 4), dtype=np.uint8)
        second = np.full((2, 2, 4), 255, dtype=np.uint8)
        features = routes.content_features([first, second], 8)
        self.assertEqual(features, routes.content_features([first, second], 8))
        for predictor in ("left", "top", "gradient"):
            self.assertEqual(features[predictor + "_residual_zero_tuple_fraction"], 1.0)
            self.assertEqual(features[predictor + "_residual_tuple_cardinality"],
                             0.5 if predictor == "gradient" else 0.25)

    def test_16bit_gray_preserved_and_packed_trns_expanded(self):
        raw = b"\0" + struct.pack(">HH", 1, 65535)
        a, info = routes.decode_samples(png_bytes(color=0, depth=16, width=2, height=1, raw=raw))
        self.assertEqual(a.dtype, np.uint16)
        self.assertEqual(info["sample_depth"], 16)
        self.assertEqual(a[0, 0, 0], 1)
        a, info = routes.decode_samples(png_bytes(color=0, depth=1, width=2, height=1,
                    raw=b"\0\x40", extra=((b"tRNS", b"\0\1"),)))
        self.assertEqual((info["source_bit_depth"], info["sample_depth"]), (1, 8))
        np.testing.assert_array_equal(a[0, :, 0], [0, 255])
        np.testing.assert_array_equal(a[0, :, 3], [255, 0])

    def test_16bit_rgba_retains_low_bits_and_hidden_rgb(self):
        try:
            import cv2  # noqa: F401 - optional exact-depth decoder
        except ImportError:
            self.skipTest("Optional OpenCV not installed")
        values = np.array([[[1, 256, 65535, 0], [2, 257, 65534, 65535]]], dtype=np.uint16)
        raw = b"\0" + values.astype(">u2").tobytes()
        decoded, info = routes.decode_samples(png_bytes(depth=16, width=2, height=1, raw=raw))
        np.testing.assert_array_equal(decoded, values)
        self.assertEqual(info["sample_depth"], 16)

    def test_palette_hidden_rgb_and_alpha(self):
        a, info = routes.decode_samples(png_bytes(color=3, depth=1, width=2, height=1,
                raw=b"\0\x40", extra=((b"PLTE", bytes([23, 45, 67, 90, 80, 70])),
                                       (b"tRNS", bytes([0, 255])))))
        np.testing.assert_array_equal(a[0, 0], [23, 45, 67, 0])
        self.assertTrue(info["alpha_present"])

    def test_single_pixel_has_finite_features(self):
        values = routes.content_features([np.zeros((1, 1, 4), dtype=np.uint8)], 8)
        self.assertTrue(all(np.isfinite(v) for v in values.values()))
        for predictor in ("left", "top", "gradient"):
            self.assertEqual(values[predictor + "_residual_tuple_cardinality"], 0.0)
            self.assertEqual(values[predictor + "_residual_zero_tuple_fraction"], 0.0)

    def test_tiny_extraction_no_report_or_encoding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "original.png"
            data = png_bytes(12)
            path.write_bytes(data)
            manifest = Path(temp) / "manifest.json"
            manifest.write_text(json.dumps(dict(cases=[dict(original=str(path),
                         sha256=hashlib.sha256(data).hexdigest())])), encoding="utf-8")
            result = routes.extract(manifest)
            row = result["cases"][0]
            self.assertEqual(set(row["features"]), set(routes.FEATURES))
            self.assertGreater(row["timing"]["total_seconds"], 0)
            self.assertEqual(path.read_bytes(), data)


class AnalysisTests(unittest.TestCase):
    def test_calibration_stump_byte_regret_and_runtime_estimates(self):
        result = routes.analyze(*fixtures())
        self.assertEqual(result["model"]["kind"], "stump")
        held = result["evaluations"]["holdout"]
        self.assertEqual(held["byte_regret"], 0)
        self.assertEqual(held["byte_savings_vs_baseline"], 40)
        self.assertEqual(held["exhaustive_seconds"], 12)
        self.assertEqual(held["chosen_route_seconds_estimate"], 6)
        self.assertAlmostEqual(held["independently_measured_feature_overhead_seconds"], 0.08)
        self.assertIsNone(held["actual_chosen_route_runs"])

    def test_holdout_labels_do_not_change_fitted_model(self):
        study, features, reports = fixtures()
        first = routes.analyze(study, features, reports)
        for row in reports[0]["cases"][4:]:
            row["variants"]["delta"][0]["measurement"]["bytes"] = 1000
        second = routes.analyze(study, features, reports)
        self.assertEqual(first["model"], second["model"])
        self.assertGreater(second["evaluations"]["holdout"]["byte_regret"], 0)
        self.assertLess(second["evaluations"]["holdout"]["byte_savings_vs_baseline"], 0)

    def test_no_gains_does_not_force_classifier(self):
        result = routes.analyze(*fixtures(collapsed=True))
        self.assertEqual(result["model"]["kind"], "constant")
        self.assertEqual(result["model"]["route"], "baseline")
        self.assertEqual(result["model"]["status"], "no_candidate_byte_gains_on_calibration")
        self.assertEqual(result["evaluations"]["holdout"]["byte_savings_vs_baseline"], 0)

    def test_uniform_positive_candidate_is_a_constant_not_a_tree(self):
        study, features, reports = fixtures()
        for row in reports[0]["cases"]:
            row["variants"]["delta"][0]["measurement"]["bytes"] = 80
        result = routes.analyze(study, features, reports)
        self.assertEqual(result["model"]["kind"], "constant")
        self.assertEqual(result["model"]["route"], "delta")
        self.assertEqual(result["model"]["status"], "collapsed_calibration_labels_constant_route")

    def test_no_outputsize_or_extra_feature_allowed(self):
        study, features, reports = fixtures()
        features["cases"][0]["features"]["png_bytes"] = 100
        with self.assertRaisesRegex(ValueError, "whitelisted"):
            routes.analyze(study, features, reports)

    def test_scalar_only_feature_cache_is_rejected_before_fit(self):
        study, features, reports = fixtures()
        features["feature_names"] = [name for name in features["feature_names"] if "tuple" not in name]
        with self.assertRaisesRegex(ValueError, "feature schema"):
            routes.analyze(study, features, reports)

    def test_group_leakage_and_partial_holdout_rejected(self):
        study, features, reports = fixtures()
        study["cases"][4]["source_group"] = study["cases"][0]["source_group"]
        with self.assertRaisesRegex(ValueError, "group crosses"):
            routes.analyze(study, features, reports)
        study, features, reports = fixtures()
        features["cases"].pop()
        with self.assertRaisesRegex(ValueError, "partial holdout"):
            routes.analyze(study, features, reports)

    def test_native_nested_route_explicit_adapter(self):
        study, features, reports = fixtures()
        for row in reports[0]["cases"]:
            row["evidence"] = dict(native=dict(delta=row["variants"]["delta"]))
        specs = dict(baseline=dict(report=0, path=["variants", "baseline"]),
                     native_delta=dict(report=0, path=["evidence", "native", "delta"]))
        result = routes.analyze(study, features, reports, route_specs=specs)
        self.assertEqual(result["evaluations"]["holdout"]["byte_regret"], 0)
        self.assertEqual(result["model"]["right_route"], "native_delta")

    def test_invalid_unverified_or_unstable_reports_rejected(self):
        study, features, reports = fixtures()
        reports[0]["cases"][0]["variants"]["delta"][0]["verified"] = False
        with self.assertRaisesRegex(ValueError, "verified"):
            routes.analyze(study, features, reports)
        study, features, reports = fixtures()
        sample = copy.deepcopy(reports[0]["cases"][0]["variants"]["delta"][0])
        sample["measurement"]["bytes"] += 1
        reports[0]["cases"][0]["variants"]["delta"].append(sample)
        with self.assertRaisesRegex(ValueError, "Nondeterministic"):
            routes.analyze(study, features, reports)

    def test_sha_mismatch_and_depth_narrowing_rejected(self):
        study, features, reports = fixtures()
        reports[0]["cases"][0]["sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 disagrees"):
            routes.analyze(study, features, reports)
        study, features, reports = fixtures()
        features["cases"][0]["semantics"]["sample_depth"] = 16
        with self.assertRaisesRegex(ValueError, "semantics mismatch"):
            routes.analyze(study, features, reports)

    def test_false_skips_misses_and_baseline_ties(self):
        study, features, reports = fixtures(collapsed=True)
        for row in reports[0]["cases"][4:]:
            row["variants"]["delta"][0]["measurement"]["bytes"] = 80
        result = routes.analyze(study, features, reports)
        held = result["evaluations"]["holdout"]
        self.assertEqual((held["false_skips"], held["misses"], held["byte_regret"]), (4, 4, 80))
        for row in reports[0]["cases"]:
            row["variants"]["delta"][0]["measurement"]["bytes"] = 100
        result = routes.analyze(study, features, reports)
        self.assertEqual(result["model"]["route"], "baseline")

    def test_real_selected_runs_are_separate_from_estimates(self):
        study, features, reports = fixtures()
        result = routes.analyze(study, features, reports)
        runs = dict(cases=[dict(sha256=row["sha256"], verified=True, route=row["chosen_route"],
                               bytes=row["chosen_bytes"], seconds=0.5)
                           for row in result["evaluations"]["holdout"]["cases"]])
        result = routes.analyze(study, features, reports, measured_runs=runs)
        held = result["evaluations"]["holdout"]
        self.assertEqual(held["actual_chosen_route_runs"]["measured_route_seconds"], 2)
        self.assertEqual(held["chosen_route_seconds_estimate"], 6)


if __name__ == "__main__":
    unittest.main()
