import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_ma_kernels import ema_arguments, parse_ema, validate_ema_receipt


class EmaBenchmarkTests(unittest.TestCase):
    def test_parse_repeatable_options_and_boundary_passes(self):
        self.assertEqual(parse_ema([], {"baseline"}), {})
        self.assertEqual(parse_ema(["random_1=1:random", "far-255=255:farthest"],
                                   {"baseline", "random_1", "far-255"}), {
            "random_1": {"passes": 1, "initialization": "random"},
            "far-255": {"passes": 255, "initialization": "farthest"},
        })

    def test_reject_invalid_options(self):
        for value in ("trial=0:random", "trial=256:random", "trial=-1:random",
                      "trial=1.5:random", "trial=1:other", "trial=1:Random",
                      "trial=1", "trial:1:random", "trial=1:random:extra",
                      "trial= 1:random", "trial=1:random ", "trial=+1:random",
                      "../trial=1:random", "bad name=1:random", "=1:random",
                      "missing=1:random"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_ema([value], {"trial"})
        with self.assertRaises(ValueError):
            parse_ema(["trial=1:random", "trial=2:farthest"], {"trial"})

    def test_arguments_only_enable_the_selected_variant(self):
        ema = parse_ema(["random=3:random", "far=7:farthest"],
                        {"baseline", "random", "far"})
        self.assertEqual(ema_arguments(ema.get("baseline")), [])
        self.assertEqual(ema_arguments(ema["random"]), ["--ema-passes", "3"])
        self.assertEqual(ema_arguments(ema["far"]),
                         ["--ema-passes", "7", "--ema-farthest"])

    def test_parse_distance_and_polish_per_variant(self):
        ema = parse_ema(
            ["cross=1:random", "hell=3:farthest", "merge=255:random"],
            {"baseline", "cross", "hell", "merge"},
            ["cross=cross-entropy", "hell=hellinger", "merge=merge-entropy"],
            ["cross=0", "hell=255"],
        )
        self.assertEqual(ema["cross"], dict(passes=1, initialization="random",
                                           distance="cross-entropy", polish=0))
        self.assertEqual(ema["hell"], dict(passes=3, initialization="farthest",
                                          distance="hellinger", polish=255))
        self.assertEqual(ema["merge"], dict(passes=255, initialization="random",
                                           distance="merge-entropy"))
        self.assertNotIn("baseline", ema)
        self.assertEqual(ema_arguments(ema["cross"]),
                         ["--ema-passes", "1", "--ema-distance", "cross-entropy",
                          "--ema-polish", "0"])
        self.assertEqual(ema_arguments(ema["hell"]),
                         ["--ema-passes", "3", "--ema-farthest",
                          "--ema-distance", "hellinger", "--ema-polish", "255"])
        self.assertEqual(ema_arguments(ema["merge"]),
                         ["--ema-passes", "255", "--ema-distance", "merge-entropy"])

    def test_reject_invalid_distance_and_polish(self):
        for field, invalid in (
                ("distances", ("trial=unknown", "trial=Hellinger", "trial=CrossEntropy",
                               "trial=hellinger ", "trial= hellinger", "trial=",
                               "trial:hellinger", "trial=hellinger:extra",
                               "../trial=hellinger", "bad name=hellinger", "=hellinger",
                               "baseline=hellinger", "missing=hellinger")),
                ("polish", ("trial=256", "trial=-1", "trial=1.5", "trial=+1",
                            "trial= 1", "trial=1 ", "trial=", "trial:1", "trial=1:extra",
                            "../trial=1", "bad name=1", "=1", "baseline=1", "missing=1"))):
            for value in invalid:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    parse_ema(["trial=3:random"], {"trial", "baseline"}, **{field: [value]})
        for settings in (dict(distances=["trial=hellinger", "trial=cross-entropy"]),
                         dict(distances=["trial=hellinger", "trial=hellinger"]),
                         dict(polish=["trial=0", "trial=255"]),
                         dict(polish=["trial=1", "trial=1"])):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                parse_ema(["trial=3:random"], {"trial"}, **settings)
        for settings in (dict(distances=["trial=hellinger"]), dict(polish=["trial=0"])):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                parse_ema([], {"trial"}, **settings)

    def test_distance_and_polish_are_independent_opt_ins(self):
        distance_only = parse_ema(["trial=2:random"], {"trial"}, ["trial=hellinger"])
        polish_only = parse_ema(["trial=2:random"], {"trial"}, polish=["trial=1"])
        self.assertEqual(ema_arguments(distance_only["trial"]),
                         ["--ema-passes", "2", "--ema-distance", "hellinger"])
        self.assertEqual(ema_arguments(polish_only["trial"]),
                         ["--ema-passes", "2", "--ema-polish", "1"])

    def test_receipts_match_exact_passes_and_initialization(self):
        for initialization in ("random", "farthest"):
            options = {"passes": 3, "initialization": initialization}
            farthest = initialization == "farthest"
            validate_ema_receipt({"ema_passes": 3, "ema_farthest": farthest}, options)
            for row in ({}, {"ema_passes": 3}, {"ema_farthest": farthest},
                        {"ema_passes": None, "ema_farthest": farthest},
                        {"ema_passes": 2, "ema_farthest": farthest},
                        {"ema_passes": 3.0, "ema_farthest": farthest},
                        {"ema_passes": "3", "ema_farthest": farthest},
                        {"ema_passes": 3, "ema_farthest": not farthest},
                        {"ema_passes": 3, "ema_farthest": int(farthest)}):
                with self.subTest(initialization=initialization, row=row):
                    with self.assertRaises(ValueError):
                        validate_ema_receipt(row, options)
        with self.assertRaises(ValueError):
            validate_ema_receipt({"ema_passes": True, "ema_farthest": False},
                                 {"passes": 1, "initialization": "random"})

    def test_final_assignment_only_enables_selected_ema_variants(self):
        ema = parse_ema(["trial_1=2:random", "trial-2=3:farthest", "ordinary=1:random"],
                        {"baseline", "trial_1", "trial-2", "ordinary"},
                        final_assignment=["trial-2", "trial_1"])
        self.assertIs(ema["trial_1"]["final_assignment"], True)
        self.assertIs(ema["trial-2"]["final_assignment"], True)
        self.assertNotIn("final_assignment", ema["ordinary"])
        self.assertEqual(ema_arguments(ema["trial_1"]),
                         ["--ema-passes", "2", "--ema-final-assignment"])
        self.assertEqual(ema_arguments(ema["trial-2"]),
                         ["--ema-passes", "3", "--ema-farthest", "--ema-final-assignment"])
        self.assertEqual(ema_arguments(ema["ordinary"]), ["--ema-passes", "1"])
        self.assertEqual(ema_arguments(ema.get("baseline")), [])
        combined = parse_ema(["trial=2:farthest"], {"trial"},
                             ["trial=merge-entropy"], ["trial=1"], ["trial"])
        self.assertEqual(ema_arguments(combined["trial"]),
                         ["--ema-passes", "2", "--ema-farthest", "--ema-distance",
                          "merge-entropy", "--ema-polish", "1", "--ema-final-assignment"])

    def test_final_assignment_requires_safe_unique_existing_ema_names(self):
        for name in ("baseline", "missing", "../trial", "bad name", "", "trial ",
                     " trial", "trial=true", "trial:1", "trial/extra"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_ema(["trial=2:random"], {"trial", "baseline"}, final_assignment=[name])
        with self.assertRaises(ValueError):
            parse_ema(["trial=2:random"], {"trial"}, final_assignment=["trial", "trial"])
        with self.assertRaises(ValueError):
            parse_ema([], {"trial"}, final_assignment=["trial"])

    def test_final_assignment_receipts_require_exact_booleans(self):
        old = dict(ema_passes=2, ema_farthest=False)
        defaults = dict(passes=2, initialization="random")
        enabled = dict(defaults, final_assignment=True)
        validate_ema_receipt(old, defaults)
        validate_ema_receipt(old, dict(defaults, final_assignment=False))
        validate_ema_receipt(dict(old, ema_final_assignment=False), defaults)
        validate_ema_receipt(dict(old, ema_final_assignment=True), enabled)
        with self.assertRaises(ValueError):
            validate_ema_receipt(old, enabled)
        with self.assertRaises(ValueError):
            validate_ema_receipt(dict(old, ema_final_assignment=False), enabled)
        with self.assertRaises(ValueError):
            validate_ema_receipt(dict(old, ema_final_assignment=True), defaults)
        for options in (defaults, enabled):
            for value in (None, 0, 1, 0.0, 1.0, "true", "false", [], {}):
                with self.subTest(options=options, value=value), self.assertRaises(ValueError):
                    validate_ema_receipt(dict(old, ema_final_assignment=value), options)
        baseline = dict(ema_passes=None, ema_farthest=False,
                        ema_distance=None, ema_polish=0)
        validate_ema_receipt(dict(baseline, ema_final_assignment=False), None)
        for value in (True, None, 0, 1, 0.0, "false"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_ema_receipt(dict(baseline, ema_final_assignment=value), None)
        for value in (False, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_ema_receipt(dict(ema_final_assignment=value), None)

    def test_baseline_accepts_old_or_disabled_receipts_only(self):
        validate_ema_receipt({}, None)
        validate_ema_receipt({"ema_passes": None, "ema_farthest": False}, None)
        validate_ema_receipt({"ema_passes": None, "ema_farthest": False,
                              "ema_distance": None, "ema_polish": 0}, None)
        for row in ({"ema_passes": None}, {"ema_farthest": False},
                    {"ema_passes": 0, "ema_farthest": False},
                    {"ema_passes": 1, "ema_farthest": False},
                    {"ema_passes": None, "ema_farthest": True},
                    {"ema_passes": None, "ema_farthest": 0}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_ema_receipt(row, None)

    def test_receipts_match_distance_and_polish_exactly(self):
        for distance in ("cross-entropy", "hellinger", "merge-entropy"):
            for polish in (0, 1, 255):
                options = dict(passes=3, initialization="farthest", distance=distance, polish=polish)
                row = dict(ema_passes=3, ema_farthest=True,
                           ema_distance=distance, ema_polish=polish)
                validate_ema_receipt(row, options)
                for field, invalid in (
                        ("ema_distance", (None, "unknown", "Hellinger", 0, False)),
                        ("ema_polish", (None, str(polish), float(polish), False, True, -1, 256))):
                    for value in invalid:
                        with self.subTest(options=options, field=field, value=value):
                            with self.assertRaises(ValueError):
                                validate_ema_receipt(dict(row, **{field: value}), options)
                wrong_distance = "hellinger" if distance == "cross-entropy" else "cross-entropy"
                with self.assertRaises(ValueError):
                    validate_ema_receipt(dict(row, ema_distance=wrong_distance), options)
                with self.assertRaises(ValueError):
                    validate_ema_receipt(dict(row, ema_polish=(polish + 1) % 256), options)

    def test_legacy_receipts_only_match_default_distance_and_polish(self):
        old = dict(ema_passes=3, ema_farthest=False)
        defaults = dict(passes=3, initialization="random")
        validate_ema_receipt(old, defaults)
        validate_ema_receipt(dict(old, ema_distance="cross-entropy", ema_polish=0), defaults)
        validate_ema_receipt(old, dict(defaults, distance="cross-entropy", polish=0))
        for distance in ("hellinger", "merge-entropy"):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                validate_ema_receipt(old, dict(defaults, distance=distance))
        with self.assertRaises(ValueError):
            validate_ema_receipt(old, dict(defaults, polish=1))
        with self.assertRaises(ValueError):
            validate_ema_receipt(dict(old, ema_distance="hellinger"),
                                 dict(defaults, distance="hellinger", polish=1))
        with self.assertRaises(ValueError):
            validate_ema_receipt(dict(old, ema_polish=1),
                                 dict(defaults, distance="hellinger", polish=1))

    def test_disabled_receipts_reject_distance_or_polish_experiments(self):
        baseline = dict(ema_passes=None, ema_farthest=False)
        for row in (dict(ema_distance=None), dict(ema_polish=0),
                    dict(baseline, ema_distance="cross-entropy"),
                    dict(baseline, ema_distance="hellinger"),
                    dict(baseline, ema_polish=1), dict(baseline, ema_polish=False),
                    dict(baseline, ema_polish=0.0), dict(baseline, ema_polish="0"),
                    dict(baseline, ema_polish=None)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_ema_receipt(row, None)


if __name__ == "__main__":
    unittest.main()
