"""Admission runner contracts; no builds or real codec processes."""
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_palette_admission as bench


def event(stage="local", stream=2, ordinal=3, choice="keep", unavailable=False):
    forced = choice not in ("keep", "skip")
    delta, zero = choice.startswith("exact-delta-"), choice.startswith("exact-zero-")
    return dict(stage=stage, stream_id=stream, ordinal=ordinal, choice=choice, unavailable=unavailable,
                cost_forced=forced,
                attempt=dict(mode="absolute" if choice in ("keep", "skip") else choice,
                             width=2, height=1, begin=0, channels=3, limit=1024, colors=0 if delta else 2,
                             deltas=2 if delta else 1 if zero else 0, delta_pixels=2 if delta or zero else 0,
                             cost_before=10, cost_after=20, cost_bypassed=False,
                             exact_delta_attempted=delta, exact_delta_produced=delta,
                             exact_zero_attempted=zero, zero_mask_eligible=zero,
                             zero_mask_matches=2 if zero else 0,
                             absolute_attempted=not delta, absolute_produced=not delta,
                             outcome="accepted" if forced else "cost_rejected"))


def trial(name, unavailable=False):
    changed = event(choice="keep" if name == "baseline" else name, unavailable=unavailable)
    if name == "skip" or unavailable:
        changed["attempt"] = None
    return dict(name=name, status="unavailable" if unavailable else "ok",
                file=None if unavailable else f"{name}.jxl", bytes=None if unavailable else 10,
                seconds=0 if unavailable else 0.25, bpp=None if unavailable else 40,
                mpix_per_second=None if unavailable else 0.000008, fixed_input_verified=True,
                events=[changed])


def receipt(target, choices=("skip", "absolute")):
    value = dict(schema=1, width=2, height=1, bits=8, effort=9, threads=8,
                 baseline_mode="absolute", target=target, search=False, patches_cleared=True,
                 configured=dict(palette_colors=1024, groupwise=True),
                 trials=[trial("baseline"), *([trial(c) for c in choices] if target is not None else [])])
    if target is None:
        value["trials"][0]["fixed_input_verified"] = False
    return value


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expected = SimpleNamespace(width=2, height=1, depth=8)
        self.target = bench.target_dict(event())
        self.case = dict(id="case", path=str(self.root / "original.png"), baseline_mode="absolute",
                         choices=["skip", "absolute"], selector=dict(stage="local"))
        (self.root / "original.png").write_bytes(b"original")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(dict(cases=[self.case])), encoding="utf-8")
        self.argv = ["--encoder", str(self.root / "lab.exe"), "--decoder", str(self.root / "djxl.exe"),
                     "--manifest", str(self.manifest), "--out", str(self.root / "report.json")]

    def validate(self, value):
        verified = set()
        for t in value["trials"]:
            if t["status"] == "ok":
                path = self.root / t["file"]
                path.write_bytes(b"0123456789")
                verified.add(path)
        bench.validate_receipt(value, self.case, value["target"], self.expected, 8, self.root, verified)

    def test_parser_and_manifest(self):
        args = bench.arguments(self.argv)
        self.assertEqual((args.threads, args.repeats, args.timeout), (8, 1, 600))
        self.assertEqual(bench.load_manifest(self.manifest)[0]["max_targets"], 1)
        for extra in (["--threads", "0"], ["--repeats", "0"], ["--timeout", "nan"], ["--timeout", "inf"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments(self.argv + extra)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.arguments(["--encoder", "lab", "--decoder", "djxl", "--out", "report"])
        for changes in (dict(choices=["skip", "skip"]), dict(choices=["unknown"]),
                        dict(max_targets=101), dict(max_targets=True), dict(selector={"stage": "bad"}),
                        dict(selector={"stage": "local", "unknown": 3})):
            self.manifest.write_text(json.dumps(dict(cases=[dict(self.case, **changes)])), encoding="utf-8")
            with self.assertRaises(ValueError):
                bench.load_manifest(self.manifest)
        (self.root / "report.json").write_text("preserve", encoding="utf-8")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.arguments(self.argv)

    def test_selection_permutations_and_missing_target(self):
        events = [event(stream=4), event(stream=1, ordinal=8), event(stream=1, ordinal=2), event(stage="global")]
        for permutation in itertools.permutations(events):
            selection = bench.select_targets(permutation, {"stage": "local", "outcome": "cost_rejected"}, 2)
            self.assertEqual(selection["matched_count"], 3)
            self.assertEqual([(t["stream_id"], t["ordinal"]) for t in selection["selected"]], [(1, 2), (1, 8)])
        self.assertEqual(bench.select_targets(events, {"stage": "local", "colors": 99}, 1),
                         dict(matched_count=0, selected=[]))
        value = receipt(self.target)
        value["trials"][1]["events"] = []
        with self.assertRaisesRegex(ValueError, "missing"):
            self.validate(value)

    def test_forced_metadata_and_natural_suffix_may_differ(self):
        value = receipt(self.target)
        # Forced acceptance with a worse cost is valid evidence, not a natural-policy failure.
        value["trials"][2]["events"][0]["attempt"]["outcome"] = "accepted"
        value["trials"][2]["events"][0]["attempt"]["cost_bypassed"] = True
        value["trials"][2]["events"].append(event(stream=9))
        self.validate(value)
        for mutate in (
            lambda v: v["trials"].pop(),
            lambda v: v["trials"].append(deepcopy(v["trials"][1])),
            lambda v: v["trials"][0].update(fixed_input_verified=False),
            lambda v: v["trials"][2].update(seconds=0),
            lambda v: v["trials"][2].update(bpp=1),
            lambda v: v["trials"][2].update(bytes=11),
            lambda v: v["trials"][2]["events"][0].update(choice="skip"),
            lambda v: v["trials"][2]["events"][0].update(cost_forced=False),
            lambda v: v["trials"][0]["events"][0].update(cost_forced=True),
            lambda v: v["trials"][2]["events"][0].pop("cost_forced"),
            lambda v: v["trials"][2]["events"].append(event(stream=9, choice="absolute")),
            lambda v: v["trials"][2]["events"][0]["attempt"].update(delta_pixels=3),
        ):
            bad = deepcopy(value)
            mutate(bad)
            with self.assertRaises(ValueError):
                self.validate(bad)
        with self.assertRaisesRegex(ValueError, "target mismatch"):
            bench.validate_receipt(value, self.case, dict(self.target, ordinal=99), self.expected, 8, self.root, set())

    def test_discovery_fixed_input_and_policy_contracts(self):
        for groupwise in (False, True):
            for target in (None, self.target):
                value = receipt(target)
                value["configured"]["groupwise"] = groupwise
                self.validate(value)
        for target in (None, self.target):
            for mutate in (
                lambda v: v.update(search=True),
                lambda v: v.pop("search"),
                lambda v: v.update(patches_cleared=False),
                lambda v: v["configured"].update(palette_colors=10000),
                lambda v: v["configured"].pop("groupwise"),
                lambda v: v["configured"].update(groupwise=1),
                lambda v: v["trials"][0].update(fixed_input_verified=target is None),
            ):
                value = receipt(target)
                mutate(value)
                with self.assertRaises(ValueError):
                    self.validate(value)

    def test_forced_delta_and_zero_require_kernel_evidence(self):
        for choice in ("absolute", "exact-delta-gradient", "exact-zero-left"):
            value = receipt(self.target, (choice,))
            spec = dict(self.case, choices=[choice])
            verified = set()
            for t in value["trials"]:
                path = self.root / t["file"]
                path.write_bytes(b"0123456789")
                verified.add(path)
            bench.validate_receipt(value, spec, self.target, self.expected, 8, self.root, verified)
            mutations = [lambda a: a.pop("exact_delta_produced"),
                         lambda a: a.update(cost_bypassed=1),
                         lambda a: a.update(outcome="cost_rejected")]
            if choice.startswith("exact-delta-"):
                mutations += [lambda a: a.update(exact_delta_produced=False),
                              lambda a: a.update(exact_delta_attempted=False),
                              lambda a: a.update(delta_pixels=0)]
            if choice.startswith("exact-zero-"):
                mutations += [lambda a: a.update(zero_mask_eligible=False),
                              lambda a: a.update(absolute_produced=False)]
            for mutate in mutations:
                bad = deepcopy(value)
                mutate(bad["trials"][1]["events"][0]["attempt"])
                with self.assertRaises(ValueError):
                    bench.validate_receipt(bad, spec, self.target, self.expected, 8, self.root, verified)

    def test_unsafe_output_paths(self):
        for name in ("../outside.jxl", "/outside.jxl", "C:/outside.jxl", "C:outside.jxl",
                     "sub/../../outside.jxl", "sub\\outside.jxl", "./file.jxl", "file.png", "", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                bench.output_path(self.root, name)
        self.assertEqual(bench.output_path(self.root, "safe.jxl"), self.root / "safe.jxl")

    def test_unavailable_is_not_a_win_or_codec_failure(self):
        value = receipt(self.target)
        value["trials"][2] = trial("absolute", unavailable=True)
        self.validate(value)
        bad = deepcopy(value)
        bad["trials"][2]["fixed_input_verified"] = False
        with self.assertRaisesRegex(ValueError, "fixed-input"):
            self.validate(bad)
        sample = dict(verified=True, receipt=value)
        summary = bench.summarize_target([sample], self.case["choices"], 1)
        self.assertTrue(summary["complete"])
        self.assertFalse(summary["alternatives"]["absolute"]["scored"])
        self.assertIsNone(summary["alternatives"]["absolute"]["byte_ratio"])
        self.assertEqual(summary["alternatives"]["skip"]["byte_ratio"], 1)
        summary = bench.summarize_target([sample, dict(verified=False)], self.case["choices"], 2)
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["alternatives"]["skip"]["scored"])
        malformed = bench.summarize_target([dict(verified=False, receipt=["invalid"])], self.case["choices"], 1)
        self.assertFalse(malformed["complete"])
        value["trials"][2]["bytes"] = 0
        with self.assertRaises(ValueError):
            self.validate(value)

    def test_unavailable_requested_kernel_with_unused_absolute_fallback(self):
        for choice in ("exact-delta-gradient", "exact-zero-left", "absolute"):
            spec = dict(self.case, choices=[choice])
            value = receipt(self.target, (choice,))
            value["trials"][1] = trial(choice, unavailable=True)
            declined = event(choice=choice, unavailable=True)
            attempt = declined["attempt"]
            attempt.update(outcome="kernel_declined", colors=0, deltas=0, delta_pixels=0,
                           exact_delta_produced=False, absolute_attempted=True,
                           absolute_produced=choice != "absolute", zero_mask_eligible=False)
            # Other workers can complete natural attempts before/after one
            # unavailable target; only that target has unavailable semantics.
            value["trials"][1]["events"] = [declined, event(stream=9)]
            path = self.root / "baseline.jxl"
            path.write_bytes(b"0123456789")
            verified = {path}
            bench.validate_receipt(value, spec, self.target, self.expected, 8, self.root, verified)
            mutations = [lambda a: a.update(outcome="accepted"),
                         lambda a: a.update(colors=1),
                         lambda a: a.update(deltas=1),
                         lambda a: a.update(delta_pixels=1)]
            if choice.startswith("exact-delta-"):
                mutations.append(lambda a: a.update(exact_delta_produced=True))
            elif choice.startswith("exact-zero-"):
                mutations.append(lambda a: a.update(zero_mask_eligible=True))
                # Eligibility alone is insufficient if the absolute kernel declined.
                eligible = deepcopy(value)
                eligible["trials"][1]["events"][0]["attempt"].update(
                    zero_mask_eligible=True, absolute_produced=False)
                bench.validate_receipt(eligible, spec, self.target, self.expected, 8, self.root, verified)
            else:
                mutations.append(lambda a: a.update(absolute_produced=True))
            for mutate in mutations:
                bad = deepcopy(value)
                mutate(bad["trials"][1]["events"][0]["attempt"])
                with self.assertRaises(ValueError):
                    bench.validate_receipt(bad, spec, self.target, self.expected, 8, self.root, verified)

    def test_emitted_files_checked_after_exit_or_json_failure(self):
        args = bench.arguments(self.argv)
        for bad in ("exit", "json", "pixels"):
            sample, calls = {}, []
            with tempfile.TemporaryDirectory(dir=self.root) as directory:
                folder = Path(directory) / "trials"
                def run(command, env, timeout):
                    calls.append(command)
                    if len(calls) == 1:
                        self.assertFalse(folder.exists())
                        folder.mkdir()
                        (folder / "baseline.jxl").write_bytes(b"0123456789")
                        (folder / "extra.jxl").write_bytes(b"0123456789")
                        return dict(returncode=1 if bad == "exit" else 0, stderr="failure",
                                    stdout="invalid" if bad == "json" else json.dumps(receipt(None)))
                    Path(command[2]).write_bytes(b"oracle")
                    return dict(returncode=0, stderr="", stdout="")
                with patch.object(bench.port, "run_process", side_effect=run), \
                        patch.object(bench.port, "png_samples", return_value=self.expected), \
                        patch.object(bench.port, "compare_pixels", side_effect=ValueError("pixels") if bad == "pixels" else None):
                    with self.assertRaises(ValueError):
                        bench.invoke(args, self.case, self.root / "original.png", self.expected, folder, {}, sample)
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(sample["outputs"]), 2)

    def test_invoke_requires_new_directory_and_unchanged_policy(self):
        args = bench.arguments(self.argv)
        with patch.object(bench.port, "run_process") as run:
            with self.assertRaisesRegex(ValueError, "must not already exist"):
                bench.invoke(args, self.case, self.root / "original.png", self.expected, self.root, {}, {})
            run.assert_not_called()
        folder = self.root / "new-trials"
        def run(command, env, timeout):
            if command[0] == str(args.encoder):
                self.assertFalse(folder.exists())
                folder.mkdir()
                value = receipt(self.target)
                value["configured"]["groupwise"] = False
                for t in value["trials"]:
                    (folder / t["file"]).write_bytes(b"0123456789")
                return dict(returncode=0, stderr="", stdout=json.dumps(value))
            Path(command[2]).write_bytes(b"oracle")
            return dict(returncode=0, stderr="", stdout="")
        with patch.object(bench.port, "run_process", side_effect=run), \
                patch.object(bench.port, "png_samples", return_value=self.expected), \
                patch.object(bench.port, "compare_pixels"):
            with self.assertRaisesRegex(ValueError, "policy changed"):
                bench.invoke(args, self.case, self.root / "original.png", self.expected, folder, {}, {},
                             self.target, dict(palette_colors=1024, groupwise=True))

    def test_mocked_runner_partial_report_and_missing_selection(self):
        for mode in ("ok", "missing", "malformed", "changed-original", "interrupt"):
            with tempfile.TemporaryDirectory(dir=self.root) as directory:
                root = Path(directory)
                argv = self.argv[:-1] + [str(root / "report.json")]
                args = bench.arguments(argv)
                args.encoder.write_bytes(b"encoder")
                args.decoder.write_bytes(b"decoder")
                commands = []
                def run(command, env, timeout):
                    commands.append(command)
                    if command[0] == str(args.encoder):
                        folder = Path(command[command.index("--out-dir") + 1])
                        self.assertFalse(folder.exists())
                        folder.mkdir()
                        replay = "--target" in command
                        value = receipt(self.target if replay else None)
                        if mode == "missing":
                            value["trials"][0]["events"][0]["attempt"]["colors"] = 99
                        if replay:
                            value["trials"][2] = trial("absolute", unavailable=True)
                        for t in value["trials"]:
                            if t["file"]:
                                (folder / t["file"]).write_bytes(b"0123456789")
                        if replay and mode == "interrupt":
                            raise KeyboardInterrupt()
                        if replay and mode == "changed-original":
                            (self.root / "original.png").write_bytes(b"changed")
                        if replay and mode == "malformed":
                            value["trials"] = ["invalid"]
                        return dict(returncode=0, stderr="", stdout=json.dumps(value))
                    Path(command[2]).write_bytes(b"oracle")
                    return dict(returncode=0, stderr="", stdout="")
                spec = deepcopy(self.case)
                spec["selector"]["colors"] = 2
                self.manifest.write_text(json.dumps(dict(cases=[spec])), encoding="utf-8")
                with patch.object(bench, "prepare_png", return_value=(b"original", self.expected, "mock")), \
                        patch.object(bench.port, "run_process", side_effect=run), \
                        patch.object(bench.port, "png_samples", return_value=self.expected), \
                        patch.object(bench.port, "compare_pixels"):
                    report = bench.benchmark(args)
                saved = json.loads(args.out.read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], "passed" if mode == "ok" else "failed", saved["errors"])
                self.assertTrue(report["temporary_outputs_cleaned"])
                self.assertEqual(saved["cases"][0]["selection"]["matched_count"], 0 if mode == "missing" else 1)
                self.assertEqual(args.encoder.read_bytes(), b"encoder")
                if mode == "ok":
                    summary = saved["cases"][0]["targets"][0]["summary"]
                    self.assertEqual(summary["alternatives"]["absolute"]["status"], "unavailable")
                    self.assertEqual(len([c for c in commands if "--target" in c]), 1)
                with self.assertRaises(FileExistsError):
                    bench.benchmark(args)


if __name__ == "__main__":
    unittest.main()
