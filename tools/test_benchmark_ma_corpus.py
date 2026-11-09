import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_ma_corpus as bench


class MaCorpusTests(unittest.TestCase):
    def fixture(self, root, count=2):
        for name in ("encoder", "djxl"):
            (root / name).write_bytes(name.encode())
        cases = []
        for index in range(count):
            source = root / f"source{index}.png"
            source.write_bytes(bench.PNG_SIGNATURE + f"original{index}".encode())
            cases.append(dict(original=str(source), stratum=f"group{index}",
                              sha256=bench.file_sha256(source), known_case=index == 0,
                              width=3, height=2, bit_depth=16))
        manifest = root / "sample.json"
        manifest.write_text(json.dumps(dict(cases=cases)), encoding="utf-8")
        argv = ["--encoder", str(root / "encoder"), "--decoder", str(root / "djxl"),
                "--manifest", str(manifest), "--out", str(root / "run")]
        return SimpleNamespace(root=root, cases=cases, manifest=manifest, argv=argv,
                               args=bench.arguments(argv), calls=[], artifacts=[], prepared=[],
                               progress=[], effect=None, phase_effect=None, compare_effect=None,
                               measurement_effect=None, variant=None)

    def measurement(self, variant, pixels=6):
        params = bench.variant_parameters(variant)
        fast = params["fast_context"]
        if fast is not None:
            fast = dict(fast, scan="auto", trees=1, leaves=2, pixels=pixels,
                        offset_leaves=0, horizontal=1, vertical=0, single=0)
        seed = params["ma_seed"]
        if seed is not None:
            seed = dict(seed, **dict.fromkeys(bench.SEED_COUNTERS, 0))
            seed["learners"] = 1
        return dict(bytes=100, seconds=0.25, bpp=800 / pixels,
                    mpix_per_second=pixels / 1e6 / 0.25, search=False,
                    model_costs=dict(tree_bits=10, entropy_header_bits=20, total_bits=30),
                    fast_context=fast, ma_seed=seed)

    def invoke(self, fixture, extra=()):
        f = fixture
        args = bench.arguments(f.argv + list(extra))
        oracle = SimpleNamespace(width=3, height=2, depth=16)

        def prepare(source, directory):
            f.prepared.append(str(source))
            snapshot = directory / "snapshot.png"
            snapshot.write_bytes(bench.PNG_SIGNATURE + b"canonical")
            f.artifacts.append(snapshot)
            index = next(i for i, case in enumerate(f.cases) if case["original"] == str(source))
            f.index = index
            depth = max(8, f.cases[index].get("bit_depth", f.cases[index].get("depth", 16)))
            oracle.depth = depth
            info = dict(kind="png", width=3, height=2, pixels=6, depth=depth,
                        input_sha256=bench.file_sha256(source),
                        payload_sha256=bench.file_sha256(snapshot), input_bytes=source.stat().st_size,
                        conversion="png-packed2-to-rgba8" if depth == 8 else "metadata-strip",
                        removed_metadata_chunks=["tEXt", "iCCP"])
            return info, snapshot, oracle

        def process(command, environment, timeout):
            self.assertEqual(timeout, args.timeout)
            self.assertFalse(any(k.startswith(("JXL_", "RS_JXL_")) for k in environment))
            record = dict(command=command, returncode=0, stdout="", stderr="", wall_seconds=1.5)
            if command[0] == str(args.encoder):
                variant = "baseline"
                if "--ma-seed" in command:
                    variant = "seed-" + command[command.index("--ma-seed") + 1]
                elif "--context-predictors" in command:
                    variant = "fast-" + command[command.index("--context-predictors") + 1]
                f.variant = variant
                f.calls.append((f.index, variant, command))
                output = Path(command[command.index("--output") + 1])
                self.assertTrue(Path(command[1]).read_bytes().startswith(bench.PNG_SIGNATURE))
                self.assertFalse(output.exists())
                data = self.measurement(variant)
                if f.measurement_effect:
                    f.measurement_effect(data, variant)
                output.write_bytes(b"x" * 100)
                f.artifacts.append(output)
                record["stdout"] = json.dumps(data)
            else:
                self.assertIn(f"--bits_per_sample={oracle.depth}", command)
                self.assertEqual(command[3:], [f"--bits_per_sample={oracle.depth}", "--num_threads=1", "--quiet"])
                output = Path(command[2])
                self.assertFalse(output.exists())
                output.write_bytes(b"decoded")
                f.artifacts.append(output)
            if f.effect:
                f.effect(command, record)
            return record

        original_write = bench.atomic_write

        def atomic_write(path, values):
            original_write(path, values)
            if path.name == "progress.json":
                f.progress.append(copy.deepcopy(values[0]))
                if f.phase_effect:
                    f.phase_effect(values[0])

        with patch.object(bench, "prepare", side_effect=prepare), \
                patch.object(bench, "run_process", side_effect=process), \
                patch.object(bench, "png_samples", return_value=oracle), \
                patch.object(bench, "compare_pixels", side_effect=f.compare_effect) as pixels, \
                patch.object(bench, "atomic_write", side_effect=atomic_write), \
                patch.dict(os.environ, {"JXL_TEST_LEAK": "1", "RS_JXL_TEST_LEAK": "1"}), \
                patch.object(bench.os, "fsync", wraps=os.fsync) as sync:
            report = bench.benchmark(args)
        f.pixel_checks = pixels.call_count
        f.sync_calls = sync.call_count
        f.rows = [json.loads(line) for line in (args.out / "trials.jsonl").read_text().splitlines()]
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertTrue(all(not artifact.exists() for artifact in f.artifacts))
        self.assertEqual(sorted(path.name for path in args.out.iterdir() if path.is_dir()), [])
        self.assertEqual(json.loads((args.out / "report.json").read_text()), report)
        return report

    def test_defaults_and_invalid_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory))
            self.assertEqual((f.args.effort, f.args.threads, f.args.timeout), (9, 8, 600))
            self.assertEqual(f.args.variants, list(bench.VARIANTS))
            for extra in (["--effort", "0"], ["--threads", "0"], ["--threads", "257"],
                          ["--timeout", "0"], ["--timeout", "inf"], ["--timeout", "nan"],
                          ["--variants", "seed-only"], ["--variants", "baseline", "baseline"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        bench.arguments(f.argv + extra)

    def test_all_six_exact_flags_rotation_provenance_progress_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory))
            report = self.invoke(f)
            self.assertEqual((report["status"], report["completed_cases"], report["passed_trials"]),
                             ("completed", 2, 12))
            for index in range(2):
                calls = [call for call in f.calls if call[0] == index]
                self.assertEqual([v for _, v, _ in calls],
                                 [v for _, v in bench.matrix_order(f.args, 0, index)])
                for _, variant, command in calls:
                    expected = [str(f.args.encoder), command[1], "--effort", "9", "--threads", "8",
                                "--model-costs", "--output", command[8]]
                    if variant.startswith("fast-"):
                        expected += ["--fast-context", "auto12", "--context-bias", "static",
                                     "--context-predictors", variant[5:]]
                    elif variant.startswith("seed-"):
                        expected += ["--ma-seed", variant[5:], "--ma-seed-depth", "3"]
                    self.assertEqual(command, expected)
            self.assertEqual(f.pixel_checks, 12)
            self.assertGreater(f.sync_calls, len(f.rows))
            for row in f.rows:
                self.assertEqual(row["known_case"], row["index"] == 0)
                self.assertEqual(row["preparation"]["conversion"], "metadata-strip")
                self.assertEqual(row["preparation"]["removed_metadata_chunks"], ["tEXt", "iCCP"])
                self.assertEqual(row["measurement"]["seconds"], 0.25)
                self.assertEqual(row["encoder"]["wall_seconds"], 1.5)
            run = json.loads((f.args.out / "run.json").read_text())
            self.assertEqual(run["identity"]["encoder"]["sha256"], bench.file_sha256(f.args.encoder))
            self.assertEqual(len(run["identity"]["scripts"]), len(bench.SCRIPTS))
            for state in f.progress:
                self.assertTrue({"index", "source", "variant", "phase", "pid", "total_cases",
                                 "completed_cases", "completed_trials", "updated_utc", "status"} <= state.keys())
            self.assertEqual(f.progress[-1]["completed_trials"], 12)
            self.assertEqual(f.progress[-1]["status"], "completed")
            self.assertTrue({"prepare", "encode", "decode", "verify_pixels", "verify_source",
                             "case_integrity"} <= {s["phase"] for s in f.progress})

    def test_encoder_failure_timeout_and_pixel_mismatch_do_not_drop_trials(self):
        for failure in ("encoder", "timeout", "pixels", "decoder"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(Path(directory))

                def effect(command, record):
                    encode = command[0] == str(f.args.encoder)
                    if f.index == 0 and f.variant == "baseline":
                        if failure == "encoder" and encode:
                            record.update(returncode=1, stderr="encoder failure")
                        elif failure == "timeout" and encode:
                            record.update(returncode=None, error="Process timeout after 600s")
                        elif failure == "decoder" and not encode:
                            record.update(returncode=1, stderr="decoder failure")

                def compare(expected, actual):
                    if failure == "pixels" and f.index == 0 and f.variant == "baseline":
                        raise ValueError("hidden RGB mismatch")

                f.effect, f.compare_effect = effect, compare
                report = self.invoke(f)
                self.assertEqual((report["completed_trials"], report["failed_trials"]), (12, 1))
                self.assertEqual(len(f.calls), 12)
                failed = next(row for row in f.rows if row["status"] == "failed")
                self.assertIn("error", failed)
                self.assertIsNotNone(failed["encoder"])
                if failure == "timeout":
                    self.assertIn("timeout", failed["encoder"]["error"])

    def test_invalid_input_missing_source_and_hash_mismatch_record_all_variants(self):
        for failure in ("invalid", "jpeg", "missing", "hash"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(Path(directory))
                source = Path(f.cases[0]["original"])
                if failure == "missing":
                    source.unlink()
                else:
                    source.write_bytes(b"\xff\xd8not-png" if failure == "jpeg" else b"broken")
                    if failure != "hash":
                        f.cases[0]["sha256"] = bench.file_sha256(source)
                f.manifest.write_text(json.dumps(dict(cases=f.cases)))
                report = self.invoke(f)
                self.assertEqual((report["failed_trials"], report["completed_trials"]), (6, 12))
                self.assertEqual(len(f.calls), 6)
                self.assertEqual(f.prepared, [f.cases[1]["original"]])
                self.assertTrue(all(r["measurement"] is None for r in f.rows if r["index"] == 0))

    def test_prepare_failure_and_packed_conversion_are_not_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory), count=1)
            f.cases[0]["bit_depth"] = 2
            f.manifest.write_text(json.dumps(dict(cases=f.cases)))
            self.assertEqual(self.invoke(f)["failed_trials"], 0)
            self.assertTrue(all(r["depth"] == 8 and r["preparation"]["conversion"] == "png-packed2-to-rgba8"
                                for r in f.rows))
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory), count=1)
            with patch.object(bench, "prepare", side_effect=ValueError("original oracle mismatch")), \
                    patch.object(bench, "run_process") as process:
                report = bench.benchmark(f.args)
            self.assertEqual((report["failed_trials"], report["completed_cases"]), (6, 1))
            process.assert_not_called()

    def test_mutation_invalidates_already_durable_case_rows_including_final_check(self):
        for when in ("variant", "final"):
            with self.subTest(when=when), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(Path(directory))

                def effect(command, record):
                    if when == "variant" and f.index == 0 and f.variant == "seed-only":
                        prior = (f.args.out / "trials.jsonl").read_text().splitlines()
                        self.assertTrue(any(json.loads(line)["status"] == "passed" for line in prior))
                        Path(f.cases[0]["original"]).write_bytes(b"mutated")

                def phase_effect(state):
                    if when == "final" and state["phase"] == "case_integrity" and state["index"] == 0:
                        Path(f.cases[0]["original"]).write_bytes(b"mutated at case end")

                f.effect, f.phase_effect = effect, phase_effect
                report = self.invoke(f)
                self.assertEqual((report["failed_trials"], report["integrity_failed_cases"]), (6, 1))
                self.assertTrue(all(r["status"] == "failed" and "integrity_error" in r
                                    for r in f.rows if r["index"] == 0))
                self.assertTrue(all(r["status"] == "passed" for r in f.rows if r["index"] == 1))
                self.assertEqual(len({(r["index"], r["variant"]) for r in f.rows}), 12)

    def test_zero_work_receipts_and_full_predictor_confirmation(self):
        for variant in bench.VARIANTS:
            data = self.measurement(variant)
            if variant.startswith("seed-"):
                data["ma_seed"].update(dict.fromkeys(bench.SEED_COUNTERS, 0))
                self.assertEqual(bench.receipt_validation(data, variant, 100, 6)["status"], "bypassed")
            elif variant.startswith("fast-"):
                data["fast_context"].update(dict.fromkeys(bench.FAST_COUNTERS, 0))
                self.assertEqual(bench.receipt_validation(data, variant, 100, 6)["status"], "bypassed")
        for predictors in (None, "basic", "wrong"):
            data = self.measurement("fast-full")
            if predictors is None:
                del data["fast_context"]["predictors"]
            else:
                data["fast_context"]["predictors"] = predictors
            with self.assertRaisesRegex(ValueError, "predictors"):
                bench.receipt_validation(data, "fast-full", 100, 6)
        data = self.measurement("seed-only")
        data["ma_seed"]["learners"] = 0
        data["ma_seed"]["prepared_records"] = 1
        with self.assertRaises(ValueError):
            bench.receipt_validation(data, "seed-only", 100, 6)
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory), count=1)

            def idle(data, variant):
                if data["ma_seed"]:
                    data["ma_seed"].update(dict.fromkeys(bench.SEED_COUNTERS, 0))

            f.measurement_effect = idle
            self.assertEqual(self.invoke(f)["failed_trials"], 0)
            self.assertTrue(all(r["pixels_verified"] for r in f.rows))

    def test_bad_receipt_is_separate_from_verified_pixels_and_raw_measurement_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory), count=1)

            def bad_receipt(data, variant):
                if variant == "fast-full":
                    del data["fast_context"]["predictors"]

            f.measurement_effect = bad_receipt
            self.assertEqual(self.invoke(f)["failed_trials"], 1)
            row = next(r for r in f.rows if r["variant"] == "fast-full")
            self.assertTrue(row["pixels_verified"])
            self.assertIsNotNone(row["measurement"])
            self.assertEqual(row["receipt_validation"]["status"], "failed")

    def test_stop_then_resume_skips_failures_prepares_only_remaining_and_keeps_run_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory))

            def stop(command, record):
                (f.args.out / "STOP").touch()
                if f.variant == "baseline":
                    record.update(returncode=1, stderr="intentional failure")

            f.effect = stop
            report = self.invoke(f)
            self.assertEqual((report["status"], report["completed_trials"]), ("stopped", 6))
            first_rows = copy.deepcopy(f.rows)
            run_bytes = (f.args.out / "run.json").read_bytes()
            (f.args.out / "STOP").unlink()
            f.effect = None
            f.calls.clear()
            f.prepared.clear()
            report = self.invoke(f, ["--resume"])
            self.assertEqual((report["status"], report["completed_trials"], report["failed_trials"]),
                             ("completed", 12, 1))
            self.assertEqual(f.rows[:6], first_rows)
            self.assertEqual(f.prepared, [f.cases[1]["original"]])
            self.assertTrue(all(index == 1 for index, _, _ in f.calls))
            self.assertEqual((f.args.out / "run.json").read_bytes(), run_bytes)
            f.calls.clear()
            f.prepared.clear()
            self.invoke(f, ["--resume"])
            self.assertEqual((f.calls, f.prepared), ([], []))

    def test_partial_case_resume_does_not_repeat_terminal_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory), count=1)
            self.invoke(f)
            first = (f.args.out / "trials.jsonl").read_text().splitlines(keepends=True)[0]
            (f.args.out / "trials.jsonl").write_text(first)
            f.calls.clear()
            self.assertEqual(self.invoke(f, ["--resume"])["completed_trials"], 6)
            self.assertEqual(len(f.calls), 5)
            self.assertNotIn(json.loads(first)["variant"], [v for _, v, _ in f.calls])

    def test_resume_rejects_binary_manifest_script_and_config_identity_changes(self):
        for change in ("encoder", "decoder", "manifest", "script", "config"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(Path(directory), count=1)
                self.invoke(f)
                extra = ["--resume"]
                if change in ("encoder", "decoder"):
                    getattr(f.args, change).write_bytes(b"changed binary")
                elif change == "manifest":
                    f.manifest.write_text(f.manifest.read_text() + " ")
                elif change == "config":
                    extra += ["--threads", "7"]
                else:
                    path = f.args.out / "run.json"
                    run = json.loads(path.read_text())
                    run["identity"]["scripts"][0]["sha256"] = "0" * 64
                    path.write_text(json.dumps(run))
                journal_bytes = (f.args.out / "trials.jsonl").read_bytes()
                with patch.object(bench, "run_process") as process, self.assertRaisesRegex(ValueError, "Resume"):
                    bench.benchmark(bench.arguments(f.argv + extra))
                process.assert_not_called()
                self.assertEqual((f.args.out / "trials.jsonl").read_bytes(), journal_bytes)

    def test_resume_rejects_duplicate_malformed_torn_and_wrong_identity_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory), count=1)
            self.invoke(f)
            journal = f.args.out / "trials.jsonl"
            original = journal.read_text()
            wrong = copy.deepcopy(f.rows[0])
            wrong["source"] = "wrong.png"
            no_measurement = copy.deepcopy(f.rows[0])
            no_measurement["measurement"] = None
            malformed_process = copy.deepcopy(f.rows[0])
            malformed_process["encoder"] = dict(returncode=0)
            for text in (original + original.splitlines()[0] + "\n", original.rstrip("\n"),
                         "{}\n", "not-json\n", "\n", json.dumps(wrong) + "\n",
                         json.dumps(no_measurement) + "\n", json.dumps(malformed_process) + "\n",
                         '{"index":0,"index":1}\n'):
                journal.write_text(text)
                with self.subTest(text=text[:40]), patch.object(bench, "run_process") as process:
                    with self.assertRaises(ValueError):
                        bench.benchmark(bench.arguments(f.argv + ["--resume"]))
                    process.assert_not_called()

    def test_resume_rehashes_completed_sources_without_preparing_or_running_again(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory), count=1)
            self.invoke(f)
            Path(f.cases[0]["original"]).write_bytes(b"changed after completed run")
            f.calls.clear()
            f.prepared.clear()
            report = self.invoke(f, ["--resume"])
            self.assertEqual((f.calls, f.prepared), ([], []))
            self.assertEqual((report["failed_trials"], report["integrity_failed_cases"]), (6, 1))

    def test_journal_write_failure_stops_without_silently_running_more_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory), count=1)
            original_open = Path.open

            def open_path(path, mode="r", *args, **kwargs):
                if path.name == "trials.jsonl" and mode == "a":
                    raise OSError("journal disk failure")
                return original_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", new=open_path):
                report = self.invoke(f)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(len(f.calls), 1)
            self.assertIn("journal disk failure", report["errors"][0])

    def test_new_output_is_exclusive_and_stop_before_start_is_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory), count=1)
            f.args.out.mkdir()
            sentinel = f.args.out / "keep.txt"
            sentinel.write_text("untouched")
            with self.assertRaises(FileExistsError):
                bench.benchmark(f.args)
            self.assertEqual(sentinel.read_text(), "untouched")
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory), count=1)
            self.invoke(f)
            (f.args.out / "trials.jsonl").write_text("")
            (f.args.out / "STOP").touch()
            f.prepared.clear()
            f.calls.clear()
            report = self.invoke(f, ["--resume"])
            self.assertEqual((report["status"], report["completed_trials"]), ("stopped", 0))
            self.assertEqual((f.prepared, f.calls), ([], []))

    def test_manifest_validation_and_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(Path(directory), count=1)
            invalid = [[], {}, dict(cases=[]), dict(cases=[{}]), dict(cases=f.cases * 2)]
            for key, value in (("original", "relative.png"), ("original", str(f.root / "input.jpg")),
                               ("sha256", "wrong"), ("known_case", 1), ("width", True)):
                row = dict(f.cases[0])
                row[key] = value
                invalid.append(dict(cases=[row]))
            for spec in invalid:
                f.manifest.write_text(json.dumps(spec))
                with self.subTest(spec=spec), self.assertRaises(ValueError):
                    bench.load_cases(f.manifest)
            for text in ('{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}'):
                with self.assertRaises(ValueError):
                    bench.loads(text)


if __name__ == "__main__":
    unittest.main()
