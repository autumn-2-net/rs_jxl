"""Argument and command checks only; no codec subprocesses or benchmark runs."""

from contextlib import redirect_stderr
from io import StringIO
import unittest
from unittest.mock import patch

import compare_cpp_port as port
import compare_rust_checkpoints as checkpoints
import compare_three_encoders as three
import verify_port_corpus as corpus


class SampleLimitTests(unittest.TestCase):
    def comparisons(self):
        return (
            (checkpoints, ["--before", "before", "--after", "after", "--decoder", "djxl",
                           "--json", "report.json", "image.png"]),
            (three, ["--official", "official", "--cpp", "cpp", "--rust", "rust",
                     "--decoder", "djxl", "--json", "report.json", "image.png"]),
        )

    def test_comparison_defaults_do_not_inject_sample_limits(self):
        for module, argv in self.comparisons():
            with self.subTest(module=module.__name__):
                parser = module.argument_parser()
                args = parser.parse_args(argv)
                self.assertIsNone(args.sample_limit)
                self.assertIn("omitted", parser.format_help())
                command = port.encoder_command("rust", "rust", "in", "out", args.threads,
                                               args.effort, "full-scheduler", args.sample_limit)
                self.assertNotIn("--source-sample-limit", command)

    def test_explicit_comparison_caps_are_forwarded_and_validated(self):
        for module, argv in self.comparisons():
            for limit in (1, 1_048_576, 33_554_432):
                args = module.argument_parser().parse_args([*argv, "--sample-limit", str(limit)])
                command = port.encoder_command("rust", "rust", "in", "out", args.threads,
                                               args.effort, "full-scheduler", args.sample_limit)
                self.assertEqual(command[command.index("--source-sample-limit") + 1], str(limit))
            for limit in (0, -1):
                with self.subTest(module=module.__name__, limit=limit), \
                        redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
                    module.main([*argv, "--sample-limit", str(limit)])
                self.assertEqual(caught.exception.code, 2)

    def test_corpus_defaults_and_explicit_sample_cap(self):
        args = corpus.argument_parser().parse_args([])
        self.assertIsNone(args.source_sample_limit)
        self.assertNotIn("--source-sample-limit", corpus.encoder_command(args, "rust", "in", "out"))
        args = corpus.argument_parser().parse_args(["--source-sample-limit", "123"])
        command = corpus.encoder_command(args, "rust", "in", "out")
        self.assertEqual(command[command.index("--source-sample-limit") + 1], "123")
        for argv in (["--source-sample-limit", "0"], ["--source-sample-limit", "-1"],
                     ["--mode", "legacy", "--source-sample-limit", "1"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
                corpus.main(argv)
            self.assertEqual(caught.exception.code, 2)

    def test_corpus_size_limits_are_explicit_positive_options(self):
        parser = corpus.argument_parser()
        defaults = parser.parse_args([])
        for name in ("max_cases", "max_pixels", "max_total_pixels", "max_input_bytes", "max_output_bytes"):
            flag = "--" + name.replace("_", "-")
            with self.subTest(flag=flag):
                self.assertIsNone(getattr(defaults, name))
                self.assertIn(flag, parser.format_help())
                for limit in (1, 1 << 100):
                    args = parser.parse_args([flag, str(limit)])
                    self.assertEqual(getattr(args, name), limit)
                for limit in (0, -1):
                    with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
                        corpus.main([flag, str(limit)])
                    self.assertEqual(caught.exception.code, 2)

    def test_corpus_worker_and_time_settings_do_not_have_private_caps(self):
        args = corpus.argument_parser().parse_args(["--threads", "256"])
        self.assertEqual(args.threads, 256)
        with patch.object(corpus, "select_sources", return_value=([], [])), \
                patch.object(corpus, "verify") as verify:
            corpus.main(["--threads", "16", "--timeout", "900", "--total-timeout", "7200"])
            self.assertEqual(verify.call_args.args[0].timeout, 900)
            self.assertEqual(verify.call_args.args[0].total_timeout, 7200)
        for value in ("0", "-1", "inf", "nan"):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                corpus.main(["--timeout", value])


if __name__ == "__main__":
    unittest.main()
