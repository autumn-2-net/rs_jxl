import copy
import unittest

from summarize_jpeg_entropy import collect, validate_audit


def measurement():
    return dict(audit_ac=True, ac_lz_enabled=False, bytes=200, seconds=1.0,
                size_breakdown=dict(ac_data_bits=1000, ac_model_bits=100),
                ac_audit=dict(streams=2, nonempty_streams=1, tokens=100, raw_contexts=6,
                    active_contexts=3, histograms=2, extra_bits=200, selector_bits=0,
                    ans=True, lz77=False, model_symbol_bits=700.0, cluster_entropy_bits=680.0,
                    raw_context_entropy_bits=580.0, extra_bit_entropy_bits=150.0, seconds=0.1))


class AuditTests(unittest.TestCase):
    def test_valid_and_zero_token_diagnostics(self):
        validate_audit(measurement())
        m = measurement()
        for key in ("tokens", "nonempty_streams", "active_contexts", "extra_bits",
                    "model_symbol_bits", "cluster_entropy_bits", "raw_context_entropy_bits",
                    "extra_bit_entropy_bits"):
            m["ac_audit"][key] = 0
        validate_audit(m)

    def test_invalid_numbers_counts_and_receipts(self):
        for key in ("streams", "tokens", "selector_bits"):
            for value in (-1, True, 1.5, None):
                m = measurement()
                m["ac_audit"][key] = value
                with self.assertRaises(ValueError):
                    validate_audit(m)
        for key in ("model_symbol_bits", "cluster_entropy_bits", "seconds"):
            for value in (-1, True, float("nan"), float("inf"), None):
                m = measurement()
                m["ac_audit"][key] = value
                with self.assertRaises(ValueError):
                    validate_audit(m)
        for key, value in (("raw_context_entropy_bits", 800), ("cluster_entropy_bits", 900),
                           ("extra_bit_entropy_bits", 201), ("extra_bits", 1001),
                           ("seconds", 2), ("lz77", True), ("ans", 1), ("histograms", 257),
                           ("active_contexts", 7), ("nonempty_streams", 3)):
            m = measurement()
            m["ac_audit"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_audit(m)
        with self.assertRaises(ValueError):
            validate_audit({})

    @staticmethod
    def report():
        m = measurement()
        return dict(status="passed", cases=[dict(source="original.jpg", sha256="abc", pixels=100,
                    original_bytes=300, variants=dict(candidate=[dict(verified=True, measurement=m),
                    dict(verified=True, measurement=copy.deepcopy(m))]))])

    def test_sum_once_per_image_and_separate_gap_meanings(self):
        report = self.report()
        report["cases"][0]["variants"]["candidate"][1]["measurement"]["ac_audit"]["seconds"] = 0.2
        r = collect(report, "candidate")
        self.assertEqual(r["images"], 1)
        t = r["totals"]
        self.assertEqual(t["jxl_bytes"], 200)
        self.assertEqual(t["weighted_bpp"], 16)
        self.assertEqual(t["table_fit_gap_bits"], 20)
        self.assertEqual(t["cluster_gap_bits"], 100)
        self.assertEqual(t["extra_bit_bias_gap_bits"], 50)
        self.assertEqual(t["coder_framing_residual_bits"], 100)

    def test_failed_unverified_duplicate_and_inconsistent_repeats_rejected(self):
        r = self.report()
        r["status"] = "failed"
        with self.assertRaises(ValueError):
            collect(r, "candidate")

        r = self.report()
        r["cases"][0]["variants"]["candidate"][0]["verified"] = False
        with self.assertRaises(ValueError):
            collect(r, "candidate")
        r = self.report()
        r["cases"].append(copy.deepcopy(r["cases"][0]))
        with self.assertRaises(ValueError):
            collect(r, "candidate")
        r = self.report()
        r["cases"][0]["variants"]["candidate"][1]["measurement"]["ac_audit"]["tokens"] = 99
        with self.assertRaises(ValueError):
            collect(r, "candidate")

    def test_repeated_wire_cost_shift_cannot_be_hidden_by_same_total_size(self):
        r = self.report()
        wire = r["cases"][0]["variants"]["candidate"][1]["measurement"]["size_breakdown"]
        wire["ac_data_bits"] -= 100
        wire["ac_model_bits"] += 100
        with self.assertRaisesRegex(ValueError, "wire costs"):
            collect(r, "candidate")


if __name__ == "__main__":
    unittest.main()
