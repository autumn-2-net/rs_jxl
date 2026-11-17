"""Analyze opt-in JPEG AC audits; empirical references are not JXL size bounds."""

import argparse
import json
import math
from pathlib import Path

from jpeg_lab_receipts import AUDIT_COUNTS as COUNTS, AUDIT_COSTS as COSTS, validate_audit


def collect(report, variant):
    if report.get("status") != "passed" or not report.get("cases"):
        raise ValueError("Expected a completed nonempty report")
    rows = []
    identities = set()
    for case in report["cases"]:
        identity = (case["source"], case["sha256"])
        if identity in identities:
            raise ValueError("Duplicate report input")
        identities.add(identity)
        samples = case["variants"][variant]
        if not samples:
            raise ValueError("Empty variant")
        selected = samples[0]["measurement"]
        for sample in samples:
            if sample.get("verified") is not True:
                raise ValueError("Unverified sample")
            m = sample["measurement"]
            validate_audit(m)
            for field in (*COUNTS, *COSTS[:-1], "ans", "lz77"):
                if m["ac_audit"][field] != selected["ac_audit"][field]:
                    raise ValueError("Repeated diagnostic counts differ")
            if m["bytes"] != selected["bytes"]:
                raise ValueError("Repeated output sizes differ")
            if m["size_breakdown"] != selected["size_breakdown"]:
                raise ValueError("Repeated actual wire costs differ")
        a = selected["ac_audit"]
        wire = selected["size_breakdown"]
        rows.append({
            "source": case["source"], "pixels": case["pixels"],
            "jpeg_bytes": case["original_bytes"], "jxl_bytes": selected["bytes"],
            "total_bits": 8 * selected["bytes"], "ac_data_bits": wire["ac_data_bits"],
            "ac_model_bits": wire["ac_model_bits"],
            "symbol_and_state_bits": wire["ac_data_bits"] - a["extra_bits"] - a["selector_bits"],
            **{k: a[k] for k in COUNTS + COSTS},
            "ans": a["ans"], "lz77": a["lz77"],
            "table_fit_gap_bits": a["model_symbol_bits"] - a["cluster_entropy_bits"],
            "cluster_gap_bits": a["cluster_entropy_bits"] - a["raw_context_entropy_bits"],
            "extra_bit_bias_gap_bits": a["extra_bits"] - a["extra_bit_entropy_bits"],
            "coder_framing_residual_bits": wire["ac_data_bits"] - a["selector_bits"]
                - a["extra_bits"] - a["model_symbol_bits"],
        })
    keys = ("pixels", "jpeg_bytes", "jxl_bytes", "total_bits", "ac_data_bits", "ac_model_bits", "symbol_and_state_bits",
            "tokens", "extra_bits", "selector_bits", "model_symbol_bits", "cluster_entropy_bits",
            "raw_context_entropy_bits", "extra_bit_entropy_bits", "table_fit_gap_bits",
            "cluster_gap_bits", "extra_bit_bias_gap_bits", "coder_framing_residual_bits")
    totals = {k: math.fsum(row[k] for row in rows) for k in keys}
    totals["weighted_bpp"] = totals["total_bits"] / totals["pixels"]
    totals["ans_images"] = sum(row["ans"] for row in rows)
    totals["lz_images"] = sum(row["lz77"] for row in rows)
    return dict(images=len(rows), totals=totals, cases=rows,
                interpretation="Fixed selected tokens/configs. Entropies omit model signaling; "
                "extra-bit entropy is NOT an available JXL coding mode. Diagnostic time is not encode speed.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--variant", default="candidate")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = collect(json.loads(args.report.read_text(encoding="utf-8")), args.variant)
    if args.out:
        with args.out.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
    t = result["totals"]
    print(f"{result['images']} images, {int(t['jxl_bytes']):,} JXL bytes, {t['weighted_bpp']:.6f} weighted bpp")
    for key in ("ac_data_bits", "extra_bits", "symbol_and_state_bits", "model_symbol_bits", "table_fit_gap_bits",
                "cluster_gap_bits", "extra_bit_bias_gap_bits", "coder_framing_residual_bits"):
        print(f"{key:30s} {t[key] / 8:14,.3f} byte equivalents  {100*t[key]/t['total_bits']:8.4f}% of file")
    print(result["interpretation"])


if __name__ == "__main__":
    main()
