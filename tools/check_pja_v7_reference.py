"""Development-only independent verification of frozen Pascal probability tables.

Uses 70-digit Decimal exp, rather than the binary64 exp used to generate
the Rust constants. No encoder or native code is invoked.
"""
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / "crates/rs-jxl-plus/src/probability_tables.rs").read_text()
    arrays = {}
    for name in ("SQUASH", "STRETCH"):
        body = re.search(rf"const {name}:.*?=\s*\[(.*?)\];", source, re.S).group(1)
        arrays[name] = [int(v) for v in re.findall(r"-?\d+", body)]
    with localcontext() as ctx:
        ctx.prec = 70
        squash = []
        clearance = Decimal(1)
        for i in range(4096):
            q = Decimal(4096) / (1 + (Decimal(2048-i) / 256).exp())
            integer = q.to_integral_value(rounding=ROUND_HALF_EVEN)
            clearance = min(clearance, abs(abs(q-integer)-Decimal("0.5")))
            squash.append(max(1, min(4095, int(integer))))
        stretch = [2047] * 4096
        pi = 0
        for x in range(-2048, 2048):
            while pi <= 4095 and pi <= squash[x+2048]:
                stretch[pi] = x
                pi += 1
        stretch[0] = stretch[1]
        assert arrays["SQUASH"] == squash, "squash differs from high precision Pascal formula"
        assert arrays["STRETCH"] == stretch, "stretch differs from Pascal inverse recurrence"
        print(f"All 8192 frozen entries agree; minimum rounding-boundary distance {clearance}")


if __name__ == "__main__":
    main()
