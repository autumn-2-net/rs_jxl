"""Small deterministic recurrence/cache fixtures; no input or encoded files written."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

import benchmark_multiframe_native as bench


def fixtures():
    rng = np.random.default_rng(20261027)
    x, a, b = [rng.integers(0, 256, (192, 256, 3), dtype=np.uint8)
               for _ in range(3)]
    x2, a2 = x.copy(), a.copy()
    x2[64:72, 80:88] ^= np.uint8(0x7f)
    a2[64:72, 80:88] ^= np.uint8(0x7f)
    crops = [x]
    for value in (17, 69, 128):
        current = x.copy()
        current[32:160, 32:224] ^= np.uint8(value)
        crops.append(current)
    sparse = [x]
    for value in (17, 69, 128):
        current = x.copy()
        for y, z in [(1, 1), (25, 25), (161, 225), (185, 249)]:
            current[y:y + 3, z:z + 3] ^= np.uint8(value)
        sparse.append(current)
    return {"revisit-first": [x, a, b, x2],
            "revisit-middle": [x, a, b, a2],
            "repeated-crop": crops, "subtile-changes": sparse}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--effort", type=int, default=9, choices=range(1, 10))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--modes", nargs="+", choices=bench.MODES,
                        default=["auto-adaptive", "auto-retained", "auto-lookahead", "independent"])
    parser.add_argument("--cases", nargs="+", choices=tuple(fixtures()), help="optional fixture subset")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    args.reference, args.legacy = "auto", False
    os.environ["RS_JXL_NATIVE_LIB"] = str(args.native.resolve(strict=True))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    import rs_jxl

    report = {"created": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "native": str(args.native), "effort": args.effort, "threads": args.threads,
              "reference": args.reference, "legacy": False, "datasets": []}
    for name, frames in fixtures().items():
        if args.cases and name not in args.cases:
            continue
        height, width = frames[0].shape[:2]
        dataset = {"folder": name, "width": width, "height": height,
                   "settings": {"layout": "hwc", "bits_per_sample": 8, "color_encoding": "srgb"},
                   "sources": [{"path": Path(f"frame{i}.png"), "raw": b"", "pixels": frame,
                                "info": {"extra_channels": []}, "jpeg": False}
                               for i, frame in enumerate(frames)]}
        entry = {"folder": name, "width": width, "height": height, "frames": len(frames),
                 "input_seconds": 0.0, "results": []}
        report["datasets"].append(entry)
        for mode in args.modes:
            try:
                row = bench.measure(rs_jxl, dataset, mode, args)
            except Exception as error:
                report["status"] = "failed"
                row = {"mode": mode, "status": "failed", "error": str(error)}
            entry["results"].append(row)
            print(json.dumps({"fixture": name, **{k: v for k, v in row.items()
                                                 if k != "search_frames"}}), flush=True)
    print(*bench.write_reports(report, args.out), sep="\n")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
