"""Create fixed official lossless JXL streams for an existing PNG subset report."""

import argparse
import json
import os
from pathlib import Path
import tempfile

from compare_cpp_port import (
    binary_info, check_process, clean_environment, compare_pixels,
    file_sha256, png_payload, png_samples, run_process,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--cjxl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="NEW directory")
    parser.add_argument("--labels", nargs="+", help="Subset report labels to prepare")
    parser.add_argument("--fixed-gradient", action="store_true",
                        help="Explicit predictor-5 route probe, not default e9")
    args = parser.parse_args()
    cases = json.loads(args.report.read_text(encoding="utf-8"))["cases"]
    if args.labels:
        missing = set(args.labels) - {c["label"] for c in cases if c["kind"] == "png"}
        if missing:
            parser.error(f"Unknown PNG labels: {sorted(missing)}")
        cases = [c for c in cases if c["label"] in args.labels]
    args.out.mkdir(parents=False, exist_ok=False)
    env, environment = clean_environment(os.environ)
    binary = binary_info(args.cjxl, env, 10)
    check_process(binary["version_probe"])
    manifest = dict(binary=binary, environment=environment, cases=[], status="running")
    try:
        with tempfile.TemporaryDirectory(prefix="source-", dir=args.out) as temp:
            for index, case in enumerate(cases):
                if case["kind"] != "png":
                    continue
                source = Path(case["path"]).resolve(strict=True)
                original_hash = file_sha256(source)
                original = source.read_bytes()
                payload, _ = png_payload(original)
                expected = png_samples(original)
                compare_pixels(expected, png_samples(payload))
                snapshot = Path(temp) / "pixels.png"
                snapshot.write_bytes(payload)
                output = (args.out / f"{index:02d}.jxl").resolve()
                command = [binary["path"], str(snapshot), str(output), "-d", "0",
                           "-e", "9", "--lossless_jpeg=0", "--keep_invisible=1",
                           "--num_threads=8"]
                if args.fixed_gradient:
                    command += ["-P", "5", "-I", "0", "-E", "0"]
                result = run_process(command, env, 300)
                check_process(result)
                if file_sha256(source) != original_hash:
                    raise ValueError(f"Source changed while preparing: {source}")
                size = output.stat().st_size
                manifest["cases"].append(dict(
                    label=case["label"], original=str(source), jxl=str(output),
                    original_sha256=original_hash, jxl_sha256=file_sha256(output),
                    width=expected.width, height=expected.height, depth=expected.depth,
                    jxl_bytes=size, bpp=8 * size / (expected.width * expected.height),
                    command=command, encode=result,
                ))
                print(f"{case['label']}: {size} bytes", flush=True)
        if file_sha256(Path(binary["path"])) != binary["sha256"]:
            raise ValueError("Encoder changed during preparation")
        manifest["status"] = "prepared; decoder verification still required"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
