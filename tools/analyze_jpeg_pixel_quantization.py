"""Isolate output quantization from float reconstruction using frozen decoders.

Development diagnostic only. Does not modify either codec or relax comparisons.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from benchmark_jpeg_pixel_decode import command, differences
from compare_cpp_port import check_process, clean_environment, file_sha256, run_process
from verify_decoder_features import portable


def reference_table(repository):
    source = subprocess.run(
        ["git", "-C", str(repository), "show", "v0.12.0:lib/jxl/render_pipeline/stage_write.cc"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout
    initializer = source.split("const float kDither[48 * 32] = {", 1)[1].split("};", 1)[0]
    values = json.loads("[" + initializer.rstrip().rstrip(",") + "]")
    table = np.array(values, dtype=np.float32).reshape(32, 48)
    np.testing.assert_array_equal(table[:, :16], table[:, 32:])
    return table[:, :32].copy(), hashlib.sha256(source.encode()).hexdigest()


def quantize(pixels, table=None):
    result = np.empty(pixels.shape, dtype=np.uint8)
    x = np.arange(pixels.shape[1])
    for y in range(pixels.shape[0]):
        for c in range(pixels.shape[2]):
            scaled = pixels[y, :, c] * np.float32(255)
            if table is not None:
                scaled += table[(y + c * 13) % 32, (x + c * 23) % 32]
            result[y, :, c] = np.rint(np.clip(scaled, 0, 255)).astype(np.uint8)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--label", default="wallhaven-yq5e9x.jpg")
    parser.add_argument("--cpp-repo", type=Path, default=Path("../jxl_python/libjxl"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.report.read_text(encoding="utf-8"))
    case = next(r["case"] for r in baseline["results"] if r["case"]["label"] == args.label)
    jxl = Path(case["jxl"])
    if file_sha256(jxl) != case["jxl_sha256"]:
        raise ValueError("Changed input")
    cpp, source_hash = reference_table(args.cpp_repo)
    ranks = Path("vendor/jxl-rs/jxl/src/util/dither_32x32.bin").read_bytes()
    rust = (np.frombuffer(ranks, dtype=np.uint8).astype(np.float32).reshape(32, 32) * 2 - 255) / 512
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(case=case, binaries=baseline["binaries"], trials=[],
                  cpp_source_sha256=source_hash, rust_ranks_sha256=hashlib.sha256(ranks).hexdigest())
    env, result["environment"] = clean_environment(os.environ)
    floats, integers = {}, {}
    with tempfile.TemporaryDirectory(prefix="quantizer-", dir=args.out) as temp:
        for name, binary in baseline["binaries"].items():
            path = Path(binary["path"])
            if file_sha256(path) != binary["sha256"]:
                raise ValueError("Changed binary")
            for bits, extension, dest in [(8, "pam", integers), (32, "pfm", floats)]:
                output = Path(temp) / f"{name}.{extension}"
                argv = command(name, path, jxl, 1, output=output, bits=bits)
                trial = run_process(argv, env, 60)
                result["trials"].append(trial)
                check_process(trial)
                dest[name] = portable(output)
        result["actual_u8"] = differences(integers["rust"], integers["official"])
        result["actual_f32"] = differences(floats["rust"], floats["official"])
        result["same_quantizer"] = {}
        for name, table in [("none", None), ("cpp", cpp), ("rust_upstream", rust)]:
            result["same_quantizer"][name] = differences(
                quantize(floats["rust"], table), quantize(floats["official"], table))
        result["float_to_actual_u8"] = {
            f"{name}_{table_name}": differences(quantize(floats[name], table), integers[name])
            for name in ("rust", "official")
            for table_name, table in [("cpp", cpp), ("rust_upstream", rust)]
        }
    result["temporary_outputs_cleaned"] = not any(args.out.glob("quantizer-*"))
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, metrics in result["same_quantizer"].items():
        print(f"Common {name}: changed={metrics['different_pixel_percent']:.7f}% max={metrics['max_abs_error']}")
    for name, metrics in result["float_to_actual_u8"].items():
        print(f"Replay {name}: changed={metrics['different_pixel_percent']:.7f}% max={metrics['max_abs_error']}")


if __name__ == "__main__":
    main()
