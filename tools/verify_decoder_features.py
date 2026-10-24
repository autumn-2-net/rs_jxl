"""Development-only decoder feature oracle; no fixture paths are embedded.

Requires NumPy and the pinned jxl-rs resources/test corpus. All raster outputs
are temporary. Nonzero tolerance is explicit and only for rendered floating
pixels; ordinary lossless exactness is covered separately by verify_decoder.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np


def run(command: list[str], timeout: float) -> float:
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-6000:])
    return time.perf_counter() - start


def arrays(path: Path) -> list[np.ndarray]:
    # djxl can concatenate multiple NPY arrays for distinct-sized raw layers.
    result = []
    with path.open("rb") as stream:
        while stream.tell() < path.stat().st_size:
            values = np.load(stream, allow_pickle=False)
            if values.ndim == 4:
                result.extend(values)
            else:
                result.append(values)
    return result


def compare(actual: list[np.ndarray], expected: list[np.ndarray], atol: float) -> dict:
    if not actual or not expected:
        raise AssertionError("oracle comparison requires decoded frames")
    if len(actual) != len(expected):
        raise AssertionError(f"frame count differs: {len(actual)} != {len(expected)}")
    maximum = 0.0
    different = 0
    nonfinite = 0
    for a, b in zip(actual, expected):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"shape/type differ: {a.shape}/{a.dtype} != {b.shape}/{b.dtype}")
        same = a.view(np.uint32) == b.view(np.uint32) if a.dtype == np.float32 else a == b
        finite = np.isfinite(a) & np.isfinite(b)
        different += int(np.count_nonzero(~same))
        nonfinite += int(np.count_nonzero(~same & ~finite))
        if np.any(finite):
            maximum = max(maximum, float(np.max(np.abs(a[finite].astype(np.float64) - b[finite]))))
    return {"frames": len(actual), "max_abs_error": maximum, "different_words": different,
            "nonfinite_mismatches": nonfinite,
            "passed": different == 0 or (atol > 0 and nonfinite == 0 and maximum <= atol)}


def portable(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        magic = stream.readline().strip()
        if magic == b"P7":
            header = {}
            while (line := stream.readline().strip()) != b"ENDHDR":
                if not line:
                    raise ValueError("truncated PAM header")
                key, value = line.split(maxsplit=1)
                header[key] = value
            width, height, channels = (int(header[k]) for k in (b"WIDTH", b"HEIGHT", b"DEPTH"))
            dtype = np.uint8 if int(header[b"MAXVAL"]) <= 255 else np.dtype(">u2")
            return np.frombuffer(stream.read(), dtype=dtype).reshape(height, width, channels)
        if magic not in (b"PF", b"Pf"):
            raise ValueError("unsupported portable oracle output")
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        dtype = np.dtype("<f4" if scale < 0 else ">f4")
        values = np.frombuffer(stream.read(), dtype=dtype).reshape(height, width, 3 if magic == b"PF" else 1)
        return values[::-1].astype(np.float32) * abs(scale)


def raw_layers(args: argparse.Namespace, name: str, source: str, folder: Path) -> dict:
    cmyk = name == "layers"
    own = folder / "rust.npy"
    reference = folder / ("reference.pam" if cmyk else "reference.pfm")
    rust_seconds = run([str(args.rs_jxl.resolve()), "decode", source, str(own), "--bits",
        "8" if cmyk else "32", "--no-coalescing", "--output-frames", "--metadata-out",
        str(folder / "rust.json"), "--allow-implicit-level10"], args.timeout)
    official_seconds = run([str(args.djxl.resolve()), source, str(reference), "--no_coalescing",
        "--output_frames", "--output_extra_channels"], args.timeout)
    natural = lambda p: int(p.stem.rsplit("-", 1)[1])
    actual = [a for p in sorted(folder.glob("rust-*.npy"), key=natural) for a in arrays(p)]
    primary = [p for p in folder.glob(reference.name + "-*" + reference.suffix) if "-ec" not in p.stem]
    expected = []
    for path in sorted(primary, key=natural):
        values = portable(path)
        if cmyk:
            # This fixture's header order is Black, Alpha. djxl PAM interleaves
            # primary alpha first (CMYAK); our NPY follows source order (CMYKA).
            if values.shape[-1] != 5:
                raise ValueError("CMYK raw-layer fixture must have five channels")
            values = values[..., [0, 1, 2, 4, 3]]
        else:
            extras = list(folder.glob(path.stem + "-ec*.pfm"))
            if len(extras) != 1:
                raise ValueError("RGBA fixture must export exactly one alpha plane")
            alpha = portable(extras[0])
            values = np.concatenate((values, alpha), axis=-1)
        expected.append(values)
    tolerance = 0.0 if cmyk else args.atol
    return {"rust_seconds": rust_seconds, "official_seconds": official_seconds,
            "oracle_format": "PAM native CMYKA" if cmyk else "PFM RGB + alpha",
            "atol": tolerance, **compare(actual, expected, tolerance)}


def profile_inspection(args: argparse.Namespace, source: str, folder: Path) -> dict:
    profiles = []
    for bits in (8, 32):
        inspect_icc, decode_icc = folder / f"inspect-{bits}.icc", folder / f"decode-{bits}.icc"
        common = [str(args.rs_jxl.resolve()), "decode", source, "--bits", str(bits),
                  "--allow-implicit-level10"]
        run([*common, "--icc-out", str(inspect_icc)], args.timeout)
        run([*common, str(folder / f"pixels-{bits}.npy"), "--icc-out", str(decode_icc)],
            args.timeout)
        inspected, decoded = inspect_icc.read_bytes(), decode_icc.read_bytes()
        if not inspected or inspected != decoded:
            raise AssertionError(f"inspection/pixel output ICC differs at {bits} bits")
        profiles.append(inspected)
    # XYB plus embedded ICC negotiates sRGB for integer output, linear sRGB for
    # float output. A default-format header scan would miss this distinction.
    if profiles[0] == profiles[1]:
        raise AssertionError("XYB ICC fixture did not exercise format-dependent profiles")
    return {"passed": True, "profile_bytes": [len(p) for p in profiles]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rs-jxl", type=Path, required=True)
    parser.add_argument("--djxl", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--case", action="append", default=[], help="Select named cases for targeted reruns")
    args = parser.parse_args()
    if args.atol < 0 or args.timeout <= 0:
        parser.error("invalid tolerance or timeout")
    cases = [
        ("preview", "with_preview.jxl", [], []),
        ("layers", "conformance_test_images/cmyk_layers.jxl", ["--no-coalescing", "--output-frames"], ["--no_coalescing", "--output_frames"]),
        ("rgba-layers", "multiple_layers_noise_spline.jxl", ["--no-coalescing", "--output-frames"], ["--no_coalescing", "--output_frames"]),
        ("icc", "with_icc.jxl", ["--color-space", "RGB_D65_SRG_Rel_SRG"], ["--color_space=RGB_D65_SRG_Rel_SRG"]),
        ("profile-inspection", "lossy_with_icc.jxl", [], []),
        ("linear", "3x3_srgb_lossless.jxl", ["--color-space", "RGB_D65_SRG_Rel_Lin"], ["--color_space=RGB_D65_SRG_Rel_Lin"]),
        ("cmyk", "conformance_test_images/cmyk_layers.jxl", ["--color-space", "RGB_D65_SRG_Rel_SRG"], ["--color_space=RGB_D65_SRG_Rel_SRG"]),
        ("pq-display", "hdr_pq_test.jxl", ["--display-nits", "250"], ["--display_nits=250"]),
        ("hlg-display", "hdr_hlg_test.jxl", ["--display-nits", "250"], ["--display_nits=250"]),
        ("partial", "progressive_ac.jxl", ["--allow-partial-files"], ["--allow_partial_files"]),
    ]
    for ratio in (2, 4, 8):
        cases.append((f"progressive-{ratio}", "progressive_ac.jxl",
                      ["--downsampling", str(ratio)], [f"--downsampling={ratio}"]))
    for name in ("extra_channels", "ec_upsampling8_multi_group", "ec_dim_shift_no_upsampling", "upsampled_alpha"):
        cases.append((name, name + ".jxl", [], []))
    selected = set(args.case)
    known = {case[0] for case in cases}
    if selected - known:
        parser.error(f"unknown cases: {sorted(selected - known)}")
    rows = []
    with tempfile.TemporaryDirectory(prefix="rs-jxl-features-") as temporary:
        root = Path(temporary)
        for name, relative, rust_args, official_args in cases:
            if selected and name not in selected:
                continue
            row = {"case": name, "fixture": relative, "atol": args.atol}
            try:
                folder = root / name
                folder.mkdir()
                source = str((args.corpus / relative).resolve())
                if name == "profile-inspection":
                    row.update(profile_inspection(args, source, folder))
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=True), flush=True)
                    continue
                if "layers" in name:
                    row.update(raw_layers(args, name, source, folder))
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=True), flush=True)
                    continue
                if name == "partial":
                    encoded = Path(source).read_bytes()
                    truncated = folder / "truncated.jxl"
                    truncated.write_bytes(encoded[:len(encoded) * 3 // 4])
                    source = str(truncated)
                own, reference = folder / "rust.npy", folder / "reference.npy"
                extra_rust = []
                extra_official = []
                if name == "preview":
                    extra_rust = ["--preview-out", str(folder / "rust-preview.npy")]
                    extra_official = ["--preview_out=" + str(folder / "reference-preview.npy")]
                row["rust_seconds"] = run([str(args.rs_jxl.resolve()), "decode", source, str(own),
                    "--bits", "32", "--allow-implicit-level10", *rust_args, *extra_rust], args.timeout)
                row["official_seconds"] = run([str(args.djxl.resolve()), source, str(reference),
                    *official_args, *extra_official], args.timeout)
                actual, expected = arrays(own), arrays(reference)
                row.update(compare(actual, expected, args.atol))
                if name == "preview":
                    row["preview"] = compare(arrays(folder / "rust-preview.npy"), arrays(folder / "reference-preview.npy"), args.atol)
                    row["passed"] = row["passed"] and row["preview"]["passed"]
            except (OSError, ValueError, RuntimeError, AssertionError, subprocess.TimeoutExpired) as error:
                row.update(passed=False, error=str(error))
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    report = {"passed": sum(row["passed"] for row in rows), "total": len(rows), "cases": rows}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"RESULT: {report['passed']}/{report['total']} cases passed")
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
