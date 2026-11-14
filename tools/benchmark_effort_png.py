"""Run an effort refresh, cache lossless PNG baselines, and render both charts.

PNG timing excludes input decoding, pixel conversion and verification. Pillow
preserves packed/palette inputs; OpenCV handles 16-bit samples without narrowing.
Original PNG size has no measured encode time and is drawn only as a guide.
"""

import argparse
import copy
import csv
import io
import os
from pathlib import Path
import statistics
import tempfile
import time

import benchmark_context_corpus as corpus
import benchmark_effort_refresh as refresh
from benchmark_palette_runs import prepare_png
import compare_cpp_port as port
import plot_effort_refresh as plot


def png_policy():
    import PIL
    import zlib
    cv2, _ = port.pixel_libraries()
    return dict(schema="png-baseline-v1", pillow=PIL.__version__, opencv=cv2.__version__,
                zlib=zlib.ZLIB_RUNTIME_VERSION, level=9, optimize_8bit=True,
                strategy_16bit="default", threads=1,
                timing="memory encode only; excludes decode, conversion, validation and IO",
                runner=port.file_sha256(Path(__file__)))


def encode_png(payload, expected):
    from PIL import Image
    cv2, np = port.pixel_libraries()
    if expected.depth == 16:
        # Canonical RGBA retains transparency keys and invisible RGB. Avoid
        # redundant alpha/color channels only when the input layout permits it.
        color = payload[25]
        _, info = port.png_payload(payload)
        has_alpha = color in (4, 6) or "transparency_key" in info
        if color == 0 and not has_alpha:
            pixels = np.ascontiguousarray(expected.rgba[:, :, 0])
        else:
            order = [2, 1, 0, 3] if has_alpha else [2, 1, 0]
            pixels = np.ascontiguousarray(expected.rgba[:, :, order])
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9,
                  cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT]
        start = time.perf_counter()
        ok, encoded = cv2.imencode(".png", pixels, params)
        seconds = time.perf_counter() - start
        corpus.require(ok, "OpenCV PNG encode failed")
        return encoded.tobytes(), seconds, "OpenCV/libpng level 9 (16-bit)"
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        transparency = image.info.get("transparency")
        image.info.clear()
        options = dict(format="PNG", optimize=True, compress_level=9)
        if transparency is not None:
            options["transparency"] = transparency
        with io.BytesIO() as output:
            start = time.perf_counter()
            image.save(output, **options)
            seconds = time.perf_counter() - start
            return output.getvalue(), seconds, "Pillow optimized level 9 (1-8-bit)"


def validate_png_row(row, identity, index):
    digest = corpus.fingerprint(identity)
    case = identity["inputs"][index]
    corpus.require(row["row_sha256"] == corpus.fingerprint(
        {k: v for k, v in row.items() if k != "row_sha256"}), "PNG row checksum mismatch")
    corpus.require(row["identity"] == digest and row["index"] == index
                   and row["source_sha256"] == case["sha256"]
                   and row["pixels"] == case["width"] * case["height"]
                   and row["verified"] is True, "PNG cache does not match input/policy")
    corpus.require(all(type(row[k]) is int and row[k] > 0
                       for k in ("bytes", "original_bytes", "payload_bytes"))
                   and corpus.positive(row["seconds"]), "Invalid PNG measurements")


def png_baseline(inputs, folder, progress):
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(policy=png_policy(), inputs=inputs)
    digest = corpus.fingerprint(identity)
    identity_path = folder / "identity.json"
    if identity_path.exists():
        corpus.require(corpus.loads(identity_path.read_bytes()) == identity,
                       "PNG cache policy or input list changed")
    else:
        refresh.atomic_json(identity_path, identity)
    rows = []
    for index, case in enumerate(inputs):
        path = folder / f"{index:04d}.json"
        original = Path(case["path"]).read_bytes()
        corpus.require(port.sha256(original) == case["sha256"], "PNG input changed")
        progress.write(dict(status="png", pid=os.getpid(), completed=index,
                            total=len(inputs), active=case["path"], updated_at=corpus.now()))
        if path.exists():
            row = corpus.loads(path.read_bytes())
            validate_png_row(row, identity, index)
        else:
            with tempfile.TemporaryDirectory(prefix="png-check-", dir=folder) as tmp:
                scratch = Path(tmp)
                payload, expected, _ = prepare_png(Path(case["path"]), original, scratch)
                encoded, seconds, backend = encode_png(payload, expected)
                # Packed output needs the existing packed-depth oracle adapter.
                decoded_path = scratch / "verify.png"
                decoded_path.write_bytes(encoded)
                _, actual, _ = prepare_png(decoded_path, encoded, scratch)
                port.compare_pixels(expected, actual)
            row = corpus.seal_row(dict(identity=digest, index=index,
                source_sha256=case["sha256"], pixels=case["width"] * case["height"],
                original_bytes=len(original), payload_bytes=len(payload), bytes=len(encoded),
                seconds=seconds, backend=backend, depth=expected.depth, verified=True,
                includes_hidden_rgb=True, output_sha256=port.sha256(encoded)))
            validate_png_row(row, identity, index)
            refresh.atomic_json(path, row)
        rows.append(row)
        print(f"PNG [{index + 1}/{len(inputs)}] {row['seconds']:.4f}s {row['bytes']}B verified", flush=True)
    sizes = sum(r["bytes"] for r in rows)
    pixels = sum(r["pixels"] for r in rows)
    seconds = sum(r["seconds"] for r in rows)
    point = dict(n=len(rows), summed_bytes=sizes, summed_pixels=pixels,
                 mean_bpp=statistics.mean(8 * r["bytes"] / r["pixels"] for r in rows),
                 pixel_weighted_bpp=8 * sizes / pixels, native_seconds=seconds,
                 mpix_per_second=pixels / 1e6 / seconds)
    original = dict(summed_bytes=sum(r["original_bytes"] for r in rows),
                    mean_bpp=statistics.mean(8 * r["original_bytes"] / r["pixels"] for r in rows),
                    payload_bytes=sum(r["payload_bytes"] for r in rows))
    report = dict(identity=identity, fingerprint=digest, rows=rows, point=point,
                  original=original, status="complete", updated_at=corpus.now())
    refresh.atomic_json(folder / "summary.json", report)
    return report


def comparison_series(current, previous):
    series = copy.deepcopy(refresh.validate_report(current))
    old = refresh.validate_report(previous)
    a, b = current["identity"], previous["identity"]
    corpus.require(a["inputs"] == b["inputs"]
                   and a["previous_fingerprint"] == b["previous_fingerprint"],
                   "Comparison reports use different cohorts/baselines")
    for key in ("threads", "repeats", "efforts"):
        corpus.require(a["config"][key] == b["config"][key], "Comparison settings differ")
    series["previous"] = copy.deepcopy(old["new"])
    series["previous"]["label"] = previous["identity"]["label"] + " (cached)"
    return series


def draw(current, series, png, outdir):
    import matplotlib.pyplot as plt
    corpus.require(current["identity"]["inputs"] == png["identity"]["inputs"],
                   "PNG/JXL cohorts differ")
    for metric, filename in (("mean-bpp", "bpp-time.png"), ("total-mib", "size-time.png")):
        key, divisor, _ = plot.METRICS[metric]
        fig = plot.draw_plot(current, series, metric)
        try:
            ax = fig.axes[0]
            point = png["point"]
            ax.plot([point["native_seconds"]], [point[key] / divisor],
                    marker="D", ms=7, color="#805a9d", linestyle="none",
                    label="PNG level 9 (1 thread)\nPillow 8-bit / OpenCV 16-bit")
            ax.axhline(png["original"][key] / divisor, color="#888888", lw=.8, ls=":",
                       label="Original PNG size\n(encode time unknown)")
            ax.legend(loc="upper left", bbox_to_anchor=(1.025, 1), frameon=False,
                      fontsize=9, labelspacing=1.2)
            fig.texts[-1].set_text(
                ("Arithmetic mean per-image bpp; equal image weights." if metric == "mean-bpp"
                 else "Total compressed bytes / 1,048,576 (MiB).") + " No error bars.\n"
                "PNG: level 9 lossless memory encode; decoding/conversion excluded; not an optimal PNG search.\n"
                "Historical JXL baselines are cached, non-simultaneous. All timing excludes file I/O.")
            fig.savefig(outdir / filename, dpi=200)
            fig.savefig((outdir / filename).with_suffix(".svg"))
        finally:
            plt.close(fig)
    fields = ["series", "effort", "n", "mean_bpp", "summed_bytes", "pixel_weighted_bpp",
              "summed_pixels", "native_seconds", "mpix_per_second", "cached", "threads"]
    with (outdir / "points.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for name, entry in series.items():
            for point in entry["points"]:
                writer.writerow(dict(point, series=name, cached=name != "new",
                                     threads=current["identity"]["config"]["threads"]))
        writer.writerow(dict(png["point"], series="png_level9", effort="", cached=False, threads=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "comparison", "encoder", "decoder", "outdir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--label", default="rs_jxl current")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    progress = refresh.ProgressWriter(args.outdir / "progress.json")
    with corpus.run_lock(args.outdir):
        try:
            progress.write(dict(status="jxl", pid=os.getpid(), updated_at=corpus.now(),
                                details=str(args.outdir / "jxl/progress.json")))
            arguments = ["--previous", str(args.previous), "--encoder", str(args.encoder),
                         "--decoder", str(args.decoder), "--outdir", str(args.outdir / "jxl"),
                         "--label", args.label]
            if (args.outdir / "jxl/identity.json").exists():
                arguments.append("--resume")
            current = refresh.benchmark(refresh.arguments(arguments))
            corpus.require(current["status"] == "complete", "JXL matrix incomplete; see jxl/summary.json")
            previous = corpus.loads(args.comparison.read_bytes())
            series = comparison_series(current, previous)
            png = png_baseline(current["identity"]["inputs"], args.outdir / "png", progress)
            progress.write(dict(status="plotting", pid=os.getpid(), updated_at=corpus.now()))
            draw(current, series, png, args.outdir)
            refresh.atomic_json(args.outdir / "summary.json", dict(status="complete",
                new_report=corpus.hashed(args.outdir / "jxl/summary.json"),
                previous_report=corpus.hashed(args.comparison),
                png_report=corpus.hashed(args.outdir / "png/summary.json"),
                series=series, png=png["point"], original_png=png["original"], updated_at=corpus.now()))
            progress.write(dict(status="complete", pid=os.getpid(), updated_at=corpus.now(),
                                verified_jxl=current["verified_trials"], verified_png=len(png["rows"])))
        except BaseException as error:
            progress.write(dict(status="failed", pid=os.getpid(), updated_at=corpus.now(),
                                error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    main()
