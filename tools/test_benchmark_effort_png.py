"""PNG cache/precision and plot checks; no JXL encoder execution."""

import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark_effort_png as bench
from benchmark_palette_runs import prepare_png
import compare_cpp_port as port
from test_plot_effort_refresh import fake_report


class PngTests(unittest.TestCase):
    def fixtures(self):
        from PIL import Image
        cv2, np = port.pixel_libraries()
        rgba = np.arange(19 * 13 * 4, dtype=np.uint8).reshape(13, 19, 4)
        rgba[:, :, 3] = 0
        for mode in ("RGB", "RGBA", "P", "1"):
            image = Image.fromarray(rgba, "RGBA").convert(mode)
            if mode == "P":
                image.info["transparency"] = 0
            output = io.BytesIO()
            image.save(output, format="PNG", bits=4 if mode == "P" else 8)
            yield output.getvalue()
        for channels in (1, 3, 4):
            shape = (13, 19) if channels == 1 else (13, 19, channels)
            pixels = (np.arange(13 * 19 * channels, dtype=np.uint16) * 67).reshape(shape)
            if channels == 4:
                pixels[:, :, 3] = 0
            ok, data = cv2.imencode(".png", pixels)
            self.assertTrue(ok)
            yield data.tobytes()

    def test_roundtrip_layout_depth_and_invisible_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for data in self.fixtures():
                path = folder / "original.png"
                path.write_bytes(data)
                payload, expected, _ = prepare_png(path, data, folder)
                result, seconds, backend = bench.encode_png(payload, expected)
                path = folder / "result.png"
                path.write_bytes(result)
                _, actual, _ = prepare_png(path, result, folder)
                port.compare_pixels(expected, actual)
                self.assertGreater(seconds, 0)
                self.assertEqual("16-bit" in backend, expected.depth == 16)

    def test_cache_reuses_verified_rows_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "input.png"
            data = next(self.fixtures())
            path.write_bytes(data)
            inputs = [dict(path=str(path), sha256=port.sha256(data), width=19, height=13)]
            progress = bench.refresh.ProgressWriter(folder / "progress.json")
            report = bench.png_baseline(inputs, folder / "cache", progress)
            with patch.object(bench, "encode_png", side_effect=AssertionError("encoded twice")):
                cached = bench.png_baseline(inputs, folder / "cache", progress)
            self.assertEqual(report["rows"], cached["rows"])
            row = copy.deepcopy(report["rows"][0])
            row["bytes"] += 1
            with self.assertRaisesRegex(ValueError, "checksum"):
                bench.validate_png_row(row, report["identity"], 0)
            path.write_bytes(data + b"corrupt")
            with self.assertRaisesRegex(ValueError, "input changed"):
                bench.png_baseline(inputs, folder / "cache", progress)

    def test_plots_include_png_and_do_not_mutate_measurements(self):
        report = fake_report()
        original = copy.deepcopy(report)
        png = dict(identity=dict(inputs=report["identity"]["inputs"]),
                   point=dict(report["series"]["new"]["points"][0]),
                   original=dict(summed_bytes=9e6, mean_bpp=9))
        with tempfile.TemporaryDirectory() as tmp:
            bench.draw(report, report["series"], png, Path(tmp))
            self.assertGreater((Path(tmp) / "bpp-time.png").stat().st_size, 1000)
            self.assertGreater((Path(tmp) / "size-time.png").stat().st_size, 1000)
            self.assertIn("png_level9", (Path(tmp) / "points.csv").read_text())
        self.assertEqual(report, original)


if __name__ == "__main__":
    unittest.main()
