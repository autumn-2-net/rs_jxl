"""Benchmark plumbing tests; no DLL loading or real codec/benchmark execution."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import benchmark_multiframe_native as bench


class FakeCodec:
    def __init__(self):
        self.calls = []
        self.outputs = {}
        self.inputs = {}

    def decode(self, data, **options):
        self.calls.append(("decode", data, options))
        if data in self.inputs:
            return self.inputs[data]
        return self.outputs[data]

    def encode_multiframe(self, frames, **options):
        self.calls.append(("multiframe", frames, options))
        encoded = b"multiframe"
        self.outputs[encoded] = (frames, {"frames": [{"duration": 1, "extra_channels": []} for _ in frames]})
        reports = [{"candidates": [{"trial_pixels": frame.shape[0] * frame.shape[1],
                                    "elapsed_seconds": 0.25}],
                    "pruned": [], "analysis_seconds": 0.125} for frame in frames]
        return encoded, {"frames": reports}

    def encode(self, frame, **options):
        self.calls.append(("encode", frame, options))
        encoded = b"still" + bytes([len(self.outputs)])
        self.outputs[encoded] = ([frame], {"frames": [{"extra_channels": []}]})
        return encoded

    def transcode_jpeg(self, raw, **options):
        self.calls.append(("jpeg", raw, options))
        return b"JXL" + raw

    def reconstruct_jpeg(self, data, **options):
        self.calls.append(("reconstruct", data, options))
        return data[3:]


def dataset():
    settings = {"layout": "hwc", "bits_per_sample": 16, "color_encoding": "linear_srgb"}
    sources = [{"path": Path(f"frame{i}.png"), "raw": bytes([i]),
                "pixels": np.full((3, 5, 3), 1000 + i, dtype=np.uint16),
                "info": {"extra_channels": []}, "jpeg": False} for i in range(3)]
    return {"width": 5, "height": 3, "settings": settings, "sources": sources}


def args(legacy=False):
    return SimpleNamespace(effort=5, threads=4, reference="auto", legacy=legacy)


class BenchmarkTests(unittest.TestCase):
    def test_multiscale_mode_is_opt_in_and_unknown_input_time_is_not_zero(self):
        parsed = bench.parse_args(["one", "--modes", "auto-multiscale", "--out", "report.json"])
        self.assertEqual(parsed.modes, ["auto-multiscale"])
        self.assertNotIn("auto-multiscale", bench.DEFAULT_MODES)
        codec = FakeCodec()
        bench.measure(codec, dataset(), "auto-multiscale", args())
        self.assertEqual(codec.calls[0][2]["auto_options"]["extra_tile_sizes"], [8, 16, 64])
        report = {"status": "passed", "native": "fixture", "effort": 9, "threads": 4,
                  "reference": "auto", "legacy": False,
                  "datasets": [{"folder": "subset", "width": 5, "height": 3,
                                "frames": 2, "input_seconds": None, "results": []}]}
        self.assertIn("input read/decode: not measured", bench.markdown(report))

    def test_cli_folders_defaults_and_legacy_modes(self):
        parsed = bench.parse_args(["one", "--folders", "two", "three", "--folders", "four",
                                   "--out", "report.md"])
        self.assertEqual(parsed.folders, list(map(Path, ["one", "two", "three", "four"])))
        self.assertEqual((parsed.effort, parsed.threads), (9, 4))
        self.assertEqual(parsed.modes, list(bench.DEFAULT_MODES))
        parsed = bench.parse_args(["one", "--legacy", "--modes", "crop,auto", "--out", "report.json"])
        self.assertEqual(parsed.modes, ["crop", "auto-legacy"])

    def test_retained_reference_is_explicit_and_not_an_old_library_mode(self):
        parsed = bench.parse_args(["one", "--modes", "auto-retained", "--out", "report.json"])
        self.assertEqual(parsed.modes, ["auto-retained"])
        for extra in (["--legacy"], ["--reference", "first"]):
            with self.assertRaises(SystemExit):
                bench.parse_args(["one", "--modes", "auto-retained", "--out", "report.json", *extra])
        codec = FakeCodec()
        row = bench.measure(codec, dataset(), "auto-retained", args())
        self.assertEqual(row["status"], "passed")
        self.assertTrue(codec.calls[0][2]["auto_options"]["retained_reference"])

    def test_lookahead_mode_forwards_native_option(self):
        parsed = bench.parse_args(["one", "--modes", "auto-lookahead", "--out", "report.json"])
        self.assertEqual(parsed.modes, ["auto-lookahead"])
        self.assertNotIn("auto-lookahead", bench.DEFAULT_MODES)
        codec = FakeCodec()
        bench.measure(codec, dataset(), "auto-lookahead", args())
        self.assertTrue(codec.calls[0][2]["auto_options"]["reference_lookahead"])
        for extra in (["--legacy"], ["--reference", "previous"]):
            with self.assertRaises(SystemExit):
                bench.parse_args(["one", "--modes", "auto-lookahead", "--out", "report.json", *extra])

    def test_natural_filename_order(self):
        paths = list(map(Path, ["frame10.png", "frame2.png", "frame1.png"]))
        self.assertEqual([p.name for p in sorted(paths, key=bench.natural_key)],
                         ["frame1.png", "frame2.png", "frame10.png"])

    def test_actual_native_structured_color_metadata(self):
        color = {"color_space": "rgb", "kind": "structured", "primaries": "Some(Srgb)",
                 "rendering_intent": "Perceptual", "transfer_function": "srgb", "white_point": "D65"}
        info = {"bits_per_sample": 8, "orientation": 1, "extra_channels": [],
                "color_encoding": color}
        for primaries in ("Some(Srgb)", "Srgb", "srgb"):
            for transfer, expected in (("srgb", "srgb"), ("linear", "linear_srgb")):
                with self.subTest(primaries=primaries, transfer=transfer):
                    current = {**info, "color_encoding": {**color, "primaries": primaries,
                                                         "transfer_function": transfer}}
                    self.assertEqual(bench.pixel_settings(current),
                                     {"layout": "hwc", "bits_per_sample": 8, "color_encoding": expected})
        for unsupported in ({"primaries": "Some(P3)"}, {"primaries": "None"}, {"white_point": "D50"}):
            with self.subTest(unsupported=unsupported), self.assertRaises(ValueError):
                bench.pixel_settings({**info, "color_encoding": {**color, **unsupported}})

    def test_mixed_dimensions_are_rejected_without_resizing(self):
        codec = FakeCodec()
        metadata = {"bits_per_sample": 8, "orientation": 1, "extra_channels": [],
                    "color_encoding": {"name": "srgb"}}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for index, shape in enumerate(((7, 10, 3), (6, 9, 3))):
                raw = b"PNG input" + bytes([index])
                (folder / f"frame{index}.png").write_bytes(raw)
                codec.inputs[raw] = (np.zeros(shape, dtype=np.uint8), metadata)
            with self.assertRaisesRegex(ValueError, "geometry, sample type or color metadata differs"):
                bench.load_folder(codec, folder, 4)

    def test_only_pixel_arrays_are_encoded_with_identical_metadata(self):
        codec, data = FakeCodec(), dataset()
        for mode in bench.MODES[:-1]:
            row = bench.measure(codec, data, mode, args())
            self.assertEqual(row["status"], "passed")
            self.assertEqual(row["bpp"], row["total_bytes"] * 8 / (5 * 3 * 3))
        for operation, value, options in codec.calls:
            if operation in ("multiframe", "encode"):
                for key, expected in data["settings"].items():
                    self.assertEqual(options[key], expected)
                self.assertEqual((options["effort"], options["threads"]), (5, 4))
                if operation == "encode":
                    self.assertIsInstance(value, np.ndarray)
                else:
                    for supplied, source in zip(value, data["sources"]):
                        self.assertIs(supplied, source["pixels"])
        multiframe = [call for call in codec.calls if call[0] == "multiframe"]
        self.assertNotIn("auto_options", multiframe[0][2])
        for call, mode in zip(multiframe[1:], bench.AUTO_OPTIONS):
            self.assertEqual(call[2]["auto_options"], bench.AUTO_OPTIONS[mode])

    def test_legacy_call_omits_auto_options_entirely(self):
        codec = FakeCodec()
        bench.measure(codec, dataset(), "auto-legacy", args(legacy=True))
        self.assertNotIn("auto_options", codec.calls[0][2])

    def test_source_decode_occurs_only_during_loading(self):
        codec = FakeCodec()
        metadata = {"bits_per_sample": 8, "orientation": 1, "extra_channels": [],
                    "color_encoding": {"name": "srgb"}}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for i in range(3):
                raw = b"PNG input" + bytes([i])
                (folder / f"frame{i}.png").write_bytes(raw)
                codec.inputs[raw] = (np.full((3, 5, 3), i, dtype=np.uint8), metadata)
            loaded = bench.load_folder(codec, folder, 4)
            self.assertEqual(len(codec.calls), 3)
            for mode in bench.MODES[:-1]:
                bench.measure(codec, loaded, mode, args())
            self.assertEqual(sum(call[0] == "decode" and call[1] in codec.inputs for call in codec.calls), 3)

    def test_statistics_include_pruned_work_and_missing_is_not_zero(self):
        info = {"frames": [{"candidates": [{"trial_pixels": 15, "elapsed_seconds": 0.5}],
                            "pruned": [{"trial_pixels": 7, "elapsed_seconds": 0.25}],
                            "analysis_seconds": 0.125}]}
        stats = bench.search_statistics(info, require_new=True)
        self.assertEqual(stats, {"candidates_completed": 1, "candidates_pruned": 1,
                                 "trial_pixels": 22, "candidate_seconds": 0.75,
                                 "analysis_seconds": 0.125, "lookahead_comparisons": None,
                                 "reused_payloads": None, "reused_payload_pixels": None})
        old = {"frames": [{"candidates": [{"encoded_bytes": 123}]}]}
        self.assertIsNone(bench.search_statistics(old)["candidates_pruned"])
        self.assertIsNone(bench.search_statistics(old)["analysis_seconds"])
        with self.assertRaises(ValueError):
            bench.search_statistics(old, require_new=True)

    def test_hidden_rgb_float_words_and_extra_planes_are_exact(self):
        hidden = np.array([[[1, 2, 3, 0]]], dtype=np.uint8)
        changed = hidden.copy()
        changed[0, 0, 0] = 7
        with self.assertRaises(ValueError):
            bench.exact_array(changed, hidden, "hidden RGB")
        words = np.array([0x80000000, 0x7fc12345], dtype=np.uint32).view(np.float32)
        bench.exact_array(words.copy(), words, "float bits")
        changed = words.copy()
        changed[0] = 0.0
        with self.assertRaises(ValueError):
            bench.exact_array(changed, words, "signed zero")
        source = dataset()["sources"][0]
        extra = {"type": "depth", "bits_per_sample": 16, "exponent_bits_per_sample": 0,
                 "dim_shift": 0, "data": np.full((3, 5, 1), 7, dtype=np.uint16)}
        source["info"]["extra_channels"] = [extra]
        metadata = {"frames": [{"duration": 1, "extra_channels": [deepcopy(extra)]}]}
        bench.verify_pixels([source["pixels"]], metadata, [source], animated=True)
        metadata["frames"][0]["extra_channels"][0]["data"][0, 0, 0] = 9
        with self.assertRaises(ValueError):
            bench.verify_pixels([source["pixels"]], metadata, [source], animated=True)

    def test_jpeg_uses_separate_byte_transcode_sum_and_reconstruction(self):
        codec, data = FakeCodec(), dataset()
        for i in (0, 2):
            data["sources"][i].update(jpeg=True, raw=b"\xff\xd8original" + bytes([i]))
        row = bench.measure(codec, data, "jpeg", args())
        expected = sum(3 + len(source["raw"]) for source in data["sources"] if source["jpeg"])
        self.assertEqual(row["total_bytes"], expected)
        self.assertEqual(row["bpp"], expected * 8 / (5 * 3 * 2))
        self.assertEqual(row["frames"], 2)
        self.assertEqual([call[0] for call in codec.calls], ["jpeg", "jpeg", "reconstruct", "reconstruct"])
        codec.reconstruct_jpeg = lambda *a, **kw: b"wrong bytes"
        with self.assertRaises(ValueError):
            bench.measure(codec, data, "jpeg", args())


if __name__ == "__main__":
    unittest.main()
