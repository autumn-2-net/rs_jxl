"""Sampler contracts using temporary original files, without codecs or image libraries."""
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sample_palette_manifest as sampler


class Sampling(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = self.root / "source.json"
        self.out = self.root / "nested" / "sample.json"
        cases = []
        for stratum, count in (("z", 5), ("a", 3), ("m", 2)):
            for index in range(count):
                path = self.root / f"{stratum}-{index}.png"
                data = path.name.encode()
                path.write_bytes(data)
                cases.append(dict(stratum=stratum, original=path.name,
                                  sha256=hashlib.sha256(data).hexdigest(), width=17, height=23,
                                  extra=dict(dataset="synthetic", tag=index)))
        self.source = dict(cases=cases, count=10, prior_results="must-not-be-opened.json")
        self.write_source()

    def write_source(self):
        self.manifest.write_text(json.dumps(self.source), encoding="utf-8")

    def sample(self, count=4, seed=20261110, out=None):
        return sampler.sample_manifest(self.manifest, count, seed, out or self.out)

    def test_largest_remainder_and_generic_counts(self):
        self.assertEqual(sampler.largest_remainder({"c": 1, "b": 1, "a": 1}, 2),
                         dict(a=1, b=1, c=0))
        sizes = dict(photo_highres=80, photo_other=60, urban=40, manga=40, mixed_local=40,
                     document_web=60, game_ui=40, alpha_icon=50, texture=40, format_mixed=50)
        expected = dict(photo_highres=32, photo_other=24, urban=16, manga=16, mixed_local=16,
                        document_web=24, game_ui=16, alpha_icon=20, texture=16, format_mixed=20)
        self.assertEqual(sampler.largest_remainder(sizes, 200), expected)
        for count in range(1, 501):
            quotas = sampler.largest_remainder(sizes, count)
            self.assertEqual(sum(quotas.values()), count)
            self.assertTrue(all(0 <= quota <= sizes[name] for name, quota in quotas.items()))
        for count in (0, -1, 501, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                sampler.largest_remainder(sizes, count)

    def test_determinism_metadata_and_originals_only(self):
        opened = []
        real_open = Path.open

        def track(path, *args, **kwargs):
            opened.append(path.resolve())
            return real_open(path, *args, **kwargs)

        with patch.object(Path, "open", track):
            result = self.sample()
        selected = result["cases"]
        self.assertEqual(set(opened), {self.manifest, self.out, *(Path(c["original"]) for c in selected)})
        self.assertEqual(Counter(c["stratum"] for c in selected), result["quotas"])
        self.assertEqual(result["verified_originals"], 4)
        self.assertEqual(result["source_manifest"]["sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        self.assertEqual(result["source_manifest"]["metadata"]["prior_results"], "must-not-be-opened.json")
        self.assertEqual(json.loads(self.out.read_text()), result)
        originals = {str((self.root / c["original"]).resolve()): c for c in self.source["cases"]}
        for case in selected:
            expected = dict(originals[case["original"]], original=case["original"])
            self.assertEqual(case, expected)
        self.assertEqual(len({c["original"] for c in selected}), 4)
        self.assertEqual(len({c["sha256"] for c in selected}), 4)
        again = self.sample(out=self.root / "again.json")
        self.assertEqual(result, again)
        self.source["cases"].reverse()
        self.write_source()
        self.assertEqual(selected, self.sample(out=self.root / "reordered.json")["cases"])
        changed = self.sample(seed=17, out=self.root / "other-seed.json")
        self.assertNotEqual(selected, changed["cases"])

    def test_duplicate_resolved_paths_or_hashes_rejected(self):
        baseline = deepcopy(self.source)
        for duplicate_hash in (False, True):
            self.source = deepcopy(baseline)
            first, second = self.source["cases"][:2]
            if duplicate_hash:
                second["sha256"] = first["sha256"].upper()
            else:
                second["original"] = str(self.root / first["original"])
            self.write_source()
            with self.subTest(hash=duplicate_hash), self.assertRaisesRegex(ValueError, "Duplicate"):
                self.sample()
            self.assertFalse(self.out.exists())

    def test_verification_errors_never_publish_output(self):
        path = self.root / self.source["cases"][0]["original"]
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.sample(count=10)
        self.assertFalse(self.out.exists())
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.sample(count=10)
        self.assertFalse(self.out.exists())

    def test_cli_and_no_overwrite(self):
        argv = ["--manifest", str(self.manifest), "--count", "4", "--seed", "20261110",
                "--out", str(self.out)]
        with redirect_stdout(StringIO()):
            self.assertEqual(sampler.main(argv), 0)
        before = self.out.read_bytes()
        with redirect_stderr(StringIO()):
            self.assertEqual(sampler.main(argv), 1)
        self.assertEqual(self.out.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            self.sample(out=self.manifest)


if __name__ == "__main__":
    unittest.main()
