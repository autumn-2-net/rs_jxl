import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import prepare_representation_corpus as corpus


def png_bytes(value=0, color=6, depth=8, width=4, height=4, raw=None, extra=()):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    if raw is None:
        row = bytes([value]) * ((width * channels * depth + 7) // 8)
        raw = (b"\0" + row) * height
    header = struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
            b"".join(chunk(k, v) for k, v in extra) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.population = self.root / "population.json"
        self.known = self.root / "known.json"
        rows = []
        for index in range(10):
            path = self.root / f"original-{index}.png"
            data = png_bytes(index)
            path.write_bytes(data)
            rows.append(dict(original=str(path), sha256=hashlib.sha256(data).hexdigest(),
                             stratum="document_web", source_group=f"collection-{index // 2}"))
        self.rows = rows
        self.population.write_text(json.dumps(dict(cases=rows)), encoding="utf-8")
        self.known.write_text(json.dumps(dict(cases=[dict(original=rows[0]["original"])])), encoding="utf-8")

    def prepare(self, **kwargs):
        return corpus.prepare(self.population, self.known, quotas=dict(document_web=4),
                              diagnostic_count=2, **kwargs)

    def test_repeatable_group_separated_originals_only(self):
        first = self.prepare()
        self.assertEqual(first, self.prepare())
        cal, hold = first["calibration"], first["holdout"]
        self.assertEqual((len(cal), len(hold), len(first["patch24"])), (2, 2, 2))
        self.assertFalse({r["source_group"] for r in cal} & {r["source_group"] for r in hold})
        self.assertFalse({r["sha256"] for r in cal} & {r["sha256"] for r in hold})
        self.assertFalse({r["sha256"] for r in first["study"]} & {self.rows[0]["sha256"]})
        self.assertTrue({r["sha256"] for r in first["patch24"]} <= {r["sha256"] for r in hold})
        self.assertNotIn("input_bytes", first["study"][0])
        self.assertEqual(first["protocol"]["exclusion_counts"], dict(known12_sha256=1))

    def test_dedup_sha_and_path_before_choice(self):
        rows = copy.deepcopy(self.rows)
        duplicate = self.root / "duplicate.png"
        duplicate.write_bytes(Path(rows[1]["original"]).read_bytes())
        rows.extend([dict(rows[1], original=str(duplicate)), rows[2]])
        self.population.write_text(json.dumps(dict(cases=rows)), encoding="utf-8")
        result = self.prepare()
        self.assertEqual(result["protocol"]["exclusion_counts"]["duplicate_sha256"], 1)
        self.assertEqual(result["protocol"]["exclusion_counts"]["duplicate_path"], 1)

    def test_original_validation_sha_crc_animation(self):
        row = self.rows[1]
        path = Path(row["original"])
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            corpus.checked_original(row, self.population)
        with self.assertRaisesRegex(ValueError, "CRC mismatch"):
            corpus.checked_original(dict(original=str(path)), self.population)
        path.write_bytes(png_bytes(1, extra=((b"acTL", struct.pack(">II", 1, 0)),)))
        with self.assertRaisesRegex(ValueError, "APNG"):
            corpus.checked_original(dict(original=str(path)), self.population)

    def test_insufficient_source_groups_fails(self):
        self.population.write_text(json.dumps(dict(cases=[dict(r, source_group="one") for r in self.rows])), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "whole-source split"):
            self.prepare()

    def test_frozen_outputs_never_overwrite(self):
        result = self.prepare(references=[self.population])
        output = self.root / "frozen"
        corpus.write_manifests(result, output)
        self.assertEqual(corpus.read_json(output / "study.json")["count"], 4)
        self.assertEqual(result["protocol"]["overlaps"][str(self.population.resolve())]["main"], 4)
        with self.assertRaises(FileExistsError):
            corpus.write_manifests(result, output)

    def test_groups_are_semantic_not_filehash(self):
        group = corpus.source_group
        self.assertEqual(group(dict(dataset="manga109", relative="manga109/LoveHina_vol01.png")),
                         group(dict(dataset="manga109", relative="manga109/LoveHina_vol02.png")))
        self.assertEqual(group(dict(dataset="jxl-testdata", relative="jxl-testdata/images/icon_64/a.png")),
                         group(dict(dataset="jxl-testdata", relative="jxl-testdata/images/icon_512/a.png")))
        self.assertEqual(group(dict(relative="diverse/qoi/screenshot_web/a.png")),
                         group(dict(relative="jxl-testdata/images/screenshot_web/b.png")))
        self.assertNotEqual(group(dict(dataset="DIV2K", relative="DIV2K/0001.png")),
                            group(dict(dataset="BSDS100", relative="BSDS100/0001.png")))

    def test_invalid_quotas_rejected(self):
        with self.assertRaisesRegex(ValueError, "even quotas"):
            corpus.prepare(self.population, self.known, quotas=dict(document_web=3))


if __name__ == "__main__":
    unittest.main()
