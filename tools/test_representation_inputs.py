import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import representation_inputs as inputs


class RepresentationInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.first, self.second = self.root / "first.png", self.root / "second.png"
        # Deliberately not images: preflight must only read bytes, never decode.
        self.first.write_bytes(b"original first bytes")
        self.second.write_bytes(b"original second bytes")
        self.args = SimpleNamespace(case=[], manifest=self.manifest)

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_manifest(self, cases):
        self.manifest.write_text(json.dumps(dict(cases=cases)), encoding="utf-8")

    def test_relative_absolute_and_uppercase_hashes_preserve_order(self):
        self.write_manifest([
            dict(original=self.second.name, sha256=self.sha(self.second).upper()),
            dict(original=str(self.first.resolve()), sha256=self.sha(self.first))])
        with warnings.catch_warnings(record=True) as notes:
            warnings.simplefilter("always")
            result = inputs.load_verified_cases(self.args)
        self.assertEqual(result, [self.second.resolve(), self.first.resolve()])
        self.assertEqual(notes, [])
        self.assertTrue(all(isinstance(path, Path) for path in result))

    def test_changed_original_fails_before_returning_cases(self):
        self.write_manifest([dict(original=self.first.name, sha256=self.sha(self.first))])
        self.first.write_bytes(b"changed since freezing")
        with self.assertRaisesRegex(ValueError, "Frozen manifest SHA256 mismatch") as error:
            inputs.load_verified_cases(self.args)
        self.assertIn(str(self.first.resolve()), str(error.exception))
        self.assertIn("expected", str(error.exception))
        self.assertIn("actual", str(error.exception))

    def test_legacy_missing_hash_is_accepted_with_note(self):
        self.write_manifest([dict(original=self.first.name), dict(original=self.second.name)])
        with self.assertWarnsRegex(UserWarning, "2 legacy case.*not frozen-hash verified"):
            result = inputs.load_verified_cases(self.args)
        self.assertEqual(result, [self.first.resolve(), self.second.resolve()])

    def test_mixed_manifest_still_enforces_declared_hash(self):
        self.write_manifest([dict(original=self.first.name),
                             dict(original=self.second.name, sha256=self.sha(self.second))])
        with self.assertWarnsRegex(UserWarning, "1 legacy case"):
            self.assertEqual(inputs.load_verified_cases(self.args),
                             [self.first.resolve(), self.second.resolve()])
        self.second.write_bytes(b"replacement")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            inputs.load_verified_cases(self.args)

    def test_malformed_present_hash_is_not_a_legacy_opt_out(self):
        for value in (None, "", "0" * 63, "g" * 64, "0" * 64 + "\n", 123, True):
            with self.subTest(value=value):
                self.write_manifest([dict(original=self.first.name, sha256=value)])
                with patch.object(inputs, "file_sha256") as hash_file:
                    with self.assertRaisesRegex(ValueError, "Invalid manifest sha256"):
                        inputs.load_verified_cases(self.args)
                    hash_file.assert_not_called()

    def test_explicit_case_inputs_are_preserved_without_hash_requirement(self):
        self.args.manifest = None
        self.args.case = [self.second, self.first]
        with patch.object(inputs, "file_sha256") as hash_file:
            with warnings.catch_warnings(record=True) as notes:
                warnings.simplefilter("always")
                result = inputs.load_verified_cases(self.args)
            hash_file.assert_not_called()
        self.assertEqual(result, [self.second.resolve(), self.first.resolve()])
        self.assertEqual(notes, [])

    def test_explicit_cases_precede_verified_manifest_cases(self):
        self.args.case = [self.second]
        self.write_manifest([dict(original=self.first.name, sha256=self.sha(self.first))])
        self.assertEqual(inputs.load_verified_cases(self.args),
                         [self.second.resolve(), self.first.resolve()])

    def test_existing_duplicate_and_missing_file_errors_are_preserved(self):
        row = dict(original=self.first.name, sha256=self.sha(self.first))
        self.write_manifest([row, row])
        with self.assertRaisesRegex(ValueError, "distinct"):
            inputs.load_verified_cases(self.args)
        self.write_manifest([dict(original="missing.png", sha256="0" * 64)])
        with self.assertRaises(FileNotFoundError):
            inputs.load_verified_cases(self.args)

    def test_equal_content_at_distinct_paths_is_not_silently_deduplicated(self):
        self.second.write_bytes(self.first.read_bytes())
        self.write_manifest([dict(original=str(p), sha256=self.sha(p))
                             for p in (self.first, self.second)])
        self.assertEqual(inputs.load_verified_cases(self.args),
                         [self.first.resolve(), self.second.resolve()])

    def test_manifest_change_during_wrapped_load_is_rejected(self):
        self.write_manifest([dict(original=self.first.name, sha256=self.sha(self.first))])

        def changed_load(args):
            self.write_manifest([dict(original=self.second.name, sha256=self.sha(self.second))])
            return [self.second.resolve()]

        with patch.object(inputs, "load_cases", side_effect=changed_load):
            with self.assertRaisesRegex(ValueError, "Manifest changed during input loading"):
                inputs.load_verified_cases(self.args)

    def test_wrapped_path_list_must_match_manifest(self):
        self.write_manifest([dict(original=self.first.name, sha256=self.sha(self.first))])
        with patch.object(inputs, "load_cases", return_value=[self.second.resolve()]):
            with self.assertRaisesRegex(ValueError, "Manifest paths changed"):
                inputs.load_verified_cases(self.args)


if __name__ == "__main__":
    unittest.main()
