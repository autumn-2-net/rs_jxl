import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import select_ma_corpus as select


class SelectionTests(unittest.TestCase):
    def test_exact_balanced_quotas(self):
        self.assertEqual(select.quotas(500), select.WEIGHTS)
        for count in [25, 101, 499, 501]:
            self.assertEqual(sum(select.quotas(count).values()), count)
        with self.assertRaises(ValueError):
            select.quotas(2)

    def test_classification_and_subgroups(self):
        cases = [('DIV2K', 'DIV2K/a.png', 'photo_highres'),
                 ('urban100', 'urban100/a.png', 'urban'),
                 ('local-screenshots', 'local-screenshots/a.png', 'document_web'),
                 ('jxl-testdata', 'jxl-testdata/images/screenshot_web/a.png', 'document_web'),
                 ('jxl-testdata', 'jxl-testdata/images/textures_pk/a.png', 'texture'),
                 ('jxl-testdata', 'jxl-testdata/images/icon_512/a.png', 'alpha_icon')]
        for dataset, relative, expected in cases:
            self.assertEqual(select.classify(dict(dataset=dataset, relative=relative)), expected)

    def test_deterministic_exact_selection_known_and_duplicate_handling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for group in ['A', 'B']:
                for i in range(5):
                    path = root / f'{group}{i}.png'
                    path.write_bytes(select.PNG_SIGNATURE + f'{group}{i}'.encode())
                    rows.append(dict(path=str(path), dataset=group, relative=path.name))
            known = root / 'known.json'
            known.write_text(json.dumps(dict(cases=[dict(source=rows[0]['path'])])))
            catalog = root / 'catalog.json'
            catalog.write_text(json.dumps(dict(inputs=rows)))
            framing = dict(width=2, height=3, bit_depth=8, color_type=6)
            with patch.object(select, 'WEIGHTS', {'A': 2, 'B': 2}), \
                 patch.object(select, 'classify', side_effect=lambda r: r['dataset']), \
                 patch.object(select, 'png_framing', return_value=(b'', framing)):
                first = select.select(catalog, known, 4, 17)
                second = select.select(catalog, known, 4, 17)
                self.assertEqual(first, second)
                self.assertEqual(first['known_cases'], 1)
                self.assertEqual(len({r['sha256'] for r in first['cases']}), 4)
                self.assertEqual([r['stratum'] for r in first['cases']].count('A'), 2)
                for row in rows[5:]:
                    Path(row['path']).write_bytes(Path(rows[0]['path']).read_bytes())
                with self.assertRaisesRegex(ValueError, 'Not enough'):
                    select.select(catalog, known, 4, 17)


if __name__ == '__main__':
    unittest.main()
