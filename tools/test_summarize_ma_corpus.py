import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import summarize_ma_corpus as summary


def case(index, known=False, stratum='photos'):
    return dict(original=f'C:/fixtures/{index}.png', stratum=stratum,
                sha256=f'{index:064x}', known_case=known)


def trial(cases, index, variant='baseline', size=100, seconds=1, pixels=100):
    c = cases[index]
    return dict(index=index, source=c['original'], stratum=c['stratum'], variant=variant,
                effort=9, status='passed', pixels=pixels, width=pixels, height=1, depth=8,
                original_sha256=c['sha256'], payload_sha256='a' * 64, error=None,
                known_case=c['known_case'], pixels_verified=True,
                receipt_validation={'status': 'passed'}, encoder={'returncode': 0}, decoder={'returncode': 0},
                receipts={'verification': 'fixture'},
                measurement=dict(bytes=size, seconds=seconds, bpp=8 * size / pixels,
                                 mpix_per_second=pixels / 1e6 / seconds))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.cases = [case(0, True), case(1), case(2, stratum='graphics')]

    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trials.jsonl'
            path.write_text(text, encoding='utf-8')
            return summary.load_trials(path, self.cases)

    def test_weighted_totals_not_mean_rates_or_ratios(self):
        rows = [trial(self.cases, 0, size=100, seconds=1, pixels=100),
                trial(self.cases, 1, size=900, seconds=9, pixels=900),
                trial(self.cases, 0, 'fast-basic', size=200, seconds=2, pixels=100),
                trial(self.cases, 1, 'fast-basic', size=450, seconds=3, pixels=900)]
        data, regressions = summary.summarize({'cases': self.cases}, rows)
        g = data['overall']['fast-basic']
        self.assertEqual(g['baseline_paired_denominator'], 2)
        self.assertEqual(g['baseline']['bytes'], 1000)
        self.assertEqual(g['trial']['bytes'], 650)
        self.assertEqual(g['trial']['pixels'], 1000)
        self.assertEqual(g['trial']['bpp'], 5.2)
        self.assertEqual(g['trial']['mpix_per_second'], .0002)
        self.assertEqual(g['total_bytes_ratio'], .65)
        self.assertEqual(g['total_seconds_ratio'], .5)
        self.assertEqual(g['median_size_ratio'], 1.25)
        self.assertEqual(g['median_size_delta_bytes'], -175)
        self.assertEqual(g['median_absolute_size_delta_bytes'], 275)
        self.assertAlmostEqual(g['median_time_ratio'], 7 / 6)
        self.assertEqual(g['min_size_change_pct'], -50)
        self.assertEqual(g['max_size_change_pct'], 100)
        self.assertEqual(g['wins'], 1)
        self.assertEqual(g['larger_over_0_25_pct'], 1)
        self.assertEqual(g['larger_over_5_pct'], 1)
        self.assertEqual(g['both_bigger_and_slower'], 1)
        self.assertEqual(data['strata']['photos']['fast-basic']['paired'], 2)
        self.assertEqual(data['strata']['graphics']['fast-basic']['paired'], 0)
        worst = regressions['worst20']['largest_expansions'][0]
        self.assertEqual(worst['relative_basename'], '0.png')
        self.assertEqual(worst['stratum'], 'photos')
        self.assertEqual(worst['trial']['receipts'], {'verification': 'fixture'})

    def test_failures_and_unpaired_are_explicit_and_never_in_totals(self):
        failed = trial(self.cases, 0, 'fast-basic')
        failed.update(status='failed', measurement=None, error='decoder failed')
        mismatch = trial(self.cases, 1, 'fast-basic')
        mismatch['payload_sha256'] = 'b' * 64
        rows = [trial(self.cases, 0), trial(self.cases, 1), failed, mismatch]
        data, regressions = summary.summarize({'cases': self.cases}, rows)
        g = data['overall']['fast-basic']
        self.assertEqual(g['selected'], 3)
        self.assertEqual(g['paired'], 0)
        self.assertEqual(g['failures'], 1)
        self.assertEqual(g['failure_records'][0]['error'], 'decoder failed')
        self.assertEqual(g['unpaired'], dict(failed_trial=1, payload_mismatch=1, missing_trial=1))
        self.assertEqual(g['trial']['bytes'], 0)
        self.assertIsNone(g['total_bytes_ratio'])
        self.assertEqual(regressions['per_image'], [])
        self.assertEqual(data['status'], 'partial')
        self.assertFalse(data['sweep_complete'])
        self.assertEqual(data['missing_trials'], 14)

    def test_same_effort_and_passed_baseline_required(self):
        base = trial(self.cases, 0)
        row = trial(self.cases, 0, 'seed-only')
        row['effort'] = 8
        self.assertEqual(summary._reason(row, base), 'effort_mismatch')
        base['status'] = 'failed'
        self.assertEqual(summary._reason(row, base), 'failed_baseline')
        self.assertEqual(summary._reason(row, None), 'missing_baseline')

    def test_known_and_heldout_are_separate_not_inferred_from_paths(self):
        rows = [trial(self.cases, i, v, size=(200 if i == 0 and v != 'baseline' else 100))
                for i in range(3) for v in ('baseline', 'fast-basic')]
        data, _ = summary.summarize({'cases': self.cases}, rows)
        known = data['cohorts']['known_case']['fast-basic']
        heldout = data['cohorts']['heldout']['fast-basic']
        self.assertEqual(known['selected'], 1)
        self.assertEqual(known['total_bytes_ratio'], 2)
        self.assertEqual(heldout['selected'], 2)
        self.assertEqual(heldout['total_bytes_ratio'], 1)
        text = summary.markdown(data, summary.summarize({'cases': self.cases}, rows)[1])
        self.assertIn('previous 11 MA experiment cases', text)
        self.assertIn('not globally unseen data', text)
        self.assertIn('Median delta bytes', text)

    def test_duplicate_nan_malformed_and_truncated_jsonl_rejected(self):
        row = json.dumps(trial(self.cases, 0))
        for text in (row + '\n' + row + '\n', row, '{\n', '\n', 'null\n',
                     row.replace('"seconds": 1', '"seconds": NaN') + '\n',
                     row.replace('"seconds": 1', '"seconds": 1e999') + '\n',
                     row[:-1] + ', "index": 0}\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_bad_identity_dimensions_and_numbers_rejected(self):
        original = trial(self.cases, 0)
        updates = [dict(index=True), dict(index=3), dict(index=-1), dict(variant='other'),
                   dict(source='C:/fixtures/other.png'), dict(stratum='wrong'),
                   dict(original_sha256='b' * 64), dict(original_sha256=None),
                   dict(payload_sha256='bad'), dict(effort=True), dict(effort=0), dict(effort=11),
                   dict(effort=1.5), dict(pixels=0), dict(width=2), dict(depth=False),
                   dict(depth=1), dict(depth=4), dict(depth=12), dict(depth=32), dict(status='ok'),
                   dict(pixels_verified=False), dict(pixels_verified=1),
                   dict(receipt_validation=None), dict(receipt_validation={'status': 'validated'}),
                   dict(receipt_validation={'status': 'failed'}), dict(encoder={'returncode': 1}),
                   dict(decoder=None), dict(integrity_error='source changed'), dict(known_case=False),
                   dict(event='unknown')]
        for key, value in (('bytes', 0), ('bytes', 1.5), ('seconds', -1), ('seconds', True),
                           ('seconds', float('inf')), ('bpp', 0), ('bpp', 42),
                           ('mpix_per_second', 42)):
            metrics = dict(original['measurement'], **{key: value})
            updates.append(dict(measurement=metrics))
        for update in updates:
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.load(json.dumps(dict(original, **update)) + '\n')
        self.assertEqual(self.load(json.dumps(original) + '\n'), [original])

    def test_effort_boundaries_canonical_depths_and_bypass_accepted(self):
        for effort in (1, 10):
            for depth in (8, 16):
                row = trial(self.cases, 0)
                row.update(effort=effort, depth=depth, receipt_validation={'status': 'bypassed'})
                self.assertEqual(self.load(json.dumps(row) + '\n'), [row])

    def test_integrity_failed_forensics_retained_but_excluded(self):
        row = trial(self.cases, 0)
        row.update(status='failed', integrity_error='Source changed', error='Source changed')
        self.assertEqual(self.load(json.dumps(row) + '\n'), [row])
        data, regressions = summary.summarize({'cases': self.cases}, [row])
        baseline = data['overall']['baseline']
        self.assertEqual(baseline['paired'], 0)
        self.assertEqual(baseline['trial']['bytes'], 0)
        self.assertTrue(baseline['failure_records'][0]['pixels_verified'])
        self.assertEqual(baseline['failure_records'][0]['measurement'], row['measurement'])
        self.assertEqual(regressions['per_image'], [])
        row.update(payload_sha256=None, measurement=None, pixels_verified=False)
        self.assertEqual(self.load(json.dumps(row) + '\n'), [row])
        row['original_sha256'] = None
        with self.assertRaises(ValueError):
            self.load(json.dumps(row) + '\n')

    def test_unknown_event_row_rejected(self):
        with self.assertRaises(ValueError):
            self.load(json.dumps({'event': 'unknown', 'index': 0}) + '\n')

    def test_rewritten_integrity_rows_exclude_baseline_and_variant(self):
        rows = [trial(self.cases, 0), trial(self.cases, 0, 'fast-basic')]
        for row in rows:
            row.update(status='failed', integrity_error='Source changed', error='Source changed')
        rows += [trial(self.cases, 1), trial(self.cases, 1, 'fast-basic', size=90)]
        loaded = self.load(''.join(json.dumps(row) + '\n' for row in rows))
        data, regressions = summary.summarize({'cases': self.cases}, loaded)
        group = data['overall']['fast-basic']
        self.assertEqual(group['failures'], 1)
        self.assertEqual(group['paired'], 1)
        self.assertEqual(group['baseline']['bytes'], 100)
        self.assertEqual(group['trial']['bytes'], 90)
        self.assertEqual(group['failure_records'][0]['integrity_error'], 'Source changed')
        self.assertEqual([p['index'] for p in regressions['per_image']], [1])
        self.assertFalse(data['sweep_complete'])

    def test_malformed_integrity_marker_rejected(self):
        for status, error in (('passed', ''), ('passed', 'changed'),
                              ('failed', ''), ('failed', None), ('failed', True)):
            row = trial(self.cases, 0)
            row.update(status=status, integrity_error=error, error='Source changed')
            with self.subTest(status=status, error=error), self.assertRaises(ValueError):
                self.load(json.dumps(row) + '\n')

    def test_manifest_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.json'
            for value in ({'cases': []}, {'cases': [self.cases[0]] * 2},
                          {'cases': [dict(self.cases[0], known_case=1)]},
                          {'cases': [dict(self.cases[0], original='relative.png')]},
                          {'cases': [dict(self.cases[0], sha256='bad')]}):
                path.write_text(json.dumps(value), encoding='utf-8')
                with self.subTest(value=value), self.assertRaises(ValueError):
                    summary.load_manifest(path)
            path.write_text(json.dumps({'cases': self.cases}), encoding='utf-8')
            self.assertEqual(summary.load_manifest(path)['cases'], self.cases)

    def test_full_coverage_with_failure_not_success_or_full_500(self):
        rows = [trial(self.cases, i, v) for i in range(3) for v in summary.VARIANTS]
        rows[0].update(status='failed', measurement=None)
        data, _ = summary.summarize({'cases': self.cases}, rows)
        self.assertEqual(data['status'], 'completed_with_failures')
        self.assertTrue(data['sweep_complete'])
        self.assertFalse(data['full_500_sweep_complete'])
        self.assertEqual(data['overall']['fast-basic']['paired'], 2)

    def test_worst20_keeps_full_per_image_records(self):
        cases = [case(i) for i in range(25)]
        rows = [trial(cases, i, v, size=(100 if v == 'baseline' else 101 + i))
                for i in range(25) for v in ('baseline', 'seed-repair')]
        _, regressions = summary.summarize({'cases': cases}, rows)
        self.assertEqual(len(regressions['per_image']), 25)
        view = regressions['variants']['seed-repair']['largest_expansions']
        self.assertEqual(len(view), 20)
        self.assertEqual(view[0]['index'], 24)

    def test_cli_uses_atomic_helpers_and_does_not_modify_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, journal = root / 'sample.json', root / 'trials.jsonl'
            manifest.write_text(json.dumps({'cases': self.cases}), encoding='utf-8')
            journal.write_text(json.dumps(trial(self.cases, 0)) + '\n', encoding='utf-8')
            before = manifest.read_bytes(), journal.read_bytes()
            with mock.patch.object(summary, 'write_json') as write_json, \
                    mock.patch.object(summary, 'write_text') as write_text:
                summary.main(['--manifest', str(manifest), '--trials', str(journal),
                              '--out', str(root / 'reports')])
            self.assertEqual([c.args[0].name for c in write_json.call_args_list],
                             ['summary.json', 'regressions.json'])
            self.assertEqual(write_text.call_args.args[0].name, 'SUMMARY.md')
            self.assertIn('Status: partial', write_text.call_args.args[1])
            self.assertEqual(before, (manifest.read_bytes(), journal.read_bytes()))


if __name__ == '__main__':
    unittest.main()
