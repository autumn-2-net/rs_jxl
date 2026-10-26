import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import benchmark_encoder_corpus as corpus


class CorpusTests(unittest.TestCase):
    def setUp(self):
        corpus._report_warnings.clear()

    def test_atomic_write_retries_sharing_failure_and_preserves_old_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.json'
            path.write_text('old', encoding='utf-8')
            replace = Path.replace
            attempts = []
            def busy_once(temporary, target):
                attempts.append(target)
                if len(attempts) == 1:
                    self.assertEqual(path.read_text(encoding='utf-8'), 'old')
                    raise PermissionError('sharing violation')
                return replace(temporary, target)
            with mock.patch.object(Path, 'replace', busy_once), mock.patch.object(corpus, 'sleep') as sleep:
                corpus.write_json(path, {'completed': 988})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'completed': 988})
            self.assertEqual(len(attempts), 2)
            sleep.assert_called_once()
            self.assertFalse(path.with_name('progress.json.tmp').exists())

    def test_persistent_reporting_failure_warns_once_and_can_recover(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.json'
            with mock.patch.object(Path, 'replace', side_effect=PermissionError('busy')) as replace, \
                    mock.patch.object(corpus, 'sleep'), mock.patch('sys.stderr', new_callable=io.StringIO) as log:
                self.assertFalse(corpus.write_derived(path, {'a': 1}, as_json=True))
                self.assertFalse(corpus.write_derived(path, {'a': 2}, as_json=True))
                self.assertEqual(replace.call_count, 8)
                self.assertEqual(log.getvalue().count('Warning:'), 1)
            self.assertTrue(corpus.write_derived(path, {'a': 3}, as_json=True))
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'a': 3})
            self.assertNotIn(path, corpus._report_warnings)

    def test_nonsharing_io_error_is_not_retried_and_critical_write_stays_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            with mock.patch.object(Path, 'write_text', side_effect=OSError('disk full')), \
                    mock.patch.object(corpus, 'sleep') as sleep, \
                    mock.patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    corpus.write_json(path, {})
                self.assertFalse(corpus.write_derived(path.with_name('summary.json'), {}, as_json=True))
                sleep.assert_not_called()

    def test_serialization_errors_are_not_hidden_as_reporting_io_errors(self):
        with self.assertRaises(TypeError):
            corpus.write_derived(Path('progress.json'), object(), as_json=True)

    def test_reporting_resume_keeps_all_other_provenance_strict(self):
        original = dict(scripts=[dict(path=str(Path('tools/benchmark_encoder_corpus.py')), sha256='old', bytes=10),
                                 dict(path='corpus_inputs.py', sha256='adapter', bytes=20)],
                        inputs=[0, 1], config={'effort': 9}, encoder={'sha256': 'frozen'},
                        cached_origins=['baseline'])
        current = copy.deepcopy(original)
        current['scripts'][0].update(sha256='new', bytes=30)
        self.assertFalse(corpus.validate_resume(original, original))
        with self.assertRaises(ValueError):
            corpus.validate_resume(original, current)
        with self.assertRaises(ValueError):
            corpus.validate_resume(original, current, 'wrong original hash')
        self.assertTrue(corpus.validate_resume(original, current, 'old'))
        for key, value in [('inputs', [0]), ('config', {'effort': 8}),
                           ('encoder', {'sha256': 'rebuilt'}), ('cached_origins', ['different'])]:
            changed = copy.deepcopy(current)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                corpus.validate_resume(original, changed, 'old')
        changed = copy.deepcopy(current)
        changed['scripts'][1]['sha256'] = 'new adapter'
        with self.assertRaises(ValueError):
            corpus.validate_resume(original, changed, 'old')
        self.assertEqual(original['scripts'][0]['sha256'], 'old')

    def prepared_run(self, directory):
        root = Path(directory)
        for name in ('a.png', 'b.png'):
            (root / name).write_bytes(b'fixture; no encoder launched')
        catalog = root / 'catalog.json'
        catalog.write_text(json.dumps(dict(version=1,
            datasets=[dict(name='fixture', paths=['a.png', 'b.png'])])), encoding='utf-8')
        binary = root / 'frozen.exe'
        binary.write_bytes(b'fixture; never executed')
        out = root / 'results'
        args = ['benchmark_encoder_corpus.py', '--catalog', str(catalog),
                '--encoder', str(binary), '--jpeg-encoder', str(binary), '--djxl', str(binary),
                '--out', str(out)]
        with mock.patch('sys.argv', args + ['--prepare-only']), mock.patch('sys.stdout', new_callable=io.StringIO):
            corpus.main()
        manifest_bytes = (out / 'manifest.json').read_bytes()
        inputs = json.loads(manifest_bytes)['inputs']
        prefix = json.dumps(dict(inputs[0], status='unsupported')) + '\n'
        (out / 'trials.jsonl').write_text(prefix, encoding='utf-8')
        return args, out, inputs, (out / 'trials.jsonl').read_bytes(), manifest_bytes

    def test_resume_completes_when_all_derived_files_are_busy(self):
        with tempfile.TemporaryDirectory() as directory:
            args, out, inputs, prefix, manifest_bytes = self.prepared_run(directory)
            replace = Path.replace
            def busy_reports(temporary, target):
                if target.name in {'progress.json', 'summary.json', 'failures.json', 'REPORT.md'}:
                    raise PermissionError('persistent sharing violation')
                return replace(temporary, target)
            def measure(row, args, env, phase):
                self.assertEqual(row['index'], 1)
                phase('encode')
                phase('verify')
                return dict(row, status='unsupported')
            with mock.patch('sys.argv', args + ['--resume']), \
                    mock.patch('sys.stdout', new_callable=io.StringIO), \
                    mock.patch('sys.stderr', new_callable=io.StringIO) as log, \
                    mock.patch.object(Path, 'replace', busy_reports), mock.patch.object(corpus, 'sleep'), \
                    mock.patch.object(corpus, 'measure', side_effect=measure) as measured:
                corpus.main()
            measured.assert_called_once()
            self.assertEqual(log.getvalue().count('Warning:'), 4)
            self.assertFalse((out / 'run.lock').exists())
            self.assertTrue((out / 'trials.jsonl').read_bytes().startswith(prefix))
            self.assertEqual([r['index'] for r in corpus.load_journal(out / 'trials.jsonl', inputs)], [0, 1])
            self.assertEqual((out / 'manifest.json').read_bytes(), manifest_bytes)
            session = json.loads((out / 'sessions.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(session['completed_at_start'], 1)

    def test_journal_io_error_stops_run_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            args, out, _, prefix, _ = self.prepared_run(directory)
            open_path = Path.open
            def fail_journal(path, *args, **kwargs):
                if path.name == 'trials.jsonl' and args and args[0] == 'a':
                    raise OSError('journal unavailable')
                return open_path(path, *args, **kwargs)
            with mock.patch('sys.argv', args + ['--resume']), \
                    mock.patch.object(Path, 'open', fail_journal), mock.patch.object(corpus, 'measure') as measured:
                with self.assertRaisesRegex(OSError, 'journal unavailable'):
                    corpus.main()
            measured.assert_not_called()
            self.assertEqual((out / 'trials.jsonl').read_bytes(), prefix)
            self.assertFalse((out / 'run.lock').exists())
            self.assertEqual(json.loads((out / 'progress.json').read_text(encoding='utf-8'))['status'], 'interrupted')

    def test_cache_requires_verified_source_and_payload(self):
        previous = dict(status='ok', input_sha256='source', payload_sha256='payload',
                        bytes=100, seconds=2, bpp=1, checks=[dict(status='ok')])
        result = dict(status='passed', input_sha256='source', payload_sha256='payload', bytes=80)
        match = corpus.match_cache({'old': previous}, result)['old']
        self.assertTrue(match['size_comparable'])
        self.assertAlmostEqual(match['size_change_pct'], -20)
        self.assertFalse(match['timing_comparable'])
        for change in ({'input_sha256': 'changed'}, {'payload_sha256': 'different metadata'},
                       {'status': 'failed'}):
            match = corpus.match_cache({'old': previous}, dict(result, **change))['old']
            self.assertFalse(match['size_comparable'])
            self.assertNotIn('size_change_pct', match)
        for change in ({'status': 'failed'}, {'checks': []}, {'checks': [dict(status='failed')]}):
            match = corpus.match_cache({'old': dict(previous, **change)}, result)['old']
            self.assertFalse(match['size_comparable'])

    def test_aggregate_keeps_jpeg_wall_out_of_pixel_call_rate(self):
        png = dict(status='passed', pixels=2_000_000, bytes=100, input_bytes=200,
                   wall_seconds=3, call_seconds=2)
        jpeg = dict(status='passed', pixels=1_000_000, bytes=50, input_bytes=100,
                    wall_seconds=1, call_seconds=None)
        total = corpus.totals([png, jpeg, dict(status='unsupported')])
        self.assertEqual(total['processed'], 3)
        self.assertEqual(total['passed'], 2)
        self.assertEqual(total['call_mpix_s'], 1)
        self.assertEqual(total['wall_mpix_s'], .75)
        self.assertEqual(total['bpp'], 150 * 8 / 3_000_000)

    def test_inventory_keeps_invalid_static_files_for_attempt_and_other_formats_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            images.mkdir()
            for name in ('a.png', 'b.jpg', 'c.pfm', 'd.jxl', 'e.json'):
                (images / name).write_bytes(b'invalid fixture retained for input validation')
            catalog = root / 'sources.json'
            spec = dict(version=1, datasets=[dict(name='example', paths=['images'])])
            catalog.write_text(json.dumps(spec), encoding='utf-8')
            rows, excluded = corpus.inventory(catalog)
            self.assertEqual([r['relative'] for r in rows], ['example/a.png', 'example/b.jpg', 'example/c.pfm'])
            self.assertEqual([r['index'] for r in rows], [0, 1, 2])
            self.assertEqual([r['relative'] for r in excluded], ['example/d.jxl'])
            spec['datasets'].append(dict(name='overlap', paths=['images/a.png']))
            catalog.write_text(json.dumps(spec), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Overlapping'):
                corpus.inventory(catalog)

    def test_resume_never_reruns_completed_failures_or_accepts_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / 'trials.jsonl'
            inputs = [dict(index=0, relative='x')]
            row = dict(index=0, relative='x', status='failed')
            journal.write_text(json.dumps(row) + '\n', encoding='utf-8')
            self.assertEqual(corpus.load_journal(journal, inputs), [row])
            with journal.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                corpus.load_journal(journal, inputs)

    def test_latest_cached_attempt_wins_even_when_it_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text('{"config":{"threads":8}}', encoding='utf-8')
            base = dict(relative='x', encoder='baseline', status='ok', bytes=100)
            (root / 'trials.jsonl').write_text(json.dumps(base) + '\n' +
                json.dumps(dict(base, status='failed')) + '\n', encoding='utf-8')
            records, origins = corpus.cached_records([root])
            self.assertEqual(next(iter(records['x'].values()))['status'], 'failed')
            self.assertEqual(origins[0]['config']['threads'], 8)


if __name__ == '__main__':
    unittest.main()
