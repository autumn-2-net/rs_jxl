"""One-pass current-Rust corpus benchmark; cached codecs are never executed.

PNG timing includes the entire content pilot, excludes input decode and IO.
JPEG uses the frozen CLI's byte-reversible source search and reports wall time
only. Every output is independently verified by djxl, then removed. JSONL is the
resume journal; failures remain visible and do not terminate the corpus sweep.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter, sleep

import compare_cpp_port as port
from corpus_inputs import UnsupportedInput, prepare


FORMATS = {'.png', '.jpg', '.jpeg', '.pgm', '.ppm', '.pnm', '.pam', '.pfm'}
OTHER_IMAGES = {'.gif', '.jxl', '.y4m', '.apng', '.webp', '.bmp', '.tif', '.tiff'}
PILOT_ARGS = ['--search', '--content-pilot', '--fast-ranges', '--subtrees',
              '--concat-batches', '--pilot-parallel-initial',
              '--pilot-overlap-automatic', '--pilot-near-tie-consensus']


def now():
    return datetime.now(timezone.utc).isoformat()


def write_text(path, value):
    temporary = path.with_name(path.name + '.tmp')
    delays = (0.01, 0.03, 0.1)
    for attempt in range(len(delays) + 1):
        try:
            temporary.write_text(value, encoding='utf-8')
            temporary.replace(path)
            return
        except OSError as error:
            sharing = isinstance(error, PermissionError) or getattr(error, 'winerror', None) in (5, 32, 33)
            if not sharing or attempt == len(delays):
                raise
            sleep(delays[attempt])


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, ensure_ascii=True) + '\n')


_report_warnings = set()


def write_derived(path, value, *, as_json=False):
    """Reports are replaceable views; only the manifest and journal are critical."""
    # Serialize outside the IO handler: programming errors must remain visible.
    text = json.dumps(value, indent=2, ensure_ascii=True) + '\n' if as_json else value
    try:
        write_text(path, text)
    except OSError as error:
        if path not in _report_warnings:
            _report_warnings.add(path)
            try:
                print(f'Warning: cannot update {path.name}: {error}; '
                      'continuing with trials.jsonl as the checkpoint.', file=sys.stderr, flush=True)
            except OSError:
                pass
        return False
    _report_warnings.discard(path)
    return True


def validate_resume(original, current, reporting_update=None):
    """Permit an explicitly reviewed runner-only fix, never changed codec inputs."""
    if reporting_update is not None and reporting_update != original['scripts'][0]['sha256']:
        raise ValueError('Reporting update must name the original runner SHA256')
    if original == current:
        return False
    allowed = dict(original)
    old_runner, new_runner = original['scripts'][0], current['scripts'][0]
    if (reporting_update is not None and old_runner['path'] == new_runner['path']
            and Path(old_runner['path']).name == 'benchmark_encoder_corpus.py'):
        allowed['scripts'] = [new_runner, *original['scripts'][1:]]
        if allowed == current:
            return True
    raise ValueError('Resume requires unchanged inventory, binaries, input adapters and settings; '
                     'runner-only reporting repairs require --resume-reporting-update ORIGINAL_SHA256')


def inventory(catalog):
    spec = json.loads(catalog.read_text(encoding='utf-8'))
    if spec.get('version') != 1:
        raise ValueError('Expected corpus catalog version 1')
    base = (catalog.parent / spec.get('base_dir', '.')).resolve(strict=True)
    rows, excluded, seen, names = [], [], set(), set()
    for dataset in spec['datasets']:
        name, roots = dataset['name'], dataset['paths']
        if name in names or not roots:
            raise ValueError('Duplicate or empty dataset')
        names.add(name)
        for root_index, source in enumerate(roots):
            root = (base / source).resolve(strict=True)
            paths = [root] if root.is_file() else sorted(root.rglob('*'))
            prefix = name if len(roots) == 1 else f'{name}/{root_index:02d}'
            for path in paths:
                if not path.is_file() or path.suffix.lower() not in FORMATS | OTHER_IMAGES:
                    continue
                path = path.resolve(strict=True)
                if path in seen:
                    raise ValueError(f'Overlapping catalog roots: {path}')
                seen.add(path)
                relative = path.name if root.is_file() else path.relative_to(root).as_posix()
                stat = path.stat()
                row = dict(path=str(path), relative=f'{prefix}/{relative}', dataset=name,
                           input_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
                if path.suffix.lower() not in FORMATS:
                    excluded.append(dict(row, reason='Not a supported still PNG/JPEG/Netpbm input'))
                else:
                    rows.append(dict(row, index=len(rows)))
    if not rows:
        raise ValueError('No corpus inputs')
    return rows, excluded


def cached_records(folders):
    """Freeze historic records, including failed ones; no best-result cherry picking."""
    cache, origins = {}, []
    for folder in folders:
        manifest = folder / 'manifest.json'
        journal = folder / 'trials.jsonl'
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        origins.append(dict(path=str(folder), manifest_sha256=port.file_sha256(manifest),
                            journal_sha256=port.file_sha256(journal),
                            config=metadata.get('config', metadata.get('baseline_config'))))
        with journal.open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                key = record.get('relative')
                if not key:
                    continue
                name = f'{folder.name}/{record["encoder"]}'
                selected = {k: record.get(k) for k in ('status', 'input_sha256', 'payload_sha256',
                                                       'bytes', 'seconds', 'bpp', 'checks')}
                cache.setdefault(key, {})[name] = selected
    return cache, origins


def select_sample(inputs, sample):
    """Replay an archived sample in its original order; never replace missing cases."""
    spec = json.loads(sample.read_text(encoding='utf-8-sig'))
    selected = spec.get('inputs')
    if not isinstance(selected, list) or not selected:
        raise ValueError('Sample manifest must contain a nonempty inputs list')
    by_path = {Path(row['path']).resolve(): row for row in inputs}
    seen, rows = set(), []
    for source in selected:
        if not isinstance(source, dict) or not isinstance(source.get('path'), str):
            raise ValueError('Invalid sample input path')
        path = Path(source['path'])
        if not path.is_absolute():
            raise ValueError('Sample input paths must be absolute')
        path = path.resolve()
        if path in seen:
            raise ValueError(f'Duplicate sample input: {path}')
        seen.add(path)
        if path not in by_path:
            raise ValueError(f'Sample input missing from current catalog: {path}')
        current = by_path[path]
        if source.get('relative') != current['relative']:
            raise ValueError(f'Sample dataset identity changed: {path}')
        rows.append(dict(current, index=len(rows)))
    return rows


def match_cache(records, result):
    matched = {}
    for name, old in records.items():
        same_source = bool(result.get('input_sha256')) and old['input_sha256'] == result['input_sha256']
        same_payload = bool(result.get('payload_sha256')) and old['payload_sha256'] == result['payload_sha256']
        checks = old.get('checks') or []
        verified = (old.get('status') == 'ok' and bool(checks)
                    and all(check.get('status') == 'ok' for check in checks))
        comparable = verified and same_source and same_payload and result.get('status') == 'passed'
        item = dict(**old, same_source=same_source, same_payload=same_payload,
                    verified=verified, size_comparable=comparable,
                    timing_comparable=False)
        if comparable and old.get('bytes', 0) > 0:
            item['size_change_pct'] = 100 * (result['bytes'] / old['bytes'] - 1)
        matched[name] = item
    return matched


def totals(rows):
    passed = [r for r in rows if r['status'] == 'passed']
    pixels = sum(r['pixels'] for r in passed)
    size = sum(r['bytes'] for r in passed)
    wall = sum(r['wall_seconds'] for r in passed)
    calls = [r for r in passed if r.get('call_seconds') is not None]
    call_seconds = sum(r['call_seconds'] for r in calls)
    return dict(processed=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
                passed=len(passed), pixels=pixels, bytes=size,
                input_bytes=sum(r['input_bytes'] for r in passed),
                bpp=8 * size / pixels if pixels else None, wall_seconds=wall,
                wall_mpix_s=pixels / 1e6 / wall if wall else None,
                call_samples=len(calls), call_seconds=call_seconds,
                call_mpix_s=sum(r['pixels'] for r in calls) / 1e6 / call_seconds if call_seconds else None)


def report(out, manifest, results, status, active=None):
    groups = {}
    for row in results:
        groups.setdefault(f'{row["dataset"]}/{row.get("kind", "unknown")}', []).append(row)
    data = dict(status=status, updated_utc=now(), total=len(manifest['inputs']),
                excluded=len(manifest['excluded']), active=active, totals=totals(results),
                datasets={key: totals(rows) for key, rows in sorted(groups.items())},
                baselines_recomputed=False, comparisons={})
    for row in results:
        for name, record in row.get('cached', {}).items():
            group = data['comparisons'].setdefault(name, dict(source_matches=0, payload_matches=0,
                                                             pairs=0, old_bytes=0, new_bytes=0))
            group['source_matches'] += int(record['same_source'])
            group['payload_matches'] += int(record['same_payload'])
            if record['size_comparable']:
                group['pairs'] += 1
                group['old_bytes'] += record['bytes']
                group['new_bytes'] += row['bytes']
    for group in data['comparisons'].values():
        group['size_change_pct'] = (100 * (group['new_bytes'] / group['old_bytes'] - 1)
                                    if group['old_bytes'] else None)
    write_derived(out / 'summary.json', data, as_json=True)
    failures = [r for r in results if r['status'] != 'passed']
    write_derived(out / 'failures.json', failures, as_json=True)
    title = 'Archived Sample' if 'sample_manifest' in manifest else 'Full Corpus'
    lines = [f'# Rust Content-Pilot {title}', '', f'Status: **{status}**.',
             f'Processed {len(results)}/{data["total"]}; verified {data["totals"]["passed"]}; '
             f'failures/unsupported {len(failures)}; other-format exclusions {data["excluded"]}.', '',
             'One pass, e9, serial files, 16 threads by default. No old encoder is executed.',
             'PNG call time includes the complete pilot/search; startup, input decode and output IO excluded.',
             'JPEG timing is CLI wall only; original-byte reconstruction, no lossy/pixel fallback.',
             'Every result is checked with official djxl. Transient images and JXL files are deleted.',
             'Timeouts and resource limits are failures, not successful bounded optimization.',
             'Rates are total pixels / total time, not a mean of individual rates.',
             'Cached sizes are paired only with verified original AND payload identities.',
             'Source-only matches remain in trials.jsonl; differing metadata/payloads are not strict size pairs.',
             'Historic threading/concurrency/timing differ: cached wall time is context, not a controlled speedup.',
             '', '| Dataset / kind | Passed/processed | bpp | Call MPix/s | Wall MPix/s |',
             '| --- | ---: | ---: | ---: | ---: |']
    fmt = lambda n: '-' if n is None else f'{n:.4f}'
    for key, values in data['datasets'].items():
        lines.append(f'| {key} | {values["passed"]}/{values["processed"]} | {fmt(values["bpp"])} | '
                     f'{fmt(values["call_mpix_s"])} | {fmt(values["wall_mpix_s"])} |')
    lines += ['', '## Records', '', '- `manifest.json`: inputs, binaries, tool hashes, old-result provenance.',
              '- `progress.json`: active image/phase and process identity.',
              '- `trials.jsonl`: flushed per-image timings, bpp, commands, verification and cached results.',
              '- `failures.json`: all failed/unsupported attempts, not silently dropped.',
              '- `summary.json`: aggregate and dataset statistics; no old codec reruns.',
              '- Create a `STOP` file here to stop after the current image; resume retains all completed attempts.']
    write_derived(out / 'REPORT.md', '\n'.join(lines) + '\n')
    return data


def measure(row, args, env, phase):
    result = dict(row, status='failed', started_utc=now())
    stage = 'prepare'
    try:
        with tempfile.TemporaryDirectory(prefix='case-', dir=args.out / 'scratch') as tmp:
            directory = Path(tmp)
            path = Path(row['path'])
            stat = path.stat()
            if stat.st_size != row['input_bytes'] or stat.st_mtime_ns != row['mtime_ns']:
                raise ValueError('Input changed after inventory; refusing stale comparison')
            info, source, expected = prepare(path, directory)
            result.update(info)
            output = directory / 'encoded.jxl'
            if info['kind'] == 'png':
                command = [str(args.encoder), str(source), '--output', str(output),
                           '--threads', str(args.threads), '--ma-workers', str(min(args.threads, 64)),
                           '--effort', '9', *args.pilot_args]
            else:
                command = [str(args.jpeg_encoder), 'transcode-jpeg', str(source), str(output),
                           '--effort', '9', '--source-search', '--threads', str(args.threads)]
            stage = 'encode'
            phase(stage)
            process = port.run_process(command, env, args.timeout)
            result['encode'] = process
            port.check_process(process)
            size = output.stat().st_size
            if not 0 < size <= port.MAX_INPUT_BYTES:
                raise ValueError('Empty or oversized JXL output')
            result.update(bytes=size, bpp=8 * size / info['pixels'],
                          wall_seconds=process['wall_seconds'], call_seconds=None)
            if info['kind'] == 'png':
                metrics = json.loads(process['stdout'])
                if (metrics['bytes'] != size or metrics['width'] != info['width']
                        or metrics['height'] != info['height'] or not metrics['search']
                        or metrics['search_policy'] != 'content-pilot'
                        or not math.isfinite(metrics['seconds']) or metrics['seconds'] <= 0):
                    raise ValueError('Invalid encoder metrics or wrong search mode')
                result['call_seconds'] = metrics['seconds']
                result['call_mpix_s'] = info['pixels'] / 1e6 / metrics['seconds']
            result['wall_mpix_s'] = info['pixels'] / 1e6 / process['wall_seconds']
            stage = 'verify'
            phase(stage)
            restored = directory / ('restored.png' if info['kind'] == 'png' else 'restored.jpg')
            command = (port.decoder_command(args.djxl, output, restored, info['depth'])
                       if info['kind'] == 'png' else
                       [str(args.djxl), str(output), str(restored), '--reconstruct_jpeg',
                        '--num_threads=1', '--quiet'])
            result['decode'] = port.run_process(command, env, args.decoder_timeout)
            port.check_process(result['decode'])
            if info['kind'] == 'png':
                port.compare_pixels(expected, port.png_samples(restored.read_bytes()))
                result['verification'] = 'original samples including invisible RGB'
            elif restored.read_bytes() != expected:
                raise ValueError('Reconstructed JPEG bytes differ from original')
            else:
                result['verification'] = 'original JPEG bytes'
            result['status'] = 'passed'
    except UnsupportedInput as error:
        result.update(status='unsupported', error=str(error), stage=stage)
    except Exception as error:
        timeout = any('timeout' in str(result.get(key, {}).get('error', '')).lower()
                      for key in ('encode', 'decode'))
        result.update(status='timeout' if timeout else 'failed', stage=stage,
                      error=f'{type(error).__name__}: {error}')
    result['finished_utc'] = now()
    return result


def load_journal(path, inputs):
    result, seen = [], set()
    if not path.exists():
        return result
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            index = row['index']
            if (index in seen or not 0 <= index < len(inputs)
                    or row['relative'] != inputs[index]['relative']):
                raise ValueError('Invalid/duplicate journal record; refusing ambiguous resume')
            seen.add(index)
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('catalog', 'encoder', 'jpeg-encoder', 'djxl', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cached-run', type=Path, action='append', default=[])
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--decoder-timeout', type=float, default=300)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-reporting-update', metavar='ORIGINAL_SHA256',
                        help='Explicitly accept a reviewed reporting-only runner repair; '
                             'all other provenance must match the immutable manifest')
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument('--limit', type=int, help='Explicit smoke subset, never used for the full run')
    subset.add_argument('--sample-manifest', type=Path,
                        help='Replay exactly the archived inputs and order, without resampling')
    parser.add_argument('--pilot-progressive-groups', action='store_true',
                        help='Use the opt-in progressive group scheduler for pixel inputs only')
    parser.add_argument('--pilot-source-palette', action='store_true',
                        help='Use source palette defaults instead of forced Morton literal tools')
    parser.add_argument('--pilot-family-tournament', action='store_true',
                        help='Enable the experimental pixel pilot family tournament')
    args = parser.parse_args()
    if args.resume_reporting_update and not args.resume:
        parser.error('--resume-reporting-update requires --resume')
    if (not 1 <= args.threads <= 64 or (args.limit is not None and args.limit < 1)
            or any(not math.isfinite(n) or n <= 0 for n in (args.timeout, args.decoder_timeout))):
        parser.error('Invalid threads, limits, or timeouts')
    for key in ('catalog', 'encoder', 'jpeg_encoder', 'djxl'):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.cached_run = [p.resolve(strict=True) for p in args.cached_run]
    args.out = args.out.resolve()
    inputs, excluded = inventory(args.catalog)
    if args.sample_manifest is not None:
        args.sample_manifest = args.sample_manifest.resolve(strict=True)
        inputs = select_sample(inputs, args.sample_manifest)
        excluded = []
    if args.limit is not None:
        inputs = inputs[:args.limit]
    args.pilot_args = PILOT_ARGS + (['--pilot-progressive-groups'] if args.pilot_progressive_groups else [])
    if args.pilot_source_palette:
        args.pilot_args.append('--pilot-source-palette')
    if args.pilot_family_tournament:
        args.pilot_args.append('--pilot-family-tournament')
    cache, origins = cached_records(args.cached_run)
    env, environment = port.clean_environment(os.environ)
    binary = lambda p: dict(path=str(p), sha256=port.file_sha256(p), bytes=p.stat().st_size)
    manifest = dict(schema=1, inputs=inputs, excluded=excluded,
                    dataset_counts=dict(Counter(r['dataset'] for r in inputs)),
                    catalog=binary(args.catalog), encoder=binary(args.encoder),
                    jpeg_encoder=binary(args.jpeg_encoder), decoder=binary(args.djxl),
                    scripts=[binary(Path(__file__).with_name(name)) for name in
                             ('benchmark_encoder_corpus.py', 'corpus_inputs.py', 'compare_cpp_port.py')],
                    config=dict(threads=args.threads, effort=9, timeout=args.timeout,
                                decoder_timeout=args.decoder_timeout, concurrent_files=1,
                                repeats=1, limit=args.limit, pilot_args=args.pilot_args),
                    environment=environment, cached_origins=origins)
    if args.sample_manifest is not None:
        manifest['sample_manifest'] = binary(args.sample_manifest)
    reporting_update = False
    if args.resume:
        original = json.loads((args.out / 'manifest.json').read_text(encoding='utf-8'))
        try:
            reporting_update = validate_resume(original, manifest, args.resume_reporting_update)
        except ValueError as error:
            parser.error(str(error))
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / 'manifest.json', manifest)
    results = load_journal(args.out / 'trials.jsonl', inputs)
    if args.prepare_only:
        report(args.out, manifest, results, 'prepared')
        print(json.dumps(dict(count=len(inputs), datasets=manifest['dataset_counts'],
                              excluded=len(excluded), out=str(args.out)), indent=2), flush=True)
        return
    lock = args.out / 'run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), started_utc=now()), stream)
    status, active = 'running', None
    started = perf_counter()
    try:
        # Keep the initial manifest immutable; audit each resumed runner separately.
        with (args.out / 'sessions.jsonl').open('a', encoding='utf-8') as sessions:
            sessions.write(json.dumps(dict(pid=os.getpid(), started_utc=now(),
                completed_at_start=len(results), resume=args.resume,
                reporting_only_update=reporting_update,
                manifest_sha256=port.file_sha256(args.out / 'manifest.json'),
                journal_sha256_at_start=(port.file_sha256(args.out / 'trials.jsonl')
                                         if (args.out / 'trials.jsonl').exists() else None),
                scripts=manifest['scripts']), ensure_ascii=True) + '\n')
            sessions.flush()
            os.fsync(sessions.fileno())
        (args.out / 'scratch').mkdir(exist_ok=True)
        report(args.out, manifest, results, status)
        done = {r['index'] for r in results}
        with (args.out / 'trials.jsonl').open('a', encoding='utf-8', buffering=1) as journal:
            for row in inputs:
                if row['index'] in done:
                    continue
                if (args.out / 'STOP').exists():
                    status = 'stopped'
                    break
                active = dict(relative=row['relative'], index=row['index'], started_utc=now())
                def phase(stage):
                    write_derived(args.out / 'progress.json', dict(status='running', pid=os.getpid(),
                               total=len(inputs), completed=len(results), updated_utc=now(),
                               active=dict(**active, stage=stage)), as_json=True)
                phase('prepare')
                record = measure(row, args, env, phase)
                record['cached'] = match_cache(cache.get(row['relative'], {}), record)
                journal.write(json.dumps(record, ensure_ascii=True) + '\n')
                journal.flush()
                results.append(record)
                if len(results) % 10 == 0:
                    os.fsync(journal.fileno())
                if len(results) % 25 == 0 or record['status'] != 'passed' or len(results) == 1:
                    summary = report(args.out, manifest, results, status, active)
                    print(f'[{len(results)}/{len(inputs)}] {summary["totals"]["statuses"]} '
                          f'elapsed={perf_counter() - started:.1f}s {row["relative"]}', flush=True)
            else:
                status = 'completed' if all(r['status'] == 'passed' for r in results) else 'completed_with_failures'
        for key in ('encoder', 'jpeg_encoder', 'decoder'):
            if port.file_sha256(Path(manifest[key]['path'])) != manifest[key]['sha256']:
                raise ValueError(f'{key} changed during benchmark; results are invalid')
    except BaseException:
        status = 'interrupted'
        raise
    finally:
        try:
            report(args.out, manifest, results, status)
            write_derived(args.out / 'progress.json', dict(status=status, pid=os.getpid(),
                       total=len(inputs), completed=len(results), active=None, updated_utc=now()), as_json=True)
        finally:
            lock.unlink()
    print(f'{status}: {len(results)}/{len(inputs)} {args.out}', flush=True)


if __name__ == '__main__':
    main()
