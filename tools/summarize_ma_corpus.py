"""Read-only MA journal analysis; never reads PNGs or executes a codec."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path, PureWindowsPath
import re
from statistics import median

from benchmark_encoder_corpus import write_json, write_text


VARIANTS = ('baseline', 'fast-basic', 'fast-full', 'seed-only', 'seed-refine', 'seed-repair')
HASH = re.compile(r'[0-9a-fA-F]{64}\Z')


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON member: {key}')
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f'Nonfinite JSON number: {value}')


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'Nonfinite JSON number: {value}')
    return number


def _json(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant, parse_float=_float)


def _hash(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def _positive(value, integer=False):
    return (type(value) is int if integer else type(value) in (int, float)) and value > 0 and math.isfinite(value)


def load_manifest(path):
    manifest = _json(path.read_text(encoding='utf-8-sig'))
    cases = manifest.get('cases') if isinstance(manifest, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError('Manifest requires nonempty cases')
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('Invalid manifest case')
        source = case.get('original')
        if (not isinstance(source, str)
                or not (Path(source).is_absolute() or PureWindowsPath(source).is_absolute())
                or PureWindowsPath(source).suffix.lower() != '.png'
                or not isinstance(case.get('stratum'), str) or not case['stratum']
                or not _hash(case.get('sha256')) or type(case.get('known_case')) is not bool):
            raise ValueError('Invalid manifest source, stratum, sha256 or known_case')
        if source in seen:
            raise ValueError('Duplicate manifest source')
        seen.add(source)
    return manifest


def load_trials(path, cases):
    rows, seen = [], set()
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith('\n'):
                raise ValueError(f'Truncated JSONL line {number}: missing newline')
            try:
                row = _json(line)
                required = {'index', 'source', 'stratum', 'variant', 'effort', 'status', 'pixels',
                            'width', 'height', 'depth', 'original_sha256', 'payload_sha256',
                            'measurement', 'encoder', 'decoder', 'receipt_validation', 'pixels_verified'}
                if not isinstance(row, dict) or not required <= row.keys() or 'event' in row:
                    raise ValueError('Expected trial object; journal events are not supported')
                index, variant = row.get('index'), row.get('variant')
                if type(index) is not int or not 0 <= index < len(cases) or variant not in VARIANTS:
                    raise ValueError('Invalid index or variant')
                case = cases[index]
                if row.get('source') != case['original'] or row.get('stratum') != case['stratum']:
                    raise ValueError('Source or stratum differs from manifest')
                if (not _hash(row['original_sha256'])
                        or row['original_sha256'].lower() != case['sha256'].lower()):
                    raise ValueError('Original hash differs from manifest')
                if (row.get('status') not in ('passed', 'failed')
                        or type(row.get('effort')) is not int or not 1 <= row['effort'] <= 10):
                    raise ValueError('Invalid status or effort')
                if row.get('known_case') != case['known_case'] or type(row.get('known_case')) is not bool:
                    raise ValueError('Known-case identity differs from manifest')
                if (index, variant) in seen:
                    raise ValueError('Duplicate trial key')
                if 'integrity_error' in row and (
                        row['status'] != 'failed' or not isinstance(row['integrity_error'], str)
                        or not row['integrity_error']):
                    raise ValueError('Integrity error requires a failed trial and nonempty message')
                if row['status'] == 'passed':
                    _validate_passed(row)
                elif not isinstance(row.get('error'), str) or not row['error']:
                    raise ValueError('Failed trial requires an error')
                seen.add((index, variant))
                rows.append(row)
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                raise ValueError(f'Invalid JSONL line {number}: {error}') from error
    return rows


def _validate_passed(row):
    receipt = row.get('receipt_validation')
    if (row.get('pixels_verified') is not True or row.get('integrity_error')
            or not isinstance(receipt, dict) or receipt.get('status') not in ('passed', 'bypassed')):
        raise ValueError('Passed trial lacks pixel/receipt evidence or has an integrity error')
    if any(not isinstance(row.get(k), dict) or row[k].get('returncode') != 0
           for k in ('encoder', 'decoder')):
        raise ValueError('Passed trial lacks successful process evidence')
    if not _hash(row.get('original_sha256')) or not _hash(row.get('payload_sha256')):
        raise ValueError('Passed trial requires original and payload SHA256')
    if any(not _positive(row.get(k), True) for k in ('width', 'height', 'depth', 'pixels')):
        raise ValueError('Invalid passed dimensions')
    if row['depth'] not in (8, 16):
        raise ValueError('Canonical PNG depth must be 8 or 16')
    if row['pixels'] != row['width'] * row['height']:
        raise ValueError('Pixels differ from dimensions')
    metrics = row.get('measurement')
    if not isinstance(metrics, dict) or not _positive(metrics.get('bytes'), True):
        raise ValueError('Invalid passed bytes')
    if any(not _positive(metrics.get(k)) for k in ('seconds', 'bpp', 'mpix_per_second')):
        raise ValueError('Invalid passed measurement')
    for key, expected in (('bpp', 8 * metrics['bytes'] / row['pixels']),
                          ('mpix_per_second', row['pixels'] / 1e6 / metrics['seconds'])):
        if not math.isclose(metrics[key], expected, rel_tol=1e-5, abs_tol=1e-9):
            raise ValueError(f'Inconsistent passed {key}')


def _totals(rows):
    pixels = sum(r['pixels'] for r in rows)
    size = sum(r['measurement']['bytes'] for r in rows)
    seconds = sum(r['measurement']['seconds'] for r in rows)
    return dict(count=len(rows), pixels=pixels, bytes=size, seconds=seconds,
                bpp=8 * size / pixels if pixels else None,
                mpix_per_second=pixels / 1e6 / seconds if seconds else None)


def _reason(row, baseline):
    if row is None:
        return 'missing_trial'
    if row['status'] != 'passed':
        return 'failed_trial'
    if baseline is None:
        return 'missing_baseline'
    if baseline['status'] != 'passed':
        return 'failed_baseline'
    if row['effort'] != baseline['effort']:
        return 'effort_mismatch'
    if row['payload_sha256'].lower() != baseline['payload_sha256'].lower():
        return 'payload_mismatch'
    if any(row[k] != baseline[k] for k in ('width', 'height', 'depth', 'pixels')):
        return 'dimensions_mismatch'
    return None


def _pair(case, row, baseline):
    size = row['measurement']['bytes'] / baseline['measurement']['bytes']
    time = row['measurement']['seconds'] / baseline['measurement']['seconds']
    return dict(index=row['index'], source=row['source'],
                relative_basename=PureWindowsPath(case.get('relative', row['source'])).as_posix()
                if case.get('relative') else PureWindowsPath(row['source']).name,
                stratum=case['stratum'], known_case=case['known_case'], variant=row['variant'],
                effort=row['effort'], size_ratio=size, time_ratio=time,
                size_change_pct=100 * (size - 1), time_change_pct=100 * (time - 1),
                byte_change=row['measurement']['bytes'] - baseline['measurement']['bytes'],
                trial=row, baseline=baseline)


def _group(cases, indexed, indices):
    result, pairs = {}, []
    for variant in VARIANTS:
        matched, reasons, failures = [], Counter(), []
        for index in indices:
            row = indexed.get((index, variant))
            reason = _reason(row, indexed.get((index, 'baseline')))
            if row is not None and row['status'] == 'failed':
                failures.append(row)
            if reason:
                reasons[reason] += 1
            else:
                matched.append(_pair(cases[index], row, indexed[index, 'baseline']))
        baseline = _totals([p['baseline'] for p in matched])
        trial = _totals([p['trial'] for p in matched])
        sizes = [p['size_ratio'] for p in matched]
        times = [p['time_ratio'] for p in matched]
        result[variant] = dict(
            selected=len(indices), attempted=sum((i, variant) in indexed for i in indices),
            failures=len(failures), failure_records=failures,
            paired=len(matched), baseline_paired_denominator=len(matched),
            unpaired=dict(reasons), baseline=baseline, trial=trial,
            total_bytes_ratio=trial['bytes'] / baseline['bytes'] if matched else None,
            total_seconds_ratio=trial['seconds'] / baseline['seconds'] if matched else None,
            median_size_ratio=median(sizes) if sizes else None,
            median_size_delta_bytes=median(p['byte_change'] for p in matched) if matched else None,
            median_absolute_size_delta_bytes=median(abs(p['byte_change']) for p in matched) if matched else None,
            median_time_ratio=median(times) if times else None,
            min_size_change_pct=min(p['size_change_pct'] for p in matched) if matched else None,
            max_size_change_pct=max(p['size_change_pct'] for p in matched) if matched else None,
            wins=sum(s < 1 for s in sizes), larger_over_0_25_pct=sum(s > 1.0025 for s in sizes),
            larger_over_5_pct=sum(s > 1.05 for s in sizes),
            both_bigger_and_slower=sum(p['size_ratio'] > 1 and p['time_ratio'] > 1 for p in matched))
        pairs.extend(matched)
    return result, pairs


def summarize(manifest, rows):
    cases = manifest['cases']
    indexed = {(r['index'], r['variant']): r for r in rows}
    overall, pairs = _group(cases, indexed, list(range(len(cases))))
    strata = {s: _group(cases, indexed, [i for i, c in enumerate(cases) if c['stratum'] == s])[0]
              for s in sorted({c['stratum'] for c in cases})}
    cohorts = {name: _group(cases, indexed, [i for i, c in enumerate(cases) if c['known_case'] == known])[0]
               for name, known in (('known_case', True), ('heldout', False))}
    expected = len(cases) * len(VARIANTS)
    failures = sum(r['status'] == 'failed' for r in rows)
    complete = len(rows) == expected
    status = 'partial' if not complete else ('completed_with_failures' if failures else 'completed')
    summary = dict(status=status, sweep_complete=complete, selected=len(cases),
                   selected_500=len(cases) == 500, full_500_sweep_complete=complete and len(cases) == 500,
                   expected_trials=expected, recorded_trials=len(rows), missing_trials=expected - len(rows),
                   failed_trials=failures, overall=overall, strata=strata, cohorts=cohorts)
    comparisons = [p for p in pairs if p['variant'] != 'baseline']

    def views(items):
        return {name: sorted((p for p in items if p[field] > threshold),
                             key=lambda p: (-p[field], p['index'], p['variant']))[:20]
                for name, field, threshold in (('largest_expansions', 'size_ratio', 1),
                                                ('largest_byte_increases', 'byte_change', 0),
                                                ('largest_slowdowns', 'time_ratio', 1))}

    regressions = dict(per_image=comparisons, worst20=views(comparisons),
                       variants={v: views([p for p in comparisons if p['variant'] == v])
                                 for v in VARIANTS[1:]})
    return summary, regressions


def markdown(summary, regressions):
    lines = ['# MA Corpus Summary', '',
             f"Status: {summary['status']}; selected {summary['selected']}/500; "
             f"recorded {summary['recorded_trials']}/{summary['expected_trials']}; "
             f"missing {summary['missing_trials']}; failed {summary['failed_trials']}.", '',
             'Matched comparisons require passed trials, manifest original hashes, identical payloads,',
             'dimensions and effort. Each variant has its own paired baseline denominator.',
             'One-pass native timings only; no statistical significance is claimed.',
             'Bytes and seconds are summed; bpp = 8*bytes/pixels; MPix/s = pixels/1e6/seconds.',
             'Median byte deltas show signed trial minus baseline and absolute magnitude per image.',
             'Known cases are the previous 11 MA experiment cases; heldout means outside those cases,',
             'not globally unseen data. Cohort membership comes only from manifest known_case flags.',
             'Compressed-file hashes are not a goal. Failures remain in selected denominators.', '']
    lines += ['Integrity invalidation rewrites affected trials as failed with integrity_error;',
              'retained measurements and verification receipts are forensic only, never paired.',
              'No separate integrity-event records or recovery are inferred.', '']
    fmt = lambda x: '-' if x is None else f'{x:.5g}'
    groups = [('Overall', summary['overall'])]
    groups += [(f'Stratum: {k}', v) for k, v in summary['strata'].items()]
    groups += [(f'Cohort: {k}', v) for k, v in summary['cohorts'].items()]
    for title, group in groups:
        lines += [f'## {title}', '',
                  '| Variant | Pairs/selected | Failed | Bytes ratio | Seconds ratio | Median size/time | Median delta bytes (signed / absolute) | bpp | MPix/s | Wins | >0.25% / >5% | Bigger+slower |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for v, g in group.items():
            lines.append(f"| {v} | {g['paired']}/{g['selected']} | {g['failures']} | "
                         f"{fmt(g['total_bytes_ratio'])} | {fmt(g['total_seconds_ratio'])} | "
                         f"{fmt(g['median_size_ratio'])}/{fmt(g['median_time_ratio'])} | "
                         f"{fmt(g['median_size_delta_bytes'])}/{fmt(g['median_absolute_size_delta_bytes'])} | "
                         f"{fmt(g['trial']['bpp'])} | {fmt(g['trial']['mpix_per_second'])} | "
                         f"{g['wins']} | {g['larger_over_0_25_pct']}/{g['larger_over_5_pct']} | "
                         f"{g['both_bigger_and_slower']} |")
        lines.append('')
    for name, items in regressions['worst20'].items():
        lines += [f'## {name.replace("_", " ").title()}', '']
        for p in items:
            label = f"{p['relative_basename']} [{p['stratum']}]".replace('\n', ' ')
            lines.append(f"- {label}: {p['variant']}, size {p['size_change_pct']:+.3f}%, "
                         f"time {p['time_change_pct']:+.3f}% (index {p['index']}).")
        lines.append('')
    lines += ['Full per-image pairs, receipts and per-variant worst20: `regressions.json`.',
              'Failure records, exclusion reasons and paired baseline totals: `summary.json`.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'trials', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        rows = load_trials(args.trials, manifest['cases'])
        summary, regressions = summarize(manifest, rows)
        targets = [args.out / name for name in ('SUMMARY.md', 'summary.json', 'regressions.json')]
        if any(t.resolve() in (args.manifest.resolve(), args.trials.resolve()) for t in targets):
            raise ValueError('Reports must not overwrite inputs')
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / 'summary.json', summary)
        write_json(args.out / 'regressions.json', regressions)
        write_text(args.out / 'SUMMARY.md', markdown(summary, regressions))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
