"""Freeze a stratified PNG sample without running an encoder or inspecting wins."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import struct

from compare_cpp_port import PNG_SIGNATURE, file_sha256
from corpus_inputs import png_framing


# Relative weights, not production image-classification rules.
WEIGHTS = dict(photo_highres=80, photo_other=60, urban=40, manga=40,
               mixed_local=40, document_web=60, game_ui=40, alpha_icon=50,
               texture=40, format_mixed=50)


def classify(row):
    dataset = row['dataset']
    parts = Path(row['relative'].replace('\\', '/')).parts
    text = '/'.join(parts).lower()
    if dataset == 'DIV2K':
        return 'photo_highres'
    if dataset == 'urban100':
        return 'urban'
    if dataset == 'manga109':
        return 'manga'
    if (dataset == 'local-screenshots' or '/screenshot_web/' in text
            or 'graveyard_' in text or any(name in text for name in (
                '5000-national', '5200-epa', '5300-noaa', '6800-ia-',
                '7000-lilith-plots', '8100-lilith-web', '6000-lilith-scans'))):
        return 'document_web'
    if dataset.startswith('local-'):
        return 'mixed_local'
    if '/screenshot_game/' in text or 'kenney_pixel_platformer' in text:
        return 'game_ui'
    if '/icon_' in text or '/pngimg/' in text or 'kenney_particle' in text:
        return 'alpha_icon'
    if '/textures_' in text or '/polyhaven/' in text:
        return 'texture'
    if (dataset in ('BSDS100', 'Set14', 'Set5') or '/photo_' in text
            or '/clic2025/' in text or '1400-lilith-nature' in text):
        return 'photo_other'
    return 'format_mixed'


def subgroup(row):
    parts = row['relative'].replace('\\', '/').split('/')
    if row['dataset'] == 'jxl-testdata':
        return '/'.join(parts[:3])
    if row['dataset'] == 'diverse-downloads':
        return '/'.join(parts[:3])
    return row['dataset']


def quotas(count):
    if count < len(WEIGHTS):
        raise ValueError('Count must cover every declared stratum')
    total = sum(WEIGHTS.values())
    result = {key: count * weight // total for key, weight in WEIGHTS.items()}
    rest = sorted(WEIGHTS, key=lambda key: (-(count * WEIGHTS[key] % total), key))
    for key in rest[:count - sum(result.values())]:
        result[key] += 1
    return result


def fair_order(rows, rng):
    """Shuffle within source subgroups, then round-robin instead of file-count dominance."""
    groups = defaultdict(list)
    for row in rows:
        groups[subgroup(row)].append(row)
    names = sorted(groups)
    rng.shuffle(names)
    for name in names:
        groups[name].sort(key=lambda row: row['original'].casefold())
        rng.shuffle(groups[name])
    while any(groups.values()):
        for name in names:
            if groups[name]:
                yield groups[name].pop()


def select(catalog, known_report, count, seed):
    catalog = catalog.resolve(strict=True)
    known_report = known_report.resolve(strict=True)
    inventory = json.loads(catalog.read_text(encoding='utf-8'))['inputs']
    known = {Path(c['source']).resolve() for c in
             json.loads(known_report.read_text(encoding='utf-8'))['cases']}
    pools, exclusions, seen_paths = defaultdict(list), [], set()
    for row in inventory:
        path = Path(row['path']).resolve()
        if path.suffix.lower() != '.png' or path in seen_paths:
            continue
        seen_paths.add(path)
        pools[classify(row)].append(dict(row, original=str(path), known_case=path in known))
    if known - seen_paths:
        raise ValueError('Known cases absent from catalog: ' + str(sorted(map(str, known - seen_paths))))
    target = quotas(count)
    rng = random.Random(seed)
    selected, hashes = defaultdict(list), {}

    def accept(row, stratum):
        path = Path(row['original'])
        try:
            # Reject malformed framing/APNG before selection; never use codec outcomes.
            data = path.read_bytes()
            if not data.startswith(PNG_SIGNATURE):
                raise ValueError('Not a PNG')
            digest = hashlib.sha256(data).hexdigest()
            _, info = png_framing(data)
            if digest in hashes:
                exclusions.append(dict(original=str(path), reason='exact-file-duplicate',
                                       duplicate_of=hashes[digest]))
                return False
            result = dict(original=str(path), stratum=stratum, dataset=row['dataset'],
                          relative=row['relative'], known_case=row['known_case'],
                          sha256=digest, input_bytes=len(data), width=info['width'],
                          height=info['height'], bit_depth=info['bit_depth'],
                          color_type=info['color_type'])
            selected[stratum].append(result)
            hashes[digest] = str(path)
            return True
        except (OSError, ValueError, struct.error) as error:
            if row['known_case']:
                raise ValueError(f'Invalid required known case: {path}: {error}') from error
            exclusions.append(dict(original=str(path), reason=str(error)))
            return False

    for stratum in WEIGHTS:
        required = sorted((r for r in pools[stratum] if r['known_case']),
                          key=lambda r: r['original'].casefold())
        if len(required) > target[stratum]:
            raise ValueError('Quota too small for known cases in ' + stratum)
        for row in required:
            if not accept(row, stratum):
                raise ValueError('Required known cases must be distinct')
        for row in fair_order([r for r in pools[stratum] if not r['known_case']], rng):
            if len(selected[stratum]) == target[stratum]:
                break
            accept(row, stratum)
        if len(selected[stratum]) != target[stratum]:
            raise ValueError(f'Not enough eligible unique PNGs in {stratum}')
    # Interleave strata so progress/ETA does not see just the cheap images first.
    names = list(WEIGHTS)
    rng.shuffle(names)
    for rows in selected.values():
        rng.shuffle(rows)
    cases = []
    while any(selected.values()):
        for name in names:
            if selected[name]:
                cases.append(selected[name].pop())
    assert len(cases) == count
    return dict(schema=1, seed=seed, count=count, quotas=target, cases=cases,
                catalog=dict(path=str(catalog), sha256=file_sha256(catalog)),
                prior_small_matrix=dict(path=str(known_report), sha256=file_sha256(known_report)),
                exclusions=exclusions,
                population_counts={key: len(pools[key]) for key in WEIGHTS},
                known_cases=sum(c['known_case'] for c in cases),
                scope='PNG pixel-lossless only; no JPEG transcoding, no crop/downsample',
                sampling='Fixed stratum quotas; shuffled subgroup round-robin; original-file SHA256 dedup',
                heldout_definition='Not in preceding 11-image MA experiment; not claimed never benchmarked before',
                exclusion_policy='Input framing/animation/duplicate exclusions only; no encoding outcomes used',
                pixel_format_counts=dict(Counter(f"{c['bit_depth']}bit-type{c['color_type']}" for c in cases)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--known-report', type=Path, required=True)
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('Manifest already exists; frozen samples are never overwritten')
    result = select(args.catalog, args.known_report, args.count, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in
                      ('count', 'quotas', 'known_cases', 'pixel_format_counts')}, indent=2))


if __name__ == '__main__':
    main()
