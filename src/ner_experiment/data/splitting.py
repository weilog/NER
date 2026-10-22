"""Deterministic, sentence-label iterative stratification of text groups."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import random
import secrets
import zipfile

LABELS = ('address', 'book', 'company', 'game', 'government', 'movie',
          'name', 'organization', 'position', 'scene')
ALGORITHM = 'text-group-iterative-first-order-v1'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def jsonl_bytes(rows):
    return ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')


def sha(body):
    return hashlib.sha256(body).hexdigest()


def iterative_groups(groups, test_size, seed):
    """Rare-label-first assignment with hard capacity and seeded tie breaking.

    Stratification uses sentence presence, not entity occurrence counts. Groups
    remain indivisible. Groups which fit neither remaining bin are quarantined.
    """
    rng = random.Random(seed)
    sizes = [len(group) for group in groups]
    n = sum(sizes)
    capacity = [n - test_size, test_size]
    if min(capacity) < 0:
        raise ValueError('Requested test size exceeds eligible source pool')
    labels = [{e['type'] for row in group for e in row['entities']} for group in groups]
    totals = Counter(label for i, labs in enumerate(labels) for label in labs for _ in range(sizes[i]))
    desired = [{label: totals[label] * count / n for label in LABELS} for count in capacity] if n else [{}, {}]
    priority = list(range(len(groups)))
    rng.shuffle(priority)
    rank = {index: order for order, index in enumerate(priority)}
    pending = set(priority)
    assigned = [[], []]
    quarantine = []
    while pending:
        remaining = Counter(label for i in pending for label in labels[i] for _ in range(sizes[i]))
        rare = min(remaining, key=lambda label: (remaining[label], label)) if remaining else None
        candidates = sorted((i for i in pending if rare is None or rare in labels[i]), key=rank.get)
        for i in candidates:
            bins = [b for b in (0, 1) if capacity[b] >= sizes[i]]
            if not bins:
                quarantine.extend(groups[i])
            else:
                scores = {b: (desired[b].get(rare, 0), capacity[b]) for b in bins}
                best = max(scores.values())
                ties = [b for b in bins if scores[b] == best]
                b = rng.choice(ties)
                assigned[b].extend(groups[i])
                capacity[b] -= sizes[i]
                for label in labels[i]:
                    desired[b][label] -= sizes[i]
            pending.remove(i)
    return assigned[0], assigned[1], quarantine


def statistics(rows):
    presence = Counter(label for row in rows for label in {e['type'] for e in row['entities'] or []})
    occurrences = Counter(e['type'] for row in rows for e in row['entities'] or [])
    return {'sentences': len(rows), 'characters': sum(len(row['text']) for row in rows),
            'sentence_label_counts': {label: presence[label] for label in LABELS},
            'entity_occurrences': {label: occurrences[label] for label in LABELS},
            'overlap_ids': [row['id'] for row in rows if row['bio_compatible'] is False]}


def prepare_split(root):
    root = Path(root)
    source = root / 'data/processed/cluener2020'
    original = {split: read_jsonl(source / f'{split}.jsonl') for split in ('train', 'dev', 'test')}
    assert [len(original[s]) for s in ('train', 'dev', 'test')] == [10748, 1343, 1345]
    by_text = defaultdict(list)
    for split in ('train', 'dev'):
        for row in original[split]:
            by_text[row['text']].append(row)
    conflicts = []
    for text, rows in by_text.items():
        annotations = {tuple(sorted((e['start'], e['end'], e['type']) for e in row['entities'])) for row in rows}
        if len(annotations) > 1:
            conflicts.append({'text': text, 'records': rows})
    # Raw text in review records stays local and ignored by Git.
    review = root / 'runs/split-review.json'
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_bytes(json_bytes({'conflicts': conflicts}))
    if conflicts:
        raise ValueError(f'Annotation conflicts: review {review}; no split generated')
    if any(row['bio_compatible'] is not True for row in original['dev']):
        raise ValueError('Official Dev has overlapping gold; explicit protocol decision required')
    pinned = [row for row in original['train'] if not row['bio_compatible']]
    if [row['id'] for row in pinned] != ['train:1506']:
        raise ValueError('Unexpected overlapping Train annotations; review before splitting')
    dev_text = {row['text'] for row in original['dev']}
    groups = defaultdict(list)
    quarantine = []
    pinned_text = {row['text'] for row in pinned}
    fixed = []
    for row in original['train']:
        if row['text'] in dev_text:
            quarantine.append(row)
        elif row['text'] in pinned_text:
            fixed.append(row)
        else:
            groups[row['text']].append(row)
    if any(row['id'] in {p['id'] for p in pinned} for row in quarantine):
        raise ValueError('Pinned overlap sample duplicates Dev; review required')
    seed_path = root / 'configs/split_seed.json'
    if not seed_path.exists():
        state = {'seed': secrets.randbits(32), 'algorithm': ALGORITHM,
                 'generated_once': True, 'purpose': 'split and derived Dev subset/example selection'}
        with seed_path.open('xb') as stream:
            stream.write(json_bytes(state))
    state = json.loads(seed_path.read_text(encoding='utf-8'))
    if state['algorithm'] != ALGORITHM:
        raise ValueError('Seed algorithm changed; archive and explicitly migrate first')
    train, test, extra = iterative_groups(list(groups.values()), 1209, state['seed'])
    quarantine += extra
    train += fixed
    train.sort(key=lambda row: int(row['id'].split(':')[1]))
    test.sort(key=lambda row: int(row['id'].split(':')[1]))
    sets = {'train': train, 'dev': original['dev'], 'test': test}
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        assert not ({row['text'] for row in sets[a]} & {row['text'] for row in sets[b]})
    assert {row['id'] for row in train + test + quarantine} == {row['id'] for row in original['train']}
    assert len(train + test + quarantine) == len(original['train'])
    assert all(row['bio_compatible'] is True for row in test)
    if not quarantine and all(len(group) == 1 for group in groups.values()):
        assert (len(train), len(test)) == (9539, 1209)

    # Deterministic nested Dev subsets; format pilot is a prefix of the tune set.
    dev_indices = list(range(len(original['dev'])))
    random.Random(state['seed'] ^ 0xD3E).shuffle(dev_indices)
    tune = [original['dev'][i] for i in dev_indices[:300]]
    pilot = tune[:20]
    valid_train = [row for row in train if row['bio_compatible']]
    pool = valid_train.copy()
    random.Random(state['seed'] ^ 0xE8A).shuffle(pool)
    rank = {row['id']: i for i, row in enumerate(pool)}
    covered, examples = set(), []
    for _ in range(8):
        best = min(pool, key=lambda row: (-len({e['type'] for e in row['entities']} - covered), rank[row['id']]))
        examples.append(best)
        covered.update(e['type'] for e in best['entities'])
        pool.remove(best)
    outputs = {f'{name}.jsonl': jsonl_bytes(rows) for name, rows in sets.items()}
    outputs['train_valid.jsonl'] = jsonl_bytes(valid_train)
    # Copy bytes to preserve the official Dev and unlabeled Test serialization.
    outputs['dev.jsonl'] = (source / 'dev.jsonl').read_bytes()
    outputs['official_test_unlabeled.jsonl'] = (source / 'test.jsonl').read_bytes()
    outputs['quarantine.jsonl'] = jsonl_bytes(quarantine)
    outputs['dev_tune.jsonl'] = jsonl_bytes(tune)
    outputs['dev_pilot.jsonl'] = jsonl_bytes(pilot)
    outputs['examples_8.jsonl'] = jsonl_bytes(examples)
    manifest = {**state, 'protocol': 'dual-evaluation-v2',
                'source_hashes': {name: sha((source / f'{name}.jsonl').read_bytes()) for name in original},
                'archive_sha256': sha((root / 'data/raw/cluener2020/cluener_public.zip').read_bytes()),
                'ids': {name: [row['id'] for row in rows] for name, rows in sets.items()},
                'excluded_training_ids': [row['id'] for row in pinned],
                'quarantine_ids': [row['id'] for row in quarantine],
                'dev_pilot_ids': [row['id'] for row in pilot], 'dev_tune_ids': [row['id'] for row in tune],
                'examples_order': [row['id'] for row in examples],
                'examples_coverage': {str(k): sorted({e['type'] for row in examples[:k] for e in row['entities']}) for k in (2, 4, 8)},
                'statistics': {name: statistics(rows) for name, rows in sets.items()},
                'valid_training_sentences': len(valid_train),
                'output_sha256': {name: sha(body) for name, body in outputs.items()}}
    destination = root / 'data/processed/experiment_v2'
    manifest_path = root / 'data/splits/experiment_v2.json'
    if manifest_path.exists():
        if manifest_path.read_bytes() != json_bytes(manifest):
            raise ValueError('Frozen split manifest differs; refusing to overwrite')
        for name, body in outputs.items():
            if (destination / name).exists() and (destination / name).read_bytes() != body:
                raise ValueError(f'Existing split changed: {name}')
    else:
        old_paths = [path for path in (root / 'data/splits').rglob('*') if path.is_file() and path.name != '.gitkeep']
        if old_paths:
            archive = root / 'artifacts/split-archives/before-experiment-v2.zip'
            archive.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
                for path in old_paths:
                    stream.write(path, path.relative_to(root).as_posix())
    destination.mkdir(parents=True, exist_ok=True)
    for name, body in outputs.items():
        (destination / name).write_bytes(body)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(json_bytes(manifest))
    return manifest
