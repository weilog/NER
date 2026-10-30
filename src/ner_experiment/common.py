"""Shared formats and reproducibility helpers."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
LABELS = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
TAGS = ['O'] + [prefix + label for prefix in ('B-', 'I-') for label in LABELS]

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')

def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def project_data(split):
    config = load_json(ROOT / 'configs/experiment.json')
    return read_jsonl(ROOT / config['dataset'][split + '_file'])

def valid_train():
    return [r for r in project_data('train') if r['bio_compatible']]

def normalize_bio(tags):
    normalized = []; illegal = 0
    for tag in tags:
        if tag not in TAGS:
            raise ValueError(f'Unknown BIO label: {tag}')
        if tag.startswith('I-') and (not normalized or normalized[-1] not in ('B-' + tag[2:], tag)):
            illegal += 1; tag = 'B-' + tag[2:]
        normalized.append(tag)
    return normalized, illegal

def bio_to_entities(tags):
    tags, illegal = normalize_bio(tags)
    entities = []; i = 0
    while i < len(tags):
        if tags[i] == 'O': i += 1; continue
        j = i + 1; category = tags[i][2:]
        while j < len(tags) and tags[j] == 'I-' + category: j += 1
        entities.append({'start': i, 'end': j, 'type': category}); i = j
    return entities, illegal

def entities_to_bio_set(entities, sample_id):
    """Expand ALL valid spans; overlapping BIO labels coexist at a character.

    Set semantics deduplicate identical (sample, position, BIO label) items.
    No O items are inserted. This is a BIO extension, not EDS-NLP's type-only metric.
    """
    items = set()
    for start, end, category in entities:
        items.add((sample_id, start, 'B-' + category))
        items.update((sample_id, i, 'I-' + category) for i in range(start + 1, end))
    return items


def overlapping_entities(entities):
    """Diagnostic only: unique spans participating in overlap, and pair count."""
    ordered = sorted(entities)
    involved = set()
    pairs = 0
    for i, first in enumerate(ordered):
        for second in ordered[i + 1:]:
            if second[0] >= first[1]:
                break
            involved.update((first, second))
            pairs += 1
    return involved, pairs


def resolve_overlaps(entities):
    """Select valid spans without consulting gold; never mutate the input.

    Priority: earliest start, longest at the same start, then LABELS order.
    Input must already have passed checked_entities().
    """
    category_order = {category: i for i, category in enumerate(LABELS)}
    kept, removed, occupied = set(), set(), set()
    ordered = sorted(entities, key=lambda e: (e[0], -e[1], category_order[e[2]]))
    for entity in ordered:
        start, end, category = entity
        positions = set(range(start, end))
        if positions & occupied:
            removed.add(entity)
        else:
            kept.add(entity)
            occupied.update(positions)
    return kept, removed


def spans_to_bio(entities, length):
    """Project valid spans to BIO using the shared overlap policy."""
    kept, removed = resolve_overlaps(entities)
    tags = ['O'] * length
    for start, end, category in kept:
        tags[start] = 'B-' + category
        tags[start + 1:end] = ['I-' + category] * (end - start - 1)
    return tags, len(removed)
