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

def spans_to_bio(entities, length):
    """Deterministic projection only; strict entity scoring retains overlaps."""
    tags = ['O'] * length; occupied = set(); dropped = 0
    for start, end, category in sorted(entities, key=lambda e: (e[0], -e[1], e[2])):
        positions = set(range(start, end))
        if positions & occupied: dropped += 1; continue
        occupied |= positions; tags[start] = 'B-' + category
        tags[start + 1:end] = ['I-' + category] * (end - start - 1)
    return tags, dropped
