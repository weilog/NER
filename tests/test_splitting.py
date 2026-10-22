from pathlib import Path
import json
import pytest
from ner_experiment.data import splitting

ROOT = Path(__file__).resolve().parents[1]


def test_indivisible_groups_and_seed_reproducibility():
    groups = [[{'id': str(i), 'entities': [{'type': 'name'}]}] for i in range(20)]
    group = [{'id': 'a', 'entities': [{'type': 'company'}]}, {'id': 'b', 'entities': [{'type': 'company'}]}]
    groups.append(group)
    first = splitting.iterative_groups(groups, 5, 123)
    assert first == splitting.iterative_groups(groups, 5, 123)
    train, test, quarantine = first
    assert len(train) == 17 and len(test) == 5 and not quarantine
    assert any({'a', 'b'} <= {row['id'] for row in rows} for rows in (train, test))
    assert len({row['id'] for row in train + test}) == 22


def test_no_label_sentence_allocation():
    groups = [[{'id': str(i), 'entities': []}] for i in range(10)]
    train, test, quarantined = splitting.iterative_groups(groups, 3, 9)
    assert (len(train), len(test), len(quarantined)) == (7, 3, 0)


def test_frozen_real_split_invariants_and_rerun():
    manifest_path = ROOT / 'data/splits/experiment_v2.json'
    if not manifest_path.exists():
        pytest.skip('Run prepare_cluener.py and prepare_split.py first')
    before = manifest_path.read_bytes()
    manifest = splitting.prepare_split(ROOT)
    assert manifest_path.read_bytes() == before
    directory = ROOT / 'data/processed/experiment_v2'
    rows = {name: splitting.read_jsonl(directory / f'{name}.jsonl') for name in ('train', 'dev', 'test')}
    assert [len(rows[name]) for name in rows] == [9539, 1343, 1209]
    assert manifest['valid_training_sentences'] == 9538
    for name, expected_hash in manifest['output_sha256'].items():
        assert splitting.sha((directory / name).read_bytes()) == expected_hash
    assert (directory / 'dev.jsonl').read_bytes() == (ROOT / 'data/processed/cluener2020/dev.jsonl').read_bytes()
    assert {row['id'] for row in rows['train'] + rows['test']} == {f'train:{i}' for i in range(10748)}
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        assert not ({row['text'] for row in rows[a]} & {row['text'] for row in rows[b]})
    assert [row['id'] for row in rows['train'] if not row['bio_compatible']] == ['train:1506']
    assert all(row['bio_compatible'] for row in rows['dev'] + rows['test'])
    assert not manifest['quarantine_ids']
    assert manifest['dev_pilot_ids'] == manifest['dev_tune_ids'][:20]
    assert len(manifest['dev_tune_ids']) == 300
    assert set(manifest['examples_order']) <= set(manifest['ids']['train'])
    assert not set(manifest['examples_order']) & set(manifest['excluded_training_ids'])
    assert manifest['examples_coverage']['4'] == list(splitting.LABELS)


def test_conflict_stops_before_seed_or_split(tmp_path, monkeypatch):
    source = ROOT / 'data/processed/cluener2020'
    if not (source / 'train.jsonl').exists():
        pytest.skip('Dataset unavailable')
    datasets = {name: splitting.read_jsonl(source / f'{name}.jsonl') for name in ('train', 'dev', 'test')}
    datasets['train'][0]['text'] = datasets['dev'][0]['text']
    datasets['train'][0]['entities'] = []
    monkeypatch.setattr(splitting, 'read_jsonl', lambda path: datasets[path.stem])
    with pytest.raises(ValueError, match='Annotation conflicts'):
        splitting.prepare_split(tmp_path)
    report = json.loads((tmp_path / 'runs/split-review.json').read_text(encoding='utf-8'))
    assert report['conflicts']
    assert not (tmp_path / 'configs/split_seed.json').exists()
    assert not (tmp_path / 'data/splits/experiment_v2.json').exists()
