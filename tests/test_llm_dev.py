"""Selection/leakage gates and a network-free durable two-phase run."""
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ner_experiment.data.splitting import json_bytes, sha
from ner_experiment.experiments import llm_dev as experiment
from ner_experiment.llm.prompts import request_for
from ner_experiment.llm.budget import Ledger

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/llm_dev_v1.json').read_text(encoding='utf-8'))


def test_few_shot_selection_excludes_zero_and_uses_declared_ties():
    arms = {f'shot_{n}': {'shots': n, 'entity': {'f1': f}, 'latency_mean_seconds': t}
            for n, f, t in [(0, 1, 1), (2, .7, 2), (4, .8, 3), (8, .8, 4)]}
    assert experiment.choose_shots(arms) == 4
    arms['shot_8']['latency_mean_seconds'] = 2
    assert experiment.choose_shots(arms) == 8
    arms['shot_4']['latency_mean_seconds'] = 2
    assert experiment.choose_shots(arms) == 4


def test_real_plan_has_fixed_prefixes_and_no_target_annotation():
    config, pricing, rows, requests, context, credential = experiment.prepare(ROOT, 'tune')
    assert len(rows) == 300 and len(requests) == 1200
    assert context['shots'] == [0, 2, 4, 8] and not context['test_accessed']
    assert all(r['id'].startswith('dev:') for r in rows)
    first = {r['shots']: r['request'] for r in requests[:4]}
    assert len({r['messages'][0]['content'] for r in first.values()}) == 1
    for n, request in first.items():
        assert len(request['messages']) == 2 + 2*n
        assert request['messages'][1:-1] == first[8]['messages'][1:1+2*n]
        target = json.loads(request['messages'][-1]['content'])
        assert 'entities' not in target and 'label' not in target
        assert target['text'] == rows[0]['text']


def test_frozen_artifact_refuses_silent_replacement(tmp_path):
    path = tmp_path / 'artifact'
    experiment.unchanged_write(path, b'original')
    experiment.unchanged_write(path, b'original')
    with pytest.raises(ValueError, match='Frozen artifact'):
        experiment.unchanged_write(path, b'changed')


def test_mock_tuning_cache_and_later_ledger_growth_do_not_change_results(tmp_path, monkeypatch):
    config = {**CONFIG, 'run_directory': 'runs/dev', 'ledger_path': 'runs/budget.sqlite3'}
    pricing = json.loads((ROOT / 'configs/deepseek_pricing_2026-10-26.json').read_text(encoding='utf-8'))
    pricing['verified_date_shanghai'] = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    rows = [{'id': 'dev:0', 'text': '张三', 'tokens': list('张三'), 'has_labels': True,
             'bio_compatible': True, 'bio_tags': ['B-name', 'I-name'],
             'entities': [{'start': 0, 'end': 2, 'type': 'name', 'text': '张三'}]}]
    requests = []
    for n in (0, 2, 4, 8):
        request = request_for(rows[0]['text'], config)
        request['messages'][0]['content'] += f' test arm {n}'
        requests.append({'id': 'dev:0', 'arm': f'shot_{n}', 'shots': n, 'request': request})
    context = {'source_sha256': {}, 'shots': [0, 2, 4, 8], 'endpoint': 'https://api.deepseek.com'}
    monkeypatch.setattr(experiment, 'prepare', lambda root, phase: (config, pricing, rows, requests, context, 'test-only'))
    calls = []
    class Fake:
        def __init__(self, *args): pass
        def send(self, request):
            calls.append(request)
            return {'model': 'deepseek-flash', 'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'entities': rows[0]['entities']})}}]}
        def close(self): pass
    monkeypatch.setattr(experiment, 'DeepSeekTransport', Fake)
    first = experiment.run(tmp_path, 'tune')
    assert first['complete'] and len(calls) == 4
    assert all(a['entity']['f1'] == 1 for a in first['arms'].values())
    paths = list((tmp_path / 'runs/dev/tune').rglob('*'))
    before = {str(p): sha(p.read_bytes()) for p in paths if p.is_file() and 'partial' not in p.name}
    ledger = Ledger(tmp_path / config['ledger_path'])
    attempt = ledger.reserve('other-stage', 1, 'another-stage', .01, 1)
    ledger.finish(attempt, error={'kind': 'rejected'}, cost=0)
    ledger.close()
    second = experiment.run(tmp_path, 'tune', offline=True)
    assert second == first and len(calls) == 4
    assert before == {str(p): sha(p.read_bytes()) for p in paths if p.is_file() and 'partial' not in p.name}
    assert not (tmp_path / 'runs/llm/dev_experiment.lock').exists()


def test_full_dev_refuses_without_completed_selection(tmp_path, monkeypatch):
    # Verify the actual prepare gate, with real immutable Dev fixtures and no API.
    monkeypatch.setattr(experiment, 'assert_branch', lambda root: None)
    import shutil
    for name in ['configs/llm_dev_v1.json', 'configs/llm_runtime_frozen.json', 'data/splits/experiment_v2.json',
                 'data/processed/experiment_v2/dev.jsonl', 'data/processed/experiment_v2/dev_tune.jsonl',
                 'data/processed/experiment_v2/examples_8.jsonl']:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    with pytest.raises(FileNotFoundError):
        experiment.prepare(tmp_path, 'full')
