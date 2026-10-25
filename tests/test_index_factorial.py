from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import pytest

from ner_experiment.data.splitting import json_bytes
from ner_experiment.evaluation.index_diagnostics import diagnose, occurrences
from ner_experiment.evaluation.metrics import evaluate
from ner_experiment.experiments import index_factorial as experiment
from ner_experiment.llm.budget import Ledger, BudgetStopped
from ner_experiment.llm.index_prompts import build_request, indexed_text, system_prompt
from ner_experiment.llm.prompts import canonical, request_key, PROMPT_VERSION, PARSER_VERSION

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/index_factorial_v1.json').read_text(encoding='utf-8'))


def gold(identifier='dev:0', text='张三', entities=None, tags=None):
    return {'id': identifier, 'text': text, 'tokens': list(text), 'has_labels': True,
            'bio_compatible': True, 'bio_tags': tags or ['B-name', 'I-name'],
            'entities': entities or [{'start': 0, 'end': 2, 'type': 'name', 'text': '张三'}]}


def test_factorial_changes_only_two_declared_factors():
    example = {**gold('train:1'), 'bio_compatible': True}
    requests = {a['name'][0]: build_request('测试原文', [example], a, CONFIG) for a in CONFIG['arms']}
    assert requests['A']['messages'][0] == requests['B']['messages'][0]
    assert requests['C']['messages'][0] == requests['D']['messages'][0]
    for letter, request in requests.items():
        target = json.loads(request['messages'][-1]['content'])
        assert target['text'] == '测试原文' and 'entities' not in target
        assert request['messages'][2] == requests['A']['messages'][2]
        assert {k:v for k,v in request.items() if k != 'messages'} == {k:v for k,v in requests['A'].items() if k != 'messages'}
    assert requests['A']['messages'][-1] == requests['C']['messages'][-1]
    assert requests['B']['messages'][-1] == requests['D']['messages'][-1]
    assert json.loads(indexed_text('𠮷😀', 'columns'))['characters'] == [[0, '𠮷', 1], [1, '😀', 2]]


def test_fixed_dev_selection_excludes_all_adaptation_ids():
    ids = [f'dev:{i}' for i in range(1000)]
    excluded = ids[:300]
    picked = experiment.select_ids(ids, excluded, 141595842, 200)
    assert picked == experiment.select_ids(list(reversed(ids)), excluded, 141595842, 200)
    assert len(set(picked)) == 200 and not set(picked) & set(excluded)


def test_surface_diagnostic_disentangles_text_and_wrong_coordinate():
    prediction = {'id': 'dev:0', 'status': 'ok', 'entities': [{'start': 0, 'end': 1, 'type': 'name', 'text': '张三'}]}
    d = diagnose([gold()], [prediction])
    assert d['surface_type_bag']['f1'] == 1
    assert evaluate([gold()], [prediction])['entity']['f1'] == 0
    assert d['position_error_sample_rate'] == 1
    assert d['counts']['inclusive_end_signature_entities'] == 1
    assert d['counts']['unique_surface_relocatable_entities'] == 1


def test_repeated_mentions_are_not_uniquely_relocatable():
    row = gold(text='张三和张三', tags=['B-name','I-name','O','B-name','I-name'],
               entities=[{'start': 0,'end': 2,'type': 'name','text':'张三'}, {'start':3,'end':5,'type':'name','text':'张三'}])
    entry = {'start': 0, 'end': 1, 'type': 'name', 'text': '张三'}
    d = diagnose([row], [{'id':'dev:0','status':'ok','entities':[entry,entry]}])
    assert d['counts'].get('unique_surface_relocatable_entities', 0) == 0
    assert d['surface_type_bag']['tp'] == 1 and d['surface_type_bag']['fn'] == 1
    assert occurrences('哈哈哈','哈哈') == [0,1]


def test_cache_versions_are_isolated_without_invalidating_old_keys():
    request = {'messages': [], 'model': 'test'}
    material = {'id':'dev:0','endpoint':'https://api.deepseek.com','request':request,
                'prompt_version':PROMPT_VERSION,'parser_version':PARSER_VERSION}
    original = hashlib.sha256(canonical(material).encode()).hexdigest()
    assert request_key('dev:0',request,material['endpoint']) == original
    assert request_key('dev:0',request,material['endpoint'],prompt_version='new-arm') != original


def test_parallel_budget_reservations_cannot_overspend(tmp_path):
    path = tmp_path / 'budget.sqlite3'
    Ledger(path, total_limit=1).close()
    def reserve(i):
        ledger = Ledger(path, total_limit=1)
        try:
            ledger.reserve(str(i),1,'experiment',.3,1)
            return True
        except BudgetStopped:
            return False
        finally:
            ledger.close()
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(reserve, range(10))) == 3
    ledger = Ledger(path,total_limit=1)
    assert ledger.spent() == pytest.approx(.9)
    ledger.close()


def test_main_branch_is_rejected(monkeypatch):
    monkeypatch.setattr(experiment.subprocess,'check_output',lambda *args,**kwargs:b'main\n')
    with pytest.raises(ValueError,match='user branch'):
        experiment.assert_branch(ROOT)


def test_full_mock_batch_and_offline_replay(tmp_path,monkeypatch):
    config = {**CONFIG, 'concurrency':2,'run_directory':'runs/experiment','ledger_path':'runs/budget.sqlite3'}
    pricing = json.loads((ROOT / 'configs/deepseek_pricing_2026-10-25.json').read_text(encoding='utf-8'))
    pricing['verified_date_shanghai'] = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    rows = [gold('dev:0'),gold('dev:1')]
    requests = [{'id':row['id'],'arm':arm['name'],'prompt_version':'test/'+arm['name'],
                 'request':build_request(row['text'],[],arm,config)} for row in rows for arm in config['arms']]
    context = {'split_record':{},'source_sha256':{},'endpoint':'https://api.deepseek.com'}
    monkeypatch.setattr(experiment,'prepare',lambda root:(config,pricing,rows,requests,context,'test-placeholder'))
    calls = []
    class Fake:
        def __init__(self,*args): pass
        def send(self,request):
            calls.append(request)
            return {'model':'deepseek-flash','usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120},
                    'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'entities':gold()['entities']})}}]}
        def close(self): pass
    monkeypatch.setattr(experiment,'DeepSeekTransport',Fake)
    first = experiment.run(tmp_path)
    assert first['complete'] and first['api_attempts_this_invocation'] == 8
    assert all(a['entity']['f1'] == 1 for a in first['arms'].values())
    replay = experiment.run(tmp_path,offline=True)
    assert replay['complete'] and replay['api_attempts_this_invocation'] == 0 and len(calls) == 8
    assert replay['arms'] == first['arms']
    assert not (tmp_path / 'runs/llm/index_experiment.lock').exists()
