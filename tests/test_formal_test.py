from pathlib import Path
import json

import numpy as np
import pytest

from ner_experiment.data.splitting import json_bytes, sha
from ner_experiment.evaluation.error_analysis import attribute
from ner_experiment.experiments import formal_test as experiment
from ner_experiment.experiments.formal_analysis import novelty_counts, paired_bootstrap
from ner_experiment.llm.prompts import request_for

ROOT = Path(__file__).resolve().parents[1]


def row(identifier='train:0'):
    return {'id':identifier,'text':'张三','tokens':['张','三'],'bio_tags':['B-name','I-name'],
        'has_labels':True,'bio_compatible':True,
        'entities':[{'start':0,'end':2,'type':'name','text':'张三'}]}


def test_novelty_partition_includes_false_positives_and_new_types():
    rows=[{**row(),'text':'张三李四','entities':row()['entities']+[{'start':2,'end':4,'type':'name','text':'李四'}]}]
    predicted={'train:0':{'entities':[row()['entities'][0],{'start':2,'end':4,'type':'company','text':'李四'}]}}
    result=novelty_counts(rows,predicted,{('张三','name'),('李四','name')})
    assert [result['seen_train_surface_type'][k] for k in ('tp','fp','fn')]==[1,0,1]
    assert [result['unseen_train_surface_type'][k] for k in ('tp','fp','fn')]==[0,1,0]


def test_paired_bootstrap_recalculates_global_micro_and_checks_pairing():
    counts={'a':[{'id':'1','entity':[1,0,0]},{'id':'2','entity':[0,0,9]}],
            'b':[{'id':'1','entity':[0,0,1]},{'id':'2','entity':[0,0,9]}]}
    result=paired_bootstrap(counts,[['a','b']],123,2000)
    assert result[0]['entity_f1_difference']==pytest.approx(2/11)
    assert result==paired_bootstrap(counts,[['a','b']],123,2000)
    assert result[0]['ci95_low']==0 and result[0]['ci95_high']==1
    counts['b'].reverse()
    with pytest.raises(ValueError,match='identical'):
        paired_bootstrap(counts,[['a','b']],123,10)


def test_error_pairing_never_reuses_gold_for_two_split_predictions():
    errors=attribute({(0,4,'organization')},{(0,2,'organization'),(2,4,'organization')})
    assert sum(e['gold'] is not None for e in errors)==1
    assert sorted(e['category'] for e in errors)==['boundary','spurious']


def test_freeze_does_not_open_test_rows(monkeypatch):
    original_bytes,original_text=Path.read_bytes,Path.read_text
    def guarded_bytes(path,*args,**kwargs):
        if path.name=='test.jsonl':
            pytest.fail('Test opened while building the pre-Test freeze')
        return original_bytes(path,*args,**kwargs)
    def guarded_text(path,*args,**kwargs):
        if path.name=='test.jsonl':
            pytest.fail('Test opened while building the pre-Test freeze')
        return original_text(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',guarded_bytes)
    monkeypatch.setattr(Path,'read_text',guarded_text)
    captured=[]
    monkeypatch.setattr(experiment,'unchanged_write',lambda path,body:captured.append((path,body)))
    gate=experiment.freeze(ROOT)
    assert len(gate['test_ids'])==1209 and not gate['test_predictions_or_scores_used_for_selection']
    assert captured[0][0].name=='test_frozen_v1.json'


def test_uncommitted_gate_is_rejected_before_test_read(tmp_path,monkeypatch):
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/test_frozen_v1.json').write_text('{}',encoding='utf8')
    monkeypatch.setattr(experiment,'assert_branch',lambda root:None)
    monkeypatch.setattr(experiment.subprocess,'check_output',lambda *a,**k:b'{"different":true}')
    with pytest.raises(ValueError,match='committed'):
        experiment.verify_gate(tmp_path)


def test_mock_five_method_run_and_strict_offline_replay(tmp_path,monkeypatch):
    config=json.loads((ROOT/'configs/formal_test_v1.json').read_text(encoding='utf8'))
    parameters=json.loads((ROOT/'configs/llm_selected.json').read_text(encoding='utf8'))['parameters']
    pricing=json.loads((ROOT/config['pricing_file']).read_text(encoding='utf8'))
    from datetime import datetime
    from zoneinfo import ZoneInfo
    pricing['verified_date_shanghai']=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    (tmp_path/'configs').mkdir()
    (tmp_path/config['pricing_file']).write_bytes(json_bytes(pricing))
    rows=[row('train:1'),row('train:2')]
    gate={'config':config,'source_sha256':{},'llm':{'parameters':parameters},
        'traditional':{'selected':{n:{'model_path':'model','model_sha256':'mock'} for n in ('dictionary','hmm','crf')}}}
    requests=[{'id':r['id'],'method':m,'request':request_for(r['text'],parameters,[] if m=='llm_zero' else [row()]*4)}
        for r in rows for m in ('llm_zero','llm_4_shot')]
    context={'gate_sha256':'mock','endpoint':'https://api.deepseek.com'}
    monkeypatch.setattr(experiment,'prepare',lambda root:(gate,rows,requests,context,'test-only'))
    monkeypatch.setattr(experiment.subprocess,'check_output',lambda *a,**k:b'mock-commit')
    monkeypatch.setattr(experiment.joblib,'load',lambda path:object())
    prediction_calls=[]
    def predict(model,rows,family):
        prediction_calls.append(family)
        return [{'id':r['id'],'status':'ok','entities':r['entities'],'elapsed_seconds':.001} for r in rows]
    monkeypatch.setattr(experiment,'predict_rows',predict)
    calls=[]
    class Fake:
        def __init__(self,*args): pass
        def send(self,request):
            calls.append(request)
            return {'model':'deepseek-flash','usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120},
                'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'entities':row()['entities']})}}]}
        def close(self): pass
    monkeypatch.setattr(experiment,'DeepSeekTransport',Fake)
    with pytest.raises(ValueError,match='Offline mode refuses'):
        experiment.run(tmp_path,offline=True)
    assert not calls and not prediction_calls
    first=experiment.run(tmp_path)
    assert len(calls)==4 and len(prediction_calls)==3
    assert all(m['entity']['f1']==1 for m in first['methods'].values())
    files=[p for p in (tmp_path/config['run_directory']).rglob('*') if p.is_file() and p.name!='records.partial.jsonl']
    before={p:sha(p.read_bytes()) for p in files}
    replay=experiment.run(tmp_path,offline=True)
    assert replay==first and len(calls)==4 and len(prediction_calls)==3
    assert before=={p:sha(p.read_bytes()) for p in files}
