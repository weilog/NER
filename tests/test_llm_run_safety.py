"""No paid requests: all transports and data are local fixtures."""
import importlib.util
import sys
from pathlib import Path
import pytest
from ner_experiment.common import save_json,load_json
from ner_experiment.models.llm import Client,BudgetStop

@pytest.fixture
def runner(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('llm_runner_safety',scripts/'run_llm.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

ROW={'id':'a','text':'张三','tokens':list('张三'),'entities':[{'start':0,'end':2,'type':'name'}]}

def response(payload):
    return {'model':'mock','choices':[{'finish_reason':'stop','message':{'content':'{"entities":[{"start":0,"end":2,"type":"name"}]}'}}],
            'usage':{'prompt_tokens':10,'completion_tokens':5}}

def test_protocol_checks_before_writes(tmp_path,runner):
    base=tmp_path/'new';protocol={'parameters':{'temperature':0}}
    runner.check_protocol(base,protocol);assert not base.exists()
    save_json(base/'request_protocol.json',protocol)
    runner.check_protocol(base,protocol)
    with pytest.raises(RuntimeError):runner.check_protocol(base,{'parameters':{'temperature':1}})
    save_json(base/'interface_frozen.json',{'protocol':{'different':True}})
    with pytest.raises(RuntimeError):runner.check_protocol(base,protocol)
    legacy=tmp_path/'legacy';save_json(legacy/'ledger.json',{'attempts':[]})
    with pytest.raises(RuntimeError):runner.check_protocol(legacy,protocol)

def test_group_resume_and_damage_no_request(tmp_path,runner):
    calls=[]
    def transport(payload):calls.append(payload);return response(payload)
    client=Client(tmp_path/'client',transport=transport);folder=tmp_path/'group'
    first=runner.run_group(client,[ROW],[],folder,1)
    before={p.name:p.read_bytes() for p in folder.iterdir()}
    assert runner.run_group(client,[ROW],[],folder,1)==first and len(calls)==1
    assert before=={p.name:p.read_bytes() for p in folder.iterdir()}
    (folder/'predictions.jsonl').write_text('damaged',encoding='utf-8')
    with pytest.raises(RuntimeError):runner.run_group(client,[ROW],[],folder,1)
    assert len(calls)==1

def test_budget_interruption_resumes_without_repeating_paid_requests(tmp_path,runner):
    calls=[]
    def transport(payload):calls.append(payload);return response(payload)
    client=Client(tmp_path/'client',budget=0,transport=transport);folder=tmp_path/'group'
    with pytest.raises(BudgetStop):runner.run_group(client,[ROW],[],folder,1)
    assert not calls and not (folder/'run.json').exists()
    client.budget=100
    result=runner.run_group(client,[ROW],[],folder,1)
    assert len(calls)==1 and result['entity']['f1']==1

def test_legacy_group_is_not_overwritten(tmp_path,runner):
    folder=tmp_path/'group';save_json(folder/'run.json',{'old':True})
    def forbidden(payload):raise AssertionError('Unexpected request')
    client=Client(tmp_path/'client',transport=forbidden)
    before=(folder/'run.json').read_bytes()
    with pytest.raises(RuntimeError):runner.run_group(client,[ROW],[],folder,1)
    assert (folder/'run.json').read_bytes()==before

def test_changed_protocol_main_stops_before_client(tmp_path,runner,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'verify_data',lambda root:({}, {'dataset':{}}))
    monkeypatch.setattr(runner,'valid_train',lambda:[])
    monkeypatch.setattr(runner,'project_data',lambda split:[ROW])
    monkeypatch.setattr(runner,'choose_examples',lambda rows,seed:[])
    monkeypatch.setattr(runner,'group_fingerprint',lambda *a:'implementation')
    save_json(tmp_path/'configs/split_seed.json',{'seed':1})
    save_json(tmp_path/'configs/pricing.json',{})
    for name in ['pilot_ids','dev_tuning_ids']:save_json(tmp_path/f'data/splits/cluener2020/{name}.json',['a'])
    save_json(tmp_path/'runs/new/request_protocol.json',{'old':True})
    def forbidden(*a,**kw):raise AssertionError('Client must not be constructed')
    monkeypatch.setattr(runner,'Client',forbidden)
    monkeypatch.setattr(sys,'argv',['run_llm.py','--output-dir','runs/new'])
    with pytest.raises(RuntimeError,match='protocol changed'):runner.main()
    assert load_json(tmp_path/'runs/new/request_protocol.json')=={'old':True}
