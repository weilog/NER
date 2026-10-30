import urllib.error
import pytest
from ner_experiment.models.llm import Client, BudgetStop, choose_examples

ROW={'id':'x','text':'张三','tokens':list('张三')}
def response(content='{"entities": []}'):
    return {'model':'test-only','choices':[{'finish_reason':'stop','message':{'content':content}}], 'usage':{'prompt_tokens':10,'completion_tokens':5}}

def test_cache_and_budget(tmp_path):
    calls=[]
    def mock(p):calls.append(1);return response()
    c=Client(tmp_path,transport=mock)
    assert c.predict(ROW,[])['status']=='ok'
    assert Client(tmp_path,transport=mock).predict(ROW,[])['status']=='ok' and len(calls)==1
    with pytest.raises(BudgetStop):Client(tmp_path/'empty',budget=0,transport=mock).predict(ROW,[])
    assert len(calls)==1

def test_failure_and_invalid(tmp_path,monkeypatch):
    monkeypatch.setattr('ner_experiment.models.llm.time.sleep',lambda n:None)
    calls=[]
    def fail(p):calls.append(1);raise urllib.error.URLError('test')
    c=Client(tmp_path/'failure',transport=fail)
    assert c.predict(ROW,[])['status']=='api_error' and len(calls)==3
    assert c.total()>0 # uncertain failed requests retain reservations
    invalid=Client(tmp_path/'invalid',transport=lambda p:response('{"entities":[{"start":0,"end":99,"type":"other"}]}')).predict(ROW,[])
    assert invalid['status']=='ok' # V3 diagnoses invalid spans; auxiliary metric counts FP
    assert Client(tmp_path/'bad',transport=lambda p:response('oops')).predict(ROW,[])['status']=='format_error'

def test_no_retry_auth(tmp_path):
    calls=[]
    def fail(p):calls.append(1);raise urllib.error.HTTPError('url',401,'test',None,None)
    assert Client(tmp_path,transport=fail).predict(ROW,[])['status']=='api_error'
    assert len(calls)==1

def test_durable_unknown_charge(tmp_path,monkeypatch):
    monkeypatch.setattr('ner_experiment.models.llm.time.sleep',lambda n:None)
    def fail(p):raise TimeoutError()
    c=Client(tmp_path,transport=fail);c.predict(ROW,[])
    assert Client(tmp_path,transport=fail).total()==c.total()>0

def test_prompt_examples_from_valid_training():
    from ner_experiment.common import valid_train,load_json,ROOT
    rows=valid_train();seed=load_json(ROOT/'configs/split_seed.json')['seed']
    selected=choose_examples(rows,seed)
    assert len(selected)==8 and len({r['id'] for r in selected})==8
    assert all(r['bio_compatible'] and r['id']!='train:1506' for r in selected)
    assert [r['id'] for r in selected]==[r['id'] for r in choose_examples(rows,seed)]

@pytest.mark.parametrize('damage',['prediction','payload','raw_response','legacy','rehashed_prediction'])
def test_cache_corruption_stops_without_request(tmp_path,damage):
    from ner_experiment.common import load_json,save_json,object_hash
    calls=[]
    def mock(payload):calls.append(payload);return response()
    client=Client(tmp_path,transport=mock);client.predict(ROW,[])
    path=next(client.cache.glob('*.json'));record=load_json(path)
    assert record['payload']['messages'][-1]['role']=='user'
    if damage in ('prediction','rehashed_prediction'):
        record['prediction']['entities']=[{'start':0,'end':2,'type':'name'}]
    elif damage=='payload':record['payload']['temperature']=1
    elif damage=='raw_response':record['raw_response']['model']='changed'
    else:del record['cache_version']
    if damage=='rehashed_prediction':
        record['record_sha256']=object_hash({k:v for k,v in record.items() if k!='record_sha256'})
    save_json(path,record);before=path.read_bytes()
    with pytest.raises(ValueError):client.predict(ROW,[])
    assert len(calls)==1 and path.read_bytes()==before
