import importlib.util
from pathlib import Path
import json
import pytest
from ner_experiment.models.llm import Client

@pytest.fixture
def pilot(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('shot_budget',scripts/'pilot_shot_budget.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def response(entities):
    return {'model':'mock','choices':[{'finish_reason':'stop','message':{'content':json.dumps({'entities':entities})}}],
            'usage':{'prompt_tokens':100,'completion_tokens':20}}

def test_text_schema_and_cache_reparse(tmp_path,pilot):
    row={'id':'a','text':'张三','tokens':list('张三')}
    calls=[]
    def mock(payload):
        calls.append(payload)
        return response([{'start':0,'end':2,'type':'name','text':'张三'}])
    client=Client(tmp_path,transport=mock,message_builder=pilot.pilot_messages,response_parser=pilot.parse_text_response)
    result=client.predict(row,[])
    assert result['status']=='ok' and result['entities'][0]['text']=='张三'
    assert client.predict(row,[])==result and len(calls)==1
    assert 'text字段' in calls[0]['messages'][0]['content']
    assert pilot.parse_text_response(response([{'start':0,'end':2,'type':'name'}]))==('format_error',[])

def test_cost_arithmetic_and_conservative_no_cache(pilot):
    prices={'input_cache_hit':.04,'input_cache_miss':2,'output':8}
    usage={'prompt_tokens':1000,'completion_tokens':100,'prompt_cache_hit_tokens':800}
    assert pilot.summarize_usage(usage,prices)==pytest.approx(.0028)
    assert pilot.summarize_usage(usage,prices,cache=True)==pytest.approx(.001232)
