import importlib.util
from pathlib import Path
import pytest
from ner_experiment.common import save_json

@pytest.fixture
def runner(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('shot_test_runner',scripts/'run_shot_test.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_freeze_is_stable_and_rejects_changes(tmp_path,runner):
    path=tmp_path/'new/frozen_protocol.json'
    protocol={'shots':[0,2,4,8],'prompt':'fixed'}
    first=runner.freeze_once(path,protocol)
    before=path.read_bytes()
    assert runner.freeze_once(path,protocol)==first
    with pytest.raises(RuntimeError):runner.freeze_once(path,{**protocol,'prompt':'changed'})
    assert path.read_bytes()==before

def test_nonempty_run_without_freeze_is_protected(tmp_path,runner):
    save_json(tmp_path/'run/old.json',{})
    with pytest.raises(RuntimeError):runner.freeze_once(tmp_path/'run/frozen_protocol.json',{'shots':[0]})

def test_pilot_mismatch_stops_before_calls(runner):
    with pytest.raises(RuntimeError):runner.validate_pilot({'system':'changed','parameters':{},'shots':[0]}, {})

def test_text_mismatch_diagnosed_without_relocation(tmp_path,runner):
    from types import SimpleNamespace
    row={'text':'张三','tokens':list('张三')}
    prediction={'cache_key':'key','entities':[{'text':'张三','start':0,'end':1,'type':'name'}],
                'usage':{'prompt_tokens':10,'completion_tokens':5}}
    save_json(tmp_path/'key.json',{'raw_response':{'choices':[{'message':{}}]}})
    client=SimpleNamespace(cache=tmp_path,ledger={'attempts':[{'key':'key','charged_upper_cny':.01,'usage':prediction['usage']}]})
    result=runner.diagnostic_summary([row],[prediction],client)
    assert result['text_span_mismatches']==1 and result['cost_peak_ledger_cny']==.01
    assert prediction['entities'][0]['end']==1
