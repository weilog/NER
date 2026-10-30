"""Temporary fixtures only: no real experiments or API calls."""
import importlib.util
from pathlib import Path
import pytest
from ner_experiment.common import save_json, file_hash

spec=importlib.util.spec_from_file_location('baseline_runner',Path(__file__).resolve().parents[1]/'scripts/run_baselines.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

def make_cache(folder):
    folder.mkdir()
    for name in runner.artifact_names('dictionary'):
        (folder/name).write_text('{}',encoding='utf-8')
    save_json(folder/'metrics.json',{'entity':{'f1':.5},'token':{'f1':.6}})
    hashes={name:file_hash(folder/name) for name in runner.artifact_names('dictionary')}
    info={'fingerprint':'abc','method':'dictionary','config':{},'artifact_sha256':hashes,
          'model_sha256':hashes['model.pkl'],'entity_f1':.5,'token_f1':.6}
    save_json(folder/'run.json',info)
    return info

def test_verified_cache_and_missing_run(tmp_path):
    folder=tmp_path/'run'
    assert runner.checked_cache(folder,'abc','dictionary',{}) is None
    expected=make_cache(folder)
    assert runner.checked_cache(folder,'abc','dictionary',{})==expected

@pytest.mark.parametrize('damage',['changed_code','missing','modified','incomplete','legacy'])
def test_cache_fails_without_overwriting(tmp_path,damage):
    folder=tmp_path/'run';info=make_cache(folder);fingerprint='abc'
    if damage=='changed_code': fingerprint='different'
    if damage=='missing': (folder/'model.pkl').unlink()
    if damage=='modified': (folder/'predictions.jsonl').write_text('changed',encoding='utf-8')
    if damage=='incomplete': (folder/'run.json').unlink()
    if damage=='legacy':
        del info['artifact_sha256'];save_json(folder/'run.json',info)
    before={p.name:p.read_bytes() for p in folder.iterdir()}
    with pytest.raises(RuntimeError):runner.checked_cache(folder,fingerprint,'dictionary',{})
    assert before=={p.name:p.read_bytes() for p in folder.iterdir()}

def test_data_file_and_configuration_checked(tmp_path):
    data=tmp_path/'data.jsonl';data.write_text('{}',encoding='utf-8')
    save_json(tmp_path/'data/splits/cluener2020/manifest.json',{'files':{'train':{'path':'data.jsonl','sha256':file_hash(data)}}})
    config={'dataset':{s+'_file':'data.jsonl' for s in ('train','dev','test')}}
    save_json(tmp_path/'configs/experiment.json',config)
    runner.verify_data(tmp_path)
    data.write_text('changed',encoding='utf-8')
    with pytest.raises(RuntimeError,match='hash mismatch'):runner.verify_data(tmp_path)
    data.write_text('{}',encoding='utf-8')
    config['dataset']['train_file']='untracked.jsonl'
    save_json(tmp_path/'configs/experiment.json',config)
    with pytest.raises(RuntimeError,match='absent from manifest'):runner.verify_data(tmp_path)
