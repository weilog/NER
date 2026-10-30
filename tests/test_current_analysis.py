import importlib.util
from pathlib import Path
import numpy as np
import pytest
from ner_experiment.common import save_json

def module(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

def test_bootstrap_recomputes_global_micro_with_paired_draws():
    m=module('analyze_current_results')
    a=[{'id':'x','entity':[1,0,0],'token':[2,0,0]},{'id':'y','entity':[0,0,9],'token':[0,0,9]}]
    b=[{'id':'x','entity':[0,1,1],'token':[0,2,2]},{'id':'y','entity':[1,0,8],'token':[1,0,8]}]
    result=m.shared_bootstrap({'a':a,'b':b},17,repeats=50,pairs=[('a','b')])
    assert result['methods']['a']['entity']['estimate']==pytest.approx(2/11)
    aa=np.array([r['entity'] for r in a]);bb=np.array([r['entity'] for r in b])
    rng=np.random.default_rng(17);diff=[]
    for _ in range(50):
        idx=rng.integers(0,2,2)
        diff.append(float(m.micro_f1(aa[idx].sum(0))-m.micro_f1(bb[idx].sum(0))))
    np.testing.assert_allclose(result['comparisons'][0]['entity']['ci95'],np.quantile(diff,[.025,.975]))
    same=m.shared_bootstrap({'a':a,'b':a},17,50,pairs=[('a','b')])
    assert same['comparisons'][0]['entity']['ci95']==[0,0]
    with pytest.raises(ValueError):m.shared_bootstrap({'a':a,'b':b[::-1]},17,50,pairs=[('a','b')])

def test_review_sampling_unique_and_category_balanced():
    m=module('prepare_current_review');events=[]
    for model in m.METHODS:
        for kind in m.KINDS+['invalid','overlap','text_mismatch']:
            for i in range(8):events.append({'method':model,'sample_id':f'{model}_{kind}_{i}','kind':kind,'key':f'{model}_{kind}_{i}'})
    chosen=m.select_events(events,3)
    assert len(chosen)==len({r['sample_id'] for r in chosen})==40
    assert {r['method'] for r in chosen}==set(m.METHODS)
    for kind in m.KINDS:assert sum(r['kind']==kind for r in chosen)==7
    assert chosen==m.select_events(events,3)
    assert 'automatic_kind' not in m.FIELDS and 'method' not in m.FIELDS

def test_human_edits_preserved(tmp_path):
    m=module('prepare_current_review');folder=tmp_path/'packet';folder.mkdir()
    save_json(folder/'packet_manifest.json',{'source_fingerprint':'same'})
    for name in ['independent_A_20.csv','independent_B_20.csv','adjudication_40.csv','private_key.json','README.md']:
        (folder/name).write_text('human edits must remain',encoding='utf-8')
    before={p.name:p.read_bytes() for p in folder.iterdir()}
    assert m.preserve_packet(folder,'same')
    with pytest.raises(RuntimeError):m.preserve_packet(folder,'different')
    assert before=={p.name:p.read_bytes() for p in folder.iterdir()}
