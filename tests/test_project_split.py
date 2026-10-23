from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def load(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]

def test_project_split_integrity():
    base=ROOT/'data/processed'
    original=load(base/'cluener2020/train.jsonl')
    folds={s:load(base/'cluener2020_project'/f'{s}.jsonl') for s in ['train','dev','test']}
    assert [len(folds[s]) for s in folds]==[9539,1343,1209]
    assert (base/'cluener2020/dev.jsonl').read_bytes()==(base/'cluener2020_project/dev.jsonl').read_bytes()
    merged={r['id']:r for s in ['train','test'] for r in folds[s]}
    assert len(merged)==10748 and merged=={r['id']:r for r in original}
    textsets={s:{r['text'] for r in rows} for s,rows in folds.items()}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:assert not textsets[a]&textsets[b]
    for rows in folds.values():
        assert len({e['type'] for r in rows for e in r['entities']})==10
        assert all(r['has_labels'] for r in rows)

def test_reserved_sample_and_hashes():
    from ner_experiment.common import file_hash, load_json, valid_train
    manifest=load_json(ROOT/'data/splits/cluener2020/manifest.json')
    assert len(valid_train())==9538
    assert manifest['stats']['train']['skipped_training_ids']==['train:1506']
    for item in manifest['files'].values():assert file_hash(ROOT/item['path'])==item['sha256']

def test_repeat_stratification():
    import importlib.util
    spec=importlib.util.spec_from_file_location('split',ROOT/'scripts/split_cluener.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[r for r in load(ROOT/'data/processed/cluener2020/train.jsonl') if r['bio_compatible']]
    seed=json.loads((ROOT/'configs/split_seed.json').read_text())['seed']
    _,indices=module.split_indices(rows,test_size=1209,seed=seed)
    assert [rows[i]['id'] for i in indices]==json.loads((ROOT/'data/splits/cluener2020/test_ids.json').read_text())

def test_manifest_preserves_byte_order(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('refresh',ROOT/'scripts/refresh_split.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'manifest.json';path.write_text('{"b": 2, "a": 1}',encoding='utf-8')
    before=path.read_bytes();module.save_manifest(path,{'a':1,'b':2})
    assert path.read_bytes()==before
