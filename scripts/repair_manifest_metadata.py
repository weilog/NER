"""One-time audit of serialization-only manifest drift. Never changes model/data/selection."""
from pathlib import Path
from ner_experiment.common import ROOT,load_json,save_json,file_hash,object_hash,utc_now

def main():
    freeze=ROOT/'configs/frozen_experiment.json';record=load_json(freeze);manifest=ROOT/'data/splits/cluener2020/manifest.json'
    changed=[p for p,h in record['snapshot']['files'].items() if file_hash(ROOT/p)!=h]
    if not changed:return
    assert changed==['data/splits/cluener2020/manifest.json'],changed
    # Baseline fingerprint was recorded before Test and hashes sorted semantic JSON.
    fingerprint=object_hash({'manifest':load_json(manifest),'smoke':False,
       'code':{p.name:file_hash(p) for p in [ROOT/'src/ner_experiment/models/baselines.py',ROOT/'src/ner_experiment/evaluation/metrics.py',ROOT/'src/ner_experiment/common.py',ROOT/'scripts/run_baselines.py']}})
    assert fingerprint==load_json(ROOT/'runs/baselines/selection.json')['fingerprint']
    for item in load_json(manifest)['files'].values():assert file_hash(ROOT/item['path'])==item['sha256']
    backup=ROOT/'configs/frozen_experiment_original.json'
    if backup.exists():raise RuntimeError('One-time repair already recorded')
    backup.write_bytes(freeze.read_bytes());old=record['snapshot']['files'][changed[0]];new=file_hash(manifest)
    record['snapshot']['files'][changed[0]]=new
    save_json(ROOT/'configs/freeze_metadata_migration.json',{'date':utc_now(),'reason':'JSON object key insertion order only; semantic fingerprint unchanged from pre-Test baseline record',
        'original_manifest_sha256':old,'current_manifest_sha256':new,'pre_test_semantic_fingerprint_verified':fingerprint,
        'data_model_predictions_and_selections_changed':False,'original_freeze_sha256':file_hash(backup)})
    save_json(freeze,record);print('Recorded serialization-only repair; original freeze preserved.')

if __name__=='__main__':main()
