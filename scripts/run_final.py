"""Freeze Dev selections, then evaluate the fixed Test exactly once (cache-resumable)."""
import os
import pickle
import argparse
import time
import numpy as np
from ner_experiment.common import ROOT, project_data, valid_train, load_json, save_json, file_hash, read_jsonl, utc_now
from ner_experiment.models.llm import Client
from run_llm import run_group
from run_baselines import save_evaluation

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    traditional=load_json(ROOT/'runs/baselines/selection.json');llm=load_json(ROOT/'runs/llm/selection.json')
    freeze_path=ROOT/'configs/frozen_experiment.json'
    tracked=[ROOT/'data/splits/cluener2020/manifest.json',ROOT/'configs/experiment.json',ROOT/'configs/pricing.json',
        ROOT/'runs/baselines/selection.json',ROOT/'runs/llm/selection.json',ROOT/'runs/llm/interface_frozen.json']
    tracked+=list((ROOT/'src/ner_experiment').rglob('*.py'))
    snapshot={'files':{p.relative_to(ROOT).as_posix():file_hash(p) for p in tracked},'selected':traditional['selected'],
         'best_traditional':traditional['best_traditional'],'best_shots':llm['best_shots'],'test_size':1209}
    if freeze_path.exists():
        old=load_json(freeze_path)
        if old['snapshot']!=snapshot:raise RuntimeError('Frozen code/config/data changed. Do not tune on Test.')
    else:save_json(freeze_path,{'frozen_at':utc_now(),'snapshot':snapshot})
    test=project_data('test');manifest=load_json(ROOT/'data/splits/cluener2020/manifest.json')
    for item in manifest['files'].values():
        if file_hash(ROOT/item['path'])!=item['sha256']:raise RuntimeError('Data hash changed')
    assert len(test)==1209
    for method,name in traditional['selected'].items():
        source=ROOT/'runs/baselines'/name
        if file_hash(source/'model.pkl')!=load_json(source/'run.json')['model_sha256']:raise RuntimeError('Model hash mismatch')
        folder=ROOT/'runs/final'/method
        if (folder/'predictions.jsonl').exists():predictions=read_jsonl(folder/'predictions.jsonl')
        else:
            # Only locally trained, hash-verified artifacts are deserialized.
            with (source/'model.pkl').open('rb') as f:model=pickle.load(f)
            predictions=[];latencies=[]
            for row in test:
                started=time.perf_counter();predictions.append(model.predict(row));latencies.append(time.perf_counter()-started)
            save_json(folder/'run.json',{'selected_run':name,'samples':len(test),
                'latency_median_seconds':float(np.median(latencies)),'latency_p95_seconds':float(np.quantile(latencies,.95))})
        metrics=save_evaluation(folder,test,predictions);print('TEST',method,metrics['entity']['f1'],flush=True)
    client=Client();lock=client.folder/'process.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode())
    try:
        rows={r['id']:r for r in valid_train()};examples=[rows[i] for i in load_json(ROOT/'data/splits/cluener2020/example_ids.json')]
        for shots,name in [(0,'llm_zero'),(llm['best_shots'],'llm_few')]:
            run_group(client,test,examples[:shots],ROOT/'runs/final'/name,args.workers)
    finally:os.close(fd);lock.unlink()

if __name__=='__main__':main()
