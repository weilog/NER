"""Pilot -> tuning -> complete Dev; Test is a separate frozen step. Cache resume is automatic."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import numpy as np
from ner_experiment.common import ROOT, valid_train, project_data, load_json, save_json, write_jsonl, file_hash, object_hash, utc_now
from ner_experiment.models.llm import Client, choose_examples, PARAMETERS, SYSTEM, BudgetStop
from run_baselines import save_evaluation

def run_group(client,rows,examples,folder,workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results=[]
        for p in pool.map(lambda r:client.predict(r,examples),rows):
            results.append(p)
            if len(results)%50==0:print(folder.name,len(results),'/',len(rows),'budget_upper=',round(client.total(),3),flush=True)
    metrics=save_evaluation(folder,rows,results)
    latency=[p['latency_seconds'] for p in results]
    save_json(folder/'run.json',{'created_at':utc_now(),'examples':[r['id'] for r in examples],'parameters':PARAMETERS,
       'actual_models':sorted({p.get('actual_model','unknown') for p in results}), 'samples':len(rows),
       'latency_median_seconds':float(np.median(latency)),'latency_p95_seconds':float(np.quantile(latency,.95)),
       'ledger_upper_cny':client.total(),'predictions_sha256':file_hash(folder/'predictions.jsonl')})
    print('DONE',folder.name,'entity_f1=',round(metrics['entity']['f1'],4),'failure=',metrics['failure_rate'],flush=True)
    return metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','dev'],default='pilot');p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    base=ROOT/'runs/llm';base.mkdir(parents=True,exist_ok=True);lock=base/'process.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode())
    try:
        client=Client();seed=load_json(ROOT/'configs/split_seed.json')['seed'];train=valid_train();dev=project_data('dev');lookup={r['id']:r for r in dev}
        examples=choose_examples(train,seed);save_json(ROOT/'data/splits/cluener2020/example_ids.json',[r['id'] for r in examples])
        pilot=[lookup[i] for i in load_json(ROOT/'data/splits/cluener2020/pilot_ids.json')]
        metrics=run_group(client,pilot,[],base/'pilot',args.workers)
        if metrics['failure_rate'] or metrics['invalid_sample_rate']:raise RuntimeError('Pilot has failed/invalid outputs; inspect before freezing interface')
        protocol={'parameters':PARAMETERS,'system_prompt':SYSTEM,'example_ids':[r['id'] for r in examples], 'seed':seed,
                  'pricing':load_json(ROOT/'configs/pricing.json'),'pilot_passed':True,'created_at':utc_now()}
        freeze=base/'interface_frozen.json'
        if freeze.exists():
            previous=load_json(freeze)
            for key in ['parameters','system_prompt','example_ids','seed','pricing']:
                if previous[key]!=protocol[key]:raise RuntimeError('Frozen interface changed')
        else:save_json(freeze,protocol)
        if args.stage=='pilot':return
        tuning=[lookup[i] for i in load_json(ROOT/'data/splits/cluener2020/dev_tuning_ids.json')]
        # Keep a conservative 80 CNY for complete Dev/Test. Fixed Test is never reduced.
        client.budget=20
        options=[]
        for shots in [0,2,4,8]:
            m=run_group(client,tuning,examples[:shots],base/f'tune_{shots}',args.workers)
            run=load_json(base/f'tune_{shots}'/'run.json')
            options.append({'shots':shots,'entity_f1':m['entity']['f1'],'token_f1':m['token']['f1'],'latency':run['latency_median_seconds']})
        best=min([o for o in options if o['shots']>0],key=lambda o:(-o['entity_f1'],o['latency'],o['shots']))['shots']
        client.budget=100
        for shots in [0,best]:run_group(client,dev,examples[:shots],base/f'dev_{shots}',args.workers)
        selection={'best_shots':best,'options':options,'criterion':'tuning Dev entity micro F1; tie: median latency then fewer examples',
           'interface_sha256':file_hash(freeze)}
        path=base/'selection.json'
        if path.exists():
            previous=load_json(path)
            if any(previous[k]!=v for k,v in selection.items()):raise RuntimeError('Selection changed; inspect before replacing frozen results')
        else:save_json(path,{**selection,'completed_at':utc_now()})
    finally:os.close(fd);lock.unlink()

if __name__=='__main__':main()
