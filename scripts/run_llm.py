"""Pilot -> tuning -> complete Dev; Test is a separate frozen step. Cache resume is automatic."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import numpy as np
from ner_experiment.common import ROOT, valid_train, project_data, load_json, save_json, write_jsonl, file_hash, object_hash, utc_now
from ner_experiment.models.llm import Client, choose_examples, PARAMETERS, SYSTEM, BudgetStop
from run_baselines import save_evaluation, verify_data

ARTIFACTS=['predictions.jsonl','metrics.json','sentence_counts.jsonl','errors.jsonl']

def group_fingerprint(rows,examples):
    return object_hash({'rows':rows,'examples':examples,'parameters':PARAMETERS,'system':SYSTEM,
        'code':{str(p.relative_to(ROOT)):file_hash(p) for p in [Path(__file__).resolve(),
            ROOT/'src/ner_experiment/models/llm.py',ROOT/'src/ner_experiment/evaluation/metrics.py',ROOT/'src/ner_experiment/common.py']}})

def check_group(folder,fingerprint):
    if not folder.exists():return None
    pending=folder/'request_manifest.json'
    if not pending.exists() or load_json(pending).get('fingerprint')!=fingerprint:
        raise RuntimeError(f'Existing group is unverified or changed: {folder}. Use a new --output-dir.')
    meta=folder/'run.json'
    if not meta.exists():
        if any((folder/p).exists() for p in ARTIFACTS):
            raise RuntimeError(f'Partial result files: {folder}. Preserve and use a new directory.')
        return None # interrupted requests can resume from per-request cache
    info=load_json(meta)
    if info.get('fingerprint')!=fingerprint:raise RuntimeError('Group fingerprint mismatch')
    for name in ARTIFACTS:
        path=folder/name
        if not path.is_file() or info.get('artifact_sha256',{}).get(name)!=file_hash(path):
            raise RuntimeError(f'Damaged group artifact: {path}')
    return load_json(folder/'metrics.json')

def check_protocol(base,protocol):
    context=base/'request_protocol.json'
    if context.exists():
        if load_json(context)!=protocol:raise RuntimeError('Frozen request protocol changed; use a new --output-dir')
    elif base.exists() and any(base.iterdir()):
        raise RuntimeError('Legacy/nonempty LLM directory is protected; use a new --output-dir')
    freeze=base/'interface_frozen.json'
    if freeze.exists() and load_json(freeze).get('protocol')!=protocol:
        raise RuntimeError('Frozen interface changed')

def run_group(client,rows,examples,folder,workers):
    fingerprint=group_fingerprint(rows,examples)
    cached=check_group(folder,fingerprint)
    if cached is not None:
        print('CACHED',folder.name,flush=True);return cached
    if workers<1 or not rows:raise ValueError('Positive workers and nonempty rows required')
    save_json(folder/'request_manifest.json',{'fingerprint':fingerprint})
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
       'ledger_upper_cny':client.total(),'predictions_sha256':file_hash(folder/'predictions.jsonl'),
       'fingerprint':fingerprint,'artifact_sha256':{name:file_hash(folder/name) for name in ARTIFACTS}})
    print('DONE',folder.name,'entity_f1=',round(metrics['entity']['f1'],4),'failure=',metrics['failure_rate'],flush=True)
    return metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','dev'],default='pilot');p.add_argument('--workers',type=int,default=8)
    p.add_argument('--output-dir',type=Path,default=Path('runs/llm'))
    p.add_argument('--tuning-shots',nargs='+',type=int,choices=[0,2,4,8],default=[0,2,4,8],
                   help='Freeze candidate counts before requests; e.g. 0 4 reduces tuning calls, not evaluation samples.')
    args=p.parse_args()
    if args.workers<1:raise ValueError('workers must be positive')
    candidates=sorted(set(args.tuning_shots))
    if 0 not in candidates or len(candidates)<2:raise ValueError('Include 0 and at least one positive shot count')
    base=(ROOT/args.output_dir).resolve()
    if not base.is_relative_to((ROOT/'runs').resolve()) or base==(ROOT/'runs').resolve():
        raise ValueError('Output directory must be a subdirectory of project runs/')
    manifest,data_config=verify_data(ROOT)
    seed=load_json(ROOT/'configs/split_seed.json')['seed'];train=valid_train();dev=project_data('dev');lookup={r['id']:r for r in dev}
    examples=choose_examples(train,seed)
    pilot_ids=load_json(ROOT/'data/splits/cluener2020/pilot_ids.json')
    tuning_ids=load_json(ROOT/'data/splits/cluener2020/dev_tuning_ids.json')
    pilot=[lookup[i] for i in pilot_ids];tuning=[lookup[i] for i in tuning_ids]
    protocol={'parameters':PARAMETERS,'system_prompt':SYSTEM,'example_ids':[r['id'] for r in examples],
              'examples_sha256':object_hash(examples),'seed':seed,'pricing':load_json(ROOT/'configs/pricing.json'),
              'data_manifest':manifest,'dataset_config':data_config['dataset'],'pilot_ids':pilot_ids,'tuning_ids':tuning_ids,
              'tuning_shots':candidates,'implementation':group_fingerprint([],[])}
    # Read-only preflight happens before Client construction, writes, or any request.
    check_protocol(base,protocol)
    groups=[(pilot,[],base/'pilot')]
    groups += [(tuning,examples[:n],base/f'tune_{n}') for n in candidates]
    groups += [(dev,examples[:n],base/f'dev_{n}') for n in [0,2,4,8]]
    for rows,shots,folder in groups:check_group(folder,group_fingerprint(rows,shots))
    # Share the old lock location across output directories to prevent parallel ledgers.
    lock=ROOT/'runs/llm/process.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode())
    try:
        other_cost=sum(a['charged_upper_cny'] for path in (ROOT/'runs').rglob('ledger.json')
                       if path.resolve()!=(base/'ledger.json').resolve() for a in load_json(path)['attempts'])
        save_json(base/'request_protocol.json',protocol)
        save_json(base/'example_ids.json',[r['id'] for r in examples])
        client=Client(folder=base,budget=max(0,100-other_cost))
        metrics=run_group(client,pilot,[],base/'pilot',args.workers)
        if metrics['failure_rate'] or metrics['invalid_sample_rate']:raise RuntimeError('Pilot has failed/invalid outputs; inspect before freezing interface')
        freeze=base/'interface_frozen.json'
        if not freeze.exists():save_json(freeze,{'protocol':protocol,'pilot_passed':True,'created_at':utc_now()})
        if args.stage=='pilot':return
        # Keep a conservative 80 CNY for complete Dev/Test. Fixed Test is never reduced.
        client.budget=max(0,20-other_cost)
        options=[]
        for shots in candidates:
            m=run_group(client,tuning,examples[:shots],base/f'tune_{shots}',args.workers)
            run=load_json(base/f'tune_{shots}'/'run.json')
            options.append({'shots':shots,'entity_f1':m['entity']['f1'],'token_f1':m['token']['f1'],'latency':run['latency_median_seconds']})
        best=min([o for o in options if o['shots']>0],key=lambda o:(-o['entity_f1'],o['latency'],o['shots']))['shots']
        client.budget=max(0,100-other_cost)
        for shots in [0,best]:run_group(client,dev,examples[:shots],base/f'dev_{shots}',args.workers)
        selection={'best_shots':best,'options':options,'criterion':'tuning Dev entity micro F1; tie: median latency then fewer examples',
           'interface_sha256':file_hash(freeze)}
        path=base/'selection.json'
        if path.exists():
            previous=load_json(path)
            if any(previous[k]!=v for k,v in selection.items()):raise RuntimeError('Selection changed; inspect before replacing frozen results')
        else:save_json(path,{**selection,'completed_at':utc_now()})
    except BudgetStop:
        print('Budget stop: cached requests preserved. No automatic sample reduction or recharge; review before continuing.',flush=True)
        raise
    finally:os.close(fd);lock.unlink()

if __name__=='__main__':main()
