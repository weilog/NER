"""Train CPU baselines and select on Dev only. Resumable with code/data fingerprints."""
import argparse
import pickle
import platform
import time
from pathlib import Path
import numpy as np
from ner_experiment.common import ROOT, valid_train, project_data, load_json, save_json, write_jsonl, file_hash, object_hash, utc_now
from ner_experiment.models.baselines import Dictionary, HMM, CRF
from ner_experiment.evaluation.metrics import evaluate

def save_evaluation(folder, rows, predictions):
    metrics, counts, cases=evaluate(rows,predictions)
    write_jsonl(folder/'predictions.jsonl',predictions); save_json(folder/'metrics.json',metrics)
    write_jsonl(folder/'sentence_counts.jsonl',counts);write_jsonl(folder/'errors.jsonl',cases)
    return metrics

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    train=valid_train();dev=project_data('dev')
    if args.smoke:train=train[:100];dev=dev[:20]
    base=ROOT/('runs/smoke_baselines' if args.smoke else 'runs/baselines');base.mkdir(parents=True,exist_ok=True)
    fingerprint=object_hash({'manifest':load_json(ROOT/'data/splits/cluener2020/manifest.json'),'smoke':args.smoke,
       'code':{p.name:file_hash(p) for p in [ROOT/'src/ner_experiment/models/baselines.py',ROOT/'src/ner_experiment/evaluation/metrics.py',ROOT/'src/ner_experiment/common.py',Path(__file__)]}})
    specs=[('dictionary',{})]+[('hmm',{'alpha':a,'min_freq':f}) for a in [.01,.1,1.] for f in [1,2,3]]+[('crf',{'c1':a,'c2':b}) for a,b in [(0,.1),(.1,.1),(.1,1.)]]
    results=[]
    for index,(kind,config) in enumerate(specs):
        name=f'{index:02d}_{kind}';folder=base/name;meta=folder/'run.json'
        if meta.exists() and load_json(meta)['fingerprint']==fingerprint:
            info=load_json(meta);results.append(info);print('CACHED',name,flush=True);continue
        model={'dictionary':Dictionary,'hmm':HMM,'crf':CRF}[kind](**config)
        print('FIT',name,config,flush=True);t=time.perf_counter();model.fit(train);fit=time.perf_counter()-t
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/'model.pkl').open('wb') as f:pickle.dump(model,f)
        predictions=[];latencies=[]
        for row in dev:
            t=time.perf_counter();predictions.append(model.predict(row));latencies.append(time.perf_counter()-t)
        metrics=save_evaluation(folder,dev,predictions)
        info={'name':name,'method':kind,'config':config,'order':index,'fingerprint':fingerprint,'created_at':utc_now(),
           'train_samples':len(train),'dev_samples':len(dev),'fit_seconds':fit,'predict_seconds':sum(latencies),
           'latency_median_seconds':float(np.median(latencies)),'latency_p95_seconds':float(np.quantile(latencies,.95)),
           'entity_f1':metrics['entity']['f1'],'token_f1':metrics['token']['f1'],'model_sha256':file_hash(folder/'model.pkl'),
           'environment':{'python':platform.python_version(),'platform':platform.platform()}}
        if kind=='hmm':
            other=save_evaluation(folder/'unconstrained',dev,[model.predict(r,constrained=False) for r in dev])
            chars=[c for r in dev for c in r['tokens']]
            info.update(vocabulary_size=len(model.vocabulary),dev_unknown_char_rate=sum(c not in model.vocabulary for c in chars)/len(chars),
               unconstrained_entity_f1=other['entity']['f1'],unconstrained_token_f1=other['token']['f1'],unconstrained_illegal_bio_rate=other['illegal_bio_rate'])
        save_json(meta,info);results.append(info);print('DONE',name,'entity=',round(info['entity_f1'],4),'token=',round(info['token_f1'],4),flush=True)
    selected={kind:min([r for r in results if r['method']==kind],key=lambda r:(-r['entity_f1'],r['fit_seconds']+r['predict_seconds'],r['order']))['name'] for kind in ['dictionary','hmm','crf']}
    best=min([r for r in results if r['name'] in selected.values()],key=lambda r:(-r['entity_f1'],r['fit_seconds']+r['predict_seconds'],r['order']))['name']
    save_json(base/'selection.json',{'fingerprint':fingerprint,'criterion':'Dev strict entity micro F1; tie: lower fit+predict time, then grid order','selected':selected,'best_traditional':best,'runs':results})
    print('SELECTED',selected,'BEST',best,flush=True)

if __name__=='__main__':main()
