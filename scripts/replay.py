"""Recompute every saved evaluation without network/model access and require exact equality."""
from ner_experiment.common import ROOT, read_jsonl, load_json
from ner_experiment.evaluation.metrics import evaluate
from ner_experiment.evaluation.metrics_v1 import evaluate as evaluate_v1
from ner_experiment.evaluation.metrics_v2 import evaluate as evaluate_v2

def evaluator_for(saved_metrics):
    version=saved_metrics.get('protocol_version','v1')
    if version=='v1':return evaluate_v1
    if version=='v2':return evaluate_v2
    if version=='v3':return evaluate
    raise ValueError(f'Unknown evaluation protocol: {version}')

def main():
    all_rows={r['id']:r for s in ['train','dev','test'] for r in read_jsonl(ROOT/f'data/processed/cluener2020_project/{s}.jsonl')}
    total=0
    for path in (ROOT/'runs').rglob('predictions.jsonl'):
        if not (path.parent/'metrics.json').exists():continue
        predictions=read_jsonl(path)
        gold=[all_rows[r['id']] for r in predictions]
        saved=load_json(path.parent/'metrics.json')
        metrics,counts,cases=evaluator_for(saved)(gold,predictions)
        if metrics!=saved:raise AssertionError(f'Metric mismatch: {path}')
        if counts!=read_jsonl(path.parent/'sentence_counts.jsonl'):raise AssertionError(f'Count mismatch: {path}')
        total+=1
    print(f'Offline replay passed: {total} saved evaluations; no API calls.')

if __name__=='__main__':main()
