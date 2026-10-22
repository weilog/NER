"""Reload selected local models and exactly reproduce saved Dev predictions."""
from pathlib import Path
import json
import joblib
from threadpoolctl import threadpool_limits
from ner_experiment.data.splitting import read_jsonl, json_bytes, sha
from ner_experiment.evaluation.metrics import evaluate
from ner_experiment.experiments.traditional import predict_rows, atomic_write


def main(root):
    root = Path(root)
    selected = json.loads((root / 'configs/traditional_selected.json').read_text(encoding='utf-8'))
    run = root / selected['run_directory']
    context = json.loads((run / 'context.json').read_text(encoding='utf-8'))
    for name, digest in context['source_sha256'].items():
        if sha((root / name).read_bytes()) != digest:
            raise ValueError(f'Source changed: {name}')
    dev_path = root / 'data/processed/experiment_v2/dev.jsonl'
    if sha(dev_path.read_bytes()) != context['dev_sha256']:
        raise ValueError('Dev hash changed')
    dev = read_jsonl(dev_path)
    result = {'test_accessed': False, 'selected_models': {}}
    with threadpool_limits(limits=1):
        for family, trial in selected['selected'].items():
            model_path = root / trial['model_path']
            if sha(model_path.read_bytes()) != trial['model_sha256']:
                raise ValueError('Model hash changed')
            model = joblib.load(model_path)
            restored = predict_rows(model, dev, family)
            saved = read_jsonl(run / trial['name'] / 'predictions.jsonl')
            stripped = lambda rows: [{key: value for key, value in row.items() if key != 'elapsed_seconds'} for row in rows]
            if stripped(saved) != stripped(restored):
                raise ValueError(f'Reloaded model predictions differ: {family}')
            metrics = evaluate(dev, restored)
            metrics.pop('normalized_predictions')
            if metrics != json.loads((run / trial['name'] / 'metrics.json').read_text(encoding='utf-8')):
                raise ValueError(f'Reloaded model metrics differ: {family}')
            result['selected_models'][family] = {'samples': len(dev), 'predictions_identical': True, 'metrics_identical': True}
            print(f'PASS {family}: reloaded model reproduces all {len(dev)} Dev predictions and metrics')
    atomic_write(run / 'verification/model-reload.json', json_bytes(result))


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
