"""Verify real persisted pilot cache and offline metrics without network access."""
from pathlib import Path
import json
from ner_experiment.data.splitting import read_jsonl, json_bytes
from ner_experiment.evaluation.metrics import evaluate
from ner_experiment.experiments.traditional import atomic_write
from ner_experiment.llm.budget import Ledger
from ner_experiment.llm.client import execute_request
from ner_experiment.llm.pilot import prepare
from ner_experiment.llm.prompts import request_key


def main(root):
    root = Path(root)
    config, pricing, rows, requests, context, _ = prepare(root)
    directory = root / config['run_directory']
    ledger = Ledger(root / config['ledger_path'], config['budget_cny'])
    try:
        before = ledger.summary()
        restored = []
        for item in requests:
            key = request_key(item['id'], item['request'], context['endpoint'])
            if ledger.cached(key) is None:
                raise ValueError('A real cached response is missing; verification never sends requests')
            record, cached = execute_request(item['id'], item['request'], context['endpoint'], ledger,
                 None, pricing, config['stage'], config['pilot_budget_cny'])
            if not cached:
                raise ValueError('Verification unexpectedly missed cache')
            restored.append(record['prediction'])
        saved = read_jsonl(directory / 'predictions.jsonl')
        for actual, expected in zip(restored, saved):
            if actual['id'] != expected['id'] or actual['status'] != expected['status'] or actual['entities'] != expected['entities']:
                raise ValueError('Cached predictions differ from persisted pilot')
        metrics = evaluate(rows, restored)
        metrics.pop('normalized_predictions')
        if metrics != json.loads((directory / 'metrics.json').read_text(encoding='utf-8')):
            raise ValueError('Cached/offline metrics differ')
        after = ledger.summary()
        if before != after:
            raise ValueError('Offline verification changed API accounting')
        result = {'samples': 20, 'cache_hits': 20, 'additional_api_attempts': 0,
                  'offline_metrics_identical': True, 'accounting_unchanged': True,
                  'total_accounted_cny': after['accounted_cny'], 'test_accessed': False}
        atomic_write(root / 'runs/llm/verification/cache-replay.json', json_bytes(result))
        print(json.dumps(result, indent=2))
    finally:
        ledger.close()


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
