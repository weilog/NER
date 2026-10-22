"""Offline scoring of prediction JSONL against labeled experiment data."""
import argparse
from pathlib import Path
from ner_experiment.data.splitting import read_jsonl, json_bytes, jsonl_bytes, sha
from ner_experiment.evaluation.metrics import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(read_jsonl(args.gold), read_jsonl(args.predictions))
    normalized = result.pop('normalized_predictions')
    result['input_sha256'] = {'gold': sha(args.gold.read_bytes()), 'predictions': sha(args.predictions.read_bytes())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json_bytes(result))
    args.output.with_suffix('.normalized.jsonl').write_bytes(jsonl_bytes(normalized))
    print('Token:', result['token'])
    print('Entity:', result['entity'])
