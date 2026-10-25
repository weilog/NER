"""Replay all 800 cached predictions, protecting original records and accounting."""
from pathlib import Path
import json
from ner_experiment.data.splitting import json_bytes, jsonl_bytes, sha
from ner_experiment.evaluation.metrics import evaluate
from ner_experiment.evaluation.index_diagnostics import diagnose
from ner_experiment.experiments.index_factorial import prepare
from ner_experiment.experiments.traditional import atomic_write
from ner_experiment.llm.budget import Ledger
from ner_experiment.llm.prompts import canonical, request_key


def main(root):
    root = Path(root)
    config, pricing, rows, requests, context, _ = prepare(root)
    directory = root / config['run_directory']
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    if not summary['complete'] or sha(json_bytes(context)) != summary['context_sha256']:
        raise ValueError('Completed identical experiment context required')
    protected = [directory / 'records.jsonl', directory / 'summary.json']
    protected += [directory / arm['name'] / filename for arm in config['arms']
                  for filename in ('predictions.jsonl','metrics.json','normalized.jsonl','diagnostics.json')]
    before = {p.relative_to(root).as_posix():sha(p.read_bytes()) for p in protected}
    ledger = Ledger(root / config['ledger_path'],config['budget_cny'])
    try:
        accounting = ledger.summary()
        arm_predictions = {arm['name']:{} for arm in config['arms']}
        for item in requests:
            key = request_key(item['id'],item['request'],context['endpoint'],prompt_version=item['prompt_version'])
            record = ledger.cached(key)
            if record is None:
                raise ValueError('Missing cache; verifier never calls API')
            arm_predictions[item['arm']][item['id']] = record['prediction']
        stable = lambda value:json.loads(canonical(value))
        for arm in config['arms']:
            predictions = [arm_predictions[arm['name']][row['id']] for row in rows]
            metrics = evaluate(rows,predictions)
            normalized = metrics.pop('normalized_predictions')
            diagnostic = diagnose(rows,predictions)
            computed = {'predictions.jsonl':jsonl_bytes(stable(predictions)),
                        'metrics.json':json_bytes(stable(metrics)),
                        'normalized.jsonl':jsonl_bytes(stable(normalized)),
                        'diagnostics.json':json_bytes(stable(diagnostic))}
            for name,body in computed.items():
                if (directory / arm['name'] / name).read_bytes() != body:
                    raise ValueError('Offline replay changed '+arm['name']+'/'+name)
        if ledger.summary() != accounting:
            raise ValueError('Verifier altered API accounting')
    finally:
        ledger.close()
    after = {p.relative_to(root).as_posix():sha(p.read_bytes()) for p in protected}
    if after != before:
        raise ValueError('Protected experiment files changed')
    result = {'cache_hits':len(requests),'additional_api_attempts':0,
              'all_four_arm_predictions_metrics_and_diagnostics_byte_identical':True,
              'original_records_and_summary_unchanged':True,'accounting_unchanged':True,
              'protected_sha256':before,'test_accessed':False}
    atomic_write(directory / 'verification/offline-replay.json',json_bytes(result))
    print(json.dumps({k:v for k,v in result.items() if k!='protected_sha256'},indent=2))


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
