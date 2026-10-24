"""Project future Dev/Test costs from pilot usage; never reads Test text."""
from pathlib import Path
import json
import math
from ner_experiment.data.splitting import read_jsonl, json_bytes, sha
from ner_experiment.experiments.traditional import atomic_write
from ner_experiment.llm.pilot import prepare
from ner_experiment.llm.prompts import canonical, request_for


def main(root):
    root = Path(root)
    config, pricing, pilot_rows, _, _, _ = prepare(root)
    frozen = json.loads((root / 'configs/llm_runtime_frozen.json').read_text(encoding='utf-8'))
    directory = root / config['run_directory']
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    if not summary['runtime_freeze_ready'] or frozen['pilot_context_sha256'] != summary['context_sha256']:
        raise ValueError('Matching successful pilot freeze required')
    manifest = json.loads((root / 'data/splits/experiment_v2.json').read_text(encoding='utf-8'))
    folder = root / 'data/processed/experiment_v2'
    for filename in ('dev_tune.jsonl', 'examples_8.jsonl'):
        if sha((folder / filename).read_bytes()) != manifest['output_sha256'][filename]:
            raise ValueError('Fixed examples or tuning subset hash mismatch')
    tune, examples = read_jsonl(folder / 'dev_tune.jsonl'), read_jsonl(folder / 'examples_8.jsonl')
    allowed = set(manifest['ids']['train']) - set(manifest['excluded_training_ids'])
    if [row['id'] for row in examples] != manifest['examples_order'] or not all(row['id'] in allowed and row['bio_compatible'] for row in examples):
        raise ValueError('Few-shot examples must come only from effective Train')
    records = read_jsonl(directory / 'records.jsonl')
    ratios = [record['raw_response']['usage']['prompt_tokens'] / len(canonical(record['request']).encode('utf-8')) for record in records]
    max_ratio = max(ratios)
    output_tokens = sorted(record['raw_response']['usage']['completion_tokens'] for record in records)
    # Max observed input ratio, 95th-percentile output, 1.5x token safety factor.
    output_projection = math.ceil(output_tokens[math.ceil(.95 * len(output_tokens)) - 1] * 1.5)
    rates = pricing['per_million_tokens']['peak']
    plans = []
    for shots in (0, 2, 4, 8):
        tokens = [math.ceil(len(canonical(request_for(row['text'], config, examples[:shots])).encode('utf-8')) * max_ratio * 1.5) for row in tune]
        cost = (sum(tokens) * rates['input_cache_miss'] + len(tune) * output_projection * rates['output']) / 1_000_000
        plans.append({'shots': shots, 'samples': 300, 'input_token_projection_mean': sum(tokens) / len(tokens),
                      'output_token_projection_per_sample': output_projection, 'peak_uncached_projected_cny': cost})
    max_few_per_call = max(plan['peak_uncached_projected_cny'] / 300 for plan in plans if plan['shots'] > 0)
    zero_per_call = plans[0]['peak_uncached_projected_cny'] / 300
    remaining = sum(plan['peak_uncached_projected_cny'] for plan in plans) - 20 * zero_per_call
    full_dev_new = 1343 - 300
    remaining += (full_dev_new + 1209) * (zero_per_call + max_few_per_call)
    result = {'basis': '20-row v2 actual usage; input max token/byte ratio and output P95, both 1.5x; peak prices; no server cache benefit',
              'estimates_are_not_guarantees': True, 'pilot_total_accounted_cny': summary['accounting']['accounted_cny'],
              'tuning_plans': plans, 'future_calls_without_retries_with_cache_reuse': 5684,
              'projected_remaining_cny_with_cache_reuse': remaining,
              'projected_total_cny': remaining + summary['accounting']['accounted_cny'],
              'budget_cny': 100, 'test_text_accessed': False,
              'test_count_from_manifest': len(manifest['ids']['test'])}
    atomic_write(root / 'reports/llm-budget-projection.json', json_bytes(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
