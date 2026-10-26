"""Fixed-prefix shot selection on 300 Dev rows, then full Dev validation.

Reuses the frozen pilot prompt and shared durable per-request ledger. This
module never opens the project Test data or changes the frozen prompt modules.
"""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import json
import os
from pathlib import Path
import threading
from zoneinfo import ZoneInfo

from dotenv import dotenv_values
import numpy as np

from ner_experiment.data.splitting import json_bytes, jsonl_bytes, read_jsonl, sha
from ner_experiment.evaluation.metrics import evaluate
from ner_experiment.evaluation.index_diagnostics import diagnose
from ner_experiment.experiments.index_factorial import assert_branch
from ner_experiment.experiments.traditional import atomic_write
from ner_experiment.llm.budget import Ledger, reserve_cost
from ner_experiment.llm.client import DeepSeekTransport, execute_request
from ner_experiment.llm.prompts import PARSER_VERSION, PROMPT_VERSION, canonical, request_for, request_key


def stable(value):
    return json.loads(canonical(value))


def unchanged_write(path, body):
    path = Path(path)
    if path.exists() and path.read_bytes() != body:
        raise ValueError('Frozen artifact changed: ' + str(path))
    if not path.exists():
        atomic_write(path, body)


def choose_shots(arms):
    candidates = [arms[f'shot_{n}'] for n in (2, 4, 8)]
    return min(candidates, key=lambda a: (-a['entity']['f1'],
               a['latency_mean_seconds'], (2, 4, 8).index(a['shots'])))['shots']


def prepare(root, phase):
    root = Path(root)
    assert_branch(root)
    if phase not in ('tune', 'full'):
        raise ValueError('Expected tune or full Dev phase')
    config = json.loads((root / 'configs/llm_dev_v1.json').read_text(encoding='utf-8'))
    manifest = json.loads((root / 'data/splits/experiment_v2.json').read_text(encoding='utf-8'))
    frozen = json.loads((root / 'configs/llm_runtime_frozen.json').read_text(encoding='utf-8'))
    if config['shots'] != [0, 2, 4, 8] or config['tune_samples'] != 300 or config['full_dev_samples'] != 1343:
        raise ValueError('Frozen search design changed')
    if config['prompt_version'] != PROMPT_VERSION or config['prompt_version'] != frozen['prompt_version']:
        raise ValueError('Prompt differs from the frozen pilot')
    for key in ('temperature', 'thinking', 'response_format', 'max_tokens', 'timeout_seconds', 'max_retries'):
        if config[key] != frozen['parameters'][key]:
            raise ValueError('Runtime differs from frozen pilot: ' + key)
    if config['model'] != frozen['requested_model']:
        raise ValueError('Model differs from frozen pilot')
    directory = root / 'data/processed/experiment_v2'
    names = ['dev.jsonl', 'dev_tune.jsonl', 'examples_8.jsonl']
    for name in names:
        if sha((directory / name).read_bytes()) != manifest['output_sha256'][name]:
            raise ValueError('Frozen data changed: ' + name)
    dev = read_jsonl(directory / 'dev.jsonl')
    tune = read_jsonl(directory / 'dev_tune.jsonl')
    examples = read_jsonl(directory / 'examples_8.jsonl')
    if len(dev) != 1343 or len(tune) != 300 or len(examples) != 8:
        raise ValueError('Unexpected data counts')
    if [r['id'] for r in dev] != manifest['ids']['dev'] or [r['id'] for r in tune] != manifest['dev_tune_ids']:
        raise ValueError('Dev IDs differ from manifest')
    if [r['id'] for r in examples] != manifest['examples_order']:
        raise ValueError('Example order differs from frozen prefix')
    allowed = set(manifest['ids']['train']) - set(manifest['excluded_training_ids'])
    dev_lookup = {r['id']: r for r in dev}
    if any(r != dev_lookup.get(r['id']) for r in tune):
        raise ValueError('Tuning rows differ from full Dev')
    if any(r['id'] not in allowed or not r['bio_compatible'] for r in examples):
        raise ValueError('Examples are not effective Train rows')
    if {r['text'] for r in examples} & {r['text'] for r in dev}:
        raise ValueError('Example text leaks into Dev')
    if any(not r['bio_compatible'] or not r['has_labels'] for r in dev):
        raise ValueError('Dev must contain compatible labeled rows')
    selection = None
    shots = config['shots']
    if phase == 'full':
        selection_path = root / config['run_directory'] / 'selection.json'
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        tune_context = root / config['run_directory'] / 'tune/context.json'
        tune_summary = root / config['run_directory'] / 'tune/summary.json'
        summary = json.loads(tune_summary.read_text(encoding='utf-8'))
        if (not summary['complete'] or selection['tune_context_sha256'] != sha(tune_context.read_bytes())
                or selection['tune_summary_sha256'] != sha(tune_summary.read_bytes())
                or selection['selected_shots'] != choose_shots(summary['arms'])):
            raise ValueError('Selection is not backed by completed frozen tuning')
        if json.loads(tune_context.read_text(encoding='utf-8'))['config'] != config:
            raise ValueError('Configuration changed after tuning')
        shots = [0, selection['selected_shots']]
    rows = tune if phase == 'tune' else dev
    env = dotenv_values(root / '.env') if (root / '.env').exists() else {}
    credential = os.getenv('DEEPSEEK_API_KEY') or env.get('DEEPSEEK_API_KEY') or ''
    endpoint = (os.getenv('DEEPSEEK_BASE_URL') or env.get('DEEPSEEK_BASE_URL') or 'https://api.deepseek.com').rstrip('/')
    pricing = json.loads((root / config['pricing_file']).read_text(encoding='utf-8'))
    requests = []
    for row in rows:
        order = sorted(shots, key=lambda n: sha(f'{manifest["seed"]}|llm-dev-v1|{row["id"]}|{n}'.encode()))
        for n in order:
            requests.append({'id': row['id'], 'shots': n, 'arm': f'shot_{n}',
                             'request': request_for(row['text'], config, examples[:n])})
    source_names = ['src/ner_experiment/experiments/llm_dev.py',
        'src/ner_experiment/experiments/index_factorial.py',
        'src/ner_experiment/experiments/traditional.py', 'src/ner_experiment/data/splitting.py',
        'src/ner_experiment/evaluation/metrics.py', 'src/ner_experiment/evaluation/index_diagnostics.py',
        'src/ner_experiment/llm/prompts.py', 'src/ner_experiment/llm/client.py', 'src/ner_experiment/llm/budget.py']
    context = {'experiment': config['experiment'], 'phase': phase, 'config': config,
        'endpoint': endpoint, 'pricing_snapshot': pricing, 'prompt_version': PROMPT_VERSION,
        'parser_version': PARSER_VERSION, 'ids': [r['id'] for r in rows], 'shots': shots,
        'example_ids': [r['id'] for r in examples], 'selection': selection,
        'input_sha256': {name: manifest['output_sha256'][name] for name in names},
        'source_sha256': {name: sha((root / name).read_bytes()) for name in source_names},
        'requests_sha256': sha(jsonl_bytes(requests)), 'test_accessed': False}
    return config, pricing, rows, requests, context, credential


def run(root, phase, dry_run=False, offline=False):
    root = Path(root)
    config, pricing, rows, requests, context, credential = prepare(root, phase)
    directory = root / config['run_directory'] / phase
    unchanged_write(directory / 'context.json', json_bytes(context))
    unchanged_write(directory / 'requests.jsonl', jsonl_bytes(requests))
    for name in context['source_sha256']:
        unchanged_write(directory / 'source_snapshot' / name, (root / name).read_bytes())
    db = Ledger(root / config['ledger_path'], config['budget_cny'])
    try:
        before = db.summary()
        cached = sum(db.cached(request_key(i['id'], i['request'], context['endpoint'])) is not None for i in requests)
    finally:
        db.close()
    plan = {'phase': phase, 'requests': len(requests), 'cached': cached,
        'new_requests': len(requests)-cached, 'context_sha256': sha(json_bytes(context)),
        'stage_budget_cny': config['stage_budget_cny'], 'accounting_before': before,
        'all_requests_peak_reservation_sum_cny': sum(reserve_cost(i['request'], pricing) for i in requests),
        'test_accessed': False}
    if dry_run:
        atomic_write(directory / 'plan.json', json_bytes(plan))
        print(json.dumps(plan, indent=2), flush=True)
        return plan
    if offline and cached != len(requests):
        raise ValueError('Offline mode refuses any missing cache')
    if cached != len(requests) and not credential:
        raise ValueError('Missing local credential; no request sent')
    if before['unknown_or_pending_attempts']:
        raise ValueError('Unresolved accounting; no new experiment')
    lock = root / 'runs/llm/dev_experiment.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    local, transports, resource_lock = threading.local(), [], threading.Lock()
    records, stop_reason = {}, None
    stage = config['experiment'] + '/' + phase

    def execute(item):
        ledger = Ledger(root / config['ledger_path'], config['budget_cny'])
        try:
            key = request_key(item['id'], item['request'], context['endpoint'])
            if ledger.cached(key) is None:
                if offline:
                    raise ValueError('Offline cache miss')
                if pricing['verified_date_shanghai'] != datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat():
                    raise ValueError('Price verification does not match request date')
                if not hasattr(local, 'transport'):
                    local.transport = DeepSeekTransport(credential, context['endpoint'], config['timeout_seconds'])
                    with resource_lock:
                        transports.append(local.transport)
            record, _ = execute_request(item['id'], item['request'], context['endpoint'], ledger,
                getattr(local, 'transport', None), pricing, stage, config['stage_budget_cny'],
                max_retries=config['max_retries'])
            return record
        finally:
            ledger.close()

    try:
        with ThreadPoolExecutor(max_workers=config['concurrency']) as pool:
            iterator = iter(requests)
            pending = {}
            while True:
                if not stop_reason:
                    while len(pending) < config['concurrency']:
                        item = next(iterator, None)
                        if item is None:
                            break
                        pending[pool.submit(execute, item)] = item
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    item = pending.pop(future)
                    try:
                        record = future.result()
                    except Exception as exc:
                        stop_reason = 'worker_error_' + type(exc).__name__
                        continue
                    records[(item['arm'], item['id'])] = record
                    if record.get('fatal'):
                        stop_reason = record['prediction'].get('error_code', 'fatal_request_or_accounting')
                if len(records) % 40 == 0 or not pending:
                    ordered = [{**records[(i['arm'], i['id'])], 'arm': i['arm'], 'shots': i['shots']}
                               for i in requests if (i['arm'], i['id']) in records]
                    atomic_write(directory / 'records.partial.jsonl', jsonl_bytes(stable(ordered)))
                    print(f'{phase.upper()} {len(records)}/{len(requests)}', flush=True)
        ordered = [{**records[(i['arm'], i['id'])], 'arm': i['arm'], 'shots': i['shots']}
                   for i in requests if (i['arm'], i['id']) in records]
        ledger = Ledger(root / config['ledger_path'], config['budget_cny'])
        try:
            after, spent = ledger.summary(), ledger.spent(stage)
        finally:
            ledger.close()
        if after['unknown_or_pending_attempts']:
            stop_reason = stop_reason or 'unresolved_accounting'
        complete = len(records) == len(requests) and not stop_reason
        if complete:
            unchanged_write(directory / 'records.jsonl', jsonl_bytes(stable(ordered)))
        else:
            atomic_write(directory / 'records.incomplete.jsonl', jsonl_bytes(stable(ordered)))
        summary = {'phase': phase, 'complete': complete, 'stop_reason': stop_reason,
            'context_sha256': sha(json_bytes(context)), 'samples_per_arm': len(rows),
            'completed_records': len(records), 'stage_accounted_cny': spent,
            'accounting_at_first_export': after, 'arms': {}, 'test_accessed': False}
        if complete:
            for shots in context['shots']:
                name = f'shot_{shots}'
                arm_records = [records[(name, r['id'])] for r in rows]
                predictions = [r['prediction'] for r in arm_records]
                metrics = evaluate(rows, predictions)
                normalized = metrics.pop('normalized_predictions')
                diagnostic = diagnose(rows, predictions)
                outputs = {'predictions.jsonl': jsonl_bytes(stable(predictions)),
                    'normalized.jsonl': jsonl_bytes(stable(normalized)),
                    'metrics.json': json_bytes(stable(metrics)), 'diagnostics.json': json_bytes(stable(diagnostic))}
                for filename, body in outputs.items():
                    unchanged_write(directory / name / filename, body)
                times = [r['network_seconds'] for r in arm_records]
                usage = Counter()
                for record in arm_records:
                    for attempt in record['attempts']:
                        detail = json.loads(attempt['cost_detail_json'] or '{}')
                        usage.update({k: detail.get(k, 0) for k in ('prompt_tokens', 'completion_tokens', 'cache_hit_tokens', 'cache_miss_tokens')})
                summary['arms'][name] = {'shots': shots, 'token': metrics['token'], 'entity': metrics['entity'],
                    'rates': metrics['rates'], 'position_error_sample_rate': diagnostic['position_error_sample_rate'],
                    'surface_type_bag': diagnostic['surface_type_bag'],
                    'latency_mean_seconds': float(np.mean(times)),
                    'latency_median_seconds': float(np.median(times)), 'latency_p95_seconds': float(np.percentile(times, 95)),
                    'latency_scope': 'original request network time including retries, excluding backoff; reused caches retain original times',
                    'usage': dict(usage), 'response_models': sorted({r['response_model'] for r in arm_records if r.get('response_model')}),
                    'system_fingerprints': sorted({r['system_fingerprint'] for r in arm_records if r.get('system_fingerprint')}),
                    'unique_prediction_cost_cny_including_reused_requests': sum(r['accounted_cny'] for r in arm_records),
                    'output_sha256': {filename: sha(body) for filename, body in outputs.items()}}
        summary_path = directory / 'summary.json'
        # First export is immutable; later shared-ledger spending cannot change
        # the recorded historical accounting context of this completed phase.
        if summary_path.exists() and complete:
            old = json.loads(summary_path.read_text(encoding='utf-8'))
            if old['complete']:
                summary['accounting_at_first_export'] = old['accounting_at_first_export']
                unchanged_write(summary_path, json_bytes(summary))
            else:
                atomic_write(summary_path, json_bytes(summary))
        else:
            atomic_write(summary_path, json_bytes(summary))
        if not complete:
            raise RuntimeError('Dev run stopped safely: ' + str(stop_reason))
        if phase == 'tune':
            selection = {'selected_shots': choose_shots(summary['arms']), 'selection_rule': config['selection'],
                'tune_context_sha256': sha((directory / 'context.json').read_bytes()),
                'tune_summary_sha256': sha(summary_path.read_bytes()), 'test_accessed': False}
            unchanged_write(root / config['run_directory'] / 'selection.json', json_bytes(selection))
        print(json.dumps({'phase': phase, 'complete': complete,
            'new_api_attempts': after['attempt_count']-before['attempt_count'], 'stage_accounted_cny': spent,
            'global_accounted_cny': after['accounted_cny'],
            'arms': {n: {'token_f1': a['token']['f1'], 'entity_f1': a['entity']['f1']} for n,a in summary['arms'].items()}}, indent=2), flush=True)
        return summary
    finally:
        for transport in transports:
            transport.close()
        lock.unlink(missing_ok=True)
