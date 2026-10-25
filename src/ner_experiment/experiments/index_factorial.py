"""Preregistered 2x2 Dev experiment; all real attempts use the shared ledger."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import json
import os
import subprocess
import threading

from dotenv import dotenv_values
from ner_experiment.data.splitting import read_jsonl, json_bytes, jsonl_bytes, sha
from ner_experiment.evaluation.metrics import evaluate
from ner_experiment.evaluation.index_diagnostics import diagnose
from ner_experiment.experiments.traditional import atomic_write
from ner_experiment.llm.budget import Ledger, reserve_cost
from ner_experiment.llm.client import DeepSeekTransport, execute_request
from ner_experiment.llm.prompts import PARSER_VERSION, request_key, canonical
from ner_experiment.llm.index_prompts import VERSION, build_request

BRANCH = 'codex/cluener-dual-eval-traditional-dev'


def select_ids(dev_ids, excluded, seed, count):
    eligible = [identifier for identifier in dev_ids if identifier not in set(excluded)]
    key = lambda identifier: hashlib.sha256(f'{seed}|index-factorial-v1|{identifier}'.encode()).hexdigest()
    if len(eligible) < count:
        raise ValueError('Not enough unused Dev IDs')
    return sorted(eligible, key=key)[:count]


def assert_branch(root):
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=root).decode().strip()
    if branch != BRANCH:
        raise ValueError('This experiment may only run on the user branch: ' + BRANCH)


def prepare(root):
    root = Path(root)
    assert_branch(root)
    config = json.loads((root / 'configs/index_factorial_v1.json').read_text(encoding='utf-8'))
    manifest_path = root / 'data/splits/experiment_v2.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    frozen = json.loads((root / 'configs/llm_runtime_frozen.json').read_text(encoding='utf-8'))
    if config['sample_count'] != 200 or config['shot_count'] != 4 or config['concurrency'] != 4:
        raise ValueError('Declared sample/shot/concurrency changed')
    if config['model'] != frozen['requested_model']:
        raise ValueError('Model differs from pilot')
    for key in ('temperature', 'thinking', 'response_format', 'max_tokens', 'timeout_seconds', 'max_retries'):
        if config[key] != frozen['parameters'][key]:
            raise ValueError('Runtime differs from pilot: ' + key)
    designs = {(a['representation'], a['explanation']) for a in config['arms']}
    if len(config['arms']) != 4 or designs != {(r, e) for r in ('pairs', 'columns') for e in ('brief', 'explicit')}:
        raise ValueError('Expected complete 2x2 factorial')
    if len({a['name'] for a in config['arms']}) != 4:
        raise ValueError('Duplicate arm name')
    directory = root / 'data/processed/experiment_v2'
    for name in ('dev.jsonl', 'examples_8.jsonl'):
        if sha((directory / name).read_bytes()) != manifest['output_sha256'][name]:
            raise ValueError('Frozen data changed: ' + name)
    dev = read_jsonl(directory / 'dev.jsonl')
    excluded = set(manifest['dev_pilot_ids']) | set(manifest['dev_tune_ids'])
    ids = select_ids([r['id'] for r in dev], excluded, manifest['seed'], config['sample_count'])
    lookup = {row['id']: row for row in dev}
    rows = [lookup[identifier] for identifier in ids]
    examples = read_jsonl(directory / 'examples_8.jsonl')[:config['shot_count']]
    allowed = set(manifest['ids']['train']) - set(manifest['excluded_training_ids'])
    if [r['id'] for r in examples] != manifest['examples_order'][:4] or any(r['id'] not in allowed or not r['bio_compatible'] for r in examples):
        raise ValueError('Examples are not the fixed effective Train prefix')
    if set(ids) & (set(manifest['ids']['train']) | set(manifest['ids']['test']) | excluded):
        raise ValueError('Evaluation ID contamination')
    dotenv = dotenv_values(root / '.env') if (root / '.env').exists() else {}
    credential = os.getenv('DEEPSEEK_API_KEY') or dotenv.get('DEEPSEEK_API_KEY') or ''
    endpoint = (os.getenv('DEEPSEEK_BASE_URL') or dotenv.get('DEEPSEEK_BASE_URL') or 'https://api.deepseek.com').rstrip('/')
    pricing = json.loads((root / config['pricing_file']).read_text(encoding='utf-8'))
    requests = []
    # All four arms for one sentence are interleaved in a seeded order, reducing
    # arm/time confounding. No labels enter target messages.
    for row in rows:
        arms = sorted(config['arms'], key=lambda a: sha(f'{manifest["seed"]}|{row["id"]}|{a["name"]}'.encode()))
        for arm in arms:
            requests.append({'id': row['id'], 'arm': arm['name'],
                             'prompt_version': VERSION + '/' + arm['name'],
                             'request': build_request(row['text'], examples, arm, config)})
    tracked = ['src/ner_experiment/data/splitting.py', 'src/ner_experiment/evaluation/metrics.py',
               'src/ner_experiment/evaluation/index_diagnostics.py', 'src/ner_experiment/llm/budget.py',
               'src/ner_experiment/llm/client.py', 'src/ner_experiment/llm/prompts.py',
               'src/ner_experiment/llm/index_prompts.py', 'src/ner_experiment/experiments/index_factorial.py']
    split_record = {'experiment': config['experiment'], 'seed': manifest['seed'],
                    'selection_rule': config['sample_rule'], 'dev_ids': ids,
                    'example_ids': [r['id'] for r in examples], 'excluded_pilot_and_tune_ids': sorted(excluded),
                    'source_dev_sha256': manifest['output_sha256']['dev.jsonl'], 'test_accessed': False}
    context = {'config': config, 'split_record': split_record, 'endpoint': endpoint,
               'pricing_snapshot': pricing, 'parser_version': PARSER_VERSION,
               'source_sha256': {name: sha((root / name).read_bytes()) for name in tracked},
               'pilot_runtime_frozen_sha256': sha((root / 'configs/llm_runtime_frozen.json').read_bytes()),
               'original_split_manifest_sha256': sha(manifest_path.read_bytes()),
               'requests_sha256': sha(jsonl_bytes(requests)), 'test_accessed': False}
    return config, pricing, rows, requests, context, credential


def freeze_plan(root, config, requests, context):
    root = Path(root)
    directory = root / config['run_directory']
    for path, body in [(directory / 'context.json', json_bytes(context)),
                       (directory / 'requests.jsonl', jsonl_bytes(requests)),
                       (root / 'data/splits/index_factorial_v1.json', json_bytes(context['split_record']))]:
        if path.exists() and path.read_bytes() != body:
            raise ValueError('Preregistered content changed: ' + str(path))
        if not path.exists():
            atomic_write(path, body)
    for name in context['source_sha256']:
        snapshot = directory / 'source_snapshot' / name
        if not snapshot.exists():
            atomic_write(snapshot, (root / name).read_bytes())


def run(root, dry_run=False, offline=False):
    root = Path(root)
    config, pricing, rows, requests, context, credential = prepare(root)
    freeze_plan(root, config, requests, context)
    directory = root / config['run_directory']
    ledger = Ledger(root / config['ledger_path'], config['budget_cny'])
    try:
        baseline = ledger.summary()
        cached_count = sum(ledger.cached(request_key(item['id'], item['request'], context['endpoint'],
                           prompt_version=item['prompt_version'])) is not None for item in requests)
    finally:
        ledger.close()
    if dry_run:
        plan = {'requests': len(requests), 'cached': cached_count, 'new_requests': len(requests)-cached_count,
                'stage_budget_cny': config['stage_budget_cny'], 'accounting_before': baseline,
                'sum_peak_reservations_without_settlements_cny': sum(reserve_cost(i['request'], pricing) for i in requests),
                'context_sha256': sha(json_bytes(context)), 'test_accessed': False}
        atomic_write(directory / 'plan.json', json_bytes(plan))
        print(json.dumps(plan, indent=2)); return plan
    if offline and cached_count != len(requests):
        raise ValueError('Offline replay refuses missing cache')
    if cached_count != len(requests) and not credential:
        raise ValueError('Local API credential is missing; no request sent')
    if baseline['unknown_or_pending_attempts']:
        raise ValueError('Unresolved accounting must be reviewed before new experiment')
    lock_path = root / 'runs/llm/index_experiment.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    local, transports, resource_lock = threading.local(), [], threading.Lock()
    records, stop_reason = {}, None

    def execute(item):
        db = Ledger(root / config['ledger_path'], config['budget_cny'])
        try:
            key = request_key(item['id'], item['request'], context['endpoint'], prompt_version=item['prompt_version'])
            if db.cached(key) is None:
                if offline:
                    raise ValueError('Offline replay encountered a cache miss')
                if pricing['verified_date_shanghai'] != datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat():
                    raise ValueError('Price snapshot is not verified for this request date')
                if not hasattr(local, 'transport'):
                    local.transport = DeepSeekTransport(credential, context['endpoint'], config['timeout_seconds'])
                    with resource_lock:
                        transports.append(local.transport)
            record, cached = execute_request(item['id'], item['request'], context['endpoint'], db,
                  getattr(local, 'transport', None), pricing, config['experiment'], config['stage_budget_cny'],
                  max_retries=config['max_retries'], prompt_version=item['prompt_version'])
            return {**record, 'arm': item['arm'], 'cache_hit_this_invocation': cached}
        finally:
            db.close()

    try:
        with ThreadPoolExecutor(max_workers=config['concurrency']) as pool:
            iterator = iter(requests)
            pending = {pool.submit(execute, item): item for item in [next(iterator) for _ in range(min(config['concurrency'], len(requests)))]}
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    item = pending.pop(future)
                    try:
                        record = future.result()
                    except Exception as exc:
                        # Exception strings may contain SDK data: persist only class.
                        stop_reason = 'worker_error_' + type(exc).__name__
                        continue
                    records[(item['arm'], item['id'])] = record
                    if record.get('fatal'):
                        stop_reason = record['prediction'].get('error_code', 'fatal_request_or_accounting')
                if len(records) % 40 == 0 or not pending:
                    ordered = [records[(i['arm'], i['id'])] for i in requests if (i['arm'], i['id']) in records]
                    atomic_write(directory / 'records.partial.jsonl', jsonl_bytes(ordered))
                    print(f'PROGRESS {len(records)}/{len(requests)}', flush=True)
                if not stop_reason:
                    while len(pending) < config['concurrency']:
                        item = next(iterator, None)
                        if item is None:
                            break
                        pending[pool.submit(execute, item)] = item
        ordered = [records[(i['arm'], i['id'])] for i in requests if (i['arm'], i['id']) in records]
        atomic_write(directory / 'records.jsonl', jsonl_bytes(ordered))
        db = Ledger(root / config['ledger_path'], config['budget_cny'])
        try:
            after = db.summary()
            stage_cost = db.spent(config['experiment'])
        finally:
            db.close()
        if after['unknown_or_pending_attempts']:
            stop_reason = stop_reason or 'unresolved_accounting_reservations'
        complete = len(records) == len(requests) and not stop_reason
        summary = {'experiment': config['experiment'], 'complete': complete, 'stop_reason': stop_reason,
                   'context_sha256': sha(json_bytes(context)), 'samples_per_arm': len(rows),
                   'completed_records': len(records), 'api_attempts_this_invocation': after['attempt_count'] - baseline['attempt_count'],
                   'stage_accounted_cny': stage_cost, 'accounting': after, 'test_accessed': False, 'arms': {}}
        if complete:
            for arm in config['arms']:
                predictions = [records[(arm['name'], row['id'])]['prediction'] for row in rows]
                score = evaluate(rows, predictions)
                normalized = score.pop('normalized_predictions')
                diagnostic = diagnose(rows, predictions)
                folder = directory / arm['name']
                # SQLite caches canonicalize JSON object key order. Normalize
                # that order before first export as well, so cache replay keeps
                # the same bytes without ever changing values or list order.
                stable = lambda value: json.loads(canonical(value))
                outputs = {'predictions.jsonl': jsonl_bytes(stable(predictions)), 'metrics.json': json_bytes(stable(score)),
                           'normalized.jsonl': jsonl_bytes(stable(normalized)), 'diagnostics.json': json_bytes(stable(diagnostic))}
                for name, body in outputs.items():
                    # Stable evaluation artifacts must survive replay byte-for-byte.
                    path = folder / name
                    if path.exists() and path.read_bytes() != body:
                        raise ValueError('Offline evaluation changed: ' + str(path))
                    atomic_write(path, body)
                summary['arms'][arm['name']] = {'token': score['token'], 'entity': score['entity'],
                    'position_error_sample_rate': diagnostic['position_error_sample_rate'],
                    'surface_type_bag': diagnostic['surface_type_bag'], 'rates': score['rates'],
                    'output_sha256': {name: sha(body) for name, body in outputs.items()}}
        atomic_write(directory / 'summary.json', json_bytes(summary))
        print(json.dumps(summary, indent=2), flush=True)
        if not complete:
            raise RuntimeError('Experiment stopped safely; inspect stop_reason and accounting')
        return summary
    finally:
        for transport in transports:
            transport.close()
        lock_path.unlink(missing_ok=True)
