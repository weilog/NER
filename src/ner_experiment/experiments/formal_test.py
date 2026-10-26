"""Immutable Test gate and one-shot five-method experiment with offline replay."""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import threading
from zoneinfo import ZoneInfo

from dotenv import dotenv_values
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from ner_experiment.data.splitting import json_bytes, jsonl_bytes, read_jsonl, sha
from ner_experiment.evaluation.metrics import evaluate, PROTOCOL
from ner_experiment.evaluation.index_diagnostics import diagnose
from ner_experiment.experiments.index_factorial import assert_branch
from ner_experiment.experiments.llm_dev import stable, unchanged_write
from ner_experiment.experiments.traditional import atomic_write, predict_rows
from ner_experiment.llm.budget import Ledger, now, reserve_cost
from ner_experiment.llm.client import DeepSeekTransport, execute_request
from ner_experiment.llm.prompts import PARSER_VERSION, PROMPT_VERSION, request_for, request_key


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze(root):
    """Build the gate using existing manifests/Dev only; never open Test rows."""
    root = Path(root)
    assert_branch(root)
    config = load(root / 'configs/formal_test_v1.json')
    traditional = load(root / 'configs/traditional_selected.json')
    llm = load(root / 'configs/llm_selected.json')
    manifest = load(root / 'data/splits/experiment_v2.json')
    if config['sample_count'] != 1209 or llm['selected_few_shot_count'] != 4:
        raise ValueError('Declared Test design differs from completed Dev selection')
    if llm['prompt_version'] != PROMPT_VERSION or llm['parser_version'] != PARSER_VERSION:
        raise ValueError('Frozen LLM protocol changed')
    if len(manifest['ids']['test']) != 1209 or len(set(manifest['ids']['test'])) != 1209:
        raise ValueError('Test IDs do not satisfy the audited manifest')
    if set(manifest['ids']['test']) & (set(manifest['ids']['train']) | set(manifest['ids']['dev'])):
        raise ValueError('Split IDs overlap')
    for name, digest in llm['source_sha256'].items():
        if sha((root / name).read_bytes()) != digest:
            raise ValueError('Source changed since Dev: ' + name)
    for chosen in traditional['selected'].values():
        if sha((root / chosen['model_path']).read_bytes()) != chosen['model_sha256']:
            raise ValueError('Dev-selected model changed')
    dev_summary = root / 'runs/llm/dev_v1/full/summary.json'
    if sha(dev_summary.read_bytes()) != llm['full_dev_summary_sha256'] or not load(dev_summary)['complete']:
        raise ValueError('Completed Dev evidence changed')
    names = ['configs/formal_test_v1.json', 'configs/llm_selected.json',
        'configs/traditional_selected.json', 'data/splits/experiment_v2.json', config['pricing_file']]
    names += [p.relative_to(root).as_posix() for p in sorted((root / 'src/ner_experiment').rglob('*.py'))]
    names += ['scripts/run_formal_test.py', 'scripts/analyze_formal_test.py']
    gate = {'protocol': PROTOCOL, 'config': config, 'traditional': traditional, 'llm': llm,
        'test_ids': manifest['ids']['test'], 'seed': manifest['seed'],
        'input_sha256': {n: manifest['output_sha256'][n] for n in ('test.jsonl','train_valid.jsonl','dev.jsonl','examples_8.jsonl')},
        'source_sha256': {n: sha((root / n).read_bytes()) for n in names},
        'best_traditional_selected_on_dev': traditional['best_traditional_by_dev'],
        'test_predictions_or_scores_used_for_selection': False}
    unchanged_write(root / 'configs/test_frozen_v1.json', json_bytes(gate))
    print(json.dumps({'frozen': True, 'gate_sha256': sha(json_bytes(gate)), 'test_rows_opened': False}, indent=2))
    return gate


def verify_gate(root):
    root = Path(root)
    assert_branch(root)
    gate_path = root / 'configs/test_frozen_v1.json'
    gate = load(gate_path)
    # Require the exact gate in HEAD, so a local uncommitted rewrite cannot
    # silently choose new models or change the protocol after seeing results.
    committed = subprocess.check_output(['git','show','HEAD:configs/test_frozen_v1.json'], cwd=root)
    if committed != gate_path.read_bytes():
        raise ValueError('Test freeze must be committed before accessing Test')
    for name, digest in gate['source_sha256'].items():
        if sha((root / name).read_bytes()) != digest:
            raise ValueError('Frozen source/config changed: ' + name)
    for chosen in gate['traditional']['selected'].values():
        if sha((root / chosen['model_path']).read_bytes()) != chosen['model_sha256']:
            raise ValueError('Frozen model changed')
    return gate


def prepare(root):
    root = Path(root)
    gate = verify_gate(root)
    config = gate['config']
    data = root / 'data/processed/experiment_v2'
    for name in ('test.jsonl', 'examples_8.jsonl'):
        if sha((data / name).read_bytes()) != gate['input_sha256'][name]:
            raise ValueError('Frozen input changed: ' + name)
    rows = read_jsonl(data / 'test.jsonl')
    examples = read_jsonl(data / 'examples_8.jsonl')[:gate['llm']['selected_few_shot_count']]
    if [r['id'] for r in rows] != gate['test_ids'] or len(rows) != config['sample_count']:
        raise ValueError('Frozen Test IDs changed')
    if any(not r['bio_compatible'] or not r['has_labels'] for r in rows):
        raise ValueError('Test contains incompatible/unlabeled gold')
    if [r['id'] for r in examples] != gate['llm']['example_ids']:
        raise ValueError('Frozen Train prefix changed')
    parameters = gate['llm']['parameters']
    requests = []
    for row in rows:
        for shots in sorted([0,4], key=lambda n: sha(f'{gate["seed"]}|formal-test-v1|{row["id"]}|{n}'.encode())):
            requests.append({'id': row['id'], 'method': 'llm_zero' if shots == 0 else 'llm_4_shot',
                'request': request_for(row['text'], parameters, examples[:shots])})
    env = dotenv_values(root / '.env') if (root / '.env').exists() else {}
    credential = os.getenv('DEEPSEEK_API_KEY') or env.get('DEEPSEEK_API_KEY') or ''
    endpoint = (os.getenv('DEEPSEEK_BASE_URL') or env.get('DEEPSEEK_BASE_URL') or 'https://api.deepseek.com').rstrip('/')
    context = {'gate_sha256': sha((root / 'configs/test_frozen_v1.json').read_bytes()),
        'requests_sha256': sha(jsonl_bytes(requests)), 'endpoint': endpoint,
        'input_sha256': gate['input_sha256'], 'test_ids': gate['test_ids']}
    return gate, rows, requests, context, credential


def request_batch(root, gate, requests, context, credential, offline):
    config, parameters = gate['config'], gate['llm']['parameters']
    pricing = load(root / config['pricing_file'])
    local, transports, resources = threading.local(), [], threading.Lock()
    records, stop = {}, None
    directory = root / config['run_directory']

    def execute(item):
        db = Ledger(root / config['ledger_path'], config['budget_cny'])
        try:
            key = request_key(item['id'], item['request'], context['endpoint'])
            if db.cached(key) is None:
                if offline:
                    raise ValueError('Offline mode encountered missing cache')
                if pricing['verified_date_shanghai'] != datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat():
                    raise ValueError('Reverify pricing for request date')
                if not hasattr(local, 'transport'):
                    local.transport = DeepSeekTransport(credential, context['endpoint'], parameters['timeout_seconds'])
                    with resources:
                        transports.append(local.transport)
            record, _ = execute_request(item['id'], item['request'], context['endpoint'], db,
                getattr(local, 'transport', None), pricing, config['experiment'], config['stage_budget_cny'],
                max_retries=parameters['max_retries'])
            return record
        finally:
            db.close()
    try:
        with ThreadPoolExecutor(max_workers=parameters['concurrency']) as pool:
            iterator, pending = iter(requests), {}
            while True:
                if not stop:
                    while len(pending) < parameters['concurrency']:
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
                        stop = 'worker_error_' + type(exc).__name__
                        continue
                    records[(item['method'], item['id'])] = record
                    if record.get('fatal'):
                        stop = record['prediction'].get('error_code', 'fatal_api_or_accounting')
                if len(records) % 40 == 0 or not pending:
                    ordered = [{**records[(i['method'], i['id'])], 'method': i['method']}
                        for i in requests if (i['method'], i['id']) in records]
                    atomic_write(directory / 'records.partial.jsonl', jsonl_bytes(stable(ordered)))
                    print(f'TEST {len(records)}/{len(requests)}', flush=True)
        if stop or len(records) != len(requests):
            atomic_write(directory / 'stop.json', json_bytes({'stop_reason': stop, 'completed_records': len(records)}))
            raise RuntimeError('Test safely stopped: ' + str(stop))
        return records
    finally:
        for transport in transports:
            transport.close()


def run(root, dry_run=False, offline=False):
    root = Path(root)
    gate, rows, requests, context, credential = prepare(root)
    config = gate['config']
    directory = root / config['run_directory']
    unchanged_write(directory / 'context.json', json_bytes(context))
    unchanged_write(directory / 'requests.jsonl', jsonl_bytes(requests))
    for name in gate['source_sha256']:
        unchanged_write(directory / 'source_snapshot' / name, (root / name).read_bytes())
    db = Ledger(root / config['ledger_path'], config['budget_cny'])
    try:
        before = db.summary()
        cached = sum(db.cached(request_key(i['id'], i['request'], context['endpoint'])) is not None for i in requests)
    finally:
        db.close()
    if dry_run:
        plan = {'test_samples': len(rows), 'llm_requests': len(requests), 'cached': cached,
            'new_requests': len(requests)-cached, 'stage_budget_cny': config['stage_budget_cny'],
            'accounting_before': before, 'no_predictions_or_scores_generated': True,
            'all_requests_peak_reservation_sum_cny': sum(reserve_cost(i['request'], load(root / config['pricing_file'])) for i in requests)}
        atomic_write(directory / 'plan.json', json_bytes(plan)); print(json.dumps(plan, indent=2)); return plan
    if offline and cached != len(requests):
        raise ValueError('Offline mode refuses missing cache before predicting/scoring')
    if before['unknown_or_pending_attempts'] or (cached != len(requests) and not credential):
        raise ValueError('Unresolved accounting or missing credential')
    lock = directory / 'experiment.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        unchanged_write(directory / 'started.json', json_bytes(load(directory / 'started.json') if (directory / 'started.json').exists()
            else {'started_at': now(), 'gate_sha256': context['gate_sha256'],
                  'preregistered_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root).decode().strip()}))
        for family, chosen in gate['traditional']['selected'].items():
            folder = directory / family
            predictions_path = folder / 'predictions.jsonl'
            manifest_path = folder / 'prediction-manifest.json'
            if predictions_path.exists():
                if not manifest_path.exists() or sha(predictions_path.read_bytes()) != load(manifest_path)['sha256']:
                    raise ValueError('Saved traditional predictions failed integrity check')
            else:
                model = joblib.load(root / chosen['model_path'])
                with threadpool_limits(limits=1):
                    predictions = predict_rows(model, rows, family)
                unchanged_write(predictions_path, jsonl_bytes(stable(predictions)))
                unchanged_write(manifest_path, json_bytes({'sha256': sha(predictions_path.read_bytes()),
                    'gate_sha256': context['gate_sha256'], 'model_sha256': chosen['model_sha256']}))
        records = request_batch(root, gate, requests, context, credential, offline)
        ordered = [{**records[(i['method'], i['id'])], 'method': i['method']} for i in requests]
        unchanged_write(directory / 'records.jsonl', jsonl_bytes(stable(ordered)))
        for method in ('llm_zero','llm_4_shot'):
            predictions = [records[(method,r['id'])]['prediction'] for r in rows]
            unchanged_write(directory / method / 'predictions.jsonl', jsonl_bytes(stable(predictions)))
        db = Ledger(root / config['ledger_path'], config['budget_cny'])
        try:
            after, spent = db.summary(), db.spent(config['experiment'])
        finally:
            db.close()
        if after['unknown_or_pending_attempts']:
            raise ValueError('Unresolved accounting; no completed Test report')
        # No gold-based scores are generated until all five prediction files exist.
        summary = {'complete': True, 'stage': config['experiment'], 'context_sha256': sha(json_bytes(context)),
            'gate_sha256': context['gate_sha256'], 'samples': len(rows), 'stage_accounted_cny': spent,
            'accounting_at_first_export': after, 'methods': {}}
        for method in config['methods']:
            folder = directory / method
            predictions = read_jsonl(folder / 'predictions.jsonl')
            metrics = evaluate(rows, predictions)
            normalized = metrics.pop('normalized_predictions')
            diagnostic = diagnose(rows, predictions)
            outputs = {'metrics.json': json_bytes(stable(metrics)), 'normalized.jsonl': jsonl_bytes(stable(normalized)),
                'diagnostics.json': json_bytes(stable(diagnostic))}
            for name, body in outputs.items():
                unchanged_write(folder / name, body)
            times = ([records[(method,r['id'])]['network_seconds'] for r in rows] if method.startswith('llm_')
                     else [r['elapsed_seconds'] for r in predictions])
            stats = {'token': metrics['token'], 'entity': metrics['entity'], 'rates': metrics['rates'],
                'position_error_sample_rate': diagnostic['position_error_sample_rate'],
                'latency_median_seconds': float(np.median(times)), 'latency_p95_seconds': float(np.percentile(times,95)),
                'latency_scope': 'network including retries excluding backoff' if method.startswith('llm_') else 'local features and prediction',
                'output_sha256': {name: sha((folder/name).read_bytes()) for name in ('predictions.jsonl',*outputs)}}
            if method.startswith('llm_'):
                own = [records[(method,r['id'])] for r in rows]
                usage = Counter()
                for record in own:
                    for attempt in record['attempts']:
                        detail = json.loads(attempt['cost_detail_json'] or '{}')
                        usage.update({k:detail.get(k,0) for k in ('prompt_tokens','completion_tokens','cache_hit_tokens','cache_miss_tokens')})
                stats.update(usage=dict(usage), response_models=sorted({r['response_model'] for r in own if r.get('response_model')}),
                    system_fingerprints=sorted({r['system_fingerprint'] for r in own if r.get('system_fingerprint')}),
                    accounted_cny=sum(r['accounted_cny'] for r in own))
            summary['methods'][method] = stats
        summary_path = directory / 'summary.json'
        if summary_path.exists():
            summary['accounting_at_first_export'] = load(summary_path)['accounting_at_first_export']
        unchanged_write(summary_path, json_bytes(summary))
        print(json.dumps({'complete': True, 'new_api_attempts': after['attempt_count']-before['attempt_count'],
            'stage_accounted_cny': spent, 'global_accounted_cny': after['accounted_cny'],
            'methods': {n: {'token_f1':a['token']['f1'],'entity_f1':a['entity']['f1']} for n,a in summary['methods'].items()}},indent=2),flush=True)
        return summary
    finally:
        lock.unlink(missing_ok=True)
