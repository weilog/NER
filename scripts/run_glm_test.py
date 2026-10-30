"""Explicit GLM full-Test extension. Default is an entirely offline preflight."""
import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import math
import os
from pathlib import Path
import sys

import numpy as np
from ner_experiment.common import (
    ROOT, file_hash, load_json, object_hash, project_data, read_jsonl, save_json,
    utc_now, valid_train, write_jsonl,
)
from ner_experiment.evaluation.metrics import PROTOCOL_VERSION, checked_entities, evaluate
from ner_experiment.models.glm import GLMClient, InterfaceStop, atomic_json, settings
from ner_experiment.models.llm import BudgetStop
from pilot_shot_budget import SYSTEM_TEXT, pilot_messages
from run_baselines import verify_data


def build_context(config):
    manifest, _ = verify_data(ROOT)
    if PROTOCOL_VERSION != 'v3':
        raise RuntimeError('GLM comparison requires the unchanged V3 evaluator')
    if config['shots'] != [0, 2, 4, 8]:
        raise RuntimeError('Keep all four predeclared shot settings')
    by_id = {r['id']: r for r in valid_train()}
    examples = [by_id[i] for i in config['example_ids']]
    if len(by_id) != 9538 or len(examples) != 8 or len(set(config['example_ids'])) != 8:
        raise RuntimeError('Effective Train/examples differ from frozen experiment')
    test = project_data('test')
    if len(test) != 1209 or len({r['id'] for r in test}) != 1209:
        raise RuntimeError('Use the entire fixed 1209-sentence project Test')
    if {r['text'] for r in examples} & {r['text'] for r in test}:
        raise RuntimeError('Train examples overlap Test')
    if config.get('system_sha256') != object_hash(SYSTEM_TEXT):
        raise RuntimeError('System prompt differs from frozen comparison prompt')
    if config.get('examples_sha256') != object_hash(examples):
        raise RuntimeError('Few-shot examples differ from frozen reference')
    return manifest, examples, test


def freeze_once(base, protocol):
    path = base / 'frozen_protocol.json'
    if path.exists():
        record = load_json(path)
        if record['protocol'] != protocol:
            migrated = migrated_freeze(base)
            if migrated is not None and migrated['protocol'] == protocol:
                return migrated
            raise RuntimeError('Frozen GLM configuration changed; preserve this run, use a new directory')
        return record
    if base.exists() and any(base.iterdir()):
        raise RuntimeError('Nonempty output without a verified freeze is protected')
    record = {'frozen_at': utc_now(), 'protocol': protocol}
    atomic_json(path, record)
    return record


def migrated_freeze(base):
    """Explicit scheduling-only migration; original freeze and all cached records stay intact."""
    audit = base / 'scheduling_migration_v1'
    path = audit / 'migration.json'
    if not path.exists():
        return None
    migration = load_json(path)
    if file_hash(audit / 'before_manifest.json') != migration['before_manifest_sha256']:
        raise RuntimeError('Scheduling migration baseline manifest changed')
    original = base / 'frozen_protocol.json'
    active = base / 'execution_protocol_v2.json'
    if file_hash(original) != migration['original_freeze_sha256'] or file_hash(active) != migration['active_freeze_sha256']:
        raise RuntimeError('Scheduling migration freeze hash mismatch')
    old, new = load_json(original)['protocol'], load_json(active)['protocol']
    def semantics(protocol):
        return {k: v for k, v in protocol.items() if k not in {'workers', 'source_sha256'}}
    if semantics(old) != semantics(new) or old.get('workers', 1) != 1 or new.get('workers') != 4:
        raise RuntimeError('Migration may change scheduling from 1 to 4 only; model/data/prompt/budget must stay fixed')
    old_sources = {k.replace('\\', '/'): v for k, v in old['source_sha256'].items()}
    new_sources = {k.replace('\\', '/'): v for k, v in new['source_sha256'].items()}
    changed_sources = {k for k in old_sources.keys() | new_sources.keys()
                       if old_sources.get(k) != new_sources.get(k)}
    allowed = {'scripts/run_glm_test.py', 'src/ner_experiment/models/glm.py'}
    if not changed_sources <= allowed:
        raise RuntimeError('Migration changed unrelated experimental source files')
    snapshot = load_json(audit / 'before_manifest.json')
    for name, digest in snapshot['files'].items():
        if name.replace('\\', '/').startswith('cache/') and file_hash(base / name) != digest:
            raise RuntimeError('Preserved serial cache changed')
    return load_json(active)


def accepted_failures(base):
    if migrated_freeze(base) is None:
        return []
    audit = base / 'scheduling_migration_v1'
    migration = load_json(audit / 'migration.json')
    expected = {name.replace('\\', '/'): digest for name, digest
                in load_json(audit / 'before_manifest.json')['files'].items()}
    keys = migration.get('retained_http400_failure_keys', [])
    for key in keys:
        name = f'cache/{key}.json'
        if name not in expected or file_hash(base / name) != expected[name]:
            raise RuntimeError('Retained failure was not in preserved serial cache')
        prediction = load_json(base / name)['prediction']
        if prediction['status'] != 'api_error' or prediction.get('error', {}).get('http_status') != 400:
            raise RuntimeError('Only the explicitly audited historical HTTP400 can be retained without a batch stop')
    return keys


def prior_project_cost(base):
    # Archived duplicates are not scanned; only live ledgers beneath runs/ count.
    total = 0.0
    for path in (ROOT / 'runs').rglob('ledger.json'):
        if path.resolve() == (base / 'ledger.json').resolve():
            continue
        for entry in load_json(path)['attempts']:
            value = entry['charged_upper_cny']
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RuntimeError('Invalid existing project ledger; review before any call')
            total += value
    return total


def predict_rows(client, rows, examples, workers, progress=None):
    """At most workers active futures; preserve input order and drain in-flight responses on stop."""
    if not 1 <= workers <= 4 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Use 1..4 workers and unique row IDs')
    results = [None] * len(rows)
    iterator = iter(enumerate(rows))
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = {}
    count = 0
    def fill():
        while len(pending) < workers:
            try:
                index, row = next(iterator)
            except StopIteration:
                return
            pending[pool.submit(client.predict, row, examples)] = index
    try:
        fill()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda f: pending[f]):
                index = pending.pop(future)
                results[index] = future.result()
                count += 1
                if progress:
                    progress(count)
            fill()
    except BaseException:
        client.stop_event.set()
        for future in pending:
            future.cancel()
        raise
    finally:
        # Running requests finish and cache their responses before process lock is released.
        pool.shutdown(wait=True, cancel_futures=True)
    return results


def run_group(client, rows, examples, folder, protocol_hash, workers=1):
    """Bounded parallel calls; resume only from verified individual request caches."""
    done = folder / 'run.json'
    if done.exists():
        run = load_json(done)
        if run['protocol_sha256'] != protocol_hash:
            raise RuntimeError('GLM group protocol differs')
        for name, digest in run['artifact_sha256'].items():
            if file_hash(folder / name) != digest:
                raise RuntimeError('Damaged GLM result artifact; no requests sent')
        predictions = read_jsonl(folder / 'predictions.jsonl')
        metrics, _, _ = evaluate(rows, predictions)
        if metrics != load_json(folder / 'metrics.json'):
            raise RuntimeError('Saved GLM score differs from offline re-evaluation')
        print('CACHED', folder.name, flush=True)
        return load_json(folder / 'diagnostic_summary.json')

    if folder.exists() and any(folder.iterdir()):
        # Full group files are written only after requests; interrupted finalization is audit-only.
        raise RuntimeError('Partial group artifacts: preserve and inspect; request cache is retained')
    def progress(index):
        if index == 1 or index % 20 == 0:
            print(folder.name, index, '/', len(rows), 'workers=', workers,
                  'GLM ledger upper CNY=', round(client.total(), 4), flush=True)
    predictions = predict_rows(client, rows, examples, workers, progress)
    metrics, counts, cases = evaluate(rows, predictions)
    mismatch = valid_total = reasoning_samples = 0
    truncations = 0
    keys = {p['cache_key'] for p in predictions}
    for row, pred in zip(rows, predictions):
        raw = load_json(client.cache / (pred['cache_key'] + '.json'))['raw_response'] or {}
        choices = raw.get('choices') or []
        first = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = first.get('message') or {}
        reasoning_samples += bool(message.get('reasoning_content'))
        truncations += first.get('finish_reason') == 'length'
        for entity in pred['entities']:
            valid, _, _ = checked_entities([entity], len(row['tokens']))
            if valid:
                valid_total += 1
                mismatch += row['text'][entity['start']:entity['end']] != entity['text']
    attempts = [a for a in client.ledger['attempts'] if a['key'] in keys]
    latency = [p['latency_seconds'] for p in predictions]
    diag = {
        'shots': len(examples), 'samples': len(rows), 'token': metrics['token'], 'entity': metrics['entity'],
        'entity_per_type': metrics['entity_per_type'], 'failure_rate': metrics['failure_rate'],
        'diagnostics': metrics['diagnostics'], 'error_counts': metrics['error_counts'],
        'text_span_mismatches': mismatch, 'valid_spans': valid_total,
        'text_span_mismatch_rate': mismatch / valid_total if valid_total else 0,
        'truncated_samples': truncations, 'nonempty_reasoning_samples': reasoning_samples,
        'api_error_samples': sum(p['status'] == 'api_error' for p in predictions),
        'format_error_samples': sum(p['status'] == 'format_error' for p in predictions),
        'actual_models': sorted({p.get('actual_model') or 'unknown' for p in predictions}),
        'prompt_tokens': sum((a.get('usage') or {}).get('prompt_tokens', 0) for a in attempts),
        'completion_tokens': sum((a.get('usage') or {}).get('completion_tokens', 0) for a in attempts),
        'attempts_including_retries': len(attempts),
        'cost_ledger_upper_cny': sum(a['charged_upper_cny'] for a in attempts),
        'latency_median_seconds': float(np.median(latency)),
        'latency_p95_seconds': float(np.quantile(latency, .95)),
        'latency_definition': 'Original request elapsed time including in-process retries; cache restores retain original latency; wall-clock interrupted retries are not included',
        'execution_workers_counts': {str(w): sum(p.get('execution_workers', 1) == w for p in predictions)
                                     for w in sorted({p.get('execution_workers', 1) for p in predictions})},
        'scheduling_note': 'Earlier cached records retain original serial timings; subsequent requests use the declared worker count. Mixed scheduling is disclosed.',
    }
    # Save complete groups only. Missing/unexecuted sentences never become empty predictions.
    write_jsonl(folder / 'predictions.jsonl', predictions)
    save_json(folder / 'metrics.json', metrics)
    write_jsonl(folder / 'sentence_counts.jsonl', counts)
    write_jsonl(folder / 'errors.jsonl', cases)
    save_json(folder / 'diagnostic_summary.json', diag)
    names = ['predictions.jsonl', 'metrics.json', 'sentence_counts.jsonl', 'errors.jsonl', 'diagnostic_summary.json']
    atomic_json(done, {'completed_at': utc_now(), 'protocol_sha256': protocol_hash,
                       'artifact_sha256': {n: file_hash(folder / n) for n in names}})
    print('DONE', folder.name, 'token_F1=', round(metrics['token']['f1'], 4),
          'entity_F1=', round(metrics['entity']['f1'], 4), flush=True)
    return diag


def replay(base, rows):
    """Read existing complete results only; no client, key requirement or writes."""
    frozen = migrated_freeze(base) or load_json(base / 'frozen_protocol.json')
    if frozen['protocol']['test_ids'] != [r['id'] for r in rows]:
        raise RuntimeError('Frozen Test ID sequence differs')
    digest = object_hash(frozen['protocol'])
    for shots in [0, 2, 4, 8]:
        folder = base / f'shot_{shots}'
        if not (folder / 'run.json').is_file():
            raise RuntimeError(f'Incomplete {folder.name}; replay cannot send requests')
        result = run_group(None, rows, [], folder, digest)
        print('VERIFIED', shots, 'token_F1=', result['token']['f1'], 'entity_F1=', result['entity']['f1'])


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Explicitly send paid requests for all four full-Test settings')
    parser.add_argument('--check-config', action='store_true', help='Offline configuration/data/key-presence checks only')
    parser.add_argument('--replay', action='store_true', help='Offline verification of all four completed prediction groups; never calls API')
    parser.add_argument('--budget-cny', type=float, default=30)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/glm_flashx_test_v1'))
    args = parser.parse_args()
    if sum([args.run, args.check_config, args.replay]) > 1:
        parser.error('--run, --check-config and --replay are separate actions')
    if not math.isfinite(args.budget_cny) or args.budget_cny <= 0 or args.budget_cny > 100:
        parser.error('Use a finite budget in (0,100] CNY; project budget remains 100 CNY')
    config_path = ROOT / 'configs/glm_flashx.json'
    config = load_json(config_path)
    connection = settings(config)
    manifest, examples, test = build_context(config)
    base = (ROOT / args.output_dir).resolve()
    if not base.is_relative_to((ROOT / 'runs').resolve()) or base == (ROOT / 'runs').resolve():
        raise ValueError('Use a dedicated subdirectory of runs/')
    if args.replay:
        replay(base, test)
        return
    prior = prior_project_cost(base)
    protocol = build_protocol(config, connection, manifest, examples, test, args.budget_cny, args.workers)
    existing = sum(a['charged_upper_cny'] for a in load_json(base / 'ledger.json')['attempts']) if (base / 'ledger.json').exists() else 0
    preview = {
        'mode': 'PAID FULL TEST' if args.run else 'OFFLINE CHECK ONLY',
        'glm_key_present': bool(connection['key']), 'endpoint': connection['endpoint'],
        'parameters': protocol['parameters'], 'shots': config['shots'], 'test_samples_each': len(test),
        'workers': args.workers,
        'original_requests_total': len(test) * len(config['shots']),
        'example_ids': config['example_ids'], 'evaluation_protocol': PROTOCOL_VERSION,
        'budget_cny_requested': args.budget_cny, 'prior_project_ledger_upper_cny': prior,
        'existing_glm_ledger_upper_cny': existing, 'project_remaining_cny': max(0, 100 - prior - existing),
        'output_dir': str(base), 'notice': 'Full-run cost is not guaranteed. Request reservations enforce the ceiling; no automatic recharge or Test reduction. Offline checks cannot validate Key/permissions/account balance.',
    }
    print(__import__('json').dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if not args.run:
        if not connection['key']:
            print('Set GLM_API_KEY in project .env, then rerun this offline check. Key value is never printed.')
        return
    if not connection['key']:
        raise RuntimeError('GLM_API_KEY is empty; no request sent')
    if prior + existing >= 100:
        raise BudgetStop('Project budget exhausted; no request sent')

    lock = ROOT / 'runs/llm/process.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        prior = prior_project_cost(base)
        freeze = freeze_once(base, protocol)
        for shots in config['shots']:
            folder = base / f'shot_{shots}'
            if (folder / 'run.json').exists():
                run_group(None, test, [], folder, object_hash(protocol))
            elif folder.exists() and any(folder.iterdir()):
                raise RuntimeError('Partial GLM group files require audit before any new request')
        client = GLMClient(base, config, pilot_messages, budget=max(0, min(args.budget_cny, 100 - prior)),
                           workers=args.workers, accepted_failure_keys=accepted_failures(base))
        groups = []
        for shots in config['shots']:
            groups.append(run_group(client, test, examples[:shots], base / f'shot_{shots}', object_hash(protocol), args.workers))
        result = {'completed_at': utc_now(), 'frozen_at': freeze['frozen_at'], 'groups': groups,
                  'glm_ledger_upper_cny': client.total(), 'project_ledger_upper_cny': prior + client.total(),
                  'offline_replay': True, 'selection': 'Report all four settings; do not select from Test',
                  'disclosure': config['notes']}
        save_json(base / 'summary.json', result)
        with (base / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            columns = ['shots', 'samples', 'token_p', 'token_r', 'token_f1', 'entity_p', 'entity_r', 'entity_f1',
                       'cost_ledger_upper_cny', 'latency_median_seconds', 'latency_p95_seconds', 'failure_rate']
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for group in groups:
                row = {k: group[k] for k in columns if k in group}
                for kind in ['token', 'entity']:
                    for short, key in [('p', 'precision'), ('r', 'recall'), ('f1', 'f1')]:
                        row[f'{kind}_{short}'] = group[kind][key]
                writer.writerow(row)
        print('COMPLETE', str(base / 'summary.csv'), flush=True)
    except (BudgetStop, InterfaceStop):
        print('STOPPED. Cached requests and ledger preserved. No automatic recharge, repair call, or Test reduction.', flush=True)
        raise
    finally:
        os.close(fd)
        lock.unlink()


def build_protocol(config, connection, manifest, examples, test, budget, workers):
    sources = [Path(__file__).resolve(), ROOT / 'configs/glm_flashx.json', ROOT / 'scripts/pilot_shot_budget.py',
               ROOT / 'scripts/run_baselines.py', ROOT / 'src/ner_experiment/models/glm.py',
               ROOT / 'src/ner_experiment/models/llm.py', ROOT / 'src/ner_experiment/common.py',
               ROOT / 'src/ner_experiment/evaluation/metrics.py']
    return {
        'version': 'glm-flashx-full-test-v1', 'endpoint': connection['endpoint'],
        'parameters': {'model': connection['model'], **config['parameters']},
        'system': SYSTEM_TEXT, 'examples': examples, 'shots': config['shots'],
        'dataset_manifest': manifest, 'test_ids': [r['id'] for r in test],
        'evaluation': 'V3 retain valid deduplicated overlaps; full BIO-position sets and exact spans',
        'pricing': config['pricing'], 'budget_cny': budget, 'project_budget_cny': 100, 'workers': workers,
        'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
        'disclosure': config['notes'], 'retry_policy': 'At most two extra network/429/5xx attempts; no format or semantic repair',
    }


if __name__ == '__main__':
    main()
