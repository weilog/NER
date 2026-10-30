"""MiMo 0/2/4/8-shot fixed Test. Default/check-config/replay never call an API."""
import argparse
import csv
import math
import os
from pathlib import Path
import sys

import numpy as np
from ner_experiment.common import ROOT, file_hash, load_json, object_hash, read_jsonl, save_json, utc_now
from ner_experiment.models.mimo import MiMoClient, settings
from ner_experiment.models.glm import atomic_json
from pilot_shot_budget import SYSTEM_TEXT, pilot_messages
import run_glm_test as shared


def freeze_once(base, protocol):
    path = base / 'frozen_protocol.json'
    if path.exists():
        frozen = load_json(path)
        if object_hash(frozen['protocol']) != object_hash(protocol):
            raise RuntimeError('MiMo frozen protocol changed; preserve existing results')
        return frozen
    if base.exists() and any(base.iterdir()):
        raise RuntimeError('Nonempty MiMo directory without verified protocol is protected')
    record = {'frozen_at': utc_now(), 'protocol': protocol}
    atomic_json(path, record)
    return record


def make_protocol(config, connection, manifest, examples, test, budget):
    sources = [Path(__file__).resolve(), ROOT / 'configs/mimo_flash.json',
               ROOT / 'src/ner_experiment/models/mimo.py', ROOT / 'src/ner_experiment/models/glm.py',
               ROOT / 'src/ner_experiment/models/llm.py', ROOT / 'src/ner_experiment/common.py',
               ROOT / 'src/ner_experiment/evaluation/metrics.py', ROOT / 'scripts/run_glm_test.py',
               ROOT / 'scripts/run_baselines.py', ROOT / 'scripts/pilot_shot_budget.py']
    return {'version': 'mimo-flash-full-test-v1', 'endpoint': connection['endpoint'],
            'parameters': {'model': connection['model'], **config['parameters']},
            'system': SYSTEM_TEXT, 'examples': examples, 'shots': [0, 2, 4, 8],
            'dataset_manifest': manifest, 'test_ids': [r['id'] for r in test],
            'workers': config['workers'], 'global_rpm': config['rpm'],
            'budget_cny': budget, 'project_budget_cny': 100, 'pricing': config['pricing'],
            'source_sha256': {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources},
            'evaluation': 'V3 valid deduplicated spans including overlaps; character/full-BIO sets and strict entities',
            'selection': 'Report all four settings. No Test-based prompt/example/parameter selection.',
            'disclosure': 'Cross-model extension after previous Test inspection; not a first blind evaluation. MiMo thinking disabled; GLM thinking enabled.',
            'failure_policy': 'At most three attempts including retries; no hidden LLM repair; retain failures; stop on fatal interface or consecutive transport failures; TLS verified.'}


def verify_saved_groups(base, rows, digest, require_all=False):
    groups = []
    for shot in [0, 2, 4, 8]:
        folder = base / f'shot_{shot}'
        if (folder / 'run.json').exists():
            groups.append(shared.run_group(None, rows, [], folder, digest))
        elif require_all:
            raise RuntimeError(f'{folder.name} incomplete; offline replay never sends requests')
        elif folder.exists() and any(folder.iterdir()):
            raise RuntimeError('Partial MiMo group artifacts require review, cache is preserved')
    return groups


def timing_summary(base, shot):
    predictions = read_jsonl(base / f'shot_{shot}/predictions.jsonl')
    throttles = [p.get('throttle_seconds', 0) for p in predictions]
    http = [p.get('http_seconds', 0) for p in predictions]
    return {'throttle_median_seconds': float(np.median(throttles)),
            'http_attempts_median_seconds': float(np.median(http)),
            'http_attempts_p95_seconds': float(np.quantile(http, .95)),
            'definition': 'latency_seconds includes global RPM waiting, HTTP and in-process retries; http_seconds sums only original HTTP attempt durations. Cached rows retain original measurements.'}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check-config', action='store_true')
    action.add_argument('--run', action='store_true', help='Explicitly send paid requests for all four full-Test settings')
    action.add_argument('--replay', action='store_true', help='Offline verify saved predictions and metrics')
    parser.add_argument('--budget-cny', type=float, default=20)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/mimo_flash_test_v1'))
    args = parser.parse_args()
    if not math.isfinite(args.budget_cny) or not 0 < args.budget_cny <= 100:
        parser.error('Use finite local budget in (0,100], within the original 100 CNY project cap')
    config = load_json(ROOT / 'configs/mimo_flash.json')
    connection = settings(config)
    manifest, examples, test = shared.build_context(config)
    base = (ROOT / args.output_dir).resolve()
    if not base.is_relative_to((ROOT / 'runs').resolve()) or base == (ROOT / 'runs').resolve():
        raise ValueError('Use a dedicated subdirectory of runs/')
    if args.replay:
        frozen = load_json(base / 'frozen_protocol.json')
        if frozen['protocol']['test_ids'] != [r['id'] for r in test]:
            raise RuntimeError('Frozen Test IDs differ')
        verify_saved_groups(base, test, object_hash(frozen['protocol']), require_all=True)
        print('PASS: all four MiMo groups re-evaluated offline; no API requests')
        return
    prior = shared.prior_project_cost(base)  # includes every unreconciled GLM reservation
    existing = sum(a['charged_upper_cny'] for a in load_json(base / 'ledger.json')['attempts']) if (base / 'ledger.json').exists() else 0
    protocol = make_protocol(config, connection, manifest, examples, test, args.budget_cny)
    print({'mode': 'PAID FULL TEST' if args.run else 'OFFLINE CHECK ONLY',
           'mimo_key_present': bool(connection['key']), 'model': connection['model'],
           'endpoint': connection['endpoint'], 'parameters': protocol['parameters'],
           'samples_per_shot': len(test), 'shots': [0, 2, 4, 8], 'original_requests': 4836,
           'workers': config['workers'], 'global_rpm': config['rpm'], 'evaluation_protocol': 'v3',
           'local_budget_cny': args.budget_cny, 'other_project_ledger_upper_cny': prior,
           'existing_mimo_upper_cny': existing, 'project_remaining_cny': max(0, 100 - prior - existing),
           'notice': 'Key presence is not validity/credit verification; no prospective bill guarantee. GLM stays paused.'}, flush=True)
    if not args.run:
        return
    if not connection['key']:
        raise RuntimeError('MIMO_API_KEY is empty; no request sent')
    lock = ROOT / 'runs/llm/process.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        prior = shared.prior_project_cost(base)
        frozen = freeze_once(base, protocol)
        protocol = frozen['protocol']
        digest = object_hash(protocol)
        verify_saved_groups(base, test, digest)
        client = MiMoClient(base, config, pilot_messages,
                            budget=max(0, min(args.budget_cny, 100 - prior)), workers=config['workers'])
        groups = []
        for shot in [0, 2, 4, 8]:
            group = shared.run_group(client, test, examples[:shot], base / f'shot_{shot}', digest, config['workers'])
            timing = timing_summary(base, shot)
            save_json(base / f'shot_{shot}/rate_timing.json', timing)
            groups.append({**group, 'rate_timing': timing})
        attempts = client.ledger['attempts']
        summary = {'completed_at': utc_now(), 'frozen_at': frozen['frozen_at'], 'groups': groups,
                   'mimo_ledger_upper_cny': client.total(), 'project_ledger_upper_cny': prior + client.total(),
                   'returned_usage_estimate_cny': sum(a['charged_upper_cny'] for a in attempts if a['status'] == 'response'),
                   'unknown_reservations_cny': sum(a['charged_upper_cny'] for a in attempts if a['status'] != 'response'),
                   'protocol_sha256': digest, 'selection': protocol['selection'],
                   'disclosure': protocol['disclosure'], 'offline_replay_verified': True}
        save_json(base / 'summary.json', summary)
        columns = ['shots', 'samples', 'token_p', 'token_r', 'token_f1', 'entity_p', 'entity_r', 'entity_f1',
                   'failure_rate', 'cost_ledger_upper_cny', 'latency_median_seconds', 'latency_p95_seconds']
        with (base / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for group in groups:
                row = {key: group[key] for key in columns if key in group}
                for kind in ['token', 'entity']:
                    for short, full in [('p', 'precision'), ('r', 'recall'), ('f1', 'f1')]:
                        row[f'{kind}_{short}'] = group[kind][full]
                writer.writerow(row)
        print('COMPLETE', str(base / 'summary.csv'), flush=True)
    finally:
        os.close(fd)
        lock.unlink()


if __name__ == '__main__':
    main()
