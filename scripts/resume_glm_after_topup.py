"""Explicit top-up recovery, with independent provider lock and reserved two-model ceilings."""
import argparse
from contextlib import contextmanager
import csv
import math
import os
from pathlib import Path
import re
import sys

from ner_experiment.common import ROOT, file_hash, load_json, object_hash, save_json, utc_now
from ner_experiment.models.glm_recovery import RecoveryGLMClient
from ner_experiment.models.glm import settings
from pilot_shot_budget import pilot_messages
import run_glm_test as shared


def allocation_limit(other_cost, glm_limit=40, mimo_limit=20):
    values = (other_cost, glm_limit, mimo_limit)
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Invalid project budget allocation')
    upper = sum(values)
    if upper > 100:
        raise RuntimeError('Combined reserved model ceilings exceed project 100 CNY cap')
    return upper


def quota_candidates(base):
    ledger = load_json(base / 'ledger.json')['attempts']
    if any(a['status'] == 'reserved' for a in ledger):
        raise RuntimeError('GLM has unresolved in-flight requests; do not start another worker')
    candidates = []
    for path in (base / 'cache').glob('*.json'):
        record = load_json(path)
        p = record['prediction']
        error = p.get('error', {})
        if p['status'] != 'api_error' or error.get('provider_error_code') != '1113':
            continue
        key = p['cache_key']
        attempts = [a for a in ledger if a['key'] == key]
        if (error.get('http_status') != 429 or record['raw_response'] is not None or p['entities']
                or not re.fullmatch('[0-9a-f]{64}', key) or path.stem != key
                or not 0 < len(attempts) < 3 or any(a['status'] == 'response' for a in attempts)):
            raise RuntimeError('Only verified unanswered insufficient-balance failures may be resumed')
        candidates.append({'key': key, 'id': p['id'], 'shots': len(record['example_ids']),
                           'sha256': file_hash(path), 'previous_attempts': len(attempts)})
    return sorted(candidates, key=lambda p: p['key'])


def archive_quota(base, candidates):
    """Move only verified no-answer quota failures; immutable history remains inside the run."""
    archive = base / 'quota_recovery_v1/prior_cache'
    archive.mkdir(parents=True, exist_ok=True)
    for p in candidates:
        source = base / 'cache' / (p['key'] + '.json')
        target = archive / source.name
        if not source.resolve().is_relative_to(base.resolve()) or not target.resolve().is_relative_to(base.resolve()):
            raise RuntimeError('Archive path must stay inside the GLM experiment')
        if target.exists() or file_hash(source) != p['sha256']:
            raise RuntimeError('Quota recovery history exists or original cache changed')
        source.rename(target)
        if file_hash(target) != p['sha256']:
            raise RuntimeError('Archived quota record verification failed')


@contextmanager
def provider_lock(base, shared_lock, expected_mimo_pid):
    # Never replace/remove MiMo's global lock. Explicit owner is verified by the operator.
    if shared_lock.exists() and int(shared_lock.read_text()) != expected_mimo_pid:
        raise RuntimeError('Global lock owner changed; verify the MiMo process before concurrent start')
    path = base / 'provider.lock'
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        path.unlink()


def check_allocation(base, mimo):
    frozen_mimo = load_json(mimo / 'frozen_protocol.json')
    if frozen_mimo['protocol']['budget_cny'] != 20:
        raise RuntimeError('Expected MiMo frozen local ceiling of 20 CNY')
    other = 0.0
    for path in (ROOT / 'runs').rglob('ledger.json'):
        if path.resolve() in {(base / 'ledger.json').resolve(), (mimo / 'ledger.json').resolve()}:
            continue
        other += sum(a['charged_upper_cny'] for a in load_json(path)['attempts'])
    upper = allocation_limit(other)
    for folder, ceiling in [(base, 40), (mimo, 20)]:
        current = sum(a['charged_upper_cny'] for a in load_json(folder / 'ledger.json')['attempts'])
        if current > ceiling:
            raise RuntimeError('Existing model ledger already exceeds reserved ceiling')
    return {'glm_reserved_ceiling_cny': 40, 'mimo_reserved_ceiling_cny': 20,
            'other_provider_cost_upper_cny': other, 'combined_worst_case_cny': upper,
            'project_ceiling_cny': 100, 'mimo_freeze_sha256': file_hash(mimo / 'frozen_protocol.json'),
            'policy': 'Keep all historical unknown fees; do not run a third paid experiment during this allocation'}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--mimo-owner-pid', type=int, required=True,
                        help='PID of the already-verified MiMo owner of runs/llm/process.lock')
    args = parser.parse_args()
    base, mimo = ROOT / 'runs/glm_flashx_test_v1', ROOT / 'runs/mimo_flash_test_v1'
    config = load_json(ROOT / 'configs/glm_flashx.json')
    manifest, examples, test = shared.build_context(config)
    frozen = shared.migrated_freeze(base)
    candidate = shared.build_protocol(config, settings(config), manifest, examples, test,
                                      frozen['protocol']['budget_cny'], 4)
    protocol = shared.freeze_once(base, candidate)['protocol']
    allocation = check_allocation(base, mimo)
    audit = base / 'quota_recovery_v1'
    plan_path = audit / 'plan.json'
    retries = load_json(plan_path)['quota_retries'] if plan_path.exists() else quota_candidates(base)
    print({'mode': 'PAID RESUME' if args.run else 'OFFLINE CHECK ONLY', 'workers_glm': 4,
           'quota_retry_ids': [p['id'] for p in retries], **allocation}, flush=True)
    if not args.run:
        return
    if not settings(config)['key']:
        raise RuntimeError('GLM_API_KEY empty; no request sent')
    with provider_lock(base, ROOT / 'runs/llm/process.lock', args.mimo_owner_pid):
        check_allocation(base, mimo)
        for shot in [0, 2, 4]:
            shared.run_group(None, test, [], base / f'shot_{shot}', object_hash(protocol))
        if not plan_path.exists():
            retries = quota_candidates(base)
            audit.mkdir(exist_ok=True)
            backup = audit / 'ledger_at_topup.json'
            if backup.exists():
                raise RuntimeError('Partial quota audit exists; review before continuing')
            backup.write_bytes((base / 'ledger.json').read_bytes())
            files = list((base / 'cache').glob('*.json')) + [p for p in base.glob('shot_*/*') if p.is_file()]
            save_json(audit / 'before_manifest.json', {'files': {
                p.relative_to(base).as_posix(): file_hash(p) for p in files}})
            save_json(plan_path, {'created_at': utc_now(), 'quota_retries': retries,
                'authorization': 'User topped up BigModel and explicitly requested simultaneous GLM/MiMo continuation',
                'allocation': allocation, 'prior_ledger_sha256': file_hash(backup),
                'note': 'Retry only unanswered balance1113 errors, within original maximum-three lifetime attempts; never choose between model answers'})
            archive_quota(base, retries)
        else:
            for p in retries:
                if file_hash(audit / 'prior_cache' / (p['key'] + '.json')) != p['sha256']:
                    raise RuntimeError('Archived balance failure changed')
        execution = {'model_protocol_sha256': object_hash(protocol), 'allocation': allocation,
                     'sources': {p.relative_to(ROOT).as_posix(): file_hash(p) for p in [Path(__file__).resolve(),
                         ROOT / 'src/ner_experiment/models/glm_recovery.py']},
                     'plan_sha256': file_hash(plan_path), 'workers': 4,
                     'note': 'Earlier completed group results retained; unknown reservations never released'}
        execution_path = audit / 'execution.json'
        if execution_path.exists():
            if load_json(execution_path)['protocol'] != execution:
                raise RuntimeError('Quota recovery configuration changed')
        else:
            save_json(execution_path, {'created_at': utc_now(), 'protocol': execution})
        client = RecoveryGLMClient(base, config, pilot_messages, 40, workers=4,
                                   accepted_failure_keys=shared.accepted_failures(base))
        groups = []
        for shot in [0, 2, 4, 8]:
            groups.append(shared.run_group(client, test, examples[:shot], base / f'shot_{shot}',
                                          object_hash(protocol), workers=4))
        summary = {'completed_at': utc_now(), 'groups': groups, 'glm_ledger_upper_cny': client.total(),
                   'other_models_allocated': allocation,
                   'quota_recovery': load_json(plan_path),
                   'disclosure': 'Two unanswered balance failures resumed after explicit user top-up; initial failure records preserved. Network-failed predictions remain failures.'}
        save_json(base / 'summary.json', summary)
        columns = ['shots', 'token_p', 'token_r', 'token_f1', 'entity_p', 'entity_r', 'entity_f1',
                   'failure_rate', 'cost_ledger_upper_cny', 'latency_median_seconds', 'latency_p95_seconds']
        with (base / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for g in groups:
                row = {k: g[k] for k in columns if k in g}
                for kind in ['token', 'entity']:
                    for short, key in [('p', 'precision'), ('r', 'recall'), ('f1', 'f1')]:
                        row[f'{kind}_{short}'] = g[kind][key]
                writer.writerow(row)
        print('GLM COMPLETE', str(base / 'summary.csv'), flush=True)


if __name__ == '__main__':
    main()
