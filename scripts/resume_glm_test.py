"""Resume unchanged GLM requests after network repair and explicit bill reconciliation."""
import argparse
import csv
import sys
import math
from pathlib import Path

from ner_experiment.common import ROOT, file_hash, load_json, object_hash, save_json, utc_now
from ner_experiment.models.glm import settings
from ner_experiment.models.glm_recovery import RecoveryGLMClient, validate_reconciliation, reconciled_total
from pilot_shot_budget import pilot_messages
import run_glm_test as original


def canonical_protocol(base, candidate):
    # Equality may accept 30 == 30.0, but hashes preserve their JSON representation.
    # Always use the actual frozen bytes' parsed representation for existing group hashes.
    return original.freeze_once(base, candidate)['protocol']


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--budget-cny', type=float, default=40,
                        help='Local gross/effective ceiling; project remains capped at 100 CNY')
    parser.add_argument('--retain-unknown-costs', action='store_true',
                        help='Explicitly retain all unknown historical reservations instead of requiring bill reconciliation')
    args = parser.parse_args()
    if not math.isfinite(args.budget_cny) or not 0 < args.budget_cny <= 100:
        parser.error('Budget must be finite, positive and no greater than the 100 CNY project cap')
    base = ROOT / 'runs/glm_flashx_test_v1'
    config = load_json(ROOT / 'configs/glm_flashx.json')
    manifest, examples, test = original.build_context(config)
    frozen = original.migrated_freeze(base)
    if frozen is None:
        raise RuntimeError('Expected the explicitly migrated four-worker experiment')
    protocol = original.build_protocol(config, settings(config), manifest, examples, test,
                                       frozen['protocol']['budget_cny'], 4)
    protocol = canonical_protocol(base, protocol)
    ledger = load_json(base / 'ledger.json')
    billing_path = base / 'billing_reconciliation.json'
    if not billing_path.exists() and not args.retain_unknown_costs:
        print('STOP: need user-verified BigModel bill reconciliation before resuming. No requests sent.')
        return
    billing = validate_reconciliation(ledger, load_json(billing_path)) if billing_path.exists() else None
    prior = original.prior_project_cost(base)
    effective = reconciled_total(ledger, billing)
    print({'mode': 'PAID RESUME' if args.run else 'OFFLINE CHECK ONLY',
           'workers': 4, 'glm_effective_upper_cny': effective, 'glm_budget_cny': args.budget_cny,
           'project_effective_upper_cny': prior + effective, 'key_present': bool(settings(config)['key']),
           'preserve_failed_predictions': True, 'retry_failed_cached_rows': False,
           'retain_all_historical_unknown_costs': billing is None})
    if not args.run:
        return
    baseline_path = base / 'network_recovery_v1/before_manifest.json'
    baseline = load_json(baseline_path)
    for name, digest in baseline['files'].items():
        if file_hash(base / name) != digest:
            raise RuntimeError('Existing prediction or result changed before recovery: ' + name)
    # A separate, immutable transport execution record supplements the unchanged model protocol.
    recovery = {
        'base_model_protocol_sha256': object_hash(protocol),
        'billing_reconciliation_sha256': file_hash(billing_path) if billing_path.exists() else None,
        'before_manifest_sha256': file_hash(baseline_path),
        'sources': {str(p.relative_to(ROOT)): file_hash(p) for p in [Path(__file__).resolve(),
                    ROOT / 'src/ner_experiment/models/glm_recovery.py']},
        'workers': 4, 'glm_budget_cny': args.budget_cny, 'project_budget_cny': 100,
        'budget_authorization': 'User requested continued execution without billing reconciliation; retain every unknown reservation and use 40 CNY local gross ceiling within original 100 CNY project authorization',
        'network_policy': 'Stop on DNS/refused/TLS verification failure before sending, or eight consecutive connection failures. Existing per-row maximum-three attempts unchanged. TLS verification enabled.',
        'history': 'Original 0/2-shot files and all cached 4-shot successes/failures remain unchanged; no historical failure is re-requested.',
    }
    audit_path = base / 'network_recovery_v1/execution.json'
    if audit_path.exists():
        if load_json(audit_path)['protocol'] != recovery:
            raise RuntimeError('Network recovery execution changed; preserve and audit before sending requests')
    elif not settings(config)['key']:
        raise RuntimeError('GLM_API_KEY is empty')
    import os
    lock = ROOT / 'runs/llm/process.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        if not audit_path.exists():
            save_json(audit_path, {'created_at': utc_now(), 'protocol': recovery})
        # Check completed artifacts before any new call.
        for shot in [0, 2, 4, 8]:
            folder = base / f'shot_{shot}'
            if (folder / 'run.json').exists():
                original.run_group(None, test, [], folder, object_hash(protocol))
            elif folder.exists() and any(folder.iterdir()):
                raise RuntimeError('Partial group artifacts require audit')
        client = RecoveryGLMClient(base, config, pilot_messages, min(args.budget_cny, max(0, 100 - prior)),
                                   workers=4, accepted_failure_keys=original.accepted_failures(base))
        groups = []
        for shot in [0, 2, 4, 8]:
            groups.append(original.run_group(client, test, examples[:shot], base / f'shot_{shot}',
                                              object_hash(protocol), workers=4))
        summary = {'completed_at': utc_now(), 'groups': groups, 'glm_effective_upper_cny': client.total(),
                   'project_effective_upper_cny': prior + client.total(),
                   'raw_ledger_upper_cny': sum(a['charged_upper_cny'] for a in client.ledger['attempts']),
                   'billing_reconciliation': billing, 'network_recovery': recovery,
                   'selection': 'All four predeclared shot settings; no Test-based selection',
                   'disclosure': 'Keep historical network-failed predictions; outages affect performance comparison.'}
        save_json(base / 'summary.json', summary)
        with (base / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            columns = ['shots', 'samples', 'token_p', 'token_r', 'token_f1', 'entity_p', 'entity_r', 'entity_f1',
                       'failure_rate', 'cost_raw_ledger_upper_cny', 'latency_median_seconds', 'latency_p95_seconds']
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for group in groups:
                row = {k: group[k] for k in ['shots', 'samples', 'failure_rate']}
                row['cost_raw_ledger_upper_cny'] = group['cost_ledger_upper_cny']
                row['latency_median_seconds'] = group['latency_median_seconds']
                row['latency_p95_seconds'] = group['latency_p95_seconds']
                for kind in ['token', 'entity']:
                    for short, key in [('p', 'precision'), ('r', 'recall'), ('f1', 'f1')]:
                        row[f'{kind}_{short}'] = group[kind][key]
                writer.writerow(row)
        print('COMPLETE', str(base / 'summary.csv'))
    finally:
        os.close(fd)
        lock.unlink()


if __name__ == '__main__':
    main()
