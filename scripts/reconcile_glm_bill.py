"""Record a user-verified, scoped BigModel bill without rewriting the original ledger."""
import argparse
import math
from ner_experiment.common import ROOT, file_hash, load_json, object_hash, save_json, utc_now
from ner_experiment.models.glm_recovery import validate_reconciliation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmed-cny', type=float, required=True,
                        help='User-confirmed charge for THIS GLM run up to the stopped ledger cutoff')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.confirmed_cny) or args.confirmed_cny < 0:
        parser.error('Need a finite nonnegative actual bill amount')
    base = ROOT / 'runs/glm_flashx_test_v1'
    if (ROOT / 'runs/llm/process.lock').exists():
        raise RuntimeError('Stop requests and resolve the process lock before reconciling')
    ledger = load_json(base / 'ledger.json')
    if any(a['status'] == 'reserved' for a in ledger['attempts']):
        raise RuntimeError('In-flight reservations remain; do not reconcile')
    if (base / 'billing_reconciliation.json').exists():
        raise RuntimeError('Existing reconciliation is protected')
    known = sum(a['charged_upper_cny'] for a in ledger['attempts'] if a['status'] == 'response')
    record = {
        'created_at': utc_now(), 'source': 'User verified the official BigModel billing console for this GLM run',
        'confirmed_account_charge_cny': args.confirmed_cny,
        'confirmed_charge_upper_cny': max(known, args.confirmed_cny),
        'known_usage_estimate_cny': known,
        'original_ledger_upper_cny': sum(a['charged_upper_cny'] for a in ledger['attempts']),
        'prefix_attempts': len(ledger['attempts']), 'prefix_sha256': object_hash(ledger['attempts']),
        'original_ledger_sha256': file_hash(base / 'ledger.json'),
        'scope': 'This stopped GLM run only; no other provider, no hidden recharge, 30 CNY ceiling retained',
    }
    validate_reconciliation(ledger, record)
    print({**record, 'applied': bool(args.apply), 'api_calls': 0})
    if args.apply:
        audit = base / 'network_recovery_v1'
        audit.mkdir(exist_ok=True)
        backup = audit / 'billing_ledger_before_reconciliation.json'
        if backup.exists():
            raise RuntimeError('Existing network recovery ledger backup is protected')
        backup.write_bytes((base / 'ledger.json').read_bytes())
        paths = list((base / 'cache').glob('*.json'))
        paths += [p for p in base.glob('shot_*/*') if p.is_file()]
        paths += [base / 'frozen_protocol.json', base / 'execution_protocol_v2.json']
        if not (audit / 'before_manifest.json').exists():
            save_json(audit / 'before_manifest.json', {
                'created_at': utc_now(),
                'files': {p.relative_to(base).as_posix(): file_hash(p) for p in paths},
            })
        save_json(base / 'billing_reconciliation.json', record)


if __name__ == '__main__':
    main()
