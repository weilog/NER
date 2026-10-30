"""One audited, offline scheduling migration for the existing serial GLM run."""
import argparse
from pathlib import Path

from ner_experiment.common import ROOT, file_hash, load_json, object_hash, save_json, utc_now
from ner_experiment.models.glm import settings
from run_glm_test import build_context, build_protocol, freeze_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not args.apply:
        parser.error('Use --apply only for the user-authorized serial-to-four-worker migration; no API calls')
    base = ROOT / 'runs/glm_flashx_test_v1'
    audit = base / 'scheduling_migration_v1'
    if (ROOT / 'runs/llm/process.lock').exists():
        raise RuntimeError('Old process must finish and release its lock first')
    if (audit / 'migration.json').exists() or (base / 'execution_protocol_v2.json').exists():
        raise RuntimeError('Migration already exists; do not overwrite it')
    snapshot = load_json(audit / 'before_manifest.json')
    for name, digest in snapshot['files'].items():
        if file_hash(base / name) != digest:
            raise RuntimeError('Pre-migration files changed: ' + name)
    old_freeze = load_json(base / 'frozen_protocol.json')
    old = old_freeze['protocol']
    old_sources = {k.replace('\\', '/'): v for k, v in old['source_sha256'].items()}
    for name in ['scripts/run_glm_test.py', 'src/ner_experiment/models/glm.py']:
        if file_hash(audit / 'original_source' / name) != old_sources[name]:
            raise RuntimeError('Original source backup differs from original freeze')
    config = load_json(ROOT / 'configs/glm_flashx.json')
    manifest, examples, test = build_context(config)
    new = build_protocol(config, settings(config), manifest, examples, test, old['budget_cny'], 4)
    semantics = lambda p: {k: v for k, v in p.items() if k not in {'workers', 'source_sha256'}}
    if semantics(old) != semantics(new):
        raise RuntimeError('Model/data/examples/parameters/budget changed; migration refused')
    if any(base.glob('shot_*/run.json')):
        raise RuntimeError('This migration is limited to the existing incomplete serial first group')
    ledger = load_json(base / 'ledger.json')
    if any(a['status'] == 'reserved' for a in ledger['attempts']):
        raise RuntimeError('Unresolved in-flight reservation requires manual audit')
    retained = []
    for name in snapshot['files']:
        if not name.replace('\\', '/').startswith('cache/'):
            continue
        p = load_json(base / name)['prediction']
        if p.get('fatal_error'):
            if p['status'] != 'api_error' or p.get('error', {}).get('http_status') != 400:
                raise RuntimeError('Unexpected fatal cached failure; audit required')
            retained.append(p['cache_key'])
    active = {'frozen_at': utc_now(), 'protocol': new,
              'previous_protocol_sha256': object_hash(old),
              'note': 'Execution scheduling migration only; all original cached responses/failures retained without another request'}
    save_json(base / 'execution_protocol_v2.json', active)
    record = {
        'created_at': utc_now(), 'authorization': 'User requested preservation of current results and parallel execution of remaining requests',
        'workers_before': 1, 'workers_after': 4,
        'original_freeze_sha256': file_hash(base / 'frozen_protocol.json'),
        'active_freeze_sha256': file_hash(base / 'execution_protocol_v2.json'),
        'before_manifest_sha256': file_hash(audit / 'before_manifest.json'),
        'baseline_cache_count': snapshot['cache_count'],
        'baseline_ledger_upper_cny': sum(a['charged_upper_cny'] for a in ledger['attempts']),
        'retained_http400_failure_keys': retained,
        'retained_failure_policy': 'Original HTTP400 body was not recorded; reason unknown. Keep API failure exactly as cached, empty prediction for scoring, no re-request.',
        'new_error_diagnostics': 'Subsequent HTTP errors preserve sanitized provider code/message. Only documented HTTP400/code1301 is a non-retried per-sample refusal; unknown interface errors stop the batch.',
        'latency_policy': '333 serial caches retain original elapsed times; later responses record execution_workers=4. Mixed scheduling must be disclosed; cache benefits may change.',
        'script_sha256': file_hash(Path(__file__).resolve()),
    }
    save_json(audit / 'migration.json', record)
    assert freeze_once(base, new) == active
    print({'migration': 'verified', 'preserved_cache_count': snapshot['cache_count'],
           'workers_after': 4, 'retained_failures_without_retry': len(retained), 'api_calls': 0})


if __name__ == '__main__':
    main()
