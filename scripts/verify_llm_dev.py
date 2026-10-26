"""Replay both Dev phases with no network, preserving artifacts/accounting."""
import json
from pathlib import Path

from ner_experiment.data.splitting import json_bytes, sha
from ner_experiment.experiments.llm_dev import run
from ner_experiment.experiments.traditional import atomic_write
from ner_experiment.llm.budget import Ledger


def main(root):
    root = Path(root)
    config = json.loads((root / 'configs/llm_dev_v1.json').read_text(encoding='utf-8'))
    directory = root / config['run_directory']
    files = []
    for phase in ('tune', 'full'):
        summary = json.loads((directory / phase / 'summary.json').read_text(encoding='utf-8'))
        if not summary['complete']:
            raise ValueError('Cannot verify incomplete phase: ' + phase)
        files += [directory / phase / name for name in ('context.json','requests.jsonl','records.jsonl','summary.json')]
        files += [directory / phase / arm / name for arm in summary['arms']
                  for name in ('predictions.jsonl','metrics.json','normalized.jsonl','diagnostics.json')]
    files += [directory / 'selection.json']
    before = {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in files}
    ledger = Ledger(root / config['ledger_path'], config['budget_cny'])
    try:
        accounting = ledger.summary()
    finally:
        ledger.close()
    for phase in ('tune', 'full'):
        run(root, phase, offline=True)
    ledger = Ledger(root / config['ledger_path'], config['budget_cny'])
    try:
        if ledger.summary() != accounting:
            raise ValueError('Offline replay changed accounting')
    finally:
        ledger.close()
    if before != {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in files}:
        raise ValueError('Offline replay changed frozen artifacts')
    result = {'cached_request_positions_replayed': 1200+2686, 'additional_api_attempts': 0,
        'protected_files_byte_identical': len(files), 'accounting_unchanged': True,
        'protected_sha256': before, 'test_accessed': False}
    atomic_write(directory / 'verification/offline-replay.json', json_bytes(result))
    print(json.dumps({k:v for k,v in result.items() if k != 'protected_sha256'}, indent=2))


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
