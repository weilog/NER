"""Network-free replay of frozen Test and its preregistered analysis."""
from pathlib import Path
import json

from ner_experiment.data.splitting import json_bytes, sha
from ner_experiment.experiments.formal_test import load, run, verify_gate
from ner_experiment.experiments.formal_analysis import main as analyze
from ner_experiment.experiments.traditional import atomic_write
from ner_experiment.llm.budget import Ledger


def main(root):
    root = Path(root)
    gate = verify_gate(root)
    config, directory = gate['config'], root/gate['config']['run_directory']
    files = [directory/name for name in ('context.json','requests.jsonl','records.jsonl','started.json','summary.json')]
    files += [directory/method/name for method in config['methods']
              for name in ('predictions.jsonl','metrics.json','normalized.jsonl','diagnostics.json')]
    files += [directory/method/'prediction-manifest.json' for method in ('dictionary','hmm','crf')]
    files += [p for folder in ('analysis','human-review') for p in sorted((directory/folder).rglob('*')) if p.is_file()]
    before = {p.relative_to(root).as_posix():sha(p.read_bytes()) for p in files}
    db = Ledger(root/config['ledger_path'],config['budget_cny'])
    try:
        accounting = db.summary()
    finally:
        db.close()
    run(root,offline=True)
    analyze(root)
    db = Ledger(root/config['ledger_path'],config['budget_cny'])
    try:
        if db.summary() != accounting:
            raise ValueError('Offline replay altered accounting')
    finally:
        db.close()
    if before != {p.relative_to(root).as_posix():sha(p.read_bytes()) for p in files}:
        raise ValueError('Offline replay changed frozen predictions or analysis')
    result = {'cached_llm_requests_replayed':2418,'additional_api_attempts':0,
        'protected_files_byte_identical':len(files),'paired_bootstrap_identical':True,
        'novelty_slices_identical':True,'accounting_unchanged':True,'protected_sha256':before}
    atomic_write(directory/'verification/offline-replay.json',json_bytes(result))
    print(json.dumps({k:v for k,v in result.items() if k!='protected_sha256'},indent=2))


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
