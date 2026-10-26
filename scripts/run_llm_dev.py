"""Select shot count on fixed 300 Dev rows, then validate on full Dev."""
import argparse
from pathlib import Path
from ner_experiment.experiments.llm_dev import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['tune', 'full'], required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.phase, dry_run=args.dry_run, offline=args.offline)
