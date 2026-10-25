"""Prepare, run, or replay the distinct 2x2 character-position experiment."""
import argparse
from pathlib import Path
from ner_experiment.experiments.index_factorial import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], dry_run=args.dry_run, offline=args.offline)
