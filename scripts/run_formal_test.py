"""Commit the freeze before running formal Test; --offline never calls API."""
import argparse
from pathlib import Path
from ner_experiment.experiments.formal_test import freeze, run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    freeze(root) if args.freeze else run(root, dry_run=args.dry_run, offline=args.offline)
