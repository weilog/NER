"""Plan or run the fixed zero-shot 20-Dev pilot; never reads Test."""
import argparse
from pathlib import Path
from ner_experiment.llm.pilot import run_pilot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        run_pilot(Path(__file__).resolve().parents[1], dry_run=args.dry_run)
    except (ValueError, RuntimeError, FileExistsError) as exc:
        # These exceptions are controlled internal validation errors, not SDK errors.
        parser.exit(2, f'Pilot stopped: {exc}\n')
