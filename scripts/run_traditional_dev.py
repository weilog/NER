"""Run only traditional-model Dev experiments; never opens Test JSONL."""
import argparse
from pathlib import Path
from ner_experiment.experiments.traditional import run_development

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='runs/traditional_dev')
    args = parser.parse_args()
    summary = run_development(Path(__file__).resolve().parents[1], args.run_dir)
    print('Best traditional by Dev entity F1:', summary['best_traditional_by_dev'])
