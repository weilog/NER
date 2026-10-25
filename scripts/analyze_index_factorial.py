"""Offline reporting; no model/API calls."""
from pathlib import Path
from ner_experiment.experiments.index_analysis import main

if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
