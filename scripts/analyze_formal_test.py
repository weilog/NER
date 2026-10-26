"""Offline fixed-protocol Test analysis; does not call or change any model."""
from pathlib import Path
from ner_experiment.experiments.formal_analysis import main

if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
