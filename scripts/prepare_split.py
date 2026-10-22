"""Create or verify the frozen experiment split; never regenerate its seed."""
from pathlib import Path
import json
from ner_experiment.data.splitting import prepare_split

if __name__ == '__main__':
    manifest = prepare_split(Path(__file__).resolve().parents[1])
    print(json.dumps({key: manifest[key] for key in ('seed', 'algorithm', 'statistics',
          'valid_training_sentences', 'excluded_training_ids', 'quarantine_ids', 'examples_coverage')}, indent=2))
