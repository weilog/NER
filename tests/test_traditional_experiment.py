import json
import pytest
from ner_experiment.data.splitting import json_bytes, sha
from ner_experiment.experiments.traditional import (choose_best, completed_trial,
                                                  predict_rows)
from ner_experiment.models.dictionary import DictionaryNER


def test_dev_selection_uses_entity_then_fit_time_then_order():
    def trial(entity, token, elapsed, order):
        return {'entity': {'f1': entity}, 'token': {'f1': token},
                'fit_seconds': elapsed, 'configuration_order': order}
    a = trial(0.7, 0.95, 1, 0)
    b = trial(0.8, 0.5, 20, 1)
    c = trial(0.8, 0.5, 10, 2)
    d = trial(0.8, 0.4, 10, 0)
    assert choose_best([a, b]) is b
    assert choose_best([a, b, c]) is c
    assert choose_best([a, b, c, d]) is d


def test_cached_trial_checks_context_and_artifact_bytes(tmp_path):
    assert completed_trial(tmp_path, 'context') is None
    outputs = {'predictions.jsonl': b'[]', 'metrics.json': b'{}', 'normalized.jsonl': b'[]'}
    trial = {'context_sha256': 'context', 'output_sha256': {name: sha(body) for name, body in outputs.items()}}
    for name, body in outputs.items():
        (tmp_path / name).write_bytes(body)
    (tmp_path / 'trial.json').write_bytes(json_bytes(trial))
    assert completed_trial(tmp_path, 'context') == trial
    with pytest.raises(ValueError, match='Different experiment context'):
        completed_trial(tmp_path, 'new-context')
    (tmp_path / 'predictions.jsonl').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Corrupted cached trial'):
        completed_trial(tmp_path, 'context')


def test_prediction_adapter_keeps_ids_raw_tags_and_latencies():
    class Model:
        def predict(self, text, constrained=True):
            return ['I-name'] * len(text)
    rows = [{'id': 'dev:0', 'text': '张三'}, {'id': 'dev:1', 'text': '李四'}]
    predictions = predict_rows(Model(), rows, 'hmm', constrained=False)
    assert [p['id'] for p in predictions] == [r['id'] for r in rows]
    assert predictions[0]['raw_bio_tags'] == ['I-name', 'I-name']
    assert predictions[0]['entities'] == [{'start': 0, 'end': 2, 'type': 'name', 'text': '张三'}]
    assert all(p['elapsed_seconds'] >= 0 and p['status'] == 'ok' for p in predictions)
