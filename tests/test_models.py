from itertools import product
import joblib
import numpy as np
import pytest
from ner_experiment.evaluation.metrics import bio_to_spans, entities_list
from ner_experiment.models.dictionary import DictionaryNER
from ner_experiment.models.hmm import CharacterHMM, bio_masks, viterbi
from ner_experiment.models.crf import CharacterCRF, sentence_features


def row(text, tags):
    spans, normalized, illegal = bio_to_spans(tags)
    assert not illegal and normalized == tags
    return {'text': text, 'bio_tags': tags, 'bio_compatible': True,
            'entities': entities_list(spans, text)}


def test_dictionary_longest_frequency_ties_and_unseen():
    rows = [row('北京大学', ['B-organization'] + ['I-organization'] * 3),
            row('北京', ['B-address', 'I-address']),
            row('苹果', ['B-name', 'I-name']), row('苹果', ['B-company', 'I-company'])]
    dictionary = DictionaryNER().fit(rows)
    predictions = dictionary.predict('北京大学苹果未知')
    assert [(e['start'], e['end'], e['type']) for e in predictions] == [(0, 4, 'organization'), (4, 6, 'company')]
    assert dictionary.predict('未知') == []
    assert DictionaryNER().fit(rows + [rows[2]]).lexicon_['苹果'] == 'name'


@pytest.mark.parametrize('constrained', [False, True])
@pytest.mark.parametrize('length', [1, 2, 3, 4])
def test_viterbi_matches_exhaustive_paths(constrained, length):
    rng = np.random.default_rng(20261022 + length)
    states = ('O', 'B-name', 'I-name', 'B-company', 'I-company')
    initial = rng.random(len(states)); initial /= initial.sum()
    transition = rng.random((len(states), len(states))); transition /= transition.sum(axis=1, keepdims=True)
    emission = rng.random((length, len(states))); emission /= emission.sum(axis=0, keepdims=True)
    initial, transition, emission = map(np.log, (initial, transition, emission))
    initial_mask, transition_mask = bio_masks(states)
    scored = []
    for path in product(range(len(states)), repeat=length):
        if constrained and (not initial_mask[path[0]] or any(not transition_mask[a, b] for a, b in zip(path, path[1:]))):
            continue
        score = initial[path[0]] + sum(emission[t, state] for t, state in enumerate(path))
        score += sum(transition[a, b] for a, b in zip(path, path[1:]))
        scored.append((score, path))
    expected_score, expected_path = max(scored, key=lambda item: item[0])
    path, score = viterbi(initial, transition, emission,
                          initial_mask if constrained else None,
                          transition_mask if constrained else None)
    assert path == list(expected_path)
    assert score == pytest.approx(expected_score)


def test_viterbi_manual_sequence_constraint_and_empty():
    states = ('O', 'B-name', 'I-name')
    initial = np.log([0.1, 0.3, 0.6])
    transitions = np.log([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7], [0.2, 0.1, 0.7]])
    emissions = np.log([[0.1, 0.4, 0.5], [0.1, 0.1, 0.8]])
    masks = bio_masks(states)
    assert viterbi(initial, transitions, emissions)[0] == [2, 2]
    assert viterbi(initial, transitions, emissions, *masks)[0] == [1, 2]
    assert viterbi(initial, transitions, np.empty((0, 3))) == ([], 0.0)


def test_hmm_supervised_counts_unk_and_probability_normalization():
    rows = [row('张三在京', ['B-name', 'I-name', 'O', 'B-address']),
            row('李四在京', ['B-name', 'I-name', 'O', 'B-address'])]
    model = CharacterHMM(alpha=0.1, min_char_count=2).fit(rows)
    assert set(model.vocabulary_) == {'在', '京'}
    assert model.training_unk_characters_ == 4
    assert np.exp(model.log_initial_).sum() == pytest.approx(1)
    assert np.exp(model.log_transition_).sum(axis=1) == pytest.approx(np.ones(21))
    assert np.exp(model.log_emission_).sum(axis=1) == pytest.approx(np.ones(21))
    b_name = model.states_.index('B-name')
    assert np.exp(model.log_initial_[b_name]) == pytest.approx(2.1 / 4.1)
    assert np.exp(model.log_emission_[b_name, 0]) == pytest.approx(2.1 / 2.3)
    predicted = model.predict('未知在京')
    assert len(predicted) == 4 and bio_to_spans(predicted)[2] == 0
    assert model.predict('') == []
    stats = model.vocabulary_statistics(['未知在京'])
    assert stats['dev_unk_rate'] == 0.5


def test_crf_features_boundaries_and_local_bigrams():
    features = sentence_features('张3，A')
    assert features[0]['-1:BOS'] and features[0]['-2:BOS']
    assert features[1]['+0:isdigit']
    assert features[2]['+0:ispunct']
    assert features[3]['+0:isascii_alpha'] and features[3]['+1:EOS']
    assert features[1]['bigram-1'] == '张3' and features[1]['bigram+0'] == '3，'
    assert sentence_features('') == []


def test_model_save_restore_predictions(tmp_path):
    rows = [row('张三在京', ['B-name', 'I-name', 'O', 'B-address'])] * 3
    crf = CharacterCRF(max_iterations=20).fit_features([sentence_features(r['text']) for r in rows], [r['bio_tags'] for r in rows])
    for model in [DictionaryNER().fit(rows), CharacterHMM().fit(rows), crf]:
        expected = model.predict('张三在京')
        path = tmp_path / f'{type(model).__name__}.joblib'
        joblib.dump(model, path)
        assert joblib.load(path).predict('张三在京') == expected
    assert crf.predict('张三在京') == rows[0]['bio_tags']
