import json
import pytest
from ner_experiment.evaluation.metrics import (PredictionFileError, bio_to_spans,
    counts, evaluate, normalize_prediction, spans_to_bio, token_counts)


def gold(text='北京大学', tags=None, sample_id='sample:0'):
    tags = tags or ['B-organization', 'I-organization', 'I-organization', 'I-organization']
    spans, fixed, illegal = bio_to_spans(tags)
    assert tags == fixed and not illegal
    return {'id': sample_id, 'text': text, 'tokens': list(text), 'has_labels': True,
            'bio_compatible': True, 'bio_tags': tags,
            'entities': [{'start': a, 'end': b, 'type': typ, 'text': text[a:b]} for a, b, typ in sorted(spans)]}


def predict(*spans, **kwargs):
    return {'id': 'sample:0', 'status': 'ok', 'entities': [
        {'start': a, 'end': b, 'type': typ} for a, b, typ in spans], **kwargs}


@pytest.mark.parametrize('prediction,token,entity', [
    (predict((0, 4, 'organization')), (4, 0, 0), (1, 0, 0)),
    (predict((0, 2, 'organization')), (2, 0, 2), (0, 1, 1)),
    (predict((0, 4, 'name')), (0, 4, 4), (0, 1, 1)),
    (predict(), (0, 0, 4), (0, 0, 1)),
    (predict((0, 1, 'organization'), (1, 4, 'organization')), (3, 1, 1), (0, 2, 1)),
])
def test_hand_calculated_examples(prediction, token, entity):
    result = evaluate([gold()], [prediction])
    for name, expected in [('token', token), ('entity', entity)]:
        assert tuple(result[name][key] for key in ('tp', 'fp', 'fn')) == expected
    if token == (2, 0, 2):
        assert result['token']['f1'] == pytest.approx(2 / 3)
        assert result['entity']['f1'] == 0


def test_background_and_false_positive():
    background = gold(tags=['O'] * 4)
    empty = evaluate([background], [predict()])
    assert empty['token']['f1'] == empty['entity']['f1'] == 0
    assert empty['character_accuracy_including_O'] == 1
    false_positive = evaluate([background], [predict((0, 2, 'name'))])
    assert false_positive['token']['fp'] == 2
    assert false_positive['entity']['fp'] == 1
    assert false_positive['entity_macro_f1_fixed_10'] == 0


def test_micro_over_all_characters_and_fixed_macro():
    rows = [gold(), gold(tags=['O'] * 4, sample_id='sample:1')]
    predictions = [predict((0, 4, 'organization')), {**predict((0, 4, 'name')), 'id': 'sample:1'}]
    result = evaluate(rows, predictions)
    assert result['token']['precision'] == result['entity']['precision'] == 0.5
    assert result['token']['recall'] == result['entity']['recall'] == 1
    assert result['entity_macro_f1_fixed_10'] == 0.1


@pytest.mark.parametrize('predictions', [[], [predict(), predict()], [{**predict(), 'id': 'extra'}], [predict(), {**predict(), 'id': 'extra'}]])
def test_prediction_id_errors(predictions):
    with pytest.raises(PredictionFileError):
        evaluate([gold()], predictions)


def test_explicit_failures_are_empty_predictions():
    result = evaluate([gold()], [predict((0, 4, 'organization'), status='api_error')])
    assert result['entity']['fn'] == 1 and result['entity']['tp'] == 0
    assert result['rates']['failure_rate'] == 1


def test_illegal_bio_repair_and_strict_sensitivity():
    prediction = {'id': 'sample:0', 'status': 'ok', 'raw_bio_tags': ['I-organization'] * 4}
    result = evaluate([gold()], [prediction])
    assert result['token']['f1'] == result['entity']['f1'] == 1
    assert result['rates']['illegal_bio_transition_rate_before_repair'] == 0.25
    assert result['sensitivity']['raw_bio_token_on_bio_samples_only']['f1'] == 0.75
    assert result['sensitivity']['strict_bio_entity_on_bio_samples_only']['f1'] == 0
    spans, tags, illegal = bio_to_spans(['I-name', 'I-name', 'I-company', 'O'])
    assert spans == {(0, 2, 'name'), (2, 3, 'company')}
    assert illegal == 2 and tags == ['B-name', 'I-name', 'B-company', 'O']


def test_same_prediction_set_and_raw_overlap_diagnostic():
    prediction = predict((0, 4, 'organization'), (0, 2, 'name'), (0, 4, 'organization'))
    result = evaluate([gold()], [prediction])
    assert result['entity']['f1'] == result['token']['f1'] == 1
    assert result['sensitivity']['raw_span_entity']['fp'] == 1
    assert result['diagnostics']['duplicate_entities'] == 1
    assert result['diagnostics']['overlap_entities_removed'] == 1
    assert result['rates']['overlap_affected_sample_rate'] == 1


def test_overlap_ties_use_type_name_and_leftmost_rule():
    norm = normalize_prediction(predict((0, 2, 'organization'), (0, 2, 'name'), (1, 4, 'name'), (2, 4, 'organization')), '北京大学')
    assert norm['spans'] == {(0, 2, 'name'), (2, 4, 'organization')}
    assert norm['bio'] == ['B-name', 'I-name', 'B-organization', 'I-organization']


def test_invalid_spans_filtered_only_from_primary():
    prediction = predict((-1, 4, 'name'), (0, 99, 'name'), (0, 2, 'UNKNOWN'), (2, 1, 'name'))
    prediction['entities'] += [{'start': True, 'end': 2, 'type': 'name'}, {'start': 0, 'end': 4, 'type': 'organization', 'text': '错误'}]
    result = evaluate([gold()], [prediction])
    assert result['entity']['fp'] == 0 and result['entity']['fn'] == 1
    assert result['diagnostics']['invalid_entities'] == 6
    assert result['rates']['invalid_entity_rate'] == 1
    assert result['sensitivity']['raw_span_entity']['fp'] == 4


def test_invalid_duplicate_does_not_hide_valid_entry_and_is_counted():
    valid = {'start': 0, 'end': 4, 'type': 'organization', 'text': '北京大学'}
    invalid = {**valid, 'text': '错误'}
    for entries in ([valid, invalid], [invalid, valid]):
        result = evaluate([gold()], [{**predict(), 'entities': entries}])
        assert result['entity']['f1'] == 1
        assert result['diagnostics']['duplicate_entities'] == 1
        assert result['rates']['invalid_entity_rate'] == 0.5


@pytest.mark.parametrize('prediction', [
    {'id': 'sample:0', 'entities': []},
    {'id': 'sample:0', 'status': 'ok', 'entities': {}},
    {'id': 'sample:0', 'status': 'ok', 'raw_bio_tags': ['O']},
    {'id': 'sample:0', 'status': 'ok', 'raw_bio_tags': ['X'] * 4},
    {**predict(), 'raw_bio_tags': ['B-name', 'I-name', 'O', 'O']},
])
def test_malformed_file_stops_scoring(prediction):
    with pytest.raises(PredictionFileError):
        evaluate([gold()], [prediction])


def test_serialized_predictions_recompute_identically(tmp_path):
    predictions = [predict((0, 2, 'organization'))]
    path = tmp_path / 'predictions.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in predictions), encoding='utf-8')
    restored = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert evaluate([gold()], predictions) == evaluate([gold()], restored)


def test_unlabeled_or_nested_gold_rejected():
    for field, value in [('has_labels', False), ('bio_compatible', False)]:
        with pytest.raises(ValueError):
            evaluate([{**gold(), field: value}], [predict()])


def test_unicode_positions_do_not_use_utf16_or_bytes():
    row = gold(text='𠮷😀北', tags=['B-name', 'I-name', 'O'])
    result = evaluate([row], [predict((0, 2, 'name'))])
    assert result['token']['tp'] == 2 and result['entity']['tp'] == 1
