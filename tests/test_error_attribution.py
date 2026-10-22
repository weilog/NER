from pathlib import Path
import runpy

analysis = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/analyze_traditional_dev.py'))
attribute = analysis['attribute']


def test_one_to_one_error_pairing_and_conservation():
    gold = {(0, 4, 'name'), (5, 9, 'company'), (10, 12, 'address'), (20, 23, 'book')}
    predicted = {(0, 4, 'name'), (5, 8, 'company'), (10, 12, 'name'), (30, 32, 'movie')}
    errors = attribute(gold, predicted)
    assert [e['category'] for e in errors] == ['type', 'boundary', 'missed', 'spurious']
    assert sum(e['gold'] is not None for e in errors) == len(gold - predicted)
    assert sum(e['prediction'] is not None for e in errors) == len(predicted - gold)


def test_competing_predictions_pair_once_with_fixed_priority():
    gold = {(0, 4, 'name')}
    predicted = {(0, 4, 'company'), (0, 3, 'name'), (1, 4, 'name')}
    errors = attribute(gold, predicted)
    assert errors[0] == {'category': 'type', 'gold': (0, 4, 'name'), 'prediction': (0, 4, 'company')}
    assert [e['category'] for e in errors].count('spurious') == 2
    assert attribute(set(gold), set(reversed(sorted(predicted)))) == errors


def test_boundary_and_type_remains_separate():
    errors = attribute({(0, 4, 'name')}, {(1, 3, 'company')})
    assert errors == [{'category': 'boundary_and_type', 'gold': (0, 4, 'name'), 'prediction': (1, 3, 'company')}]


def test_character_errors_separate_bi_and_type_confusion():
    actual = analysis['character_errors'](['B-name', 'I-name', 'O', 'B-company', 'O'],
                                         ['B-name', 'B-name', 'B-name', 'O', 'B-company'])
    assert actual == {'token_bi_confusion': 1, 'token_missed': 1, 'token_spurious': 2}
    assert analysis['character_errors'](['B-name'], ['B-company']) == {'token_type_confusion': 1}
