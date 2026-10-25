import numpy as np
import pytest
from ner_experiment.experiments.index_analysis import NAMES, f1, paired_contrasts


def test_global_f1_is_not_mean_of_sentence_f1():
    counts = np.array([[1,0,0],[0,0,9]])
    assert f1(counts.sum(axis=0)) == pytest.approx(2/11)
    assert f1(counts).mean() == .5
    assert f1(np.array([0,0,0])) == 0


def test_paired_zero_effect_and_known_interaction():
    def rows(tp,fp,fn,position):
        return [{'id':str(i),'token':[tp,fp,fn],'entity':[tp,fp,fn],
                 'surface':[tp,fp,fn],'position_error':position} for i in range(3)]
    arms = {name:rows(1,0,0,0) for name in NAMES}
    zero = paired_contrasts(arms,123,100)
    assert all(item['difference'] == item['ci95_low'] == item['ci95_high'] == 0 for item in zero)
    arms[NAMES[0]] = rows(0,1,1,1)
    actual = paired_contrasts(arms,123,100)
    effect = next(item for item in actual if item['contrast']=='B-A' and item['endpoint']=='entity_f1')
    assert effect['difference'] == effect['ci95_low'] == effect['ci95_high'] == 1
    interaction = next(item for item in actual if item['contrast']=='(D-C)-(B-A)' and item['endpoint']=='position_error_rate')
    assert interaction['difference'] == 1
    arms[NAMES[-1]][0]['id'] = 'wrong-order'
    with pytest.raises(ValueError,match='identical'):
        paired_contrasts(arms,123,100)
