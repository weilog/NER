import pytest
from ner_experiment.evaluation.metrics import evaluate, paired_bootstrap, error_pairs
from ner_experiment.evaluation.metrics_v2 import evaluate as evaluate_v2

def row(entities=None, text='北京大学'):
    return {'id':'x','text':text,'tokens':list(text),'entities':entities if entities is not None else [{'start':0,'end':4,'type':'organization'}]}

def pred(entities=None, **kwargs):
    return {'id':'x','status':'ok','entities':entities or [], **kwargs}

@pytest.mark.parametrize('entities,token,entity',[
    ([{'start':0,'end':4,'type':'organization'}],(4,0,0),(1,0,0)),
    ([{'start':0,'end':2,'type':'organization'}],(2,0,2),(0,1,1)),
    ([{'start':0,'end':4,'type':'company'}],(0,4,4),(0,1,1)),
    ([],(0,0,4),(0,0,1)),
])
def test_manual_cases(entities,token,entity):
    m,_,_=evaluate([row()],[pred(entities)])
    for kind,expected in [('token',token),('entity',entity)]:
        assert tuple(m[kind][k] for k in ['tp','fp','fn'])==expected

def test_bi_confusion_and_normalization():
    m,_,_=evaluate([row()],[pred(raw_bio=['B-organization','B-organization','I-organization','I-organization'])])
    assert (m['token']['tp'],m['token']['fp'],m['token']['fn'])==(3,1,1)
    assert m['entity']['f1']==0
    m,_,_=evaluate([row()],[pred(raw_bio=['I-organization']*4)])
    assert m['entity']['f1']==1 and m['illegal_bio_rate']==.25

def test_empty_spurious_failure():
    m,_,_=evaluate([row([])],[pred()]); assert m['token']['f1']==0 and m['character_accuracy']==1
    m,_,_=evaluate([row([])],[pred([{'start':0,'end':2,'type':'name'}])]); assert m['token']['fp']==2
    m,_,_=evaluate([row()],[pred(status='api_error')]); assert m['entity']['fn']==1 and m['failure_rate']==1

def test_invalid_overlap_duplicates():
    a={'start':0,'end':4,'type':'organization'}
    m,_,cases=evaluate_v2([row()],[pred([a,a,{'start':0,'end':2,'type':'address'},{'start':0,'end':99,'type':'name'},{'start':0,'end':1,'type':'unknown'}])])
    assert m['entity']['tp']==1 and m['entity']['fp']==0
    assert m['entity_before_normalization']['fp']==3
    assert m['entity']['f1']==1
    assert m['token']['f1']==1
    assert m['diagnostics']['duplicate_entities']==1
    assert m['diagnostics']['overlap_projection_samples']==1
    assert m['diagnostics']['invalid_entities']==2
    assert cases[0]['errors']==[]
    assert cases[0]['normalization']['removed_overlap_entities']==[(0,2,'address')]

def test_same_predictions_used_by_both_metrics():
    from copy import deepcopy
    predictions=[pred([{'start':0,'end':4,'type':'organization'},{'start':0,'end':2,'type':'name'}])]
    original=deepcopy(predictions)
    m,counts,cases=evaluate_v2([row()],predictions)
    assert m['protocol_version']=='v2'
    assert m['entity']['f1']==m['token']['f1']==1
    assert m['entity_before_normalization']['f1']==pytest.approx(2/3)
    assert m['entity_f1_normalization_delta']==pytest.approx(1/3)
    assert counts[0]['entity_before_normalization']==[1,1,0]
    assert predictions==original and cases[0]['raw_prediction']==original[0]

def test_overlap_priority_and_gold_independence():
    from ner_experiment.common import resolve_overlaps, LABELS
    # Earlier start wins even when a later span is longer; touching is legal.
    spans={(0,2,'name'),(1,5,'organization'),(2,4,'name')}
    kept,removed=resolve_overlaps(spans)
    assert kept=={(0,2,'name'),(2,4,'name')}
    assert removed=={(1,5,'organization')}
    # Same bounds: fixed category order wins, NOT the gold category.
    m,_,_=evaluate_v2([row()],[pred([{'start':0,'end':4,'type':'organization'},{'start':0,'end':4,'type':'name'}])])
    assert LABELS.index('name')<LABELS.index('organization')
    assert m['entity']['f1']==m['token']['f1']==0
    assert m['entity_before_normalization']['tp']==1

def test_all_invalid_predictions_remain_visible():
    m,_,cases=evaluate_v2([row()],[pred([{'start':0,'end':99,'type':'name'}])])
    assert m['entity']['fp']==m['token']['fp']==0
    assert m['entity']['fn']==1 and m['token']['fn']==4
    assert m['entity_before_normalization']['fp']==1
    assert cases[0]['normalization']['invalid_entities']==1

def test_bio_and_span_interfaces_agree():
    spans=[{'start':0,'end':4,'type':'organization'}]
    a,_,_=evaluate([row()],[pred(spans)])
    b,_,cases=evaluate([row()],[pred(raw_bio=['I-organization']*4)])
    assert a['entity']==b['entity'] and a['token']==b['token']
    assert cases[0]['raw_prediction']['raw_bio']==['I-organization']*4
    assert cases[0]['normalization']['illegal_bio_positions']==1

def test_preserved_v1_and_replay_dispatch():
    import importlib.util
    from ner_experiment.common import ROOT
    from ner_experiment.evaluation.metrics_v1 import evaluate as v1
    predictions=[pred([{'start':0,'end':4,'type':'organization'},{'start':0,'end':2,'type':'name'}])]
    old,_,_=v1([row()],predictions)
    assert old['entity']['f1']==pytest.approx(2/3) and old['token']['f1']==1
    spec=importlib.util.spec_from_file_location('replay',ROOT/'scripts/replay.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.evaluator_for(old) is v1
    assert module.evaluator_for({'protocol_version':'v2'}) is evaluate_v2
    assert module.evaluator_for({'protocol_version':'v3'}) is evaluate
    with pytest.raises(ValueError):module.evaluator_for({'protocol_version':'unknown'})

@pytest.mark.parametrize('predictions',[[],[pred(),pred()],[pred(),{'id':'extra'}]])
def test_id_errors(predictions):
    with pytest.raises(ValueError):evaluate([row()],predictions)

def test_bootstrap_and_pairing():
    _,counts,_=evaluate([row()],[pred()])
    assert paired_bootstrap(counts,counts,123)['ci95']==[0,0]
    errors=error_pairs({(0,4,'name'),(6,8,'name')},{(0,3,'name'),(6,8,'company')})
    assert sorted(e['kind'] for e in errors)==['boundary','type']

def test_v3_overlap_is_penalized_in_both_metrics():
    from copy import deepcopy
    predictions=[pred([{'start':0,'end':4,'type':'organization'},{'start':0,'end':2,'type':'name'}])]
    before=deepcopy(predictions)
    m,counts,cases=evaluate([row()],predictions)
    assert m['protocol_version']=='v3'
    assert counts[0]['token']==[4,2,0]
    assert counts[0]['entity']==[1,1,0]
    assert m['token']['f1']==pytest.approx(.8)
    assert m['entity']['f1']==pytest.approx(2/3)
    assert m['character_accuracy']==.5
    assert m['diagnostics']['overlapping_entities']==2
    assert m['diagnostics']['overlap_pairs']==1
    assert len(cases[0]['normalization']['kept_entities'])==2
    assert predictions==before and cases[0]['raw_prediction']==before[0]

def test_v3_same_type_prefix_shares_token_items():
    m,_,_=evaluate([row()],[pred([{'start':0,'end':4,'type':'organization'},{'start':0,'end':2,'type':'organization'}])])
    assert m['token']['f1']==1 # same BIO items deduplicate; entity boundaries still matter
    assert m['entity']['f1']==pytest.approx(2/3)

def test_v3_nested_gold_and_sentence_identity():
    from ner_experiment.common import entities_to_bio_set
    spans=[{'start':0,'end':4,'type':'organization'},{'start':1,'end':3,'type':'name'}]
    m,counts,_=evaluate([row(spans)],[pred(spans)])
    assert m['token']['f1']==m['entity']['f1']==1
    assert counts[0]['token']==[6,0,0]
    a=entities_to_bio_set({(0,2,'name')},'a')
    b=entities_to_bio_set({(0,2,'name')},'b')
    assert not a & b

def test_v3_invalid_and_duplicate_outputs_visible():
    a={'start':0,'end':4,'type':'organization'}
    m,_,cases=evaluate([row()],[pred([a,a,{'start':0,'end':99,'type':'name'}])])
    assert m['token']['f1']==m['entity']['f1']==1
    assert m['entity_with_invalid_fp']['f1']==pytest.approx(2/3)
    assert m['diagnostics']['invalid_entities']==1
    assert m['diagnostics']['duplicate_entities']==1
    assert len(cases[0]['raw_prediction']['entities'])==3

def test_v3_flat_bio_counts_equal_single_label_protocol():
    import itertools
    from ner_experiment.evaluation.metrics_v1 import evaluate as v1
    # Exhaustive short BIO sequences exercise B/I confusion and normalization.
    for tags in itertools.product(['O','B-name','I-name'],repeat=3):
        gold=row([{'start':0,'end':2,'type':'name'}],text='abc')
        predictions=[pred(raw_bio=list(tags))]
        old,_,_=v1([gold],predictions);new,_,_=evaluate([gold],predictions)
        assert new['token']==old['token']
        assert new['entity']==old['entity']
        assert new['character_accuracy']==old['character_accuracy']
