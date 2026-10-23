import pytest
from ner_experiment.evaluation.metrics import evaluate, paired_bootstrap, error_pairs

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
    m,_,_=evaluate([row()],[pred([a,a,{'start':0,'end':2,'type':'address'},{'start':0,'end':99,'type':'name'},{'start':0,'end':1,'type':'unknown'}])])
    assert m['entity']['tp']==1 and m['entity']['fp']==3
    assert m['token']['f1']==1
    assert m['diagnostics']['duplicate_entities']==1
    assert m['diagnostics']['overlap_projection_samples']==1

@pytest.mark.parametrize('predictions',[[],[pred(),pred()],[pred(),{'id':'extra'}]])
def test_id_errors(predictions):
    with pytest.raises(ValueError):evaluate([row()],predictions)

def test_bootstrap_and_pairing():
    _,counts,_=evaluate([row()],[pred()])
    assert paired_bootstrap(counts,counts,123)['ci95']==[0,0]
    errors=error_pairs({(0,4,'name'),(6,8,'name')},{(0,3,'name'),(6,8,'company')})
    assert sorted(e['kind'] for e in errors)==['boundary','type']
