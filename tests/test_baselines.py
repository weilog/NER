import itertools
import numpy as np
from ner_experiment.models.baselines import viterbi, legal_masks, Dictionary, HMM, features
from ner_experiment.common import TAGS

def test_hmm_supervised_counts_and_unknown_characters():
    rows=[{'tokens':list('aab'),'bio_tags':['B-name','I-name','O']},
          {'tokens':list('ac'),'bio_tags':['O','B-name']}]
    model=HMM(alpha=.1,min_freq=2).fit(rows)
    assert model.vocabulary=={'a':1}
    k=len(TAGS); b=TAGS.index('B-name'); i=TAGS.index('I-name'); o=TAGS.index('O')
    initial=np.full(k,.1); initial[b]+=1; initial[o]+=1
    transition=np.full((k,k),.1)
    transition[b,i]+=1; transition[i,o]+=1; transition[o,b]+=1
    emission=np.full((k,2),.1)
    emission[b,1]+=1; emission[i,1]+=1; emission[o,0]+=1; emission[o,1]+=1; emission[b,0]+=1
    np.testing.assert_allclose(np.exp(model.initial),initial/initial.sum())
    np.testing.assert_allclose(np.exp(model.transition),transition/transition.sum(axis=1,keepdims=True))
    np.testing.assert_allclose(np.exp(model.emission),emission/emission.sum(axis=1,keepdims=True))
    assert len(model.predict({'id':'unknown','tokens':['z']})['raw_bio'])==1

def test_bio_constraints_between_entity_types():
    start,trans=legal_masks(TAGS)
    b=TAGS.index('B-name'); i=TAGS.index('I-name'); o=TAGS.index('O')
    assert not start[i] and not trans[o,i]
    assert trans[b,i] and trans[i,i]
    assert not trans[TAGS.index('B-company'),i]
    assert trans[i,b] and trans[i,o]

def test_crf_features_do_not_use_gold_labels():
    row={'tokens':list('A1中。'),'bio_tags':['O']*4,'entities':[]}
    altered=dict(row,bio_tags=['B-name']*4,entities=[{'start':0,'end':4,'type':'name'}])
    assert features(row)==features(altered)
    assert features(row)[0]['char-1']=='<BOS>'
    assert features(row)[-1]['char1']=='<EOS>'

def test_viterbi_against_enumeration():
    tags=['O','B-name','I-name']; rng=np.random.default_rng(7)
    for constrained in [False,True]:
        for length in [1,2,4]:
            initial=rng.normal(size=3); trans=rng.normal(size=(3,3)); emission=rng.normal(size=(length,3))
            start,legal=legal_masks(tags)
            paths=[p for p in itertools.product(range(3),repeat=length) if not constrained or (start[p[0]] and all(legal[a,b] for a,b in zip(p,p[1:])))]
            def score(p):return initial[p[0]]+sum(emission[i,s] for i,s in enumerate(p))+sum(trans[a,b] for a,b in zip(p,p[1:]))
            assert viterbi(initial,trans,emission,constrained,tags)==list(max(paths,key=score))

def test_dictionary_longest_and_type_tie():
    training=[{'text':'北京大学','entities':[{'start':0,'end':2,'type':'address'},{'start':0,'end':4,'type':c}]} for c in ['organization','company']]
    m=Dictionary().fit(training)
    assert m.predict({'id':'x','text':'北京大学','tokens':list('北京大学')})['entities']==[{'start':0,'end':4,'type':'company'}]
