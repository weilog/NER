import itertools
import numpy as np
from ner_experiment.models.baselines import viterbi, legal_masks, Dictionary

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
