"""Small, explicitly constructed verification examples plus one actual CRF feature record.

These are implementation checks, not new benchmark runs or invented experiment scores.
"""
import itertools
import numpy as np
from ner_experiment.common import ROOT, project_data, save_json
from ner_experiment.models.baselines import viterbi, legal_masks, features
from ner_experiment.evaluation.metrics import evaluate


def collect():
    tags=['O','B-name','I-name']
    pi=np.log([.6,.3,.1])
    transition=np.log([[.6,.3,.1],[.2,.2,.6],[.3,.2,.5]])
    emissions=np.log([[.6,.2,.2],[.2,.5,.3],[.1,.2,.7]])
    def score(path):
        return float(pi[path[0]]+sum(emissions[t,s] for t,s in enumerate(path))+
                     sum(transition[a,b] for a,b in zip(path,path[1:])))
    checks=[]
    for constrained in [False,True]:
        start,legal=legal_masks(tags)
        paths=[p for p in itertools.product(range(3),repeat=3)
               if not constrained or (start[p[0]] and all(legal[a,b] for a,b in zip(p,p[1:])))]
        brute=max(paths,key=score)
        dynamic=viterbi(pi,transition,emissions,constrained,tags)
        if list(brute)!=dynamic:raise RuntimeError('Viterbi/exhaustive verification failed')
        a,b=pi.copy(),transition.copy()
        if constrained:a[~start]=-np.inf;b[~legal]=-np.inf
        delta=a+emissions[0];states=[delta.tolist()];parents=[]
        for e in emissions[1:]:
            candidates=delta[:,None]+b;parent=candidates.argmax(axis=0)
            delta=candidates[parent,np.arange(3)]+e
            states.append(delta.tolist());parents.append(parent.tolist())
        checks.append({'constrained':constrained,'enumerated_paths':len(paths),
                       'path':[tags[i] for i in dynamic],'log_score':score(dynamic),
                       'delta':[[None if not np.isfinite(x) else x for x in row] for row in states],
                       'backpointers':parents,'matches_exhaustive':True})
    actual=next(r for r in project_data('test') if r['id']=='train:50')
    position=1;actual_features=features(actual)[position]
    gold=[{'id':'unit:overlap','text':'北京大学','tokens':list('北京大学'),
           'entities':[{'start':0,'end':4,'type':'organization'}]}]
    overlapping=[{'id':'unit:overlap','status':'ok','entities':[
        {'start':0,'end':4,'type':'organization'},{'start':0,'end':2,'type':'name'}]}]
    overlap,_,_=evaluate(gold,overlapping)
    invalid=[{'id':'unit:overlap','status':'ok','entities':[
        {'start':0,'end':4,'type':'organization'},{'start':0,'end':2,'type':'unknown_type'}]}]
    bad,_,_=evaluate(gold,invalid)
    assert [overlap['token'][k] for k in ['tp','fp','fn']]==[4,2,0]
    assert [overlap['entity'][k] for k in ['tp','fp','fn']]==[1,1,0]
    assert [bad['entity_with_invalid_fp'][k] for k in ['tp','fp','fn']]==[1,1,0]
    result={'scope':'Explicitly constructed implementation checks; not CLUENER benchmark scores',
        'viterbi':{'tags':tags,'initial_probabilities':np.exp(pi).tolist(),
                   'transition_probabilities':np.exp(transition).tolist(),
                   'emission_probabilities_per_step':np.exp(emissions).tolist(),'checks':checks},
        'actual_crf_feature':{'id':actual['id'],'text':actual['text'],'character_position':position,
                              'character':actual['tokens'][position],'features':actual_features},
        'overlap_example':{'gold':gold,'predictions':overlapping,'token':overlap['token'],'entity':overlap['entity']},
        'invalid_example':{'predictions':invalid,'token':bad['token'],'entity':bad['entity'],
                            'entity_with_invalid_fp':bad['entity_with_invalid_fp']}}
    save_json(ROOT/'reports/tables/course_report_revision/implementation_evidence.json',result)
    return result


if __name__=='__main__':
    result=collect()
    for row in result['viterbi']['checks']:
        print(row['constrained'],row['enumerated_paths'],row['path'],row['log_score'])
