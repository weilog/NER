"""Deterministic, balanced human-review packet. No model calls and no human-label fabrication."""
from collections import Counter
import csv
import io
import json
from pathlib import Path
from ner_experiment.common import ROOT,read_jsonl,load_json,save_json,object_hash,file_hash,bio_to_entities

METHODS=['dictionary','hmm','crf','llm_0','llm_2','llm_4','llm_8']
KINDS=['boundary','type','boundary_and_type','missed','spurious']
PACKET=ROOT/'reports/review/current_v3_20261024'
FIELDS=['review_id','sample_id','prediction_code','text','character_indices_json','gold_entities_json','predicted_entities_json',
        'focus_gold_json','focus_predicted_json','reviewer','human_kind','annotation_concern','explanation','suggested_action','reviewed_at']

def write_csv(path,rows,fields=FIELDS):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader();writer.writerows(rows)
    path.write_text(stream.getvalue(),encoding='utf-8-sig',newline='')

def select_events(events,seed):
    """35 primary errors (7 per kind) + 5 structural diagnostics, all unique sentences."""
    used=set();selected=[];method_counts=Counter()
    def take(kind):
        pool=[e for e in events if e['kind']==kind and e['sample_id'] not in used]
        if not pool:raise ValueError(f'Cannot fill required review stratum: {kind}')
        best=min(pool,key=lambda e:(method_counts[e['method']],object_hash([seed,e['key']])))
        selected.append(best);used.add(best['sample_id']);method_counts[best['method']]+=1
    for kind in ['invalid','overlap','text_mismatch','text_mismatch','text_mismatch']:take(kind)
    for _ in range(7):
        for kind in KINDS:take(kind)
    if len(selected)!=40 or len(used)!=40:raise ValueError('Expected 40 distinct sentence IDs')
    return sorted(selected,key=lambda e:object_hash([seed,'review_order',e['key']]))

def preserve_packet(folder,fingerprint):
    """Never overwrite an existing packet: human answers may already be present."""
    if not folder.exists():return False
    path=folder/'packet_manifest.json'
    if not path.exists() or load_json(path).get('source_fingerprint')!=fingerprint:
        raise RuntimeError('Existing review packet has another source version; preserve it and use a new packet directory')
    for name in ['independent_A_20.csv','independent_B_20.csv','adjudication_40.csv','private_key.json','README.md']:
        if not (folder/name).exists():raise RuntimeError('Incomplete review packet; inspect before regenerating')
    return True

def main():
    seed=load_json(ROOT/'configs/split_seed.json')['seed']
    gold=read_jsonl(ROOT/'data/processed/cluener2020_project/test.jsonl');lookup={r['id']:r for r in gold}
    folders={m:ROOT/'runs/final'/m for m in METHODS[:3]}
    folders.update({f'llm_{n}':ROOT/f'runs/llm_text_test_v1/shot_{n}' for n in [0,2,4,8]})
    sources={}
    for m,folder in folders.items():
        for path in [folder/'predictions.jsonl',ROOT/f'runs/current_v3_analysis/{m}/errors.jsonl']:
            sources[path.relative_to(ROOT).as_posix()]=file_hash(path)
    source_fingerprint=object_hash({'sources':sources,'seed':seed,'algorithm':'35-primary-balanced-plus-5-diagnostics-v1'})
    if preserve_packet(PACKET,source_fingerprint):
        print('PRESERVED existing packet, including human edits; API calls=0');return
    events=[];pred_maps={}
    for m,folder in folders.items():
        preds={r['id']:r for r in read_jsonl(folder/'predictions.jsonl')};pred_maps[m]=preds
        def add(sid,kind,g,p,index=0):
            key=object_hash([m,sid,kind,g,p,index])
            events.append({'method':m,'sample_id':sid,'kind':kind,'gold':g,'predicted':p,'key':key})
        for case in read_jsonl(ROOT/f'runs/current_v3_analysis/{m}/errors.jsonl'):
            for j,e in enumerate(case['errors']):add(case['id'],e['kind'],e['gold'],e['predicted'],j)
            norm=case['normalization']
            if norm['invalid_entities']:add(case['id'],'invalid',None,preds[case['id']]['entities'])
            if norm['overlap_pairs']:add(case['id'],'overlap',None,norm['overlapping_entities'])
        for sid,pred in preds.items():
            for j,e in enumerate(pred.get('entities',[])):
                a,b=e.get('start'),e.get('end');row=lookup[sid]
                if isinstance(e.get('text'),str) and type(a) is int and type(b) is int and 0<=a<b<=len(row['text']):
                    if row['text'][a:b]!=e['text']:add(sid,'text_mismatch',None,e,j)
    selected=select_events(events,seed)
    codes={m:f'P{i+1:02d}' for i,m in enumerate(METHODS)}
    rows=[];private=[]
    def j(value):return json.dumps(value,ensure_ascii=False)
    for i,e in enumerate(selected,1):
        row=lookup[e['sample_id']];pred=pred_maps[e['method']][e['sample_id']]
        ents=bio_to_entities(pred['raw_bio'])[0] if 'raw_bio' in pred else pred['entities']
        display_entities=[]
        for entity in ents:
            item=dict(entity);a,b=item.get('start'),item.get('end')
            item['span_text']=row['text'][a:b] if type(a) is int and type(b) is int and 0<=a<b<=len(row['text']) else None
            display_entities.append(item)
        rid=f'V3R{i:03d}'
        record={k:'' for k in FIELDS}
        record.update(review_id=rid,sample_id=row['id'],prediction_code=codes[e['method']],text=row['text'],
            character_indices_json=j(list(enumerate(row['text']))),gold_entities_json=j(row['entities']),
            predicted_entities_json=j(display_entities),focus_gold_json=j(e['gold']),focus_predicted_json=j(e['predicted']))
        rows.append(record);private.append({'review_id':rid,**e,'prediction_code':codes[e['method']]})
    PACKET.mkdir(parents=True)
    first=rows[:20];second=sorted(first,key=lambda r:object_hash([seed,'reviewer_B_order',r['review_id']]))
    write_csv(PACKET/'independent_A_20.csv',first);write_csv(PACKET/'independent_B_20.csv',second)
    master_fields=FIELDS+['reviewer_A_kind','reviewer_B_kind','adjudicators','adjudicated_kind','adjudication_notes','selected_for_report']
    write_csv(PACKET/'adjudication_40.csv',[{**r,**{k:'' for k in master_fields if k not in r}} for r in rows],master_fields)
    save_json(PACKET/'private_key.json',{'warning':'Do not read before independent judgments. Automatic attribution is not human ground truth.',
        'prediction_codes':codes,'items':private})
    manifest={'source_fingerprint':source_fingerprint,'sources':sources,'seed':seed,'algorithm':'35-primary-balanced-plus-5-diagnostics-v1',
        'selection_unit':'one sentence/model plus one focus event; sentence IDs unique across 40',
        'sampling':'balanced targeted error review, not a representative random estimate of population error frequencies',
        'review_items':40,'independent_items':20,'unique_sentence_ids':40,
        'by_method':dict(Counter(e['method'] for e in selected)),'by_automatic_kind':dict(Counter(e['kind'] for e in selected)),
        'independent_review_ids':[r['review_id'] for r in first],
        'initial_template_hashes':{n:file_hash(PACKET/n) for n in ['independent_A_20.csv','independent_B_20.csv','adjudication_40.csv']},
        'human_review_completed':False,'new_api_calls':0}
    save_json(PACKET/'packet_manifest.json',manifest)
    text='''# 新版V3人工错误审核包

本包来自固定Test的当前七组模型预测，40个不同句子，每项聚焦一处错误或结构问题。它是按错误类别与模型覆盖的定向抽样，不能用于估计总体错误比例。全部人工判断栏保持空白。

## 两人独立判断

1. A只打开independent_A_20.csv，B只打开independent_B_20.csv；两份包含相同20个review_id，但B的顺序不同。
2. 独立阶段不要互看答案，不打开private_key.json，也不先查看模型成绩或自动归因。prediction_code为模型匿名编号。
3. 阅读原文、字符索引、完整金标/预测和focus字段；[start,end)为左闭右开。null表示该侧没有对应实体；结构问题focus可能是列表或原始对象。
4. 每人填写reviewer、human_kind、annotation_concern、explanation、suggested_action、reviewed_at。annotation_concern可写yes/no/unclear。允许发现标注疑点，不要因为金标已给出就声称其语义必然正确。

human_kind建议：boundary、type、boundary_and_type、missed、spurious、invalid、overlap、text_mismatch、annotation_issue、unclear。可在explanation注明多个相关问题，但以focus为本项主判断。

## 共同复核

完成独立初判后，将20项两人的判断按review_id填入adjudication_40.csv的reviewer_A_kind和reviewer_B_kind，保留各自原表。再共同核验全部40项，填写adjudicators、adjudicated_kind、adjudication_notes。分歧必须解释；不要把自动标签直接复制成人工结论。

从共同确认案例中挑选至少6个用于报告，覆盖至少3类原因，在selected_for_report写yes。该列不是已完成声明。之后才可查看private_key.json，对比自动配对与人工判断。

## 文件保护

prepare_current_review.py重复执行时保留已有CSV及人工填写，不覆盖。不要删除审核包重新生成来改变样本。旧reports/review中的历史审核表不变。

程序只负责准备表格，没有替两名成员完成审核或给出人工结论。原始语料与审核材料仅本地保存。
'''
    (PACKET/'README.md').write_text(text,encoding='utf-8')
    print('PREPARED 40 distinct sentences; two blind 20-item forms; human answers blank; API calls=0')

if __name__=='__main__':main()
