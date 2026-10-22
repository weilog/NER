"""Convert the supplied CLUENER archive without filtering or resplitting."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import zipfile

LABELS = ['address','book','company','game','government','movie','name','organization','position','scene']

def dump(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def convert(row, split, index):
    text=row['text']
    result={'id':f'{split}:{index}','source_id':row.get('id'), 'text':text,'tokens':list(text),'has_labels':'label' in row}
    if 'label' not in row:
        result.update(entities=None,bio_tags=None,bio_compatible=None)
        return result
    entities=[]
    for category,mentions in row['label'].items():
        if category not in LABELS: raise ValueError(f'Unexpected category: {category}')
        for mention,positions in mentions.items():
            for start,end in positions:
                if not 0<=start<=end<len(text) or text[start:end+1]!=mention:
                    raise ValueError(f'Invalid span: {split}:{index}')
                entities.append({'start':start,'end':end+1,'type':category,'text':mention})
    entities.sort(key=lambda e:(e['start'],e['end'],e['type']))
    occupied=set(); overlap=False
    for entity in entities:
        span=set(range(entity['start'],entity['end']))
        overlap |= bool(occupied & span)
        occupied |= span
    bio=None
    if not overlap:
        bio=['O']*len(text)
        for e in entities:
            bio[e['start']]='B-'+e['type']
            bio[e['start']+1:e['end']]=['I-'+e['type']]*(e['end']-e['start']-1)
    result.update(entities=entities,bio_tags=bio,bio_compatible=not overlap)
    return result

def main(root):
    archive=root/'data/raw/cluener2020/cluener_public.zip'
    raw=root/'data/raw/cluener2020'; processed=root/'data/processed/cluener2020'
    meta=root/'data/manifests/cluener2020'
    for p in [raw,processed,meta]:p.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list); stats={}; anomalies=[]; hashes={}
    with zipfile.ZipFile(archive) as z:
        for name in ['train.json','dev.json','test.json','README.md','cluener_predict.json']:
            body=z.read(name)
            path=raw/name
            if path.exists() and path.read_bytes()!=body:raise ValueError(f'Raw file differs: {name}')
            path.write_bytes(body)
        for split in ['train','dev','test']:
            body=z.read(split+'.json')
            rows=[json.loads(line) for line in body.decode('utf-8-sig').splitlines() if line.strip()]
            converted=[convert(row,split,i) for i,row in enumerate(rows)]
            raw_out=raw/(split+'.jsonl')
            raw_out.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
            assert [json.loads(s) for s in raw_out.read_text(encoding='utf-8').splitlines()]==rows
            out=processed/(split+'.jsonl')
            content=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in converted)
            out.write_text(content,encoding='utf-8')
            for row in converted:
                groups[row['text']].append({'id':row['id'],'entities':row['entities']})
                if row['bio_compatible'] is False:anomalies.append(row)
            stats[split]={'sentences':len(rows),'characters':sum(len(r['text']) for r in rows),
                          'labeled_sentences':sum(r['has_labels'] for r in converted),
                          'entities':sum(len(r['entities'] or []) for r in converted) if all(r['has_labels'] for r in converted) else None,
                          'entity_counts':dict(Counter(e['type'] for r in converted for e in (r['entities'] or []))),
                          'overlapping_sentences':sum(r['bio_compatible'] is False for r in converted)}
            hashes[split]={'raw_sha256':hashlib.sha256(body).hexdigest(),'processed_sha256':hashlib.sha256(content.encode()).hexdigest()}
    duplicates=[]
    for text,items in groups.items():
        if len(items)<2:continue
        labeled=[x['entities'] for x in items if x['entities'] is not None]
        duplicates.append({'text':text,'occurrences':items,'cross_split':len({x['id'].split(':')[0] for x in items})>1,
                           'labeled_annotation_conflict':any(x!=labeled[0] for x in labeled[1:])})
    audit={'source':'User-provided Downloads/cluener_public.zip','license':'Not specified in supplied archive README; not inferred from code license',
           'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'stats':stats,'hashes':hashes,
           'policy':'No deduplication, sampling, relabeling or resplitting. Original inclusive end converted to exclusive end.',
           'duplicate_groups':len(duplicates),'cross_split_duplicate_groups':sum(x['cross_split'] for x in duplicates),
           'annotation_conflict_groups':sum(x['labeled_annotation_conflict'] for x in duplicates)}
    dump(meta/'audit.json',audit); dump(meta/'duplicates.json',duplicates);dump(meta/'overlaps.json',anomalies)
    dump(meta/'labels.json',{'entity_types':LABELS,'bio_labels':['O']+[prefix+label for prefix in ['B-','I-'] for label in LABELS]})
    report=['# CLUENER2020 数据准备','', '原始划分完整保留，未去重、未抽样、未重新划分。',
            '用户提供的压缩包已复制归档，并保留原始 JSON；原 JSON 实际为逐行 JSON，同时提供原字段不变的 .jsonl。','',
            '| 划分 | 文本数 | 带标签文本数 | 实体出现次数 | 重叠实体文本数 |','|---|---:|---:|---:|---:|']
    for split,s in stats.items():report.append(f"| {split} | {s['sentences']} | {s['labeled_sentences']} | {s['entities']} | {s['overlapping_sentences']} |")
    report += ['',f"精确重复文本组：{len(duplicates)}；跨划分重复组：{audit['cross_split_duplicate_groups']}；带标签记录间冲突组：{audit['annotation_conflict_groups']}。仅记录，未删除。",
               '', '原实体位置是字符级闭区间 [start,end]；标准格式改为 [start,end)，并逐实体校验实体文字与原文切片一致。',
               '重叠实体完整保留，bio_tags=null、bio_compatible=false，单层 BIO 模型的处理策略需在训练前明确，不静默覆盖。',
               'test 没有真实标签，entities/bio_tags=null，不能当成没有实体的负样本或直接用于有监督评分。cluener_predict.json 是随包预测示例，不能当测试真值。',
               '当前未指定独立的项目最终评估集；在划分协议确定前不进行正式选参或测试。',
               '许可：随包 README 未明确说明数据许可；保留来源记录，不自行标成开放许可证。',
               '', '运行：`.venv\\Scripts\\python.exe scripts\\prepare_cluener.py`。详细审计、重复和重叠样本位于 data/manifests/cluener2020。']
    (root/'docs/data_preparation.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=True,indent=2))

if __name__=='__main__':main(Path(__file__).resolve().parents[1])
