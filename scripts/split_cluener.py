"""Deterministic capacity-constrained iterative multilabel split, seed 42."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import random

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['address','book','company','game','government','movie','name','organization','position','scene']

def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def split_indices(rows, test_size=1209, seed=42):
    """Rare-label-first iterative assignment; exact capacities, no single-label stratify.

    Membership is binary per sentence, not entity occurrence count. At each
    step process the rarest remaining class, assign to the fold with greatest
    unmet class demand, then remaining capacity; random ties use a local RNG.
    """
    rng=random.Random(seed)
    memberships=[{e['type'] for e in row['entities']} for row in rows]
    remaining=set(range(len(rows)))
    active={label:{i for i,m in enumerate(memberships) if label in m} for label in LABELS}
    capacity=[len(rows)-test_size,test_size]
    demand=[{label:len(active[label])*n/len(rows) for label in LABELS} for n in capacity]
    folds=[[],[]]
    while remaining:
        available=[label for label in LABELS if active[label]]
        if not available:
            pending=sorted(remaining);rng.shuffle(pending)
            for i in pending:
                choices=[f for f in [0,1] if capacity[f]>0]
                largest=max(capacity[f] for f in choices)
                f=rng.choice([f for f in choices if capacity[f]==largest])
                folds[f].append(i);capacity[f]-=1
            break
        smallest=min(len(active[label]) for label in available)
        label=rng.choice([label for label in available if len(active[label])==smallest])
        pending=sorted(active[label]);rng.shuffle(pending)
        for i in pending:
            choices=[f for f in [0,1] if capacity[f]>0]
            highest=max(demand[f][label] for f in choices)
            choices=[f for f in choices if demand[f][label]==highest]
            largest=max(capacity[f] for f in choices)
            f=rng.choice([f for f in choices if capacity[f]==largest])
            folds[f].append(i);capacity[f]-=1;remaining.remove(i)
            for c in memberships[i]:
                demand[f][c]-=1;active[c].remove(i)
    return tuple(sorted(fold) for fold in folds)

def main():
    source=ROOT/'data/processed/cluener2020'
    train=read(source/'train.jsonl');dev=read(source/'dev.jsonl')
    groups=defaultdict(list)
    for row in train+dev:groups[row['text']].append(row)
    conflicts=[];duplicates=[]
    for text,rows in groups.items():
        if len(rows)<2:continue
        signatures={json.dumps(sorted((e['start'],e['end'],e['type']) for e in r['entities'])) for r in rows}
        entry={'ids':[r['id'] for r in rows], 'text':text}
        (conflicts if len(signatures)>1 else duplicates).append(entry)
    if conflicts or duplicates:
        raise RuntimeError('Duplicate text found; stop for review rather than apply label priority: '+json.dumps({'conflicts':conflicts,'duplicates':duplicates},ensure_ascii=False))
    assert len(train)==10748 and len(dev)==1343
    ti,si=split_indices(train)
    assert (ti,si)==split_indices(train)
    assert len(ti)==9539 and len(si)==1209 and set(ti).isdisjoint(si)
    assert set(ti)|set(si)==set(range(len(train)))
    folds={'train':[train[i] for i in ti],'dev':dev,'test':[train[i] for i in si]}
    folder=ROOT/'data/processed/cluener2020_project'
    ids_folder=ROOT/'data/splits/cluener2020';ids_folder.mkdir(parents=True,exist_ok=True)
    folder.mkdir(parents=True,exist_ok=True)
    stats={};files={}
    for split,rows in folds.items():
        target=folder/(split+'.jsonl')
        content=(source/'dev.jsonl').read_text(encoding='utf-8') if split=='dev' else ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
        if target.exists() and target.read_text(encoding='utf-8')!=content:raise RuntimeError('Frozen split differs: '+str(target))
        target.write_text(content,encoding='utf-8')
        (ids_folder/(split+'_ids.json')).write_text(json.dumps([r['id'] for r in rows],indent=2)+'\n',encoding='utf-8')
        occurrences=Counter(e['type'] for r in rows for e in r['entities'])
        coverage=Counter(c for r in rows for c in {e['type'] for e in r['entities']})
        assert set(coverage)==set(LABELS)
        stats[split]={'sentences':len(rows),'entity_counts':dict(occurrences),'sentence_coverage':dict(coverage),
                      'overlap_ids':[r['id'] for r in rows if not r['bio_compatible']]}
        files[split]={'path':target.relative_to(ROOT).as_posix(),'sha256':sha(target)}
    texts={s:{r['text'] for r in rows} for s,rows in folds.items()}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:assert not texts[a]&texts[b]
    assert (folder/'dev.jsonl').read_bytes()==(source/'dev.jsonl').read_bytes()
    report={'seed':42,'algorithm':'capacity-constrained rare-label-first iterative multilabel stratification v1',
            'stratification_target':'per-sentence binary presence of each entity type, not occurrence counts',
            'original_dev_unchanged':True,'duplicate_groups':0,'conflict_groups':0,
            'source_hashes':{s:sha(source/(s+'.jsonl')) for s in ['train','dev']},'files':files,'stats':stats,
            'source_ids_preserved':True,'unlabeled_official_test_used':False}
    (ids_folder/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    out=ROOT/'reports/tables/cluener_split_distribution.csv'
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['category','train_entities','dev_entities','test_entities','train_sentences','dev_sentences','test_sentences'])
        for c in LABELS:writer.writerow([c]+[stats[s]['entity_counts'].get(c,0) for s in folds]+[stats[s]['sentence_coverage'].get(c,0) for s in folds])
    lines=['# CLUENER 项目划分','', '固定原 dev，Test 从原 train 抽取；seed=42。Train/Dev/Test=9539/1343/1209（约78.9%/11.1%/10.0%）。',
           '划分前检查完整文本精确重复及标签冲突，均为0；没有删除记录。',
           '算法以每句所含的多个实体类别作为多热标签，按剩余最稀有类别优先迭代分配，兼顾目标类别需求与精确集合容量；同分使用固定随机数。不是单标签 stratify。',
           '分层针对句子类别覆盖，实体出现次数只做审计，不保证完全相同比例。Dev 固定，因此不对 Dev 重新平衡。',
           '', '| 类别 | Train实体数 | Dev实体数 | Test实体数 |','|---|---:|---:|---:|']
    for c in LABELS:lines.append('| '+c+' | '+' | '.join(str(stats[s]['entity_counts'].get(c,0)) for s in folds)+' |')
    lines+=['', '原始样本ID保留：项目 test 中的 train:xxx 表示来自官方训练集，不代表它仍属于项目训练集。',
            '官方无标签 test 独立保留；不进入本次划分或指标计算。',
            '重叠实体的实际归属：'+json.dumps({s:stats[s]['overlap_ids'] for s in folds})+'。未强制调换，也未修改标注；单层BIO模型仍需显式处理。',
            '复现命令：`.venv\\Scripts\\python.exe scripts\\split_cluener.py`。重复运行若结果不同会拒绝覆盖。',
            '通过检查：原训练样本全量且恰好分配一次、三集合文本互斥、Dev文件逐字节不变、每个集合覆盖10类、同种子重复划分一致。']
    (ROOT/'docs/project_split.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':
    from refresh_split import main as refresh_main
    refresh_main()
