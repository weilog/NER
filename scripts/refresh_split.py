"""Archive old project split, generate one persisted seed, constrain overlap to train."""
from pathlib import Path
import secrets
import zipfile
import hashlib
import importlib.util
from ner_experiment.common import ROOT, read_jsonl, save_json, load_json, write_jsonl, file_hash

def save_manifest(path, value):
    # Set iteration can change Counter key insertion order between processes.
    # Preserve committed bytes when semantic content is identical.
    if path.exists() and load_json(path)==value:return
    if (ROOT/'configs/frozen_experiment.json').exists():raise RuntimeError('Frozen manifest content would change')
    save_json(path,value)

def main():
    seedfile = ROOT/'configs/split_seed.json'
    if not seedfile.exists():
        archive = ROOT.parent/'NLP_archives'/'project_split_seed42_before_v2.zip'
        files = [p for folder in ['data/processed/cluener2020_project','data/splits/cluener2020'] for p in (ROOT/folder).rglob('*') if p.is_file()]
        files += [ROOT/p for p in ['configs/experiment.json','docs/project_split.md','reports/tables/cluener_split_distribution.csv'] if (ROOT/p).exists()]
        if archive.exists(): raise RuntimeError('Existing archive; inspect before generating a new seed')
        hashes = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files}
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for p in files: z.write(p, p.relative_to(ROOT).as_posix())
        with zipfile.ZipFile(archive) as z:
            for name, digest in hashes.items(): assert hashlib.sha256(z.read(name)).hexdigest() == digest
        save_json(seedfile, {'seed': secrets.randbelow(2**31), 'generation': 'secrets.randbelow once; not selected using outcomes', 'archive':str(archive), 'archive_sha256':file_hash(archive)})
    seed = load_json(seedfile)['seed']
    spec=importlib.util.spec_from_file_location('split_module',ROOT/'scripts/split_cluener.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=ROOT/'data/processed/cluener2020'; train=read_jsonl(source/'train.jsonl');dev=read_jsonl(source/'dev.jsonl')
    from collections import defaultdict
    by_text=defaultdict(list)
    for row in train+dev:by_text[row['text']].append(row)
    duplicates=[{'text':text,'records':rows,'conflicting_labels':len({str(r['entities']) for r in rows})>1}
                for text,rows in by_text.items() if len(rows)>1]
    if duplicates:
        save_json(ROOT/'runs/data_review/duplicates.json',duplicates)
        raise RuntimeError('Duplicate/conflicting texts require manual review; see runs/data_review/duplicates.json')
    reserved=[r for r in train if not r['bio_compatible']]; eligible=[r for r in train if r['bio_compatible']]
    assert len(reserved)==1
    a,b=module.split_indices(eligible,test_size=1209,seed=seed)
    assert (a,b)==module.split_indices(eligible,test_size=1209,seed=seed)
    folds={'train':sorted([eligible[i] for i in a]+reserved,key=lambda r:int(r['id'].split(':')[1])), 'dev':dev, 'test':[eligible[i] for i in b]}
    output=ROOT/'data/processed/cluener2020_project'; ids=ROOT/'data/splits/cluener2020'
    for split,rows in folds.items():
        write_jsonl(output/(split+'.jsonl'), rows)
        save_json(ids/(split+'_ids.json'),[r['id'] for r in rows])
    # Preserve original dev bytes, including formatting.
    (output/'dev.jsonl').write_bytes((source/'dev.jsonl').read_bytes())
    tuning_pool=dev
    _, subset=module.split_indices(tuning_pool,test_size=300,seed=seed)
    save_json(ids/'dev_tuning_ids.json',[dev[i]['id'] for i in subset])
    save_json(ids/'pilot_ids.json',[dev[i]['id'] for i in subset[:20]])
    from collections import Counter
    stats={s:{'sentences':len(rows),'entity_counts':dict(Counter(e['type'] for r in rows for e in r['entities'])),
              'sentence_coverage':dict(Counter(c for r in rows for c in {e['type'] for e in r['entities']})),
              'skipped_training_ids':[r['id'] for r in rows if not r['bio_compatible']]} for s,rows in folds.items()}
    save_manifest(ids/'manifest.json',{'seed':seed,'algorithm':'rare-label-first capacity-constrained iterative stratification v2; overlap forced into train',
              'source_hashes':{s:file_hash(source/(s+'.jsonl')) for s in ['train','dev']},'original_dev_unchanged':True,
              'stats':stats,'files':{s:{'path':(output/(s+'.jsonl')).relative_to(ROOT).as_posix(),'sha256':file_hash(output/(s+'.jsonl'))} for s in folds}})
    config=load_json(ROOT/'configs/experiment.json');config['dataset']['seed']=seed;config['evaluation']['seed']=seed
    config['baseline_methods']=['dictionary','hmm','crf'];config['dataset']['effective_train_size']=9538
    config['dataset']['overlap_policy']='preserve raw; fixed in train and excluded from all training and prompt examples'
    config['llm']['model']='deepseek-flash';save_json(ROOT/'configs/experiment.json',config)
    import csv
    with (ROOT/'reports/tables/cluener_split_distribution.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['category']+[s+'_entities' for s in folds]+[s+'_sentence_coverage' for s in folds])
        for c in module.LABELS:w.writerow([c]+[stats[s]['entity_counts'].get(c,0) for s in folds]+[stats[s]['sentence_coverage'].get(c,0) for s in folds])
    (ROOT/'docs/project_split.md').write_text(f'# Project split v2\n\nSeed: {seed}, generated once from the system RNG.\nTrain/Dev/Test: 9539/1343/1209; effective training: 9538.\nOriginal Dev preserved byte-for-byte. One nested sample is reserved in Train and excluded from model training and prompt examples.\nNo duplicate texts across labeled source splits; all original Train samples are assigned exactly once.\nDetailed counts and hashes: data/splits/cluener2020/manifest.json.\nReproduce: python scripts/refresh_split.py (reuses the saved seed).\nOld seed42 split is archived; do not run the old split_cluener.py entry point.\n',encoding='utf-8')
    print('New persisted seed:',seed);print({s:len(rows) for s,rows in folds.items()})

if __name__=='__main__':main()
