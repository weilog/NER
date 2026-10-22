from pathlib import Path
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def rows(split):
    return [json.loads(x) for x in (ROOT/'data/processed/cluener2020'/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]

def test_raw_conversion():
    with zipfile.ZipFile(ROOT/'data/raw/cluener2020/cluener_public.zip') as z:
        for split,n in [('train',10748),('dev',1343),('test',1345)]:
            original=[json.loads(x) for x in z.read(split+'.json').decode('utf-8').splitlines() if x.strip()]
            saved=[json.loads(x) for x in (ROOT/'data/raw/cluener2020'/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
            assert original==saved and len(saved)==n
            assert [x['text'] for x in rows(split)]==[x['text'] for x in original]

def test_spans_and_bio():
    for split in ['train','dev']:
        for row in rows(split):
            assert row['tokens']==list(row['text']) and row['has_labels']
            for e in row['entities']:
                assert row['text'][e['start']:e['end']]==e['text']
            if row['bio_compatible']:
                decoded=[]; i=0; tags=row['bio_tags']
                while i<len(tags):
                    if tags[i]=='O': i+=1; continue
                    assert tags[i].startswith('B-')
                    typ=tags[i][2:]; j=i+1
                    while j<len(tags) and tags[j]=='I-'+typ:j+=1
                    decoded.append((i,j,typ)); i=j
                assert decoded==[(e['start'],e['end'],e['type']) for e in row['entities']]

def test_overlapping_annotation_preserved():
    overlap=[r for r in rows('train') if not r['bio_compatible']]
    assert len(overlap)==1 and overlap[0]['id']=='train:1506'
    assert overlap[0]['bio_tags'] is None and len(overlap[0]['entities'])>=2

def test_unlabeled_is_not_negative():
    for row in rows('test'):
        assert row['has_labels'] is False
        assert row['entities'] is None and row['bio_tags'] is None
