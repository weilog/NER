"""Create a single external resource ZIP without source files or unrelated history."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import uuid
import zipfile
from build_submission_zip import collect, secret_values
from external_resources import sha256, resource_path

ROOT=Path(__file__).resolve().parents[1]
DATA_PREFIXES=['data/raw/cluener2020/','data/processed/cluener2020/','data/processed/cluener2020_project/',
               'data/manifests/cluener2020/','data/splits/cluener2020/']
RUN_PREFIXES=['runs/baselines/','runs/final/dictionary/','runs/final/hmm/','runs/final/crf/',
              'runs/llm_text_test_v1/','runs/glm_flashx_test_v1/','runs/mimo_flash_test_v1/','runs/network_recovery_20261030/']
NECESSARY={'runs/integrated_v1_analysis/verification.json','runs/llm/selection.json','runs/llm/interface_frozen.json'}
NECESSARY.update(f'runs/evaluation_versions/v3/final/{m}/metrics.json' for m in ['dictionary','hmm','crf'])


def include(name):
    return resource_path(name) and (any(name.startswith(p) for p in DATA_PREFIXES+RUN_PREFIXES)
        or name in NECESSARY or (name.startswith('runs/') and name.endswith('/ledger.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-archive',type=Path,required=True)
    parser.add_argument('--input-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--secret-env-file',type=Path)
    args=parser.parse_args()
    if sha256(args.input_archive)!=args.input_sha256:raise RuntimeError('Verified original archive SHA256 mismatch')
    sources={p.relative_to(ROOT).as_posix():p for p in collect(ROOT)}
    snapshot=json.loads((ROOT/'reports/demo_inputs_manifest.json').read_text(encoding='utf-8-sig'))
    secrets=secret_values(args.secret_env_file)
    target=args.output.resolve();target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.building')
    inventory={};skipped_duplicates=[]
    try:
        with zipfile.ZipFile(args.input_archive) as original,zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as output:
            old_inventory=json.loads(original.read('PACKAGE_MANIFEST.json'))['sha256']
            names=sorted(n for n in original.namelist() if include(n))
            for index,name in enumerate(names,1):
                body=original.read(name);digest=hashlib.sha256(body).hexdigest()
                if digest!=old_inventory[name]:raise RuntimeError('Original member digest differs: '+name)
                if name in sources:
                    if sha256(sources[name])!=digest:raise RuntimeError('Source/resource content conflict: '+name)
                    skipped_duplicates.append(name);continue
                if any(value in body for value in secrets):raise RuntimeError('Credential detected in selected resource: '+name)
                inventory[name]={'bytes':len(body),'sha256':digest};output.writestr(name,body)
                if index==1 or index%4000==0 or index==len(names):print('Resource files packaged',index,'/',len(names),flush=True)
            for name,digest in snapshot['files_sha256'].items():
                if name in sources:
                    if sha256(sources[name])!=digest:raise RuntimeError('Frozen source changed: '+name)
                elif name not in inventory or inventory[name]['sha256']!=digest:raise RuntimeError('Required frozen resource missing: '+name)
            selection=json.loads(original.read('runs/baselines/selection.json'))
            for run in selection['runs']:
                name=f"runs/baselines/{run['name']}/model.pkl"
                if name not in inventory or inventory[name]['sha256']!=run['model_sha256']:
                    raise RuntimeError('Trained baseline model missing or changed: '+name)
            payload=json.dumps({'schema':'cluener-resource-members-1','files':inventory,
                'scope':'Fixed CLUENER data, trained baselines, current predictions/API records and necessary audit metadata only',
                'not_included':'source, configs, tests, docs, report PDFs, source ZIP, old project archives, venv, credentials'},ensure_ascii=False,indent=2).encode('utf-8')
            output.writestr('MATERIALS_MANIFEST.json',payload)
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:raise RuntimeError('Resource ZIP CRC failed')
            overlap=set(z.namelist())&set(sources)
            if overlap:raise RuntimeError('Resource overlaps source: '+repr(sorted(overlap)))
            for name,entry in inventory.items():
                if hashlib.sha256(z.read(name)).hexdigest()!=entry['sha256']:raise RuntimeError('Resource ZIP member differs: '+name)
        temporary.replace(target)
    finally:
        if temporary.exists():temporary.unlink()
    catalog={'schema':'cluener-external-resources-1','archive_name':target.name,'archive_bytes':target.stat().st_size,
             'archive_sha256':sha256(target),'manifest_sha256':hashlib.sha256(payload).hexdigest(),'file_count':len(inventory),
             'source_overlap_files':0,'frozen_input_manifest_sha256':sha256(ROOT/'reports/demo_inputs_manifest.json')}
    (ROOT/'reproducibility/manifest.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'reproducibility/source.json').write_text(json.dumps({'schema':'cluener-submission-source-2','download_url':None,
             'publication_status':'not_uploaded','notice':'One external ZIP; download manually and place beside extracted source directory, or use --materials / --url.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={'catalog':catalog,'selected_models':selection['selected'],'trained_models':sum(n.endswith('/model.pkl') for n in inventory),
             'source_path_overlap':0,'skipped_identical_source_paths':skipped_duplicates,'new_model_api_calls':0,
             'folder_counts':dict(Counter('/'.join(n.split('/')[:2]) for n in inventory)),
             'old_full_backup_preserved':True,'publication_performed':False}
    (target.parent/'资源包构建核验.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'resource_bytes':catalog['archive_bytes'],'resource_files':len(inventory),'models':summary['trained_models'],'source_overlap':0},ensure_ascii=True))

if __name__=='__main__':main()
