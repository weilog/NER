"""Private reproducibility archive of all 15 methods; never calls model APIs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid
import zipfile
from dotenv import dotenv_values
from ner_experiment.common import ROOT, file_hash, load_json, save_json

FOLDERS=['src','scripts','tests','docs','configs','reports','data/raw/cluener2020',
         'data/processed/cluener2020','data/processed/cluener2020_project','data/manifests','data/splits',
         'runs/baselines','runs/llm','runs/llm_text_pilot_v1','runs/llm_text_test_v1',
         'runs/glm_flashx_test_v1','runs/mimo_flash_test_v1','runs/final',
         'runs/network_recovery_20261030','runs/demo_recordings','runs/evaluation_versions','runs/integrated_v1_analysis','runs/glm_network_recovery_plan','runs/integrated_v3_analysis','runs/current_v3_analysis']
ROOT_FILES=['README.md','pyproject.toml','requirements-lock.txt','requirements-figure-qa.txt',
            '.python-version','.env.example','.gitattributes','.gitignore']


def allowed(path):
    name=path.name.lower()
    return not (any(x.lower() in {'.git','.venv','__pycache__','.pytest_cache'} for x in path.parts)
        or (name.startswith('.env') and name!='.env.example')
        or path.suffix.lower() in {'.pyc','.pyo','.tmp','.pem','.key','.pfx','.lock'}
        or name=='process.lock')


def collect(root):
    root=root.resolve();files=set()
    for directory in FOLDERS:
        for p in (root/directory).rglob('*'):
            if p.is_file() and allowed(p):
                if not p.resolve().is_relative_to(root):raise RuntimeError('Archive input resolves outside project')
                files.add(p)
    files.update(root/n for n in ROOT_FILES if (root/n).is_file() and allowed(root/n))
    return sorted(files)


def verify_archive(path,expected):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(set(names))!=len(names) or set(names)!=set(expected)|{'PACKAGE_MANIFEST.json'}:
            raise RuntimeError('Archive member mismatch')
        if z.testzip() is not None:raise RuntimeError('Archive CRC failed')
        for name,digest in expected.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise RuntimeError('Archive SHA256 failed: '+name)


def build_archive(root,out,files,secrets):
    root=root.resolve();out.mkdir(parents=True,exist_ok=False)
    target=out/'cluener_15_methods_private.zip';temp=out/'cluener_15_methods_private.tmp'
    hashes={}
    try:
        with zipfile.ZipFile(temp,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for number,p in enumerate(files,1):
                if not allowed(p) or not p.resolve().is_relative_to(root):raise RuntimeError('Disallowed archive input')
                name=p.relative_to(root).as_posix();content=p.read_bytes()
                if any(secret and secret in content for secret in secrets):
                    raise RuntimeError('Credential found in candidate file; archive blocked: '+name)
                hashes[name]=hashlib.sha256(content).hexdigest();z.writestr(name,content)
                if number%4000==0:print(f'PACKED {number}/{len(files)}',flush=True)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps({'sha256':hashes,
                'scope':'Private local archive. Dataset license unresolved; do not publish corpus/raw responses.',
                'environment':'No venv included; rebuild from requirements-lock.txt using Python 3.13.',
                'new_api_calls':0},ensure_ascii=False,indent=2))
        for p in files:
            if file_hash(p)!=hashes[p.relative_to(root).as_posix()]:raise RuntimeError('Input changed while packaging')
        verify_archive(temp,hashes);temp.replace(target)
    except Exception:
        if temp.exists():temp.unlink()
        raise
    save_json(out/'files_sha256.json',hashes)
    save_json(out/'verification.json',{'verified_at':datetime.now(timezone.utc).isoformat(),
        'files':len(files),'archive_sha256':file_hash(target),'archive_bytes':target.stat().st_size,
        'all_member_sha256_checked':True,'zip_crc_checked':True,'source_files_unchanged':True,
        'credential_values_scanned':True,'actual_env_excluded':True,'new_api_calls':0,
        'visibility':'LOCAL PRIVATE; not a public submission or GitHub upload'})
    return target


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',type=Path)
    args=parser.parse_args()
    if args.verify:
        with zipfile.ZipFile(args.verify) as z:manifest=json.loads(z.read('PACKAGE_MANIFEST.json'))
        verify_archive(args.verify,manifest['sha256']);print('ARCHIVE VERIFIED');return
    from verify_v1_artifacts import main as verify_artifacts
    verify_artifacts()
    if any((ROOT/'runs').rglob('*.lock')):raise RuntimeError('Run lock exists; stop before snapshotting')
    for ledger in (ROOT/'runs').rglob('ledger.json'):
        if any(a['status']=='reserved' for a in load_json(ledger)['attempts']):
            raise RuntimeError('Unresolved in-flight ledger reservation')
    # Read credential values only to prevent accidental inclusion; never print or archive them.
    values=dict(os.environ)
    if (ROOT/'.env').exists():values.update(dotenv_values(ROOT/'.env',encoding='utf-8-sig',interpolate=False))
    secrets=[v.encode('utf-8') for k,v in values.items() if k.endswith('API_KEY') and v and len(v)>=12]
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
    out=ROOT/'runs/submission_packages'/tag
    files=collect(ROOT)
    print(f'PRIVATE SNAPSHOT: {len(files)} files, {sum(p.stat().st_size for p in files)/1048576:.1f} MiB before compression',flush=True)
    target=build_archive(ROOT,out,files,secrets)
    print('VERIFIED PRIVATE ARCHIVE:',target,flush=True)


if __name__=='__main__':main()
