"""Build a <=50 MB source-only submission ZIP from an explicit allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ROOT_FILES={'README.md','pyproject.toml','requirements-lock.txt','requirements-figure-qa.txt','.python-version','.env.example','.gitattributes','.gitignore','项目摘要.md','AI使用声明.md','资源链接.md','报告.pdf'}
EXTRA_FILES={'reproducibility/manifest.json','reproducibility/source.json','reports/demo_inputs_manifest.json','reports/tables/network_recovery/main_results.csv'}
CODE_FOLDERS={'src','scripts','tests','docs','configs'}
TEXT_SUFFIXES={'.py','.md','.json','.ps1','.cjs','.js','.txt','.toml','.yaml','.yml'}
FORBIDDEN={'.git','.venv','venv','__pycache__','.pytest_cache','data','runs','artifacts','checkpoint','checkpoints'}


def eligible(relative):
    name=relative.as_posix()
    if any(part.lower() in FORBIDDEN for part in relative.parts):return False
    if relative.name.startswith('.env') and relative.name!='.env.example':return False
    if name in EXTRA_FILES:return True
    if len(relative.parts)==1:
        return relative.name in ROOT_FILES or (relative.name.startswith('贡献声明_') and relative.suffix=='.md')
    return relative.parts[0] in CODE_FOLDERS and relative.suffix.lower() in TEXT_SUFFIXES


def collect(root):
    root=Path(root).resolve();files=[]
    for path in root.rglob('*'):
        if path.is_file() and eligible(path.relative_to(root)):
            if not path.resolve().is_relative_to(root):raise RuntimeError('Source symlink resolves outside root')
            files.append(path)
    return sorted(files)


def secret_values(path):
    if path is None or not path.is_file():return []
    values=[]
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if '=' not in line or line.strip().startswith('#'):continue
        key,value=line.split('=',1);value=value.strip().strip('\"\'')
        if key.strip().endswith('API_KEY') and value and len(value)>=12:values.append(value.encode('utf-8'))
    return values


def build(root,target,secrets):
    root=Path(root).resolve();target=Path(target).resolve();target.parent.mkdir(parents=True,exist_ok=True)
    files=collect(root)
    if not files:raise RuntimeError('No eligible source files')
    temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.building');hashes={}
    try:
        with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for path in files:
                relative=path.relative_to(root).as_posix();body=path.read_bytes()
                if any(value and value in body for value in secrets):raise RuntimeError('Credential value detected in source file: '+relative)
                if path.name=='.env.example':
                    for line in body.decode('utf-8-sig').splitlines():
                        if '=' in line and not line.strip().startswith('#'):
                            key,value=line.split('=',1)
                            if key.strip().endswith('API_KEY') and value.strip().strip('\"\''):
                                raise RuntimeError('Credential template must contain empty API keys')
                hashes[relative]=hashlib.sha256(body).hexdigest();z.writestr(relative,body)
            z.writestr('SOURCE_PACKAGE_MANIFEST.json',json.dumps({'schema':'cluener-source-zip-1','files_sha256':hashes,
                'excluded':'data, weights, material parts, raw API responses, venv, video, actual credentials',
                'scope':'Source-only package. External materials and complete host scoring still require validation.'},ensure_ascii=False,indent=2))
        if temporary.stat().st_size>50_000_000:raise RuntimeError('Source ZIP exceeds 50 MB upload limit')
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:raise RuntimeError('ZIP CRC check failed')
            if set(z.namelist())!=set(hashes)|{'SOURCE_PACKAGE_MANIFEST.json'} or len(set(z.namelist()))!=len(z.namelist()):
                raise RuntimeError('ZIP member inventory mismatch')
            for name,digest in hashes.items():
                if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise RuntimeError('ZIP member SHA256 mismatch: '+name)
                if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Source changed during packaging: '+name)
        temporary.replace(target)
    finally:
        if temporary.exists():temporary.unlink()
    result={'status':'PASS','archive':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'source_files':len(files),'zip_members':len(files)+1,'all_member_hashes_verified':True,'crc_verified':True,
            'actual_credentials_scanned':bool(secrets),'source_only':True,'upload_limit_bytes':50_000_000}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--leader-id',default='20246118');parser.add_argument('--leader-name',default='王学锋')
    parser.add_argument('--project',default='CLUENER2020命名实体识别')
    parser.add_argument('--secret-env-file',type=Path,default=ROOT/'.env')
    args=parser.parse_args()
    for value in [args.leader_id,args.leader_name,args.project]:
        if not value or re.search(r'[\\/:*?"<>|]',value):raise RuntimeError('Invalid submission filename component')
    target=args.output or ROOT.parent/(args.leader_id+'_'+args.leader_name+'_'+args.project+'.zip')
    result=build(ROOT,target,secret_values(args.secret_env_file))
    (target.parent/(target.stem+'_源码包校验.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
