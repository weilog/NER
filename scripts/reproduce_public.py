"""Restore the complete PUBLIC materials shipped in this repository, then replay final results."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, name):
    value=PurePosixPath(name)
    if not name or '\\' in name or value.is_absolute() or '..' in value.parts or ':' in name:
        raise RuntimeError('Unsafe archive/part path: '+repr(name))
    target=(root/name).resolve()
    if not target.is_relative_to(root.resolve()):raise RuntimeError('Path escapes project: '+name)
    if target.name.startswith('.env') and target.name!='.env.example':
        raise RuntimeError('Actual credential file is forbidden: '+name)
    return target


def assemble(root, manifest):
    folder=root/'runs/public_reproduction';folder.mkdir(parents=True,exist_ok=True)
    target=safe_path(folder,manifest['archive_name'])
    if target.exists() and target.stat().st_size==manifest['archive_bytes'] and sha256(target)==manifest['archive_sha256']:
        print('公开材料完整包已存在，SHA256通过。',flush=True)
        return target
    parts=manifest['parts']
    if not parts or len({p['path'] for p in parts})!=len(parts):raise RuntimeError('Missing/duplicate public material parts')
    temporary=folder/'materials.assembling'
    whole=hashlib.sha256();total=0
    try:
        with temporary.open('wb') as output:
            for i,part in enumerate(parts,1):
                path=safe_path(root,part['path'])
                if not path.is_file():raise RuntimeError('Missing PUBLIC material part: '+part['path']+'; clone/download the complete repository')
                data=path.read_bytes()
                if len(data)!=part['bytes'] or hashlib.sha256(data).hexdigest()!=part['sha256']:
                    raise RuntimeError('Public part SHA256/size mismatch: '+part['path'])
                output.write(data);whole.update(data);total+=len(data)
                if i==1 or i%25==0 or i==len(parts):print(f'公开材料校验/合并：{i}/{len(parts)}',flush=True)
        if total!=manifest['archive_bytes'] or whole.hexdigest()!=manifest['archive_sha256']:
            raise RuntimeError('Whole public archive SHA256/size mismatch')
        temporary.replace(target)
    except BaseException:
        if temporary.exists():temporary.unlink()
        raise
    return target


def restore(root, archive, manifest):
    snapshot_path=root/'reports/demo_inputs_manifest.json'
    if not snapshot_path.exists():raise RuntimeError('Missing published final input inventory')
    required=json.loads(snapshot_path.read_text(encoding='utf-8-sig'))['files_sha256']
    restored=0;preserved=0
    with zipfile.ZipFile(archive) as bundle:
        names=bundle.namelist()
        if len(names)!=len(set(names)):raise RuntimeError('Duplicate archive members')
        inventory=json.loads(bundle.read('PACKAGE_MANIFEST.json'))['sha256']
        if len(inventory)!=manifest['file_count'] or set(names)!=set(inventory)|{'PACKAGE_MANIFEST.json'}:
            raise RuntimeError('Archive member inventory differs')
        for index,info in enumerate(bundle.infolist(),1):
            target=safe_path(root,info.filename)
            if stat.S_ISLNK(info.external_attr>>16):raise RuntimeError('Symlink archive entry forbidden')
            if info.filename=='PACKAGE_MANIFEST.json':continue
            content=bundle.read(info)
            if hashlib.sha256(content).hexdigest()!=inventory[info.filename]:raise RuntimeError('Member SHA256 mismatch: '+info.filename)
            material=PurePosixPath(info.filename).parts[0] in {'data','runs','artifacts','reports'}
            if target.exists():
                if info.filename in required and sha256(target)!=required[info.filename]:
                    raise RuntimeError('Existing required input differs; preserved without overwrite: '+info.filename)
                preserved+=1
            elif material or info.filename in required:
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_name(target.name+'.restoring')
                temporary.write_bytes(content);temporary.replace(target);restored+=1
            if index%5000==0:print(f'逐文件SHA256校验：{index}/{len(names)}',flush=True)
    for name,digest in required.items():
        path=safe_path(root,name)
        if not path.is_file() or sha256(path)!=digest:raise RuntimeError('Final frozen input missing/changed: '+name)
    print(f'完整公开材料已恢复：新增{restored}个文件，保留{preserved}个已有文件；最终输入SHA256通过。',flush=True)
    return {'restored':restored,'preserved':preserved,'final_inputs_verified':len(required)}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true',help='Verify/restore public materials without model inference')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'reproducibility/manifest.json').read_text(encoding='utf-8-sig'))
    if manifest.get('schema')!='cluener-public-materials-1':raise RuntimeError('Unknown public material inventory')
    print('完整复现材料随公开仓库提供，无需私有包、API Key或重新请求大模型。',flush=True)
    archive=assemble(ROOT,manifest);result=restore(ROOT,archive,manifest)
    (ROOT/'runs/public_reproduction/verification.json').write_text(json.dumps({**result,'archive_sha256':manifest['archive_sha256'],
        'public_materials':True,'private_download_required':False,'new_model_api_calls':0},indent=2),encoding='utf-8')
    if args.prepare_only:
        print('PASS: 公开材料还原与完整性校验通过。',flush=True);return
    command=[sys.executable,str(ROOT/'scripts/demo_offline.py'),'--name','公开离线复现','--student-id','NOT_APPLICABLE','--auto']
    print('开始：现场CRF推断 → 最终15组重评 → 两类真实难例。',flush=True)
    outcome=subprocess.run(command,cwd=ROOT)
    if outcome.returncode:raise SystemExit(outcome.returncode)
    print('PASS: PUBLIC REPRODUCTION COMPLETE; no private package or model API requests.',flush=True)

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,KeyError,zipfile.BadZipFile) as exc:
        print('公开复现停止：'+str(exc),file=sys.stderr)
        raise SystemExit(1)
