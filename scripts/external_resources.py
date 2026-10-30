"""Restore one source-disjoint data/model/result ZIP. Standard library only; no API clients."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import uuid
import zipfile


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def safe_target(root,name):
    if not isinstance(name,str) or not name or '\\' in name or ':' in name or '\x00' in name:
        raise RuntimeError('Unsafe resource path: '+repr(name))
    value=PurePosixPath(name)
    if value.is_absolute() or '..' in value.parts or name!=value.as_posix():
        raise RuntimeError('Unsafe resource path: '+repr(name))
    root=Path(root).resolve();target=(root/name).resolve()
    if not target.is_relative_to(root):raise RuntimeError('Resource path escapes project: '+name)
    return target


def resource_path(name):
    parts=PurePosixPath(name).parts
    blocked={'.git','.venv','venv','__pycache__','.pytest_cache','source_snapshot','backups','backup','archives','archive'}
    return (bool(parts) and parts[0] in {'data','runs'}
            and not any(p.lower() in blocked for p in parts)
            and not any(p.startswith('.env') for p in parts)
            and PurePosixPath(name).suffix.lower() not in {'.py','.pyc','.pyo','.ps1','.js','.cjs','.ipynb','.mp4','.7z','.rar','.pem','.key'})


def _unique_json(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise RuntimeError('Duplicate JSON key in material manifest: '+key)
        result[key]=value
    return result


def restore_resources(root,archive,catalog,source_files=None):
    root=Path(root).resolve();archive=Path(archive).resolve()
    if catalog.get('schema')!='cluener-external-resources-1':raise RuntimeError('Unknown external resource catalog')
    if archive.stat().st_size!=catalog['archive_bytes'] or sha256(archive)!=catalog['archive_sha256']:
        raise RuntimeError('External archive SHA256/size mismatch: '+str(archive))
    if source_files is None:
        source_manifest=root/'SOURCE_PACKAGE_MANIFEST.json'
        source_files=set(json.loads(source_manifest.read_text(encoding='utf-8-sig'))['files_sha256']) if source_manifest.exists() else set()
    else:source_files=set(source_files)
    frozen_path=root/'reports/demo_inputs_manifest.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf-8-sig'))['files_sha256']
    restored=preserved=0
    with zipfile.ZipFile(archive) as bundle:
        infos=bundle.infolist();names=[i.filename for i in infos]
        if len(set(names))!=len(names):raise RuntimeError('Duplicate resource ZIP members')
        if 'MATERIALS_MANIFEST.json' not in names:raise RuntimeError('Missing external material manifest')
        if bundle.getinfo('MATERIALS_MANIFEST.json').file_size>10*1024*1024:raise RuntimeError('Material manifest size exceeded')
        raw=bundle.read('MATERIALS_MANIFEST.json')
        if hashlib.sha256(raw).hexdigest()!=catalog['manifest_sha256']:raise RuntimeError('External manifest SHA256 differs')
        manifest=json.loads(raw.decode('utf-8-sig'),object_pairs_hook=_unique_json)
        if manifest.get('schema')!='cluener-resource-members-1' or not isinstance(manifest.get('files'),dict):raise RuntimeError('Unknown member manifest')
        files=manifest['files']
        if len(files)!=catalog['file_count'] or set(names)!=set(files)|{'MATERIALS_MANIFEST.json'}:raise RuntimeError('External ZIP inventory differs')
        overlap=set(names)&source_files
        if overlap:raise RuntimeError('Source/resource path overlap: '+repr(sorted(overlap)))
        for info in infos:
            if info.filename=='MATERIALS_MANIFEST.json':continue
            target=safe_target(root,info.filename)
            if not resource_path(info.filename):raise RuntimeError('Forbidden resource path: '+info.filename)
            if stat.S_ISLNK(info.external_attr>>16):raise RuntimeError('Symlink resource member forbidden: '+info.filename)
            entry=files[info.filename]
            if type(entry.get('bytes')) is not int or entry['bytes']<0 or info.file_size!=entry['bytes']:
                raise RuntimeError('Resource size metadata differs: '+info.filename)
            if target.exists() and (not target.is_file() or sha256(target)!=entry['sha256']):
                raise RuntimeError('Existing resource differs; preserved without overwrite: '+info.filename)
        # Validate the entire source/resource dependency union BEFORE extracting anything.
        for name,expected in frozen.items():
            if name in files:
                if files[name]['sha256']!=expected:raise RuntimeError('Frozen input resource hash differs: '+name)
            else:
                path=safe_target(root,name)
                if not path.is_file() or sha256(path)!=expected:raise RuntimeError('Frozen input absent/changed in source package: '+name)
        for index,(name,entry) in enumerate(files.items(),1):
            target=safe_target(root,name)
            if target.exists():preserved+=1
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.restoring')
                digest=hashlib.sha256();total=0
                try:
                    with bundle.open(name) as source,temporary.open('xb') as output:
                        for block in iter(lambda:source.read(1024*1024),b''):
                            total+=len(block)
                            if total>entry['bytes']:raise RuntimeError('Resource exceeded size allowance: '+name)
                            digest.update(block);output.write(block)
                    if total!=entry['bytes'] or digest.hexdigest()!=entry['sha256']:raise RuntimeError('Resource member SHA256 differs: '+name)
                    temporary.replace(target);restored+=1
                finally:
                    if temporary.exists():temporary.unlink()
            if index==1 or index%4000==0 or index==len(files):print(f'外部资源校验/还原：{index}/{len(files)}',flush=True)
    for name,expected in frozen.items():
        path=safe_target(root,name)
        if not path.is_file() or sha256(path)!=expected:raise RuntimeError('Final frozen input mismatch: '+name)
    return {'restored':restored,'preserved':preserved,'final_inputs_verified':len(frozen),
            'source_resource_overlap':0,'resource_files_verified':len(files),'archive_sha256':catalog['archive_sha256']}
