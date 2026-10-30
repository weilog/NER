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
        all_infos=bundle.infolist();raw_names=[i.filename for i in all_infos]
        if len(set(raw_names))!=len(raw_names):raise RuntimeError('Duplicate resource ZIP members')
        prefix=catalog.get('archive_prefix','')
        if prefix:
            if not isinstance(prefix,str) or not prefix.endswith('/'):raise RuntimeError('Unsafe archive prefix')
            safe_target(root,prefix[:-1])
        mapped={}
        for info in all_infos:
            safe_target(root,info.filename.rstrip('/') if info.is_dir() else info.filename)
            if stat.S_ISLNK(info.external_attr>>16):raise RuntimeError('Symlink resource member forbidden: '+info.filename)
            if prefix and not info.filename.startswith(prefix):raise RuntimeError('External ZIP inventory outside declared prefix')
            if info.is_dir():continue
            name=info.filename[len(prefix):];safe_target(root,name)
            mapped[name]=info
        names=set(mapped)
        if 'MATERIALS_MANIFEST.json' not in names:raise RuntimeError('Missing external material manifest')
        if mapped['MATERIALS_MANIFEST.json'].file_size>10*1024*1024:raise RuntimeError('Material manifest size exceeded')
        raw=bundle.read(mapped['MATERIALS_MANIFEST.json'])
        if hashlib.sha256(raw).hexdigest()!=catalog['manifest_sha256']:raise RuntimeError('External manifest SHA256 differs')
        manifest=json.loads(raw.decode('utf-8-sig'),object_pairs_hook=_unique_json)
        if manifest.get('schema')!='cluener-resource-members-1' or not isinstance(manifest.get('files'),dict):raise RuntimeError('Unknown member manifest')
        files=manifest['files']
        auxiliary=catalog.get('auxiliary_files',{})
        if not isinstance(auxiliary,dict):raise RuntimeError('Invalid auxiliary inventory')
        for name,entry in auxiliary.items():
            safe_target(root,name)
            if len(PurePosixPath(name).parts)!=1 or PurePosixPath(name).suffix.lower()!='.mp4':raise RuntimeError('Unsupported auxiliary file: '+name)
            if name in files:raise RuntimeError('Auxiliary/resource overlap')
        if len(files)!=catalog['file_count'] or names!=set(files)|{'MATERIALS_MANIFEST.json'}|set(auxiliary):raise RuntimeError('External ZIP inventory differs')
        overlap=set(names)&source_files
        if overlap:raise RuntimeError('Source/resource path overlap: '+repr(sorted(overlap)))
        # The recording is verified but never restored into experiment inputs.
        for name,entry in auxiliary.items():
            digest=hashlib.sha256();total=0
            with bundle.open(mapped[name]) as source:
                for block in iter(lambda:source.read(1024*1024),b''):digest.update(block);total+=len(block)
            if total!=entry.get('bytes') or digest.hexdigest()!=entry.get('sha256'):raise RuntimeError('Auxiliary size/hash mismatch: '+name)
        for name,entry in files.items():
            info=mapped[name];target=safe_target(root,name)
            if not resource_path(name):raise RuntimeError('Forbidden resource path: '+name)
            if type(entry.get('bytes')) is not int or entry['bytes']<0 or info.file_size!=entry['bytes']:
                raise RuntimeError('Resource size metadata differs: '+name)
            if target.exists() and (not target.is_file() or sha256(target)!=entry['sha256']):
                raise RuntimeError('Existing resource differs; preserved without overwrite: '+name)
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
                    with bundle.open(mapped[name]) as source,temporary.open('xb') as output:
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
            'source_resource_overlap':0,'resource_files_verified':len(files),'archive_sha256':catalog['archive_sha256'],
            'auxiliary_files_verified':len(auxiliary)}
