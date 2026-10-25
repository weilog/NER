"""Private experiment/learning handoff with verified SQLite backup and hashes."""
from contextlib import closing
from pathlib import Path
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import zipfile
from dotenv import dotenv_values
from ner_experiment.data.splitting import json_bytes, sha


def main(root):
    root = Path(root)
    config = json.loads((root / 'configs/index_factorial_v1.json').read_text(encoding='utf-8'))
    directory = root / config['run_directory']
    verification = json.loads((directory / 'verification/offline-replay.json').read_text(encoding='utf-8'))
    if not verification['accounting_unchanged']:
        raise ValueError('Offline verification required')
    folders = ['src/ner_experiment','scripts','tests','configs','docs','reports','data/splits',
               'data/manifests/cluener2020','data/raw/cluener2020','data/processed/cluener2020',
               'data/processed/experiment_v2','runs/traditional_dev','runs/llm','artifacts/models/traditional_dev']
    paths = set()
    for folder in folders:
        for path in (root / folder).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.name == '.gitkeep':
                continue
            if path.name.startswith('.env') or path.suffix in {'.key','.pem','.tmp','.lock'} or path.name.endswith(('-wal','-shm')):
                continue
            paths.add(path)
    paths.update(root/name for name in ('README.md','pyproject.toml','requirements-lock.txt','.python-version','.gitignore','.gitattributes'))
    secret = (os.getenv('DEEPSEEK_API_KEY') or dotenv_values(root / '.env').get('DEEPSEEK_API_KEY') or '').encode()
    patterns = [rb'sk-[A-Za-z0-9_-]{24,}',rb'ghp_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{30,}',rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    destination = root/'artifacts/result-packages/cluener-index-factorial-2026-10-25.zip'
    destination.parent.mkdir(parents=True,exist_ok=True)
    ledger = root/config['ledger_path']
    manifest = {}
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary)/'budget.sqlite3'
        with closing(sqlite3.connect(f'file:{ledger.as_posix()}?mode=ro',uri=True)) as source,closing(sqlite3.connect(snapshot)) as target:
            source.backup(target)
            if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Ledger backup failed')
        with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(paths):
                name = path.relative_to(root).as_posix()
                body = snapshot.read_bytes() if path == ledger else path.read_bytes()
                if secret and secret in body or any(re.search(pattern,body) for pattern in patterns):
                    raise ValueError('Credential scan rejected '+name)
                manifest[name] = {'sha256':sha(body),'bytes':len(body)}
                archive.writestr(name,body)
            provenance = {'branch':subprocess.check_output(['git','branch','--show-current'],cwd=root).decode().strip(),
                          'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),
                          'note':'Historical source snapshots are inside run folders; package files individually hashed.'}
            archive.writestr('GIT-PROVENANCE.json',json_bytes(provenance))
            archive.writestr('PACKAGE-MANIFEST.json',json_bytes(manifest))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('CRC failure')
        for name,info in manifest.items():
            if sha(archive.read(name)) != info['sha256']:
                raise ValueError('Archive hash mismatch '+name)
    checksum=sha(destination.read_bytes())
    destination.with_suffix('.zip.sha256').write_text(f'{checksum}  {destination.name}\n',encoding='utf-8')
    print(json.dumps({'path':str(destination),'sha256':checksum,'files_verified':len(manifest),'bytes':destination.stat().st_size},indent=2))


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
