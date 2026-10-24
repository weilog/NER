"""Private local handoff containing datasets, models and pilot records, no keys."""
from pathlib import Path
from contextlib import closing
import json
import re
import sqlite3
import tempfile
import zipfile
from dotenv import dotenv_values
from ner_experiment.data.splitting import json_bytes, sha


def main(root):
    root = Path(root)
    config = json.loads((root / 'configs/llm_pilot_v2.json').read_text(encoding='utf-8'))
    context = json.loads((root / config['run_directory'] / 'context.json').read_text(encoding='utf-8'))
    for name, digest in context['source_sha256'].items():
        if sha((root / name).read_bytes()) != digest:
            raise ValueError(f'Pilot source no longer matches: {name}')
    folders = ['src/ner_experiment', 'scripts', 'tests', 'configs', 'docs', 'reports',
               'data/raw/cluener2020', 'data/processed/cluener2020',
               'data/processed/experiment_v2', 'data/manifests/cluener2020', 'data/splits',
               'artifacts/models/traditional_dev', 'runs/traditional_dev', 'runs/llm']
    paths = set()
    for folder in folders:
        for path in (root / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.name != '.gitkeep':
                if path.name.startswith('.env') or path.suffix in {'.key', '.pem', '.tmp', '.lock'} or path.name.endswith(('-wal', '-shm')):
                    continue
                paths.add(path)
    paths.update(root / name for name in ('README.md', 'pyproject.toml', 'requirements-lock.txt', '.python-version', '.gitignore', '.gitattributes'))
    key = (dotenv_values(root / '.env').get('DEEPSEEK_API_KEY') or '').encode('utf-8')
    patterns = [rb'(?:sk-[A-Za-z0-9_-]{24,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',
                rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    output = root / 'artifacts/result-packages/cluener-llm-pilot-2026-10-24.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger = root / config['ledger_path']
    manifest = {}
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary) / 'budget.sqlite3'
        with closing(sqlite3.connect(f'file:{ledger.as_posix()}?mode=ro', uri=True)) as source, closing(sqlite3.connect(snapshot)) as target:
            source.backup(target)
            if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Ledger snapshot integrity check failed')
        with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(paths):
                name = path.relative_to(root).as_posix()
                body = snapshot.read_bytes() if path == ledger else path.read_bytes()
                if key and key in body or any(re.search(pattern, body) for pattern in patterns):
                    raise ValueError(f'Credential-like content detected; package rejected: {name}')
                manifest[name] = {'sha256': sha(body), 'bytes': len(body)}
                archive.writestr(name, body)
            archive.writestr('PACKAGE-MANIFEST.json', json_bytes(manifest))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC validation failed')
        for name, info in manifest.items():
            if sha(archive.read(name)) != info['sha256']:
                raise ValueError(f'Archive content hash mismatch: {name}')
    checksum = sha(output.read_bytes())
    output.with_suffix('.zip.sha256').write_text(f'{checksum}  {output.name}\n', encoding='utf-8')
    print(json.dumps({'path': str(output), 'sha256': checksum,
                      'files_verified': len(manifest), 'credential_scan_passed': True}, indent=2))


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
