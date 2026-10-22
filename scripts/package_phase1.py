"""Build and hash-verify a local data/evaluator handoff, excluding credentials."""
from pathlib import Path
import importlib.metadata
import json
import platform
import sys
import zipfile

from ner_experiment.data.splitting import json_bytes, sha


def main(root):
    root = Path(root)
    split = json.loads((root / 'data/splits/experiment_v2.json').read_text(encoding='utf-8'))
    report = {'stage': 'data_and_dual_evaluator', 'date': '2026-10-22',
              'python': platform.python_version(), 'platform': platform.platform(),
              'seed': split['seed'], 'archive_sha256': split['archive_sha256'],
              'counts': {name: item['sentences'] for name, item in split['statistics'].items()},
              'valid_training_sentences': split['valid_training_sentences'],
              'models_run': False, 'test_predictions_scored': False,
              'api_calls': 0, 'api_cost_cny': 0,
              'license': 'unresolved',
              'packages': {line.split('==')[0]: importlib.metadata.version(line.split('==')[0])
                           for line in (root / 'requirements-lock.txt').read_text().splitlines()
                           if '==' in line and not line.startswith('#')}}
    (root / 'reports/data-readiness.json').write_bytes(json_bytes(report))
    folders = ['src/ner_experiment', 'scripts', 'tests', 'configs', 'docs', 'reports',
               'data/raw/cluener2020', 'data/processed/cluener2020',
               'data/processed/experiment_v2', 'data/manifests/cluener2020', 'data/splits',
               'runs/phase1']
    paths = []
    for folder in folders:
        for path in sorted((root / folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.name not in {'.env', '.gitkeep'}:
                if path.name.startswith('.env') or path.suffix in {'.key', '.pem'}:
                    continue
                paths.append(path)
    paths += [root / name for name in ['README.md', 'pyproject.toml', 'requirements-lock.txt', '.python-version', '.gitignore']]
    content = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
    manifest = {name: {'sha256': sha(body), 'bytes': len(body)} for name, body in content.items()}
    content['PACKAGE-MANIFEST.json'] = json_bytes(manifest)
    destination = root / 'artifacts/result-packages'
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / 'cluener-phase1-2026-10-22.zip'
    # Refuse accidental replacement of a previous handoff artifact.
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, body in content.items():
            archive.writestr(name, body)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC validation failed')
        for name, entry in manifest.items():
            if sha(archive.read(name)) != entry['sha256']:
                raise ValueError(f'Archive hash mismatch: {name}')
    checksum = sha(output.read_bytes())
    output.with_suffix('.zip.sha256').write_text(f'{checksum}  {output.name}\n', encoding='utf-8')
    print(json.dumps({'path': str(output), 'sha256': checksum, 'files_verified': len(manifest)}, indent=2))


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
