"""Package data, baseline models and Dev predictions locally, without secrets."""
from pathlib import Path
import json
import zipfile
from ner_experiment.data.splitting import json_bytes, sha


def main(root):
    root = Path(root)
    selected = json.loads((root / 'configs/traditional_selected.json').read_text(encoding='utf-8'))
    run = root / selected['run_directory']
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    context = json.loads((run / 'context.json').read_text(encoding='utf-8'))
    for name, digest in context['source_sha256'].items():
        if sha((root / name).read_bytes()) != digest:
            raise ValueError(f'Current source differs from fitted experiment: {name}')
    folders = ['src/ner_experiment', 'scripts', 'tests', 'configs', 'docs', 'reports',
               'data/raw/cluener2020', 'data/processed/cluener2020',
               'data/processed/experiment_v2', 'data/manifests/cluener2020', 'data/splits',
               selected['run_directory']]
    paths = set()
    for folder in folders:
        for path in sorted((root / folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.name != '.gitkeep':
                if path.name.startswith('.env') or path.suffix in {'.pem', '.key', '.tmp'}:
                    continue
                paths.add(path)
    for trial in summary['trials']:
        path = root / trial['model_path']
        if sha(path.read_bytes()) != trial['model_sha256']:
            raise ValueError(f'Model hash changed: {path}')
        paths.add(path)
    paths.update(root / name for name in ('README.md', 'pyproject.toml', 'requirements-lock.txt', '.python-version', '.gitignore'))
    output = root / 'artifacts/result-packages/cluener-traditional-dev-2026-10-22.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            name = path.relative_to(root).as_posix()
            body = path.read_bytes()
            manifest[name] = {'sha256': sha(body), 'bytes': len(body)}
            archive.writestr(name, body)
        archive.writestr('PACKAGE-MANIFEST.json', json_bytes(manifest))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC validation failed')
        for name, metadata in manifest.items():
            if sha(archive.read(name)) != metadata['sha256']:
                raise ValueError(f'Archive content differs: {name}')
    checksum = sha(output.read_bytes())
    output.with_suffix('.zip.sha256').write_text(f'{checksum}  {output.name}\n', encoding='utf-8')
    print(json.dumps({'path': str(output), 'sha256': checksum, 'files_verified': len(manifest), 'bytes': output.stat().st_size}, indent=2))


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
