"""Local-only archive; allowlist excludes credentials and environments."""
import zipfile
import hashlib
import json
from ner_experiment.common import ROOT, file_hash, utc_now

def main():
    if (ROOT/'runs/llm/process.lock').exists():raise RuntimeError('Wait until API run finishes')
    folders=['data/raw/cluener2020','data/processed/cluener2020','data/processed/cluener2020_project','data/manifests','data/splits',
             'runs/baselines','runs/llm','runs/final','configs','src','scripts','tests','docs','reports']
    files=[p for name in folders for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.tmp','.pyc']]
    files += [ROOT/n for n in ['README.md','pyproject.toml','requirements-lock.txt','.env.example','.gitattributes','.gitignore']]
    files=sorted(set(files));manifest={p.relative_to(ROOT).as_posix():file_hash(p) for p in files}
    archive=ROOT.parent/'NLP_archives'/'cluener_experiment_results.zip';archive.parent.mkdir(exist_ok=True)
    temp=archive.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('PACKAGE_MANIFEST.json',json.dumps({'created_at':utc_now(),'sha256':manifest},ensure_ascii=False,indent=2))
    with zipfile.ZipFile(temp) as z:
        assert '.env' not in z.namelist()
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    temp.replace(archive)
    print('Verified local result archive:',archive,'files=',len(files),'sha256=',file_hash(archive))

if __name__=='__main__':main()
