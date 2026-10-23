"""Single entry point. all includes paid API; offline never calls it."""
import argparse
import subprocess
import sys
import shutil
from ner_experiment.common import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['all','offline']);args=p.parse_args()
    # A clean clone contains the published freeze, but no local run artifacts.
    # Preserve that reference and generate a fresh freeze after this run's Dev.
    freeze=ROOT/'configs/frozen_experiment.json'
    if args.mode=='all' and freeze.exists() and not (ROOT/'runs/baselines/selection.json').exists():
        if any((ROOT/'runs/final').glob('*/predictions.jsonl')):raise RuntimeError('Incomplete existing run; inspect before reinitializing')
        reference=ROOT/'runs/reference/frozen_experiment.json';reference.parent.mkdir(parents=True,exist_ok=True)
        if reference.exists():raise RuntimeError('Existing reference freeze; inspect before starting a new run')
        shutil.copy2(freeze,reference);freeze.unlink()
        print('Clean run: published freeze preserved in runs/reference; a new freeze will be created after Dev.',flush=True)
    commands=[['scripts/replay.py'],['scripts/report.py']] if args.mode=='offline' else [
        ['scripts/prepare_cluener.py'],['scripts/refresh_split.py'],['-m','pytest','-q'],
        ['scripts/run_baselines.py'],['scripts/run_llm.py','--stage','dev'],['scripts/run_final.py'],
        ['scripts/replay.py'],['scripts/report.py'],['scripts/package_results.py']]
    for cmd in commands:
        print('RUN',*cmd,flush=True);subprocess.run([sys.executable,*cmd],cwd=ROOT,check=True)

if __name__=='__main__':main()
