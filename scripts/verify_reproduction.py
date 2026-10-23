"""Verify a complete cache-resumed rerun preserves data, results, and API ledger."""
import subprocess
import sys
from ner_experiment.common import ROOT, file_hash, load_json, save_json, utc_now

def main():
    paths=[ROOT/'configs/frozen_experiment.json',ROOT/'runs/llm/ledger.json']
    paths+=list((ROOT/'data/processed/cluener2020_project').glob('*.jsonl'))
    paths+=list((ROOT/'runs/final').glob('*/predictions.jsonl'))
    before={p.relative_to(ROOT).as_posix():file_hash(p) for p in paths}
    log=ROOT/'reports/reproduction.log'
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'scripts/run_experiment.py','all'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Reproduction failed: inspect reports/reproduction.log')
    after={p.relative_to(ROOT).as_posix():file_hash(p) for p in paths}
    if before!=after:raise AssertionError('Reproduction changed protected artifacts or made new API requests')
    save_json(ROOT/'reports/tables/verification.json',{'checked_at':utc_now(),'full_pipeline_resume':True,
        'no_new_api_requests':True,'data_and_predictions_hashes_unchanged':True,'files':after,
        'note':'Same-machine cache-resume verification; independent-machine reproduction and human review remain pending.'})
    # Rebuild package after the completed verification record/log exist.
    subprocess.run([sys.executable,'scripts/package_results.py'],cwd=ROOT,check=True)
    print('PASS: full pipeline resumed; frozen data/predictions and API ledger unchanged.')

if __name__=='__main__':main()
