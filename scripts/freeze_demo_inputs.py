"""Explicitly freeze verified final demo inputs; never refresh silently and never call APIs."""
import argparse
from ner_experiment.common import ROOT, load_json, save_json, utc_now
from demo_evidence import build_snapshot, verify_snapshot

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reports/demo_inputs_manifest.json')
    args=parser.parse_args()
    path=ROOT/args.output
    if path.exists():
        snapshot=verify_snapshot(ROOT,load_json(path))
        print('Existing manifest verified; not overwritten:',path)
    else:
        snapshot=build_snapshot(ROOT)
        snapshot['created_at']=utc_now()
        save_json(path,snapshot)
        print('FROZEN:',path)
    print('Final methods:',len(snapshot['methods']),'Hashed inputs:',len(snapshot['files_sha256']),'API calls: 0')

if __name__=='__main__':main()
