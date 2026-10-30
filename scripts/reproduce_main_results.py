"""One offline entry for the current 15-method results. Never starts an API runner."""
import subprocess
import sys
from pathlib import Path
from ner_experiment.common import ROOT


def main():
    required=['data/processed/cluener2020_project/test.jsonl','runs/baselines/selection.json']
    required += [f'runs/final/{m}/predictions.jsonl' for m in ['dictionary','hmm','crf']]
    required += [f'runs/{base}/shot_{s}/predictions.jsonl' for base in
        ['llm_text_test_v1','glm_flashx_test_v1','mimo_flash_test_v1'] for s in [0,2,4,8]]
    missing=[p for p in required if not (ROOT/p).is_file()]
    if missing:
        raise SystemExit('Local evidence is missing. Restore the authorized private result package into the project first. '
                         'Cloning GitHub alone does not include data or predictions. No paid runner was started.\n'+'\n'.join(missing))
    log=ROOT/'runs/integrated_v1_analysis/reproduction_current.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    commands=[['scripts/integrate_v1_results.py'],['scripts/plot_v1_results.py'],
              ['scripts/verify_v1_artifacts.py'],['-m','pytest','tests','-q']]
    with log.open('w',encoding='utf-8') as stream:
        for command in commands:
            label='RUN '+ ' '.join(command);print(label,flush=True);stream.write(label+'\n');stream.flush()
            result=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            stream.write(result.stdout);stream.flush();print(result.stdout,flush=True)
            if result.returncode:raise SystemExit(result.returncode)
    print('PASS: current 15-method offline reproduction; no model API requests. Log:',log)


if __name__=='__main__':main()
