"""Optional local PDF QA. Requires an explicit directory of external QA tools."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from ner_experiment.common import ROOT, file_hash, load_json, save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qa-tools-dir',required=True,type=Path)
    args=parser.parse_args();out=ROOT/'reports/figures/integrated_v1'
    def check(script,arguments,target):
        p=subprocess.run([sys.executable,str(args.qa_tools_dir/script),*map(str,arguments),'--json'],
                         cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        try:r=json.loads(p.stdout)
        except ValueError:raise RuntimeError(f'{script} did not return JSON: '+p.stderr)
        save_json(out/target,r)
        return {'exit_code':p.returncode,'report':target}
    result={'source':check('validate_figure.py',['scripts/plot_v1_results.py'],'source_qa.json'),'figures':{}}
    for name in load_json(out/'figure_manifest.json')['figures']:
        result['figures'][name]={
            'font':check('audit_pdf_text.py',[out/f'{name}.pdf','--min-pt','7.5'],f'{name}.fonts.json'),
            'collision':check('audit_figure_collisions.py',[out/f'{name}.pdf'],f'{name}.collisions.json')}
    result['tools_sha256']={name:file_hash(args.qa_tools_dir/name) for name in
        ['validate_figure.py','audit_pdf_text.py','audit_figure_collisions.py']}
    result['raster_dpi_exception']='300dpi course preview; editable SVG/PDF are primary, not a raster journal submission'
    save_json(out/'qa_summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['source']['exit_code'] or any(x['exit_code'] for v in result['figures'].values() for x in v.values()):
        raise SystemExit(1)


if __name__=='__main__':main()
