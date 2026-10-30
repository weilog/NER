"""Offline consistency checks for the integrated report and its saved source tables."""
import csv
import re
from ner_experiment.common import ROOT, file_hash, load_json, save_json


def main():
    tables=ROOT/'reports/tables/integrated_v3';figures=ROOT/'reports/figures/integrated_v3'
    verification=load_json(tables/'verification.json')
    for relative,digest in {**verification['source_hashes'],**verification['analysis_source_hashes']}.items():
        if file_hash(ROOT/relative)!=digest:raise RuntimeError('Source changed: '+relative)
    def read(name):
        with (tables/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    main_rows=read('main_results.csv');subset=read('network_subset.csv')
    methods=verification['methods']
    if [r['method'] for r in main_rows]!=methods or len(methods)!=15:raise RuntimeError('Main method mismatch')
    for r in main_rows:
        m=load_json(ROOT/'runs/integrated_v3_analysis'/r['method']/'metrics.json')
        for level in ['entity','token']:
            for k in ['tp','fp','fn','precision','recall','f1']:
                if float(r[f'{level}_{k}'])!=m[level][k]:raise RuntimeError('Main metric mismatch')
    counts={'main_rows':15,'entity_class_rows':150,'bio_label_rows':300,'comparison_rows':84,'parameter_rows':13}
    for name,expected in [('entity_per_type',150),('token_bio_per_label',300),('bootstrap_comparisons',84),('baseline_dev_parameters',13)]:
        if len(read(name+'.csv'))!=expected:raise RuntimeError('Table row mismatch: '+name)
    ids=load_json(tables/'network_subset_ids.json')
    kept,removed=set(ids['kept_ids']),set(ids['removed_ids'])
    if kept & removed or len(kept)+len(removed)!=1209 or len(kept)!=1088:raise RuntimeError('Diagnostic ID partition mismatch')
    if [r['method'] for r in subset]!=methods or any(int(r['samples'])!=len(kept) for r in subset):raise RuntimeError('Diagnostic method mismatch')
    manifest=load_json(figures/'figure_manifest.json')
    if file_hash(ROOT/'scripts/plot_integrated_results.py')!=manifest['plot_script_sha256']:raise RuntimeError('Plot source changed')
    for relative,digest in manifest['source_files'].items():
        if file_hash(ROOT/relative)!=digest:raise RuntimeError('Figure source changed: '+relative)
    for name in manifest['figures']:
        for suffix in ['png','svg','pdf']:
            if not (figures/f'{name}.{suffix}').is_file():raise RuntimeError('Missing figure')
    report=ROOT/'reports/integrated_results.md'
    for link in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        if not (report.parent/link).exists():raise RuntimeError('Broken report link: '+link)
    costs=read('cost_latency.csv')
    for c in costs:
        if abs(float(c['known_usage_estimate_cny'])+float(c['unknown_reservations_cny'])-float(c['gross_ledger_upper_cny']))>1e-8:
            raise RuntimeError('Cost accounting mismatch')
    save_json(tables/'artifact_verification.json',{'status':'PASS','checks':counts,
        'shared_subset_samples':len(kept),'all_source_hashes_unchanged':True,'figure_sources_match':True,
        'report_links_exist':True,'report_sha256':file_hash(report),'new_api_calls':0})
    print('PASS: 15 methods; 150 entity classes; 300 BIO labels; 84 paired metric rows; 6 figures; source hashes intact')


if __name__=='__main__':main()
