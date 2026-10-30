"""Offline audit of the completed fixed-ID network recovery. Does not load keys or call APIs."""
import csv
from ner_experiment.common import ROOT, file_hash, load_json, object_hash, project_data, read_jsonl, save_json
from ner_experiment.evaluation.metrics_v1 import evaluate
from ner_experiment.models.llm import validate_cache
from ner_experiment.models.glm import parse_glm_response, usage_cost


def main():
    base=ROOT/'runs/network_recovery_20261030'
    freeze=load_json(base/'protocol.json')['protocol'];summary=load_json(base/'summary.json')
    assert summary['status']=='COMPLETE'
    assert file_hash(ROOT/'scripts/recover_network_failures.py')==freeze['script_sha256']
    assert all(file_hash(ROOT/p)==h for p,h in freeze['protected_inputs_sha256'].items())
    gold=project_data('test');gold_ids=[r['id'] for r in gold]
    verified=[];all_attempts=0;known_total=0.0
    methods={}
    for group in freeze['groups']:
        family=group['family'];folder=base/family;shot=group['shots']
        original_dir=ROOT/'runs'/('glm_flashx_test_v1' if family=='glm' else 'mimo_flash_test_v1')
        original=read_jsonl(original_dir/f'shot_{shot}/predictions.jsonl')
        old_by={r['id']:r for r in original}
        merged=read_jsonl(folder/'merged_predictions.jsonl');replacements=read_jsonl(folder/'recovery_predictions.jsonl')
        replace_by={r['id']:r for r in replacements};merged_by={r['id']:r for r in merged}
        allowed=set(group['ids'])
        assert len(merged)==1209 and [r['id'] for r in merged]==[r['id'] for r in original]
        assert set(replace_by)==allowed and len(replacements)==len(allowed)
        assert all(merged_by[i]==old_by[i] for i in merged_by.keys()-allowed)
        assert all(merged_by[i]==replace_by[i] for i in allowed)
        assert all(r['status']=='ok' for r in replacements)
        metrics,counts,_=evaluate(gold,merged)
        assert metrics==load_json(folder/'metrics.json') and counts==read_jsonl(folder/'sentence_counts.jsonl')
        assert evaluate(gold,original)[0]==load_json(folder/'metrics_before.json')
        for prediction in replacements:
            record=load_json(folder/'cache'/(prediction['cache_key']+'.json'))
            original_record=load_json(original_dir/'cache'/(prediction['cache_key']+'.json'))
            assert record['payload']==original_record['payload'] and record['endpoint']==original_record['endpoint']
            assert validate_cache(record,record['payload'],record['endpoint'],parse_glm_response)==prediction
        ledger=load_json(folder/'ledger.json')['attempts']
        assert all(a['status']=='response' for a in ledger)
        prices=freeze['parameters'][family]['pricing']
        for attempt in ledger:assert abs(usage_cost(attempt['usage'],prices)-attempt['charged_upper_cny'])<1e-12
        cost=sum(a['charged_upper_cny'] for a in ledger)
        assert cost<=group['budget']
        all_attempts+=len(ledger);known_total+=cost
        methods[f'{family}_{shot}']=metrics
        verified.append({'family':family,'samples':1209,'recovered':len(replacements),
                         'preserved_samples':1209-len(replacements),'api_attempts':len(ledger),
                         'actual_models':sorted({r.get('actual_model','unknown') for r in replacements}),
                         'known_cost_cny':cost,'metrics_replayed_exactly':True,
                         'same_original_payloads':True,'other_predictions_unchanged':True,
                         'merged_predictions_sha256':file_hash(folder/'merged_predictions.jsonl')})
    assert all_attempts==summary['new_api_attempts']==121
    assert abs(known_total-summary['new_ledger_upper_cny'])<1e-12
    # A separate complete score table; confidence intervals are omitted until regenerated.
    with (ROOT/'reports/tables/integrated_v1/main_results.csv').open(encoding='utf-8-sig',newline='') as f:old_rows=list(csv.DictReader(f))
    table=[]
    for row in old_rows:
        name=row['method'];m=methods.get(name)
        out={'method':name,'display':row['display'],'samples':1209,'evaluation_protocol':'v1',
             'prediction_scope':'fixed network recovery merge' if m else 'unchanged original predictions'}
        for level in ['token','entity']:
            for key in ['tp','fp','fn','precision','recall','f1']:
                out[f'{level}_{key}']=m[level][key] if m else float(row[f'{level}_{key}'])
        table.append(out)
    path=ROOT/'reports/tables/network_recovery/main_results.csv';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    audit={'status':'PASS','groups':verified,'api_attempts':all_attempts,'total_known_usage_cny':known_total,
           'original_inputs_sha256_unchanged':True,'offline_replay_new_api_calls':0,
           'new_15_method_table':path.relative_to(ROOT).as_posix(),
           'formal_word_report_updated':False}
    save_json(base/'verification.json',audit)
    print('PASS: 121 recovered; same payloads; all other predictions unchanged; full-Test scores replay exactly; added estimate CNY',round(known_total,6))

if __name__=='__main__':main()
