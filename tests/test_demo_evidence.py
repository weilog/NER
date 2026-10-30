"""Synthetic guardrail tests; fixtures are not claimed as real experiment predictions."""
import csv
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from demo_evidence import (DemoError, count_value, compare_metrics, read_table,
    recovery_mapping, checked_rows, select_case, case_error)
from ner_experiment.common import file_hash, save_json, write_jsonl


def scores():
    return {level:{'tp':2,'fp':1,'fn':1,'precision':2/3,'recall':2/3,'f1':2/3} for level in ['token','entity']}

def reference():
    return {f'{level}_{key}':('2.0' if key=='tp' else '1.0' if key in {'fp','fn'} else '0.6667')
            for level in ['token','entity'] for key in ['tp','fp','fn','precision','recall','f1']}

def test_rounded_probability_and_integral_float_csv_counts_are_accepted():
    compare_metrics('glm_8',scores(),reference())


def test_full_precision_float_noise_is_allowed():
    actual=scores();row=reference()
    row['token_f1']=str(2/3+5e-13)
    compare_metrics('glm_8',actual,row)


def test_count_mismatch_is_strict_and_identifies_model_metric_values():
    row=reference();row['entity_fn']='2.0'
    with pytest.raises(DemoError,match=r'glm_8 entity.fn: actual=1, reference='):
        compare_metrics('glm_8',scores(),row)


@pytest.mark.parametrize('raw',['1.0000000000000001','NaN','-1','Infinity'])
def test_noninteger_or_nonfinite_counts_rejected(raw):
    with pytest.raises(DemoError):count_value(raw,'model token.tp')


def test_probability_drift_is_not_hidden_by_tolerance():
    row=reference();row['token_f1']='0.6670'
    with pytest.raises(DemoError,match=r'glm_8 token.f1: actual=.*reference=.*abs_tol='):
        compare_metrics('glm_8',scores(),row)


def test_duplicate_csv_methods_fail_instead_of_overwrite(tmp_path):
    path=tmp_path/'scores.csv';path.write_text('method\nglm_8\nglm_8\n',encoding='utf-8')
    with pytest.raises(DemoError,match='Duplicate result-table method'):read_table(path)


def recovery_fixture(root,shots=(2,8)):
    audit={'status':'PASS','groups':[]};protocol={'groups':[]};summary={'status':'COMPLETE','groups':[]}
    for index,shot in enumerate(shots):
        folder=root/'runs/network_recovery_20261030'/f'actual_folder_{index}'
        save_json(folder/'summary.json',{'family':'glm','shots':shot,'full_test_samples':2,'recovered_ok':1})
        write_jsonl(folder/'merged_predictions.jsonl',[{'id':f'fixture:{index}','status':'ok','entities':[]}])
        digest=file_hash(folder/'merged_predictions.jsonl')
        audit['groups'].append({'family':'glm','samples':2,'recovered':1,'merged_predictions_sha256':digest})
        protocol['groups'].append({'family':'glm','shots':shot})
        summary['groups'].append({'family':'glm','shots':shot,'full_test_samples':2,'recovered_ok':1})
    return audit,protocol,summary


def test_recovery_shots_come_from_real_metadata_not_family_inference(tmp_path):
    audit,protocol,summary=recovery_fixture(tmp_path)
    mapping=recovery_mapping(tmp_path,audit,protocol,summary)
    assert set(mapping)=={('glm',2),('glm',8)}
    assert mapping[('glm',2)]['path'].endswith('actual_folder_0/merged_predictions.jsonl')


def test_duplicate_recovery_protocol_configuration_rejected(tmp_path):
    audit,protocol,summary=recovery_fixture(tmp_path)
    protocol['groups'].append(protocol['groups'][0].copy())
    with pytest.raises(DemoError,match='Duplicate recovery protocol configuration'):
        recovery_mapping(tmp_path,audit,protocol,summary)


def test_duplicate_audit_entry_rejected(tmp_path):
    audit,protocol,summary=recovery_fixture(tmp_path)
    audit['groups'].append(audit['groups'][0].copy())
    with pytest.raises(DemoError,match='Duplicate audited recovery configuration'):
        recovery_mapping(tmp_path,audit,protocol,summary)


def test_duplicate_actual_directory_configuration_rejected(tmp_path):
    audit,protocol,summary=recovery_fixture(tmp_path,shots=(2,2))
    # Keep declarations unique to exercise the independent directory duplicate check.
    protocol['groups']=protocol['groups'][:1];summary['groups']=summary['groups'][:1]
    with pytest.raises(DemoError,match='Duplicate recovery directory configuration'):
        recovery_mapping(tmp_path,audit,protocol,summary)


def test_corrupted_final_prediction_fails_sha256(tmp_path):
    path=tmp_path/'predictions.jsonl';write_jsonl(path,[{'id':'a'}]);digest=file_hash(path)
    path.write_text('{"id":"b"}\n',encoding='utf-8')
    with pytest.raises(DemoError,match='SHA256 mismatch'):
        checked_rows(tmp_path,'predictions.jsonl',digest)


def sample(identity,text,category):
    return {'id':identity,'text':text,'tokens':list(text),'entities':[{'start':0,'end':len(text),'type':category}]}


def test_type_case_is_verified_from_actual_spans_and_scored():
    gold=sample('synthetic:a','作品名','movie')
    prediction={'id':gold['id'],'status':'ok','entities':[{'start':0,'end':3,'type':'game'}]}
    case=case_error(gold,prediction,'type')
    assert case is not None
    assert case['metrics']['token']['fp']==3 and case['metrics']['token']['fn']==3
    assert case['metrics']['entity']['fp']==1 and case['metrics']['entity']['f1']==0
    assert case_error(gold,prediction,'boundary') is None


def test_preferred_case_corrected_uses_verified_final_file_alternative(tmp_path):
    gold={'first':sample('first','甲乙','movie'),'second':sample('second','丙丁','movie')}
    predictions=[{'id':'first','status':'ok','entities':gold['first']['entities']},
                 {'id':'second','status':'ok','entities':[{'start':0,'end':2,'type':'game'}]}]
    path=tmp_path/'final.jsonl';write_jsonl(path,predictions)
    entry={'method':'glm_8','prediction_path':'final.jsonl','prediction_sha256':file_hash(path)}
    result=select_case(tmp_path,entry,gold,'first','type')
    assert result['used_alternative'] is True and result['row']['id']=='second'
    assert result['source_path']=='final.jsonl'


def test_no_actual_error_does_not_fabricate_example(tmp_path):
    row=sample('first','甲乙','book');path=tmp_path/'final.jsonl'
    write_jsonl(path,[{'id':'first','status':'ok','entities':row['entities']}])
    entry={'method':'mimo_4','prediction_path':'final.jsonl','prediction_sha256':file_hash(path)}
    with pytest.raises(DemoError,match='No verified boundary error'):
        select_case(tmp_path,entry,{'first':row},'first','boundary')
