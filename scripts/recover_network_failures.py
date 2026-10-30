"""One explicit recovery batch of prelisted connection failures; original runs stay untouched."""
import argparse
from copy import deepcopy
from collections import Counter
import json
import os
import ssl
import threading
import urllib.error
from pathlib import Path
from ner_experiment.common import ROOT, file_hash, load_json, object_hash, project_data, read_jsonl, save_json, utc_now, write_jsonl
from ner_experiment.evaluation.metrics_v1 import evaluate
from ner_experiment.models.glm import GLMClient, InterfaceStop, atomic_json, parse_glm_response, settings as glm_settings
from ner_experiment.models.mimo import MiMoClient, settings as mimo_settings
from ner_experiment.models.llm import BudgetStop, validate_cache
from pilot_shot_budget import pilot_messages, SYSTEM_TEXT
from run_glm_test import predict_rows
from integrate_results import failure_kind

OUT=ROOT/'runs/network_recovery_20261030'
SPECS=[('glm','glm_flashx_test_v1','glm_flashx.json',4,117,4.8),
       ('mimo','mimo_flash_test_v1','mimo_flash.json',8,4,.2)]


def merge_predictions(original, replacements, allowed_ids):
    allowed=set(allowed_ids)
    if len(allowed)!=len(allowed_ids):raise ValueError('Duplicate allowed recovery IDs')
    ids=[r['id'] for r in original]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate original IDs')
    lookup={r['id']:r for r in replacements}
    if len(lookup)!=len(replacements) or set(lookup)!=allowed:raise ValueError('Recovery IDs differ from frozen list')
    if not allowed<=set(ids):raise ValueError('Unknown recovery ID')
    if any(failure_kind(r)!='connection' for r in original if r['id'] in allowed):
        raise ValueError('Only connection failures may be replaced')
    return [deepcopy(lookup[r['id']]) if r['id'] in allowed else deepcopy(r) for r in original]


class GuardedGLMClient(GLMClient):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.connection_guard=threading.Lock()
        self.consecutive_network_errors=0
    def http(self,payload):
        try:
            response=super().http(payload)
            with self.connection_guard:self.consecutive_network_errors=0
            return response
        except (urllib.error.URLError,TimeoutError,ConnectionError,OSError) as exc:
            if not isinstance(exc,urllib.error.HTTPError):
                reason=exc.reason if isinstance(exc,urllib.error.URLError) else exc
                with self.connection_guard:
                    self.consecutive_network_errors+=1
                    if isinstance(reason,ssl.SSLCertVerificationError) or self.consecutive_network_errors>=8:
                        self.stop_event.set()
            raise


def context():
    gold=project_data('test');lookup={r['id']:r for r in gold}
    if len(gold)!=1209 or len(lookup)!=1209:raise RuntimeError('Fixed full Test required')
    prepared=[];protected={}
    def protect(path):protected[path.relative_to(ROOT).as_posix()]=file_hash(path)
    for family,dirname,config_name,shots,expected,budget in SPECS:
        base=ROOT/'runs'/dirname;config=load_json(ROOT/'configs'/config_name)
        connection=(glm_settings if family=='glm' else mimo_settings)(config)
        freeze=load_json(base/'frozen_protocol.json')['protocol']
        if {'model':connection['model'],**config['parameters']}!=freeze['parameters']:
            raise RuntimeError('Frozen parameters changed')
        if freeze['system']!=SYSTEM_TEXT:raise RuntimeError('System prompt changed')
        original=read_jsonl(base/f'shot_{shots}/predictions.jsonl')
        if len(original)!=1209 or {r['id'] for r in original}!=set(lookup):raise RuntimeError('Original IDs changed')
        failures=[r for r in original if failure_kind(r)=='connection']
        if len(failures)!=expected:raise RuntimeError('Connection failure count changed')
        examples=freeze['examples'][:shots]
        for p in failures:
            row=lookup[p['id']]
            payload={**freeze['parameters'],'messages':pilot_messages(row,examples)}
            cache=base/'cache'/(p['cache_key']+'.json');record=load_json(cache)
            if record['payload']!=payload or record['endpoint']!=connection['endpoint']:
                raise RuntimeError('Recovery request differs from exact original request')
            if validate_cache(record,payload,connection['endpoint'],parse_glm_response)!=p:
                raise RuntimeError('Original cached prediction differs')
            protect(cache)
        for p in [base/f'shot_{shots}/predictions.jsonl',base/f'shot_{shots}/metrics.json',
                  base/'frozen_protocol.json',base/'ledger.json',ROOT/'configs'/config_name]:protect(p)
        if family=='glm':
            previous=load_json(ROOT/'runs/glm_network_recovery_plan/plan.json')
            if previous['network_failure_ids']!=[p['id'] for p in failures]:raise RuntimeError('Planned GLM IDs changed')
        prepared.append(dict(family=family,base=base,config=config,shots=shots,budget=budget,
                             connection=connection,original=original,examples=examples,
                             rows=[lookup[p['id']] for p in failures],ids=[p['id'] for p in failures]))
    protect(ROOT/'data/processed/cluener2020_project/test.jsonl')
    for file in (ROOT/'configs').glob('*seed*.json'):protect(file)
    return gold,prepared,protected


def cost_total(exclude_recovery=False):
    total=0.0
    for path in (ROOT/'runs').rglob('ledger.json'):
        if exclude_recovery and path.is_relative_to(OUT):continue
        attempts=load_json(path)['attempts']
        if any(a['status']=='reserved' for a in attempts):raise InterfaceStop('Unresolved in-flight reservation; no new requests')
        total+=sum(a['charged_upper_cny'] for a in attempts)
    return total


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    gold,groups,protected=context()
    prior=cost_total(True)
    if prior+5>100:raise BudgetStop('Insufficient original project budget for recovery cap')
    protocol={'purpose':'one explicit post-outage recovery batch, no quality selection','evaluation':'original evaluator',
              'groups':[{k:g[k] for k in ['family','shots','budget','ids']} for g in groups],
              'parameters':{g['family']:{'endpoint':g['connection']['endpoint'],'parameters':{'model':g['connection']['model'],**g['config']['parameters']},
                  'examples_sha256':object_hash(g['examples']),'pricing':g['config']['pricing']} for g in groups},
              'pricing_verified_date':'2026-10-30','new_budget_cap_cny':5,'project_cap_cny':100,'workers_per_provider':4,
              'max_new_attempts_per_sample':3,'no_hidden_repair_or_quality_retry':True,
              'protected_inputs_sha256':protected,'script_sha256':file_hash(Path(__file__))}
    freeze=OUT/'protocol.json'
    if freeze.exists():
        if load_json(freeze)['protocol']!=protocol:raise RuntimeError('Recovery freeze changed')
    else:save_json(freeze,{'frozen_at':utc_now(),'protocol':protocol})
    print(json.dumps({'mode':'PAID RECOVERY' if args.run else 'OFFLINE PREFLIGHT','groups':[{k:g[k] for k in ['family','shots','budget']}|{'samples':len(g['ids']),'key_present':bool(g['connection']['key'])} for g in groups],
                      'prior_project_upper_cny':prior,'new_budget_cap_cny':5,'project_cap_cny':100},ensure_ascii=True),flush=True)
    if not args.run:return
    if not all(g['connection']['key'] for g in groups):raise RuntimeError('Required API key missing; no call sent')
    lock=ROOT/'runs/llm/process.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    summaries=[]
    try:
        os.write(fd,str(os.getpid()).encode())
        if cost_total(True)!=prior:raise InterfaceStop('Other process changed project budget')
        for g in groups:
            folder=OUT/g['family']
            client=(GuardedGLMClient if g['family']=='glm' else MiMoClient)(folder,g['config'],pilot_messages,g['budget'],workers=4)
            def progress(n):
                state={'family':g['family'],'completed':n,'total':len(g['ids']),'new_ledger_upper_cny':client.total(),'updated_at':utc_now()}
                atomic_json(folder/'progress.json',state)
                if n==1 or n%10==0 or n==len(g['ids']):print(json.dumps(state,ensure_ascii=True),flush=True)
            replacements=predict_rows(client,g['rows'],g['examples'],4,progress)
            merged=merge_predictions(g['original'],replacements,g['ids'])
            before,_,_=evaluate(gold,g['original']);after,counts,cases=evaluate(gold,merged)
            write_jsonl(folder/'recovery_predictions.jsonl',replacements)
            write_jsonl(folder/'merged_predictions.jsonl',merged)
            save_json(folder/'metrics_before.json',before);save_json(folder/'metrics.json',after)
            write_jsonl(folder/'sentence_counts.jsonl',counts);write_jsonl(folder/'errors.jsonl',cases)
            ledger=load_json(folder/'ledger.json')['attempts']
            summary={'family':g['family'],'shots':g['shots'],'full_test_samples':len(gold),'requested_recovery_samples':len(g['ids']),
                     'recovered_ok':sum(p['status']=='ok' for p in replacements),'new_statuses':dict(Counter(p['status'] for p in replacements)),
                     'preserved_original_samples':len(gold)-len(g['ids']),'entity_f1_before':before['entity']['f1'],'entity_f1_after':after['entity']['f1'],
                     'token_f1_before':before['token']['f1'],'token_f1_after':after['token']['f1'],
                     'known_usage_estimate_cny':sum(a['charged_upper_cny'] for a in ledger if a['status']=='response'),
                     'unknown_reservations_cny':sum(a['charged_upper_cny'] for a in ledger if a['status']!='response'),
                     'ledger_upper_cny':client.total(),'completed_at':utc_now(),
                     'note':'Merge first final recovery outcomes; all other predictions unchanged. Original refusal/format failures remain.'}
            save_json(folder/'summary.json',summary);summaries.append(summary)
            print('GROUP COMPLETE '+json.dumps(summary,ensure_ascii=True),flush=True)
        if not all(file_hash(ROOT/p)==h for p,h in protected.items()):raise RuntimeError('Original evidence changed')
        total=cost_total()
        if total>100+1e-9:raise RuntimeError('Project budget exceeded')
        result={'status':'COMPLETE','groups':summaries,'original_files_unchanged':True,'project_ledger_upper_cny':total,
                'new_api_attempts':sum(len(load_json(OUT/g['family']/'ledger.json')['attempts']) for g in groups),
                'new_ledger_upper_cny':sum(g['ledger_upper_cny'] for g in summaries),'completed_at':utc_now(),
                'primary_report_not_yet_updated':True}
        save_json(OUT/'summary.json',result)
        lines=['# 网络失败恢复结果','','原模型、提示、示例与请求参数不变；每条只采用恢复批次的首个最终结果，未按质量挑选。原结果与恢复记录分别保留。','','| 配置 | 恢复成功/目标 | 原实体F1% | 恢复后实体F1% | 原Token F1% | 恢复后Token F1% | 新增估算元 |','|---|---:|---:|---:|---:|---:|---:|']
        for r in summaries:lines.append(f"| {r['family']} {r['shots']}-shot | {r['recovered_ok']}/{r['requested_recovery_samples']} | {100*r['entity_f1_before']:.4f} | {100*r['entity_f1_after']:.4f} | {100*r['token_f1_before']:.4f} | {100*r['token_f1_after']:.4f} | {r['known_usage_estimate_cny']:.6f} |")
        lines+=['','均对完整1209句Test评分。原始文件哈希未变化；费用为usage估算，实际账单待平台核对。原正式报告尚未替换为恢复结果。']
        report=ROOT/'reports/network_recovery_results.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('RECOVERY COMPLETE '+json.dumps(result,ensure_ascii=True),flush=True)
    except BaseException as exc:
        save_json(OUT/'stopped.json',{'at':utc_now(),'exception':type(exc).__name__,'completed_groups':summaries,'cached_work_preserved':True})
        raise
    finally:
        os.close(fd);lock.unlink()

if __name__=='__main__':main()
