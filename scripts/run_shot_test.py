"""Frozen four-shot Test extension. No Dev selection or prompt changes on Test."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
from ner_experiment.common import ROOT, LABELS, load_json, save_json, project_data, file_hash, object_hash, read_jsonl, utc_now
from ner_experiment.models.llm import Client, PARAMETERS
from ner_experiment.evaluation.metrics import checked_entities, evaluate
from pilot_shot_budget import SHOTS, SYSTEM_TEXT, pilot_messages, parse_text_response
from run_baselines import verify_data
from run_llm import run_group, check_group, group_fingerprint

def freeze_once(path,protocol):
    if path.exists():
        old=load_json(path)
        if old['protocol']!=protocol:raise RuntimeError('Frozen Test protocol changed. Do not adjust using Test results.')
        return old
    if path.parent.exists() and any(path.parent.iterdir()):raise RuntimeError('Nonempty output directory without freeze is protected')
    record={'frozen_at':utc_now(),'protocol':protocol}
    save_json(path,record)
    return record

def validate_pilot(pilot,summary):
    if pilot['system']!=SYSTEM_TEXT or pilot['parameters']!=PARAMETERS or pilot['shots']!=SHOTS:
        raise RuntimeError('Current prompt/API configuration differs from completed pilot')
    for name,digest in pilot['code_sha256'].items():
        path=ROOT/('src/ner_experiment/models/llm.py' if name=='llm.py' else 'scripts'/Path(name))
        if file_hash(path)!=digest:raise RuntimeError('Pilot implementation changed')
    if any(g['failure_rate'] or g['invalid_sample_rate'] or g['text_span_mismatches'] for g in summary['groups']):
        raise RuntimeError('Pilot requires review before full Test')

def protected_files():
    paths=[ROOT/'configs/frozen_experiment.json',ROOT/'runs/llm/ledger.json',
           ROOT/'runs/llm/selection.json',ROOT/'runs/baselines/selection.json']
    paths+=list((ROOT/'runs/final').rglob('*.json'))+list((ROOT/'runs/final').rglob('*.jsonl'))
    paths+=list((ROOT/'runs/llm_text_pilot_v1').rglob('*.json'))
    return {str(p.relative_to(ROOT)):file_hash(p) for p in paths if p.is_file()}

def diagnostic_summary(rows,predictions,client):
    valid_total=0;mismatch=0;reasoning=0;usage_missing=0
    for row,pred in zip(rows,predictions):
        raw=load_json(client.cache/(pred['cache_key']+'.json'))['raw_response'] or {}
        reasoning+=bool(((raw.get('choices') or [{}])[0].get('message') or {}).get('reasoning_content'))
        usage=pred.get('usage',{})
        usage_missing+=not all(k in usage for k in ['prompt_tokens','completion_tokens'])
        for e in pred['entities']:
            valid,_,_=checked_entities([e],len(row['tokens']))
            if valid:
                valid_total+=1;mismatch+=row['text'][e['start']:e['end']]!=e['text']
    keys={p['cache_key'] for p in predictions}
    attempts=[a for a in client.ledger['attempts'] if a['key'] in keys]
    return {'text_span_mismatches':mismatch,'valid_spans':valid_total,
        'text_span_mismatch_rate':mismatch/valid_total if valid_total else 0,
        'nonempty_reasoning_response_samples':reasoning,'usage_missing_samples':usage_missing,
        'attempts_including_retries':len(attempts),'cost_peak_ledger_cny':sum(a['charged_upper_cny'] for a in attempts),
        'prompt_tokens':sum(a.get('usage',{}).get('prompt_tokens',0) for a in attempts),
        'completion_tokens':sum(a.get('usage',{}).get('completion_tokens',0) for a in attempts)}

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--output-dir',type=Path,default=Path('runs/llm_text_test_v1'))
    args=parser.parse_args()
    if not args.run:parser.error('Use --run for the explicitly authorized complete four-shot Test')
    if not 1<=args.workers<=16:raise ValueError('workers must be 1..16')
    base=(ROOT/args.output_dir).resolve()
    if not base.is_relative_to((ROOT/'runs').resolve()) or base==(ROOT/'runs').resolve():raise ValueError('Output must be within runs/')
    manifest,_=verify_data(ROOT)
    pilot=load_json(ROOT/'runs/llm_text_pilot_v1/protocol.json')
    trial=load_json(ROOT/'runs/llm_text_pilot_v1/budget_summary.json')
    validate_pilot(pilot,trial)
    examples=pilot['examples']
    source_paths=[Path(__file__).resolve(),ROOT/'scripts/pilot_shot_budget.py',ROOT/'scripts/run_llm.py',
        ROOT/'scripts/run_baselines.py',ROOT/'src/ner_experiment/models/llm.py',
        ROOT/'src/ner_experiment/evaluation/metrics.py',ROOT/'src/ner_experiment/common.py',ROOT/'configs/pricing.json']
    protocol={'version':'text-schema-four-shot-test-extension-v1','system':SYSTEM_TEXT,'parameters':PARAMETERS,
        'examples':examples,'example_ids':pilot['example_ids'],'shots':SHOTS,'workers':args.workers,
        'dataset_manifest':manifest,'test_size':1209,'pilot_protocol_sha256':object_hash(pilot),
        'source_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in source_paths},
        'evaluation':'V3 exact spans and overlapping (sample, character, BIO) sets; mismatched text diagnostic only',
        'selection':'No selection based on this Test. New-version Dev tuning/full Dev skipped at user request.',
        'disclosure':'Extension after historical Test inspection. New prompt has no new full-Dev-selected best-shot claim.',
        'budget_cny':100,'cache':'same request only; failures remain failures; network retries at most two; no LLM repair'}
    freeze_path=base/'frozen_protocol.json'
    # Snapshot source/configuration before loading Test examples or making requests.
    freeze=freeze_once(freeze_path,protocol)
    test=project_data('test')
    if len(test)!=1209 or len({r['id'] for r in test})!=1209:raise RuntimeError('Fixed Test size/ID mismatch')
    if {r['text'] for r in examples}&{r['text'] for r in test}:raise RuntimeError('Few-shot examples overlap Test')
    for shots in SHOTS:check_group(base/f'shot_{shots}',group_fingerprint(test,examples[:shots]))
    lock=ROOT/'runs/llm/process.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode())
    try:
        other_cost=sum(a['charged_upper_cny'] for p in (ROOT/'runs').rglob('ledger.json')
                       if p.resolve()!=(base/'ledger.json').resolve() for a in load_json(p)['attempts'])
        existing_cost=sum(a['charged_upper_cny'] for a in load_json(base/'ledger.json')['attempts']) if (base/'ledger.json').exists() else 0
        forecast=trial['forecast']['full_test_four_configs_cny']
        if other_cost+existing_cost+forecast*1.3>100:raise RuntimeError('Conservative forecast does not fit remaining project budget; stop before requests')
        protected=protected_files()
        audit_path=base/'protected_before.json'
        if audit_path.exists():
            if load_json(audit_path)!=protected:raise RuntimeError('Historical protected artifacts changed')
        else:save_json(audit_path,protected)
        client=Client(base,budget=max(0,100-other_cost),message_builder=pilot_messages,response_parser=parse_text_response)
        print('START four-shot Test; samples=1209; originals=4836; workers=',args.workers,'prior_peak_cny=',round(other_cost,4),flush=True)
        groups=[]
        for shots in SHOTS:
            folder=base/f'shot_{shots}'
            metrics=run_group(client,test,examples[:shots],folder,args.workers)
            predictions=read_jsonl(folder/'predictions.jsonl')
            # Recompute saved predictions without any API call.
            replay,_,_=evaluate(test,predictions)
            if replay!=metrics:raise RuntimeError('Offline metric replay mismatch')
            run=load_json(folder/'run.json')
            diag=diagnostic_summary(test,predictions,client)
            summary={'shots':shots,'samples':1209,'example_category_count':len({e['type'] for r in examples[:shots] for e in r['entities']}),
                'token':metrics['token'],'entity':metrics['entity'],'entity_macro_f1':metrics['entity_macro_f1'],
                'entity_per_type':metrics['entity_per_type'],'error_counts':metrics['error_counts'],
                'failure_rate':metrics['failure_rate'],'invalid_sample_rate':metrics['invalid_sample_rate'],
                'diagnostics':metrics['diagnostics'],'actual_models':run['actual_models'],
                'latency_median_seconds':run['latency_median_seconds'],'latency_p95_seconds':run['latency_p95_seconds'],**diag}
            save_json(folder/'diagnostic_summary.json',summary);groups.append(summary)
        if protected!=protected_files():raise RuntimeError('Protected historical artifact changed')
        result={'completed_at':utc_now(),'frozen_at':freeze['frozen_at'],'protocol_sha256':object_hash(protocol),
            'groups':groups,'new_peak_ledger_cny':client.total(),'prior_peak_ledger_cny':other_cost,
            'project_peak_ledger_cny':other_cost+client.total(),'new_original_requests':4836,
            'new_attempts_including_retries':len(client.ledger['attempts']),
            'offline_replay_matches':True,'historical_artifacts_unchanged':True,
            'selection':protocol['selection'],'disclosure':protocol['disclosure']}
        save_json(base/'summary.json',result)
        lines=['# 新版 Prompt 四档 Shot 完整 Test 对比','',f'冻结时间：{freeze["frozen_at"]}；完成时间：{result["completed_at"]}。每档固定 1209 条。',
            '', '按用户要求跳过新版300条Dev和完整Dev，直接做预先固定的0/2/4/8-shot Test对照。本轮不从Test选择最佳配置。属于历史Test查看后的扩展实验，不能称为首次盲测或测试前预注册。',
            '', '| Shot | Token P | Token R | Token F1 | 实体 P | 实体 R | 实体 F1 | 高峰费用估计 | 延迟中位数/P95 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for g in groups:
            t,e=g['token'],g['entity']
            lines.append(f'| {g["shots"]} | {t["precision"]:.2%} | {t["recall"]:.2%} | {t["f1"]:.2%} | {e["precision"]:.2%} | {e["recall"]:.2%} | {e["f1"]:.2%} | ¥{g["cost_peak_ledger_cny"]:.4f} | {g["latency_median_seconds"]:.3f}s / {g["latency_p95_seconds"]:.3f}s |')
        lines+=['',f'本轮原始请求4836次，含重试共{result["new_attempts_including_retries"]}次；高峰账本估计 ¥{client.total():.4f}，累计 ¥{result["project_peak_ledger_cny"]:.4f}。不是账户实扣账单。',
            '', '| Shot | 调用/格式失败率 | 无效样本率 | 重叠句数 | 重复实体 | 文本跨度不一致 |',
            '|---|---:|---:|---:|---:|---:|']
        for g in groups:
            lines.append(f'| {g["shots"]} | {g["failure_rate"]:.2%} | {g["invalid_sample_rate"]:.2%} | {g["diagnostics"].get("overlap_samples",0)} | {g["diagnostics"].get("duplicate_entities",0)} | {g["text_span_mismatches"]}/{g["valid_spans"]} |')
        lines+=['','## 各类别严格实体 F1','','| 类别 | 0-shot | 2-shot | 4-shot | 8-shot |','|---|---:|---:|---:|---:|']
        for category in LABELS:lines.append('| '+category+' | '+' | '.join(f'{g["entity_per_type"][category]["f1"]:.2%}' for g in groups)+' |')
        lines+=['','## 自动错误归因','','| 类别 | 0-shot | 2-shot | 4-shot | 8-shot |','|---|---:|---:|---:|---:|']
        for category in ['boundary','type','boundary_and_type','missed','spurious']:
            lines.append('| '+category+' | '+' | '.join(str(g['error_counts'].get(category,0)) for g in groups)+' |')
        lines+=['','所有模型采用同一套嵌套示例前缀，2-shot覆盖8类，4/8-shot覆盖10类；数量、类别覆盖与示例内容共同变化，不构成纯粹控制示例数量的因果实验。',
            '主指标使用V3：合法去重实体全部保留（包括重叠）；文本与跨度不一致只记录，不移动跨度或隐藏修复。原始请求与响应在独立cache保存。',
            'Prompt、实体text字段和书名号规则与旧版不同，不把本轮与旧版混成同一Shot曲线。实际模型标识、参数、日期、usage和诊断详见本地summary及缓存。',
            '全部预测离线重评一致；历史结果、账本与旧冻结文件哈希保持不变。']
        (ROOT/'reports/llm_text_test_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('COMPLETE total_peak_cny=',round(client.total(),4),'project_peak_cny=',round(result['project_peak_ledger_cny'],4),flush=True)
    finally:
        os.close(fd);lock.unlink()

if __name__=='__main__':main()
