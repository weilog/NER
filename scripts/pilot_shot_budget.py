"""Explicit, bounded 20-sentence x four-shot pilot; never runs full Dev or Test."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import numpy as np
from ner_experiment.common import ROOT, LABELS, project_data, valid_train, load_json, save_json, write_jsonl, object_hash, file_hash, utc_now
from ner_experiment.models.llm import Client, SYSTEM, PARAMETERS, input_text, choose_examples, parse_response
from ner_experiment.evaluation.metrics import evaluate, checked_entities, overlapping_entities
from run_baselines import verify_data

SHOTS=[0,2,4,8]
SYSTEM_TEXT=SYSTEM.replace('不带前后标点、书名号或无关修饰语',
    '不带无关的前后标点或修饰语；book/movie若以《》包围作品名，保留书名号作为实体边界')
SYSTEM_TEXT=SYSTEM_TEXT.replace('{"start":整数,"end":整数,"type":"类别英文名"}',
    '{"text":"原文中的实体文字","start":整数,"end":整数,"type":"类别英文名"}')
SYSTEM_TEXT+='\n每个实体必须包含text字段，且必须严格满足原文[start:end] == text。不要自行改写实体文字。'

def pilot_messages(row,examples):
    result=[{'role':'system','content':SYSTEM_TEXT}]
    for e in examples:
        entities=[{'text':e['text'][x['start']:x['end']],'start':x['start'],'end':x['end'],'type':x['type']} for x in e['entities']]
        result.extend([{'role':'user','content':input_text(e)},
                       {'role':'assistant','content':json.dumps({'entities':entities},ensure_ascii=False)}])
    return result+[{'role':'user','content':input_text(row)}]

def parse_text_response(raw):
    status,entities=parse_response(raw)
    if status=='ok' and any(not isinstance(e.get('text'),str) for e in entities):return 'format_error',[]
    return status,entities

def summarize_usage(usage,prices,cache=False):
    prompt=usage['prompt_tokens'];completion=usage['completion_tokens']
    hit=usage.get('prompt_cache_hit_tokens',0) if cache else 0
    return (hit*prices['input_cache_hit']+(prompt-hit)*prices['input_cache_miss']+completion*prices['output'])/1e6

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true',help='Authorize this bounded real-API pilot only')
    parser.add_argument('--output-dir',type=Path,default=Path('runs/llm_text_pilot_v1'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not args.run:parser.error('Use --run to execute up to 80 original requests, plus bounded network retries')
    if not 1<=args.workers<=4:raise ValueError('Use 1..4 workers for this pilot')
    base=(ROOT/args.output_dir).resolve()
    if not base.is_relative_to((ROOT/'runs').resolve()) or base==(ROOT/'runs').resolve():raise ValueError('Use a subdirectory of runs/')
    manifest,_=verify_data(ROOT)
    seed=load_json(ROOT/'configs/split_seed.json')['seed']
    examples=choose_examples(valid_train(),seed)
    dev=project_data('dev');lookup={r['id']:r for r in dev}
    ids=load_json(ROOT/'data/splits/cluener2020/pilot_ids.json');rows=[lookup[i] for i in ids]
    if len(rows)!=20 or len(set(ids))!=20:raise ValueError('Expected fixed 20 pilot IDs')
    pricing=load_json(ROOT/'configs/pricing.json')
    protocol={'version':'text-schema-boundary-pilot-v1','system':SYSTEM_TEXT,'parameters':PARAMETERS,
        'pilot_ids':ids,'example_ids':[r['id'] for r in examples],'examples':examples,'shots':SHOTS,
        'data_manifest_sha256':object_hash(manifest),'pricing':pricing,
        'code_sha256':{p.name:file_hash(p) for p in [Path(__file__),ROOT/'src/ner_experiment/models/llm.py']},
        'scope':'pilot only; no full Dev/Test calls; span text mismatch is diagnostic, no relocation or repair'}
    protocol_path=base/'protocol.json'
    if protocol_path.exists():
        if load_json(protocol_path)!=protocol:raise RuntimeError('Pilot protocol changed; use new output directory')
    elif base.exists() and any(base.iterdir()):raise RuntimeError('Nonempty directory is protected')
    report_path=base/'budget_summary.json'
    if report_path.exists():
        print(json.dumps(load_json(report_path),ensure_ascii=False,indent=2));return
    lock=ROOT/'runs/llm/process.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode())
    try:
        old_cost=sum(a['charged_upper_cny'] for p in (ROOT/'runs').rglob('ledger.json')
                     if p.resolve()!=(base/'ledger.json').resolve() for a in load_json(p)['attempts'])
        save_json(protocol_path,protocol)
        # This trial cannot spend more than 5 CNY, nor exceed project remainder.
        client=Client(base,budget=max(0,min(5,100-old_cost)),message_builder=pilot_messages,response_parser=parse_text_response)
        groups=[]
        for shots in SHOTS:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                predictions=list(pool.map(lambda r:client.predict(r,examples[:shots]),rows))
            folder=base/f'shot_{shots}'
            write_jsonl(folder/'predictions.jsonl',predictions)
            metrics,counts,cases=evaluate(rows,predictions)
            save_json(folder/'metrics.json',metrics);write_jsonl(folder/'errors.jsonl',cases);write_jsonl(folder/'sentence_counts.jsonl',counts)
            text_mismatch=0;valid_total=0;entity_total=0;reasoning_present=0;known_usage=[]
            actual_models=set()
            for row,pred in zip(rows,predictions):
                record=load_json(client.cache/(pred['cache_key']+'.json'))
                raw=record['raw_response'] or {}
                reasoning_present+=bool(((raw.get('choices') or [{}])[0].get('message') or {}).get('reasoning_content'))
                if pred.get('actual_model'):actual_models.add(pred['actual_model'])
                u=pred.get('usage',{})
                if 'prompt_tokens' in u and 'completion_tokens' in u:known_usage.append(u)
                for e in pred['entities']:
                    entity_total+=1
                    valid,_,_=checked_entities([e],len(row['tokens']))
                    if valid:
                        valid_total+=1;text_mismatch+=row['text'][e['start']:e['end']]!=e['text']
            if len(known_usage)!=20:raise RuntimeError('Incomplete usage: cannot forecast reliably; inspect cached failures')
            costs=[summarize_usage(u,pricing['peak'],cache=True) for u in known_usage]
            cold=[summarize_usage(u,pricing['peak']) for u in known_usage]
            lat=[p['latency_seconds'] for p in predictions]
            summary={'shots':shots,'samples':20,'example_category_count':len({e['type'] for r in examples[:shots] for e in r['entities']}),
                'prompt_tokens_mean':float(np.mean([u['prompt_tokens'] for u in known_usage])),
                'completion_tokens_mean':float(np.mean([u['completion_tokens'] for u in known_usage])),
                'prompt_tokens_total':sum(u['prompt_tokens'] for u in known_usage),'completion_tokens_total':sum(u['completion_tokens'] for u in known_usage),
                'cost_peak_with_observed_cache_cny':sum(costs),'cost_peak_no_cache_cny':sum(cold),
                'latency_median_seconds':float(np.median(lat)),'latency_p95_seconds':float(np.quantile(lat,.95)),
                'actual_models':sorted(actual_models),'nonempty_reasoning_response_samples':reasoning_present,
                'failure_rate':metrics['failure_rate'],'invalid_sample_rate':metrics['invalid_sample_rate'],
                'duplicate_entities':metrics['diagnostics'].get('duplicate_entities',0),'overlap_samples':metrics['diagnostics'].get('overlap_samples',0),
                'valid_spans':valid_total,'entities':entity_total,'text_span_mismatches':text_mismatch,
                'text_span_mismatch_rate':text_mismatch/valid_total if valid_total else 0,
                'token_f1':metrics['token']['f1'],'entity_f1':metrics['entity']['f1']}
            groups.append(summary);save_json(folder/'usage_summary.json',summary)
            print('PILOT',shots,json.dumps(summary,ensure_ascii=False),flush=True)
        per_four=sum(g['cost_peak_no_cache_cny']/20 for g in groups)
        # No Test content is read; use the manifest count only.
        n_dev=manifest['stats']['dev']['sentences'];n_test=manifest['stats']['test']['sentences']
        full=per_four*(n_dev+n_test)
        attempts=client.ledger['attempts']
        result={'created_at':utc_now(),'scope':'80 pilot requests only','groups':groups,
            'pricing_source':pricing['source'],'pricing_verified_date':'2026-10-24','pricing':pricing,
            'pilot_original_requests':80,'pilot_attempts_including_retries':len(attempts),'pilot_peak_ledger_cny':client.total(),
            'prior_peak_ledger_cny':old_cost,'project_peak_ledger_cny':old_cost+client.total(),
            'forecast':{'unique_full_dev_test_requests':4*(n_dev+n_test),
                'four_shot_configs_per_300_dev_cny':per_four*300,
                'full_dev_four_configs_cny':per_four*n_dev,'full_test_four_configs_cny':per_four*n_test,
                'full_dev_test_peak_no_cache_cny':full,
                'full_dev_test_with_30pct_margin_cny':full*1.3,
                'project_total_with_prior_and_30pct_margin_cny':old_cost+full*1.3,
                'project_total_with_prior_and_2x_stress_cny':old_cost+full*2,
                'sufficient_100cny_at_30pct_margin':old_cost+full*1.3<=100},
            'assumptions':['Means from fixed 20 Dev sentences per shot; not an F1 selection experiment',
                'High-peak prices, no provider input-cache benefit; 30% contingency is a scenario, not statistical confidence interval',
                'Pilot/300-Dev results reused inside complete Dev; no independent duplicate billing assumed',
                'Prompt/examples/API settings stay identical; re-running another prompt version costs extra',
                'Ledger amounts are estimates, not account balance or invoice; reserved uncertain attempts remain charged',
                'Thinking disabled in request; absence of reasoning content is observable but not independent server attestation']}
        save_json(report_path,result)
        lines=['# 四档 Shot 新提示试跑与预算估算','',f'生成时间：{result["created_at"]}。仅固定 20 条 Dev × 4 档，共 80 次原始请求。未运行完整 Dev/Test。','',
            '| Shot | 平均输入 token | 平均输出 token | 20条高峰费用上界估计 | 无缓存高峰估计 | 中位延迟 | 文本/跨度不一致 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for g in groups:
            lines.append(f'| {g["shots"]} | {g["prompt_tokens_mean"]:.1f} | {g["completion_tokens_mean"]:.1f} | ¥{g["cost_peak_with_observed_cache_cny"]:.4f} | ¥{g["cost_peak_no_cache_cny"]:.4f} | {g["latency_median_seconds"]:.3f}s | {g["text_span_mismatches"]}/{g["valid_spans"]} |')
        f=result['forecast']
        lines+=['',f'完整 Dev/Test 四档：{f["unique_full_dev_test_requests"]} 次独立请求，按无输入缓存、高峰价格估计 ¥{full:.2f}；加 30% 余量为 ¥{full*1.3:.2f}。',
            f'既有账本高峰上界 ¥{old_cost:.2f}，合计（含30%余量）¥{f["project_total_with_prior_and_30pct_margin_cny"]:.2f}；成本翻倍压力情景 ¥{f["project_total_with_prior_and_2x_stress_cny"]:.2f}。','',
            '20条样本只能提供初步费用估计，不能用来选最佳 Shot。费用不是实扣账单，未检查账户余额。新提示增加实体 text，并依据 Train 中书名/影视实体保留书名号的证据修改规则；不与旧提示结果混合。',
            '文本/跨度不一致仅诊断：不移动跨度、不模型修复。请求关闭 thinking；保存实际返回模型与 reasoning_content 是否非空，不宣称独立证明服务端内部运行模式。',
            f'价格来源：{pricing["source"]}（2026-10-24核对）。完整请求与响应见独立 cache，API Key 未保存。']
        path=ROOT/'reports/llm_text_pilot_budget.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('BUDGET',json.dumps(result['forecast'],ensure_ascii=False),flush=True)
    finally:
        os.close(fd);lock.unlink()

if __name__=='__main__':main()
