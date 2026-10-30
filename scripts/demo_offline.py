"""Three-stage real demo: live original CRF, final prediction replay, two verified errors; no API."""
import argparse
import json
import pickle
import sys
import time
from ner_experiment.common import ROOT, load_json, save_json, utc_now
from ner_experiment.evaluation.metrics_v1 import evaluate
from ner_experiment.models.baselines import CRF
from demo_evidence import (DemoError, require, verify_snapshot, local_path, checked_rows,
    unique, read_table, compare_metrics, load_predictions, select_case)


def print_metrics(label, metrics):
    print(f"{label}: TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print(f"  Precision={metrics['precision']*100:.2f}%  Recall={metrics['recall']*100:.2f}%  F1={metrics['f1']*100:.2f}%")


def render_entities(label, row, entities):
    print(label+':')
    if not entities:print('  （空）')
    for entity in entities:
        start,end=entity.get('start'),entity.get('end')
        valid=type(start) is int and type(end) is int and 0<=start<end<=len(row['text'])
        text=row['text'][start:end] if valid else '（无效跨度，未投影）'
        print(f"  {text!r}  [{start},{end})  type={entity.get('type')}")
        if 'text' in entity and entity['text']!=text:print('  模型text字段：'+repr(entity['text']))


def show_case(case):
    row=case['row'];error=case['pair'];g,p=error['gold'],error['predicted']
    kind='实体边界错误' if case['kind']=='boundary' else '实体类型错误'
    print(f"\n--- {kind} | {case['method']} | {row['id']} ---")
    print('最终预测文件：'+case['source_path'])
    if case['used_alternative']:
        print(f"原首选样本 {case['preferred_id']} 不再满足声明条件，已从该最终预测文件核验替代案例。")
    print('原文：'+repr(row['text']))
    render_entities('Gold',row,row['entities'])
    render_entities('Prediction',row,case['prediction']['entities'])
    if case['kind']=='boundary':
        if p[0]<=g[0] and p[1]>=g[1]:
            extra=row['text'][p[0]:g[0]]+row['text'][g[1]:p[1]]
            explanation=f'类型相同，预测额外包含{extra!r}，共{len(extra)}个字符，因此完整实体边界匹配失败。'
        else:explanation=f'类型相同，但Gold边界{g[:2]}与预测边界{p[:2]}不同，属于边界错误。'
    else:explanation=f'位置完全相同 [{g[0]},{g[1]})，但Gold类型为{g[2]}，预测为{p[2]}，仍是实体识别失败。'
    print('错误归因：'+explanation)
    print_metrics('Token',case['metrics']['token'])
    print_metrics('Entity',case['metrics']['entity'])
    if case['kind']=='boundary':print('字符部分正确不代表完整实体正确；上述指标均为当前样本现场重算值。')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--student-id',required=True)
    parser.add_argument('--auto',action='store_true',help='Rehearsal: skip lecture pauses')
    parser.add_argument('--manifest',default='reports/demo_inputs_manifest.json')
    args=parser.parse_args()
    require(bool(args.name.strip()) and bool(args.student_id.strip()),'姓名和学号不得为空')
    def banner(title):
        print('\n'+'='*64)
        print(f'演示者：{args.name}    学号：{args.student_id}')
        print(title);print('='*64,flush=True)
    def pause(message='讲解后按 Enter 继续：'):
        if not args.auto:input('\n'+message)
    banner('CLUENER2020：真实推断 → 双层评估 → 两类真实难例')
    print('现场重推完整Test的CRF；LLM为最终保存响应的离线重评。')
    print('无API请求、不读取密钥、不重新训练。姓名学号显示于各阶段。')
    pause()

    banner('1 / 3 现场使用原训练流水线CRF，推断完整1209句Test')
    started=time.perf_counter()
    manifest_path=local_path(ROOT,args.manifest)
    require(manifest_path.is_file(),'缺少独立输入清单，请先运行 scripts/freeze_demo_inputs.py')
    snapshot=verify_snapshot(ROOT,load_json(manifest_path))
    gold=checked_rows(ROOT,snapshot['test_path'],snapshot['files_sha256'][snapshot['test_path']])
    gold_map=unique(gold,lambda r:r['id'],'gold ID')
    require(len(gold)==snapshot['samples']==1209,'Fixed Test count differs')
    for row in gold:require(row['tokens']==list(row['text']),f'Tokens/text differ: {row["id"]}')
    entries=unique(snapshot['methods'],lambda e:e['method'],'final method')
    table_path=local_path(ROOT,snapshot['reference_table']);table_rows,reference=read_table(table_path)
    require([r['method'] for r in table_rows]==list(entries),'Final table and prediction registry differ')
    print(f"独立SHA256清单通过：{len(snapshot['files_sha256'])}个输入，含全部15组最终预测与数据。",flush=True)
    model_path=local_path(ROOT,snapshot['crf_model_path'])
    with model_path.open('rb') as handle:model=pickle.load(handle)
    require(type(model) is CRF,'Loaded model is not the original ner_experiment.models.baselines.CRF')
    print('模型来源：'+snapshot['crf_model_path'])
    print('调用原类 CRF.predict(row) → features(row) → predict_single；正在现场推断……',flush=True)
    # Exactly the same row dict and predict method as scripts/run_final.py; no replacement features.
    live=[model.predict(row) for row in gold]
    saved,saved_by_id=load_predictions(ROOT,entries['crf'],gold_map)
    live_by_id=unique(live,lambda p:p['id'],'live CRF prediction ID')
    require(set(live_by_id)==set(gold_map),'Live CRF IDs differ')
    for row in gold:
        identity=row['id']
        for key in ['id','status','raw_bio','entities']:
            actual=live_by_id[identity].get(key);wanted=saved_by_id[identity].get(key)
            require(actual==wanted,f'CRF live/saved prediction differs: id={identity}, field={key}, actual={actual!r}, reference={wanted!r}')
    live_metrics=evaluate(gold,live)[0];compare_metrics('crf [live]',live_metrics,reference['crf'])
    timings={'live_crf_seconds':time.perf_counter()-started}
    print('完整1209条现场预测与原保存输出逐条一致。')
    print_metrics('CRF Token',live_metrics['token']);print_metrics('CRF Entity',live_metrics['entity'])
    print(f"实际计算用时：{timings['live_crf_seconds']:.2f}秒",flush=True)
    pause()

    banner('2 / 3 对最终15组保存预测逐组重新评分（含网络恢复结果）')
    started=time.perf_counter();scores={}
    print('方法                    Token F1    Entity F1',flush=True)
    for method,entry in entries.items():
        predictions,_=load_predictions(ROOT,entry,gold_map)
        metrics=evaluate(gold,predictions)[0];compare_metrics(method,metrics,reference[method])
        print(f"{method:<23} {metrics['token']['f1']*100:7.2f}%    {metrics['entity']['f1']*100:7.2f}%",flush=True)
        scores[method]={level:metrics[level] for level in ['token','entity']}
    timings['all_scores_seconds']=time.perf_counter()-started
    print(f"15组TP/FP/FN严格一致，P/R/F1按CSV精度校验通过；用时{timings['all_scores_seconds']:.2f}秒。")
    print('恢复配置来自真实协议/summary/目录的交叉核验，不按family猜shot。',flush=True)
    pause()

    banner('3 / 3 现场读取最终预测并验证：实体边界错误与实体类型错误')
    started=time.perf_counter()
    require('mimo_4' in entries and 'glm_8' in entries,'Required final model configuration absent')
    cases=[select_case(ROOT,entries['mimo_4'],gold_map,'train:2506','boundary'),
           select_case(ROOT,entries['glm_8'],gold_map,'train:50','type')]
    timings['case_evaluation_seconds']=time.perf_counter()-started
    for index,case in enumerate(cases):
        if index:
            pause('讲解完边界错误后按 Enter 展示类型错误：')
            print(f'\n演示者：{args.name}    学号：{args.student_id}')
        show_case(case)
    total=sum(timings.values())
    output=ROOT/'runs/demo_recordings'/('demo_'+utc_now().replace(':','-')+'.json')
    save_json(output,{'presenter_name':args.name,'student_id':args.student_id,'auto_rehearsal':args.auto,
        'manifest_path':args.manifest,'live_crf_samples':len(gold),'saved_prediction_groups_recomputed':len(scores),
        'timings':timings,'measured_compute_seconds':total,'new_api_calls':0,'scores':scores,'hard_cases':cases})
    print(f'\n演示完成：现场计算合计{total:.2f}秒，不含讲解与等待。')
    print('本地记录：'+output.relative_to(ROOT).as_posix())
    print(f'演示者：{args.name}    学号：{args.student_id}',flush=True)

if __name__=='__main__':
    try:main()
    except (DemoError,FileNotFoundError,KeyError,ValueError) as error:
        print(f'\n演示停止：{error}',file=sys.stderr)
        raise SystemExit(1)
