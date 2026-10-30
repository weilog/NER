"""Eight evidence-checked AI-assisted examples, one per observed error/diagnostic category. No API."""
from collections import Counter
import json
from pathlib import Path
from ner_experiment.common import ROOT,read_jsonl,load_json,save_json,file_hash,object_hash

CASES=[
    {'id':'E01','kind':'boundary','title':'边界错误','method':'llm_0','sample_id':'train:523',
     'gold_focus':[16,18,'organization'],'predicted_focus':[16,20,'organization'],
     'finding':'金标只标“西甲”，预测多包含“联赛”；类型正确，右边界多出2个字符。',
     'attribution':'预测采用了较长称谓，未对齐本数据集的边界粒度。这是可直接核对的边界差异，不能据此推断模型内部机制。'},
    {'id':'E02','kind':'type','title':'类型错误','method':'llm_0','sample_id':'train:4449',
     'gold_focus':[16,22,'organization'],'predicted_focus':[16,22,'government'],
     'finding':'“联合国安理会”的位置完全正确，但organization被预测成government。',
     'attribution':'模型输出混淆了本项目“组织机构”和“政府机关”的类别边界。按数据集标签应为organization；这不意味着所有标注体系都采用相同分类。'},
    {'id':'E03','kind':'boundary_and_type','title':'边界与类型同时错误','method':'crf','sample_id':'train:4778',
     'gold_focus':[10,33,'scene'],'predicted_focus':[1,33,'organization'],
     'finding':'金标把中文名[1,10)和英文名[10,33)分别标为scene；预测合并为一个[1,33)的organization。',
     'attribution':'同时发生相邻实体合并和scene/organization混淆。中英文连续、缺少分隔可能增加切分困难，但未做消融，不能把这个解释当作已验证机制。'},
    {'id':'E04','kind':'missed','title':'漏抽','method':'dictionary','sample_id':'train:10124',
     'gold_focus':[28,32,'scene'],'predicted_focus':None,
     'finding':'“巧克力山”在金标中是scene，词典模型整句没有输出实体。',
     'attribution':'有效Train实体表中“巧克力山”出现次数为0。仅使用训练实体精确匹配的词典无法识别这一未收录表面词；这是有训练词表证据的覆盖不足。'},
    {'id':'E05','kind':'spurious','title':'误抽','method':'dictionary','sample_id':'train:2773',
     'gold_focus':None,'predicted_focus':[7,11,'name'],
     'finding':'原文“2009年4月6日”中的“2009”被抽成人名；金标没有此实体。',
     'attribution':'有效Train中train:3398把另一上下文的“2009”标为name。词典记住表面词却不使用上下文，因而把日期中的同形字符串也当成人名；不能将此直接归为训练标注错误。'},
    {'id':'E06','kind':'invalid','title':'越界跨度','method':'llm_4','sample_id':'train:8699',
     'gold_focus':[18,21,'name'],'predicted_focus':[19,22,'name'],
     'finding':'句子只有21个字符，预测“翟宝辉”为[19,22)，end=22超过文本长度；正确跨度为[18,21)。',
     'attribution':'识别文字和类型与金标相同，但位置偏移导致越界；同句前两个实体也有位置偏移。属于结构化位置输出错误，不能静默截断或修复后计分。'},
    {'id':'E07','kind':'overlap','title':'重叠输出','method':'llm_0','sample_id':'train:3543',
     'gold_focus':[7,10,'position'],'predicted_focus':[[6,10,'name'],[7,10,'position']],
     'finding':'同时输出“朱副主席”[6,10)/name和“副主席”[7,10)/position，后者包含在前者内；金标仅有后者。',
     'attribution':'模型把带姓氏的职务称呼当作完整人名，又单独抽取内部职务，违反提示要求的非重叠输出。V3保留两者评分，不能删除多余预测来抬高结果。'},
    {'id':'E08','kind':'text_mismatch','title':'实体文字与位置不一致','method':'llm_8','sample_id':'train:3346',
     'gold_focus':[29,33,'position'],'predicted_focus':[30,33,'position'],
     'finding':'模型的text字段写“项目经理”，但[30,33)实际切出“目经理”；金标位置为[29,33)。',
     'attribution':'实体文字正确、起点偏移1，text字段与跨度未保持一致。增加text字段使该问题可检测，但不能保证模型自动避免；主指标仍使用原预测位置。'},
]

def triple(e):return [e['start'],e['end'],e['type']]

def main():
    gold_path=ROOT/'data/processed/cluener2020_project/test.jsonl'
    train_path=ROOT/'data/processed/cluener2020_project/train.jsonl'
    gold={r['id']:r for r in read_jsonl(gold_path)}
    train=[r for r in read_jsonl(train_path) if r['bio_compatible']]
    train_words=Counter(r['text'][e['start']:e['end']] for r in train for e in r['entities'])
    number_sources=[{'sample_id':r['id'],'text':r['text'],'entity':e} for r in train for e in r['entities']
                    if r['text'][e['start']:e['end']]=='2009']
    assert train_words['巧克力山']==0
    assert any(r['sample_id']=='train:3398' and r['entity']['type']=='name' for r in number_sources)
    paths={gold_path,train_path};items=[];cache={}
    protected={}
    for path in (ROOT/'reports/review/current_v3_20261024').glob('*.csv'):protected[path.relative_to(ROOT).as_posix()]=file_hash(path)
    for case in CASES:
        name=case['method']
        folder=ROOT/'runs/final'/name if name in ['dictionary','hmm','crf'] else ROOT/f'runs/llm_text_test_v1/shot_{name.split("_")[1]}'
        pred_path=folder/'predictions.jsonl';error_path=ROOT/f'runs/current_v3_analysis/{name}/errors.jsonl'
        paths.update([pred_path,error_path])
        if name not in cache:
            cache[name]=({r['id']:r for r in read_jsonl(pred_path)},{r['id']:r for r in read_jsonl(error_path)})
        predictions,errors=cache[name]
        row=gold[case['sample_id']];pred=predictions[row['id']];error=errors[row['id']]
        expected_gold=case['gold_focus'];expected_pred=case['predicted_focus']
        if expected_gold is not None:assert expected_gold in [triple(e) for e in row['entities']]
        if case['kind'] in ['boundary','type','boundary_and_type','missed','spurious']:
            assert any(e['kind']==case['kind'] and e['gold']==expected_gold and e['predicted']==expected_pred for e in error['errors'])
        elif case['kind']=='overlap':
            assert all(e in error['normalization']['overlapping_entities'] for e in expected_pred)
            assert max(e[0] for e in expected_pred)<min(e[1] for e in expected_pred)
        else:assert expected_pred in [triple(e) for e in pred['entities']]
        if case['kind']=='invalid':assert expected_pred[1]>len(row['text'])
        if case['kind']=='text_mismatch':
            entity=next(e for e in pred['entities'] if triple(e)==expected_pred)
            assert entity['text']=='项目经理' and row['text'][expected_pred[0]:expected_pred[1]]=='目经理'
        items.append({**case,'text':row['text'],'text_length':len(row['text']),'all_gold_entities':row['entities'],
                      'raw_prediction':pred,'sentence_metrics':{'token':error['token'],'entity':error['entity']},
                      'verified_against_saved_files':True,'prediction_source':pred_path.relative_to(ROOT).as_posix()})
    assert len(items)==len({r['kind'] for r in items})==len({r['sample_id'] for r in items})==8
    hashes={p.relative_to(ROOT).as_posix():file_hash(p) for p in paths}
    record={'recorded_date':'2026-10-27','timezone':'Asia/Shanghai','author':'AI-assisted analysis (Codex)',
        'status':'eight selected examples; not two-member independent review',
        'scope':'one example for each of five primary extraction errors and three observed output diagnostics',
        'selection':'purposeful representative-case selection after inspecting saved Test errors; not prevalence sampling',
        'user_adjustment':'User replaced the optional 40/20 independent-review workload with a concise AI-assisted per-category analysis.',
        'items':items,'training_evidence':{'巧克力山_entity_occurrences':0,'2009_entity_occurrences':train_words['2009'],'2009_sources':number_sources},
        'source_sha256':hashes,'source_manifest_sha256':object_hash(hashes),'new_api_calls':0,
        'independent_review_csv_hashes':protected,'human_answers_modified':False}
    save_json(ROOT/'reports/tables/concise_error_cases.json',record)
    lines=['---','recorded_date: 2026-10-27','timezone: Asia/Shanghai','analysis_author: AI-assisted (Codex)',
        'sample_count: 8','human_independent_review: false','new_api_calls: 0','---','',
        '# 精简错误归因：每类一例','',
        '按用户要求，将先前自定的40条/20条独立复核工作量缩减为AI辅助案例分析。共8个不同句子：五类抽取错误各一例，三类结构输出问题各一例。案例均来自当前V3结果和保存预测，没有重新调用模型或修改原预测。',
        '这是有原文和标注支撑的AI辅助归因，不是两名成员完成的独立人工审核。抽样用于解释错误，不用于估计全体错误比例；有些案例同时存在其他错误，此处只分析指定焦点。',
        '', '| 编号 | 类别 | 方法 | 样本ID | 核心差异 |','|---|---|---|---|---|']
    for c in items:lines.append(f'| {c["id"]} | {c["title"]} | {c["method"]} | {c["sample_id"]} | {c["finding"]} |')
    def fmt(span,text):
        if span is None:return '无对应实体'
        if span and isinstance(span[0],list):return '；'.join(fmt(s,text) for s in span)
        a,b,t=span
        piece=text[a:b] if 0<=a<b<=len(text) else text[max(0,a):min(len(text),b)]+'（该跨度越界，仅展示可切出的部分）'
        return f'`{piece}`，`[{a},{b})`，`{t}`'
    for c in items:
        lines+=['',f'## {c["id"]} {c["title"]}','',f'**原文**：{c["text"]}','',
            f'**模型/样本**：`{c["method"]}` / `{c["sample_id"]}`。',
            f'**金标焦点**：{fmt(c["gold_focus"],c["text"])}。',
            f'**预测焦点**：{fmt(c["predicted_focus"],c["text"])}。','',
            '**核对结果**：'+c['finding'],'', '**归因**：'+c['attribution']]
        if c['kind']=='spurious':
            ev=next(r for r in number_sources if r['sample_id']=='train:3398')
            lines+=['',f'训练证据 `{ev["sample_id"]}`：{ev["text"]} 其中“2009”被标为name。Test中则紧接“年4月6日”，因此词典的跨上下文误匹配有直接证据。']
        if c['kind']=='invalid':lines+=['','模型同时输出的文字是“翟宝辉”，不是模型把“宝辉”识别成另一人；问题在位置字段。V3将这一无效项单独记录，并通过entity_with_invalid_fp附加FP。']
    lines+=['','## 与两级评估的关系','',
        '边界或类型不完全匹配会令该实体贡献一个FP和一个FN；字符级仍可能有部分正确BIO项。漏抽贡献FN，误抽贡献FP。复合错误和一对一配对规则避免重复归因。',
        '重叠输出保留参与V3评分；越界项无法展开字符BIO，排除合法跨度主指标并另报附加FP指标；text不一致只诊断，不按模型文字重新定位。错误类型不是彼此完全独立，八个案例不应相加充当总体错误率。',
        '', '## 对应课程要求与范围','',
        '本节覆盖超过三类错误归因和至少三个真实失败案例，服务于课程“失败讲清楚、结论有数据支撑”的要求。Token/实体差异可以支撑评估口径原理点；其他原理点仍需结合已有词表、HMM约束和参数实验。',
        '40例、20例独立判断、至少6个案例是原先自定的加严计划，不是用户贴出的教师硬性数量要求。现按用户指示采用8例精简分析，不再把完成40/20表格作为当前额外前置任务。',
        '原独立审核CSV保持原样，不填写虚构成员姓名、审核时间或“已人工确认”。成员贡献声明仍应如实记录实际参与；本节作者标为AI辅助。',
        '重复实体与API/格式失败在本轮未观察到，不编造这两类失败案例；金标语义并不天然无误，但本次所选焦点按已保存金标与明确结构规则核对，不据单例更改数据集。',
        '', '## 证据与复现','',
        '完整金标、原始预测、具体跨度、训练词表证据及文件哈希见[tables/concise_error_cases.json](tables/concise_error_cases.json)。运行 `python scripts/summarize_error_examples.py` 可离线重建并重新核对这8例。原始文本和API记录保留本地。']
    (ROOT/'reports/concise_error_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert all(file_hash(ROOT/path)==h for path,h in hashes.items())
    assert all(file_hash(ROOT/path)==h for path,h in protected.items())
    print('VERIFIED 8 categories / 8 distinct samples; original predictions and independent forms unchanged; API calls=0')

if __name__=='__main__':main()
