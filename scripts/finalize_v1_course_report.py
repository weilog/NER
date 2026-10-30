"""Present the original evaluator as the sole formal report protocol."""
import copy
import csv
from ner_experiment.common import ROOT, load_json, save_json
from report_v1_implementation_evidence import collect

BASE='reports/tables/integrated_v1/'

def finalize(blocks, appendix):
    evidence=collect()
    with (ROOT/BASE/'main_results.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    pct=lambda x:f'{100*float(x):.2f}'
    replacements={
      '人工构造重叠评估校验：':'人工构造重叠评估校验：金标为“北京大学”[0,4) organization，预测另含[0,2) name。Token按固定优先级保留较长跨度，TP/FP/FN=4/0/0、F1=100%；实体保留两个输出，计1/1/0、F1=66.67%。若额外项为unknown_type，不能造字符标签，但仍增加实体FP。两级指标的差异可能同时来自评估粒度和重叠投影，不能全部归因于边界粒度。数值来自原始评估函数的人工构造校验，不是数据集分数。',
      'V3严格实体比较':'严格实体比较(样本ID,start,end,type)；Token先按起点优先、同起点较长优先、同跨度固定类别顺序去除重叠，再转为单标签BIO，对20个非O标签计算micro P/R/F1。实体评分保留全部合法去重预测。O不贡献TP，错误类别或B/I计FP+FN；汇总计数后计算micro，零分母记0。缺/重/多余ID停止评分，显式API/格式失败按空预测评分。',
      '主表并列展示V3':'主表并列报告逐Token与严格实体P/R/F1。可枚举的越界、未知类型等无效实体直接计入整体实体FP，不能映射为字符标签的项不强行投影；无法整体解析的响应按空预测评分，并单独报告失败率。Token与实体采用的重叠处理不同，报告分别说明，不将其视作同一预测集合的两种粒度。',
      '只有DeepSeek 4-shot含1个无效实体':'DeepSeek 4-shot含1个无效实体，已计入实体FP：TP/FP/FN=1764/667/884，实体F1=69.46%。其余14组无无效实体；主表不通过丢弃该项提高实体成绩。',
      'V3指标与新版Prompt':'新版Prompt经历了历史结果观察后的调整，且按用户要求跳过新版完整Dev选参，因此本报告不宣称首次独立盲测。公开数据可能进入预训练语料，零样本只表示本轮未提供任务例句，不保证模型从未接触相关数据。结果限于当前固定Test，不外推医疗、金融或其他领域。',
    }
    def convert(bs, is_appendix=False):
        out=[]
        for original in bs:
            b=copy.deepcopy(original)
            if b.get('type')=='reference' and 'EDS-NLP' in b.get('text',''):continue
            if is_appendix and b.get('title','').startswith('表B2 '):continue
            if b.get('type')=='paragraph':
                for prefix,replacement in replacements.items():
                    if b['text'].startswith(prefix):b['text']=replacement;break
                b['text']=b['text'].replace('这两组无无效输出，合法/惩罚口径相同。','')
                b['text']=b['text'].replace('实现小例属于明确标注的校验','实现小例属于明确标注的校验')
            if b.get('type')=='table':
                if b['title'].startswith('表7 '):
                    b.update(title='表7 完整Test逐Token与严格实体指标 单位为%',
                        headers=['方法','Token P','Token R','Token F1','实体P','实体R','实体F1'],
                        rows=[[r['display'],*[pct(r[f'{level}_{k}']) for level in ['token','entity'] for k in ['precision','recall','f1']]] for r in rows],
                        widths=[2300,1223,1223,1223,1223,1223,1223],source=BASE+'main_results.csv')
                for row in b['rows']:
                    if row and row[0]=='重叠在两级评分中口径不一致':
                        row[:]=['重叠输出不能直接转为单标签BIO','固定起点、长度、类别优先级用于字符投影；实体保留全部输出','两级评分分别说明，保留重叠与无效项诊断，不按Test调整规则']
            if b.get('type')=='figure':
                b['caption']=b['caption'].split(' 本图沿用')[0]
            for key in ['text','source','path','caption']:
                if key not in b or not isinstance(b[key],str):continue
                value=b[key].replace('integrated_v3','integrated_v1').replace('implementation_evidence.json','implementation_evidence_v1.json').replace('scripts/report_implementation_evidence.py','scripts/report_v1_implementation_evidence.py')
                value=value.replace('字符BIO集合式Token','字符BIO的Token').replace('字符BIO集合评估','字符BIO评估')
                value=value.replace('报告保留协议修订、API条件差异','报告说明流程迭代、API条件差异')
                value=value.replace('统一计算V3指标','统一计算逐Token与严格实体指标')
                value=value.replace('V3口径','原始评估口径').replace('V3指标','双层指标')
                value=value.replace('语料与响应仅留本地','语料与响应仅留本地')
                for a,z in [('[6]','[5]'),('[7]','[6]'),('[8]','[7]')]:value=value.replace(a,z)
                b[key]=value
            def current_paths(value):
                if isinstance(value,str):
                    return value.replace('evaluation/metrics.py','evaluation/metrics_v1.py').replace('integrate_results.py','integrate_v1_results.py').replace('docs/evaluation_v3.md','docs/evaluation_v1.md')
                if isinstance(value,list):return [current_paths(v) for v in value]
                if isinstance(value,dict):return {k:current_paths(v) for k,v in value.items()}
                return value
            out.append(current_paths(b))
        return out
    blocks=convert(blocks)
    appendix=convert(appendix,True)
    appendix.append({'type':'paragraph','text':'全部42组Token与实体配对区间均由同一原始评估器的逐句TP/FP/FN重新计算，完整结果见reports/tables/integrated_v1/bootstrap_comparisons.csv。没有模型重训或新API调用。'})
    for b in appendix:
        if b.get('text','').startswith('附录B '):b['text']='附录B 算法实现核验'
    record=load_json(ROOT/'reports/tables/course_report_revision/revision_record.json')
    record.update(formal_protocol='v1', formal_report_version_discussion=False,
                  original_predictions_unchanged=True,new_api_calls=0,
                  appendix_tables=sum(b['type']=='table' for b in appendix),
                  new_primary_table_display='token and strict entity P/R/F1; invalid entities included in overall entity FP')
    save_json(ROOT/'reports/tables/course_report_revision/revision_record.json',record)
    return blocks,appendix,['scripts/finalize_v1_course_report.py','scripts/report_v1_implementation_evidence.py',
                          'reports/tables/course_report_revision/implementation_evidence_v1.json']
