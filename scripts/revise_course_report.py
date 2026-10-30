"""Targeted review revision: retain frozen evaluation, clarify validity, move detail to appendix."""
import copy
import csv
import io
import json
from ner_experiment.common import ROOT, load_json, save_json
from report_implementation_evidence import collect


def H(text,level=2,**kw):return {'type':'heading','level':level,'text':text,**kw}
def P(text,**kw):return {'type':'paragraph','text':text,**kw}
def E(text):return {'type':'equation','text':text}
def C(text):return {'type':'code','text':text}
def T(title,headers,rows,source,widths=None):
    return {'type':'table','title':title,'headers':headers,'rows':rows,'source':source,'widths':widths}


def revise(blocks):
    original=copy.deepcopy(blocks)
    starts=[i for i,b in enumerate(blocks) if b['type']=='heading' and b.get('level')==1]
    sections=[copy.deepcopy(blocks[a:b]) for a,b in zip(starts,starts[1:]+[len(blocks)])]
    front=copy.deepcopy(blocks[:starts[0]])
    tail=next(i for i,b in enumerate(sections[7]) if b.get('text')=='参考文献与材料来源')
    final_sections=sections[7][:tail]
    references=[b for b in sections[7][tail:] if b['type']=='reference']
    def old_table(prefix):return copy.deepcopy(next(b for b in original if b['type']=='table' and b['title'].startswith(prefix+' ')))
    def read_csv(name):
        with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    base='reports/tables/integrated_v3/'
    main=read_csv(base+'main_results.csv');by={r['method']:r for r in main}
    params=read_csv(base+'baseline_dev_parameters.csv')
    costs=read_csv(base+'cost_latency.csv')
    per_type=read_csv(base+'entity_per_type.csv')
    pct=lambda x:f'{float(x)*100:.2f}'
    names={r['method']:r['display'] for r in main}
    evidence=collect()
    sources=['scripts/revise_course_report.py','scripts/report_implementation_evidence.py',
             'reports/tables/course_report_revision/implementation_evidence.json']
    # Use the already-saved V3 auxiliary counts; never modify predictions or cached metric files.
    sensitivity=[]
    for r in main:
        name=r['method'];path=f'runs/integrated_v3_analysis/{name}/metrics.json';sources.append(path)
        m=load_json(ROOT/path);a=m['entity'];b=m['entity_with_invalid_fp']
        sensitivity.append({'method':name,'valid_tp':a['tp'],'valid_fp':a['fp'],'valid_fn':a['fn'],
            'valid_precision':a['precision'],'valid_recall':a['recall'],'valid_f1':a['f1'],
            'invalid_penalized_tp':b['tp'],'invalid_penalized_fp':b['fp'],'invalid_penalized_fn':b['fn'],
            'invalid_penalized_precision':b['precision'],'invalid_penalized_recall':b['recall'],
            'invalid_penalized_f1':b['f1'],'extra_invalid_fp':b['fp']-a['fp'],
            'f1_change_pp':(b['f1']-a['f1'])*100,'failure_rate':m['failure_rate']})
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(sensitivity[0]));writer.writeheader();writer.writerows(sensitivity)
    path=ROOT/'reports/tables/course_report_revision/invalid_output_sensitivity.csv'
    path.write_text(stream.getvalue(),encoding='utf-8-sig',newline='')
    sources.append(path.relative_to(ROOT).as_posix())
    penalty={r['method']:r for r in sensitivity}

    # 1. Add a concrete audience and extraction-level success condition.
    for b in sections[0]:
        if b['type']=='paragraph' and '面向需要将中文短文本' in b['text']:
            b['text']=b['text'].replace('面向需要将中文短文本转成结构化字段的使用者','面向中文文档处理开发者、信息管理系统开发者和文本分析研究人员')
        if b['type']=='paragraph' and b['text'].startswith('这些判据'):
            b['text']='一次成功抽取要求实体起点、终点和类别同时符合当前金标；只覆盖部分字符仍不能作为完整字段使用。工程验收另要求比较公平、出处可查和离线可重算，不补造实验前并不存在的F1门槛。'

    # 2. Add direct NER literature; clarify that omitting BERT was a scope decision.
    literature=[
      P('NER方法由手工特征与序列标注发展到预训练表示和生成式抽取。BERT在上下文编码后接任务输出层，NER可据此对token表示做标签分类[6]；它是本项目尚未覆盖的合理预训练基线。未安装PyTorch/CUDA并不意味着BERT不能在CPU上微调。本轮选择“传统序列标注与API提示抽取”路线，主要为控制训练、调参与验证范围，而非判定BERT不适合。'),
      P('GPT-NER将序列标注改写为带实体标记的文本生成，并研究示例检索和自验证[7]。这说明LLM做NER不只是一般文本生成，还需设计输出表示和例句。本项目采用字符跨度JSON，不采用其实体标记输出、相似例句检索或额外自验证调用，因而不是GPT-NER的复现，也不直接对比论文中的数值。'),
      P('Lee等针对低资源NER研究了示例内容、上下文、选择和模板，发现合适的demonstration策略会影响效果[8]。该研究的训练和任务设置与本项目不同，但提示我们不能把“示例数”与“示例质量”混为一个因素。本轮仅用一套固定例句池，结论限定为这组前缀，未验证换例句后的稳定性。')]
    new2=[]
    for b in sections[1]:
        if b['type']=='heading' and b.get('text')=='2.2 为什么采用提示抽取':new2+=literature
        if b['type']=='heading' and b.get('text')=='2.3 为什么同时报告两级指标':continue
        if b['type']=='paragraph' and b['text'].startswith('完整实体匹配关注'):continue
        if b['type']=='table' and b['title'].startswith('表2 '):
            for row in b['rows']:
                if row[0]=='SFT与LoRA':row[2]='LoRA仍需训练参数[3]；本轮按课程路线控制规模，不以无CUDA为不可行证明'
        new2.append(b)
    sections[1]=new2
    references += [
      {'type':'reference','text':'[6] Jacob Devlin等. BERT Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.','url':'https://aclanthology.org/N19-1423/'},
      {'type':'reference','text':'[7] Shuhe Wang等. GPT-NER Named Entity Recognition via Large Language Models. 2023.','url':'https://arxiv.org/abs/2304.10428'},
      {'type':'reference','text':'[8] Dong-Ho Lee等. Good Examples Make A Faster Learner Simple Demonstration-based Learning for Low-resource NER. ACL 2022.','url':'https://aclanthology.org/2022.acl-long.192/'}]

    # 3. Show the actual interface instead of adopting the review's illustrative field names.
    schema={'id':'train:50','status':'ok','entities':[{'start':0,'end':3,'type':'game','text':'街霸》'}]}
    sections[2] += [H('3.3 统一预测接口'),P('下面精简自GLM 8-shot实际记录。代码字段为id和status="ok"，不是sample_id或success；方法身份由结果目录/运行元数据提供。id沿用原始来源编号，金标不进入此预测接口。'),C(json.dumps(schema,ensure_ascii=False,indent=2)),P('传统模型BIO先转跨度；LLM也输出同样的entities列表。统一评估器依据样本ID对齐，显式api_error/format_error计空预测，缺失或重复ID停止评分。原始响应、usage和耗时仍保留在完整记录中。')]

    # 4. Add executable evidence without adding model-training experiments.
    check=evidence['viterbi']['checks'][1]
    implementation=[H('4.3 算法实现与数值验证'),
      P('HMM的动态规划状态δ_t(j)表示读到第t个字符且当前标签为j时的最佳对数路径分数；ψ_t(j)保存达到该分数的前一标签。初始化、递推、终止与回溯分别为：'),
      E('δ_0(j)=log π_j+log b_j(x_0)'),
      E('δ_t(j)=log b_j(x_t)+max_i［δ_(t−1)(i)+log a_ij］'),
      E('ψ_t(j)=argmax_i［δ_(t−1)(i)+log a_ij］；从argmax_j δ_(T−1)(j)逆向回溯'),
      P('BIO约束直接把非法起点与转移的对数分数设为−∞，再执行相同递推，不在解码后删除标签。人工构造的三状态、三位置校验例（不是CLUENER实测分数）使用π=(0.6,0.3,0.1)，状态顺序为O、B-name、I-name；转移与发射矩阵、δ和回溯指针完整保存于implementation_evidence.json。'),
      P(f"实际调用viterbi并枚举全部路径：普通解码枚举27条，BIO合法解码枚举{check['enumerated_paths']}条，均得到O→B-name→I-name。该路径概率=0.6×0.6×0.3×0.5×0.6×0.7=0.02268，对数分数={check['log_score']:.6f}；动态规划与穷举一致。另有tests/test_baselines.py覆盖长度1、2、4及随机小矩阵，不把这组人工数值当作模型效果。"),
      P('CRF特征的真实例子取Test样本train:50“街霸》电影。”的第1号字符“霸”，直接调用features函数得到下列字典。它只读取字符及位置，不读取金标；相关测试通过改写金标而保持特征不变来验证这一点。'),
      C(json.dumps(evidence['actual_crf_feature']['features'],ensure_ascii=False,separators=(',',':'))),
      P('“另有1个词典基线”表示独立模型，不是CRF增加词典特征。当前没有实施CRF+Dictionary Features独立消融，无法评价该特征的增益，保留为未完成扩展。'),
      P('人工构造重叠评估校验：金标为“北京大学”[0,4) organization，预测额外含[0,2) name。两实体均保留：Token TP/FP/FN=4/2/0、F1=80%；实体为1/1/0、F1=66.67%。若额外项类型为unknown_type，则不能造Token标签；合法实体F1为100%，但无效项计FP后为66.67%。这说明只展示合法跨度分数会漏掉结构化输出错误。数值均由真实评估函数计算，原始输入与输出已保存。')]
    new4=[]
    for b in sections[3]:
        if b['type']=='paragraph':b['text']=b['text'].replace('共3组，加词典1组','共3组，另有1个独立词典基线')
        if b['type']=='heading' and b.get('text')=='4.3 LLM提示与接口实现':
            new4+=implementation;b['text']='4.4 LLM提示与接口实现'
        if b['type']=='heading' and b.get('text')=='4.4 踩坑与修正记录':b['text']='4.5 踩坑与修正记录'
        new4.append(b)
        if b['type']=='paragraph' and b['text'].startswith('Prompt列出'):
            new4 += [P('以DeepSeek实现为例：cache_key=SHA256(JSON({endpoint,payload}))，JSON按键排序且保留Unicode；payload含模型、生成参数及完整messages。命中缓存先核验内容哈希和请求一致性；未命中则持久化reserved预算，再请求。响应成功记录usage并解析为ok或format_error；网络/429/5xx最多重试两次，其他HTTP错误不盲重试，格式错误不触发LLM修复。最终缓存也保留失败；原子临时文件替换避免半条记录。GLM/MiMo另有各自限速与恢复协议，不能凭一个简化状态图抹掉历史差异。')]
    sections[3]=new4

    # 5. Compact the main evaluation; keep interpretation-changing caveats before rankings.
    main_table=old_table('表7')
    main_table['title']='表7 完整Test双层指标与无效输出惩罚 单位为%'
    main_table['headers']=['方法','Token P','Token R','Token F1','合法实体P','合法实体R','合法实体F1','无效计FP F1']
    main_table['rows']=[[r['display'],*[pct(r[f'{level}_{k}']) for level in ['token','entity'] for k in ['precision','recall','f1']],pct(penalty[r['method']]['invalid_penalized_f1'])] for r in main]
    main_table['widths']=[2260]+[1054]*7
    main_table['source']=base+'main_results.csv；course_report_revision/invalid_output_sensitivity.csv；最后一列来自V3已保存的entity_with_invalid_fp'
    class_table=old_table('表9');class_table['title']='表9 同为8-shot的逐类实体F1解释性摘录 单位为%'
    class_table['rows']=[r for r in class_table['rows'] if r[0] in ['name','position','government','organization']]
    fig=copy.deepcopy(next(b for b in sections[4] if b['type']=='figure'));fig['width']=540
    fig['caption']+=' 本图沿用冻结V3合法跨度F1，未把原区间改称无效项惩罚口径。'
    summary=[]
    for r in params:
        cfg=json.loads(r['config'])
        if r['method']=='hmm' and cfg['alpha']==.01:
            summary.append([f"HMM α=.01, f≥{cfg['min_freq']}",r['vocabulary_size'],f"{100*float(r['dev_unknown_char_rate']):.4f}",pct(r['v3_dev_token_f1']),pct(r['v3_dev_entity_f1'])])
        if r['method']=='crf':summary.append([f"CRF c1={cfg['c1']}, c2={cfg['c2']}",'—','—',pct(r['v3_dev_token_f1']),pct(r['v3_dev_entity_f1'])])
    d4=penalty['deepseek_4']
    sections[4]=[H('⑤ 评估',1,pageBreakBefore=True),
      P('证据等级：本实验是包含历史协议和Prompt修订的探索性比较。Test未用于传统模型参数训练或Train示例选取，但部分设计决策发生在观察历史Test结果后；新版LLM未重新完成完整Dev选参。以下排名与区间描述当前固定条件，不作首次独立盲测的泛化声明。Bootstrap不能消除这类历史反馈。'),
      H('5.1 数据 指标与失败处理'),
      P('主Test为从官方Train划出的1,209句、2,648个实体；有效Train 9,538句、官方Dev 1,343句。相同文本不跨集合，seed=718853259，完整类别分布及哈希见附录A1和manifest。传统参数由Dev选出，三家LLM均报告0/2/4/8-shot，不按Test择优删档。'),
      P('V3严格实体比较(样本ID,start,end,type)；Token比较(样本ID,字符位置,完整BIO标签)集合[5]，保留合法重叠并去重。O不贡献TP，错误类别或B/I计FP+FN；先汇总计数再算micro，零分母记0。缺/重/多余ID停止评分，显式API/格式失败计空预测并报失败率。'),
      E('P=TP/(TP+FP)；R=TP/(TP+FN)；F1=2TP/(2TP+FP+FN)'),
      P('主表并列展示V3合法跨度实体P/R/F1与“无效项计FP的实体F1”。后一口径给可枚举的越界、未知类型等额外FP，供端到端输出质量比较；不能映射的条目不强造Token标签。无法整体解析的响应仍为空预测，失败率另报，因此该F1也不能代替全部可靠性指标。这是报告修订后突出V3已有辅助口径，未改动冻结评估器或历史结果。'),
      H('5.2 基线与跨API系统比较'),
      P('各API的thinking、输出上限和调度不同；跨模型结果反映这些配置下的系统表现，不是统一推理条件下的架构能力排名。GLM 4-shot保留117条网络失败，运行可用性也进入其成绩。'),main_table,
      P(f"只有DeepSeek 4-shot含1个无效实体：合法实体TP/FP/FN={d4['valid_tp']}/{d4['valid_fp']}/{d4['valid_fn']}，无效计FP后FP={d4['invalid_penalized_fp']}，F1从{pct(d4['valid_f1'])}%变为{pct(d4['invalid_penalized_f1'])}%，差{d4['f1_change_pp']:.4f}个百分点；其余14组两列相同。影响虽小，但不能让非法输出无代价消失。"),
      P('CRF实体F1为68.66%，高于词典和HMM；三个LLM零样本实体点估计均低于CRF，部分少样本配置超过CRF。DeepSeek 8-shot与MiMo 4-shot分别为71.88%和71.28%；不把最高点估计称为Dev选出的最优配置。'),
      H('5.3 Shot数量与类别差异'),fig,
      P('DeepSeek的4-shot低于2-shot，MiMo的8-shot低于4-shot；MiMo 8−4差值为−0.59个百分点，95%配对区间[−1.59,0.38]含0，不证明稳定退化。增加Shot同时改变内容、类别覆盖和长度；一套示例池不足以证明数量的纯因果效应。'),class_table,
      P('name上LLM高于CRF，position与government却低于CRF；总体排名不能代表每类字段都受益。此四类为解释性摘录，所有十类和各档P/R/F1完整保留于附录及150行源表，不逐类拼接“最好模型”。'),
      H('5.4 参数实验与三个课程原理'),
      T('表10 Dev参数对照与词表证据',['配置','词表','UNK率%','Token F1%','实体F1%'],summary,base+'baseline_dev_parameters.csv',[3300,1300,1738,1650,1650]),
      P('原理一（第2讲）：固定HMM α=.01，最低词频1→2使词表3593→3019、UNK率0.1811%→0.3741%，实体F1微升48.29%→48.45%，阈值3为48.42%。词表压缩与稀有信息损失存在取舍，并非越小越好；这里没比较分词器。'),
      P('原理二（第9讲）：9组HMM复用相同概率，比较有/无BIO约束。α=.01/.1时普通解码非法率为0；α=1的3组约0.0020%，但9组实体F1最大变化仍为0。合法路径约束能排除非法序列，却未必提升完整实体得分；数值校验见④，完整参数见附录。'),
      P('原理三（第14—15讲）：HMM的Token F1高于词典，实体F1却低5.67个百分点，排名随粒度改变；⑥的真实边界例Token F1=87.50%、实体F1=0。完整字段需要严格实体指标，字符部分命中、O准确率或只看合法输出都不能替代它。CRF的Dev实体F1在(c1,c2)=(.1,.1)最高70.61%，选参标准未在Test后切换。'),
      H('5.5 区间与网络诊断的解释边界'),
      P('2,000次句子级配对Bootstrap使用同一重采样索引，先汇总TP/FP/FN再求micro F1，seed=718853259。DeepSeek 8−CRF差3.22个百分点，95%区间[1.04,5.38]；MiMo 4−CRF差2.62，[0.46,4.81]。这两组无无效输出，合法/惩罚口径相同。42组比较未校正多重比较，且不含重训、API重复或换例句的波动；区间不含0也不能恢复独立盲测性质。'),
      P('为诊断网络影响，所有15组共同排除网络失败ID并集121句，剩1,088句；GLM 4-shot实体F1由全量64.50%变为子集68.06%。这是事后非随机子集，保留格式/拒绝失败，只作条件诊断，不替代完整Test或代表“修网后的全量真分数”。完整区间与子集表见附录。')]

    # 6. Retain real cases; explicitly qualify ontology and current human-review status.
    sections[5]=[b for b in sections[5] if not (b['type']=='table' and b['title'].startswith('表14 '))]
    for i,b in enumerate(sections[5]):
        if b['type']=='heading' and b.get('text')=='6.2 三个真实失败案例':
            sections[5].insert(i+1,P('以下“错误”均指相对于CLUENER2020当前金标与本体不一致，不表示其他标签体系下也绝对错误。例如“陆军”在本数据标为government，其他体系可能归为军事组织；因此保留原文、标签和跨度供成员核验，不擅自改金标。'));break

    # 7. Keep totals, privacy, AI disclosure and reproducibility; move per-shot billing to appendix.
    aggregate=[]
    for family in ['deepseek','glm','mimo']:
        rr=[r for r in costs if r['family']==family]
        aggregate.append([family,sum(int(r['attempts_including_retries']) for r in rr),
            f"{sum(float(r['known_usage_estimate_cny']) for r in rr):.4f}",
            f"{min(float(r['latency_median_seconds']) for r in rr):.3f}—{max(float(r['latency_median_seconds']) for r in rr):.3f}"])
    sections[6]=[H('⑦ 工程与伦理',1),
      P('三家正式Test各覆盖4,836个样本配置。usage按调用时记录的输入缓存命中/未命中及输出单价核算；逐档token、单价和P95见附录，不按当前价格重算历史请求。'),
      T('表15 正式Test费用与观测延迟',['模型','尝试数','已知usage估算元','各档延迟中位数范围秒'],aggregate,base+'cost_latency.csv；范围不是合并后的中位数',[2100,1600,2400,3538]),
      P('正式Test三家已知估算合计18.9935元；全项目含旧Prompt、Dev与试跑为25.5642元，另有26.5044元未知预留，保守账本52.0686元，低于100元上限。预留不是实扣，实际支出须查平台账单。延迟含限速、排队、重试，GLM混合串并行，不作纯服务端速度排名；传统模型无API费但CPU与人工成本非零。'),
      P('密钥只通过环境变量/本地.env提供，不进入Git或私有结果包；打包校验逐文件哈希并检查凭据。未引入私有金融、病历或用户资料；公开语料中的人名不意味着任意再分发权。本地README未明确数据许可，继续保留未解决状态，完整语料与响应仅留本地，不公开上传。'),
      P('复现条件为Windows/Python 3.13环境、精确依赖、固定种子与有权使用的本地证据包；克隆源码不自动获得数据/预测。下面入口已实际在本机重评15组、核对源表并通过136项测试，不读密钥或调用API；日志见runs/integrated_v3_analysis/reproduction_current.log。缺证据包时明确停止。'),
      C('.\\.venv\\Scripts\\python.exe scripts/reproduce_main_results.py'),
      P('本机核验不等同于另一环境独立复现，重新调用API也不保证字节一致。源表与预测哈希、数字出处表以及成本账本保留；补充本轮实现小例属于明确标注的校验，不冒充新的模型实验。'),
      P('AI使用声明：ChatGPT/Codex辅助方案讨论、代码与API工程、实验执行、测试、统计、错误归因、图表和报告。大量代码/操作由AI辅助产生或执行，不写成成员全部手工完成。案例归因不是双人独立人审，40/20扩展未完成；姓名、贡献与核验由成员据实确认，详见⑧和AI声明。')]

    # Appendix preserves every displaced table and detailed accounting.
    appendix=[H('附录A 数据 参数与统计证据',1),P('本附录保存从正文移出的明细；所有条目来自同一冻结预测与源表，不新增API实验。正文第⑧节按用户指示保持原状。')]
    moved=[('表6','表A1 固定划分类别分布'),('表9','表A2 全十类同8-shot与CRF参照'),
           ('表12','表A4 关键句子配对区间'),('表13','表A5 共同网络子集诊断'),
           ('表14','表A6 错误归因计数'),('表15','表A7 调用时记录单价'),
           ('表16','表A8 四档费用 预留与延迟'),('表17','表A9 计费token汇总'),('表A1','表A10 主张与来源')]
    for prefix,title in moved:
        item=old_table(prefix);item['title']=title;appendix.append(item)
        if prefix=='表9':
            rows=[]
            for r in params:
                cfg=json.loads(r['config'])
                rows.append([r['name'],r['config'],r['vocabulary_size'] or '—',
                  f"{100*float(r['dev_unknown_char_rate']):.4f}" if r['dev_unknown_char_rate'] else '—',
                  pct(r['v3_dev_token_f1']),pct(r['v3_dev_entity_f1']),
                  pct(r['unconstrained_entity_f1']) if r['unconstrained_entity_f1'] else '—'])
            appendix.append(T('表A3 全部13组Dev参数与HMM无约束对照',['运行','配置','词表','UNK率%','Token F1%','实体F1%','无约束实体F1%'],rows,base+'baseline_dev_parameters.csv',[1500,2500,950,1100,1150,1150,1288]))
    appendix += [H('附录B 新增实现核验与指标敏感性',1),
      P('下列Viterbi与重叠例均为人工构造实现校验，实际调用项目函数并断言结果；不是新数据集分数。完整矩阵、δ、回溯指针和构造输入保存在reports/tables/course_report_revision/implementation_evidence.json。真实CRF特征另注明样本ID。'),
      T('表B1 三状态Viterbi实际校验',['模式','枚举路径数','最优序列','对数分数','与动态规划一致'],
        [['有BIO约束' if c['constrained'] else '普通',c['enumerated_paths'],' → '.join(c['path']),f"{c['log_score']:.6f}",'是'] for c in evidence['viterbi']['checks']],
        'scripts/report_implementation_evidence.py；tests/test_baselines.py',[1700,1500,3238,1700,1500]),
      T('表B2 十五组无效输出计FP敏感性',['方法','合法实体F1%','无效计FP F1%','额外FP','变化百分点'],
        [[names[r['method']],pct(r['valid_f1']),pct(r['invalid_penalized_f1']),r['extra_invalid_fp'],f"{r['f1_change_pp']:.4f}"] for r in sensitivity],
        'reports/tables/course_report_revision/invalid_output_sensitivity.csv',[3300,1700,1800,1200,1638]),
      P('全部42组Token/实体配对区间见reports/tables/integrated_v3/bootstrap_comparisons.csv；这里列的是合法跨度口径。不得把未重算的含无效FP区间套到DeepSeek 4-shot上。正文列出的DeepSeek 8、MiMo 4、CRF无无效实体，两口径的逐句计数相同，因而对应区间亦相同。')]
    # Keep the final methodological note with its table as a compact source caption.
    appendix[-2]['source'] += '；' + appendix.pop()['text']
    appendix[-1]['compact'] = True
    # Table numbers are kept stable from the original where possible; appendix has its own numbering.
    revised=front+sum(sections[:7],[])+final_sections+[H('参考文献与材料来源')]+references
    save_json(ROOT/'reports/tables/course_report_revision/revision_record.json',{
        'new_api_calls':0,'frozen_metric_protocol':'v3 unchanged',
        'new_primary_table_display':'V3 legal-span P/R/F1 plus pre-existing entity_with_invalid_fp F1',
        'section_eight_unchanged':final_sections==sections[7][:tail],
        'implementation_checks_labeled_synthetic':True,
        'invalid_entity_count_total':sum(r['extra_invalid_fp'] for r in sensitivity),
        'new_literature':['https://aclanthology.org/N19-1423/','https://arxiv.org/abs/2304.10428','https://aclanthology.org/2022.acl-long.192/'],
        'appendix_tables':len([b for b in appendix if b['type']=='table'])})
    return revised,appendix,sources
