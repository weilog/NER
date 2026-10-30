"""Build the teacher's eight-section Chinese report from saved, verified evidence."""
import csv
import json
from pathlib import Path
from ner_experiment.common import ROOT, LABELS, file_hash, load_json, save_json, read_jsonl

OUT=ROOT/'reports'
SOURCES=set()


def csv_rows(relative):
    p=ROOT/relative;SOURCES.add(relative)
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def read(relative):
    SOURCES.add(relative);return load_json(ROOT/relative)


def main():
    table='reports/tables/integrated_v1/'
    rows=csv_rows(table+'main_results.csv');by={r['method']:r for r in rows}
    parameters=csv_rows(table+'baseline_dev_parameters.csv')
    per_type=csv_rows(table+'entity_per_type.csv')
    errors=csv_rows(table+'error_counts.csv')
    diagnostics=csv_rows(table+'diagnostics.csv');diag={r['method']:r for r in diagnostics}
    costs=csv_rows(table+'cost_latency.csv');costby={r['method']:r for r in costs}
    subset=csv_rows(table+'network_subset.csv')
    comparisons=csv_rows(table+'bootstrap_comparisons.csv')
    protocols=csv_rows(table+'api_protocols.csv')
    manifest=read('data/splits/cluener2020/manifest.json')
    audit=read('data/manifests/cluener2020/audit.json')
    verification=read(table+'verification.json')
    project_cost=read(table+'project_cost.json')
    cases=read('runs/integrated_v1_analysis/representative_cases.json')
    frozen=read('runs/llm_text_test_v1/frozen_protocol.json')['protocol']
    for name,digest in verification['source_hashes'].items():
        if file_hash(ROOT/name)!=digest:raise RuntimeError('Evidence changed: '+name)
    if len(rows)!=15 or any(int(r['samples'])!=1209 for r in rows):raise RuntimeError('Unexpected main table')
    pct=lambda x:f'{100*float(x):.2f}'
    names={r['method']:r['display'] for r in rows}
    blocks=[]
    def h(text,level=2):blocks.append({'type':'heading','level':level,'text':text})
    def p(text,**options):blocks.append({'type':'paragraph','text':text,**options})
    def t(title,headers,data,source='',widths=None):
        blocks.append({'type':'table','title':title,'headers':headers,'rows':data,'source':source,'widths':widths})
    def fig(name,title,source):
        blocks.append({'type':'figure','path':name,'caption':title,'source':source})
    def code(text):blocks.append({'type':'code','text':text})
    def eq(text):blocks.append({'type':'equation','text':text})

    # Core Results are specified first; the document is then assembled in the teacher's order.
    h('摘要')
    p('本项目围绕中文细粒度命名实体识别，构建数据审计、模型推断、统一评估和错误分析的小型NLP流水线。在CLUENER2020重新划分的1,209句带标签Test上，比较词典、HMM、CRF与DeepSeek、GLM、MiMo各0/2/4/8-shot，共15组方案，同时报告字符BIO集合式Token指标和严格实体指标。CRF的实体F1为68.66%，DeepSeek 8-shot为71.88%，MiMo 4-shot为71.28%。更多示例并非始终对应更高分数，且GLM 4-shot包含117条网络失败，系统故障必须与识别错误区分。HMM词表阈值、BIO约束对照和两级指标排序差异分别支撑词表选择、标签约束与评估协议三个课程原理。所有主结果均可由保存预测离线重算；报告保留协议修订、API条件差异、费用未知预留及尚未完成独立环境复现的限制。')
    p('关键词：命名实体识别；CLUENER2020；HMM；CRF；Few-shot；字符BIO集合评估；可复现性')

    h('① 问题定义与背景',1)
    h('1.1 任务与使用场景')
    p('命名实体识别（Named Entity Recognition，NER）需要从原文中找出实体的起止位置与类别。本项目面向需要将中文短文本转成结构化字段的使用者：例如先抽取人名、公司、组织和作品名，再用于文档检索、索引或信息统计。项目只验证这一基础抽取环节，没有部署金融或医疗业务系统，也不把通用短文本结果外推为专业领域效果。')
    p('输入为未经改写的Unicode字符串x，输出为实体集合E={(start,end,type)}，位置采用左闭右开区间[start,end)。类别固定为address、book、company、game、government、movie、name、organization、position、scene共10类[1]。项目评估token定义为原文中的一个字符，绝非远程大模型的内部子词或计费token。')
    h('1.2 研究问题与可检验成功判据')
    p('本项目回答三个问题：在同一测试集上，传统方法与提示抽取各有什么性能和错误特征；0/2/4/8个示例的变化是否持续改善效果；分数收益与费用、延迟及结构化输出失败之间如何权衡。比较对象是不同监督量、预训练背景和运行条件下的系统，不是控制模型容量的纯架构实验。')
    t('表1 项目验收判据',['判据','可核验方式'],[
        ['基线与LLM能够比较','15组均覆盖同一1,209句，逐Token及实体P/R/F1并列，失败样本保留'],
        ['结论有自己的实验支撑','至少一组参数对比、三个课程原理的实测证据、三类真实错误案例'],
        ['关键数字可追溯','源CSV保留TP/FP/FN，固定数据/预测ID和SHA256，报告表格由源表生成'],
        ['他人能够复核主结果','按README重建环境并取得本地证据包后，用离线入口重算15组，结果与保存指标一致']
    ],'docs/submission_checklist.md；reports/tables/integrated_v1/verification.json',[2000,7638])
    p('这些判据是课程交付与复核标准，不补造实验前并不存在的F1门槛。较高分数不是唯一成功标准：能解释失败并重复计算结果，同样是本项目的重要产出。')

    h('② 相关工作与方案选型',1)
    h('2.1 数据与传统序列方法')
    p('CLUENER2020提供十类中文细粒度实体，除人名与机构外还包含作品、职位和景点[1]。它适合检验类别混淆和边界判断，而不是只记忆少数粗粒度标签。本地压缩包与公开README的某些类别统计不同，因此本报告以实际使用文件的逐条计数和哈希为准，不抄用网页中的历史类别数。')
    p('词典体现训练实体的记忆与匹配能力；隐马尔可夫模型（Hidden Markov Model，HMM）以字符为观测、BIO标签为隐状态，通过初始、转移和发射概率进行序列解码；条件随机场（Conditional Random Field，CRF）直接建模给定文本后的标签序列分布，能够结合上下文特征[2]。三个基线的复杂度和归纳偏置不同，便于判断LLM收益是否超过简单记忆或局部序列建模。')
    h('2.2 为什么采用提示抽取')
    t('表2 替代方案与本项目选择',['方案','本项目处理','选择依据与边界'],[
        ['词典、HMM、CRF','全部实现并比较','仅用有效Train；CPU可运行；覆盖课程传统方法'],
        ['LLM零样本与少样本','三家API各0/2/4/8-shot','无本地权重更新；可检验示例与指令对结构化抽取的作用'],
        ['BERT或BiLSTM微调','当前未实施','可补充预训练小模型基线，但增加训练与调参范围；不虚构其效果'],
        ['SFT与LoRA','当前未实施','LoRA仍需训练低秩参数[3]；本轮无PyTorch/CUDA训练环境，优先保证现有评估闭环'],
        ['RAG','当前未实施','任务要求实体来自给定原文；检索外部知识会新增召回与上下文变量，非本轮研究问题']
    ],'src/ner_experiment/models/baselines.py；三家API冻结协议；参考文献[2—4]',[1800,2100,5738])
    p('少样本提示将带标签例句放进上下文，不更新模型参数；SFT和LoRA则需要通过训练改变可训练参数[3]。本项目可以分析提示作用及其边界，却不能据此判断提示优于微调。选择API主要为控制本地算力与工程范围，不能把“未做微调”包装成架构优势。')
    p('RAG适用于让生成模型利用检索到的外部信息[4]。本项目需要在原句中给出字符位置，外部资料并不能直接替代这一定位任务。因此不引入检索模块；这是范围控制，不是通过实验否定RAG。')
    h('2.3 为什么同时报告两级指标')
    p('完整实体匹配关注“字段是否整体正确”，逐字符指标能揭示“已有多少字符和类型被识别”。EDS-NLP的Token评估允许重叠实体，并比较(token,实体类型)集合[5]；本项目借鉴集合表示，进一步保留字符级B/I标签。由此得到的V3指标不是EDS-NLP原生指标，也不是存在重叠时的普通单标签分类分数。其局限与补充实体指标见第⑤节。')

    h('③ 系统设计',1)
    h('3.1 架构与数据流')
    fig('figures/course_report/system_architecture.png','图1 实际实现的实验架构与数据流。Test金标只进入评估器；发送给API的是待测原文与固定Train示例。新版LLM直接执行预定四档Test，未重新做完整Dev选参。','scripts/prepare_cluener.py、refresh_split.py、run_baselines.py、run_shot_test.py、run_glm_test.py、run_mimo_test.py、integrate_results.py')
    p('数据层先保存官方原文件，再检查重复、标注冲突和跨度，生成统一字符JSONL与固定划分。模型层有两条分支：传统模型在有效Train上建模，由Dev选参数后推断Test；LLM分支从同一Train生成固定示例前缀，调用三家接口。全部预测均保存为带样本ID、状态与实体跨度的JSONL。评估层只读取金标与预测，统一计算V3指标、错误配对和句子级统计区间。')
    h('3.2 模块划分与持久化')
    t('表3 模块及可核查产出',['模块','主要实现','保存内容'],[
        ['数据审计与划分','prepare_cluener.py / refresh_split.py','原始数据、转换文件、split ID、seed、计数与哈希'],
        ['传统模型','models/baselines.py / run_baselines.py','13组Dev预测与模型、9组HMM无约束对照、最终Test预测'],
        ['提示与API','pilot_shot_budget.py及三家运行入口','完整提示、示例ID、参数、缓存、原始响应、usage与错误'],
        ['统一评估','evaluation/metrics.py','Token/Entity TP、FP、FN、逐类指标、非法输出和失败诊断'],
        ['分析与复现','integrate_results.py / reproduce_main_results.py','主表、配对区间、图表、重放核验与测试日志']
    ],'项目src、scripts及runs目录；代码路径均相对项目根目录',[1800,4000,3838])
    p('API缓存由请求内容、模型与参数等确定，恢复运行复用已有记录，显式失败也保留。预算账本独立于F1评分，未知费用不能因本地没有答案就删去。数据、冻结协议、预测及旧代码快照分开保存，避免后续修改覆盖原实验条件。')

    h('④ 实现细节',1)
    h('4.1 数据审计与可复现划分')
    p('用户提供的cluener_public.zip含官方Train 10,748句、Dev 1,343句及无标签Test 1,345句。官方Test没有金标，不能当成“没有实体”的负样本。原实体end包含末字，转换时执行end+1，并逐实体验证text[start:end]与实体文字一致，避免系统性偏移一位。')
    p('精确重复检查以原文字符串相同为准，另检查相同文本的标注冲突。本地审计未发现精确重复或跨集合重复，也未发现带标签记录间冲突，因此无需按优先级删除或裁定标签。若发现冲突应停止自动划分并输出记录，而不是为凑数量强行分配。审计结果见data/manifests/cluener2020。')
    p('项目保持官方Dev字节不变，从官方Train按句子包含的多标签类别做容量约束迭代分层，得到Train 9,539句、Test 1,209句；比例约78.9∶11.1∶10，不是严格8∶1∶1。随机种子718853259只生成一次并持久保存，未按表现反复挑选。样本ID保留来源前缀，例如Test中的train:50指原官方Train第50条，不表示这条又用于当前Train。')
    p('三个集合均检查重叠或嵌套。唯一嵌套记录train:1506固定在Train并保留原标注，所有传统训练及示例选择统一跳过，实际有效Train为9,538句；Dev和Test无此类金标。删除训练用途与保留数据记录是不同操作，不静默改写原始实体。')
    h('4.2 三个传统基线')
    p('词典只从有效Train实体构造Trie。相同表面词有多个类别时，取训练频次最高者，同频按类别名排序。推断从左到右扫描并采用当前位置最长匹配，匹配后移至实体末尾。它容易漏掉未见实体，也可能把曾作为人名的普通词迁移到错误上下文；这些是可检验的记忆匹配边界。')
    p('HMM使用O及20个BIO标签，共21个状态。低于min_freq的字符映射到UNK，初始、转移和发射计数均加alpha后归一化，在对数域用Viterbi求最优路径以避免概率连乘下溢。主模型不允许句首I-X，且I-X只允许接在B-X或I-X后；无约束对照复用同一组概率，只改变路径合法性。')
    eq('最优标签序列 = argmax［log初始概率 + Σlog标签转移概率 + Σlog字符发射概率］')
    p('CRF使用当前位置及前后两位字符、相邻二元字符、数字/ASCII字母/标点属性及句首句尾占位符。采用sklearn-crfsuite的L-BFGS训练，最多100次迭代，允许学习全部标签转移；主预测中的非法BIO统一按评估规则解释。CRF的正则系数c1、c2由Dev严格实体micro F1选择，而非Test选择。')
    p('传统参数网格为HMM的alpha∈{0.01,0.1,1.0}、min_freq∈{1,2,3}共9组，CRF的(c1,c2)为(0,0.1)、(0.1,0.1)、(0.1,1.0)共3组，加词典1组。并列时优先训练加预测耗时更低者，再按配置顺序。最终HMM为(0.01,2)，CRF为(0.1,0.1)。')
    h('4.3 LLM提示与接口实现')
    example_ids=[x['id'] for x in frozen['examples']]
    p('三家API共享同一基础Prompt、同一套八条Train例句以及原文字符索引。每个Shot是一整条句子及其金标，无论句内有几个实体。2/4/8-shot采用前2/4/8条的嵌套前缀；例句ID依次为'+ '、'.join(example_ids)+'。2-shot含8个实体并覆盖8类，缺movie与scene；4-shot含13个实体、8-shot含22个实体，均覆盖10类。示例按类别覆盖优先等固定规则选择，不依据Test换例句。')
    p('Prompt列出十类定义、边界规则、[start,end)和非重叠输出要求，使用JSON模式，要求实体含text/start/end/type。解析器检查JSON和必需字段，评估器检查跨度和类别；text与切片不符单独诊断，不自动移动实体位置。JSON模式不能保证类别和字符索引正确，提示要求非重叠也不能代替评估器对实际重叠的处理。')
    t('表4 本轮API固定请求条件',['模型别名','thinking请求','输出上限','温度','调度'],[
        ['deepseek-flash','disabled','2048','0','8路'],
        ['glm-5.3-flashx','enabled；low','8192','0','先串行333条，后4路'],
        ['mimo-v2.6-flash','disabled','2048','0','4路；全局90 RPM']
    ],table+'api_protocols.csv；三家runs目录中的冻结协议',[2400,2300,1500,1000,2438])
    p('表4描述请求参数，不等于能观察到服务端全部内部实现。GLM与另外两家thinking及输出上限不同，延迟与效果不能全部归于模型名称。三家返回的是模型别名，不保证不可变权重版本；temperature=0也不保证重新调用逐字节一致。DeepSeek Test调用为10月24日，GLM为10月27日，MiMo跨10月27—28日；精确UTC时间保存在账本。')
    h('4.4 踩坑与修正记录')
    t('表5 关键问题及实际处理',['问题','处理与证据','对结论的影响'],[
        ['跨度闭区间差一位','end+1并逐实体切片核验；数据转换测试','不允许错误索引污染所有实体F1'],
        ['重叠在两级评分中口径不一致','实现V3集合式字符BIO评估；V1/V2独立保存并重评','承认V3在观察历史Test后修订，不倒改旧版本标签'],
        ['书名号提示与金标边界冲突','新版Prompt保留book/movie书名号并增加text；20条Dev×四档试跑','新版按要求直接四档Test，跳过新的完整Dev选参'],
        ['热点断联与余额中断','保留GLM117条连接失败与未知费用；仅恢复未得答案的余额故障','4-shot含工程故障，不作为纯能力下降证据'],
        ['MiMo结构失败与连接拒绝','0/2-shot的17/11条格式失败及8-shot4条连接失败保留；缓存恢复','不隐藏修复、不挑更好答案；abort原因不从日志外推'],
        ['历史运行脚本被pytest误收集','testpaths限定tests；现有136项离线测试通过','不把历史源码快照当当前测试入口']
    ],'docs/progress.md、docs/evaluation_v3.md、API恢复记录及tests',[1900,4100,3638])
    p('新版DeepSeek先在固定20条Dev上对四档检查接口和费用，共80条试跑。早期旧Prompt做过Dev调参，不能把那次选出的4-shot套用为新版三家模型的Dev最优。新版的四档Test对比是已指定配置的描述性实验，Test最高分不作为事后重新选参的理由。')

    h('⑤ 评估',1)
    h('5.1 评估数据及类别分布')
    p('主结果均来自同一带金标的1,209句项目Test，共2,648个实体。Test由原官方Train抽出，未用于传统训练、示例选择或训练词典；Dev固定1,343句。划分前去重/冲突审计已通过，样本ID与SHA256固定。下表是实际文件中的实体出现次数，一句多个同类实体分别计数；Train包含保留但跳过训练的嵌套记录。')
    t('表6 固定划分的实体类别分布',['类别','Train 9539句','Dev 1343句','Test 1209句'],
      [[label,*[str(manifest['stats'][s]['entity_counts'][label]) for s in ['train','dev','test']]] for label in LABELS],
      'data/splits/cluener2020/manifest.json',[3000,2212,2212,2214])
    h('5.2 两级指标及失败处理')
    p('严格实体级比较(样本ID,start,end,type)集合；只有边界和类别全部一致才计TP。Token级将合法实体展开为(样本ID,字符位置,BIO标签)集合后比较。两者均先汇总全部句子的TP、FP、FN，再计算micro指标；不平均每句F1。O→O不增加主指标TP，O→实体计FP，实体→O计FN，B/I或类别混淆同时产生FP与FN。')
    eq('Precision = TP / (TP + FP)；Recall = TP / (TP + FN)；F1 = 2TP / (2TP + FP + FN)')
    p('V3保留全部合法重叠跨度，并在两级指标中使用相同合法实体集合。精确重复实体集合去重；BIO输入保留原始输出，非法I-X统一解释成B-X并记录原非法率。越界或未知类别不进入两套合法跨度主指标，同时报告无效项计实体FP的辅助指标。缺失、重复或多余样本ID直接停止评分；显式API或格式失败作为空预测，失败率另报。零分母P/R/F1记0并保存原计数。')
    p('辅助表提供固定10类macro F1、每类实体P/R/F1、20个BIO标签指标和字符标签集合准确率。若同起点同类型的短实体被较长实体完全覆盖，它可能不新增Token集合错误，却仍新增严格实体FP；因此Token指标不能替代实体指标。DeepSeek 4-shot有1个无效实体，合法实体F1与无效项计FP的辅助F1都留在主表CSV，避免只看清洗后分数。')
    h('5.3 完整Test主结果')
    t('表7 十五组方法的双层主指标 单位为%', ['方法','Token P','Token R','Token F1','实体P','实体R','实体F1'],
      [[r['display'],*[pct(r[f'{level}_{k}']) for level in ['token','entity'] for k in ['precision','recall','f1']]] for r in rows],
      table+'main_results.csv；每行均为1209句',[2350,1214,1214,1214,1214,1214,1218])
    p('三个传统模型中，CRF实体F1为68.66%，高于词典53.79%和HMM48.12%。三个LLM的零样本实体点估计均低于CRF；加入示例后的部分配置超过CRF，说明预训练与指令理解并不自动消除细粒度边界和类别错误。表中各档均完整报告，不只展示Test最高配置。')
    p('DeepSeek 8-shot实体F1为71.88%，较CRF高3.22个百分点，句子配对95%区间为[1.04,5.38]；MiMo 4-shot较CRF高2.62个百分点，区间为[0.46,4.81]。这些是固定预测、固定测试样本下的探索性差值区间，不是经多重比较校正的总体模型优劣结论。')
    h('5.4 Shot数量与类别差异')
    t('表8 四档Shot的实体F1 单位为%',['模型','0-shot','2-shot','4-shot','8-shot'],
      [[family,*[pct(by[f'{family}_{s}']['entity_f1']) for s in [0,2,4,8]]] for family in ['deepseek','glm','mimo']],table+'main_results.csv',[2400,1800,1800,1800,1838])
    fig('figures/integrated_v1/entity_shot_curve.png','图2 同一固定示例池下的Shot对比。点为实际0/2/4/8-shot观测，连线仅辅助阅读；区间为2,000次句子Bootstrap百分位95%区间。所有方法使用完整1,209句Test，GLM 4-shot保留117条连接失败。','reports/figures/integrated_v1/figure_manifest.json；'+table+'main_results.csv')
    p('DeepSeek从0到8-shot的实体F1增加5.14个百分点，但4-shot低于2-shot；MiMo 4-shot为71.28%，8-shot为70.69%。MiMo的8−4配对差值为−0.59个百分点，95%区间[−1.59,0.38]包含0，不能将这次下降说成稳定退化。增加Shot同时改变例句内容、类别覆盖与长度；仅一套示例池，也无法证明换例句后仍有相同排序。')
    t('表9 同为8-shot时的逐类实体F1及CRF参照 单位为%', ['类别','CRF','DeepSeek 8','GLM 8','MiMo 8'],
      [[label,*[pct(next(r['f1'] for r in per_type if r['method']==method and r['category']==label)) for method in ['crf','deepseek_8','glm_8','mimo_8']]] for label in LABELS],
      table+'entity_per_type.csv；选同一Shot口径而非逐类挑最好结果',[2600,1700,1800,1800,1738])
    p('LLM总体分数更高不等于每类都更高。以同为8-shot为例，DeepSeek的name F1为85.91%，高于CRF的69.53%；但position为49.81%，低于CRF的73.44%，government也低于CRF。逐类差异提示模型间的收益并不均匀，不能只凭总体排名选择所有字段的处理方法。全部15组共150条逐类指标及支持数保留在源CSV，不对每一列挑最好结果拼成不存在的模型。')
    h('5.5 参数对比与三个课程原理的实证')
    p('原理一，词表选择会改变未知字符处理和序列模型的统计稀疏性，对应第2讲。本项目固定alpha=0.01，只改变最低字符频次，观测如下。这里研究的是字符词表与UNK，没有实施不同分词器或子词算法对比。')
    hmm=[r for r in parameters if r['method']=='hmm' and json.loads(r['config']).get('alpha')==.01]
    t('表10 HMM词表阈值的Dev对比',['min_freq','词表大小','未知字符率%','Token F1%','实体F1%'],
      [[json.loads(r['config'])['min_freq'],r['vocabulary_size'],f"{100*float(r['dev_unknown_char_rate']):.4f}",pct(r['v3_dev_token_f1']),pct(r['v3_dev_entity_f1'])] for r in hmm],table+'baseline_dev_parameters.csv',[1600,1700,2200,2100,2038])
    p('阈值1→2使词表3593→3019，未知率0.1811%→0.3741%，实体F1从48.29%微升到48.45%；继续到3为48.42%。结果说明更小词表并非单调更好，稀有字符归并与信息损失存在取舍；差异较小，不能外推为所有任务的最优阈值。')
    p('原理二，BIO约束规定合法标签路径，却不保证实体识别一定提高，对应第9讲。9组HMM分别以相同概率参数进行有/无约束解码。alpha=0.01和0.1时，无约束非法率均为0；alpha=1.0的3组均约0.0020%。约束排除了这些非法路径，但9组实体F1最大变化为0.0000个百分点；不能为了讲概念而补造正向收益。Viterbi小序列还通过穷举结果一致性测试。')
    p('原理三，指标必须匹配业务正确条件，并显式规定后处理，对应第14—15讲。HMM的Token F1为62.96%，高于词典60.41%，实体F1却低5.67个百分点；两者排名随评估粒度改变。第⑥节边界例Token F1为87.50%，严格实体F1为0，更直接显示部分字符命中不等于完整字段正确。V3保留重叠并补充无效输出指标，是对评估口径的具体约定，而非用Accuracy掩盖大量O字符。')
    t('表11 CRF正则系数的Dev对比',['c1','c2','Token F1%','实体F1%','历史选中'],
      [[json.loads(r['config'])['c1'],json.loads(r['config'])['c2'],pct(r['v3_dev_token_f1']),pct(r['v3_dev_entity_f1']),'是' if r['selected']=='True' else '否'] for r in parameters if r['method']=='crf'],table+'baseline_dev_parameters.csv',[1400,1400,2300,2300,2238])
    p('CRF在(c1,c2)=(0.1,0.1)时Dev实体F1为70.61%，优于另外两组64.23%和68.88%，因此历史Dev选择该配置。参数实验只用Dev，未根据Test重新排序或切换选参标准。全部9组HMM与3组CRF参数及耗时在源表保留。')
    h('5.6 配对区间与网络诊断')
    p('配对Bootstrap以句子为重采样单位，seed=718853259，2,000次有放回抽样。每轮对所有模型使用相同样本索引，先合计TP、FP、FN再计算全局micro F1和差值，取2.5%与97.5%分位数。42组配对同时包含LLM对CRF、同Shot跨模型与模型内Shot对比；全部为探索性、未经多重比较校正的区间。它不覆盖重新训练、API重复调用或更换示例池的波动。')
    pairs=[('deepseek_8','crf'),('mimo_4','crf'),('glm_8','crf'),('deepseek_8','mimo_8'),('deepseek_8','deepseek_0'),('mimo_8','mimo_4')]
    selected=[next(r for r in comparisons if r['a']==a and r['b']==b and r['metric']=='entity') for a,b in pairs]
    t('表12 实体F1句子配对差值 单位为百分点',['A − B','差值','95%区间'],
      [[names[r['a']]+' − '+names[r['b']],f"{float(r['difference_pp']):+.2f}",f"[{float(r['ci_low_pp']):+.2f}, {float(r['ci_high_pp']):+.2f}]"] for r in selected],table+'bootstrap_comparisons.csv',[5600,1600,2438])
    p('DeepSeek与MiMo同为8-shot时差值区间包含0，不能凭点估计断言前者稳定领先。GLM 4-shot还有117条连接失败和1条内容拒绝，实体F1=64.50%反映的是这次完整运行结果；其下降不能单独归因为Shot过多。')
    affected=['crf','deepseek_8','glm_4','glm_8','mimo_4','mimo_8']
    t('表13 共同网络子集的事后诊断 单位为%',['方法','全量1209句实体F1','共同1088句实体F1'],
      [[names[m],pct(by[m]['entity_f1']),pct(next(r['subset_entity_f1'] for r in subset if r['method']==m))] for m in affected],table+'network_subset.csv',[3800,2919,2919])
    p('诊断删除所有方法明确网络失败句ID的并集121句，全部15组都使用相同剩余1,088句；格式失败、内容拒绝和未知HTTP错误仍保留。GLM 4-shot在子集为68.06%。这个子集是事后、非随机条件分析，不是修好网络后的全量分数，也不替代主表。其他方法子集分数同样会因句子构成改变而变化。')

    h('⑥ 错误分析与局限',1)
    h('6.1 归因规则与整体分布')
    p('先移除精确正确实体，再对剩余跨度做固定的一对一配对：同边界优先，其次同类重叠，再次其他重叠，同层按重叠程度处理。按配对分为边界错误、类型错误、边界兼类型错误；未配对金标为漏抽，未配对预测为误抽。同一实体不重复归因，类别计数不能直接相加当作句子数。')
    selected_errors=['dictionary','hmm','crf','deepseek_8','glm_4','glm_8','mimo_4','mimo_8']
    t('表14 代表配置的实体错误归因次数',['方法','边界','类型','边界兼类型','漏抽','误抽'],
      [[names[m],*[next(r[k] for r in errors if r['method']==m) for k in ['boundary','type','boundary_and_type','missed','spurious']]] for m in selected_errors],table+'error_counts.csv；其余配置同文件完整保存',[2800,1300,1300,1638,1300,1300])
    p('API失败产生的空预测也会形成漏抽。例如GLM 4-shot失败句含261个真实实体，占该组1,082个实体FN的24.12%；其中网络失败句含259个实体。因此既要看语义错误分布，也要读运行状态；不能把工程故障全部解释成模型不认识实体。')
    h('6.2 三个真实失败案例')
    kinds={'boundary':('边界错误','预测多包含“杂志”两个字符，属于实体边界扩张。金标在本段以右书名号结束，原句本身为片段，不能假定模型拥有完整标题上下文。可观察的归因是边界错误，不声称知道模型内部推理。'),
           'type':('类型错误','原文明确包含“电影”，但输出将同一跨度标为game；“街霸”可关联多个媒介是一种可能的歧义解释，日志并不能证明模型内部原因。确定的结论是同边界异类型。'),
           'missed':('漏抽','调用成功却未输出金标government“陆军”，属于任务抽取漏失。原文为截断片段，语境不足可能增加困难，但没有对照实验证明这是唯一原因；不能归为网络失败。')}
    for n,c in enumerate(cases,1):
        title,reason=kinds[c['kind']];err=c['matched_error']
        h(f'案例{n} {title}')
        p(f"样本{c['id']}，方法{names[c['method']]}。原文：{c['text']}",keepNext=True,literal=True)
        gold=[f"{x['text']} [{x['start']},{x['end']}) {x['type']}" for x in c['gold_entities']]
        pred=[f"{x.get('text',c['text'][x['start']:x['end']])} [{x['start']},{x['end']}) {x['type']}" for x in c['raw_prediction'].get('entities',[])]
        p('金标：'+'；'.join(gold)+'。预测：'+('；'.join(pred) if pred else '空实体集合')+'。',keepNext=True,literal=True)
        p(f"Token TP/FP/FN={c['token']['tp']}/{c['token']['fp']}/{c['token']['fn']}，F1={pct(c['token']['f1'])}%；严格实体TP/FP/FN={c['entity']['tp']}/{c['entity']['fp']}/{c['entity']['fn']}，F1={pct(c['entity']['f1'])}%。"+reason,keepLines=True)
    p('以上案例由AI对照保存的原文、金标、原始预测自动归因，来源为runs/integrated_v1_analysis/representative_cases.json。它们是说明性案例，不代表独立双人核验或总体错误频率；历史精简错误分析还保存误抽和结构异常实例。')
    h('6.3 结构化输出与有效性限制')
    p('DeepSeek四档调用无显式失败，但仍有文本跨度不符和少量重叠；GLM 0/8-shot各有1条格式失败；MiMo 0/2-shot分别17/11条格式失败，其中16/11条返回finish_reason=abort。日志不足以确认abort的服务器原因，不能将全部错误称为JSON语法失败。MiMo 8-shot另有4条建立HTTP连接前失败，均按空预测保留。')
    p('少样本例句中有截断片段，且2-shot未覆盖movie/scene；增加示例与增加类别覆盖混在一起。只运行一套示例池和一次固定模型预测，不支持“8-shot普遍最好”的结论。三家thinking和输出上限不同，也不能解释成单一架构效果。')
    p('V3指标与新版Prompt都经历了历史结果观察后的调整，新版又按用户要求跳过完整Dev选参，因此本报告不宣称首次盲测或完全未触碰Test。公开数据可能进入预训练语料，零样本仅表示本轮未提供任务例句，不保证模型过去没有见过数据。结果范围限于这份固定Test，不推广到医疗、金融或其他中文域。')

    h('⑦ 工程与伦理',1)
    h('7.1 调用量 单价与费用')
    p('三家主实验各为4×1,209=4,836个样本配置。DeepSeek主实验记录4,836次尝试，GLM因重试与恢复共5,080次，MiMo共4,836次；请求数不等于成功样本数。费用按实际usage与调用时记录的缓存命中/未命中/输出单价计算，未知计费请求保留预留。下表是历史记录，不是查询日重新报价。')
    t('表15 调用时记录的单价 单位为元每百万计费token',['模型','输入未命中','输入命中','输出','记录日期'],[
        ['DeepSeek峰时估算','2.00','0.04','8.00','2026-10-23'],
        ['GLM','2.00','0.57','7.00','2026-10-27'],
        ['MiMo','1.00','0.02','2.00','2026-10-27']
    ],'configs/pricing.json、glm_flashx.json、mimo_flash.json；实际账单未核验',[2300,1800,1800,1800,1938])
    t('表16 正式Test各档usage费用估算与请求延迟',['配置','已知费用元','未知预留元','延迟中位数秒','P95秒'],
      [[r['display'],f"{float(costby[r['method']]['known_usage_estimate_cny']):.4f}",f"{float(costby[r['method']]['unknown_reservations_cny']):.4f}",f"{float(costby[r['method']]['latency_median_seconds']):.3f}",f"{float(costby[r['method']]['latency_p95_seconds']):.3f}"] for r in rows if r['family'] in ['deepseek','glm','mimo']],table+'cost_latency.csv',[2700,1800,1800,1800,1538])
    t('表17 三家正式Test的计费token汇总',['模型','输入token','输出token','已知usage估算元'],
      [[family,f"{sum(int(c['prompt_tokens']) for c in costs if c['family']==family):,}",f"{sum(int(c['completion_tokens']) for c in costs if c['family']==family):,}",f"{sum(float(c['known_usage_estimate_cny']) for c in costs if c['family']==family):.4f}"] for family in ['deepseek','glm','mimo']],table+'cost_latency.csv；仅usage已返回的尝试',[2200,2500,2500,2438])
    p(f"全项目含旧Prompt、Dev和试跑的已知usage估算为{project_cost['known_usage_estimate_cny']:.4f}元，未知费用预留{project_cost['unknown_reservations_cny']:.4f}元，账本保守合计{project_cost['gross_upper_cny']:.4f}元，低于100元上限。未知预留不能称为已扣费；实际支出仍以平台账单为准。MiMo 2-shot便宜于0-shot与实际缓存命中有关，不代表Shot越多费用必然越低。")
    p('传统模型没有API收费，但CPU、训练和人工成本并非零。Test逐句推断中位数：词典0.0152毫秒、HMM0.3215毫秒、CRF0.2170毫秒。LLM延迟包含客户端限速、排队和重试，GLM还混合串行/并行，不能作为纯服务端计算速度排名。MiMo另存HTTP耗时，其他家缺同口径字段，不补造可比数据。')
    h('7.2 权限 隐私与数据许可')
    p('密钥通过环境变量或本地.env提供，真实.env、私钥与虚拟环境排除在Git和结果包外；打包时仅在内存中读取密钥值检查候选文件，不打印密钥。实验使用公开NER文本，没有引入私有金融/病历资料或用户个人数据。公开文本中的人名仍是数据内容，不能因此推定任意二次发布权限。')
    p('本地压缩包README未明确数据许可，本项目不把代码许可推定为语料许可。原始语料、模型、请求响应与完整预测只保存在本地私有证据包；不公开上传。引用CLUENER技术报告也不等于取得语料的再分发授权。许可记录继续保留未解决状态。')
    h('7.3 复现入口与核验状态')
    p('本机环境为Windows x64、Python 3.13.5，传统模型运行于CPU。依赖锁在requirements-lock.txt，随机种子、划分ID和数据哈希在configs及data/splits。另一台电脑先按README用Python与uv重建环境，将有权使用的本地证据包解压至项目根目录，再运行下面的一键离线入口；仅克隆GitHub不包含语料和预测，无法直接得到本次主表。')
    code('.\\.venv\\Scripts\\python.exe scripts/reproduce_main_results.py')
    p('该入口依次重评15组预测、重建图表、检查报告源表与哈希并运行测试；不读密钥或启动付费API运行器。缺少本地证据时明确报错并停止。已保存预测的V3重算与原指标完全一致，本机136项离线测试通过，图表也已检查；这些证据不冒称另一环境独立复现。离线重放可重复统计，不保证以后重新请求远程API返回完全相同文本。')
    p('对关键数字的复核方式是：主结果看main_results.csv的P/R/F1与TP/FP/FN；参数看baseline_dev_parameters.csv；错误看error_counts.csv和原始案例；费用看cost_latency.csv与逐尝试ledger；数据看manifest的ID和SHA256。报告另存course_report_provenance.json记录引用源表哈希，避免数字找不到出处。现有私有ZIP附逐文件SHA256与CRC验证；新增报告不会倒改10月28日旧快照。')
    h('7.4 AI使用声明')
    p('本项目使用ChatGPT/Codex辅助实验讨论、数据与模型代码、评估器、API工程、实验执行、自动测试、结果统计、错误归因、图表与报告撰写。大量代码和操作由AI辅助产生或执行，不描述为成员全部独立手工完成。实验表格来自实际保存结果，没有编造分数，没有隐藏LLM修复调用。')
    p('错误归因和图表检查属于AI辅助分析，不等同于两位成员独立人工审核；历史40/20审核扩展没有完成。成员应据实核对最终报告、补充自己的检查与修改证据，并对提交内容负责。工具界面显示型号、姓名、签字和日期由实际使用者填写，详见AI声明草稿与第⑧节。')

    h('⑧ 分工与贡献',1)
    p('个人贡献必须按实际参与填写，不能将计划分工、AI生成代码量或自动运行时长直接换算成个人工作量。下表与每人一份贡献声明使用同一组字段；当前没有成员姓名、工时及签字记录，因此保留待填写，不虚构完成人审、录屏或交叉复现。')
    t('表18 成员实际贡献汇总',['成员','实际模块','具体产出与证据','工作量自评','交叉确认'],[
        ['A 姓名学号待填','待据实填写','文件/提交/运行日志/报告段落待填','实际时间与难点待填','范围/日期待填'],
        ['B 姓名学号待填','待据实填写','文件/提交/运行日志/报告段落待填','实际时间与难点待填','范围/日期待填']
    ],'docs/contributions.md；两位成员分别完成个人声明后核对一致性',[1700,1700,2900,1900,1438])
    p('可据实记录的贡献包括：提出并修订实验方案、理解和核验AI辅助代码、实际运行与排错、检查金标和难例、复现主表、修改文字和录屏。现有文件可以作为产出证据，但归属于哪位成员必须由真实参与者确认。成员还需完成报告身份信息、AI声明确认、真实录屏和另一环境复现记录。')
    h('参考文献与材料来源')
    references=[
        ('[1] Liang Xu等. CLUENER2020 Fine-grained Named Entity Recognition Dataset and Benchmark for Chinese. 2020.','https://arxiv.org/abs/2001.04351'),
        ('[2] Charles Sutton, Andrew McCallum. An Introduction to Conditional Random Fields. 2010.','https://arxiv.org/abs/1011.4088'),
        ('[3] Edward J. Hu等. LoRA Low-Rank Adaptation of Large Language Models. 2021.','https://arxiv.org/abs/2106.09685'),
        ('[4] Patrick Lewis等. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. 2020.','https://arxiv.org/abs/2005.11401'),
        ('[5] EDS-NLP v0.22.0. NER Metrics.','https://aphp.github.io/edsnlp/v0.22.0/metrics/ner/'),
        ('CLUENER2020官方代码与数据说明。网页统计不替代本地文件审计。','https://github.com/CLUEbenchmark/CLUENER2020')]
    for title,url in references:blocks.append({'type':'reference','text':title,'url':url})
    h('附录 关键数字与文件对应')
    t('表A1 主张与直接证据',['关键内容','相对项目根目录的来源'],[
        ['主表与逐类',table+'main_results.csv；entity_per_type.csv'],
        ['词表/约束/CRF参数',table+'baseline_dev_parameters.csv'],
        ['42组配对Bootstrap',table+'bootstrap.json；bootstrap_comparisons.csv'],
        ['失败及归因',table+'diagnostics.csv；error_counts.csv；案例见正文来源'],
        ['费用、usage、延迟',table+'cost_latency.csv；project_cost.json；各API ledger.json'],
        ['数据、ID、seed与SHA256','data/splits/cluener2020/manifest.json；configs/split_seed.json'],
    ],'完整源文件哈希另见reports/course_report_provenance.json',[2800,6838])
    # Apply the review without changing frozen predictions, methods or metrics.
    from revise_course_report import revise
    blocks, appendix_blocks, revision_sources = revise(blocks)
    from finalize_v1_course_report import finalize
    blocks, appendix_blocks, formal_sources = finalize(blocks, appendix_blocks)
    SOURCES.update(formal_sources)
    SOURCES.update(revision_sources)
    # Numeric tables are rendered from objects also used for the DOCX, preventing parallel manual edits.
    content={'title':'中文命名实体识别的传统基线与大模型提示抽取对比',
        'subtitle':'基于CLUENER2020的课程实验报告','date':'2026年10月30日',
        'metadata':'课程 NLP  ｜  班级 待填写  ｜  成员A 姓名学号待填写  ｜  成员B 姓名学号待填写',
        'blocks':blocks}
    save_json(OUT/'course_report_content.json',content)
    appendix={**content,'title':'CLUENER2020课程报告附录',
        'subtitle':'数据 参数 费用与实现核验证据','metadata':'与正文使用同一保存预测与原始评估口径；非新模型实验',
        'blocks':appendix_blocks}
    save_json(OUT/'course_report_appendix_content.json',appendix)
    md=['# '+content['title'],'',content['subtitle'],'',content['metadata'],'',content['date'],'']
    for b in blocks:
        kind=b['type']
        if kind=='heading':md+=['#'*(b['level']+1)+' '+b['text'],'']
        elif kind in ['paragraph','equation']:md += [b['text'],'']
        elif kind=='code':md+=['```powershell',b['text'],'```','']
        elif kind=='reference':md += [f"{b['text']} [原始来源]({b['url']})",'']
        elif kind=='figure':md += [f"![{b['caption']}]({b['path']})",'',b['caption'],'','来源：'+b['source'],'']
        elif kind=='table':
            md += [b['title'],'','| '+' | '.join(b['headers'])+' |','|'+'|'.join(['---']*len(b['headers']))+'|']
            md += ['| '+' | '.join(str(x).replace('|','／').replace('\n',' ') for x in row)+' |' for row in b['rows']]
            md += ['','来源：'+b['source'],'']
    (OUT/'course_report.md').write_text('\n'.join(md),encoding='utf-8')
    appendix_md=['# '+appendix['title'],'',appendix['subtitle'],'']
    for b in appendix_blocks:
        if b['type']=='heading':appendix_md += ['#'*(b['level']+1)+' '+b['text'],'']
        elif b['type']=='paragraph':appendix_md += [b['text'],'']
        elif b['type']=='table':
            appendix_md += [b['title'],'','| '+' | '.join(b['headers'])+' |','|'+'|'.join(['---']*len(b['headers']))+'|']
            appendix_md += ['| '+' | '.join(str(x).replace('|','／') for x in row)+' |' for row in b['rows']]
            appendix_md += ['','来源：'+b['source'],'']
    (OUT/'course_report_appendix.md').write_text('\n'.join(appendix_md),encoding='utf-8')
    SOURCES.update(['src/ner_experiment/models/baselines.py','src/ner_experiment/evaluation/metrics.py',
        'scripts/build_course_report.py','scripts/reproduce_main_results.py','configs/pricing.json',
        'configs/glm_flashx.json','configs/mimo_flash.json','docs/progress.md','docs/evaluation_v1.md','src/ner_experiment/evaluation/metrics_v1.py'])
    save_json(OUT/'course_report_provenance.json',{'report_date':'2026-10-30','methods':15,'test_sentences':1209,
        'section_titles':[b['text'] for b in blocks if b['type']=='heading' and b['level']==1],
        'source_sha256':{name:file_hash(ROOT/name) for name in sorted(SOURCES)},
        'markdown_sha256':file_hash(OUT/'course_report.md'),'new_api_calls':0,
        'pending_human_inputs':['names and student IDs','actual contributions and workload','AI declaration confirmation','real demo recording','independent-environment reproduction'],
        'external_references':[{'title':b['text'],'url':b['url']} for b in blocks if b['type']=='reference'],
        'revision':'original evaluator formal reporting; main/appendix separated; section 8 unchanged'})
    print('WROTE eight-section report:',OUT/'course_report.md','blocks=',len(blocks))


if __name__=='__main__':main()
