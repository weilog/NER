"""Offline evidence-backed records; never constructs an API client or changes runs."""
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import io
from pathlib import Path
import zipfile
from ner_experiment.common import ROOT, LABELS, load_json, read_jsonl, save_json, file_hash, object_hash

EXP_ID='CLUENER-B-261024-001'
LOG=ROOT/'reports/实验日志/CLUENER2020/Prompt'/f'{EXP_ID}.md'
RECORDS=ROOT/'runs/experiment_records'/EXP_ID
TABLES=ROOT/'reports/tables/prompt_version_comparison'

def relative(path):return Path(path).relative_to(ROOT).as_posix()

def link(path):
    import os
    return Path(os.path.relpath(path,LOG.parent)).as_posix()

def write_csv(path,rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(stream.getvalue(),encoding='utf-8-sig')

def local_time(value):
    return datetime.fromisoformat(value).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S +08:00')

def main():
    now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    old_interface=load_json(ROOT/'runs/llm/interface_frozen.json')
    old_ledger=load_json(ROOT/'runs/llm/ledger.json')
    selection=load_json(ROOT/'runs/llm/selection.json')
    latest=load_json(ROOT/'runs/llm_text_test_v1/summary.json')
    frozen=load_json(ROOT/'runs/llm_text_test_v1/frozen_protocol.json')
    pilot=load_json(ROOT/'runs/llm_text_pilot_v1/budget_summary.json')
    protocol=frozen['protocol'];examples=protocol['examples']
    new_groups={g['shots']:g for g in latest['groups']}
    pairs=[];category_rows=[];error_rows=[];summary_rows=[];old_keys=set()
    inputs={ROOT/'configs/experiment.json',ROOT/'configs/pricing.json',ROOT/'configs/split_seed.json',
        ROOT/'configs/frozen_experiment.json',ROOT/'data/splits/cluener2020/manifest.json',
        ROOT/'runs/llm/interface_frozen.json',ROOT/'runs/llm/selection.json',ROOT/'runs/llm/ledger.json',
        ROOT/'runs/baselines/selection.json',ROOT/'reports/evaluation_versions/comparison.md',
        ROOT/'reports/evaluation_versions/selection_audit.json',ROOT/'reports/evaluation_versions/bootstrap.json'}
    for shots,name in [(0,'llm_zero'),(4,'llm_few')]:
        folder=ROOT/'runs/final'/name
        old=load_json(ROOT/'runs/evaluation_versions/v3/final'/name/'metrics.json')
        run=load_json(folder/'run.json');preds=read_jsonl(folder/'predictions.jsonl')
        new_preds=read_jsonl(ROOT/f'runs/llm_text_test_v1/shot_{shots}/predictions.jsonl')
        assert len(preds)==len(new_preds)==1209
        assert {r['id'] for r in preds}=={r['id'] for r in new_preds}
        keys={r['cache_key'] for r in preds};old_keys.update(keys)
        entries=[a for a in old_ledger['attempts'] if a['key'] in keys]
        cost=sum(a['charged_upper_cny'] for a in entries)
        new=new_groups[shots]
        pair={'shots':shots,'old':{'metrics':old,'run':run,'group_peak_cost_cny':cost},'new':new,
            'token_f1_delta_pp':100*(new['token']['f1']-old['token']['f1']),
            'entity_f1_delta_pp':100*(new['entity']['f1']-old['entity']['f1'])}
        pairs.append(pair)
        summary_rows.append({'shots':shots,'old_token_f1':old['token']['f1'],'new_token_f1':new['token']['f1'],
            'token_delta_pp':pair['token_f1_delta_pp'],'old_entity_f1':old['entity']['f1'],
            'new_entity_f1':new['entity']['f1'],'entity_delta_pp':pair['entity_f1_delta_pp'],
            'old_group_peak_cost_cny':cost,'new_group_peak_cost_cny':new['cost_peak_ledger_cny'],
            'old_latency_median_seconds':run['latency_median_seconds'],'new_latency_median_seconds':new['latency_median_seconds']})
        for category in LABELS:
            category_rows.append({'shots':shots,'category':category,'old_entity_f1':old['entity_per_type'][category]['f1'],
                'new_entity_f1':new['entity_per_type'][category]['f1'],
                'delta_pp':100*(new['entity_per_type'][category]['f1']-old['entity_per_type'][category]['f1'])})
        for kind in ['boundary','type','boundary_and_type','missed','spurious']:
            error_rows.append({'shots':shots,'kind':kind,'old':old['error_counts'].get(kind,0),'new':new['error_counts'].get(kind,0)})
        inputs.update(folder.glob('*.json'));inputs.update(folder.glob('*.jsonl'))
        inputs.update((ROOT/'runs/evaluation_versions/v3/final'/name).glob('*.json'))
        inputs.update((ROOT/'runs/evaluation_versions/v3/final'/name).glob('*.jsonl'))
    assert old_interface['parameters']==protocol['parameters']
    assert old_interface['example_ids'][:4]==protocol['example_ids'][:4]
    for name in ['dictionary','hmm','crf']:
        inputs.update((ROOT/'runs/final'/name).glob('*.json'))
        inputs.update((ROOT/'runs/final'/name).glob('*.jsonl'))
        inputs.update((ROOT/'runs/evaluation_versions/v3/final'/name).glob('*.json'))
    for folder in ['runs/llm_text_test_v1','runs/llm_text_pilot_v1']:
        inputs.update((ROOT/folder).rglob('*.json'));inputs.update((ROOT/folder).rglob('*.jsonl'))
    for key in old_keys:
        path=ROOT/'runs/llm/cache'/(key+'.json')
        if not path.exists():raise RuntimeError(f'Missing original cache {key}')
        inputs.add(path)
    inputs.update((ROOT/'data/splits/cluener2020').glob('*.json'))
    manifest=load_json(ROOT/'data/splits/cluener2020/manifest.json')
    for item in manifest['files'].values():inputs.add(ROOT/item['path'])
    inputs.update((ROOT/'src/ner_experiment').rglob('*.py'))
    inputs.update((ROOT/'scripts').glob('*.py'));inputs.update((ROOT/'tests').glob('*.py'))
    for name in ['requirements.txt','requirements-lock.txt','pyproject.toml']:
        if (ROOT/name).exists():inputs.add(ROOT/name)
    sources={relative(p):{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in sorted(inputs) if p.is_file()}
    # Only explicitly enumerated source artifacts enter the local archive. No .env or keys.
    if any(Path(p).name.startswith('.env') or Path(p).suffix in {'.pem','.key'} for p in sources):raise RuntimeError('Unexpected secret path')
    counts={n:Counter(e['type'] for row in examples[:n] for e in row['entities']) for n in [0,2,4,8]}
    evidence_example={'sample_id':'train:32','gold':'《明堂丹令论》','old_zero':'明堂丹令论','new_zero':'《明堂丹令论》'}
    # Recheck the cited case against saved predictions, rather than trusting conversation text.
    gold={r['id']:r for r in read_jsonl(ROOT/manifest['files']['test']['path'])}
    row=gold[evidence_example['sample_id']]
    sample_sets={}
    for version,path in [('old','runs/final/llm_zero/predictions.jsonl'),('new','runs/llm_text_test_v1/shot_0/predictions.jsonl')]:
        pred=next(r for r in read_jsonl(ROOT/path) if r['id']==row['id'])
        sample_sets[version]={row['text'][e['start']:e['end']] for e in pred['entities'] if e['type']=='book'}
    assert evidence_example['old_zero'] in sample_sets['old'] and evidence_example['new_zero'] in sample_sets['new']
    comparison={'experiment_id':EXP_ID,'recorded_at':now,'evaluation_version':'v3','test_samples':1209,
        'pairs':pairs,'new_all_shots':latest['groups'],'old_dev_selected_shots':selection['best_shots'],
        'same_parameters':True,'same_four_example_ids':True,'same_test_id_sets':True,
        'example_counts':{str(n):dict(counts[n]) for n in counts},'verified_case':evidence_example,
        'prompt_change':{'old':old_interface['system_prompt'],'new':protocol['system']},
        'costs':{'historical_peak_ledger_cny':pilot['prior_peak_ledger_cny'],
                 'new_pilot_peak_cny':pilot['pilot_peak_ledger_cny'],'new_test_peak_cny':latest['new_peak_ledger_cny'],
                 'project_peak_cny':latest['project_peak_ledger_cny']},
        'limitations':['Revised protocol after historical Test inspection; not an independent blind test',
            'New Dev tuning/full Dev skipped at explicit user request; no new Dev-selected best shot',
            'Prompt boundary and output text schema changed together; no single-factor ablation',
            'Provider model alias is not an immutable version; independent API calls on different dates',
            'One nested example pool only, class balance/length and content confounded; truncated examples retained',
            'Costs reflect usage and peak local prices, not actual invoice/balance; latency depends on cache/network/concurrency',
            'Standalone missed/spurious error-pair counts are not identical to total FN/FP'],
        'source_file_count':len(sources),'source_manifest_sha256':object_hash(sources)}
    save_json(TABLES/'comparison.json',comparison)
    write_csv(TABLES/'summary.csv',summary_rows);write_csv(TABLES/'per_type.csv',category_rows);write_csv(TABLES/'error_counts.csv',error_rows)
    write_csv(TABLES/'example_distribution.csv',[{'category':c,**{f'shot_{n}':counts[n][c] for n in [0,2,4,8]}} for c in LABELS])
    save_json(RECORDS/'sources.json',{'experiment_id':EXP_ID,'recorded_at':now,'files':sources})
    archive=RECORDS/'evidence.zip'
    existing=load_json(RECORDS/'archive.json') if (RECORDS/'archive.json').exists() else None
    if existing and archive.exists() and existing['source_manifest_sha256']==object_hash(sources):
        if file_hash(archive)!=existing['sha256']:raise RuntimeError('Existing evidence archive is damaged')
    else:
        if archive.exists():raise RuntimeError('Existing archive represents another source snapshot; preserve it and use a new record ID')
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for name in sources:z.write(ROOT/name,name)
            z.writestr('EVIDENCE_SOURCES.json',__import__('json').dumps(sources,ensure_ascii=False,indent=2))
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise RuntimeError('Evidence ZIP CRC verification failed')
        save_json(RECORDS/'archive.json',{'experiment_id':EXP_ID,'sha256':file_hash(archive),
            'bytes':archive.stat().st_size,'source_manifest_sha256':object_hash(sources),'source_files':len(sources),'zip_crc_verified':True})
    # Source runs and data are read-only during the entire recording operation.
    assert all(file_hash(ROOT/name)==item['sha256'] for name,item in sources.items())
    lines=['---',f'experiment_id: {EXP_ID}','dataset: CLUENER2020','task: Chinese NER',
        'experiment_type: Prompt version comparison and four-shot Test extension',f'recorded_at: "{now}"',
        'timezone: Asia/Shanghai','status: completed_with_limitations','evaluation_protocol: v3',
        'sample_batch: CLUENER2020-split-718853259','test_samples: 1209',
        f'raw_archive: runs/experiment_records/{EXP_ID}/evidence.zip','ai_assisted: true','---','',
        '# CLUENER2020：LLM Prompt 修订与四档 Shot Test 实验记录','',
        '## 记录目的与实际执行顺序','',
        '本记录汇总已完成的历史实验、评估协议修订、新版预算试跑、四档完整Test与两版比较。所有表格来自本地保存结果；本次记录本身新增API调用为0。',
        '', '1. 2026-10-23：历史提示在20条Dev试跑；300条Dev比较0/2/4/8-shot，选择4-shot；完整Dev和Test仅运行0/4-shot。',
        '2. 查看历史Test后讨论并修订评估协议，保留V1/V2，实现V3保留重叠的字符BIO集合评估；34份历史预测离线重评，未重发请求。',
        '3. 审核并修整词典/HMM/CRF和LLM运行保护，补上数据/产物哈希、冻结前检查、独立目录和模拟测试。',
        '4. 2026-10-24：依据Train标注检查书名号边界，修订新版提示并增加text字段；固定20条Dev × 四档共80次请求，估算预算。',
        f'5. 用户明确要求直接进入最后阶段：跳过新版300条Dev和完整Dev，在{local_time(frozen["frozen_at"])}冻结本轮配置，至{local_time(latest["completed_at"])}完成四档Test。',
        '6. 本次记录没有新增示例池、修改已有预测、根据Test选择最佳配置或重新发送更好答案。',
        '', '## 数据、模型与可比条件','',
        'Train/Dev/Test=9539/1343/1209；有效Train9538；seed=718853259，只生成一次。官方Dev不变，Test来自原Train；唯一嵌套样本train:1506固定Train并跳过训练和提示。',
        '旧、新对比均使用同一1209条Test及V3评估；API参数和4-shot示例ID经程序核对相同。旧Test：2026-10-23，新Test：2026-10-24。实际接口均返回deepseek-flash，未返回可验证的不可变模型版本。',
        '`temperature=0, max_tokens=2048, thinking.type=disabled, response_format=json_object, stream=false`。本轮并发8。请求参数证明客户端设置，不能独立证明服务端内部推理模式。',
        '', '## 提示版本与变更理由','',
        '旧版要求“不带书名号”，与Train中的大量实体标注冲突。Train有效样本核查发现book有717个、movie有706个实体将完整《》包含在跨度内（这两个计数不是全部类别实体总数）。',
        '新版在book/movie以《》包围作品名时保留书名号，增加必需text字段，并要求原文[start:end]==text；Few-shot参考答案同步增加text。类别定义、示例ID、模型参数不变。完整旧/新提示已写入comparison.json和原冻结记录。',
        '这两项修改同时发生，没有做单独消融。可确认边界错误显著减少，不能把全部分数提升严格归因于某一项修改。',
        '', '## 统一评估与失败处理','',
        '字符为原文Unicode字符，不是API子词。实体主指标严格匹配(start,end,type)；Token主指标比较(样本ID,字符位置,完整BIO标签)集合，保留所有合法重叠实体，不含O项。没有重叠时等价于原单标签BIO micro。',
        '非法I按固定I→B解释并记录；重复去重；失败按空预测计分；越界/未知类排除合法跨度主指标，但必须另报无效项数量和entity_with_invalid_fp。text不一致只诊断，不移动跨度、不搜索文字补位置、不进行隐藏模型修复调用。',
        'V3是在查看旧Test后修订，不称首次预注册。V1/V2历史文件保留；旧报告experiment_results.md及原tables/figures仍是历史版本，不能直接当作当前V3/新版Prompt报告。',
        '', '## 两次Test的同档位对比（V3）','',
        '| Shot | 旧Token F1 | 新Token F1 | 变化/百分点 | 旧实体F1 | 新实体F1 | 变化/百分点 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in pairs:
        o,n=p['old']['metrics'],p['new']
        lines.append(f'| {p["shots"]} | {o["token"]["f1"]:.2%} | {n["token"]["f1"]:.2%} | {p["token_f1_delta_pp"]:+.2f} | {o["entity"]["f1"]:.2%} | {n["entity"]["f1"]:.2%} | {p["entity_f1_delta_pp"]:+.2f} |')
    lines+=['','旧版完整Test没有2/8-shot，不能制造这两档的旧版Test结果；旧300条Dev曲线与新Test曲线不直接混比。',
        '', '## 新版四档结果','', '| Shot | Token P/R/F1 | 实体 P/R/F1 | 高峰费用估计/元 | 中位延迟/秒 |', '|---|---|---|---:|---:|']
    for g in latest['groups']:
        t,e=g['token'],g['entity']
        lines.append(f'| {g["shots"]} | {t["precision"]:.2%}/{t["recall"]:.2%}/{t["f1"]:.2%} | {e["precision"]:.2%}/{e["recall"]:.2%}/{e["f1"]:.2%} | {g["cost_peak_ledger_cny"]:.4f} | {g["latency_median_seconds"]:.3f} |')
    lines+=['','本轮8-shot观察分数最高，但不能据此将其称为Dev选择的最优配置；旧Dev选出的4-shot仅属于旧提示版本。2→4-shot下降，不能声称效果随数量单调上升。',
        '', '## 示例池、分布与质量局限','', '| 类别 | 2-shot | 4-shot | 8-shot |','|---|---:|---:|---:|']
    for c in LABELS:lines.append('| '+c+' | '+' | '.join(str(counts[n][c]) for n in [2,4,8])+' |')
    lines+=['| 实体总数 | 8 | 13 | 22 |','| 覆盖类别数 | 8 | 10 | 10 |','',
        '一个Shot是一条数据记录及其完整标注，不按实体数计Shot。原数据本身可能是片段，不能将全部记录称为语义完整句子。E1—E8的ID依次为：'+', '.join(protocol['example_ids'])+'。',
        'E5只有“战争黎明2》”6字，E6/E7结尾截断。此前未充分逐条人工审核示例完整性和标签语义；当前算法保证覆盖优先与可复现，不保证类别均衡或示例质量最优。这是实际局限，不应在报告中声称已完成人工清洗。',
        '仅一套嵌套示例，没有多套独立示例的稳定性实验。4→8-shot主要增加game/name；数量、内容、类别覆盖和提示长度共同变化。不得在观察本轮Test后替换示例再把同一Test当作新盲测。',
        '', '## 真实错误证据与解释边界','',
        'train:32：金标《明堂丹令论》，旧零样本输出明堂丹令论，新零样本输出《明堂丹令论》。程序已核对两份实际预测。',
        '纯边界配对错误：零样本624→229，4-shot332→184。零样本book F1为8.43%→73.93%，movie为2.67%→72.49%，与边界规则修订的影响一致。',
        '并非所有错误都减少：零样本独立漏抽配对392→441，4-shot469→482；这些是配对归因中的未配对项，不是总FN。边界/类型配对同时消耗一个FP和一个FN，所以不能据此推断总召回下降。总实体FN分别1280→909、1029→884。',
        '新版0/2/4/8-shot文本跨度不一致数31/8/3/6；重叠句数3/1/2/0；4-shot有1个无效实体。全部API/格式失败数0、重复实体数0。text不一致并未被自动纠正。',
        '旧版没有text字段，无法追溯计算同口径文本跨度不一致率。自动归因不等于两名成员人工核验；40个错误复核及其中20个独立判断仍待本人完成。',
        '', '## 费用、延迟和预算','',
        f'历史账本高峰估计¥{pilot["prior_peak_ledger_cny"]:.6f}；新版80次试跑¥{pilot["pilot_peak_ledger_cny"]:.6f}；新版4836次Test¥{latest["new_peak_ledger_cny"]:.6f}；累计¥{latest["project_peak_ledger_cny"]:.6f}。新版Test含重试总次数仍为4836，即无重试。',
        '试跑估计完整四档Test（无输入缓存、高峰）¥20.56，加30%余量约¥26.73；实际usage高峰估计¥5.66，受服务端输入缓存影响。完整Dev+Test含历史费用和30%余量的原估计¥62.84；后来按用户要求跳过新版Dev，因此不能将该估计与仅Test实测当作相同工作量。',
        '同档位费用：旧/新0-shot约¥0.9853/¥1.1979，旧/新4-shot约¥1.6672/¥1.3307。均按各自请求key在账本归因，而不是把run.json里的累计费用误当作该组费用。',
        '成本不是账户实扣账单，未查询余额。旧/新延迟中位数：0-shot约0.852/0.863秒，4-shot约0.890/0.922秒。缓存命中、网络、并发和不同日期均影响成本/延迟，不能据此证明新提示天然更省钱。',
        '', '## 可支持与不能支持的结论','',
        '- 可支持：在固定数据和V3口径下，本轮新版0/4-shot优于旧版对应档；边界错误减少；当前一套示例下Shot效果非单调；各类别变化不一致。',
        '- 不能支持：提升完全由书名号规则或text字段导致；8-shot普遍最优；模型确定未见过公开数据；LLM架构普遍优于CRF；已完成全套人工复核或异机复现。',
        '- 新版没有完整Dev选参；“新8-shot分数最高”是对已冻结四档Test的描述，不是新增模型选择准则。',
        '', '## 证据、复核与下一步','',
        f'- 原始证据本地归档：[evidence.zip]({link(RECORDS/"evidence.zip")})；来源文件哈希：[sources.json]({link(RECORDS/"sources.json")})；归档哈希及CRC检查：[archive.json]({link(RECORDS/"archive.json")})。',
        f'- 可机读对比与CSV：[comparison.json]({link(TABLES/"comparison.json")})、[summary.csv]({link(TABLES/"summary.csv")})、[per_type.csv]({link(TABLES/"per_type.csv")})、[error_counts.csv]({link(TABLES/"error_counts.csv")})。',
        f'- 新版完整结果：[llm_text_test_results.md]({link(ROOT/"reports/llm_text_test_results.md")})；示例分布：[shot_example_distribution.md]({link(ROOT/"reports/shot_example_distribution.md")})；V1/V2/V3敏感性：[comparison.md]({link(ROOT/"reports/evaluation_versions/comparison.md")})。',
        f'- 归档含{len(sources)}个明确枚举的文件：相关原始请求/响应、预测、指标、账本、冻结记录、划分及统一数据、运行和评估源码；不含.env、密钥或虚拟环境。原始数据和API材料仅本地保存，不发布语料许可。',
        '- 本轮四档保存预测均离线重评一致，历史结果哈希未变化；本次整理再次核对所有来源文件写入前后哈希一致。',
        '- 下一步：人工复核至少40个错误（20个独立判断后复核），核对标签语义；形成最终统一报告、贡献与AI声明、录屏及独立复现证据。无需为记录再次调用API。',
        '', '## AI使用记录','',
        'AI辅助完成代码实现/审核、运行保护、按用户授权的API实验、离线指标重算与文档生成。实验数字来自实际保存文件；AI未替代两名成员的人工复核、贡献核验或独立复现。']
    LOG.parent.mkdir(parents=True,exist_ok=True);LOG.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verification={'recorded_at':now,'experiment_id':EXP_ID,'source_files':len(sources),'source_files_unchanged':True,
        'comparisons':len(pairs),'test_ids_match':True,'parameters_match':True,'four_example_ids_match':True,
        'archive_sha256':file_hash(archive),'archive_crc_verified':True,'new_api_calls':0}
    save_json(RECORDS/'verification.json',verification)
    print('RECORDED',relative(LOG),'sources=',len(sources),'archive_MB=',round(archive.stat().st_size/1e6,2),'new_api_calls=0',flush=True)

if __name__=='__main__':main()
