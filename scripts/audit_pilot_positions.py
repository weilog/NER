"""Re-examine the two historical pilots without rerunning or altering them."""
import json
from pathlib import Path
from ner_experiment.data.splitting import read_jsonl, json_bytes
from ner_experiment.evaluation.metrics import evaluate
from ner_experiment.evaluation.index_diagnostics import diagnose
from ner_experiment.experiments.traditional import atomic_write


def main(root):
    root = Path(root)
    gold = read_jsonl(root / 'data/processed/experiment_v2/dev_pilot.jsonl')
    results = {}
    for version, name in [('v1', 'pilot_20_zero_shot'), ('v2', 'pilot_20_zero_shot_v2')]:
        folder = root / 'runs/llm' / name
        predicted = read_jsonl(folder / 'predictions.jsonl')
        score = evaluate(gold, predicted)
        score.pop('normalized_predictions')
        if score != json.loads((folder / 'metrics.json').read_text(encoding='utf-8')):
            raise ValueError('Historical pilot offline scores changed')
        results[version] = {'token': score['token'], 'entity': score['entity'],
                            'diagnostics': diagnose(gold, predicted)}
    atomic_write(root / 'runs/llm/index_audit/pilot-audit.json', json_bytes(results))
    lines = ['# 历史试跑定位复核', '',
             '两版原始预测离线重算均与历史指标完全一致；没有 API 请求，也没有修改旧运行目录。', '',
             '| 版本 | 主实体 F1 (%) | 文字+类别 bag F1 (%) | 坐标错误句子 (%) | 闭区间终点特征条目 | 唯一文字可定位条目 |',
             '|---|---:|---:|---:|---:|---:|']
    for version, item in results.items():
        d = item['diagnostics']; c = d['counts']
        lines.append(f'| {version} | {item["entity"]["f1"]*100:.2f} | {d["surface_type_bag"]["f1"]*100:.2f} | {d["position_error_sample_rate"]*100:.2f} | {c.get("inclusive_end_signature_entities",0)} | {c.get("unique_surface_relocatable_entities",0)} |')
    lines += ['', '文字+类别 bag 忽略坐标，只是诊断指标，不能替代 NER 主指标。唯一可定位只是原文中仅一次出现，不代表类型或语义一定正确。',
              '历史两版同时改了索引展示和说明，且 v2 根据这 20 条 Dev 调整，不能将提升归因于一个因素或当作独立验证。', '',
              '后续 2×2 实验固定 4-shot 和另一批 Dev，分别操纵索引表示及终点解释，保留所有不利结果。', '']
    atomic_write(root / 'reports/pilot-position-audit.md', '\n'.join(lines).encode('utf-8'))
    print(json.dumps({k: {"entity_f1": v['entity']['f1'], 'diagnostics': v['diagnostics']['counts']} for k,v in results.items()}, indent=2))


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
