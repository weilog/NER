"""Offline demo evidence registry, strict validation, and source-grounded case analysis."""
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
from ner_experiment.common import file_hash, load_json, read_jsonl

class DemoError(RuntimeError):
    pass

def require(condition, message):
    if not condition:
        raise DemoError(message)

def local_path(root, relative):
    root=Path(root).resolve()
    require(isinstance(relative,str) and relative, f'Invalid project path: {relative!r}')
    path=(root/relative).resolve()
    require(path.is_relative_to(root), f'Path outside project: {relative}')
    return path

def unique(items, key, label):
    result={}
    for item in items:
        identity=key(item)
        require(identity not in result, f'Duplicate {label}: {identity!r}; refusing overwrite')
        result[identity]=item
    return result

def checked_hash(path, expected):
    require(path.is_file(), f'Missing file: {path}')
    actual=file_hash(path)
    require(actual==expected, f'SHA256 mismatch: {path}; actual={actual}, reference={expected}')

def checked_rows(root, relative, expected):
    path=local_path(root,relative)
    content=path.read_bytes()
    actual=hashlib.sha256(content).hexdigest()
    require(actual==expected, f'SHA256 mismatch: {relative}; actual={actual}, reference={expected}')
    return [json.loads(line) for line in content.decode('utf-8-sig').splitlines() if line.strip()]

def count_value(raw, context):
    try:
        value=Decimal(str(raw))
    except InvalidOperation as exc:
        raise DemoError(f'{context}: invalid integer count {raw!r}') from exc
    require(value.is_finite() and value>=0 and value==value.to_integral_value(),
            f'{context}: invalid integer count {raw!r}')
    return int(value)

def score_tolerance(raw):
    try:
        value=Decimal(str(raw))
    except InvalidOperation as exc:
        raise DemoError(f'Invalid reference score: {raw!r}') from exc
    require(value.is_finite() and 0<=value<=1, f'Reference score out of range: {raw!r}')
    # Exact zero/one integer spellings do not imply a 0.5 rounding allowance.
    quantum=Decimal(10)**value.as_tuple().exponent
    return max(1e-12, float(quantum/2)) if value.as_tuple().exponent<0 else 1e-12

def compare_metrics(method, actual, row):
    for level in ['token','entity']:
        for key in ['tp','fp','fn','precision','recall','f1']:
            name=f'{level}_{key}'
            require(name in row, f'{method}: missing reference column {name}')
            reference=row[name];value=actual[level][key]
            if key in {'tp','fp','fn'}:
                wanted=count_value(reference, f'{method} {level}.{key}')
                require(type(value) is int and value==wanted,
                        f'{method} {level}.{key}: actual={value!r}, reference={reference!r}; exact integer equality required')
            else:
                tolerance=score_tolerance(reference)
                require(math.isfinite(value) and math.isclose(value,float(reference),rel_tol=0,abs_tol=tolerance),
                        f'{method} {level}.{key}: actual={value!r}, reference={reference!r}, abs_tol={tolerance:g}')

def read_table(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    return rows,unique(rows,lambda r:r['method'],'result-table method')

def recovery_mapping(root, audit, protocol, summary):
    """Audit lacks shot: join its digest to real directories, summaries and protocol."""
    base=Path(root)/'runs/network_recovery_20261030'
    require(audit.get('status')=='PASS' and summary.get('status')=='COMPLETE','Recovery not completely audited')
    declarations=unique(protocol['groups'],lambda r:(r['family'],r['shots']),'recovery protocol configuration')
    completed=unique(summary['groups'],lambda r:(r['family'],r['shots']),'completed recovery configuration')
    discovered=[]
    for path in sorted(base.glob('*/summary.json')):
        info=load_json(path);family=info.get('family');shots=info.get('shots')
        require(isinstance(family,str) and type(shots) is int and shots>=0,f'Invalid recovery metadata: {path}')
        relative=(path.parent/'merged_predictions.jsonl').relative_to(root).as_posix()
        discovered.append({'family':family,'shots':shots,'path':relative,'summary_path':path.relative_to(root).as_posix(),
                           'sha256':file_hash(local_path(root,relative)), 'info':info})
    unique(discovered,lambda r:(r['family'],r['shots']),'recovery directory configuration')
    mapping={}
    for audited in audit['groups']:
        candidates=[r for r in discovered if r['family']==audited['family'] and r['sha256']==audited['merged_predictions_sha256']]
        require(len(candidates)==1,f'Cannot uniquely resolve recovery family/digest: {audited.get("family")}')
        chosen=candidates[0];key=(chosen['family'],chosen['shots'])
        require(key not in mapping,f'Duplicate audited recovery configuration: {key}; refusing overwrite')
        require(key in declarations and key in completed,f'Recovery configuration absent from protocol/summary: {key}')
        for info in [chosen['info'],completed[key]]:
            require(info['full_test_samples']==audited['samples'] and info['recovered_ok']==audited['recovered'],
                    f'Recovery counts disagree: {key}')
        mapping[key]=chosen
    require(set(mapping)==set(declarations)==set(completed),'Unmatched recovery configurations')
    return mapping

def build_snapshot(root):
    from ner_experiment.evaluation.metrics_v1 import evaluate
    root=Path(root).resolve();files={};provenance={}
    def track(relative, expected=None, anchor='local artifact after verification'):
        path=local_path(root,relative)
        require(path.is_file(),f'Missing input: {relative}')
        digest=file_hash(path)
        if expected is not None:require(digest==expected,f'Independent SHA256 mismatch: {relative}; actual={digest}, reference={expected}')
        if relative in files:require(files[relative]==digest,f'Input changed while freezing: {relative}')
        files[relative]=digest;provenance[relative]=anchor
        return digest
    legacy_name='runs/integrated_v1_analysis/verification.json'
    old=load_json(root/legacy_name);track(legacy_name)
    source=old['source_hashes']
    data_name='data/splits/cluener2020/manifest.json';data=load_json(root/data_name)
    track(data_name,source.get(data_name))
    for item in data['files'].values():track(item['path'],item['sha256'],data_name)
    config_name='configs/experiment.json';track(config_name,source.get(config_name));config=load_json(root/config_name)
    test_path=config['dataset']['test_file']
    require(test_path in files,'Configured Test not covered by data SHA256 manifest')
    gold=checked_rows(root,test_path,files[test_path]);gold_map=unique(gold,lambda r:r['id'],'gold ID')
    require(len(gold)==1209,'Expected fixed 1209-sentence Test')
    for r in gold:require(r['tokens']==list(r['text']),f'Token/text mismatch: {r["id"]}')
    rec='runs/network_recovery_20261030/'
    for name in ['verification.json','protocol.json','summary.json']:track(rec+name)
    audit=load_json(root/(rec+'verification.json'));proto=load_json(root/(rec+'protocol.json'))['protocol'];done=load_json(root/(rec+'summary.json'))
    overlays=recovery_mapping(root,audit,proto,done)
    for entry in overlays.values():track(entry['summary_path'])
    table_path=audit['new_15_method_table'];table_rows,expected=read_table(local_path(root,table_path));track(table_path)
    require(len(table_rows)==15 and [r['method'] for r in table_rows]==old['methods'],'Final table does not match the 15 recorded methods/order')
    base_entries=[]
    for method in ['dictionary','hmm','crf']:
        base_entries.append({'method':method,'family':method,'shots':None,'original_path':f'runs/final/{method}/predictions.jsonl'})
    for family,folder in [('deepseek','llm_text_test_v1'),('glm','glm_flashx_test_v1'),('mimo','mimo_flash_test_v1')]:
        freeze_name=f'runs/{folder}/frozen_protocol.json';track(freeze_name,source.get(freeze_name));freeze=load_json(root/freeze_name)['protocol']
        shots=freeze['shots'];require(all(type(s) is int and s>=0 for s in shots),f'Invalid shots in {freeze_name}')
        for shot in shots:
            base_entries.append({'method':f'{family}_{shot}','family':family,'shots':shot,'original_path':f'runs/{folder}/shot_{shot}/predictions.jsonl'})
    unique(base_entries,lambda e:(e['family'],e['shots']),'base configuration')
    unique(base_entries,lambda e:e['method'],'base method')
    require([e['method'] for e in base_entries]==old['methods'],'Prediction method/order differs from frozen source list')
    used=set();entries=[]
    for base in base_entries:
        method=base['method'];original=base['original_path']
        require(original in source,f'No independent prediction SHA256: {method} {original}')
        track(original,source[original],legacy_name)
        entry=dict(base);key=(entry['family'],entry['shots'])
        if key in overlays:
            recovered=overlays[key];entry.update(prediction_path=recovered['path'],prediction_sha256=recovered['sha256'],recovered=True)
            track(entry['prediction_path'],entry['prediction_sha256'],rec+'verification.json');used.add(key)
        else:entry.update(prediction_path=original,prediction_sha256=files[original],recovered=False)
        row=expected[method]
        require(row.get('evaluation_protocol')=='v1' and count_value(row['samples'],method+' samples')==1209,f'Wrong reference protocol/sample count: {method}')
        predictions=checked_rows(root,entry['prediction_path'],entry['prediction_sha256'])
        pmap=unique(predictions,lambda p:p['id'],method+' prediction ID')
        require(set(pmap)==set(gold_map),f'Missing/extra prediction IDs: {method}')
        metrics=evaluate(gold,predictions)[0];compare_metrics(method,metrics,row)
        entries.append(entry)
    require(used==set(overlays),'Recovery output not used by final registry')
    require(len({e['prediction_path'] for e in entries})==15,'Two configurations point to the same prediction file')
    selection_name='runs/baselines/selection.json';track(selection_name,source[selection_name],legacy_name)
    selection=load_json(root/selection_name);selected=selection['selected']['crf']
    final_name='runs/final/crf/run.json';track(final_name,source[final_name],legacy_name)
    require(load_json(root/final_name)['selected_run']==selected,'CRF selection differs from original Test run')
    meta_name=f'runs/baselines/{selected}/run.json';track(meta_name);meta=load_json(root/meta_name)
    selected_info=unique(selection['runs'],lambda r:r['name'],'Dev run')[selected]
    require(meta['method']=='crf' and meta['model_sha256']==selected_info['model_sha256'],'CRF model metadata disagrees with Dev selection')
    model_path=f'runs/baselines/{selected}/model.pkl';track(model_path,meta['model_sha256'],selection_name+' + '+meta_name)
    frozen_name='configs/frozen_experiment.json';track(frozen_name);frozen=load_json(root/frozen_name)['snapshot']['files']
    baseline='src/ner_experiment/models/baselines.py';track(baseline,frozen[baseline],frozen_name)
    for relative in ['src/ner_experiment/evaluation/metrics_v1.py','src/ner_experiment/common.py']:
        expected_hash=old['analysis_source_hashes'].get(relative)
        require(expected_hash is not None,f'No evaluator source SHA256: {relative}')
        track(relative,expected_hash,legacy_name)
    for relative in ['scripts/demo_offline.py','scripts/demo_evidence.py','scripts/freeze_demo_inputs.py','scripts/run_baselines.py','scripts/run_final.py']:track(relative)
    return {'schema':'final-demo-inputs-1','scope':'Post-experiment integrity snapshot, not a pre-Test preregistration',
            'test_path':test_path,'samples':1209,'reference_table':table_path,'methods':entries,
            'crf_model_path':model_path,'crf_selected_run':selected,'files_sha256':files,'hash_provenance':provenance,
            'verification':'All 15 predictions replayed through metrics_v1; exact counts and CSV precision-aware probability comparison',
            'new_api_calls':0}

def verify_snapshot(root, snapshot):
    require(snapshot.get('schema')=='final-demo-inputs-1','Unknown demo manifest schema')
    for relative,digest in snapshot['files_sha256'].items():checked_hash(local_path(root,relative),digest)
    entries=snapshot['methods']
    require(len(entries)==15,'Demo manifest must contain exactly 15 methods')
    unique(entries,lambda e:e['method'],'manifest method')
    unique(entries,lambda e:(e['family'],e['shots']),'manifest configuration')
    require(len({e['prediction_path'] for e in entries})==15,'Duplicate prediction paths in manifest')
    for entry in entries:
        require(snapshot['files_sha256'].get(entry['prediction_path'])==entry['prediction_sha256'],f'Prediction absent from SHA256 inventory: {entry["method"]}')
    return snapshot

def load_predictions(root, entry, gold_ids):
    rows=checked_rows(root,entry['prediction_path'],entry['prediction_sha256'])
    lookup=unique(rows,lambda r:r['id'],entry['method']+' prediction ID')
    require(set(lookup)==set(gold_ids),f'Missing/extra prediction IDs: {entry["method"]}')
    return rows,lookup

def case_error(gold, predicted, kind):
    from ner_experiment.evaluation.metrics_v1 import evaluate
    if predicted.get('status','ok')!='ok':return None
    metrics,_,cases=evaluate([gold],[predicted])
    errors=[e for c in cases for e in c['errors'] if e['kind']==kind]
    for error in errors:
        g,p=error['gold'],error['predicted']
        if g is None or p is None:continue
        if kind=='boundary':
            valid=g[2]==p[2] and g[:2]!=p[:2] and max(g[0],p[0])<min(g[1],p[1])
        elif kind=='type':valid=g[:2]==p[:2] and g[2]!=p[2]
        else:raise DemoError(f'Unsupported case kind: {kind}')
        if valid:return {'kind':kind,'row':gold,'prediction':predicted,'pair':error,'metrics':metrics}
    return None

def select_case(root, entry, gold_map, preferred_id, kind):
    _,predicted=load_predictions(root,entry,gold_map)
    require(preferred_id in gold_map,f'Preferred case absent from Test: {preferred_id}')
    chosen=case_error(gold_map[preferred_id],predicted[preferred_id],kind)
    fallback=chosen is None
    if fallback:
        candidates=sorted(gold_map,key=lambda i:(len(gold_map[i]['entities']),len(gold_map[i]['text']),i))
        chosen=next((c for i in candidates if (c:=case_error(gold_map[i],predicted[i],kind)) is not None),None)
    require(chosen is not None,f'No verified {kind} error in final predictions: {entry["method"]}')
    return {**chosen,'method':entry['method'],'source_path':entry['prediction_path'],
            'source_sha256':entry['prediction_sha256'],'preferred_id':preferred_id,'used_alternative':fallback}
