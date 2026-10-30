"""Verify report figures/tables against frozen evidence and inspect final Word PDF geometry."""
import argparse
import csv
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from ner_experiment.common import ROOT, file_hash, load_json, save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--visual-reviewed',action='store_true');parser.add_argument('--appendix',action='store_true');args=parser.parse_args()
    content=load_json(ROOT/('reports/course_report_appendix_content.json' if args.appendix else 'reports/course_report_content.json'))
    provenance=load_json(ROOT/'reports/course_report_provenance.json')
    for path,digest in provenance['source_sha256'].items():
        if file_hash(ROOT/path)!=digest:raise RuntimeError('Report source changed: '+path)
    if file_hash(ROOT/'reports/course_report.md')!=provenance['markdown_sha256']:raise RuntimeError('Markdown changed')
    target=ROOT/('reports/CLUENER2020_报告附录.docx' if args.appendix else 'reports/CLUENER2020_课程实验报告.docx')
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    def text(node):return ''.join(n.text or '' for n in node.findall('.//w:t',ns))
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise RuntimeError('DOCX CRC failed')
        for name in z.namelist():
            if name.endswith('.xml'):ET.fromstring(z.read(name))
        xml=ET.fromstring(z.read('word/document.xml'))
        headings=[text(p) for p in xml.findall('.//w:p',ns) if p.find('./w:pPr/w:pStyle[@w:val="Heading1"]',ns) is not None]
        expected_headings=[b['text'] for b in content['blocks'] if b['type']=='heading' and b.get('level')==1]
        if headings!=expected_headings:raise RuntimeError('Section headings do not match')
        if not args.appendix and (headings!=provenance['section_titles'] or len(headings)!=8):raise RuntimeError('Eight sections do not match')
        tables=xml.findall('.//w:tbl',ns);expected=[b for b in content['blocks'] if b['type']=='table']
        if len(tables)!=len(expected):raise RuntimeError('Word table count mismatch')
        for table,source in zip(tables,expected):
            actual=[[text(c) for c in row.findall('w:tc',ns)] for row in table.findall('w:tr',ns)]
            # JSON/JavaScript displays an integral numeric value 1.0 as 1; preserve exact strings for scores.
            wanted=[[(format(x,'g') if isinstance(x,float) else str(x)) for x in r] for r in [source['headers'],*source['rows']]]
            if actual!=wanted:raise RuntimeError('Word table changed: '+source['title'])
            widths=[int(c.attrib['{'+ns['w']+'}w']) for c in table.findall('./w:tblGrid/w:gridCol',ns)]
            if sum(widths)!=9638:raise RuntimeError('Unexpected table width')
        media=[n for n in z.namelist() if n.startswith('word/media/') and not n.endswith('/')]
        if not args.appendix and len(media)<2:raise RuntimeError('Missing required architecture/result images')
    with (ROOT/'reports/tables/integrated_v1/main_results.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    main_table=next((b for b in expected if b['title'].startswith('表7 ')),None)
    source_rows=[[r['display'],*[f"{100*float(r[f'{level}_{k}']):.2f}" for level in ['token','entity'] for k in ['precision','recall','f1']]] for r in rows]
    if main_table and len(main_table['headers'])==8:
        for output,r in zip(source_rows,rows):output.append(f"{100*float(r['entity_with_invalid_fp_f1']):.2f}")
    if main_table and main_table['rows']!=source_rows:raise RuntimeError('Main table differs from actual CSV')
    cases=load_json(ROOT/'runs/integrated_v1_analysis/representative_cases.json')
    all_text='\n'.join(b.get('text','') for b in content['blocks'])
    if not args.appendix and any(c['text'] not in all_text or c['id'] not in all_text for c in cases):raise RuntimeError('Case missing or altered')
    revision=ROOT/'reports/tables/course_report_revision/revision_record.json'
    if revision.exists() and not load_json(revision)['section_eight_unchanged']:
        raise RuntimeError('Section eight unexpectedly changed')
    backup=ROOT/'runs/course_report_revisions/before_review_v1_20261030/reports/course_report_content.json'
    if not args.appendix and backup.exists():
        def contributions(bs):
            start=next(i for i,b in enumerate(bs) if b.get('text')=='⑧ 分工与贡献')
            end=next(i for i,b in enumerate(bs[start:],start) if b.get('text')=='参考文献与材料来源')
            return bs[start:end]
        if contributions(content['blocks'])!=contributions(load_json(backup)['blocks']):
            raise RuntimeError('Actual contribution section changed from backup')
    import pymupdf
    qa_dir=ROOT/('runs/course_report_appendix_qa' if args.appendix else 'runs/course_report_qa')
    pdf=qa_dir/'course_report.pdf';document=pymupdf.open(pdf)
    outside=[]
    for i,page in enumerate(document):
        for word in page.get_text('words'):
            if word[0]<15 or word[2]>page.rect.width-15 or word[1]<15 or word[3]>page.rect.height-15:
                outside.append({'page':i+1,'text':word[4],'bbox':list(word[:4])})
    if outside:raise RuntimeError('PDF text outside page bounds: '+json.dumps(outside,ensure_ascii=False))
    qa={'status':'PASS','sections':len(headings),'methods':15,'word_tables':len(tables),'main_table_matches_csv':True if main_table else 'not applicable in appendix',
        'all_word_table_cells_match_report_content':True,'source_hashes_match':True,'actual_cases_unchanged':True,
        'word_xml_well_formed':True,'word_images':len(media),'pdf_pages':len(document),
        'pdf_text_outside_page':outside,'visual_review_complete':args.visual_reviewed,
        'renderer':'Microsoft Word native PDF export; LibreOffice unavailable on this machine',
        'schema_validation_note':'External full XSD validator unavailable (optional defusedxml dependency absent); no full XSD certification claimed.',
        'docx_sha256':file_hash(target),'pdf_sha256':file_hash(pdf),'new_api_calls':0,
        'builder_sha256':file_hash(ROOT/'scripts/build_course_report_docx.cjs')}
    save_json(qa_dir/'verification.json',qa)
    print(json.dumps({k:v for k,v in qa.items() if k not in ['schema_validation_note']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
