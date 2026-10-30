"""Plot verified observations from the offline 15-method integration."""
import argparse
import csv
import importlib.metadata
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ner_experiment.common import ROOT, file_hash, load_json, save_json

COLORS={'deepseek':'#366A9B','glm':'#B76B30','mimo':'#38806B'}
MARKERS={'deepseek':'o','glm':'s','mimo':'^'}
NAMES={'deepseek':'DeepSeek','glm':'GLM','mimo':'MiMo'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qa-tools-dir',type=Path)
    args=parser.parse_args()
    tables=ROOT/'reports/tables/integrated_v1';out=ROOT/'reports/figures/integrated_v1'
    out.mkdir(parents=True,exist_ok=True)
    def read(name):
        with (tables/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    rows=read('main_results.csv');subset=read('network_subset.csv');cost=read('cost_latency.csv')
    bootstrap=load_json(tables/'bootstrap.json')
    if len(rows)!=15 or len({r['method'] for r in rows})!=15 or any(int(r['samples'])!=1209 for r in rows):
        raise ValueError('Require all 15 methods and all 1209 Test sentences')
    if len({r['samples'] for r in subset})!=1 or [r['method'] for r in subset]!=[r['method'] for r in rows]:
        raise ValueError('Shared subset does not contain the same method set')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'axes.titlesize':11,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
    alignment=None
    if args.qa_tools_dir:
        sys.path.insert(0,str(args.qa_tools_dir))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        alignment=require_matplotlib_panel_alignment
    figures=[]
    def export(fig,name):
        fig.canvas.draw()
        if alignment:
            alignment(fig,json_out=str(out/f'{name}.alignment.json'),overlay_svg=str(out/f'{name}.alignment.svg'),
                      tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
        else:save_json(out/f'{name}.alignment.json',{'status':'NOT APPLICABLE','reason':'single data axis','axes':len(fig.axes)})
        fig.savefig(out/f'{name}.png',dpi=300,facecolor='white')
        fig.savefig(out/f'{name}.svg',facecolor='white')
        fig.savefig(out/f'{name}.pdf',facecolor='white')
        plt.close(fig);figures.append(name)
    def base(height=4.4,left=.14,bottom=.23,top=.82):
        fig,ax=plt.subplots(figsize=(7.2,height));fig.subplots_adjust(left=left,right=.97,bottom=bottom,top=top)
        ax.grid(axis='x' if height>5 else 'y',color='#DDDDDD',linewidth=.5);ax.set_axisbelow(True)
        return fig,ax
    fig,ax=base(7.0,.25,.14,.88);y=np.arange(15)
    for level,offset,color,marker,label in [('token',-.13,'#366A9B','o','Character BIO F1'),('entity',.13,'#B76B30','s','Strict entity F1')]:
        point=np.array([float(r[f'{level}_f1'])*100 for r in rows])
        lo=[float(r[f'{level}_f1_ci_low'])*100 for r in rows];hi=[float(r[f'{level}_f1_ci_high'])*100 for r in rows]
        ax.hlines(y+offset,lo,hi,color=color,linewidth=1.1)
        ax.scatter(point,y+offset,color=color,marker=marker,s=21,label=label,zorder=3)
    ax.set_yticks(y,[r['display'] for r in rows]);ax.invert_yaxis();ax.set_xlim(43,84);ax.set_xlabel('F1 (%)')
    ax.set_title('All fixed Test predictions: two evaluation levels',pad=14)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.97,.98),ncol=2)
    fig.text(.25,.055,'n = 1,209; 95% sentence-bootstrap intervals; failures retained.',fontsize=8)
    export(fig,'main_results')
    for metric,name,title in [('entity','entity_shot_curve','Strict entity F1 by shot count'),
                              ('token','token_shot_curve','Character BIO F1 by shot count'),
                              ('subset_entity','network_subset_curve','Shared network-available subset (diagnostic)')]:
        source=subset if metric.startswith('subset') else rows
        fig,ax=base();ax.grid(axis='x',visible=False)
        for family in COLORS:
            rr=[r for r in source if r['family']==family];x=np.array([int(r['shots']) for r in rr])
            yy=np.array([float(r[f'{metric}_f1'])*100 for r in rr])
            lo=[float(r[f'{metric}_f1_ci_low'])*100 for r in rr];hi=[float(r[f'{metric}_f1_ci_high'])*100 for r in rr]
            ax.plot(x,yy,color=COLORS[family],marker=MARKERS[family],markersize=5,linewidth=1.2,label=NAMES[family])
            ax.vlines(x,lo,hi,color=COLORS[family],linewidth=1)
        ax.set_xticks([0,2,4,8]);ax.set_xlim(-.4,8.4);ax.set_ylim((67,83) if metric=='token' else (60,77))
        ax.set_xlabel('Number of in-context examples');ax.set_ylabel('F1 (%)');ax.set_title(title,pad=14)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.97,.96),ncol=3)
        if metric.startswith('subset'):
            n=int(subset[0]['samples']);fig.text(.14,.10,f'Same {n:,} sentences for all methods; nonrandom, post hoc subset.',fontsize=8)
            fig.text(.14,.05,'Not a repaired full-Test estimate; 95% fixed-prediction intervals.',fontsize=8)
        else:
            fig.text(.14,.10,'Full Test n = 1,209; GLM 4-shot includes 117 connection failures.',fontsize=8)
            fig.text(.14,.05,'One nested example pool; content/coverage/length change with shot count.',fontsize=8)
        export(fig,name)
    pairs=[p for p in bootstrap['comparisons'] if p['b']=='crf']
    fig,ax=base(6,.25,.16,.88)
    display={r['method']:r['display'] for r in rows}
    for i,p in enumerate(pairs):
        d=p['entity'];lo,hi=np.array(d['ci95'])*100;value=d['difference_a_minus_b']*100
        family=p['a'].split('_')[0]
        ax.hlines(i,lo,hi,color=COLORS[family],linewidth=1.4)
        ax.scatter(value,i,color=COLORS[family],marker=MARKERS[family],s=24,zorder=3)
    ax.axvline(0,color='#888888',linestyle='--',linewidth=.8)
    ax.set_yticks(range(len(pairs)),[display[p['a']] for p in pairs]);ax.invert_yaxis();ax.set_xlim(-8,8)
    ax.set_xlabel('Strict entity F1 difference versus CRF (percentage points)')
    ax.set_title('Sentence-paired differences on the full Test',pad=14)
    fig.text(.25,.065,'95% percentile intervals; 2,000 paired draws; no multiplicity correction.',fontsize=7.5)
    export(fig,'paired_differences')
    fig,ax=base();ax.grid(axis='x',visible=False)
    for family in COLORS:
        rr=[r for r in cost if r['family']==family]
        ax.plot([int(r['shots']) for r in rr],[float(r['known_usage_estimate_cny']) for r in rr],
                color=COLORS[family],marker=MARKERS[family],markersize=5,linewidth=1.2,label=NAMES[family])
    ax.set_xticks([0,2,4,8]);ax.set_xlim(-.4,8.4);ax.set_ylim(0,4)
    ax.set_xlabel('Number of in-context examples');ax.set_ylabel('Known-usage estimate (CNY / 1,209 sentences)')
    ax.set_title('Recorded API usage costs by shot count',pad=14)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.97,.96),ncol=3)
    fig.text(.14,.10,'Historical recorded rates, not invoices; unknown GLM reserves excluded.',fontsize=8)
    fig.text(.14,.05,'Cache hits and inference modes differ; no cost confidence intervals.',fontsize=8)
    export(fig,'cost_curve')
    save_json(out/'figure_manifest.json',{'backend':'Python/matplotlib','figures':figures,
        'source_files':{(tables/f).relative_to(ROOT).as_posix():file_hash(tables/f) for f in
                        ['main_results.csv','network_subset.csv','cost_latency.csv','bootstrap.json','network_subset_ids.json']},
        'plot_script_sha256':file_hash(Path(__file__).resolve()),'samples':1209,
        'network_subset_samples':int(subset[0]['samples']),'repeats':bootstrap['repeats'],'seed':bootstrap['seed'],
        'uncertainty':'Sentence-paired percentile 95%; fixed predictions; exploratory; uncorrected',
        'exclusions_from_primary':[],'outputs':'Editable SVG, PDF, PNG 300dpi preview; TIFF not required for course report',
        'final_width_inches':7.2,'versions':{p:importlib.metadata.version(p) for p in ['numpy','matplotlib','pillow']}})
    print('EXPORTED',len(figures),'figures')


if __name__=='__main__':main()
