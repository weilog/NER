"""Python/matplotlib figures from saved V3 source data; no model calls or simulated observations."""
import argparse
import csv
import importlib.metadata
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ner_experiment.common import ROOT,load_json,save_json,file_hash

BLUE='#366A9B'
ORANGE='#B76B30'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qa-tools-dir',type=Path)
    args=parser.parse_args()
    tables=ROOT/'reports/tables/current_v3';out=ROOT/'reports/figures/current_v3';out.mkdir(parents=True,exist_ok=True)
    with (tables/'main_results.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    bootstrap=load_json(tables/'bootstrap.json')
    if len(rows)!=7 or any(int(r['samples'])!=1209 for r in rows):raise ValueError('Expected all seven methods and all Test sentences')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'axes.titlesize':11,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':9,
        'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
    qa_alignment=None
    if args.qa_tools_dir:
        sys.path.insert(0,str(args.qa_tools_dir))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        qa_alignment=require_matplotlib_panel_alignment
    def export(fig,name):
        fig.canvas.draw()
        if qa_alignment:
            qa_alignment(fig,json_out=str(out/f'{name}.alignment.json'),overlay_svg=str(out/f'{name}.alignment.svg'),
                         tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
        else:save_json(out/f'{name}.alignment.json',{'status':'NOT APPLICABLE','reason':'single data axis','axes':len(fig.axes)})
        # Do not crop to a different final physical size; margins are explicit.
        fig.savefig(out/f'{name}.png',dpi=300,facecolor='white')
        fig.savefig(out/f'{name}.svg',facecolor='white')
        fig.savefig(out/f'{name}.pdf',facecolor='white')
        plt.close(fig)

    # Fig 1: all fixed observations, same percentile uncertainty for both metrics.
    fig,ax=plt.subplots(figsize=(7.2,4.7));fig.subplots_adjust(left=.22,right=.97,bottom=.19,top=.84)
    y=np.arange(len(rows))
    for level,offset,color,marker,label in [('token',-.12,BLUE,'o','Character BIO-set F1'),('entity',.12,ORANGE,'s','Strict entity F1')]:
        point=np.array([float(r[f'{level}_f1'])*100 for r in rows])
        lo=np.array([float(r[f'{level}_f1_ci_low'])*100 for r in rows]);hi=np.array([float(r[f'{level}_f1_ci_high'])*100 for r in rows])
        # Explicit interval endpoints also work if a percentile CI excludes its point estimate.
        ax.hlines(y+offset,lo,hi,color=color,linewidth=1.2)
        ax.scatter(point,y+offset,color=color,marker=marker,s=24,label=label,zorder=3)
    ax.set_yticks(y,[r['display'] for r in rows]);ax.invert_yaxis()
    ax.set_xlim(40,90);ax.set_xlabel('F1 (%)');ax.set_title('Fixed Test predictions: two evaluation levels',pad=12)
    ax.grid(axis='x',color='#DDDDDD',linewidth=.5);ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.97,.98),ncol=2)
    fig.text(.22,.07,'n = 1,209 sentences; 95% sentence-bootstrap intervals (2,000 draws).',fontsize=8)
    export(fig,'main_results')

    # Fig 2: paired differences are the inferential quantity, not overlap of marginal intervals.
    pair_rows=[r for r in bootstrap['comparisons'] if r['b']=='crf']
    fig,ax=plt.subplots(figsize=(7.2,3.6));fig.subplots_adjust(left=.22,right=.96,bottom=.26,top=.80)
    for i,pair in enumerate(pair_rows):
        score=pair['entity'];lo,hi=np.array(score['ci95'])*100;value=score['difference_a_minus_b']*100
        ax.hlines(i,lo,hi,color=BLUE,linewidth=1.8);ax.scatter([value],[i],color=BLUE,s=30,zorder=3)
    ax.axvline(0,color='#888888',linestyle='--',linewidth=.8)
    ax.set_yticks(range(len(pair_rows)),[r['a'].replace('llm_','LLM ')+'-shot' for r in pair_rows]);ax.invert_yaxis()
    all_ends=[v*100 for r in pair_rows for v in r['entity']['ci95']]
    ax.set_xlim(min(-3,min(all_ends)-1),max(4,max(all_ends)+1))
    ax.set_xlabel('Strict entity F1 difference versus CRF (percentage points)')
    ax.set_title('Sentence-paired differences versus CRF',pad=12)
    ax.grid(axis='x',color='#DDDDDD',linewidth=.5);ax.set_axisbelow(True)
    fig.text(.22,.10,'95% percentile intervals; exploratory, no multiplicity adjustment.',fontsize=8)
    fig.text(.22,.05,'Fixed predictions only; 1,209 paired sentences, 2,000 shared draws.',fontsize=8)
    export(fig,'paired_differences')

    # Fig 3: only observed shot counts; no interpolation or extrapolation.
    llm=[r for r in rows if r['method'].startswith('llm_')];x=np.array([int(r['method'].split('_')[1]) for r in llm])
    fig,ax=plt.subplots(figsize=(7.2,4.2));fig.subplots_adjust(left=.14,right=.96,bottom=.23,top=.83)
    for level,color,marker,label in [('token',BLUE,'o','Character BIO-set F1'),('entity',ORANGE,'s','Strict entity F1')]:
        point=np.array([float(r[f'{level}_f1'])*100 for r in llm])
        lo=np.array([float(r[f'{level}_f1_ci_low'])*100 for r in llm]);hi=np.array([float(r[f'{level}_f1_ci_high'])*100 for r in llm])
        ax.plot(x,point,color=color,marker=marker,markersize=5,linewidth=1.2,label=label)
        ax.vlines(x,lo,hi,color=color,linewidth=1.1)
    ax.set_xticks([0,2,4,8]);ax.set_xlim(-.4,8.4);ax.set_ylim(60,84)
    ax.set_xlabel('Number of in-context examples');ax.set_ylabel('F1 (%)')
    ax.set_title('Shot effects within one fixed nested example pool',pad=12)
    ax.grid(axis='y',color='#DDDDDD',linewidth=.5);ax.set_axisbelow(True);ax.legend(loc='upper left')
    fig.text(.14,.10,'Same 1,209 Test sentences; 95% sentence-bootstrap intervals.',fontsize=8)
    fig.text(.14,.05,'Example content, class coverage and length vary with shot count.',fontsize=8)
    export(fig,'shot_curve')
    versions={p:importlib.metadata.version(p) for p in ['numpy','matplotlib','pillow']}
    try:versions['PyMuPDF']=importlib.metadata.version('PyMuPDF')
    except importlib.metadata.PackageNotFoundError:pass
    save_json(out/'figure_manifest.json',{'backend':'Python/matplotlib','source_files':{
        str((tables/name).relative_to(ROOT)):file_hash(tables/name) for name in ['main_results.csv','bootstrap.json','bootstrap_comparisons.csv']},
        'plot_script_sha256':file_hash(Path(__file__)),'samples':1209,'repeats':2000,'seed':bootstrap['seed'],
        'exclusions':[],'statistic':'micro F1; same sentence draw for each method/metric',
        'uncertainty':'95% percentile; no training/API/example-pool uncertainty; pairwise uncorrected exploratory intervals',
        'figures':['main_results','paired_differences','shot_curve'],'versions':versions,'final_width_inches':7.2})
    print('PLOTTED three single-axis figures in PNG/SVG/PDF; API calls=0')

if __name__=='__main__':main()
