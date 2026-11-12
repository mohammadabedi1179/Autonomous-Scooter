#!/usr/bin/env python3
"""Interactive animated Shield Type Distribution (fraction area) plot.
Tiles: Episode | Caution fraction | Critical fraction | Stage history of critical fraction
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from interactive_base import *

def build_data(episodes, w, ctrans=None):
    ep, ca, da, cr, stages = [], [], [], [], []
    for i, e in enumerate(episodes):
        ep.append(int(safe_float(e.get('episode', i+1)) or (i+1)))
        ca.append(safe_float(e.get('shield_interventions_caution_ep')) or 0)
        da.append(safe_float(e.get('shield_interventions_danger_ep')) or 0)
        cr.append(safe_float(e.get('shield_interventions_critical_ep')) or 0)
        try: stages.append(int(float(e.get('curriculum_stage', 0))))
        except: stages.append(0)
    total = [a+b+c for a,b,c in zip(ca, da, cr)]
    def safe_frac(v, t): return round(v/t, 6) if t > 0 else 0
    fca = [safe_frac(a, t) for a, t in zip(ca, total)]
    fda = [safe_frac(d, t) for d, t in zip(da, total)]
    fcr = [safe_frac(c, t) for c, t in zip(cr, total)]
    rfca = rolling_mean(fca, w)
    rfda = rolling_mean(fda, w)
    rfcr = rolling_mean(fcr, w)
    transitions, stage_zones, stage_labels = build_stage_data(ep, stages, ctrans)
    completed = build_completed_avgs(ep, transitions, rfcr)
    return {
        'episodes': ep, 'fca': fca, 'fda': fda, 'fcr': fcr,
        'rfca': rfca, 'rfda': rfda, 'rfcr': rfcr,
        'stageLabels': stage_labels, 'completedStageAvgs': completed,
        'stageZones': stage_zones, 'transitions': transitions,
        'yRange': [-0.02, 1.16], 'xRange': [ep[0], ep[-1]],
        'rollingWindow': w, 'total': len(ep),
    }

def generate_html(data):
    dj = json.dumps(data, separators=(',', ':'))
    BL = """{
  paper_bgcolor:'#FAF6F0',plot_bgcolor:'#FAF6F0',
  font:{family:"'DM Sans',sans-serif",color:'#463F3A',size:12},
  margin:{t:38,r:20,b:48,l:66},
  xaxis:{title:{text:'Episode',font:{size:12}},gridcolor:'#E8DDD2',linecolor:'#D8CABD',
    tickcolor:'#6B625B',tickfont:{size:11},zeroline:false,fixedrange:false},
  yaxis:{title:{text:'Fraction of shield types',font:{size:12}},gridcolor:'#E8DDD2',linecolor:'#D8CABD',
    tickcolor:'#6B625B',tickfont:{size:11},range:D.yRange,tickformat:'.0%'},
  legend:{x:.99,y:.99,xanchor:'right',yanchor:'top',bgcolor:'rgba(250,246,240,.92)',
    bordercolor:'#D8CABD',borderwidth:1,font:{size:11}},
  shapes:[],annotations:[],hovermode:'x unified',
  hoverlabel:{bgcolor:'#FAF6F0',bordercolor:'#D8CABD',font:{family:"'DM Mono',monospace",size:11,color:'#463F3A'}},
}"""
    return generate_page_html(
        'Shield Fraction — Interactive', 'Shield Type Distribution Over Training (Fraction)', dj,
        extra_css='',
        tiles_html="""
<div class="tile tile-fixed"><div class="tile-lbl">Episode</div><div class="tile-val" id="v-ep">—</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Caution fraction</div><div class="tile-val" id="v-ca" style="color:#C98245">—</div><div class="tile-sub">rolling</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Critical fraction</div><div class="tile-val" id="v-cr" style="color:#5E6472">—</div><div class="tile-sub">rolling — most severe</div></div>
<div class="tile tile-grow"><div class="tile-lbl">Stage critical avg</div>
  <div id="stage-current"><div class="tile-val sm" id="v-savg" style="color:#5E6472">—</div><div class="tile-sub" id="v-slbl">—</div></div>
  <div id="stage-history"></div></div>""",
        plot_divs_html='<div id="main-plot" style="flex:1;min-height:0"></div>',
        js_defs=f"""
const PLOT_IDS=['main-plot'];
const BL={BL};
const TRACES=[
  {{name:'Caution',x:[],y:[],stackgroup:'one',mode:'none',fillcolor:'rgba(201,130,69,.72)',line:{{color:'#C98245',width:.6}},hovertemplate:'Caution: %{{y:.1%}}<extra></extra>'}},
  {{name:'Danger', x:[],y:[],stackgroup:'one',mode:'none',fillcolor:'rgba(164,74,63,.72)',line:{{color:'#A44A3F',width:.6}},hovertemplate:'Danger: %{{y:.1%}}<extra></extra>'}},
  {{name:'Critical',x:[],y:[],stackgroup:'one',mode:'none',fillcolor:'rgba(94,100,114,.72)',line:{{color:'#5E6472',width:.6}},hovertemplate:'Critical: %{{y:.1%}}<extra></extra>'}},
];""",
        js_init=f"""
Plotly.newPlot('main-plot',TRACES,BL,{{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']}});""",
        js_extend="""
const ax=[],ay=[],bx=[],by=[],cx=[],cy=[];
for(let i=from+1;i<=to;i++){ax.push(D.episodes[i]);ay.push(D.fca[i]);bx.push(D.episodes[i]);by.push(D.fda[i]);cx.push(D.episodes[i]);cy.push(D.fcr[i]);}
if(ax.length) Plotly.extendTraces('main-plot',{x:[ax,bx,cx],y:[ay,by,cy]},[0,1,2]);""",
        js_rebuild="""
const ax=[],ay=[],bx=[],by=[],cx=[],cy=[];
for(let i=0;i<=tgt;i++){ax.push(D.episodes[i]);ay.push(D.fca[i]);bx.push(D.episodes[i]);by.push(D.fda[i]);cx.push(D.episodes[i]);cy.push(D.fcr[i]);}
Plotly.react('main-plot',[{...TRACES[0],x:ax,y:ay},{...TRACES[1],x:bx,y:by},{...TRACES[2],x:cx,y:cy}],{...BL,shapes:[...shapes],annotations:[...annots]});""",
        js_update_tiles="""
document.getElementById('v-ep').textContent=D.episodes[i].toLocaleString();
document.getElementById('v-ca').textContent=pct(D.rfca[i]);
document.getElementById('v-cr').textContent=pct(D.rfcr[i]);
document.getElementById('v-savg').textContent=pct(D.rfcr[i]);
document.getElementById('v-slbl').textContent=(D.stageLabels?D.stageLabels[i]:'—')+' · running avg';
document.getElementById('ep-counter').textContent=(i+1)+' / '+TOTAL;""",
        js_reset_tiles="['v-ep','v-ca','v-cr','v-savg'].forEach(id=>document.getElementById(id).textContent='—');",
        js_reset_plots="Plotly.react('main-plot',TRACES.map(t=>({...t,x:[],y:[]})),BL);",
        pb_color='#C98245',
    )

def main():
    args = base_argparser('Interactive shield fraction area plot').parse_args()
    episodes, ctrans = load_and_sort(args)
    data = build_data(episodes, args.rolling, ctrans)
    save_html(generate_html(data), args, '12_shield_fraction_interactive.html')

if __name__ == '__main__': main()
