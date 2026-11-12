#!/usr/bin/env python3
"""Interactive animated Learning Rate & Entropy plots.
Tiles: Episode | Learning rate | Entropy | LR step count
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from interactive_base import *

def build_data(episodes, w, ctrans=None):
    ep, lr, ent, stages = [], [], [], []
    for i, e in enumerate(episodes):
        ep.append(int(safe_float(e.get('episode', i+1)) or (i+1)))
        lr.append(safe_float(e.get('learning_rate')))
        ent.append(safe_float(e.get('ent_coef') or e.get('entropy')))
        try: stages.append(int(float(e.get('curriculum_stage', 0))))
        except: stages.append(0)
    rent = rolling_mean(ent, w)
    transitions, stage_zones, _ = build_stage_data(ep, stages, ctrans)
    # Count LR step-downs (each drop = one scheduler step)
    lr_steps, prev, step_count = 0, None, []
    for v in lr:
        if v is not None and prev is not None and v < prev - 1e-9: lr_steps += 1
        step_count.append(lr_steps)
        if v is not None: prev = v
    return {
        'episodes': ep, 'lr': lr, 'entRaw': ent, 'entRoll': rent,
        'lrSteps': step_count,
        'stageZones': stage_zones, 'transitions': transitions,
        'yLrRange': y_range(lr, 0.05, 0.15),
        'yEntRange': y_range(ent, 0.04, 0.12),
        'xRange': [ep[0], ep[-1]], 'rollingWindow': w, 'total': len(ep),
    }

def generate_html(data):
    dj = json.dumps(data, separators=(',', ':'))
    BL = """{
  paper_bgcolor:'#FAF6F0',plot_bgcolor:'#FAF6F0',
  font:{family:"'DM Sans',sans-serif",color:'#463F3A',size:12},
  margin:{t:30,r:20,b:44,l:80},
  xaxis:{title:{text:'Episode',font:{size:12}},gridcolor:'#E8DDD2',linecolor:'#D8CABD',
    tickcolor:'#6B625B',tickfont:{size:10},zeroline:false,fixedrange:false},
  legend:{x:.99,y:.99,xanchor:'right',yanchor:'top',bgcolor:'rgba(250,246,240,.92)',
    bordercolor:'#D8CABD',borderwidth:1,font:{size:10}},
  shapes:[],annotations:[],hovermode:'x unified',
  hoverlabel:{bgcolor:'#FAF6F0',bordercolor:'#D8CABD',font:{family:"'DM Mono',monospace",size:11,color:'#463F3A'}},
}"""
    return generate_page_html(
        'LR & Entropy — Interactive', 'Learning Rate & Policy Entropy Over Training', dj,
        extra_css='',
        tiles_html="""
<div class="tile tile-fixed"><div class="tile-lbl">Episode</div><div class="tile-val" id="v-ep">—</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Learning rate</div><div class="tile-val sm" id="v-lr" style="color:var(--teal)">—</div><div class="tile-sub">current value</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Policy entropy</div><div class="tile-val" id="v-ent" style="color:var(--teal)">—</div><div class="tile-sub">rolling</div></div>
<div class="tile"><div class="tile-lbl">LR step count</div><div class="tile-val" id="v-steps" style="color:var(--warm)">—</div><div class="tile-sub">scheduler drops</div></div>""",
        plot_divs_html="""
<div style="flex:1;min-height:0;display:flex;flex-direction:column;gap:4px">
  <div style="flex:1;min-height:0" id="plot-top"></div>
  <div style="flex:1;min-height:0" id="plot-bot"></div>
</div>""",
        js_defs=f"""
const PLOT_IDS=['plot-top','plot-bot'];
const BL_TOP={{...{BL},yaxis:{{...{BL}.yaxis,title:{{text:'Learning rate',font:{{size:12}}}},range:D.yLrRange,tickformat:'.2e'}}}};
const BL_BOT={{...{BL},yaxis:{{...{BL}.yaxis,title:{{text:'Entropy',font:{{size:12}}}},range:D.yEntRange,zeroline:true,zerolinecolor:'#D0C4B8'}}}};
const TT=[{{name:'Learning rate',x:[],y:[],mode:'lines',type:'scatter',line:{{color:'#7D9D9C',width:2.2}},hovertemplate:'Ep <b>%{{x}}</b> — %{{y:.2e}}<extra></extra>'}}];
const TB=[{{name:'Entropy',x:[],y:[],mode:'markers',type:'scattergl',marker:{{color:'#7D9D9C',size:3,opacity:.45}},hoverinfo:'skip'}},
  {{name:'Entropy (rolling)',x:[],y:[],mode:'lines',type:'scatter',line:{{color:'#7D9D9C',width:2.2}},hovertemplate:'Ep <b>%{{x}}</b> — %{{y:.3f}}<extra></extra>'}}];""",
        js_init="""
Plotly.newPlot('plot-top',TT,BL_TOP,{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']});
Plotly.newPlot('plot-bot',TB,BL_BOT,{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']});""",
        js_extend="""
const lx=[],ly=[],ex=[],ey=[],er=[];
for(let i=from+1;i<=to;i++){
  if(D.lr[i]!==null){lx.push(D.episodes[i]);ly.push(D.lr[i]);}
  if(D.entRaw[i]!==null){ex.push(D.episodes[i]);ey.push(D.entRaw[i]);}
  if(D.entRoll[i]!==null){er.push(D.episodes[i]);}
}
if(lx.length) Plotly.extendTraces('plot-top',{x:[lx],y:[ly]},[0]);
if(ex.length) Plotly.extendTraces('plot-bot',{x:[ex,er],y:[ey,D.entRoll.slice(from+1,to+1).filter(v=>v!==null)]},[0,1]);""",
        js_rebuild="""
const lx=[],ly=[],ex=[],ey=[],erx=[],ery=[];
for(let i=0;i<=tgt;i++){
  if(D.lr[i]!==null){lx.push(D.episodes[i]);ly.push(D.lr[i]);}
  if(D.entRaw[i]!==null){ex.push(D.episodes[i]);ey.push(D.entRaw[i]);}
  if(D.entRoll[i]!==null){erx.push(D.episodes[i]);ery.push(D.entRoll[i]);}
}
Plotly.react('plot-top',[{...TT[0],x:lx,y:ly}],{...BL_TOP,shapes:[...shapes],annotations:[...annots]});
Plotly.react('plot-bot',[{...TB[0],x:ex,y:ey},{...TB[1],x:erx,y:ery}],{...BL_BOT,shapes:[...shapes],annotations:[...annots]});""",
        js_update_tiles="""
document.getElementById('v-ep').textContent=D.episodes[i].toLocaleString();
const lr=D.lr[i]; document.getElementById('v-lr').textContent=lr!==null?lr.toExponential(3):'—';
document.getElementById('v-ent').textContent=fmt(D.entRoll[i],3);
document.getElementById('v-steps').textContent=D.lrSteps[i];
document.getElementById('ep-counter').textContent=(i+1)+' / '+TOTAL;""",
        js_reset_tiles="['v-ep','v-lr','v-ent','v-steps'].forEach(id=>document.getElementById(id).textContent='—');",
        js_reset_plots="""
Plotly.react('plot-top',[{...TT[0],x:[],y:[]}],BL_TOP);
Plotly.react('plot-bot',TB.map(t=>({...t,x:[],y:[]})),BL_BOT);""",
        pb_color='#7D9D9C',
    )

def main():
    args = base_argparser('Interactive LR & entropy plot').parse_args()
    episodes, ctrans = load_and_sort(args)
    data = build_data(episodes, args.rolling, ctrans)
    save_html(generate_html(data), args, '06_lr_entropy_interactive.html')

if __name__ == '__main__': main()
