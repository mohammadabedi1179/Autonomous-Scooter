#!/usr/bin/env python3
"""Interactive animated Shield Stats plots.
Tiles: Episode | Interventions/ep | Intervention fraction | Stage avg + history
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from interactive_base import *

def build_data(episodes, w, ctrans=None):
    ep, iv, lns, reasons, stages = [], [], [], [], []
    for i, e in enumerate(episodes):
        ep.append(int(safe_float(e.get('episode', i+1)) or (i+1)))
        iv.append(safe_float(e.get('shield_interventions_ep')))
        lns.append(safe_float(e.get('length', 1)) or 1)
        reasons.append(str(e.get('reason', 'unknown')).lower())
        try: stages.append(int(float(e.get('curriculum_stage', 0))))
        except: stages.append(0)
    riv = rolling_mean(iv, w)
    frac = [safe_float(a/b) if a is not None and b else None for a, b in zip(iv, lns)]
    rfrac = rolling_mean(frac, w)
    transitions, stage_zones, stage_labels = build_stage_data(ep, stages, ctrans)
    completed = build_completed_avgs(ep, transitions, riv)
    return {
        'episodes': ep, 'ivRaw': iv, 'ivRoll': riv,
        'fracRaw': frac, 'fracRoll': rfrac, 'reasons': reasons,
        'stageLabels': stage_labels, 'completedStageAvgs': completed,
        'stageZones': stage_zones, 'transitions': transitions,
        'yIvRange': y_range(iv, 0.04, 0.12),
        'yFrRange': [0, 1.08],
        'xRange': [ep[0], ep[-1]], 'rollingWindow': w, 'total': len(ep),
    }

REASON_COLORS = {'goal': '#87986A', 'collision': '#A44A3F', 'timeout': '#C98245'}

def generate_html(data):
    dj = json.dumps(data, separators=(',', ':'))
    BL = """{
  paper_bgcolor:'#FAF6F0',plot_bgcolor:'#FAF6F0',
  font:{family:"'DM Sans',sans-serif",color:'#463F3A',size:12},
  margin:{t:30,r:20,b:44,l:66},
  xaxis:{title:{text:'Episode',font:{size:12}},gridcolor:'#E8DDD2',linecolor:'#D8CABD',
    tickcolor:'#6B625B',tickfont:{size:10},zeroline:false,fixedrange:false},
  legend:{x:.99,y:.99,xanchor:'right',yanchor:'top',bgcolor:'rgba(250,246,240,.92)',
    bordercolor:'#D8CABD',borderwidth:1,font:{size:10}},
  shapes:[],annotations:[],hovermode:'x unified',
  hoverlabel:{bgcolor:'#FAF6F0',bordercolor:'#D8CABD',font:{family:"'DM Mono',monospace",size:11,color:'#463F3A'}},
}"""
    return generate_page_html(
        'Shield Stats — Interactive', 'Shield Interventions Over Training', dj,
        extra_css='',
        tiles_html="""
<div class="tile tile-fixed"><div class="tile-lbl">Episode</div><div class="tile-val" id="v-ep">—</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Interventions/ep</div><div class="tile-val" id="v-iv" style="color:var(--warm)">—</div><div class="tile-sub">rolling mean</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Intervention fraction</div><div class="tile-val" id="v-fr" style="color:var(--warm)">—</div><div class="tile-sub">steps / episode length</div></div>
<div class="tile tile-grow"><div class="tile-lbl">Stage avg interventions</div>
  <div id="stage-current"><div class="tile-val sm" id="v-savg" style="color:var(--stage)">—</div><div class="tile-sub" id="v-slbl">—</div></div>
  <div id="stage-history"></div></div>""",
        plot_divs_html="""
<div style="flex:1;min-height:0;display:flex;flex-direction:column;gap:4px">
  <div style="flex:1;min-height:0" id="plot-top"></div>
  <div style="flex:1;min-height:0" id="plot-bot"></div>
</div>""",
        js_defs=f"""
const PLOT_IDS=['plot-top','plot-bot'];
const RC={{'goal':'#87986A','collision':'#A44A3F','timeout':'#C98245','unknown':'#9B8E82'}};
const BL_TOP={{...{BL},yaxis:{{...{BL}.yaxis,title:{{text:'Shield interventions / ep',font:{{size:12}}}},range:D.yIvRange}}}};
const BL_BOT={{...{BL},yaxis:{{...{BL}.yaxis,title:{{text:'Interventions / ep length',font:{{size:12}}}},range:D.yFrRange}}}};
const roll_tr={{name:'Rolling mean',mode:'lines',type:'scatter',line:{{color:'#C9A27E',width:2.2}},hovertemplate:'%{{y:.2f}}<extra></extra>'}};
const TT=[
  {{'name':'Goal',x:[],y:[],mode:'markers',type:'scattergl',marker:{{color:'#87986A',size:3,opacity:.4}},hoverinfo:'skip'}},
  {{'name':'Collision',x:[],y:[],mode:'markers',type:'scattergl',marker:{{color:'#A44A3F',size:3,opacity:.4}},hoverinfo:'skip'}},
  {{'name':'Timeout',x:[],y:[],mode:'markers',type:'scattergl',marker:{{color:'#C98245',size:3,opacity:.4}},hoverinfo:'skip'}},
  {{...roll_tr,x:[],y:[],name:'Rolling mean (w='+D.rollingWindow+')'}},
];
const TB=[
  {{'name':'Goal',x:[],y:[],mode:'markers',type:'scattergl',marker:{{color:'#87986A',size:3,opacity:.4}},hoverinfo:'skip'}},
  {{'name':'Collision',x:[],y:[],mode:'markers',type:'scattergl',marker:{{color:'#A44A3F',size:3,opacity:.4}},hoverinfo:'skip'}},
  {{'name':'Timeout',x:[],y:[],mode:'markers',type:'scattergl',marker:{{color:'#C98245',size:3,opacity:.4}},hoverinfo:'skip'}},
  {{...roll_tr,x:[],y:[]}},
];""",
        js_init="""
Plotly.newPlot('plot-top',TT,BL_TOP,{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']});
Plotly.newPlot('plot-bot',TB,BL_BOT,{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']});""",
        js_extend="""
const ex={0:{x:[],y:[]},1:{x:[],y:[]},2:{x:[],y:[]},3:{x:[],y:[]}};
const fx={0:{x:[],y:[]},1:{x:[],y:[]},2:{x:[],y:[]},3:{x:[],y:[]}};
const RTMAP={goal:0,collision:1,timeout:2};
for(let i=from+1;i<=to;i++){
  const ti=RTMAP[D.reasons[i]]??0;
  if(D.ivRaw[i]!==null){ex[ti].x.push(D.episodes[i]);ex[ti].y.push(D.ivRaw[i]);}
  if(D.fracRaw[i]!==null){fx[ti].x.push(D.episodes[i]);fx[ti].y.push(D.fracRaw[i]);}
  if(D.ivRoll[i]!==null){ex[3].x.push(D.episodes[i]);ex[3].y.push(D.ivRoll[i]);}
  if(D.fracRoll[i]!==null){fx[3].x.push(D.episodes[i]);fx[3].y.push(D.fracRoll[i]);}
}
for(let ti=0;ti<4;ti++){
  if(ex[ti].x.length) Plotly.extendTraces('plot-top',{x:[ex[ti].x],y:[ex[ti].y]},[ti]);
  if(fx[ti].x.length) Plotly.extendTraces('plot-bot',{x:[fx[ti].x],y:[fx[ti].y]},[ti]);
}""",
        js_rebuild="""
const etop=[[],[],[],[]],ebot=[[],[],[],[]],ytop=[[],[],[],[]],ybot=[[],[],[],[]];
const RTMAP={goal:0,collision:1,timeout:2};
for(let i=0;i<=tgt;i++){
  const ti=RTMAP[D.reasons[i]]??0;
  if(D.ivRaw[i]!==null){etop[ti].push(D.episodes[i]);ytop[ti].push(D.ivRaw[i]);}
  if(D.fracRaw[i]!==null){ebot[ti].push(D.episodes[i]);ybot[ti].push(D.fracRaw[i]);}
  if(D.ivRoll[i]!==null){etop[3].push(D.episodes[i]);ytop[3].push(D.ivRoll[i]);}
  if(D.fracRoll[i]!==null){ebot[3].push(D.episodes[i]);ybot[3].push(D.fracRoll[i]);}
}
Plotly.react('plot-top',TT.map((t,i)=>({...t,x:etop[i],y:ytop[i]})),{...BL_TOP,shapes:[...shapes],annotations:[...annots]});
Plotly.react('plot-bot',TB.map((t,i)=>({...t,x:ebot[i],y:ybot[i]})),{...BL_BOT,shapes:[...shapes],annotations:[...annots]});""",
        js_update_tiles="""
document.getElementById('v-ep').textContent=D.episodes[i].toLocaleString();
document.getElementById('v-iv').textContent=fmt(D.ivRoll[i],1);
document.getElementById('v-fr').textContent=D.fracRoll[i]!==null?(D.fracRoll[i]*100).toFixed(1)+'%':'—';
document.getElementById('v-savg').textContent=fmt(D.stageLabels[i],0);
const lbl=D.stageLabels?D.stageLabels[i]:'—';
document.getElementById('v-slbl').textContent=lbl+' · running avg';
document.getElementById('v-savg').textContent=fmt(D.ivRoll[i],1);
document.getElementById('ep-counter').textContent=(i+1)+' / '+TOTAL;""",
        js_reset_tiles="['v-ep','v-iv','v-fr','v-savg'].forEach(id=>document.getElementById(id).textContent='—');",
        js_reset_plots="""
Plotly.react('plot-top',TT.map(t=>({...t,x:[],y:[]})),BL_TOP);
Plotly.react('plot-bot',TB.map(t=>({...t,x:[],y:[]})),BL_BOT);""",
        pb_color='#C9A27E',
    )

def main():
    args = base_argparser('Interactive shield stats plot').parse_args()
    episodes, ctrans = load_and_sort(args)
    data = build_data(episodes, args.rolling, ctrans)
    save_html(generate_html(data), args, '07_shield_stats_interactive.html')

if __name__ == '__main__': main()
