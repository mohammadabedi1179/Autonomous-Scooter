#!/usr/bin/env python3
"""Interactive animated Mean Jerk plot.
Tiles: Episode | Jerk v | Jerk w | Smoothness index
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from interactive_base import *

def build_data(episodes, w, ctrans=None):
    ep, jv, jw, stages = [], [], [], []
    for i, e in enumerate(episodes):
        ep.append(int(safe_float(e.get('episode', i+1)) or (i+1)))
        jv.append(safe_float(e.get('ep_mean_jerk_v')))
        jw.append(safe_float(e.get('ep_mean_jerk_w')))
        try: stages.append(int(float(e.get('curriculum_stage', 0))))
        except: stages.append(0)
    rjv, rjw = rolling_mean(jv, w), rolling_mean(jw, w)
    # Smoothness: 1/(1+jv+jw), closer to 1 = smoother
    smooth = []
    for a, b in zip(rjv, rjw):
        if a is not None and b is not None: smooth.append(round(1/(1+a+b), 6))
        else: smooth.append(None)
    transitions, stage_zones, _ = build_stage_data(ep, stages, ctrans)
    all_j = [v for v in jv+jw if v is not None]
    return {
        'episodes': ep, 'jvRaw': jv, 'jwRaw': jw,
        'jvRoll': rjv, 'jwRoll': rjw, 'smooth': smooth,
        'stageZones': stage_zones, 'transitions': transitions,
        'yRange': y_range(all_j, 0.04, 0.14),
        'xRange': [ep[0], ep[-1]], 'rollingWindow': w, 'total': len(ep),
    }

def generate_html(data):
    dj = json.dumps(data, separators=(',', ':'))
    BL = """{
  paper_bgcolor:'#FAF6F0',plot_bgcolor:'#FAF6F0',
  font:{family:"'DM Sans',sans-serif",color:'#463F3A',size:12},
  margin:{t:38,r:20,b:48,l:66},
  xaxis:{title:{text:'Episode',font:{size:12}},gridcolor:'#E8DDD2',linecolor:'#D8CABD',
    tickcolor:'#6B625B',tickfont:{size:11},zeroline:false,fixedrange:false},
  yaxis:{title:{text:'Mean jerk',font:{size:12}},gridcolor:'#E8DDD2',linecolor:'#D8CABD',
    tickcolor:'#6B625B',tickfont:{size:11},range:D.yRange,zeroline:true,zerolinecolor:'#D0C4B8'},
  legend:{x:.99,y:.99,xanchor:'right',yanchor:'top',bgcolor:'rgba(250,246,240,.92)',
    bordercolor:'#D8CABD',borderwidth:1,font:{size:11}},
  shapes:[],annotations:[],hovermode:'x unified',
  hoverlabel:{bgcolor:'#FAF6F0',bordercolor:'#D8CABD',font:{family:"'DM Mono',monospace",size:11,color:'#463F3A'}},
}"""
    return generate_page_html(
        'Mean Jerk — Interactive', 'Mean Jerk (v & w) Over Training', dj,
        extra_css='',
        tiles_html="""
<div class="tile tile-fixed"><div class="tile-lbl">Episode</div><div class="tile-val" id="v-ep">—</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Jerk v</div><div class="tile-val" id="v-jv" style="color:var(--teal)">—</div><div class="tile-sub">rolling mean</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Jerk w</div><div class="tile-val" id="v-jw" style="color:var(--warm)">—</div><div class="tile-sub">rolling mean</div></div>
<div class="tile"><div class="tile-lbl">Smoothness index</div><div class="tile-val" id="v-sm" style="color:var(--goal)">—</div><div class="tile-sub">1/(1+jv+jw) → 1 = smooth</div></div>""",
        plot_divs_html='<div id="main-plot" style="flex:1;min-height:0"></div>',
        js_defs=f"""
const PLOT_IDS=['main-plot'];
const BL={BL};
const TRACES=[
  {{name:'Jerk v',x:[],y:[],mode:'markers',type:'scattergl',marker:{{color:'#7D9D9C',size:3,opacity:.4}},hoverinfo:'skip'}},
  {{name:'Jerk w',x:[],y:[],mode:'markers',type:'scattergl',marker:{{color:'#C9A27E',size:3,opacity:.4}},hoverinfo:'skip'}},
  {{name:'Mean jerk v (rolling)',x:[],y:[],mode:'lines',type:'scatter',line:{{color:'#7D9D9C',width:2.2}},hovertemplate:'v: %{{y:.3f}}<extra></extra>'}},
  {{name:'Mean jerk w (rolling)',x:[],y:[],mode:'lines',type:'scatter',line:{{color:'#C9A27E',width:2.2}},hovertemplate:'w: %{{y:.3f}}<extra></extra>'}},
];""",
        js_init=f"""
Plotly.newPlot('main-plot',TRACES,BL,{{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']}});""",
        js_extend="""
const ax=[],ay=[],bx=[],by=[],arx=[],ary=[],brx=[],bry=[];
for(let i=from+1;i<=to;i++){
  if(D.jvRaw[i]!==null){ax.push(D.episodes[i]);ay.push(D.jvRaw[i]);}
  if(D.jwRaw[i]!==null){bx.push(D.episodes[i]);by.push(D.jwRaw[i]);}
  if(D.jvRoll[i]!==null){arx.push(D.episodes[i]);ary.push(D.jvRoll[i]);}
  if(D.jwRoll[i]!==null){brx.push(D.episodes[i]);bry.push(D.jwRoll[i]);}
}
if(ax.length) Plotly.extendTraces('main-plot',{x:[ax,bx,arx,brx],y:[ay,by,ary,bry]},[0,1,2,3]);""",
        js_rebuild="""
const ax=[],ay=[],bx=[],by=[],arx=[],ary=[],brx=[],bry=[];
for(let i=0;i<=tgt;i++){
  if(D.jvRaw[i]!==null){ax.push(D.episodes[i]);ay.push(D.jvRaw[i]);}
  if(D.jwRaw[i]!==null){bx.push(D.episodes[i]);by.push(D.jwRaw[i]);}
  if(D.jvRoll[i]!==null){arx.push(D.episodes[i]);ary.push(D.jvRoll[i]);}
  if(D.jwRoll[i]!==null){brx.push(D.episodes[i]);bry.push(D.jwRoll[i]);}
}
Plotly.react('main-plot',[{...TRACES[0],x:ax,y:ay},{...TRACES[1],x:bx,y:by},{...TRACES[2],x:arx,y:ary},{...TRACES[3],x:brx,y:bry}],{...BL,shapes:[...shapes],annotations:[...annots]});""",
        js_update_tiles="""
document.getElementById('v-ep').textContent=D.episodes[i].toLocaleString();
document.getElementById('v-jv').textContent=fmt(D.jvRoll[i],3);
document.getElementById('v-jw').textContent=fmt(D.jwRoll[i],3);
const sm=D.smooth[i]; document.getElementById('v-sm').textContent=sm!==null?(sm*100).toFixed(1)+'%':'—';
document.getElementById('ep-counter').textContent=(i+1)+' / '+TOTAL;""",
        js_reset_tiles="['v-ep','v-jv','v-jw','v-sm'].forEach(id=>document.getElementById(id).textContent='—');",
        js_reset_plots="Plotly.react('main-plot',TRACES.map(t=>({...t,x:[],y:[]})),BL);",
        pb_color='#7D9D9C',
    )

def main():
    args = base_argparser('Interactive mean jerk plot').parse_args()
    episodes, ctrans = load_and_sort(args)
    data = build_data(episodes, args.rolling, ctrans)
    save_html(generate_html(data), args, '08_mean_jerk_interactive.html')

if __name__ == '__main__': main()
