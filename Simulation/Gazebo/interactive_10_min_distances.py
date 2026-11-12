#!/usr/bin/env python3
"""Interactive animated Minimum Distances plot.
Tiles: Episode | Min ultrasonic | Min obstacle depth | Stage history of obstacle depth
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from interactive_base import *

def build_data(episodes, w, ctrans=None):
    ep, us, ob, stages = [], [], [], []
    for i, e in enumerate(episodes):
        ep.append(int(safe_float(e.get('episode', i+1)) or (i+1)))
        us.append(safe_float(e.get('ep_min_ultrasonic_distance')))
        ob.append(safe_float(e.get('ep_min_obstacle_depth')))
        try: stages.append(int(float(e.get('curriculum_stage', 0))))
        except: stages.append(0)
    rus, rob = rolling_mean(us, w), rolling_mean(ob, w)
    transitions, stage_zones, stage_labels = build_stage_data(ep, stages, ctrans)
    completed = build_completed_avgs(ep, transitions, rob)
    all_v = [v for v in us+ob if v is not None]
    return {
        'episodes': ep, 'usRaw': us, 'obRaw': ob, 'usRoll': rus, 'obRoll': rob,
        'stageLabels': stage_labels, 'completedStageAvgs': completed,
        'stageZones': stage_zones, 'transitions': transitions,
        'yRange': y_range(all_v, 0.04, 0.12),
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
  yaxis:{title:{text:'Distance',font:{size:12}},gridcolor:'#E8DDD2',linecolor:'#D8CABD',
    tickcolor:'#6B625B',tickfont:{size:11},range:D.yRange,zeroline:true,zerolinecolor:'#D0C4B8'},
  legend:{x:.99,y:.99,xanchor:'right',yanchor:'top',bgcolor:'rgba(250,246,240,.92)',
    bordercolor:'#D8CABD',borderwidth:1,font:{size:11}},
  shapes:[],annotations:[],hovermode:'x unified',
  hoverlabel:{bgcolor:'#FAF6F0',bordercolor:'#D8CABD',font:{family:"'DM Mono',monospace",size:11,color:'#463F3A'}},
}"""
    return generate_page_html(
        'Min Distances — Interactive', 'Minimum Distances per Episode', dj,
        extra_css='',
        tiles_html="""
<div class="tile tile-fixed"><div class="tile-lbl">Episode</div><div class="tile-val" id="v-ep">—</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Min ultrasonic</div><div class="tile-val" id="v-us" style="color:var(--teal)">—</div><div class="tile-sub">rolling mean</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Min obstacle depth</div><div class="tile-val" id="v-ob" style="color:var(--warm)">—</div><div class="tile-sub">rolling mean</div></div>
<div class="tile tile-grow"><div class="tile-lbl">Stage obstacle avg</div>
  <div id="stage-current"><div class="tile-val sm" id="v-savg" style="color:var(--stage)">—</div><div class="tile-sub" id="v-slbl">—</div></div>
  <div id="stage-history"></div></div>""",
        plot_divs_html='<div id="main-plot" style="flex:1;min-height:0"></div>',
        js_defs=f"""
const PLOT_IDS=['main-plot'];
const BL={BL};
const TRACES=[
  {{name:'Min Ultrasonic',x:[],y:[],mode:'markers',type:'scattergl',marker:{{color:'#7D9D9C',size:3,opacity:.4}},hoverinfo:'skip'}},
  {{name:'Min Obstacle Depth',x:[],y:[],mode:'markers',type:'scattergl',marker:{{color:'#C9A27E',size:3,opacity:.4}},hoverinfo:'skip'}},
  {{name:'Min Ultrasonic (rolling)',x:[],y:[],mode:'lines',type:'scatter',line:{{color:'#7D9D9C',width:2.2}},hovertemplate:'Ultrasonic: %{{y:.3f}}<extra></extra>'}},
  {{name:'Min Obstacle Depth (rolling)',x:[],y:[],mode:'lines',type:'scatter',line:{{color:'#C9A27E',width:2.2}},hovertemplate:'Obstacle: %{{y:.3f}}<extra></extra>'}},
];""",
        js_init=f"""
Plotly.newPlot('main-plot',TRACES,BL,{{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']}});""",
        js_extend="""
const ax=[],ay=[],bx=[],by=[],arx=[],ary=[],brx=[],bry=[];
for(let i=from+1;i<=to;i++){
  if(D.usRaw[i]!==null){ax.push(D.episodes[i]);ay.push(D.usRaw[i]);}
  if(D.obRaw[i]!==null){bx.push(D.episodes[i]);by.push(D.obRaw[i]);}
  if(D.usRoll[i]!==null){arx.push(D.episodes[i]);ary.push(D.usRoll[i]);}
  if(D.obRoll[i]!==null){brx.push(D.episodes[i]);bry.push(D.obRoll[i]);}
}
if(ax.length) Plotly.extendTraces('main-plot',{x:[ax,bx,arx,brx],y:[ay,by,ary,bry]},[0,1,2,3]);""",
        js_rebuild="""
const ax=[],ay=[],bx=[],by=[],arx=[],ary=[],brx=[],bry=[];
for(let i=0;i<=tgt;i++){
  if(D.usRaw[i]!==null){ax.push(D.episodes[i]);ay.push(D.usRaw[i]);}
  if(D.obRaw[i]!==null){bx.push(D.episodes[i]);by.push(D.obRaw[i]);}
  if(D.usRoll[i]!==null){arx.push(D.episodes[i]);ary.push(D.usRoll[i]);}
  if(D.obRoll[i]!==null){brx.push(D.episodes[i]);bry.push(D.obRoll[i]);}
}
Plotly.react('main-plot',[{...TRACES[0],x:ax,y:ay},{...TRACES[1],x:bx,y:by},{...TRACES[2],x:arx,y:ary},{...TRACES[3],x:brx,y:bry}],{...BL,shapes:[...shapes],annotations:[...annots]});""",
        js_update_tiles="""
document.getElementById('v-ep').textContent=D.episodes[i].toLocaleString();
document.getElementById('v-us').textContent=fmt(D.usRoll[i],3);
document.getElementById('v-ob').textContent=fmt(D.obRoll[i],3);
document.getElementById('v-savg').textContent=fmt(D.obRoll[i],3);
document.getElementById('v-slbl').textContent=(D.stageLabels?D.stageLabels[i]:'—')+' · running avg';
document.getElementById('ep-counter').textContent=(i+1)+' / '+TOTAL;""",
        js_reset_tiles="['v-ep','v-us','v-ob','v-savg'].forEach(id=>document.getElementById(id).textContent='—');",
        js_reset_plots="Plotly.react('main-plot',TRACES.map(t=>({...t,x:[],y:[]})),BL);",
        pb_color='#7D9D9C',
    )

def main():
    args = base_argparser('Interactive min distances plot').parse_args()
    episodes, ctrans = load_and_sort(args)
    data = build_data(episodes, args.rolling, ctrans)
    save_html(generate_html(data), args, '10_min_distances_interactive.html')

if __name__ == '__main__': main()
