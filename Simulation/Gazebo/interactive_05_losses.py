#!/usr/bin/env python3
"""Interactive animated Actor/Critic Loss plots.
Tiles: Episode | Actor loss | Critic loss | Training trend (↓/→/↑)
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from interactive_base import *

def build_data(episodes, w, ctrans=None):
    ep, al, cl, stages = [], [], [], []
    for i, e in enumerate(episodes):
        ep.append(int(safe_float(e.get('episode', i+1)) or (i+1)))
        al.append(safe_float(e.get('actor_loss')))
        cl.append(safe_float(e.get('critic_loss')))
        try: stages.append(int(float(e.get('curriculum_stage', 0))))
        except: stages.append(0)
    ra, rc = rolling_mean(al, w), rolling_mean(cl, w)
    transitions, stage_zones, stage_labels = build_stage_data(ep, stages, ctrans)
    # Trend: compare rolling actor vs 60 eps ago
    trend = []
    for i in range(len(ep)):
        v, prev = ra[i], ra[max(0, i-60)]
        if v is None or prev is None: trend.append(0)
        else: trend.append(round(v - prev, 4))
    return {
        'episodes': ep, 'actorRaw': al, 'criticRaw': cl,
        'actorRoll': ra, 'criticRoll': rc, 'trend': trend,
        'stageZones': stage_zones, 'transitions': transitions,
        'yActorRange': y_range(al, 0.04, 0.12),
        'yCriticRange': y_range(cl, 0.04, 0.12),
        'xRange': [ep[0], ep[-1]], 'rollingWindow': w, 'total': len(ep),
    }

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
        'Actor & Critic Loss — Interactive', 'Actor & Critic Loss Over Training', dj,
        extra_css='',
        tiles_html="""
<div class="tile tile-fixed"><div class="tile-lbl">Episode</div><div class="tile-val" id="v-ep">—</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Actor loss</div><div class="tile-val" id="v-al" style="color:var(--teal)">—</div><div class="tile-sub">rolling</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Critic loss</div><div class="tile-val" id="v-cl" style="color:var(--warm)">—</div><div class="tile-sub">rolling</div></div>
<div class="tile"><div class="tile-lbl">Training trend</div><div class="tile-val sm" id="v-trend">—</div><div class="tile-sub" id="v-trend-sub">actor loss drift</div></div>""",
        plot_divs_html="""
<div style="flex:1;min-height:0;display:flex;flex-direction:column;gap:4px">
  <div style="flex:1;min-height:0" id="plot-top"></div>
  <div style="flex:1;min-height:0" id="plot-bot"></div>
</div>""",
        js_defs=f"""
const PLOT_IDS=['plot-top','plot-bot'];
const BL_TOP={{...{BL},yaxis:{{...{BL}.yaxis,title:{{text:'Actor loss',font:{{size:12}}}},range:D.yActorRange,zeroline:true,zerolinecolor:'#D0C4B8'}}}};
const BL_BOT={{...{BL},yaxis:{{...{BL}.yaxis,title:{{text:'Critic loss',font:{{size:12}}}},range:D.yCriticRange,zeroline:true,zerolinecolor:'#D0C4B8'}}}};
const TT=[{{name:'Actor loss',x:[],y:[],mode:'markers',type:'scattergl',marker:{{color:'#7D9D9C',size:3,opacity:.45}},hoverinfo:'skip'}},
  {{name:'Actor loss (rolling)',x:[],y:[],mode:'lines',type:'scatter',line:{{color:'#7D9D9C',width:2.2}},hovertemplate:'Ep <b>%{{x}}</b> — %{{y:.2f}}<extra></extra>'}}];
const TB=[{{name:'Critic loss',x:[],y:[],mode:'markers',type:'scattergl',marker:{{color:'#C9A27E',size:3,opacity:.45}},hoverinfo:'skip'}},
  {{name:'Critic loss (rolling)',x:[],y:[],mode:'lines',type:'scatter',line:{{color:'#C9A27E',width:2.2}},hovertemplate:'Ep <b>%{{x}}</b> — %{{y:.2f}}<extra></extra>'}}];""",
        js_init="""
Plotly.newPlot('plot-top',TT,BL_TOP,{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']});
Plotly.newPlot('plot-bot',TB,BL_BOT,{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']});""",
        js_extend="""
const ax=[],ay=[],ar=[],cx=[],cy=[],cr=[];
for(let i=from+1;i<=to;i++){
  if(D.actorRaw[i]!==null){ax.push(D.episodes[i]);ay.push(D.actorRaw[i]);}
  if(D.actorRoll[i]!==null){ar.push(D.episodes[i]);}
  if(D.criticRaw[i]!==null){cx.push(D.episodes[i]);cy.push(D.criticRaw[i]);}
  if(D.criticRoll[i]!==null){cr.push(D.episodes[i]);}
}
if(ax.length) Plotly.extendTraces('plot-top',{x:[ax,ar],y:[ay,D.actorRoll.slice(from+1,to+1).filter(v=>v!==null)]},[0,1]);
if(cx.length) Plotly.extendTraces('plot-bot',{x:[cx,cr],y:[cy,D.criticRoll.slice(from+1,to+1).filter(v=>v!==null)]},[0,1]);""",
        js_rebuild="""
const atx=[],aty=[],arx=[],ary=[],btx=[],bty=[],brx=[],bry=[];
for(let i=0;i<=tgt;i++){
  if(D.actorRaw[i]!==null){atx.push(D.episodes[i]);aty.push(D.actorRaw[i]);}
  if(D.actorRoll[i]!==null){arx.push(D.episodes[i]);ary.push(D.actorRoll[i]);}
  if(D.criticRaw[i]!==null){btx.push(D.episodes[i]);bty.push(D.criticRaw[i]);}
  if(D.criticRoll[i]!==null){brx.push(D.episodes[i]);bry.push(D.criticRoll[i]);}
}
Plotly.react('plot-top',[{...TT[0],x:atx,y:aty},{...TT[1],x:arx,y:ary}],{...BL_TOP,shapes:[...shapes],annotations:[...annots]});
Plotly.react('plot-bot',[{...TB[0],x:btx,y:bty},{...TB[1],x:brx,y:bry}],{...BL_BOT,shapes:[...shapes],annotations:[...annots]});""",
        js_update_tiles="""
document.getElementById('v-ep').textContent=D.episodes[i].toLocaleString();
document.getElementById('v-al').textContent=fmt(D.actorRoll[i],2);
document.getElementById('v-cl').textContent=fmt(D.criticRoll[i],2);
const td=D.trend[i];
const te=document.getElementById('v-trend');
if(td<-0.3){te.textContent='↓ Converging';te.style.color='var(--goal)';}
else if(td>0.3){te.textContent='↑ Rising';te.style.color='var(--collision)';}
else{te.textContent='→ Stable';te.style.color='var(--faint)';}
document.getElementById('ep-counter').textContent=(i+1)+' / '+TOTAL;""",
        js_reset_tiles="""
['v-ep','v-al','v-cl'].forEach(id=>document.getElementById(id).textContent='—');
document.getElementById('v-trend').textContent='—'; document.getElementById('v-trend').style.color='';""",
        js_reset_plots="""
Plotly.react('plot-top',TT.map(t=>({...t,x:[],y:[]})),BL_TOP);
Plotly.react('plot-bot',TB.map(t=>({...t,x:[],y:[]})),BL_BOT);""",
        pb_color='#7D9D9C',
    )

def main():
    args = base_argparser('Interactive actor/critic loss plot').parse_args()
    episodes, ctrans = load_and_sort(args)
    data = build_data(episodes, args.rolling, ctrans)
    save_html(generate_html(data), args, '05_losses_interactive.html')

if __name__ == '__main__': main()
