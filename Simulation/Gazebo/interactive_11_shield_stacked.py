#!/usr/bin/env python3
"""Interactive animated Shield Stacked bar chart.
Tiles: Episode | Total interventions | Critical count | Severity breakdown bars
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
    # Rolling fractions of each type
    def frac_roll(vals, totals, w):
        fracs = [v/t if t>0 else 0 for v,t in zip(vals, totals)]
        return rolling_mean(fracs, w)
    rca = frac_roll(ca, total, w)
    rda = frac_roll(da, total, w)
    rcr = frac_roll(cr, total, w)
    transitions, stage_zones, _ = build_stage_data(ep, stages, ctrans)
    all_t = [v for v in total if v]
    return {
        'episodes': ep, 'caution': ca, 'danger': da, 'critical': cr, 'total': total,
        'rcaution': rca, 'rdanger': rda, 'rcritical': rcr,
        'stageZones': stage_zones, 'transitions': transitions,
        'yRange': y_range(all_t, 0.02, 0.12) if all_t else [0, 100],
        'xRange': [ep[0], ep[-1]], 'rollingWindow': w, 'n': len(ep),
    }

def generate_html(data):
    data['total'] = data['n']   # template uses D.total
    dj = json.dumps(data, separators=(',', ':'))
    BL = """{
  paper_bgcolor:'#FAF6F0',plot_bgcolor:'#FAF6F0',
  font:{family:"'DM Sans',sans-serif",color:'#463F3A',size:12},
  margin:{t:38,r:20,b:48,l:66},
  xaxis:{title:{text:'Episode',font:{size:12}},gridcolor:'#E8DDD2',linecolor:'#D8CABD',
    tickcolor:'#6B625B',tickfont:{size:11},zeroline:false,fixedrange:false},
  yaxis:{title:{text:'Intervention steps',font:{size:12}},gridcolor:'#E8DDD2',linecolor:'#D8CABD',
    tickcolor:'#6B625B',tickfont:{size:11},range:D.yRange,zeroline:true,zerolinecolor:'#D0C4B8'},
  legend:{x:.99,y:.99,xanchor:'right',yanchor:'top',bgcolor:'rgba(250,246,240,.92)',
    bordercolor:'#D8CABD',borderwidth:1,font:{size:11}},
  barmode:'stack',shapes:[],annotations:[],hovermode:'x unified',
  hoverlabel:{bgcolor:'#FAF6F0',bordercolor:'#D8CABD',font:{family:"'DM Mono',monospace",size:11,color:'#463F3A'}},
}"""
    SEV_CSS = """
.frac-name.caution{color:#C98245}.frac-name.danger{color:#A44A3F}.frac-name.critical{color:#5E6472}
.frac-fill.caution{background:#C98245}.frac-fill.danger{background:#A44A3F}.frac-fill.critical{background:#5E6472}
"""
    return generate_page_html(
        'Shield Stacked — Interactive', 'Shield Interventions by Severity', dj,
        extra_css=SEV_CSS,
        tiles_html="""
<div class="tile tile-fixed"><div class="tile-lbl">Episode</div><div class="tile-val" id="v-ep">—</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Total interventions</div><div class="tile-val" id="v-tot" style="color:var(--warm)">—</div><div class="tile-sub">this episode</div></div>
<div class="tile tile-fixed"><div class="tile-lbl">Critical count</div><div class="tile-val" id="v-cr" style="color:#5E6472">—</div><div class="tile-sub">most severe type</div></div>
<div class="tile"><div class="tile-lbl">Severity breakdown</div>
  <div class="frac-row"><span class="frac-name caution">Caution</span><div class="frac-track"><div class="frac-fill caution" id="bar-ca"></div></div><span class="frac-pct" id="pct-ca">—</span></div>
  <div class="frac-row"><span class="frac-name danger">Danger</span><div class="frac-track"><div class="frac-fill danger" id="bar-da"></div></div><span class="frac-pct" id="pct-da">—</span></div>
  <div class="frac-row"><span class="frac-name critical">Critical</span><div class="frac-track"><div class="frac-fill critical" id="bar-cr"></div></div><span class="frac-pct" id="pct-cr">—</span></div>
</div>""",
        plot_divs_html='<div id="main-plot" style="flex:1;min-height:0"></div>',
        js_defs=f"""
const PLOT_IDS=['main-plot'];
const BL={BL};
const TRACES=[
  {{name:'Caution',x:[],y:[],type:'bar',marker:{{color:'rgba(201,130,69,.75)'}},hovertemplate:'Caution: %{{y}}<extra></extra>'}},
  {{name:'Danger', x:[],y:[],type:'bar',marker:{{color:'rgba(164,74,63,.75)'}},hovertemplate:'Danger: %{{y}}<extra></extra>'}},
  {{name:'Critical',x:[],y:[],type:'bar',marker:{{color:'rgba(94,100,114,.75)'}},hovertemplate:'Critical: %{{y}}<extra></extra>'}},
];""",
        js_init=f"""
Plotly.newPlot('main-plot',TRACES,BL,{{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']}});""",
        js_extend="""
const ax=[],ay=[],bx=[],by=[],cx=[],cy=[];
for(let i=from+1;i<=to;i++){ax.push(D.episodes[i]);ay.push(D.caution[i]);bx.push(D.episodes[i]);by.push(D.danger[i]);cx.push(D.episodes[i]);cy.push(D.critical[i]);}
if(ax.length) Plotly.extendTraces('main-plot',{x:[ax,bx,cx],y:[ay,by,cy]},[0,1,2]);""",
        js_rebuild="""
const ax=[],ay=[],bx=[],by=[],cx=[],cy=[];
for(let i=0;i<=tgt;i++){ax.push(D.episodes[i]);ay.push(D.caution[i]);bx.push(D.episodes[i]);by.push(D.danger[i]);cx.push(D.episodes[i]);cy.push(D.critical[i]);}
Plotly.react('main-plot',[{...TRACES[0],x:ax,y:ay},{...TRACES[1],x:bx,y:by},{...TRACES[2],x:cx,y:cy}],{...BL,shapes:[...shapes],annotations:[...annots]});""",
        js_update_tiles="""
document.getElementById('v-ep').textContent=D.episodes[i].toLocaleString();
document.getElementById('v-tot').textContent=Math.round(D.total[i]||0).toLocaleString();
document.getElementById('v-cr').textContent=Math.round(D.critical[i]||0).toLocaleString();
const setB=(fid,pid,val)=>{document.getElementById(fid).style.width=((val||0)*100)+'%';document.getElementById(pid).textContent=pct(val);};
setB('bar-ca','pct-ca',D.rcaution[i]);setB('bar-da','pct-da',D.rdanger[i]);setB('bar-cr','pct-cr',D.rcritical[i]);
document.getElementById('ep-counter').textContent=(i+1)+' / '+TOTAL;""",
        js_reset_tiles="""
['v-ep','v-tot','v-cr'].forEach(id=>document.getElementById(id).textContent='—');
['pct-ca','pct-da','pct-cr'].forEach(id=>document.getElementById(id).textContent='—');
['bar-ca','bar-da','bar-cr'].forEach(id=>document.getElementById(id).style.width='0%');""",
        js_reset_plots="Plotly.react('main-plot',TRACES.map(t=>({...t,x:[],y:[]})),BL);",
        pb_color='#C9A27E',
    )

def main():
    args = base_argparser('Interactive shield stacked plot').parse_args()
    episodes, ctrans = load_and_sort(args)
    data = build_data(episodes, args.rolling, ctrans)
    save_html(generate_html(data), args, '11_shield_stacked_interactive.html')

if __name__ == '__main__': main()
