"""
Shared utilities for interactive training analysis HTML generators.
All 7 plot scripts import from this module.
"""
import json, math


# ── Python data utilities ─────────────────────────────────────────────────────

def safe_float(x, default=None):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default

def rolling_mean(values, w):
    out = []
    for i in range(len(values)):
        j0 = max(0, i - w + 1)
        win = [v for v in values[j0:i+1] if v is not None]
        out.append(round(sum(win)/len(win), 6) if win else None)
    return out

def load_episodes(path):
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, list): return raw
    for key in ('episodes', 'data', 'training'):
        if key in raw and isinstance(raw[key], list): return raw[key]
    raise ValueError(f"Cannot find episode list in {path}")

def load_curriculum(path):
    with open(path, encoding='utf-8') as f:
        log = json.load(f)
    return sorted(
        int(e['total_episodes']) for e in log
        if e.get('event') == 'stage_advanced' and e.get('total_episodes') is not None
    )

def build_stage_data(ep_nums, stages, curriculum_transitions=None):
    """Returns (transitions, stage_zones, stage_labels). S1 zone has ep=None."""
    if curriculum_transitions is not None:
        transitions = []
        for ep in curriculum_transitions:
            idx = next((i for i, e in enumerate(ep_nums) if e >= ep), None)
            if idx is not None: transitions.append([ep, stages[idx]])
    else:
        transitions, prev = [], (stages[0] if stages else 0)
        for i in range(1, len(stages)):
            if stages[i] != prev:
                transitions.append([ep_nums[i], stages[i]]); prev = stages[i]

    stage_zones = []
    if transitions:
        stage_zones.append({"ep": None, "center": int((ep_nums[0]+transitions[0][0])/2), "label": "S1"})
        for i, (ep, _) in enumerate(transitions):
            nxt = transitions[i+1][0] if i+1 < len(transitions) else ep_nums[-1]
            stage_zones.append({"ep": ep, "center": int((ep+nxt)/2), "label": f"S{i+2}"})

    zone_idx, stage_labels = 0, []
    for i in range(len(ep_nums)):
        while (zone_idx+1 < len(stage_zones) and
               stage_zones[zone_idx+1]["ep"] is not None and
               ep_nums[i] >= stage_zones[zone_idx+1]["ep"]):
            zone_idx += 1
        stage_labels.append(stage_zones[zone_idx]["label"] if stage_zones else "S1")

    return transitions, stage_zones, stage_labels

def build_completed_avgs(ep_nums, transitions, metric_values):
    """Final metric value at last episode before each stage transition."""
    result = []
    for i, (ep, _) in enumerate(transitions):
        prev = next((j for j in range(len(ep_nums)-1, -1, -1) if ep_nums[j] < ep), None)
        if prev is not None and metric_values[prev] is not None:
            result.append({"transitionEp": ep, "completedLabel": f"S{i+1}", "finalAvg": metric_values[prev]})
    return result

def y_range(values, pad_lo=0.04, pad_hi=0.08, fallback=(-1, 1)):
    vals = [v for v in values if v is not None]
    if not vals: return list(fallback)
    lo, hi = min(vals), max(vals)
    span = hi - lo or abs(hi) or 1
    return [round(lo - span*pad_lo, 4), round(hi + span*pad_hi, 4)]


# ── HTML page template ────────────────────────────────────────────────────────

# Placeholders: <<<TITLE>>> <<<PLOT_TITLE>>> <<<DATA_JSON>>> <<<EXTRA_CSS>>>
# <<<PLOT_DIVS_HTML>>> <<<TILES_HTML>>> <<<PB_COLOR>>>
# <<<JS_DEFS>>> <<<JS_INIT>>> <<<JS_EXTEND>>> <<<JS_REBUILD>>>
# <<<JS_UPDATE_TILES>>> <<<JS_RESET_TILES>>> <<<JS_RESET_PLOTS>>>

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title><<<TITLE>>></title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#FAF6F0;--bg2:#F0E8DC;--bg3:#E8DDD2;
  --border:#D8CABD;--text:#463F3A;--muted:#6B625B;--faint:#9B8E82;
  --goal:#87986A;--collision:#A44A3F;--timeout:#C98245;
  --teal:#7D9D9C;--warm:#C9A27E;--hist:#A89080;
  --font:'DM Sans',-apple-system,sans-serif;--mono:'DM Mono',monospace;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;background:var(--bg);font-family:var(--font);color:var(--text);font-size:13px}
#app{display:flex;flex-direction:column;height:100vh;padding:12px;gap:10px}
#main-row{display:flex;flex:1;gap:10px;min-height:0}
#left{flex:1;min-width:0;position:relative;display:flex;flex-direction:column}
#plot-title{font-size:13px;font-weight:500;color:var(--muted);letter-spacing:.03em;padding:0 4px 6px;flex-shrink:0}
#right{width:200px;display:flex;flex-direction:column;gap:8px}
.tile{background:var(--bg2);border:0.5px solid var(--border);border-radius:10px;padding:11px 14px;flex:1}
.tile-fixed{flex:none}
.tile-grow{flex:1;min-height:0;display:flex;flex-direction:column}
.tile-lbl{font-size:10px;font-weight:500;color:var(--faint);letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px}
.tile-val{font-size:27px;font-weight:500;line-height:1.1;font-family:var(--mono);color:var(--text)}
.tile-val.sm{font-size:18px}
.tile-sub{font-size:11px;color:var(--faint);margin-top:3px}
#stage-history{flex:1;min-height:0;overflow-y:auto;overflow-x:hidden;margin-top:6px;scrollbar-width:none}
#stage-history::-webkit-scrollbar{display:none}
.hist-row{display:flex;justify-content:space-between;align-items:center;max-height:0;overflow:hidden;opacity:0;padding-top:0}
.hist-row.open{max-height:26px;opacity:1;padding-top:4px;transition:max-height .45s cubic-bezier(.4,0,.2,1),opacity .38s ease,padding-top .45s cubic-bezier(.4,0,.2,1)}
.hist-row.instant{max-height:26px;opacity:1;padding-top:4px}
.hist-lbl{font-size:11px;color:var(--faint);font-family:var(--mono)}
.hist-sep{flex:1;margin:0 6px;border-bottom:0.5px dotted var(--border)}
.hist-val{font-size:12px;font-weight:500;color:var(--hist);font-family:var(--mono)}
.frac-row{display:flex;align-items:center;gap:7px;margin-bottom:7px}
.frac-row:last-child{margin-bottom:0}
.frac-name{font-size:10px;font-weight:500;letter-spacing:.04em;text-transform:uppercase;width:46px;flex-shrink:0}
.frac-track{flex:1;height:5px;background:var(--bg3);border-radius:3px;overflow:hidden}
.frac-fill{height:100%;border-radius:3px;width:0%;transition:width .18s ease}
.frac-pct{font-size:11px;font-family:var(--mono);color:var(--muted);width:36px;text-align:right}
#controls{display:flex;align-items:center;gap:8px;flex-shrink:0;padding:2px 0}
.ctrl-btn{background:transparent;border:0.5px solid var(--border);border-radius:6px;padding:4px 11px;font-size:12px;font-family:var(--font);color:var(--text);cursor:pointer;white-space:nowrap;transition:background .12s}
.ctrl-btn:hover{background:var(--bg2)}.ctrl-btn:active{background:var(--bg3)}
#pb-wrap{flex:1;height:5px;background:var(--border);border-radius:3px;cursor:pointer;position:relative}
#pb{height:100%;background:<<<PB_COLOR>>>;border-radius:3px;width:0%;pointer-events:none}
#pb-thumb{position:absolute;top:50%;width:10px;height:10px;background:<<<PB_COLOR>>>;border-radius:50%;transform:translateY(-50%);transition:opacity .2s;opacity:0;pointer-events:none}
#pb-wrap:hover #pb-thumb{opacity:1}
#spd-lbl{font-size:11px;color:var(--muted);min-width:30px;text-align:center;font-family:var(--mono)}
#hover-badge{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);background:var(--bg2);border:0.5px solid var(--border);border-radius:20px;padding:4px 14px;font-size:11px;color:var(--muted);opacity:0;transition:opacity .4s;pointer-events:none;white-space:nowrap}
#hover-badge.show{opacity:1}
#ep-counter{position:absolute;top:8px;right:8px;font-size:10px;font-family:var(--mono);color:var(--faint);pointer-events:none}
<<<EXTRA_CSS>>>
</style>
</head>
<body>
<div id="app">
  <div id="main-row">
    <div id="left">
      <div id="plot-title"><<<PLOT_TITLE>>></div>
      <<<PLOT_DIVS_HTML>>>
      <div id="hover-badge">hover to inspect</div>
      <div id="ep-counter"></div>
    </div>
    <div id="right"><<<TILES_HTML>>></div>
  </div>
  <div id="controls">
    <button class="ctrl-btn" id="btn-play">⏸ Pause</button>
    <div id="pb-wrap"><div id="pb"></div><div id="pb-thumb"></div></div>
    <button class="ctrl-btn" id="btn-slower">−</button>
    <span id="spd-lbl">1×</span>
    <button class="ctrl-btn" id="btn-faster">+</button>
    <button class="ctrl-btn" id="btn-restart">↺ Restart</button>
  </div>
</div>
<script>
const D = <<<DATA_JSON>>>;
const SPEEDS=[.25,.5,1,2,4,8], ANIM_MS=18000;
let speedIdx=2, playing=true, frame=0, startTime=null, animDone=false;
let shownTrans=new Set(), shapes=[], annots=[];
const TOTAL=D.total;

<<<JS_DEFS>>>

// ── stage transitions ─────────────────────────────────────────────────────────
function applyTrans(zone, animate) {
  if (zone.ep !== null) shapes.push({type:'line',x0:zone.ep,x1:zone.ep,y0:0,y1:1,yref:'paper',
    line:{color:'#9B8E82',width:1.1,dash:'dash'}});
  annots.push({x:zone.center,y:1.04,xref:'x',yref:'paper',text:zone.label,showarrow:false,
    font:{size:11,color:'#6B625B',family:"'DM Sans',sans-serif"},bgcolor:'#FAF6F0',borderpad:2});
  if (animate) PLOT_IDS.forEach(id => Plotly.relayout(id,{shapes:[...shapes],annotations:[...annots]}));
}
function checkTrans(ep, animate) {
  for (const z of D.stageZones) {
    if (!shownTrans.has(z.ep) && ep >= (z.ep ?? -Infinity)) {
      shownTrans.add(z.ep); applyTrans(z, animate !== false);
      if (D.completedStageAvgs) {
        const c = D.completedStageAvgs.find(x => x.transitionEp === z.ep);
        if (c) addHistRow(c.completedLabel, c.finalAvg, animate !== false);
      }
    }
  }
}
function initS1() {
  const s1 = D.stageZones.find(z => z.label==='S1'); if (!s1) return;
  shownTrans.add(null);
  annots.push({x:s1.center,y:1.04,xref:'x',yref:'paper',text:'S1',showarrow:false,
    font:{size:11,color:'#6B625B',family:"'DM Sans',sans-serif"},bgcolor:'#FAF6F0',borderpad:2});
  PLOT_IDS.forEach(id => Plotly.relayout(id,{annotations:[...annots]}));
}

// ── stage history ─────────────────────────────────────────────────────────────
function addHistRow(lbl, val, animate) {
  const el = document.getElementById('stage-history'); if (!el) return;
  const row = document.createElement('div'); row.className='hist-row';
  row.innerHTML = '<span class="hist-lbl">'+lbl+'</span><span class="hist-sep"></span>'
    +'<span class="hist-val">'+fmtH(val)+'</span>';
  el.insertBefore(row, el.firstChild);
  if (animate) requestAnimationFrame(()=>requestAnimationFrame(()=>row.classList.add('open')));
  else row.classList.add('instant');
}
function clearHist() { const el=document.getElementById('stage-history'); if(el) el.innerHTML=''; }
function fmtH(v) {
  if (v===null||v===undefined) return '—';
  if (typeof v==='number' && v>0 && v<1) return (v*100).toFixed(1)+'%';
  if (typeof v==='number') return v.toFixed(2);
  return String(v);
}
function fmt(v, dec) {
  if (v===null||v===undefined||!isFinite(Number(v))) return '—';
  return Number(v).toFixed(dec !== undefined ? dec : 2);
}
function pct(v) { return v===null||v===undefined ? '—' : (Number(v)*100).toFixed(1)+'%'; }

// ── animation ─────────────────────────────────────────────────────────────────
function tgtFrame(elapsed) {
  return Math.min(TOTAL-1, Math.floor((elapsed/(ANIM_MS/SPEEDS[speedIdx]))*TOTAL));
}
function animStep(ts) {
  if (!playing) return;
  if (!startTime) startTime=ts;
  const t = tgtFrame(ts-startTime);
  if (t<=frame) { requestAnimationFrame(animStep); return; }
  extendFrame(frame, t);
  checkTrans(D.episodes[t], true);
  updateTiles(t);
  const p=(t/(TOTAL-1))*100;
  document.getElementById('pb').style.width=p+'%';
  document.getElementById('pb-thumb').style.left=`calc(${p}% - 5px)`;
  frame=t;
  if (frame>=TOTAL-1) {
    animDone=true; playing=false;
    document.getElementById('btn-play').textContent='▶ Play';
    document.getElementById('hover-badge').classList.add('show');
    setupHover(); return;
  }
  requestAnimationFrame(animStep);
}

function extendFrame(from, to) { <<<JS_EXTEND>>> }
function updateTiles(i) { <<<JS_UPDATE_TILES>>> }
function resetTiles() { <<<JS_RESET_TILES>>> }
function resetPlots() { <<<JS_RESET_PLOTS>>> }

// ── init ──────────────────────────────────────────────────────────────────────
<<<JS_INIT>>>
initS1();
requestAnimationFrame(animStep);

// ── hover ─────────────────────────────────────────────────────────────────────
function setupHover() {
  PLOT_IDS.forEach(id => {
    const el = document.getElementById(id); if (!el || !el.on) return;
    el.on('plotly_hover', function(e) {
      const ep=e.points[0].x, idx=D.episodes.findIndex(v=>v===ep);
      if (idx>=0) updateTiles(idx);
    });
  });
}

// ── rebuild ───────────────────────────────────────────────────────────────────
function rebuildToFrame(tgt) {
  if (tgt<0) tgt=0; if (tgt>=TOTAL) tgt=TOTAL-1;
  shapes=[]; annots=[]; shownTrans.clear(); clearHist();
  const s1=D.stageZones.find(z=>z.label==='S1');
  if (s1) { shownTrans.add(null); annots.push({x:s1.center,y:1.04,xref:'x',yref:'paper',
    text:'S1',showarrow:false,font:{size:11,color:'#6B625B',family:"'DM Sans',sans-serif"},
    bgcolor:'#FAF6F0',borderpad:2}); }
  for (let i=0;i<=tgt;i++) {
    for (const z of D.stageZones) {
      if (!shownTrans.has(z.ep) && D.episodes[i]>=(z.ep??-Infinity)) {
        shownTrans.add(z.ep); applyTrans(z,false);
        if (D.completedStageAvgs) {
          const c=D.completedStageAvgs.find(x=>x.transitionEp===z.ep);
          if (c) addHistRow(c.completedLabel,c.finalAvg,false);
        }
      }
    }
  }
  <<<JS_REBUILD>>>
  updateTiles(tgt);
  const p=(tgt/(TOTAL-1))*100;
  document.getElementById('pb').style.width=p+'%';
  document.getElementById('pb-thumb').style.left=`calc(${p}% - 5px)`;
  frame=tgt;
  PLOT_IDS.forEach(id => Plotly.relayout(id,{shapes:[...shapes],annotations:[...annots]}));
}

// ── controls ──────────────────────────────────────────────────────────────────
document.getElementById('btn-play').addEventListener('click',()=>{
  if(animDone) return; playing=!playing;
  document.getElementById('btn-play').textContent=playing?'⏸ Pause':'▶ Play';
  if(playing){ startTime=performance.now()-(frame/(TOTAL-1))*(ANIM_MS/SPEEDS[speedIdx]); requestAnimationFrame(animStep); }
});
document.getElementById('btn-faster').addEventListener('click',()=>{
  if(speedIdx===SPEEDS.length-1) return; speedIdx++;
  document.getElementById('spd-lbl').textContent=SPEEDS[speedIdx]+'×';
  if(playing) startTime=performance.now()-(frame/(TOTAL-1))*(ANIM_MS/SPEEDS[speedIdx]);
});
document.getElementById('btn-slower').addEventListener('click',()=>{
  if(speedIdx===0) return; speedIdx--;
  document.getElementById('spd-lbl').textContent=SPEEDS[speedIdx]+'×';
  if(playing) startTime=performance.now()-(frame/(TOTAL-1))*(ANIM_MS/SPEEDS[speedIdx]);
});
document.getElementById('btn-restart').addEventListener('click',()=>{
  playing=false; animDone=false; frame=0; startTime=null;
  shapes=[]; annots=[]; shownTrans.clear(); clearHist();
  document.getElementById('hover-badge').classList.remove('show');
  document.getElementById('btn-play').textContent='⏸ Pause';
  document.getElementById('pb').style.width='0%';
  resetTiles(); resetPlots();
  const s1=D.stageZones.find(z=>z.label==='S1');
  if(s1){ shownTrans.add(null); annots.push({x:s1.center,y:1.04,xref:'x',yref:'paper',
    text:'S1',showarrow:false,font:{size:11,color:'#6B625B',family:"'DM Sans',sans-serif"},
    bgcolor:'#FAF6F0',borderpad:2}); }
  PLOT_IDS.forEach(id=>Plotly.relayout(id,{shapes:[],annotations:[...annots]}));
  document.getElementById('ep-counter').textContent='';
  setTimeout(()=>{ playing=true; requestAnimationFrame(animStep); },80);
});
document.getElementById('pb-wrap').addEventListener('click',function(e){
  const p=Math.max(0,Math.min(1,e.offsetX/this.offsetWidth));
  playing=false; document.getElementById('btn-play').textContent='▶ Play';
  rebuildToFrame(Math.round(p*(TOTAL-1)));
  if(frame>=TOTAL-1){ animDone=true; document.getElementById('hover-badge').classList.add('show'); setupHover(); }
  else animDone=false;
});
</script>
</body>
</html>"""

def generate_page_html(
    page_title, plot_title, data_json,
    extra_css, tiles_html, plot_divs_html,
    js_defs, js_init, js_extend, js_rebuild,
    js_update_tiles, js_reset_tiles, js_reset_plots,
    pb_color='#7D9D9C',
):
    return (_TEMPLATE
        .replace('<<<TITLE>>>', page_title)
        .replace('<<<PLOT_TITLE>>>', plot_title)
        .replace('<<<DATA_JSON>>>', data_json)
        .replace('<<<EXTRA_CSS>>>', extra_css)
        .replace('<<<TILES_HTML>>>', tiles_html)
        .replace('<<<PLOT_DIVS_HTML>>>', plot_divs_html)
        .replace('<<<PB_COLOR>>>', pb_color)
        .replace('<<<JS_DEFS>>>', js_defs)
        .replace('<<<JS_INIT>>>', js_init)
        .replace('<<<JS_EXTEND>>>', js_extend)
        .replace('<<<JS_REBUILD>>>', js_rebuild)
        .replace('<<<JS_UPDATE_TILES>>>', js_update_tiles)
        .replace('<<<JS_RESET_TILES>>>', js_reset_tiles)
        .replace('<<<JS_RESET_PLOTS>>>', js_reset_plots)
    )

# ── shared CLI helper ─────────────────────────────────────────────────────────

def base_argparser(description):
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--json',    required=True)
    ap.add_argument('--curr',    default=None)
    ap.add_argument('--out',     default='./output')
    ap.add_argument('--rolling', type=int, default=20)
    return ap

def load_and_sort(args):
    import os
    episodes = load_episodes(args.json)
    episodes = sorted(episodes, key=lambda e: safe_float(e.get('episode', 0)) or 0)
    curriculum_transitions = None
    if args.curr and os.path.exists(args.curr):
        curriculum_transitions = load_curriculum(args.curr)
    return episodes, curriculum_transitions

def save_html(html, args, filename):
    import os
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved: {path}  ({os.path.getsize(path)//1024} KB)")
    return path
