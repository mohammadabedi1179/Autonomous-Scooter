#!/usr/bin/env python3
"""
Interactive Animated Success Rate Plot — standalone HTML generator.
Matches thesis_nude_contrast theme from training_analysis_creative_static.py.

Tiles:
  1. Episode number
  2. Current rolling success rate  (fixed green, no reason coloring)
  3. Stage average + animated history (completed stages slide in as rows below)
  4. Peak success rate so far

Usage:
    python interactive_02_success_rate.py --json episode_stats.json --out ./output [--rolling 20]
    python interactive_02_success_rate.py --json episode_stats.json --curr curriculum_log.json --out ./output
"""

import argparse
import json
import math
import os


# ── helpers ───────────────────────────────────────────────────────────────────

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
        win = [v for v in values[j0:i + 1] if v is not None]
        out.append(round(sum(win) / len(win), 6) if win else None)
    return out


def load_episodes(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    for key in ("episodes", "data", "training"):
        if key in raw and isinstance(raw[key], list):
            return raw[key]
    raise ValueError(f"Cannot find episode list in {path}")


def load_curriculum(path):
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    transitions = []
    for entry in log:
        if entry.get("event") == "stage_advanced":
            ep = entry.get("total_episodes")
            if ep is not None:
                transitions.append(int(ep))
    return sorted(transitions)


# ── data preparation ──────────────────────────────────────────────────────────

def build_data(episodes, rolling_w, curriculum_transitions=None):
    ep_nums, successes, stages = [], [], []

    for i, e in enumerate(episodes):
        ep_nums.append(int(safe_float(e.get("episode", i + 1)) or (i + 1)))
        reason = str(e.get("reason", "unknown")).lower().strip()
        successes.append(1 if reason == "goal" else 0)
        try:
            stages.append(int(float(e.get("curriculum_stage", 0))))
        except Exception:
            stages.append(0)

    roll = rolling_mean(successes, rolling_w)

    # Stage transitions
    if curriculum_transitions is not None:
        transitions = []
        for ep in curriculum_transitions:
            idx = next((i for i, e in enumerate(ep_nums) if e >= ep), None)
            if idx is not None:
                transitions.append([ep, stages[idx]])
    else:
        transitions = []
        prev = stages[0] if stages else 0
        for i in range(1, len(stages)):
            if stages[i] != prev:
                transitions.append([ep_nums[i], stages[i]])
                prev = stages[i]

    # Stage zone centers (S1 has no preceding line)
    stage_zones = []
    if transitions:
        first_center = int((ep_nums[0] + transitions[0][0]) / 2)
        stage_zones.append({"ep": None, "center": first_center, "label": "S1"})
        for i, (ep, _sid) in enumerate(transitions):
            next_ep = transitions[i + 1][0] if i + 1 < len(transitions) else ep_nums[-1]
            center  = int((ep + next_ep) / 2)
            stage_zones.append({"ep": ep, "center": center, "label": f"S{i + 2}"})

    # Running stage average (resets per stage)
    stage_avg = []
    stage_sums   = {}
    stage_counts = {}
    for i in range(len(ep_nums)):
        s = stages[i]
        stage_sums[s]   = stage_sums.get(s, 0)   + successes[i]
        stage_counts[s] = stage_counts.get(s, 0) + 1
        stage_avg.append(round(stage_sums[s] / stage_counts[s], 6))

    # Completed stage records: final stageAvg at the last episode of each stage.
    # Entry fires when the NEXT stage begins (at transition ep).
    completed_stage_avgs = []
    for i, (ep, _sid) in enumerate(transitions):
        # last episode index strictly before this transition
        prev_idx = next(
            (j for j in range(len(ep_nums) - 1, -1, -1) if ep_nums[j] < ep),
            None,
        )
        if prev_idx is not None:
            completed_stage_avgs.append({
                "transitionEp":  ep,
                "completedLabel": f"S{i + 1}",   # index-based, matches stageZones
                "finalAvg":       stage_avg[prev_idx],
            })

    # Peak rolling success rate (non-decreasing)
    peak = []
    cur_max = 0.0
    for v in roll:
        if v is not None and v > cur_max:
            cur_max = v
        peak.append(round(cur_max, 6))

    # Active stage label at each episode (for tile sub-label)
    stage_labels = []
    zone_idx = 0
    for i in range(len(ep_nums)):
        while (zone_idx + 1 < len(stage_zones) and
               stage_zones[zone_idx + 1]["ep"] is not None and
               ep_nums[i] >= stage_zones[zone_idx + 1]["ep"]):
            zone_idx += 1
        stage_labels.append(stage_zones[zone_idx]["label"] if stage_zones else "S1")

    return {
        "episodes":            ep_nums,
        "rolling":             roll,
        "stageAvg":            stage_avg,
        "completedStageAvgs":  completed_stage_avgs,
        "peak":                peak,
        "stageLabels":         stage_labels,
        "transitions":         transitions,
        "stageZones":          stage_zones,
        "rollingWindow":       rolling_w,
        "yRange":              [-0.04, 1.16],
        "xRange":              [ep_nums[0], ep_nums[-1]],
        "total":               len(ep_nums),
    }


# ── HTML generation ───────────────────────────────────────────────────────────

def generate_html(data):
    data_json = json.dumps(data, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Success Rate — Interactive</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:      #FAF6F0;
  --bg2:     #F0E8DC;
  --bg3:     #E8DDD2;
  --border:  #D8CABD;
  --text:    #463F3A;
  --muted:   #6B625B;
  --faint:   #9B8E82;
  --goal:    #87986A;
  --peak:    #7D9D9C;
  --stage:   #C9A27E;
  --hist:    #A89080;
  --font:    'DM Sans', -apple-system, sans-serif;
  --mono:    'DM Mono', monospace;
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{
  height: 100%; overflow: hidden;
  background: var(--bg);
  font-family: var(--font);
  color: var(--text);
  font-size: 13px;
}}
#app {{
  display: flex; flex-direction: column;
  height: 100vh; padding: 12px; gap: 10px;
}}
#main-row {{
  display: flex; flex: 1; gap: 10px; min-height: 0;
}}
#left {{
  flex: 1; min-width: 0; position: relative;
  display: flex; flex-direction: column;
}}
#plot-title {{
  font-size: 13px; font-weight: 500;
  color: var(--muted); letter-spacing: .03em;
  padding: 0 4px 6px;
}}
#main-plot {{ flex: 1; min-height: 0; }}

/* ── right panel ─────────────────────────────────────── */
#right {{
  width: 200px; display: flex; flex-direction: column; gap: 8px;
}}
.tile {{
  background: var(--bg2);
  border: 0.5px solid var(--border);
  border-radius: 10px;
  padding: 11px 14px;
}}
/* fixed height tiles */
#tile-ep, #tile-rate, #tile-peak {{ flex: none; }}
/* stage tile grows to absorb history */
#tile-stage {{ flex: 1; min-height: 0; display: flex; flex-direction: column; }}

.tile-lbl {{
  font-size: 10px; font-weight: 500; color: var(--faint);
  letter-spacing: .08em; text-transform: uppercase;
  margin-bottom: 5px; flex-shrink: 0;
}}
.tile-val {{
  font-size: 27px; font-weight: 500;
  line-height: 1.1; font-family: var(--mono);
}}
.tile-sub {{
  font-size: 11px; color: var(--faint); margin-top: 3px;
}}
#v-rate     {{ color: var(--goal); }}
#v-peak     {{ color: var(--peak); }}
#v-ep       {{ color: var(--text); }}
#v-stage-avg{{ color: var(--stage); }}

/* ── stage history list ──────────────────────────────── */
#stage-current {{ flex-shrink: 0; }}
#stage-history {{
  flex: 1; min-height: 0;
  overflow-y: auto; overflow-x: hidden;
  margin-top: 6px;
  /* hide scrollbar cosmetically */
  scrollbar-width: none;
}}
#stage-history::-webkit-scrollbar {{ display: none; }}

.hist-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 0;
  max-height: 0;
  overflow: hidden;
  opacity: 0;
}}
/* animated state — toggled via JS */
.hist-row.open {{
  max-height: 26px;
  opacity: 1;
  padding-top: 4px;
}}
/* non-animated instant state — for seek / restart rebuild */
.hist-row.instant {{
  max-height: 26px;
  opacity: 1;
  padding-top: 4px;
}}
.hist-row.open {{
  transition: max-height 0.45s cubic-bezier(0.4,0,0.2,1),
              opacity    0.38s ease,
              padding-top 0.45s cubic-bezier(0.4,0,0.2,1);
}}
.hist-lbl {{
  font-size: 11px; color: var(--faint);
  font-family: var(--mono);
}}
.hist-sep {{
  flex: 1; margin: 0 6px;
  border-bottom: 0.5px dotted var(--border);
}}
.hist-val {{
  font-size: 12px; font-weight: 500;
  color: var(--hist); font-family: var(--mono);
}}

/* ── controls ────────────────────────────────────────── */
#controls {{
  display: flex; align-items: center;
  gap: 8px; flex-shrink: 0; padding: 2px 0;
}}
.ctrl-btn {{
  background: transparent;
  border: 0.5px solid var(--border);
  border-radius: 6px; padding: 4px 11px;
  font-size: 12px; font-family: var(--font);
  color: var(--text); cursor: pointer;
  white-space: nowrap; transition: background .12s;
}}
.ctrl-btn:hover {{ background: var(--bg2); }}
.ctrl-btn:active {{ background: var(--bg3); }}
#pb-wrap {{
  flex: 1; height: 5px; background: var(--border);
  border-radius: 3px; cursor: pointer; position: relative;
}}
#pb {{
  height: 100%; background: var(--goal);
  border-radius: 3px; width: 0%; pointer-events: none;
}}
#pb-thumb {{
  position: absolute; top: 50%;
  width: 10px; height: 10px; background: var(--goal);
  border-radius: 50%; transform: translateY(-50%);
  transition: opacity .2s; opacity: 0; pointer-events: none;
}}
#pb-wrap:hover #pb-thumb {{ opacity: 1; }}
#spd-lbl {{
  font-size: 11px; color: var(--muted);
  min-width: 30px; text-align: center;
  font-family: var(--mono);
}}
#hover-badge {{
  position: absolute; bottom: 8px; left: 50%;
  transform: translateX(-50%);
  background: var(--bg2); border: 0.5px solid var(--border);
  border-radius: 20px; padding: 4px 14px;
  font-size: 11px; color: var(--muted);
  opacity: 0; transition: opacity .4s;
  pointer-events: none; white-space: nowrap;
}}
#hover-badge.show {{ opacity: 1; }}
#ep-counter {{
  position: absolute; top: 8px; right: 8px;
  font-size: 10px; font-family: var(--mono);
  color: var(--faint); pointer-events: none;
}}
@keyframes peak-flash {{
  0%   {{ transform: scale(1.07); }}
  100% {{ transform: scale(1); }}
}}
.peak-flash {{ animation: peak-flash .35s ease-out; }}
</style>
</head>
<body>
<div id="app">
  <div id="main-row">
    <div id="left">
      <div id="plot-title">Success Rate Over Training</div>
      <div id="main-plot"></div>
      <div id="hover-badge">hover over the line to inspect</div>
      <div id="ep-counter"></div>
    </div>

    <div id="right">
      <!-- tile 1: episode -->
      <div class="tile" id="tile-ep">
        <div class="tile-lbl">Episode</div>
        <div class="tile-val" id="v-ep">—</div>
      </div>

      <!-- tile 2: rolling success rate (fixed green) -->
      <div class="tile" id="tile-rate">
        <div class="tile-lbl">Success rate</div>
        <div class="tile-val" id="v-rate">—</div>
        <div class="tile-sub">rolling w={data["rollingWindow"]}</div>
      </div>

      <!-- tile 3: stage average + animated history -->
      <div class="tile" id="tile-stage">
        <div class="tile-lbl">Stage average</div>
        <div id="stage-current">
          <div class="tile-val" id="v-stage-avg">—</div>
          <div class="tile-sub" id="v-stage-sub">—</div>
        </div>
        <!-- completed stage rows slide in here -->
        <div id="stage-history"></div>
      </div>

      <!-- tile 4: peak so far -->
      <div class="tile" id="tile-peak">
        <div class="tile-lbl">Peak so far</div>
        <div class="tile-val" id="v-peak">—</div>
        <div class="tile-sub">max rolling reached</div>
      </div>
    </div>
  </div>

  <div id="controls">
    <button class="ctrl-btn" id="btn-play">⏸ Pause</button>
    <div id="pb-wrap">
      <div id="pb"></div>
      <div id="pb-thumb"></div>
    </div>
    <button class="ctrl-btn" id="btn-slower">−</button>
    <span id="spd-lbl">1×</span>
    <button class="ctrl-btn" id="btn-faster">+</button>
    <button class="ctrl-btn" id="btn-restart">↺ Restart</button>
  </div>
</div>

<script>
const D = {data_json};

const SPEEDS  = [0.25, 0.5, 1, 2, 4, 8];
const ANIM_MS = 18000;
let speedIdx = 2, playing = true, frame = 0;
let startTime = null, animDone = false;
let shownTrans = new Set();
let shapes = [], annots = [];
let lastPeak = 0;
const TOTAL = D.total;

/* ── Plotly layout ───────────────────────────────────────────────────────── */
const BASE_LAYOUT = {{
  paper_bgcolor: '#FAF6F0', plot_bgcolor: '#FAF6F0',
  font: {{ family: "'DM Sans', sans-serif", color: '#463F3A', size: 12 }},
  margin: {{ t: 38, r: 22, b: 48, l: 66 }},
  xaxis: {{
    title: {{ text: 'Episode', font: {{ size: 12 }} }},
    gridcolor: '#E8DDD2', linecolor: '#D8CABD',
    tickcolor: '#6B625B', tickfont: {{ size: 11 }},
    range: D.xRange, zeroline: false, fixedrange: false,
  }},
  yaxis: {{
    title: {{ text: 'Success rate', font: {{ size: 12 }} }},
    gridcolor: '#E8DDD2', linecolor: '#D8CABD',
    tickcolor: '#6B625B', tickfont: {{ size: 11 }},
    range: D.yRange, zeroline: true,
    zerolinecolor: '#D0C4B8', zerolinewidth: 1,
    tickformat: '.0%', fixedrange: false,
  }},
  legend: {{
    x: 0.99, y: 0.01, xanchor: 'right', yanchor: 'bottom',
    bgcolor: 'rgba(250,246,240,0.92)', bordercolor: '#D8CABD', borderwidth: 1,
    font: {{ size: 11 }},
  }},
  shapes: [], annotations: [],
  hovermode: 'x unified',
  hoverlabel: {{
    bgcolor: '#FAF6F0', bordercolor: '#D8CABD',
    font: {{ family: "'DM Mono', monospace", size: 11, color: '#463F3A' }},
  }},
}};

const BASE_TRACES = [{{
  name: `Success rate (rolling w=${{D.rollingWindow}})`,
  x: [], y: [],
  mode: 'lines', type: 'scatter',
  fill: 'tozeroy', fillcolor: 'rgba(135,152,106,0.16)',
  line: {{ color: '#87986A', width: 2.5 }},
  hovertemplate: 'Ep <b>%{{x}}</b>  —  %{{y:.1%}}<extra></extra>',
}}];

Plotly.newPlot('main-plot', BASE_TRACES, BASE_LAYOUT, {{
  responsive: true, displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
}});

/* ── tile helpers ────────────────────────────────────────────────────────── */
function pct(v) {{
  if (v === null || v === undefined) return '—';
  return (Number(v) * 100).toFixed(1) + '%';
}}

function updateTiles(i) {{
  document.getElementById('v-ep').textContent        = D.episodes[i].toLocaleString();
  document.getElementById('v-rate').textContent      = pct(D.rolling[i]);
  document.getElementById('v-stage-avg').textContent = pct(D.stageAvg[i]);
  document.getElementById('v-stage-sub').textContent = D.stageLabels[i] + ' · running avg';

  const pk = D.peak[i];
  document.getElementById('v-peak').textContent = pct(pk);
  document.getElementById('ep-counter').textContent  = `${{i + 1}} / ${{TOTAL}}`;

  if (pk > lastPeak + 0.001) {{
    lastPeak = pk;
    const el = document.getElementById('v-peak');
    el.classList.remove('peak-flash');
    void el.offsetWidth;
    el.classList.add('peak-flash');
  }}
}}

/* ── stage history ───────────────────────────────────────────────────────── */
/**
 * Add a completed-stage row to the history list.
 * animate=true  → accordion slide-in (for live animation)
 * animate=false → instant display    (for seek / restart rebuild)
 */
function addHistoryRow(label, avg, animate) {{
  const list = document.getElementById('stage-history');
  const row  = document.createElement('div');
  row.className = 'hist-row';
  row.dataset.label = label;
  row.innerHTML =
    `<span class="hist-lbl">${{label}}</span>` +
    `<span class="hist-sep"></span>` +
    `<span class="hist-val">${{pct(avg)}}</span>`;

  /* newest completed stage goes to the top of the list */
  list.insertBefore(row, list.firstChild);

  if (animate) {{
    /* next frame: add .open to trigger CSS transition */
    requestAnimationFrame(() => {{
      requestAnimationFrame(() => {{ row.classList.add('open'); }});
    }});
  }} else {{
    row.classList.add('instant');
  }}
}}

function clearHistory() {{
  document.getElementById('stage-history').innerHTML = '';
}}

/* ── stage transitions ───────────────────────────────────────────────────── */
function applyTransition(zone, animate) {{
  if (zone.ep !== null) {{
    shapes.push({{
      type: 'line', x0: zone.ep, x1: zone.ep, y0: 0, y1: 1, yref: 'paper',
      line: {{ color: '#9B8E82', width: 1.1, dash: 'dash' }},
    }});
  }}
  annots.push({{
    x: zone.center, y: 1.04, xref: 'x', yref: 'paper',
    text: zone.label, showarrow: false,
    font: {{ size: 11, color: '#6B625B', family: "'DM Sans', sans-serif" }},
    bgcolor: '#FAF6F0', borderpad: 2,
  }});
  if (animate) {{
    Plotly.relayout('main-plot', {{ shapes: [...shapes], annotations: [...annots] }});
  }}
}}

function checkTransitions(currentEp, animate) {{
  for (const zone of D.stageZones) {{
    if (!shownTrans.has(zone.ep) && currentEp >= (zone.ep ?? -Infinity)) {{
      shownTrans.add(zone.ep);
      applyTransition(zone, animate !== false);

      /* add completed stage row to history */
      const completed = D.completedStageAvgs.find(c => c.transitionEp === zone.ep);
      if (completed) {{
        addHistoryRow(completed.completedLabel, completed.finalAvg, animate !== false);
      }}
    }}
  }}
}}

/* ── animation ───────────────────────────────────────────────────────────── */
function targetFrame(elapsed) {{
  const dur = ANIM_MS / SPEEDS[speedIdx];
  return Math.min(TOTAL - 1, Math.floor((elapsed / dur) * TOTAL));
}}

function animStep(ts) {{
  if (!playing) return;
  if (!startTime) startTime = ts;
  const target = targetFrame(ts - startTime);
  if (target <= frame) {{ requestAnimationFrame(animStep); return; }}

  const newX = [], newY = [];
  for (let i = frame + 1; i <= target; i++) {{
    const v = D.rolling[i];
    if (v !== null) {{ newX.push(D.episodes[i]); newY.push(v); }}
  }}
  if (newX.length) {{
    Plotly.extendTraces('main-plot', {{ x: [newX], y: [newY] }}, [0]);
  }}

  checkTransitions(D.episodes[target], true);
  updateTiles(target);

  const p = (target / (TOTAL - 1)) * 100;
  document.getElementById('pb').style.width      = p + '%';
  document.getElementById('pb-thumb').style.left = `calc(${{p}}% - 5px)`;

  frame = target;
  if (frame >= TOTAL - 1) {{
    animDone = true; playing = false;
    document.getElementById('btn-play').textContent = '▶ Play';
    document.getElementById('hover-badge').classList.add('show');
    setupHover();
    return;
  }}
  requestAnimationFrame(animStep);
}}

/* S1 annotation at startup */
(function initS1() {{
  const s1 = D.stageZones.find(z => z.label === 'S1');
  if (!s1) return;
  shownTrans.add(null);
  annots.push({{
    x: s1.center, y: 1.04, xref: 'x', yref: 'paper',
    text: 'S1', showarrow: false,
    font: {{ size: 11, color: '#6B625B', family: "'DM Sans', sans-serif" }},
    bgcolor: '#FAF6F0', borderpad: 2,
  }});
  Plotly.relayout('main-plot', {{ annotations: [...annots] }});
}})();

requestAnimationFrame(animStep);

/* ── hover ───────────────────────────────────────────────────────────────── */
function setupHover() {{
  document.getElementById('main-plot').on('plotly_hover', function(evt) {{
    const ep  = evt.points[0].x;
    const idx = D.episodes.findIndex(e => e === ep);
    if (idx >= 0) updateTiles(idx);
  }});
}}

/* ── rebuild to frame (seek + restart) ──────────────────────────────────── */
function initS1Annot() {{
  const s1 = D.stageZones.find(z => z.label === 'S1');
  if (!s1) return;
  shownTrans.add(null);
  annots.push({{
    x: s1.center, y: 1.04, xref: 'x', yref: 'paper',
    text: 'S1', showarrow: false,
    font: {{ size: 11, color: '#6B625B', family: "'DM Sans', sans-serif" }},
    bgcolor: '#FAF6F0', borderpad: 2,
  }});
}}

function rebuildToFrame(targetIdx) {{
  if (targetIdx < 0) targetIdx = 0;
  if (targetIdx >= TOTAL) targetIdx = TOTAL - 1;

  shapes = []; annots = [];
  shownTrans.clear();
  lastPeak = 0;
  clearHistory();
  initS1Annot();

  const lineX = [], lineY = [];
  for (let i = 0; i <= targetIdx; i++) {{
    const v = D.rolling[i];
    if (v !== null) {{ lineX.push(D.episodes[i]); lineY.push(v); }}
    /* collect transitions without individual relayout */
    for (const zone of D.stageZones) {{
      if (!shownTrans.has(zone.ep) && D.episodes[i] >= (zone.ep ?? -Infinity)) {{
        shownTrans.add(zone.ep);
        applyTransition(zone, false);   /* collect shapes/annots, no relayout yet */
        const c = D.completedStageAvgs.find(x => x.transitionEp === zone.ep);
        if (c) addHistoryRow(c.completedLabel, c.finalAvg, false);
      }}
    }}
  }}

  /* single Plotly.react call with all shapes + annots */
  Plotly.react('main-plot',
    [{{ ...BASE_TRACES[0], x: lineX, y: lineY }}],
    {{ ...BASE_LAYOUT, shapes: [...shapes], annotations: [...annots] }});

  lastPeak = D.peak[targetIdx] || 0;
  updateTiles(targetIdx);

  const p = (targetIdx / (TOTAL - 1)) * 100;
  document.getElementById('pb').style.width      = p + '%';
  document.getElementById('pb-thumb').style.left = `calc(${{p}}% - 5px)`;
  frame = targetIdx;
}}

/* ── controls ────────────────────────────────────────────────────────────── */
document.getElementById('btn-play').addEventListener('click', () => {{
  if (animDone) return;
  playing = !playing;
  document.getElementById('btn-play').textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) {{
    startTime = performance.now() - (frame / (TOTAL - 1)) * (ANIM_MS / SPEEDS[speedIdx]);
    requestAnimationFrame(animStep);
  }}
}});

document.getElementById('btn-faster').addEventListener('click', () => {{
  if (speedIdx === SPEEDS.length - 1) return;
  speedIdx++;
  document.getElementById('spd-lbl').textContent = SPEEDS[speedIdx] + '×';
  if (playing) startTime = performance.now() - (frame / (TOTAL - 1)) * (ANIM_MS / SPEEDS[speedIdx]);
}});

document.getElementById('btn-slower').addEventListener('click', () => {{
  if (speedIdx === 0) return;
  speedIdx--;
  document.getElementById('spd-lbl').textContent = SPEEDS[speedIdx] + '×';
  if (playing) startTime = performance.now() - (frame / (TOTAL - 1)) * (ANIM_MS / SPEEDS[speedIdx]);
}});

document.getElementById('btn-restart').addEventListener('click', () => {{
  playing = false; animDone = false;
  frame = 0; startTime = null; lastPeak = 0;
  shapes = []; annots = [];
  shownTrans.clear();
  clearHistory();
  document.getElementById('hover-badge').classList.remove('show');
  document.getElementById('btn-play').textContent = '⏸ Pause';
  document.getElementById('pb').style.width = '0%';

  initS1Annot();
  Plotly.react('main-plot',
    [{{ ...BASE_TRACES[0], x: [], y: [] }}],
    {{ ...BASE_LAYOUT, shapes: [], annotations: [...annots] }});

  ['v-ep','v-rate','v-stage-avg','v-peak'].forEach(id =>
    document.getElementById(id).textContent = '—');
  document.getElementById('v-stage-sub').textContent = '—';
  document.getElementById('ep-counter').textContent = '';

  setTimeout(() => {{ playing = true; requestAnimationFrame(animStep); }}, 80);
}});

document.getElementById('pb-wrap').addEventListener('click', function(e) {{
  const p = Math.max(0, Math.min(1, e.offsetX / this.offsetWidth));
  playing = false;
  document.getElementById('btn-play').textContent = '▶ Play';
  rebuildToFrame(Math.round(p * (TOTAL - 1)));
  if (frame >= TOTAL - 1) {{
    animDone = true;
    document.getElementById('hover-badge').classList.add('show');
    setupHover();
  }} else {{
    animDone = false;
  }}
}});
</script>
</body>
</html>"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate interactive animated success rate plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--json",    required=True,       help="Path to episode_stats.json")
    ap.add_argument("--curr",    default=None,         help="Path to curriculum_log.json (optional)")
    ap.add_argument("--out",     default="./output",   help="Output directory")
    ap.add_argument("--rolling", type=int, default=20, help="Rolling window size")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading: {args.json}")
    episodes = load_episodes(args.json)
    episodes = sorted(episodes, key=lambda e: safe_float(e.get("episode", 0)) or 0)
    print(f"  {len(episodes)} episodes loaded.")

    curriculum_transitions = None
    if args.curr and os.path.exists(args.curr):
        curriculum_transitions = load_curriculum(args.curr)
        print(f"  Curriculum log: {len(curriculum_transitions)} stage transitions.")

    data = build_data(episodes, args.rolling, curriculum_transitions)
    print(f"  Stage zones:           {len(data['stageZones'])}")
    print(f"  Completed stage avgs:  {len(data['completedStageAvgs'])}")
    for c in data['completedStageAvgs']:
        print(f"    {c['completedLabel']} final avg: {c['finalAvg']:.3f}")
    print(f"  Peak reached:          {max(data['peak']):.3f}")

    html    = generate_html(data)
    outpath = os.path.join(args.out, "02_success_rate_interactive.html")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(outpath) / 1024
    print(f"\nSaved: {outpath}  ({size_kb:.1f} KB)")
    print("Open in any browser — fully self-contained, no server needed.")


if __name__ == "__main__":
    main()
