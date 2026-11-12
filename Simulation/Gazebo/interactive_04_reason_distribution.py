#!/usr/bin/env python3
"""
Interactive Animated Reason Distribution Plot — standalone HTML generator.
Matches thesis_nude_contrast theme from training_analysis_creative_static.py.

Tiles:
  1. Episode number
  2. Live fractions — animated mini progress bars (goal / collision / timeout)
  3. Stage goal average + animated history (completed stages slide in)
  4. Collision-free streak — consecutive non-collision episodes

Usage:
    python interactive_04_reason_distribution.py --json episode_stats.json --out ./output [--rolling 20]
    python interactive_04_reason_distribution.py --json episode_stats.json --curr curriculum_log.json --out ./output
"""

import argparse
import json
import math
import os
from collections import Counter


# ── helpers ───────────────────────────────────────────────────────────────────

def safe_float(x, default=None):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


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
    ep_nums, reasons, stages = [], [], []

    for i, e in enumerate(episodes):
        ep_nums.append(int(safe_float(e.get("episode", i + 1)) or (i + 1)))
        reasons.append(str(e.get("reason", "unknown")).lower().strip())
        try:
            stages.append(int(float(e.get("curriculum_stage", 0))))
        except Exception:
            stages.append(0)

    # Rolling fractions for all three reasons
    roll_goal, roll_collision, roll_timeout = [], [], []
    for i in range(len(reasons)):
        j0 = max(0, i - rolling_w + 1)
        window = reasons[j0:i + 1]
        total  = len(window)
        cnt    = Counter(window)
        roll_goal.append(     round(cnt.get("goal",      0) / total, 6))
        roll_collision.append(round(cnt.get("collision", 0) / total, 6))
        roll_timeout.append(  round(cnt.get("timeout",   0) / total, 6))

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

    # Stage zones (S1 has no preceding line)
    stage_zones = []
    if transitions:
        first_center = int((ep_nums[0] + transitions[0][0]) / 2)
        stage_zones.append({"ep": None, "center": first_center, "label": "S1"})
        for i, (ep, _sid) in enumerate(transitions):
            next_ep = transitions[i + 1][0] if i + 1 < len(transitions) else ep_nums[-1]
            stage_zones.append({"ep": ep, "center": int((ep + next_ep) / 2), "label": f"S{i + 2}"})

    # Running stage goal fraction (resets per stage)
    stage_goal_avg = []
    stage_sums   = {}
    stage_counts = {}
    for i in range(len(ep_nums)):
        s = stages[i]
        stage_sums[s]   = stage_sums.get(s,   0) + (1 if reasons[i] == "goal" else 0)
        stage_counts[s] = stage_counts.get(s, 0) + 1
        stage_goal_avg.append(round(stage_sums[s] / stage_counts[s], 6))

    # Completed stage goal averages (fires when next stage begins)
    completed_stage_avgs = []
    for i, (ep, _sid) in enumerate(transitions):
        prev_idx = next(
            (j for j in range(len(ep_nums) - 1, -1, -1) if ep_nums[j] < ep),
            None,
        )
        if prev_idx is not None:
            completed_stage_avgs.append({
                "transitionEp":   ep,
                "completedLabel": f"S{i + 1}",
                "finalGoalAvg":   stage_goal_avg[prev_idx],
            })

    # Collision-free streak at each episode
    streak, cur_streak = [], 0
    for r in reasons:
        cur_streak = cur_streak + 1 if r != "collision" else 0
        streak.append(cur_streak)

    # Peak streak (non-decreasing max, for flash animation)
    peak_streak, cur_max = [], 0
    for s in streak:
        cur_max = max(cur_max, s)
        peak_streak.append(cur_max)

    # Active stage label per episode
    stage_labels = []
    zone_idx = 0
    for i in range(len(ep_nums)):
        while (zone_idx + 1 < len(stage_zones) and
               stage_zones[zone_idx + 1]["ep"] is not None and
               ep_nums[i] >= stage_zones[zone_idx + 1]["ep"]):
            zone_idx += 1
        stage_labels.append(stage_zones[zone_idx]["label"] if stage_zones else "S1")

    return {
        "episodes":           ep_nums,
        "rollGoal":           roll_goal,
        "rollCollision":      roll_collision,
        "rollTimeout":        roll_timeout,
        "stageGoalAvg":       stage_goal_avg,
        "completedStageAvgs": completed_stage_avgs,
        "streak":             streak,
        "peakStreak":         peak_streak,
        "stageLabels":        stage_labels,
        "transitions":        transitions,
        "stageZones":         stage_zones,
        "rollingWindow":      rolling_w,
        "yRange":             [-0.02, 1.16],
        "xRange":             [ep_nums[0], ep_nums[-1]],
        "total":              len(ep_nums),
    }


# ── HTML generation ───────────────────────────────────────────────────────────

def generate_html(data):
    data_json = json.dumps(data, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reason Distribution — Interactive</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:        #FAF6F0;
  --bg2:       #F0E8DC;
  --bg3:       #E8DDD2;
  --border:    #D8CABD;
  --text:      #463F3A;
  --muted:     #6B625B;
  --faint:     #9B8E82;
  --goal:      #87986A;
  --collision: #A44A3F;
  --timeout:   #C98245;
  --stage:     #C9A27E;
  --hist:      #A89080;
  --font:      'DM Sans', -apple-system, sans-serif;
  --mono:      'DM Mono', monospace;
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{
  height: 100%; overflow: hidden;
  background: var(--bg); font-family: var(--font);
  color: var(--text); font-size: 13px;
}}
#app {{
  display: flex; flex-direction: column;
  height: 100vh; padding: 12px; gap: 10px;
}}
#main-row {{ display: flex; flex: 1; gap: 10px; min-height: 0; }}
#left {{
  flex: 1; min-width: 0; position: relative;
  display: flex; flex-direction: column;
}}
#plot-title {{
  font-size: 13px; font-weight: 500; color: var(--muted);
  letter-spacing: .03em; padding: 0 4px 6px;
}}
#main-plot {{ flex: 1; min-height: 0; }}
#right {{ width: 200px; display: flex; flex-direction: column; gap: 8px; }}

/* ── tiles ───────────────────────────────────────────── */
.tile {{
  background: var(--bg2); border: 0.5px solid var(--border);
  border-radius: 10px; padding: 11px 14px;
}}
#tile-ep, #tile-streak {{ flex: none; }}
#tile-fracs {{ flex: none; }}
#tile-stage {{ flex: 1; min-height: 0; display: flex; flex-direction: column; }}

.tile-lbl {{
  font-size: 10px; font-weight: 500; color: var(--faint);
  letter-spacing: .08em; text-transform: uppercase; margin-bottom: 6px;
}}
.tile-val {{
  font-size: 27px; font-weight: 500; line-height: 1.1;
  font-family: var(--mono); color: var(--text);
}}
.tile-sub {{ font-size: 11px; color: var(--faint); margin-top: 3px; }}

/* ── live fraction bars (tile 2) ─────────────────────── */
.frac-row {{
  display: flex; align-items: center;
  gap: 7px; margin-bottom: 7px;
}}
.frac-row:last-child {{ margin-bottom: 0; }}
.frac-name {{
  font-size: 10px; font-weight: 500; letter-spacing: .04em;
  text-transform: uppercase; width: 46px; flex-shrink: 0;
}}
.frac-name.goal      {{ color: var(--goal); }}
.frac-name.collision {{ color: var(--collision); }}
.frac-name.timeout   {{ color: var(--timeout); }}
.frac-track {{
  flex: 1; height: 5px;
  background: var(--bg3); border-radius: 3px; overflow: hidden;
}}
.frac-fill {{
  height: 100%; border-radius: 3px; width: 0%;
  transition: width 0.18s ease;
}}
.frac-fill.goal      {{ background: var(--goal); }}
.frac-fill.collision {{ background: var(--collision); }}
.frac-fill.timeout   {{ background: var(--timeout); }}
.frac-pct {{
  font-size: 11px; font-family: var(--mono);
  color: var(--muted); width: 36px; text-align: right;
}}

/* ── stage history (tile 3) ──────────────────────────── */
#stage-current {{ flex-shrink: 0; }}
#stage-history {{
  flex: 1; min-height: 0; overflow-y: auto; overflow-x: hidden;
  margin-top: 6px; scrollbar-width: none;
}}
#stage-history::-webkit-scrollbar {{ display: none; }}
.hist-row {{
  display: flex; justify-content: space-between; align-items: center;
  max-height: 0; overflow: hidden; opacity: 0; padding-top: 0;
}}
.hist-row.open {{
  max-height: 26px; opacity: 1; padding-top: 4px;
  transition: max-height 0.45s cubic-bezier(0.4,0,0.2,1),
              opacity    0.38s ease,
              padding-top 0.45s cubic-bezier(0.4,0,0.2,1);
}}
.hist-row.instant {{ max-height: 26px; opacity: 1; padding-top: 4px; }}
.hist-lbl  {{ font-size: 11px; color: var(--faint); font-family: var(--mono); }}
.hist-sep  {{ flex: 1; margin: 0 6px; border-bottom: 0.5px dotted var(--border); }}
.hist-val  {{ font-size: 12px; font-weight: 500; color: var(--hist); font-family: var(--mono); }}

/* ── streak tile ─────────────────────────────────────── */
#v-streak {{ color: var(--goal); }}

/* ── controls ────────────────────────────────────────── */
#controls {{
  display: flex; align-items: center;
  gap: 8px; flex-shrink: 0; padding: 2px 0;
}}
.ctrl-btn {{
  background: transparent; border: 0.5px solid var(--border);
  border-radius: 6px; padding: 4px 11px;
  font-size: 12px; font-family: var(--font);
  color: var(--text); cursor: pointer; white-space: nowrap;
  transition: background .12s;
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
  min-width: 30px; text-align: center; font-family: var(--mono);
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
@keyframes streak-flash {{
  0%   {{ transform: scale(1.08); }}
  100% {{ transform: scale(1); }}
}}
.streak-flash {{ animation: streak-flash .35s ease-out; }}
</style>
</head>
<body>
<div id="app">
  <div id="main-row">
    <div id="left">
      <div id="plot-title">Termination Reason Distribution Over Training</div>
      <div id="main-plot"></div>
      <div id="hover-badge">hover over the chart to inspect</div>
      <div id="ep-counter"></div>
    </div>

    <div id="right">

      <!-- tile 1: episode -->
      <div class="tile" id="tile-ep">
        <div class="tile-lbl">Episode</div>
        <div class="tile-val" id="v-ep">—</div>
      </div>

      <!-- tile 2: live fractions with progress bars -->
      <div class="tile" id="tile-fracs">
        <div class="tile-lbl">Current fractions <span style="font-weight:400;color:var(--faint)">w={data["rollingWindow"]}</span></div>
        <div class="frac-row">
          <span class="frac-name goal">Goal</span>
          <div class="frac-track"><div class="frac-fill goal" id="bar-goal"></div></div>
          <span class="frac-pct" id="pct-goal">—</span>
        </div>
        <div class="frac-row">
          <span class="frac-name collision">Crash</span>
          <div class="frac-track"><div class="frac-fill collision" id="bar-col"></div></div>
          <span class="frac-pct" id="pct-col">—</span>
        </div>
        <div class="frac-row">
          <span class="frac-name timeout">Timeout</span>
          <div class="frac-track"><div class="frac-fill timeout" id="bar-tmt"></div></div>
          <span class="frac-pct" id="pct-tmt">—</span>
        </div>
      </div>

      <!-- tile 3: stage goal avg + history -->
      <div class="tile" id="tile-stage">
        <div class="tile-lbl">Stage goal avg</div>
        <div id="stage-current">
          <div class="tile-val" id="v-stage-avg" style="color:var(--stage)">—</div>
          <div class="tile-sub" id="v-stage-sub">—</div>
        </div>
        <div id="stage-history"></div>
      </div>

      <!-- tile 4: collision-free streak -->
      <div class="tile" id="tile-streak">
        <div class="tile-lbl">Collision-free streak</div>
        <div class="tile-val" id="v-streak">—</div>
        <div class="tile-sub" id="v-streak-sub">consecutive episodes</div>
      </div>

    </div>
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
const D = {data_json};

const SPEEDS  = [0.25, 0.5, 1, 2, 4, 8];
const ANIM_MS = 18000;
let speedIdx = 2, playing = true, frame = 0;
let startTime = null, animDone = false;
let shownTrans = new Set();
let shapes = [], annots = [];
let lastPeakStreak = 0;
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
    title: {{ text: 'Fraction of episodes (rolling)', font: {{ size: 12 }} }},
    gridcolor: '#E8DDD2', linecolor: '#D8CABD',
    tickcolor: '#6B625B', tickfont: {{ size: 11 }},
    range: D.yRange, zeroline: false,
    tickformat: '.0%', fixedrange: false,
  }},
  legend: {{
    x: 0.99, y: 0.99, xanchor: 'right', yanchor: 'top',
    bgcolor: 'rgba(250,246,240,0.92)', bordercolor: '#D8CABD', borderwidth: 1,
    font: {{ size: 11 }},
  }},
  shapes: [], annotations: [],
  hovermode: 'x unified',
  hoverlabel: {{
    bgcolor: '#FAF6F0', bordercolor: '#D8CABD',
    font: {{ family: "'DM Mono', monospace", size: 11, color: '#463F3A' }},
  }},
  barmode: 'stack',
}};

/* Stacking order bottom→top: collision, goal, timeout  */
const BASE_TRACES = [
  {{
    name: 'Reason: collision', x: [], y: [],
    stackgroup: 'one', mode: 'none',
    fillcolor: 'rgba(164,74,63,0.72)',
    line: {{ color: '#A44A3F', width: 0.6 }},
    hovertemplate: 'Collision: %{{y:.1%}}<extra></extra>',
  }},
  {{
    name: 'Reason: goal', x: [], y: [],
    stackgroup: 'one', mode: 'none',
    fillcolor: 'rgba(135,152,106,0.72)',
    line: {{ color: '#87986A', width: 0.6 }},
    hovertemplate: 'Goal: %{{y:.1%}}<extra></extra>',
  }},
  {{
    name: 'Reason: timeout', x: [], y: [],
    stackgroup: 'one', mode: 'none',
    fillcolor: 'rgba(201,130,69,0.72)',
    line: {{ color: '#C98245', width: 0.6 }},
    hovertemplate: 'Timeout: %{{y:.1%}}<extra></extra>',
  }},
];

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
  const g  = D.rollGoal[i],  c  = D.rollCollision[i], t  = D.rollTimeout[i];
  const sg = D.stageGoalAvg[i];
  const sk = D.streak[i];

  document.getElementById('v-ep').textContent = D.episodes[i].toLocaleString();

  /* fraction bars */
  const setBar = (fillId, pctId, val) => {{
    document.getElementById(fillId).style.width = ((val || 0) * 100) + '%';
    document.getElementById(pctId).textContent  = pct(val);
  }};
  setBar('bar-goal', 'pct-goal', g);
  setBar('bar-col',  'pct-col',  c);
  setBar('bar-tmt',  'pct-tmt',  t);

  /* stage avg */
  document.getElementById('v-stage-avg').textContent = pct(sg);
  document.getElementById('v-stage-sub').textContent = D.stageLabels[i] + ' · running avg';

  /* streak */
  document.getElementById('v-streak').textContent = sk.toLocaleString();
  document.getElementById('ep-counter').textContent = `${{i + 1}} / ${{TOTAL}}`;

  /* flash streak tile on new peak */
  const pk = D.peakStreak[i];
  if (pk > lastPeakStreak) {{
    lastPeakStreak = pk;
    const el = document.getElementById('v-streak');
    el.classList.remove('streak-flash');
    void el.offsetWidth;
    el.classList.add('streak-flash');
  }}
}}

/* ── stage history ───────────────────────────────────────────────────────── */
function addHistoryRow(label, goalAvg, animate) {{
  const list = document.getElementById('stage-history');
  const row  = document.createElement('div');
  row.className = 'hist-row';
  row.innerHTML =
    `<span class="hist-lbl">${{label}}</span>` +
    `<span class="hist-sep"></span>` +
    `<span class="hist-val">${{pct(goalAvg)}}</span>`;
  list.insertBefore(row, list.firstChild);
  if (animate) {{
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
      const c = D.completedStageAvgs.find(x => x.transitionEp === zone.ep);
      if (c) addHistoryRow(c.completedLabel, c.finalGoalAvg, animate !== false);
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

  const newEp = [], newCol = [], newGoal = [], newTmt = [];
  for (let i = frame + 1; i <= target; i++) {{
    newEp.push(D.episodes[i]);
    newCol.push(D.rollCollision[i]);
    newGoal.push(D.rollGoal[i]);
    newTmt.push(D.rollTimeout[i]);
  }}
  if (newEp.length) {{
    Plotly.extendTraces('main-plot', {{
      x: [newEp, newEp, newEp],
      y: [newCol, newGoal, newTmt],
    }}, [0, 1, 2]);
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
  shownTrans.clear(); lastPeakStreak = 0;
  clearHistory(); initS1Annot();

  const ep = [], col = [], goal = [], tmt = [];
  for (let i = 0; i <= targetIdx; i++) {{
    ep.push(D.episodes[i]);
    col.push(D.rollCollision[i]);
    goal.push(D.rollGoal[i]);
    tmt.push(D.rollTimeout[i]);
    for (const zone of D.stageZones) {{
      if (!shownTrans.has(zone.ep) && D.episodes[i] >= (zone.ep ?? -Infinity)) {{
        shownTrans.add(zone.ep);
        applyTransition(zone, false);
        const c = D.completedStageAvgs.find(x => x.transitionEp === zone.ep);
        if (c) addHistoryRow(c.completedLabel, c.finalGoalAvg, false);
      }}
    }}
  }}

  Plotly.react('main-plot',
    [
      {{ ...BASE_TRACES[0], x: ep, y: col  }},
      {{ ...BASE_TRACES[1], x: ep, y: goal }},
      {{ ...BASE_TRACES[2], x: ep, y: tmt  }},
    ],
    {{ ...BASE_LAYOUT, shapes: [...shapes], annotations: [...annots] }});

  lastPeakStreak = D.peakStreak[targetIdx] || 0;
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
  frame = 0; startTime = null; lastPeakStreak = 0;
  shapes = []; annots = []; shownTrans.clear();
  clearHistory();
  document.getElementById('hover-badge').classList.remove('show');
  document.getElementById('btn-play').textContent = '⏸ Pause';
  document.getElementById('pb').style.width = '0%';

  initS1Annot();
  Plotly.react('main-plot',
    BASE_TRACES.map(t => ({{ ...t, x: [], y: [] }})),
    {{ ...BASE_LAYOUT, shapes: [], annotations: [...annots] }});

  ['v-ep','v-stage-avg','v-streak'].forEach(id =>
    document.getElementById(id).textContent = '—');
  ['pct-goal','pct-col','pct-tmt'].forEach(id =>
    document.getElementById(id).textContent = '—');
  ['bar-goal','bar-col','bar-tmt'].forEach(id =>
    document.getElementById(id).style.width = '0%');
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
  }} else {{ animDone = false; }}
}});
</script>
</body>
</html>"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate interactive animated reason distribution plot.",
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
        print(f"    {c['completedLabel']} goal avg: {c['finalGoalAvg']:.3f}")
    print(f"  Max streak:            {max(data['peakStreak'])}")

    html    = generate_html(data)
    outpath = os.path.join(args.out, "04_reason_distribution_interactive.html")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(outpath) / 1024
    print(f"\nSaved: {outpath}  ({size_kb:.1f} KB)")
    print("Open in any browser — fully self-contained, no server needed.")


if __name__ == "__main__":
    main()
