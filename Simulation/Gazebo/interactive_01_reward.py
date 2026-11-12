#!/usr/bin/env python3
"""
Interactive Animated Reward Plot — standalone HTML generator.
Matches thesis_nude_contrast theme from training_analysis_creative_static.py.

Usage:
    python interactive_01_reward.py --json episode_stats.json --out ./output [--rolling 20]
    python interactive_01_reward.py --json episode_stats.json --curr curriculum_log.json --out ./output
"""

import argparse
import json
import math
import os


# ── helpers (mirror of main script) ─────────────────────────────────────────

def safe_float(x, default=None):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def rolling_mean(values, w):
    """Pure-Python rolling mean; returns None for missing windows."""
    out = []
    for i in range(len(values)):
        j0 = max(0, i - w + 1)
        win = [v for v in values[j0:i + 1] if v is not None]
        out.append(round(sum(win) / len(win), 4) if win else None)
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
    """Extract stage transition episode numbers from curriculum log."""
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    transitions = []
    for entry in log:
        if entry.get("event") == "stage_advanced":
            ep = entry.get("total_episodes")
            if ep is not None:
                transitions.append(int(ep))
    return sorted(transitions)


# ── data preparation ─────────────────────────────────────────────────────────

def build_data(episodes, rolling_w, curriculum_transitions=None):
    ep_nums, rewards, reasons, stages = [], [], [], []

    for i, e in enumerate(episodes):
        ep_nums.append(int(safe_float(e.get("episode", i + 1)) or (i + 1)))
        rewards.append(safe_float(e.get("reward")))
        reasons.append(str(e.get("reason", "unknown")).lower().strip())
        try:
            stages.append(int(float(e.get("curriculum_stage", 0))))
        except Exception:
            stages.append(0)

    roll = rolling_mean(rewards, rolling_w)

    # Infer stage transitions from episode data if not provided externally
    if curriculum_transitions is not None:
        # Map from curriculum transition episode numbers to stage ids
        transitions = []
        for ep in curriculum_transitions:
            # Find the stage id at that episode
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

    # Stage zone centers — mid-zone labels between transition lines.
    # S1 covers ep_nums[0] to first transition (no vertical line, just a label).
    # S2, S3, ... are index-based.
    stage_zones = []
    if transitions:
        first_center = int((ep_nums[0] + transitions[0][0]) / 2)
        stage_zones.append({"ep": None, "center": first_center, "label": "S1"})
        for i, (ep, _sid) in enumerate(transitions):
            next_ep = transitions[i + 1][0] if i + 1 < len(transitions) else ep_nums[-1]
            center  = int((ep + next_ep) / 2)
            stage_zones.append({"ep": ep, "center": center, "label": f"S{i + 2}"})

    # Global max (valid rewards only)
    valid = [(i, r) for i, r in enumerate(rewards) if r is not None]
    max_idx, max_val = max(valid, key=lambda t: t[1]) if valid else (0, 0.0)
    max_ep = ep_nums[max_idx]
    max_val = round(max_val, 1)

    # Fixed y-axis range so plot doesn't jump during animation
    valid_r = [r for r in rewards if r is not None]
    y_min = min(valid_r) if valid_r else -100.0
    y_max = max(valid_r) if valid_r else 100.0
    pad_lo = abs(y_min) * 0.04 if y_min != 0 else 10
    pad_hi = abs(y_max) * 0.12 if y_max != 0 else 10  # extra top space for stage labels
    y_range = [round(y_min - pad_lo, 1), round(y_max + pad_hi, 1)]

    return {
        "episodes":      ep_nums,
        "rewards":       rewards,
        "reasons":       reasons,
        "rolling":       roll,
        "transitions":   transitions,
        "stageZones":    stage_zones,
        "maxEp":         max_ep,
        "maxVal":        max_val,
        "rollingWindow": rolling_w,
        "yRange":        y_range,
        "xRange":        [ep_nums[0], ep_nums[-1]],
        "total":         len(ep_nums),
    }


# ── HTML / JS generation ──────────────────────────────────────────────────────

def generate_html(data):
    data_json = json.dumps(data, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reward Progress — Interactive</title>
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
  --rolling:   #7D9D9C;
  --maxline:   #8B7355;
  --grid:      #E8DDD2;
  --font:      'DM Sans', -apple-system, sans-serif;
  --mono:      'DM Mono', monospace;
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{
  height: 100%; overflow: hidden;
  background: var(--bg);
  font-family: var(--font);
  color: var(--text);
  font-size: 13px;
}}

/* ── layout ──────────────────────────────────────────── */
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
.tile-lbl {{
  font-size: 10px; font-weight: 500;
  color: var(--faint);
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-bottom: 5px;
}}
.tile-val {{
  font-size: 27px; font-weight: 500;
  line-height: 1.1;
  color: var(--text);
  font-family: var(--mono);
  transition: color .18s ease;
}}
#tile-pie {{
  flex: 1; min-height: 0;
  display: flex; flex-direction: column;
}}
#pie-wrap {{
  flex: 1; min-height: 120px;
  position: relative;
}}

/* ── controls bar ────────────────────────────────────── */
#controls {{
  display: flex; align-items: center;
  gap: 8px; flex-shrink: 0;
  padding: 2px 0;
}}
.ctrl-btn {{
  background: transparent;
  border: 0.5px solid var(--border);
  border-radius: 6px;
  padding: 4px 11px;
  font-size: 12px; font-family: var(--font);
  color: var(--text); cursor: pointer;
  white-space: nowrap;
  transition: background .12s;
}}
.ctrl-btn:hover {{ background: var(--bg2); }}
.ctrl-btn:active {{ background: var(--bg3); }}

#pb-wrap {{
  flex: 1; height: 5px;
  background: var(--border);
  border-radius: 3px;
  cursor: pointer; position: relative;
}}
#pb {{
  height: 100%; background: var(--rolling);
  border-radius: 3px; width: 0%;
  pointer-events: none;
  transition: width .04s linear;
}}
#pb-thumb {{
  position: absolute; top: 50%; right: calc(100% - 100%);
  width: 10px; height: 10px;
  background: var(--rolling);
  border-radius: 50%; transform: translateY(-50%);
  transition: opacity .2s; opacity: 0;
  pointer-events: none;
}}
#pb-wrap:hover #pb-thumb {{ opacity: 1; }}
#spd-lbl {{
  font-size: 11px; color: var(--muted);
  min-width: 30px; text-align: center;
  font-family: var(--mono);
}}

/* ── hover badge ─────────────────────────────────────── */
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

/* ── episode counter overlay ─────────────────────────── */
#ep-counter {{
  position: absolute; top: 8px; right: 8px;
  font-size: 10px; font-family: var(--mono);
  color: var(--faint); pointer-events: none;
}}
</style>
</head>
<body>
<div id="app">
  <div id="main-row">

    <!-- LEFT: main plot -->
    <div id="left">
      <div id="plot-title">Reward Progress Over Training</div>
      <div id="main-plot"></div>
      <div id="hover-badge">hover over any point to inspect</div>
      <div id="ep-counter"></div>
    </div>

    <!-- RIGHT: 4 tiles -->
    <div id="right">
      <div class="tile">
        <div class="tile-lbl">Episode</div>
        <div class="tile-val" id="v-ep">—</div>
      </div>
      <div class="tile">
        <div class="tile-lbl">Reward</div>
        <div class="tile-val" id="v-rw">—</div>
      </div>
      <div class="tile">
        <div class="tile-lbl">Rolling mean <span style="font-weight:400;color:var(--faint)">w={data["rollingWindow"]}</span></div>
        <div class="tile-val" id="v-rm">—</div>
      </div>
      <div class="tile" id="tile-pie">
        <div class="tile-lbl">Outcome distribution</div>
        <div id="pie-wrap"></div>
      </div>
    </div>

  </div>

  <!-- CONTROLS -->
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
/* ── data ─────────────────────────────────────────────────────────────────── */
const D = {data_json};

/* ── constants ───────────────────────────────────────────────────────────── */
const COLORS = {{
  goal:      '#87986A',
  collision: '#A44A3F',
  timeout:   '#C98245',
  unknown:   '#9B8E82',
}};
const REASON_TRACE = {{ collision: 0, timeout: 1, goal: 2, unknown: 2 }};
const SPEEDS   = [0.25, 0.5, 1, 2, 4, 8];
const ANIM_MS  = 18000;   // base total duration at 1× speed
const PIE_FREQ = 25;      // update pie every N animation frames

/* ── state ───────────────────────────────────────────────────────────────── */
let speedIdx    = 2;          // default 1×
let playing     = true;
let frame       = 0;          // last rendered episode index
let startTime   = null;       // performance.now() anchor
let animDone    = false;
let shownTrans  = new Set();
let maxShown    = false;
let pieCounts   = {{ goal: 0, collision: 0, timeout: 0 }};
let shapes      = [];
let annots      = [];
let pieCounter  = 0;
const TOTAL     = D.total;

/* ── Plotly layout ───────────────────────────────────────────────────────── */
const BASE_LAYOUT = {{
  paper_bgcolor: '#FAF6F0',
  plot_bgcolor:  '#FAF6F0',
  font: {{ family: "'DM Sans', sans-serif", color: '#463F3A', size: 12 }},
  margin: {{ t: 38, r: 22, b: 48, l: 66 }},
  xaxis: {{
    title: {{ text: 'Episode', font: {{ size: 12 }} }},
    gridcolor: '#E8DDD2', linecolor: '#D8CABD',
    tickcolor: '#6B625B', tickfont: {{ size: 11 }},
    range: D.xRange, zeroline: false, fixedrange: false,
  }},
  yaxis: {{
    title: {{ text: 'Reward', font: {{ size: 12 }} }},
    gridcolor: '#E8DDD2', linecolor: '#D8CABD',
    tickcolor: '#6B625B', tickfont: {{ size: 11 }},
    range: D.yRange, zeroline: true,
    zerolinecolor: '#D0C4B8', zerolinewidth: 1,
    fixedrange: false,
  }},
  legend: {{
    x: 0.99, y: 0.01, xanchor: 'right', yanchor: 'bottom',
    bgcolor: 'rgba(250,246,240,0.92)',
    bordercolor: '#D8CABD', borderwidth: 1,
    font: {{ size: 11 }},
  }},
  shapes: [], annotations: [],
  hovermode: 'closest',
  hoverlabel: {{
    bgcolor: '#FAF6F0', bordercolor: '#D8CABD',
    font: {{ family: "'DM Mono', monospace", size: 11, color: '#463F3A' }},
  }},
}};

/* ── trace templates ─────────────────────────────────────────────────────── */
const BASE_TRACES = [
  {{  /* 0: collision */
    name: 'Collision', x: [], y: [],
    mode: 'markers', type: 'scattergl',
    marker: {{ color: '#A44A3F', size: 4, opacity: 0.52 }},
    hovertemplate: 'Ep <b>%{{x}}</b><br>Reward: %{{y:.1f}}<br>Collision<extra></extra>',
  }},
  {{  /* 1: timeout */
    name: 'Timeout', x: [], y: [],
    mode: 'markers', type: 'scattergl',
    marker: {{ color: '#C98245', size: 4, opacity: 0.52 }},
    hovertemplate: 'Ep <b>%{{x}}</b><br>Reward: %{{y:.1f}}<br>Timeout<extra></extra>',
  }},
  {{  /* 2: goal */
    name: 'Goal', x: [], y: [],
    mode: 'markers', type: 'scattergl',
    marker: {{ color: '#87986A', size: 4, opacity: 0.52 }},
    hovertemplate: 'Ep <b>%{{x}}</b><br>Reward: %{{y:.1f}}<br>Goal<extra></extra>',
  }},
  {{  /* 3: rolling mean */
    name: `Rolling mean (w=${{D.rollingWindow}})`,
    x: [], y: [],
    mode: 'lines', type: 'scatter',
    line: {{ color: '#7D9D9C', width: 2.6 }},
    hovertemplate: 'Ep <b>%{{x}}</b><br>Mean: %{{y:.2f}}<extra></extra>',
  }},
];

/* ── init main plot ──────────────────────────────────────────────────────── */
Plotly.newPlot('main-plot', BASE_TRACES, BASE_LAYOUT, {{
  responsive: true, displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
}});

/* ── init pie chart ──────────────────────────────────────────────────────── */
const PIE_LAYOUT = {{
  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  margin: {{ t: 4, r: 4, b: 4, l: 4 }},
  showlegend: false,
}};

Plotly.newPlot('pie-wrap', [{{
  type: 'pie',
  values: [1, 1, 1],
  labels: ['Goal', 'Collision', 'Timeout'],
  marker: {{ colors: ['#87986A', '#A44A3F', '#C98245'] }},
  hole: 0.44,
  textinfo: 'percent',
  textfont: {{ size: 10, color: '#463F3A', family: "'DM Sans', sans-serif" }},
  sort: false, direction: 'clockwise',
  hovertemplate: '%{{label}}: %{{value}} eps (%{{percent}})<extra></extra>',
}}], PIE_LAYOUT, {{ displayModeBar: false, responsive: true }});

/* ── DOM helpers ─────────────────────────────────────────────────────────── */
function fmt(v, dec) {{
  if (v === null || v === undefined) return '—';
  const n = Number(v);
  if (!isFinite(n)) return '—';
  return n.toFixed(dec !== undefined ? dec : 1);
}}

function updateTiles(i) {{
  const r      = D.rewards[i];
  const reason = D.reasons[i] || 'unknown';
  const rm     = D.rolling[i];
  const ep     = D.episodes[i];

  document.getElementById('v-ep').textContent  = ep.toLocaleString();
  document.getElementById('v-rm').textContent  = fmt(rm, 2);

  const rwEl = document.getElementById('v-rw');
  rwEl.textContent  = fmt(r, 1);
  rwEl.style.color  = COLORS[reason] || COLORS.unknown;

  document.getElementById('ep-counter').textContent =
    `${{i + 1}} / ${{TOTAL}}`;
}}

function updatePie() {{
  const g = pieCounts.goal      || 0;
  const c = pieCounts.collision || 0;
  const t = pieCounts.timeout   || 0;
  if (g + c + t === 0) return;
  Plotly.react('pie-wrap', [{{
    type: 'pie',
    values: [g, c, t],
    labels: ['Goal', 'Collision', 'Timeout'],
    marker: {{ colors: ['#87986A', '#A44A3F', '#C98245'] }},
    hole: 0.44, textinfo: 'percent',
    textfont: {{ size: 10, color: '#463F3A', family: "'DM Sans', sans-serif" }},
    sort: false, direction: 'clockwise',
    hovertemplate: '%{{label}}: %{{value}} eps (%{{percent}})<extra></extra>',
  }}], PIE_LAYOUT, {{ displayModeBar: false, responsive: true }});
}}

/* ── stage transition helpers ────────────────────────────────────────────── */
function applyTransition(zone, immediate) {{
  // Vertical dashed line — skipped for S1 (ep is null, it has no preceding line)
  if (zone.ep !== null) {{
    shapes.push({{
      type: 'line', x0: zone.ep, x1: zone.ep, y0: 0, y1: 1, yref: 'paper',
      line: {{ color: '#9B8E82', width: 1.1, dash: 'dash' }},
    }});
  }}
  // Stage label centered in zone
  annots.push({{
    x: zone.center, y: 1.025, xref: 'x', yref: 'paper',
    text: zone.label, showarrow: false,
    font: {{ size: 11, color: '#6B625B', family: "'DM Sans', sans-serif" }},
    bgcolor: '#FAF6F0', borderpad: 2,
  }});
  if (immediate) {{
    Plotly.relayout('main-plot', {{ shapes: [...shapes], annotations: [...annots] }});
  }}
}}

function checkTransitions(currentEp, immediate) {{
  for (const zone of D.stageZones) {{
    if (!shownTrans.has(zone.ep) && currentEp >= zone.ep) {{
      shownTrans.add(zone.ep);
      applyTransition(zone, immediate !== false);
    }}
  }}
}}

function showMaxLine(immediate) {{
  shapes.push({{
    type: 'line',
    x0: D.xRange[0], x1: D.xRange[1],
    y0: D.maxVal, y1: D.maxVal,
    xref: 'x', yref: 'y',
    line: {{ color: '#8B7355', width: 1.1, dash: 'dot' }},
  }});
  annots.push({{
    x: D.xRange[1], y: D.maxVal,
    xref: 'x', yref: 'y',
    text: `max = ${{D.maxVal}}`,
    showarrow: false, xanchor: 'right',
    font: {{ size: 11, color: '#8B7355', family: "'DM Sans', sans-serif" }},
    bgcolor: '#FAF6F0', borderpad: 2,
  }});
  if (immediate !== false) {{
    Plotly.relayout('main-plot', {{ shapes: [...shapes], annotations: [...annots] }});
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
  if (target <= frame) {{
    requestAnimationFrame(animStep);
    return;
  }}

  /* batch new points per reason */
  const ext = {{ 0: {{x:[],y:[]}}, 1: {{x:[],y:[]}}, 2: {{x:[],y:[]}}, 3: {{x:[],y:[]}} }};

  for (let i = frame + 1; i <= target; i++) {{
    const reason = D.reasons[i] || 'unknown';
    const ti     = REASON_TRACE[reason] ?? 2;
    const rw     = D.rewards[i];
    const rm     = D.rolling[i];

    if (rw !== null) {{ ext[ti].x.push(D.episodes[i]); ext[ti].y.push(rw); }}
    if (rm !== null) {{ ext[3].x.push(D.episodes[i]);  ext[3].y.push(rm); }}

    pieCounts[reason] = (pieCounts[reason] || 0) + 1;
  }}

  /* extend scatter traces */
  for (let ti = 0; ti < 4; ti++) {{
    if (ext[ti].x.length > 0) {{
      Plotly.extendTraces('main-plot', {{ x: [ext[ti].x], y: [ext[ti].y] }}, [ti]);
    }}
  }}

  /* stage transitions */
  checkTransitions(D.episodes[target]);

  /* max reward line */
  if (!maxShown && D.episodes[target] >= D.maxEp) {{
    showMaxLine();
    maxShown = true;
  }}

  /* tiles */
  updateTiles(target);

  /* pie (throttled) */
  pieCounter++;
  if (pieCounter % PIE_FREQ === 0) updatePie();

  /* progress bar */
  const pct = (target / (TOTAL - 1)) * 100;
  document.getElementById('pb').style.width = pct + '%';
  document.getElementById('pb-thumb').style.left = `calc(${{pct}}% - 5px)`;

  frame = target;

  if (frame >= TOTAL - 1) {{
    animDone = true;
    playing  = false;
    updatePie();
    document.getElementById('btn-play').textContent = '▶ Play';
    document.getElementById('hover-badge').classList.add('show');
    setupHover();
    return;
  }}

  requestAnimationFrame(animStep);
}}

/* add S1 annotation immediately (it has no transition line, so it never fires
   through checkTransitions during the animation loop) */
(function addS1() {{
  const s1 = D.stageZones.find(z => z.label === 'S1');
  if (!s1) return;
  shownTrans.add(null);          // mark as shown so the loop skips it
  annots.push({{
    x: s1.center, y: 1.025, xref: 'x', yref: 'paper',
    text: 'S1', showarrow: false,
    font: {{ size: 11, color: '#6B625B', family: "'DM Sans', sans-serif" }},
    bgcolor: '#FAF6F0', borderpad: 2,
  }});
  Plotly.relayout('main-plot', {{ annotations: [...annots] }});
}})();

/* kick off */
requestAnimationFrame(animStep);

/* ── hover (post-animation) ──────────────────────────────────────────────── */
function setupHover() {{
  document.getElementById('main-plot').on('plotly_hover', function(evt) {{
    const pt  = evt.points[0];
    const ep  = pt.x;
    const idx = D.episodes.findIndex(e => e === ep);
    if (idx < 0) return;

    /* recount pie up to hovered episode */
    const tmp = {{ goal: 0, collision: 0, timeout: 0 }};
    for (let i = 0; i <= idx; i++) {{
      const r = D.reasons[i] || 'unknown';
      tmp[r] = (tmp[r] || 0) + 1;
    }}
    pieCounts = tmp;

    updateTiles(idx);
    updatePie();
  }});
}}

/* ── rebuild to specific frame (for seek) ────────────────────────────────── */
function rebuildToFrame(targetIdx) {{
  if (targetIdx < 0) targetIdx = 0;
  if (targetIdx >= TOTAL) targetIdx = TOTAL - 1;

  /* reset accumulators */
  shapes = []; annots = [];
  shownTrans.clear();
  maxShown = false;
  pieCounts = {{ goal: 0, collision: 0, timeout: 0 }};

  /* always restore S1 label (no transition line, not caught by loop) */
  const s1z = D.stageZones.find(z => z.label === 'S1');
  if (s1z) {{
    shownTrans.add(null);
    annots.push({{
      x: s1z.center, y: 1.025, xref: 'x', yref: 'paper',
      text: 'S1', showarrow: false,
      font: {{ size: 11, color: '#6B625B', family: "'DM Sans', sans-serif" }},
      bgcolor: '#FAF6F0', borderpad: 2,
    }});
  }}

  const ext = [
    {{ x: [], y: [] }}, {{ x: [], y: [] }},
    {{ x: [], y: [] }}, {{ x: [], y: [] }},
  ];

  for (let i = 0; i <= targetIdx; i++) {{
    const reason = D.reasons[i] || 'unknown';
    const ti     = REASON_TRACE[reason] ?? 2;
    const rw     = D.rewards[i];
    const rm     = D.rolling[i];

    if (rw !== null) {{ ext[ti].x.push(D.episodes[i]); ext[ti].y.push(rw); }}
    if (rm !== null) {{ ext[3].x.push(D.episodes[i]);  ext[3].y.push(rm); }}

    pieCounts[reason] = (pieCounts[reason] || 0) + 1;

    /* batch transitions (no immediate relayout yet) */
    for (const zone of D.stageZones) {{
      if (!shownTrans.has(zone.ep) && D.episodes[i] >= zone.ep) {{
        shownTrans.add(zone.ep);
        applyTransition(zone, false);  /* collect, don't relayout yet */
      }}
    }}
    /* max line */
    if (!maxShown && D.episodes[i] >= D.maxEp) {{
      maxShown = true;
      showMaxLine(false);
    }}
  }}

  /* single relayout for all shapes + annotations */
  const newTraces = BASE_TRACES.map((t, ti) => ({{ ...t, x: [...ext[ti].x], y: [...ext[ti].y] }}));
  Plotly.react('main-plot', newTraces,
    {{ ...BASE_LAYOUT, shapes: [...shapes], annotations: [...annots] }});

  updateTiles(targetIdx);
  updatePie();

  const pct = (targetIdx / (TOTAL - 1)) * 100;
  document.getElementById('pb').style.width = pct + '%';
  document.getElementById('pb-thumb').style.left = `calc(${{pct}}% - 5px)`;

  frame = targetIdx;
}}

/* ── controls ────────────────────────────────────────────────────────────── */

/* play / pause */
document.getElementById('btn-play').addEventListener('click', () => {{
  if (animDone) return;
  playing = !playing;
  document.getElementById('btn-play').textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) {{
    /* re-anchor start time so animation continues from current frame */
    const elapsed = (frame / (TOTAL - 1)) * (ANIM_MS / SPEEDS[speedIdx]);
    startTime = performance.now() - elapsed;
    requestAnimationFrame(animStep);
  }}
}});

/* speed − */
document.getElementById('btn-slower').addEventListener('click', () => {{
  if (speedIdx === 0) return;
  const elapsed = (frame / (TOTAL - 1)) * (ANIM_MS / SPEEDS[speedIdx]);
  speedIdx--;
  document.getElementById('spd-lbl').textContent = SPEEDS[speedIdx] + '×';
  if (playing) startTime = performance.now() - (frame / (TOTAL - 1)) * (ANIM_MS / SPEEDS[speedIdx]);
}});

/* speed + */
document.getElementById('btn-faster').addEventListener('click', () => {{
  if (speedIdx === SPEEDS.length - 1) return;
  speedIdx++;
  document.getElementById('spd-lbl').textContent = SPEEDS[speedIdx] + '×';
  if (playing) startTime = performance.now() - (frame / (TOTAL - 1)) * (ANIM_MS / SPEEDS[speedIdx]);
}});

/* restart */
document.getElementById('btn-restart').addEventListener('click', () => {{
  playing = false;
  animDone = false;
  frame = 0; startTime = null;
  shapes = []; annots = [];
  shownTrans.clear(); maxShown = false;
  pieCounts = {{ goal: 0, collision: 0, timeout: 0 }};
  pieCounter = 0;

  /* restore S1 label immediately on restart */
  const s1r = D.stageZones.find(z => z.label === 'S1');
  if (s1r) {{
    shownTrans.add(null);
    annots.push({{
      x: s1r.center, y: 1.025, xref: 'x', yref: 'paper',
      text: 'S1', showarrow: false,
      font: {{ size: 11, color: '#6B625B', family: "'DM Sans', sans-serif" }},
      bgcolor: '#FAF6F0', borderpad: 2,
    }});
  }}
  document.getElementById('hover-badge').classList.remove('show');
  document.getElementById('btn-play').textContent = '⏸ Pause';
  document.getElementById('pb').style.width = '0%';

  /* clear traces with react */
  Plotly.react('main-plot',
    BASE_TRACES.map(t => ({{ ...t, x: [], y: [] }})),
    {{ ...BASE_LAYOUT, shapes: [], annotations: [] }});

  /* reset pie */
  Plotly.react('pie-wrap', [{{
    type: 'pie', values: [1, 1, 1], labels: ['Goal', 'Collision', 'Timeout'],
    marker: {{ colors: ['#87986A', '#A44A3F', '#C98245'] }},
    hole: 0.44, textinfo: 'percent',
    textfont: {{ size: 10, color: '#463F3A' }},
    sort: false, direction: 'clockwise',
  }}], PIE_LAYOUT, {{ displayModeBar: false, responsive: true }});

  ['v-ep', 'v-rw', 'v-rm'].forEach(id => {{
    document.getElementById(id).textContent = '—';
    document.getElementById(id).style.color = '';
  }});
  document.getElementById('ep-counter').textContent = '';

  /* small delay then play */
  setTimeout(() => {{
    playing = true;
    requestAnimationFrame(animStep);
  }}, 80);
}});

/* progress bar seek (always available) */
document.getElementById('pb-wrap').addEventListener('click', function(e) {{
  const pct      = Math.max(0, Math.min(1, e.offsetX / this.offsetWidth));
  const targetIdx = Math.round(pct * (TOTAL - 1));

  /* pause animation */
  playing = false;
  document.getElementById('btn-play').textContent = '▶ Play';

  rebuildToFrame(targetIdx);

  /* if we seeked to the end, mark done */
  if (targetIdx >= TOTAL - 1) {{
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


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate a self-contained interactive animated reward plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--json",    required=True,   help="Path to episode_stats.json")
    ap.add_argument("--curr",    default=None,    help="Path to curriculum_log.json (optional)")
    ap.add_argument("--out",     default="./output", help="Output directory")
    ap.add_argument("--rolling", type=int, default=20, help="Rolling mean window size")
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
    print(f"  Stage zones: {len(data['stageZones'])}")
    print(f"  Max reward:  {data['maxVal']} at episode {data['maxEp']}")
    print(f"  y-axis:      {data['yRange']}")

    html    = generate_html(data)
    outpath = os.path.join(args.out, "01_reward_interactive.html")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(outpath) / 1024
    print(f"\nSaved: {outpath}  ({size_kb:.1f} KB)")
    print("Open in any browser — fully self-contained, no server needed.")


if __name__ == "__main__":
    main()