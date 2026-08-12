#!/usr/bin/env python3
"""
Interactive Depth Accuracy Plot — standalone HTML generator.
Companion to plot_depth_accuracy_thesis_style_fixed.py (same THEME dict,
same metrics: MAE/RMSE/MAPE/Bias/Coverage, same ideal line + ±0.10 m band +
±5% band + quality-colored points + smoothed error line) styled after
interactive_01_reward.py / interactive_02_yaw_comparison.py (animated
play/pause/speed/seek, thesis_nude_contrast look).

Points are revealed in ascending order of true_m (same sort the static
script uses), sweeping into both the "measured vs true" panel and the
"signed relative error" panel together, with the summary metrics (n, MAE,
RMSE, MAPE, Bias, Coverage) updating live in the side panel as they go.

Usage:
    python interactive_03_depth_accuracy.py --csv depth_experiment_results.csv --out plots_depth
"""

import argparse
import csv
import json
import math
import os


THEME = {
    "BG": "#FAF6F0",
    "GRID_CLR": "#E8DDD2",
    "TEXT_CLR": "#463F3A",
    "TICK_CLR": "#6B625B",
    "EDGE_CLR": "#D8CABD",
    "PAPER_WHITE": "#FAF6F0",
    "PRIMARY": "#7D9D9C",
    "SECONDARY": "#C9A27E",
    "ACCENT": "#B7A99A",
    "GOOD": "#87986A",
    "MARGINAL": "#C98245",
    "BAD": "#A44A3F",
}

REQUIRED_COLS = {"true_m", "measured_m", "error_m", "error_pct", "depth_std_m", "coverage_pct"}


# ── color helpers ───────────────────────────────────────────────────────────

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#" + "".join(f"{max(0, min(255, int(round(c)))):02X}" for c in rgb)


def mix_hex(c1, c2, t=0.5):
    a, b = _hex_to_rgb(c1), _hex_to_rgb(c2)
    return _rgb_to_hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))


def hex_to_rgba(h, alpha):
    r, g, b = _hex_to_rgb(h)
    return f"rgba({r},{g},{b},{alpha})"


# ── basic helpers ───────────────────────────────────────────────────────────

def safe_float(x):
    if x is None:
        return None
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


def mean_of(values):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def mae_of(values):
    vals = [abs(v) for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def rmse_of(values):
    vals = [v * v for v in values if v is not None]
    return math.sqrt(sum(vals) / len(vals)) if vals else None


def rolling_mean_centered(values, window=3, min_periods=1):
    """Mirror of pandas .rolling(window, center=True, min_periods=1).mean()."""
    n = len(values)
    half = (window - 1) // 2
    after = window - 1 - half
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n - 1, i + after)
        win = [v for v in values[lo:hi + 1] if v is not None]
        out.append((sum(win) / len(win)) if len(win) >= min_periods else None)
    return out


# ── CSV loading ─────────────────────────────────────────────────────────────

def load_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_points(rows):
    if not rows:
        raise ValueError("CSV appears to be empty.")
    header = set(rows[0].keys())
    missing = REQUIRED_COLS - header
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    color_map = {"GOOD": THEME["GOOD"], "MARGINAL": THEME["MARGINAL"], "BAD": THEME["BAD"], "UNKNOWN": THEME["ACCENT"]}

    points = []
    for r in rows:
        t = safe_float(r.get("true_m"))
        if t is None:
            continue
        raw_q = (r.get("quality") or "").strip()
        key = raw_q.upper() if raw_q else "UNKNOWN"
        points.append({
            "t": t,
            "m": safe_float(r.get("measured_m")),
            "em": safe_float(r.get("error_m")),
            "ep": safe_float(r.get("error_pct")),
            "std": safe_float(r.get("depth_std_m")) or 0.0,
            "cov": safe_float(r.get("coverage_pct")),
            "qkey": key,
            "qlabel": key.capitalize(),
            "color": color_map.get(key, THEME["ACCENT"]),
        })

    points.sort(key=lambda p: p["t"])
    return points


def build_groups(points):
    keys = sorted(set(p["qkey"] for p in points))
    groups = []
    for k in keys:
        p0 = next(p for p in points if p["qkey"] == k)
        groups.append({"key": k, "label": p0["qlabel"], "color": p0["color"]})
    return groups


# ── data assembly ────────────────────────────────────────────────────────────

def build_data(points):
    true_vals = [p["t"] for p in points]
    meas_vals = [p["m"] for p in points if p["m"] is not None]

    lim_min = max(0.0, min(true_vals + meas_vals) - 0.20)
    lim_max = max(true_vals + meas_vals) + 0.30

    ep_vals = [p["ep"] for p in points if p["ep"] is not None]
    ep_min = min(ep_vals + [-5.0]) if ep_vals else -5.0
    ep_max = max(ep_vals + [5.0]) if ep_vals else 5.0
    pad = max(1.0, (ep_max - ep_min) * 0.12)
    err_y_range = [round(ep_min - pad, 2), round(ep_max + pad, 2)]

    smoothed = rolling_mean_centered([p["ep"] for p in points], window=3, min_periods=1)

    em_vals = [p["em"] for p in points]
    cov_vals = [p["cov"] for p in points]
    final = {
        "n": len(points),
        "mae": mae_of(em_vals),
        "rmse": rmse_of(em_vals),
        "mape": mae_of([p["ep"] for p in points]),
        "bias": mean_of(em_vals),
        "coverage": mean_of(cov_vals),
    }

    groups = build_groups(points)

    return {
        "points": [
            {
                "t": round(p["t"], 4),
                "m": (round(p["m"], 4) if p["m"] is not None else None),
                "em": (round(p["em"], 4) if p["em"] is not None else None),
                "ep": (round(p["ep"], 4) if p["ep"] is not None else None),
                "std": round(p["std"], 4),
                "cov": (round(p["cov"], 2) if p["cov"] is not None else None),
                "q": p["qkey"],
            }
            for p in points
        ],
        "smoothed": [(round(v, 4) if v is not None else None) for v in smoothed],
        "groups": groups,
        "limMin": round(lim_min, 3),
        "limMax": round(lim_max, 3),
        "errYRange": err_y_range,
        "final": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in final.items()},
        "total": len(points),
    }


# ── HTML / JS generation ──────────────────────────────────────────────────────

def generate_html(data, theme):
    data_json = json.dumps(data, separators=(",", ":"))

    bg, grid, text = theme["BG"], theme["GRID_CLR"], theme["TEXT_CLR"]
    tick, edge, paper = theme["TICK_CLR"], theme["EDGE_CLR"], theme["PAPER_WHITE"]
    primary, secondary, accent = theme["PRIMARY"], theme["SECONDARY"], theme["ACCENT"]
    good, marginal, bad = theme["GOOD"], theme["MARGINAL"], theme["BAD"]
    bg2 = mix_hex(bg, edge, 0.35)
    bg3 = mix_hex(bg, edge, 0.55)
    band10_fill = hex_to_rgba(primary, 0.10)
    band5_fill = hex_to_rgba(good, 0.10)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Depth Accuracy — Interactive</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:        {bg};
  --bg2:       {bg2};
  --bg3:       {bg3};
  --border:    {edge};
  --text:      {text};
  --muted:     {tick};
  --faint:     {mix_hex(text, bg, 0.45)};
  --primary:   {primary};
  --secondary: {secondary};
  --accent:    {accent};
  --good:      {good};
  --marginal:  {marginal};
  --bad:       {bad};
  --grid:      {grid};
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

#app {{ display: flex; flex-direction: column; height: 100vh; padding: 12px; gap: 10px; }}
#main-row {{ display: flex; flex: 1; gap: 10px; min-height: 0; }}
#left {{ flex: 1; min-width: 0; position: relative; display: flex; flex-direction: column; }}
#plot-title {{ font-size: 13px; font-weight: 500; color: var(--muted); letter-spacing: .03em; padding: 0 4px 6px; }}
#main-plot {{ flex: 1; min-height: 0; }}

#right {{ width: 236px; display: flex; flex-direction: column; gap: 8px; }}
.tile {{ background: var(--bg2); border: 0.5px solid var(--border); border-radius: 10px; padding: 10px 14px; }}
.tile-lbl {{ font-size: 10px; font-weight: 500; color: var(--faint); letter-spacing: .08em; text-transform: uppercase; margin-bottom: 6px; }}
.tile-val {{ font-size: 24px; font-weight: 500; line-height: 1.1; font-family: var(--mono); }}

#stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px 12px; }}
.stat-cell .stat-lbl {{ font-size: 9.5px; font-weight: 500; color: var(--faint); letter-spacing: .06em; text-transform: uppercase; margin-bottom: 2px; }}
.stat-cell .stat-val {{ font-size: 17px; font-weight: 500; font-family: var(--mono); color: var(--text); }}

#tile-pie {{ flex: 1; min-height: 0; display: flex; flex-direction: column; }}
#pie-wrap {{ flex: 1; min-height: 100px; position: relative; }}

#controls {{ display: flex; align-items: center; gap: 8px; flex-shrink: 0; padding: 2px 0; }}
.ctrl-btn {{
  background: transparent; border: 0.5px solid var(--border); border-radius: 6px;
  padding: 4px 11px; font-size: 12px; font-family: var(--font);
  color: var(--text); cursor: pointer; white-space: nowrap; transition: background .12s;
}}
.ctrl-btn:hover {{ background: var(--bg2); }}
.ctrl-btn:active {{ background: var(--bg3); }}

#pb-wrap {{ flex: 1; height: 5px; background: var(--border); border-radius: 3px; cursor: pointer; position: relative; }}
#pb {{ height: 100%; background: var(--primary); border-radius: 3px; width: 0%; pointer-events: none; transition: width .04s linear; }}
#pb-thumb {{
  position: absolute; top: 50%; right: calc(100% - 100%); width: 10px; height: 10px;
  background: var(--primary); border-radius: 50%; transform: translateY(-50%);
  transition: opacity .2s; opacity: 0; pointer-events: none;
}}
#pb-wrap:hover #pb-thumb {{ opacity: 1; }}
#spd-lbl {{ font-size: 11px; color: var(--muted); min-width: 30px; text-align: center; font-family: var(--mono); }}

#hover-badge {{
  position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
  background: var(--bg2); border: 0.5px solid var(--border); border-radius: 20px;
  padding: 4px 14px; font-size: 11px; color: var(--muted);
  opacity: 0; transition: opacity .4s; pointer-events: none; white-space: nowrap;
}}
#hover-badge.show {{ opacity: 1; }}

#sample-counter {{ position: absolute; top: 8px; right: 8px; font-size: 10px; font-family: var(--mono); color: var(--faint); pointer-events: none; }}
</style>
</head>
<body>
<div id="app">
  <div id="main-row">

    <div id="left">
      <div id="plot-title">Stereo Depth Accuracy</div>
      <div id="main-plot"></div>
      <div id="hover-badge">hover over any point to inspect</div>
      <div id="sample-counter"></div>
    </div>

    <div id="right">
      <div class="tile">
        <div class="tile-lbl">Points</div>
        <div class="tile-val" id="v-n">—</div>
      </div>
      <div class="tile">
        <div class="tile-lbl">Live metrics</div>
        <div id="stat-grid">
          <div class="stat-cell"><div class="stat-lbl">MAE</div><div class="stat-val" id="v-mae">—</div></div>
          <div class="stat-cell"><div class="stat-lbl">RMSE</div><div class="stat-val" id="v-rmse">—</div></div>
          <div class="stat-cell"><div class="stat-lbl">MAPE</div><div class="stat-val" id="v-mape">—</div></div>
          <div class="stat-cell"><div class="stat-lbl">Bias</div><div class="stat-val" id="v-bias">—</div></div>
          <div class="stat-cell" style="grid-column: 1 / span 2;"><div class="stat-lbl">Coverage</div><div class="stat-val" id="v-cov">—</div></div>
        </div>
      </div>
      <div class="tile" id="tile-pie">
        <div class="tile-lbl">Quality distribution</div>
        <div id="pie-wrap"></div>
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
/* ── data ─────────────────────────────────────────────────────────────────── */
const D = {data_json};

/* ── constants ───────────────────────────────────────────────────────────── */
const COLORS = {{
  bg: '{bg}', grid: '{grid}', text: '{text}', tick: '{tick}', edge: '{edge}', paper: '{paper}',
  primary: '{primary}', secondary: '{secondary}', accent: '{accent}',
  good: '{good}', marginal: '{marginal}', bad: '{bad}',
  band10: '{band10_fill}', band5: '{band5_fill}',
}};
const SPEEDS  = [0.25, 0.5, 1, 2, 4, 8];
const ANIM_MS = 9000;
const TOTAL   = D.total;

/* trace layout: 0 ideal, 1 band10, 2 band5, 3 errorbars,
   4..4+G-1 top group markers, 4+G..4+2G-1 bottom group markers, last: smoothed line */
const G = D.groups.length;
const TOP_START    = 4;
const BOTTOM_START = 4 + G;
const SMOOTH_IDX   = 4 + 2 * G;

/* ── state ───────────────────────────────────────────────────────────────── */
let speedIdx  = 2;
let playing   = true;
let frame     = 0;
let startTime = null;
let animDone  = false;

/* ── layout ──────────────────────────────────────────────────────────────── */
const TOP_DOMAIN    = [0.46, 0.97];
const BOTTOM_DOMAIN  = [0.0, 0.28];

const BASE_LAYOUT = {{
  paper_bgcolor: COLORS.bg,
  plot_bgcolor:  COLORS.bg,
  font: {{ family: "'DM Sans', sans-serif", color: COLORS.text, size: 12 }},
  margin: {{ t: 34, r: 26, b: 42, l: 66 }},
  showlegend: true,
  legend: {{
    x: 0.985, y: TOP_DOMAIN[0] + 0.01, xanchor: 'right', yanchor: 'bottom',
    bgcolor: 'rgba(255,253,249,0.92)', bordercolor: COLORS.edge, borderwidth: 1,
    font: {{ size: 10 }},
  }},
  xaxis: {{
    title: {{ text: 'True distance (m)', font: {{ size: 11.5 }} }},
    domain: [0, 1], anchor: 'y',
    range: [D.limMin, D.limMax],
    gridcolor: COLORS.grid, linecolor: COLORS.edge, tickcolor: COLORS.text, tickfont: {{ size: 10.5 }},
  }},
  yaxis: {{
    title: {{ text: 'Measured distance (m)', font: {{ size: 11.5 }} }},
    domain: TOP_DOMAIN, anchor: 'x',
    range: [D.limMin, D.limMax], scaleanchor: 'x', scaleratio: 1,
    gridcolor: COLORS.grid, linecolor: COLORS.edge, tickcolor: COLORS.text, tickfont: {{ size: 10.5 }},
  }},
  xaxis2: {{
    title: {{ text: 'True distance (m)', font: {{ size: 11.5 }} }},
    domain: [0, 1], anchor: 'y2',
    range: [D.limMin, D.limMax],
    gridcolor: COLORS.grid, linecolor: COLORS.edge, tickcolor: COLORS.text, tickfont: {{ size: 10.5 }},
  }},
  yaxis2: {{
    title: {{ text: 'Error (%)', font: {{ size: 11.5 }} }},
    domain: BOTTOM_DOMAIN, anchor: 'x2',
    range: D.errYRange,
    gridcolor: COLORS.grid, linecolor: COLORS.edge, tickcolor: COLORS.text, tickfont: {{ size: 10.5 }},
    zeroline: false,
  }},
  shapes: [
    {{ type: 'line', xref: 'x2', yref: 'y2', x0: D.limMin, x1: D.limMax, y0: 0, y1: 0,
       line: {{ color: COLORS.tick, width: 1.0 }}, opacity: 0.65 }},
  ],
  annotations: [
    {{ x: 0.5, y: 1.045, xref: 'paper', yref: 'paper', xanchor: 'center', yanchor: 'bottom',
       text: 'Stereo Depth Accuracy — Measured vs True Distance', showarrow: false,
       font: {{ size: 13.5, color: COLORS.text, family: "'DM Sans', sans-serif" }} }},
    {{ x: 0.5, y: BOTTOM_DOMAIN[1] + 0.045, xref: 'paper', yref: 'paper', xanchor: 'center', yanchor: 'bottom',
       text: 'Signed Relative Error Across Test Distances', showarrow: false,
       font: {{ size: 12, color: COLORS.text, family: "'DM Sans', sans-serif" }} }},
  ],
  hovermode: 'closest',
  hoverlabel: {{ bgcolor: COLORS.paper, bordercolor: COLORS.edge, font: {{ family: "'DM Mono', monospace", size: 11, color: COLORS.text }} }},
}};

/* ── static reference traces (built once) ───────────────────────────────── */
function refTraces() {{
  return [
    {{ name: 'Ideal measurement', x: [D.limMin, D.limMax], y: [D.limMin, D.limMax],
       mode: 'lines', type: 'scatter', xaxis: 'x', yaxis: 'y',
       line: {{ color: COLORS.tick, width: 1.2, dash: 'dash' }}, opacity: 0.7, hoverinfo: 'skip' }},
    {{ name: '±0.10 m band', x: [D.limMin, D.limMax, D.limMax, D.limMin],
       y: [D.limMin - 0.10, D.limMax - 0.10, D.limMax + 0.10, D.limMin + 0.10],
       mode: 'lines', type: 'scatter', xaxis: 'x', yaxis: 'y',
       fill: 'toself', fillcolor: COLORS.band10, line: {{ width: 0 }}, hoverinfo: 'skip' }},
    {{ name: '±5% band', x: [D.limMin, D.limMax, D.limMax, D.limMin], y: [-5, -5, 5, 5],
       mode: 'lines', type: 'scatter', xaxis: 'x2', yaxis: 'y2',
       fill: 'toself', fillcolor: COLORS.band5, line: {{ width: 0 }}, hoverinfo: 'skip' }},
    {{ name: 'Depth std', x: [], y: [], xaxis: 'x', yaxis: 'y',
       mode: 'markers', type: 'scatter', showlegend: false, hoverinfo: 'skip',
       marker: {{ size: 0, color: 'rgba(0,0,0,0)' }},
       error_y: {{ type: 'data', array: [], visible: true, color: COLORS.accent, thickness: 0.9, width: 4, opacity: 0.55 }} }},
  ];
}}

/* ── DOM helpers ─────────────────────────────────────────────────────────── */
function fmt(v, dec) {{
  if (v === null || v === undefined) return '—';
  const n = Number(v);
  if (!isFinite(n)) return '—';
  return n.toFixed(dec !== undefined ? dec : 1);
}}
function fmtSigned(v, dec) {{
  if (v === null || v === undefined) return '—';
  const n = Number(v);
  if (!isFinite(n)) return '—';
  return (n >= 0 ? '+' : '') + n.toFixed(dec !== undefined ? dec : 1);
}}

function updateTiles(stats) {{
  document.getElementById('v-n').textContent    = `${{stats.n}} / ${{TOTAL}}`;
  document.getElementById('v-mae').textContent  = stats.n ? fmt(stats.mae, 3) + ' m' : '—';
  document.getElementById('v-rmse').textContent = stats.n ? fmt(stats.rmse, 3) + ' m' : '—';
  document.getElementById('v-mape').textContent = stats.n ? fmt(stats.mape, 2) + '%' : '—';
  document.getElementById('v-bias').textContent = stats.n ? fmtSigned(stats.bias, 3) + ' m' : '—';
  document.getElementById('v-cov').textContent  = (stats.covCount ? fmt(stats.sumCov / stats.covCount, 1) + '%' : '—');
  document.getElementById('sample-counter').textContent = `${{stats.n}} / ${{TOTAL}}`;
}}

const PIE_LAYOUT = {{ paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', margin: {{ t: 4, r: 4, b: 4, l: 4 }}, showlegend: false }};

function pieTrace(counts) {{
  const labels = D.groups.map(g => g.label);
  const colors = D.groups.map(g => g.color);
  const values = D.groups.map(g => counts[g.key] || 0);
  const total  = values.reduce((a, b) => a + b, 0);
  return [{{
    type: 'pie', values: total ? values : D.groups.map(() => 1), labels: labels,
    marker: {{ colors: colors }}, hole: 0.44, textinfo: total ? 'percent' : 'none',
    textfont: {{ size: 10, color: COLORS.text, family: "'DM Sans', sans-serif" }},
    sort: false, direction: 'clockwise',
    hovertemplate: '%{{label}}: %{{value}} pts (%{{percent}})<extra></extra>',
  }}];
}}

Plotly.newPlot('pie-wrap', pieTrace({{}}), PIE_LAYOUT, {{ displayModeBar: false, responsive: true }});
Plotly.newPlot('main-plot', refTraces(), BASE_LAYOUT, {{
  responsive: true, displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
}});

/* ── core render: rebuild everything up to (and including) index idx ────── */
function renderUpTo(idx) {{
  const eb = {{ x: [], y: [], err: [] }};
  const top = {{}}, bottom = {{}};
  for (const g of D.groups) {{ top[g.key] = {{ x: [], y: [] }}; bottom[g.key] = {{ x: [], y: [] }}; }}
  const sm = {{ x: [], y: [] }};

  let n = 0, sumAbsEm = 0, sumSqEm = 0, sumAbsEp = 0, sumEm = 0, sumCov = 0, covCount = 0;
  const counts = {{}};

  for (let i = 0; i <= idx; i++) {{
    const p = D.points[i];
    eb.x.push(p.t); eb.y.push(p.m); eb.err.push(p.std);
    top[p.q].x.push(p.t); top[p.q].y.push(p.m);
    bottom[p.q].x.push(p.t); bottom[p.q].y.push(p.ep);
    if (D.smoothed[i] !== null) {{ sm.x.push(p.t); sm.y.push(D.smoothed[i]); }}

    if (p.em !== null) {{ sumAbsEm += Math.abs(p.em); sumSqEm += p.em * p.em; sumEm += p.em; }}
    if (p.ep !== null) {{ sumAbsEp += Math.abs(p.ep); }}
    if (p.cov !== null) {{ sumCov += p.cov; covCount++; }}
    counts[p.q] = (counts[p.q] || 0) + 1;
    n++;
  }}

  const traces = refTraces();
  traces[3].x = eb.x; traces[3].y = eb.y; traces[3].error_y.array = eb.err;

  D.groups.forEach((g, gi) => {{
    traces.push({{
      name: g.label, x: top[g.key].x, y: top[g.key].y, xaxis: 'x', yaxis: 'y',
      mode: 'markers', type: 'scatter', showlegend: true,
      marker: {{ color: g.color, size: 9, opacity: 0.92, line: {{ color: COLORS.paper, width: 1.0 }} }},
      hovertemplate: `${{g.label}}<br>true=%{{x:.2f}} m<br>measured=%{{y:.2f}} m<extra></extra>`,
    }});
  }});
  D.groups.forEach((g, gi) => {{
    traces.push({{
      name: g.label, x: bottom[g.key].x, y: bottom[g.key].y, xaxis: 'x2', yaxis: 'y2',
      mode: 'markers', type: 'scatter', showlegend: false,
      marker: {{ color: g.color, size: 8, opacity: 0.92, line: {{ color: COLORS.paper, width: 0.9 }} }},
      hovertemplate: `${{g.label}}<br>true=%{{x:.2f}} m<br>error=%{{y:.2f}}%<extra></extra>`,
    }});
  }});
  traces.push({{
    name: 'Smoothed error', x: sm.x, y: sm.y, xaxis: 'x2', yaxis: 'y2',
    mode: 'lines', type: 'scatter',
    line: {{ color: COLORS.secondary, width: 2.3 }}, hoverinfo: 'skip',
  }});

  const layout = {{ ...BASE_LAYOUT, annotations: [...BASE_LAYOUT.annotations] }};
  if (idx >= TOTAL - 1) {{
    const f = D.final;
    layout.annotations = [...layout.annotations, {{
      x: 0.015, y: TOP_DOMAIN[1] - 0.01, xref: 'paper', yref: 'paper', xanchor: 'left', yanchor: 'top',
      align: 'left', showarrow: false,
      text: `n = ${{f.n}}<br>MAE = ${{fmt(f.mae, 3)}} m<br>RMSE = ${{fmt(f.rmse, 3)}} m<br>MAPE = ${{fmt(f.mape, 2)}}%<br>Bias = ${{fmtSigned(f.bias, 3)}} m<br>Coverage = ${{fmt(f.coverage, 1)}}%`,
      font: {{ size: 10.5, color: COLORS.text, family: "'DM Mono', monospace" }},
      bgcolor: COLORS.paper, bordercolor: COLORS.edge, borderwidth: 0.9, borderpad: 6,
    }}];
  }}

  Plotly.react('main-plot', traces, layout);
  Plotly.react('pie-wrap', pieTrace(counts), PIE_LAYOUT, {{ displayModeBar: false, responsive: true }});
  updateTiles({{ n, mae: sumAbsEm / n, rmse: Math.sqrt(sumSqEm / n), mape: sumAbsEp / n, bias: sumEm / n, sumCov, covCount }});

  const pct = (idx / (TOTAL - 1)) * 100;
  document.getElementById('pb').style.width = pct + '%';
  document.getElementById('pb-thumb').style.left = `calc(${{pct}}% - 5px)`;
}}

/* ── hover (post-animation) ──────────────────────────────────────────────── */
function closestIndex(t) {{
  let lo = 0, hi = D.points.length - 1;
  while (lo < hi) {{
    const mid = (lo + hi) >> 1;
    if (D.points[mid].t < t) lo = mid + 1; else hi = mid;
  }}
  return lo;
}}

function setupHover() {{
  document.getElementById('main-plot').on('plotly_hover', function(evt) {{
    const pt = evt.points[0];
    if (pt.curveNumber < TOP_START || pt.curveNumber === SMOOTH_IDX) return;
    const idx = closestIndex(pt.x);
    let n = 0, sumAbsEm = 0, sumSqEm = 0, sumAbsEp = 0, sumEm = 0, sumCov = 0, covCount = 0;
    for (let i = 0; i <= idx; i++) {{
      const p = D.points[i];
      if (p.em !== null) {{ sumAbsEm += Math.abs(p.em); sumSqEm += p.em * p.em; sumEm += p.em; }}
      if (p.ep !== null) {{ sumAbsEp += Math.abs(p.ep); }}
      if (p.cov !== null) {{ sumCov += p.cov; covCount++; }}
      n++;
    }}
    updateTiles({{ n, mae: sumAbsEm / n, rmse: Math.sqrt(sumSqEm / n), mape: sumAbsEp / n, bias: sumEm / n, sumCov, covCount }});
  }});
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
  if (target === frame) {{ requestAnimationFrame(animStep); return; }}
  frame = target;
  renderUpTo(frame);

  if (frame >= TOTAL - 1) {{
    animDone = true; playing = false;
    document.getElementById('btn-play').textContent = '▶ Play';
    document.getElementById('hover-badge').classList.add('show');
    setupHover();
    return;
  }}
  requestAnimationFrame(animStep);
}}

/* ── controls ────────────────────────────────────────────────────────────── */
document.getElementById('btn-play').addEventListener('click', () => {{
  if (animDone) return;
  playing = !playing;
  document.getElementById('btn-play').textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) {{
    const elapsed = (frame / (TOTAL - 1)) * (ANIM_MS / SPEEDS[speedIdx]);
    startTime = performance.now() - elapsed;
    requestAnimationFrame(animStep);
  }}
}});

document.getElementById('btn-slower').addEventListener('click', () => {{
  if (speedIdx === 0) return;
  const elapsed = (frame / (TOTAL - 1)) * (ANIM_MS / SPEEDS[speedIdx]);
  speedIdx--;
  document.getElementById('spd-lbl').textContent = SPEEDS[speedIdx] + '×';
  if (playing) startTime = performance.now() - (frame / (TOTAL - 1)) * (ANIM_MS / SPEEDS[speedIdx]);
}});

document.getElementById('btn-faster').addEventListener('click', () => {{
  if (speedIdx === SPEEDS.length - 1) return;
  speedIdx++;
  document.getElementById('spd-lbl').textContent = SPEEDS[speedIdx] + '×';
  if (playing) startTime = performance.now() - (frame / (TOTAL - 1)) * (ANIM_MS / SPEEDS[speedIdx]);
}});

document.getElementById('btn-restart').addEventListener('click', () => {{
  playing = false; animDone = false;
  frame = 0; startTime = null;
  document.getElementById('hover-badge').classList.remove('show');
  document.getElementById('btn-play').textContent = '⏸ Pause';
  document.getElementById('pb').style.width = '0%';
  Plotly.react('main-plot', refTraces(), BASE_LAYOUT);
  Plotly.react('pie-wrap', pieTrace({{}}), PIE_LAYOUT, {{ displayModeBar: false, responsive: true }});
  ['v-n', 'v-mae', 'v-rmse', 'v-mape', 'v-bias', 'v-cov'].forEach(id => {{ document.getElementById(id).textContent = '—'; }});
  document.getElementById('sample-counter').textContent = '';
  setTimeout(() => {{ playing = true; requestAnimationFrame(animStep); }}, 80);
}});

document.getElementById('pb-wrap').addEventListener('click', function(e) {{
  const pct = Math.max(0, Math.min(1, e.offsetX / this.offsetWidth));
  const targetIdx = Math.round(pct * (TOTAL - 1));
  playing = false;
  document.getElementById('btn-play').textContent = '▶ Play';
  frame = targetIdx;
  renderUpTo(frame);
  if (targetIdx >= TOTAL - 1) {{
    animDone = true;
    document.getElementById('hover-badge').classList.add('show');
    setupHover();
  }} else {{
    animDone = false;
  }}
}});

/* ── kick off ────────────────────────────────────────────────────────────── */
requestAnimationFrame(animStep);
</script>
</body>
</html>"""


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate a self-contained interactive depth-accuracy HTML from depth_experiment_results.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--csv", required=True, help="Path to depth_experiment_results.csv")
    ap.add_argument("--out", default="depth_accuracy_plots", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading: {args.csv}")
    rows = load_csv_rows(args.csv)
    points = build_points(rows)
    print(f"  {len(points)} points loaded.")
    groups = build_groups(points)
    print(f"  Quality groups: {[g['label'] for g in groups]}")

    data = build_data(points)
    print(f"  MAE={data['final']['mae']:.3f} m  RMSE={data['final']['rmse']:.3f} m  "
          f"MAPE={data['final']['mape']:.2f}%  Bias={data['final']['bias']:+.3f} m  "
          f"Coverage={data['final']['coverage']:.1f}%")

    html = generate_html(data, THEME)
    outpath = os.path.join(args.out, "depth_accuracy_thesis_style_interactive.html")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(outpath) / 1024
    print(f"\nSaved: {outpath}  ({size_kb:.1f} KB)")
    print("Open in any browser — fully self-contained, no server needed.")


if __name__ == "__main__":
    main()
