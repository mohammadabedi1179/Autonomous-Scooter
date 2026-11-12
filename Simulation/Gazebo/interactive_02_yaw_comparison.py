#!/usr/bin/env python3
"""
Interactive Yaw / Heading Comparison Plot — standalone HTML generator.
Companion to analyze_tss_yaw_csv_labeled_themed.py (same window-detection
logic, reimplemented in pure Python) and styled after interactive_01_reward.py
(same animated-reveal / play-pause-seek pattern, thesis_nude_contrast theme).

Reads the same CSV that analyze_tss_yaw_csv_labeled_themed.py consumes and
produces a single self-contained HTML file: the three heading traces
(Euler yaw / Quaternion heading / Gyro heading) sweep in over time, the
0/90/180/270/360 deg hold windows shade in as the playhead enters them, and
each window's median pops in once its "summary" sub-range has been swept —
matching the static plot's shaded windows + median markers + direct end
labels, but animated and scrubbable.

Usage:
    python interactive_02_yaw_comparison.py desk_quat_test.csv --out ./output
    python interactive_02_yaw_comparison.py desk_quat_test.csv \
        --targets 0,90,180,270,360 \
        --checkpoint-centers 1.5,9.0,21.0,33.0,45.0 \
        --hold-width 5 \
        --theme thesis_nude_contrast \
        --out ./output
"""

import argparse
import csv
import json
import math
import os


# ── theme system (color values mirrored from analyze_tss_yaw_csv_labeled_themed.py) ──

THEMES = {
    "muji": {
        "BG": "#f7f3ee", "GRID_CLR": "#e6dfd5", "TEXT_CLR": "#5f5a55",
        "EDGE_CLR": "#d8d1c7", "PAPER_WHITE": "#f7f3ee",
        "PRIMARY": "#a8c5d1", "SECONDARY": "#d8b4a0", "ACCENT": "#b7c4b2",
        "STAGE_BAND_COLORS": ["#f0ebe5", "#ebe6dd", "#e8e1d8", "#f2ede6"],
    },
    "soft_linen": {
        "BG": "#F8F5F1", "GRID_CLR": "#E7E0D8", "TEXT_CLR": "#5E5A56",
        "EDGE_CLR": "#D6CCC2", "PAPER_WHITE": "#F8F5F1",
        "PRIMARY": "#D8C3A5", "SECONDARY": "#C9ADA7", "ACCENT": "#B7B7A4",
        "STAGE_BAND_COLORS": ["#f1ebe4", "#ece5dc", "#e8ddd1", "#f3eee8"],
    },
    "nude_editorial": {
        "BG": "#FAF7F2", "GRID_CLR": "#ECE6DE", "TEXT_CLR": "#605A54",
        "EDGE_CLR": "#DDD2C7", "PAPER_WHITE": "#FAF7F2",
        "PRIMARY": "#B7A99A", "SECONDARY": "#C7A9A3", "ACCENT": "#A68A64",
        "STAGE_BAND_COLORS": ["#f3eee7", "#eee5db", "#e9ddd2", "#f7f2eb"],
    },
    "japanese_earth": {
        "BG": "#F6F1EB", "GRID_CLR": "#E4DBD0", "TEXT_CLR": "#5C5752",
        "EDGE_CLR": "#D8CEC2", "PAPER_WHITE": "#F6F1EB",
        "PRIMARY": "#A3B18A", "SECONDARY": "#DDBEA9", "ACCENT": "#B08968",
        "STAGE_BAND_COLORS": ["#efe6dc", "#e7ddd1", "#ddd1c3", "#f3ece3"],
    },
    "warm_gray": {
        "BG": "#FBFAF8", "GRID_CLR": "#ECE8E2", "TEXT_CLR": "#57534E",
        "EDGE_CLR": "#D6D3D1", "PAPER_WHITE": "#FBFAF8",
        "PRIMARY": "#A8A29E", "SECONDARY": "#CBBBA0", "ACCENT": "#78716C",
        "STAGE_BAND_COLORS": ["#f2f0ec", "#ece8e2", "#e6e1da", "#f7f5f1"],
    },
    "porcelain_sage": {
        "BG": "#FAF9F6", "GRID_CLR": "#E8E4DD", "TEXT_CLR": "#4F514A",
        "EDGE_CLR": "#D9D5CC", "PAPER_WHITE": "#FAF9F6",
        "PRIMARY": "#8FA998", "SECONDARY": "#D7BFAE", "ACCENT": "#A7B8C2",
        "STAGE_BAND_COLORS": ["#F1EFEA", "#EBE8E1", "#E5E0D7", "#F5F2EC"],
    },
    "thesis_nude_contrast": {
        "BG": "#FAF6F0", "GRID_CLR": "#E8DDD2", "TEXT_CLR": "#463F3A",
        "EDGE_CLR": "#D8CABD", "PAPER_WHITE": "#FAF6F0",
        "PRIMARY": "#7D9D9C", "SECONDARY": "#C9A27E", "ACCENT": "#B7A99A",
        "STAGE_BAND_COLORS": ["#F1E8DE", "#EADBCB", "#E3D1BF", "#F6EEE6"],
    },
    "oat_blueprint": {
        "BG": "#FAF8F2", "GRID_CLR": "#E3E0D8", "TEXT_CLR": "#50545A",
        "EDGE_CLR": "#D5D2CA", "PAPER_WHITE": "#FAF8F2",
        "PRIMARY": "#8EA4B8", "SECONDARY": "#CDBBA7", "ACCENT": "#A9A58F",
        "STAGE_BAND_COLORS": ["#F0EEE8", "#E9E7E0", "#E2DFD6", "#F5F3ED"],
    },
}

DEFAULT_TARGETS = [0, 90, 180, 270, 360]
REQUIRED_TIME_COL = "time_s"
SERIES_COLS = ["euler_yaw_deg", "quat_heading_deg", "gyro_heading_deg"]
SERIES_KEYS = {"euler_yaw_deg": "euler", "quat_heading_deg": "quat", "gyro_heading_deg": "gyro"}
SERIES_LABELS = {"euler": "Euler yaw", "quat": "Quaternion heading", "gyro": "Gyro heading"}


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#" + "".join(f"{max(0, min(255, int(round(c)))):02X}" for c in rgb)


def mix_hex(c1, c2, t=0.5):
    a, b = _hex_to_rgb(c1), _hex_to_rgb(c2)
    return _rgb_to_hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))


# ── basic helpers (mirror of analyze_tss_yaw_csv_labeled_themed.py, pure python) ──

def safe_float(x):
    if x is None:
        return None
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


def parse_float_list(text):
    if text is None or str(text).strip() == "":
        return []
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def median(values):
    vals = sorted(v for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def last_finite(values):
    for v in reversed(values):
        if v is not None:
            return v
    return None


# ── CSV loading ─────────────────────────────────────────────────────────────

def load_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_series(rows):
    if not rows:
        raise ValueError("CSV appears to be empty.")
    header = rows[0].keys()
    if REQUIRED_TIME_COL not in header:
        raise ValueError(f"CSV must contain a '{REQUIRED_TIME_COL}' column.")

    present = [c for c in SERIES_COLS if c in header]
    if not present:
        raise ValueError("CSV must contain at least one of: " + ", ".join(SERIES_COLS))

    t_raw, series_raw = [], {c: [] for c in present}
    for r in rows:
        t = safe_float(r.get(REQUIRED_TIME_COL))
        if t is None:
            continue
        t_raw.append(t)
        for c in present:
            series_raw[c].append(safe_float(r.get(c)))

    order = sorted(range(len(t_raw)), key=lambda i: t_raw[i])
    t_sorted = [t_raw[i] for i in order]
    series_sorted = {c: [series_raw[c][i] for i in order] for c in present}
    return t_sorted, series_sorted, present


# ── window detection (mirror of windows_by_reference_target / windows_from_manual_centers) ──

def contiguous_segments(mask):
    idx = [i for i, m in enumerate(mask) if m]
    if not idx:
        return []
    segments = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segments.append((start, prev))
            start = prev = i
    segments.append((start, prev))
    return segments


def shrink_window(start_t, end_t, shrink_fraction):
    if end_t <= start_t:
        return start_t, end_t
    width = end_t - start_t
    margin = width * shrink_fraction
    return start_t + margin, end_t - margin


def windows_by_reference_target(t, y, targets, tolerance_deg, min_duration_s, shrink_fraction):
    windows = []
    finite_idx = [i for i, v in enumerate(y) if v is not None]

    for target in targets:
        mask = [(v is not None and abs(v - target) <= tolerance_deg) for v in y]
        segments = contiguous_segments(mask)

        candidates = []
        for a, b in segments:
            start_t, end_t = t[a], t[b]
            duration = end_t - start_t
            if duration < min_duration_s:
                continue
            med = median(y[a:b + 1])
            if med is None:
                continue
            candidates.append((duration, -abs(med - target), a, b, med))

        if candidates:
            candidates.sort(reverse=True)
            _, _, a, b, _ = candidates[0]
            start, end = t[a], t[b]
            center = 0.5 * (start + end)
            method = "auto_near_reference"
        else:
            if not finite_idx:
                continue
            nearest = min(finite_idx, key=lambda i: abs(y[i] - target))
            center = t[nearest]
            half = max(min_duration_s, 2.0) / 2.0
            start = max(t[0], center - half)
            end = min(t[-1], center + half)
            method = "fallback_nearest"

        s2, e2 = shrink_window(start, end, shrink_fraction)
        windows.append({
            "target": target, "startS": start, "endS": end,
            "summaryStartS": s2, "summaryEndS": e2, "centerS": center,
            "method": method,
        })

    windows.sort(key=lambda w: w["centerS"])
    for i, w in enumerate(windows):
        w["idx"] = i
    return windows


def windows_from_manual_centers(t, targets, centers, hold_width, shrink_fraction):
    if len(centers) != len(targets):
        raise ValueError("Number of --checkpoint-centers must match number of --targets.")
    t_min, t_max = t[0], t[-1]
    half = hold_width / 2.0
    windows = []
    for target, center in zip(targets, centers):
        start = max(t_min, center - half)
        end = min(t_max, center + half)
        s2, e2 = shrink_window(start, end, shrink_fraction)
        windows.append({
            "target": target, "startS": start, "endS": end,
            "summaryStartS": s2, "summaryEndS": e2, "centerS": center,
            "method": "manual_center",
        })
    windows.sort(key=lambda w: w["centerS"])
    for i, w in enumerate(windows):
        w["idx"] = i
    return windows


def annotate_window_medians(windows, t, series_sorted):
    for w in windows:
        lo, hi = w["summaryStartS"], w["summaryEndS"]
        idxs = [i for i, tt in enumerate(t) if lo <= tt <= hi]
        med = {}
        for col, key in SERIES_KEYS.items():
            if col not in series_sorted:
                continue
            med[key] = median([series_sorted[col][i] for i in idxs])
        w["medians"] = med


# ── data assembly ────────────────────────────────────────────────────────────

def compute_y_range(series_sorted, targets):
    all_vals = []
    for vals in series_sorted.values():
        all_vals.extend(v for v in vals if v is not None)
    all_vals.extend(targets)
    if not all_vals:
        return [-10.0, 10.0]
    y_min, y_max = min(all_vals), max(all_vals)
    pad = max(10.0, (y_max - y_min) * 0.08)
    return [round(y_min - pad, 1), round(y_max + pad, 1)]


def build_data(t, series_sorted, windows, targets, theme):
    for i, w in enumerate(windows):
        w["band"] = theme["STAGE_BAND_COLORS"][i % len(theme["STAGE_BAND_COLORS"])]
    annotate_window_medians(windows, t, series_sorted)

    out_series, final_vals = {}, {}
    for col, key in SERIES_KEYS.items():
        if col in series_sorted:
            out_series[key] = [round(v, 3) if v is not None else None for v in series_sorted[col]]
            final_vals[key] = last_finite(series_sorted[col])
        else:
            out_series[key] = None
            final_vals[key] = None

    return {
        "time": [round(v, 3) for v in t],
        "euler": out_series["euler"],
        "quat": out_series["quat"],
        "gyro": out_series["gyro"],
        "windows": [
            {
                "idx": w["idx"], "target": w["target"],
                "startS": round(w["startS"], 3), "endS": round(w["endS"], 3),
                "summaryStartS": round(w["summaryStartS"], 3), "summaryEndS": round(w["summaryEndS"], 3),
                "centerS": round(w["centerS"], 3),
                "medians": {k: (round(v, 2) if v is not None else None) for k, v in w["medians"].items()},
                "band": w["band"],
            }
            for w in windows
        ],
        "targets": targets,
        "finalVals": {k: (round(v, 2) if v is not None else None) for k, v in final_vals.items()},
        "xRange": [round(t[0], 3), round(t[-1], 3)],
        "yRange": compute_y_range(series_sorted, targets),
        "total": len(t),
    }


# ── HTML / JS generation ──────────────────────────────────────────────────────

def generate_html(data, theme):
    data_json = json.dumps(data, separators=(",", ":"))

    bg, grid, text = theme["BG"], theme["GRID_CLR"], theme["TEXT_CLR"]
    edge, paper = theme["EDGE_CLR"], theme["PAPER_WHITE"]
    primary, secondary, accent = theme["PRIMARY"], theme["SECONDARY"], theme["ACCENT"]
    bg2 = mix_hex(bg, edge, 0.35)
    bg3 = mix_hex(bg, edge, 0.55)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yaw / Heading Comparison — Interactive</title>
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
  --muted:     {text};
  --faint:     {mix_hex(text, bg, 0.45)};
  --primary:   {primary};
  --secondary: {secondary};
  --accent:    {accent};
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
  font-size: 13px; font-weight: 500;
  color: var(--muted); letter-spacing: .03em;
  padding: 0 4px 6px;
}}
#main-plot {{ flex: 1; min-height: 0; }}

#right {{
  width: 216px; display: flex; flex-direction: column; gap: 8px;
}}
.tile {{
  background: var(--bg2);
  border: 0.5px solid var(--border);
  border-radius: 10px;
  padding: 10px 14px;
}}
.tile-lbl {{
  font-size: 10px; font-weight: 500;
  color: var(--faint);
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-bottom: 5px;
}}
.tile-val {{
  font-size: 24px; font-weight: 500;
  line-height: 1.1;
  font-family: var(--mono);
  transition: color .18s ease;
}}
#tile-windows {{
  flex: 1; min-height: 0;
  display: flex; flex-direction: column;
}}
#win-list {{
  flex: 1; min-height: 60px;
  overflow-y: auto;
  display: flex; flex-direction: column; gap: 6px;
  padding-right: 2px;
}}
#win-list::-webkit-scrollbar {{ width: 4px; }}
#win-list::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}
.win-row {{
  background: var(--bg);
  border: 0.5px solid var(--border);
  border-radius: 7px;
  padding: 5px 8px;
  animation: rowIn .25s ease;
}}
@keyframes rowIn {{ from {{ opacity: 0; transform: translateY(-3px); }} to {{ opacity: 1; transform: translateY(0); }} }}
.win-row-h {{
  font-size: 11px; font-weight: 500; color: var(--text);
  display: flex; justify-content: space-between;
}}
.win-row-t {{ font-size: 10px; color: var(--faint); font-family: var(--mono); font-weight: 400; }}
.win-row-vals {{ font-size: 10.5px; color: var(--muted); margin-top: 2px; font-family: var(--mono); }}

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
  height: 100%; background: var(--primary);
  border-radius: 3px; width: 0%;
  pointer-events: none;
  transition: width .04s linear;
}}
#pb-thumb {{
  position: absolute; top: 50%; right: calc(100% - 100%);
  width: 10px; height: 10px;
  background: var(--primary);
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

#sample-counter {{
  position: absolute; top: 8px; right: 8px;
  font-size: 10px; font-family: var(--mono);
  color: var(--faint); pointer-events: none;
}}
</style>
</head>
<body>
<div id="app">
  <div id="main-row">

    <div id="left">
      <div id="plot-title">Yaw / Heading Comparison</div>
      <div id="main-plot"></div>
      <div id="hover-badge">hover over any line to inspect</div>
      <div id="sample-counter"></div>
    </div>

    <div id="right">
      <div class="tile">
        <div class="tile-lbl">Time</div>
        <div class="tile-val" id="v-time">—</div>
      </div>
      <div class="tile">
        <div class="tile-lbl">Euler yaw</div>
        <div class="tile-val" id="v-euler" style="color:var(--primary)">—</div>
      </div>
      <div class="tile">
        <div class="tile-lbl">Quaternion heading</div>
        <div class="tile-val" id="v-quat" style="color:var(--secondary)">—</div>
      </div>
      <div class="tile">
        <div class="tile-lbl">Gyro heading</div>
        <div class="tile-val" id="v-gyro" style="color:var(--accent)">—</div>
      </div>
      <div class="tile" id="tile-windows">
        <div class="tile-lbl">Window medians</div>
        <div id="win-list"></div>
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
  bg: '{bg}', grid: '{grid}', text: '{text}', edge: '{edge}', paper: '{paper}',
  primary: '{primary}', secondary: '{secondary}', accent: '{accent}',
}};
const HAS = {{ euler: D.euler !== null, quat: D.quat !== null, gyro: D.gyro !== null }};
const SPEEDS  = [0.25, 0.5, 1, 2, 4, 8];
const ANIM_MS = 16000;
const TOTAL   = D.total;

/* ── state ───────────────────────────────────────────────────────────────── */
let speedIdx       = 2;
let playing        = true;
let frame          = 0;
let startTime       = null;
let animDone       = false;
let endLabelsShown = false;
let shownBands     = new Set();
let shownMedians   = new Set();
let shapes         = [];
let annots         = [];

/* ── reference target lines (static, drawn once) ────────────────────────── */
const REF_SHAPES = D.targets.map(v => ({{
  type: 'line', xref: 'x', yref: 'y',
  x0: D.xRange[0], x1: D.xRange[1], y0: v, y1: v,
  line: {{ color: COLORS.edge, width: 0.9, dash: 'dash' }},
}}));

/* ── Plotly layout ───────────────────────────────────────────────────────── */
const BASE_LAYOUT = {{
  paper_bgcolor: COLORS.bg,
  plot_bgcolor:  COLORS.bg,
  font: {{ family: "'DM Sans', sans-serif", color: COLORS.text, size: 12 }},
  margin: {{ t: 46, r: 26, b: 48, l: 66 }},
  xaxis: {{
    title: {{ text: 'Time [s]', font: {{ size: 12 }} }},
    gridcolor: COLORS.grid, linecolor: COLORS.edge,
    tickcolor: COLORS.text, tickfont: {{ size: 11 }},
    range: D.xRange, zeroline: false, fixedrange: false,
  }},
  yaxis: {{
    title: {{ text: 'Heading [deg]', font: {{ size: 12 }} }},
    gridcolor: COLORS.grid, linecolor: COLORS.edge,
    tickcolor: COLORS.text, tickfont: {{ size: 11 }},
    range: D.yRange, zeroline: true,
    zerolinecolor: COLORS.edge, zerolinewidth: 1,
    fixedrange: false,
  }},
  legend: {{
    x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top',
    bgcolor: 'rgba(255,253,249,0.92)',
    bordercolor: COLORS.edge, borderwidth: 1,
    font: {{ size: 11 }},
  }},
  shapes: REF_SHAPES, annotations: [],
  hovermode: 'closest',
  hoverlabel: {{
    bgcolor: COLORS.paper, bordercolor: COLORS.edge,
    font: {{ family: "'DM Mono', monospace", size: 11, color: COLORS.text }},
  }},
}};

/* ── trace templates: 0 euler, 1 quat, 2 gyro, 3-5 their window medians ──── */
const BASE_TRACES = [
  {{ name: 'Euler yaw', x: [], y: [], mode: 'lines', type: 'scattergl',
     line: {{ color: COLORS.primary, width: 2.3 }}, visible: HAS.euler,
     hovertemplate: 't=%{{x:.2f}}s<br>Euler: %{{y:.1f}}°<extra></extra>' }},
  {{ name: 'Quaternion heading', x: [], y: [], mode: 'lines', type: 'scattergl',
     line: {{ color: COLORS.secondary, width: 2.1 }}, visible: HAS.quat,
     hovertemplate: 't=%{{x:.2f}}s<br>Quat: %{{y:.1f}}°<extra></extra>' }},
  {{ name: 'Gyro heading', x: [], y: [], mode: 'lines', type: 'scattergl',
     line: {{ color: COLORS.accent, width: 2.1 }}, visible: HAS.gyro,
     hovertemplate: 't=%{{x:.2f}}s<br>Gyro: %{{y:.1f}}°<extra></extra>' }},
  {{ name: 'Euler median', x: [], y: [], mode: 'markers', type: 'scatter', showlegend: false,
     marker: {{ color: COLORS.primary, size: 8, line: {{ color: COLORS.paper, width: 1.2 }} }},
     hovertemplate: 'window median<br>%{{y:.1f}}°<extra></extra>' }},
  {{ name: 'Quat median', x: [], y: [], mode: 'markers', type: 'scatter', showlegend: false,
     marker: {{ color: COLORS.secondary, size: 8, line: {{ color: COLORS.paper, width: 1.2 }} }},
     hovertemplate: 'window median<br>%{{y:.1f}}°<extra></extra>' }},
  {{ name: 'Gyro median', x: [], y: [], mode: 'markers', type: 'scatter', showlegend: false,
     marker: {{ color: COLORS.accent, size: 8, line: {{ color: COLORS.paper, width: 1.2 }} }},
     hovertemplate: 'window median<br>%{{y:.1f}}°<extra></extra>' }},
];

Plotly.newPlot('main-plot', BASE_TRACES, BASE_LAYOUT, {{
  responsive: true, displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
}});

/* ── DOM helpers ─────────────────────────────────────────────────────────── */
function fmt(v, dec) {{
  if (v === null || v === undefined) return '—';
  const n = Number(v);
  if (!isFinite(n)) return '—';
  return n.toFixed(dec !== undefined ? dec : 1);
}}

function updateTiles(i) {{
  document.getElementById('v-time').textContent  = fmt(D.time[i], 1) + 's';
  document.getElementById('v-euler').textContent = HAS.euler ? fmt(D.euler[i], 1) + '°' : '—';
  document.getElementById('v-quat').textContent  = HAS.quat  ? fmt(D.quat[i], 1) + '°'  : '—';
  document.getElementById('v-gyro').textContent  = HAS.gyro  ? fmt(D.gyro[i], 1) + '°'  : '—';
  document.getElementById('sample-counter').textContent = `${{i + 1}} / ${{TOTAL}}`;
}}

function relayoutNow() {{
  Plotly.relayout('main-plot', {{ shapes: [...REF_SHAPES, ...shapes], annotations: [...annots] }});
}}

function medianAnnot(x, y, text, color, dy) {{
  return {{
    x: x, y: y, xref: 'x', yref: 'y', text: text, showarrow: false, yshift: dy,
    font: {{ size: 9.5, color: color, family: "'DM Mono', monospace" }},
    bgcolor: COLORS.paper, bordercolor: COLORS.edge, borderwidth: 0.6, borderpad: 2,
  }};
}}

function revealBand(w, immediate) {{
  shapes.push({{
    type: 'rect', xref: 'x', yref: 'paper',
    x0: w.startS, x1: w.endS, y0: 0, y1: 1,
    fillcolor: w.band, opacity: 0.60, line: {{ width: 0 }}, layer: 'below',
  }});
  annots.push({{
    x: w.centerS, y: 1.05, xref: 'x', yref: 'paper',
    text: `${{w.target}}° window`, showarrow: false,
    font: {{ size: 10, color: COLORS.text, family: "'DM Sans', sans-serif" }},
    bgcolor: COLORS.paper, bordercolor: COLORS.edge, borderwidth: 0.75, borderpad: 3,
  }});
  if (immediate !== false) relayoutNow();
}}

function addWindowRow(w, rows) {{
  const el = document.createElement('div');
  el.className = 'win-row';
  el.innerHTML = `<div class="win-row-h"><span>${{w.target}}° window</span><span class="win-row-t">t=${{fmt(w.centerS, 1)}}s</span></div><div class="win-row-vals">${{rows.join(' &middot; ')}}</div>`;
  document.getElementById('win-list').appendChild(el);
  document.getElementById('win-list').scrollTop = document.getElementById('win-list').scrollHeight;
}}

function clearWindowRows() {{ document.getElementById('win-list').innerHTML = ''; }}

function revealMedians(w, immediate) {{
  const rows = [];
  if (HAS.euler && w.medians.euler !== null && w.medians.euler !== undefined) {{
    Plotly.extendTraces('main-plot', {{ x: [[w.centerS]], y: [[w.medians.euler]] }}, [3]);
    annots.push(medianAnnot(w.centerS, w.medians.euler, `${{fmt(w.medians.euler, 1)}}°`, COLORS.primary, -9));
    rows.push(`<span style="color:${{COLORS.primary}}">Euler ${{fmt(w.medians.euler, 1)}}°</span>`);
  }}
  if (HAS.quat && w.medians.quat !== null && w.medians.quat !== undefined) {{
    Plotly.extendTraces('main-plot', {{ x: [[w.centerS]], y: [[w.medians.quat]] }}, [4]);
    annots.push(medianAnnot(w.centerS, w.medians.quat, `${{fmt(w.medians.quat, 1)}}°`, COLORS.secondary, 12));
    rows.push(`<span style="color:${{COLORS.secondary}}">Quat ${{fmt(w.medians.quat, 1)}}°</span>`);
  }}
  if (HAS.gyro && w.medians.gyro !== null && w.medians.gyro !== undefined) {{
    Plotly.extendTraces('main-plot', {{ x: [[w.centerS]], y: [[w.medians.gyro]] }}, [5]);
    annots.push(medianAnnot(w.centerS, w.medians.gyro, `${{fmt(w.medians.gyro, 1)}}°`, COLORS.accent, -9));
    rows.push(`<span style="color:${{COLORS.accent}}">Gyro ${{fmt(w.medians.gyro, 1)}}°</span>`);
  }}
  if (immediate !== false) relayoutNow();
  addWindowRow(w, rows);
}}

function showEndLabels() {{
  if (endLabelsShown) return;
  endLabelsShown = true;
  const specs = [['euler', COLORS.primary, 'Euler'], ['quat', COLORS.secondary, 'Quat'], ['gyro', COLORS.accent, 'Gyro']];
  for (const [key, color, label] of specs) {{
    if (!HAS[key]) continue;
    const fv = D.finalVals[key];
    if (fv === null || fv === undefined) continue;
    annots.push({{
      x: D.xRange[1], y: fv, xref: 'x', yref: 'y', xshift: 8, xanchor: 'left',
      text: `${{label}}: ${{fmt(fv, 1)}}`, showarrow: false,
      font: {{ size: 10.5, color: color, family: "'DM Mono', monospace" }},
      bgcolor: COLORS.paper, bordercolor: COLORS.edge, borderwidth: 0.75, borderpad: 3,
    }});
  }}
  relayoutNow();
}}

/* ── hover (post-animation) ──────────────────────────────────────────────── */
function closestIndex(t) {{
  let lo = 0, hi = D.time.length - 1;
  while (lo < hi) {{
    const mid = (lo + hi) >> 1;
    if (D.time[mid] < t) lo = mid + 1; else hi = mid;
  }}
  return lo;
}}

function setupHover() {{
  document.getElementById('main-plot').on('plotly_hover', function(evt) {{
    const pt = evt.points[0];
    if (pt.curveNumber > 2) return;
    updateTiles(closestIndex(pt.x));
  }});
}}

/* ── animation ───────────────────────────────────────────────────────────── */
function targetFrame(elapsed) {{
  const dur = ANIM_MS / SPEEDS[speedIdx];
  return Math.min(TOTAL - 1, Math.floor((elapsed / dur) * TOTAL));
}}

function checkWindows(curT, immediate) {{
  for (const w of D.windows) {{
    if (!shownBands.has(w.idx) && curT >= w.startS) {{
      shownBands.add(w.idx);
      revealBand(w, immediate);
    }}
    if (!shownMedians.has(w.idx) && curT >= w.summaryEndS) {{
      shownMedians.add(w.idx);
      revealMedians(w, immediate);
    }}
  }}
}}

function animStep(ts) {{
  if (!playing) return;
  if (!startTime) startTime = ts;

  const target = targetFrame(ts - startTime);
  if (target <= frame) {{ requestAnimationFrame(animStep); return; }}

  const ext = {{ 0: {{ x: [], y: [] }}, 1: {{ x: [], y: [] }}, 2: {{ x: [], y: [] }} }};
  for (let i = frame + 1; i <= target; i++) {{
    const t = D.time[i];
    if (HAS.euler) {{ ext[0].x.push(t); ext[0].y.push(D.euler[i]); }}
    if (HAS.quat)  {{ ext[1].x.push(t); ext[1].y.push(D.quat[i]);  }}
    if (HAS.gyro)  {{ ext[2].x.push(t); ext[2].y.push(D.gyro[i]);  }}
  }}
  for (let ti = 0; ti < 3; ti++) {{
    if (ext[ti].x.length > 0) Plotly.extendTraces('main-plot', {{ x: [ext[ti].x], y: [ext[ti].y] }}, [ti]);
  }}

  checkWindows(D.time[target], true);
  updateTiles(target);

  const pct = (target / (TOTAL - 1)) * 100;
  document.getElementById('pb').style.width = pct + '%';
  document.getElementById('pb-thumb').style.left = `calc(${{pct}}% - 5px)`;

  frame = target;

  if (frame >= TOTAL - 1) {{
    animDone = true; playing = false;
    showEndLabels();
    document.getElementById('btn-play').textContent = '▶ Play';
    document.getElementById('hover-badge').classList.add('show');
    setupHover();
    return;
  }}
  requestAnimationFrame(animStep);
}}

/* ── rebuild to a specific frame (for seek) ──────────────────────────────── */
function rebuildToFrame(targetIdx) {{
  if (targetIdx < 0) targetIdx = 0;
  if (targetIdx >= TOTAL) targetIdx = TOTAL - 1;

  shapes = []; annots = [];
  shownBands.clear(); shownMedians.clear();
  endLabelsShown = false;
  clearWindowRows();

  const ext = {{ 0: {{ x: [], y: [] }}, 1: {{ x: [], y: [] }}, 2: {{ x: [], y: [] }} }};
  const med = {{ 3: {{ x: [], y: [] }}, 4: {{ x: [], y: [] }}, 5: {{ x: [], y: [] }} }};

  for (let i = 0; i <= targetIdx; i++) {{
    const t = D.time[i];
    if (HAS.euler) {{ ext[0].x.push(t); ext[0].y.push(D.euler[i]); }}
    if (HAS.quat)  {{ ext[1].x.push(t); ext[1].y.push(D.quat[i]);  }}
    if (HAS.gyro)  {{ ext[2].x.push(t); ext[2].y.push(D.gyro[i]);  }}

    for (const w of D.windows) {{
      if (!shownBands.has(w.idx) && t >= w.startS) {{ shownBands.add(w.idx); revealBand(w, false); }}
      if (!shownMedians.has(w.idx) && t >= w.summaryEndS) {{
        shownMedians.add(w.idx);
        const rows = [];
        if (HAS.euler && w.medians.euler !== null) {{
          med[3].x.push(w.centerS); med[3].y.push(w.medians.euler);
          annots.push(medianAnnot(w.centerS, w.medians.euler, `${{fmt(w.medians.euler, 1)}}°`, COLORS.primary, -9));
          rows.push(`<span style="color:${{COLORS.primary}}">Euler ${{fmt(w.medians.euler, 1)}}°</span>`);
        }}
        if (HAS.quat && w.medians.quat !== null) {{
          med[4].x.push(w.centerS); med[4].y.push(w.medians.quat);
          annots.push(medianAnnot(w.centerS, w.medians.quat, `${{fmt(w.medians.quat, 1)}}°`, COLORS.secondary, 12));
          rows.push(`<span style="color:${{COLORS.secondary}}">Quat ${{fmt(w.medians.quat, 1)}}°</span>`);
        }}
        if (HAS.gyro && w.medians.gyro !== null) {{
          med[5].x.push(w.centerS); med[5].y.push(w.medians.gyro);
          annots.push(medianAnnot(w.centerS, w.medians.gyro, `${{fmt(w.medians.gyro, 1)}}°`, COLORS.accent, -9));
          rows.push(`<span style="color:${{COLORS.accent}}">Gyro ${{fmt(w.medians.gyro, 1)}}°</span>`);
        }}
        addWindowRow(w, rows);
      }}
    }}
  }}

  const newTraces = BASE_TRACES.map((tr, ti) => {{
    if (ti < 3) return {{ ...tr, x: [...ext[ti].x], y: [...ext[ti].y] }};
    return {{ ...tr, x: [...med[ti].x], y: [...med[ti].y] }};
  }});
  Plotly.react('main-plot', newTraces, {{ ...BASE_LAYOUT, shapes: [...REF_SHAPES, ...shapes], annotations: [...annots] }});

  updateTiles(targetIdx);

  const pct = (targetIdx / (TOTAL - 1)) * 100;
  document.getElementById('pb').style.width = pct + '%';
  document.getElementById('pb-thumb').style.left = `calc(${{pct}}% - 5px)`;

  frame = targetIdx;

  if (targetIdx >= TOTAL - 1) {{
    showEndLabels();
    animDone = true;
    document.getElementById('hover-badge').classList.add('show');
    setupHover();
  }} else {{
    animDone = false;
  }}
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
  shapes = []; annots = [];
  shownBands.clear(); shownMedians.clear();
  endLabelsShown = false;
  clearWindowRows();

  document.getElementById('hover-badge').classList.remove('show');
  document.getElementById('btn-play').textContent = '⏸ Pause';
  document.getElementById('pb').style.width = '0%';

  Plotly.react('main-plot',
    BASE_TRACES.map(t => ({{ ...t, x: [], y: [] }})),
    {{ ...BASE_LAYOUT, shapes: [...REF_SHAPES], annotations: [] }});

  ['v-time', 'v-euler', 'v-quat', 'v-gyro'].forEach(id => {{ document.getElementById(id).textContent = '—'; }});
  document.getElementById('sample-counter').textContent = '';

  setTimeout(() => {{ playing = true; requestAnimationFrame(animStep); }}, 80);
}});

document.getElementById('pb-wrap').addEventListener('click', function(e) {{
  const pct = Math.max(0, Math.min(1, e.offsetX / this.offsetWidth));
  const targetIdx = Math.round(pct * (TOTAL - 1));
  playing = false;
  document.getElementById('btn-play').textContent = '▶ Play';
  rebuildToFrame(targetIdx);
}});

/* ── kick off ────────────────────────────────────────────────────────────── */
requestAnimationFrame(animStep);
</script>
</body>
</html>"""


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate a self-contained interactive Yaw/Heading comparison HTML from a TSS desk-test CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("csv", help="Input CSV file from the TSS logger (same file used by analyze_tss_yaw_csv_labeled_themed.py).")
    ap.add_argument("--out", default="./output", help="Output directory.")
    ap.add_argument("--targets", default=",".join(str(x) for x in DEFAULT_TARGETS), help="Comma-separated physical target angles, e.g. 0,90,180,270,360")
    ap.add_argument("--checkpoint-centers", default=None, help="Optional comma-separated center times for each target window.")
    ap.add_argument("--hold-width", type=float, default=5.0, help="Window width in seconds when using manual centers.")
    ap.add_argument("--reference-col", default="euler_yaw_deg", help="Column used to auto-detect windows.")
    ap.add_argument("--window-tolerance-deg", type=float, default=25.0, help="Auto-window tolerance around target angle.")
    ap.add_argument("--min-window-duration", type=float, default=1.0, help="Minimum duration for auto-detected window.")
    ap.add_argument("--summary-shrink", type=float, default=0.18, help="Fraction removed from each side for median calculation.")
    ap.add_argument("--theme", default="thesis_nude_contrast", choices=list(THEMES.keys()), help="Color theme, matches analyze_tss_yaw_csv_labeled_themed.py.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    theme = THEMES[args.theme]

    print(f"Loading: {args.csv}")
    rows = load_csv_rows(args.csv)
    t, series_sorted, present = build_series(rows)
    print(f"  {len(t)} samples loaded. Series present: {', '.join(present)}")

    targets = parse_float_list(args.targets)
    centers = parse_float_list(args.checkpoint_centers) if args.checkpoint_centers else []

    ref_col = args.reference_col if args.reference_col in series_sorted else present[0]
    if ref_col != args.reference_col:
        print(f"  Reference column '{args.reference_col}' not found, using '{ref_col}' instead.")

    if centers:
        windows = windows_from_manual_centers(t, targets, centers, args.hold_width, args.summary_shrink)
    else:
        windows = windows_by_reference_target(
            t, series_sorted[ref_col], targets,
            args.window_tolerance_deg, args.min_window_duration, args.summary_shrink,
        )
    print(f"  {len(windows)} windows detected.")

    data = build_data(t, series_sorted, windows, targets, theme)
    html = generate_html(data, theme)

    stem = os.path.splitext(os.path.basename(args.csv))[0]
    outpath = os.path.join(args.out, f"{stem}_yaw_comparison_interactive.html")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(outpath) / 1024
    print(f"\nSaved: {outpath}  ({size_kb:.1f} KB)")
    print("Open in any browser — fully self-contained, no server needed.")


if __name__ == "__main__":
    main()
