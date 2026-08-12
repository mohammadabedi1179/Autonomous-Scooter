#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate standalone interactive HTML dashboard for Stage-9 final comparison.

Input folder must contain exactly these six files:

final_eval_jsons/
├── test01_student_sensor_active_shield.json
├── test02_student_sensor_passive_shield.json
├── test03_teacher_gt_active_shield.json
├── test04_teacher_sensor_active_shield.json
├── test05_mpc_sensor_shield_off.json
└── test06_mpc_sensor_shield_on.json

Run:
python3 interactive_stage9_final_comparison.py \
  --dir final_eval_jsons \
  --out stage9_final_comparison_interactive.html
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


EVAL_MODES = [
    {
        "test": 1,
        "kind": "RL",
        "filename": "test01_student_sensor_active_shield.json",
        "label": "Test 1 — Student / Sensor / Active shield",
        "short": "T1 Student Sensor Active",
        "model": "Student",
        "observation": "Sensor",
        "shield": "Active",
    },
    {
        "test": 2,
        "kind": "RL",
        "filename": "test02_student_sensor_passive_shield.json",
        "label": "Test 2 — Student / Sensor / Passive shield",
        "short": "T2 Student Sensor Passive",
        "model": "Student",
        "observation": "Sensor",
        "shield": "Passive",
    },
    {
        "test": 3,
        "kind": "RL",
        "filename": "test03_teacher_gt_active_shield.json",
        "label": "Test 3 — Teacher / GT / Active shield",
        "short": "T3 Teacher GT Active",
        "model": "Teacher",
        "observation": "GT",
        "shield": "Active",
    },
    {
        "test": 4,
        "kind": "RL",
        "filename": "test04_teacher_sensor_active_shield.json",
        "label": "Test 4 — Teacher / Sensor / Active shield",
        "short": "T4 Teacher Sensor Active",
        "model": "Teacher",
        "observation": "Sensor",
        "shield": "Active",
    },
    {
        "test": 5,
        "kind": "MPC",
        "filename": "test05_mpc_sensor_shield_off.json",
        "label": "Test 5 — MPC / Sensor / Shield OFF",
        "short": "T5 MPC Shield OFF",
        "model": "MPC",
        "observation": "Sensor",
        "shield": "Off",
    },
    {
        "test": 6,
        "kind": "MPC",
        "filename": "test06_mpc_sensor_shield_on.json",
        "label": "Test 6 — MPC / Sensor / Shield ON",
        "short": "T6 MPC Shield ON",
        "model": "MPC",
        "observation": "Sensor",
        "shield": "Active",
    },
]


@dataclass
class RunSummary:
    test: int
    kind: str
    filename: str
    label: str
    short: str
    model: str
    observation: str
    shield: str

    n: int
    goal_count: int
    timeout_count: int
    collision_count: int
    other_count: int

    success_rate: float
    timeout_rate: float
    collision_rate: float
    other_rate: float

    mean_return: float | None
    mean_length: float | None
    median_clearance: float | None
    median_ttc: float | None

    shield_rate: float | None
    total_shield_activations: float | None

    median_wall_ms: float | None
    solver_fallback_rate: float | None


def _append_missing_closers(text: str) -> str:
    stack: List[str] = []
    in_string = False
    escaped = False
    matching = {"}": "{", "]": "["}

    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack and stack[-1] == matching[char]:
                stack.pop()
            else:
                raise ValueError("JSON fragment has mismatched brackets.")

    closers = {"{": "}", "[": "]"}
    return text + "".join(closers[item] for item in reversed(stack))


def load_eval_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig").strip()

    if not text:
        raise ValueError(f"JSON file is empty: {path}")

    def parse(candidate: str) -> Dict[str, Any]:
        result = json.loads(candidate, parse_constant=lambda token: float(token))
        if not isinstance(result, dict):
            raise ValueError("Top-level JSON value must be an object.")
        return result

    try:
        return parse(text)
    except Exception:
        repaired = text
        if not repaired.lstrip().startswith("{"):
            repaired = "{" + repaired
        repaired = re.sub(r",\s*$", "", repaired)
        repaired = _append_missing_closers(repaired)
        return parse(repaired)


def safe_float(value: Any, default: float = np.nan, allow_inf: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if np.isfinite(number) or (allow_inf and np.isinf(number)):
        return number

    return default


def numeric_array(values: Any, allow_inf: bool = False) -> np.ndarray:
    if not isinstance(values, (list, tuple)):
        return np.array([], dtype=float)

    return np.asarray(
        [safe_float(v, default=np.nan, allow_inf=allow_inf) for v in values],
        dtype=float,
    )


def finite_mean(values: List[float] | np.ndarray, default: float = np.nan) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else default


def finite_median(values: List[float] | np.ndarray, default: float = np.nan) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else default


def normalize_reason(reason: Any) -> str:
    r = str(reason).strip().lower()

    if "goal" in r or "success" in r or "reached" in r:
        return "goal"
    if "collision" in r or "crash" in r:
        return "collision"
    if "timeout" in r or "time_out" in r or "time-limit" in r or "time limit" in r:
        return "timeout"
    if "hard_cap" in r:
        return "timeout"

    return r or "unknown"


def percent(count: int, n: int) -> float:
    return 100.0 * count / n if n else 0.0


def none_if_nan(value: float) -> float | None:
    if value is None:
        return None
    try:
        if math.isfinite(float(value)):
            return float(value)
    except Exception:
        return None
    return None


def get_records(data: Dict[str, Any], path: Path) -> List[Dict[str, Any]]:
    records = data.get("episodes", [])
    if not isinstance(records, list) or not records:
        raise ValueError(f"No episodes found in: {path}")

    records = [r for r in records if isinstance(r, dict)]
    if not records:
        raise ValueError(f"Episode records exist but none are readable in: {path}")

    return records


def summarize_run(mode: Dict[str, Any], folder: Path) -> RunSummary:
    path = folder / mode["filename"]
    data = load_eval_json(path)
    records = get_records(data, path)

    reasons: List[str] = []
    returns: List[float] = []
    lengths: List[float] = []
    clearances: List[float] = []
    ttcs: List[float] = []
    shield_rates: List[float] = []
    shield_totals: List[float] = []
    durations: List[float] = []
    solver_counts: List[float] = []

    for record in records:
        summary = record.get("summary", record)
        raw = record.get("raw", {})

        if not isinstance(summary, dict):
            summary = {}
        if not isinstance(raw, dict):
            raw = {}

        reason = normalize_reason(summary.get("reason", "unknown"))
        reasons.append(reason)

        rewards = numeric_array(raw.get("rewards", []))
        v_cmds = numeric_array(raw.get("v_cmds", []))
        w_cmds = numeric_array(raw.get("w_cmds", []))
        goal_dists = numeric_array(raw.get("goal_dists", raw.get("dists", [])))
        min_ultras = numeric_array(raw.get("min_ultras", []))
        min_depths = numeric_array(raw.get("min_obs_depths", []))
        ttc_arr = numeric_array(raw.get("ttcs", []), allow_inf=True)

        return_sum = safe_float(summary.get("return_sum"), default=np.nan)
        if not np.isfinite(return_sum) and rewards.size:
            return_sum = float(np.nansum(rewards))
        returns.append(return_sum)

        length_steps = safe_float(summary.get("length_steps"), default=np.nan)
        if not np.isfinite(length_steps):
            length_steps = float(
                rewards.size
                or v_cmds.size
                or w_cmds.size
                or goal_dists.size
                or ttc_arr.size
            )
        lengths.append(length_steps)

        min_obs = safe_float(summary.get("min_obstacle_distance"), default=np.nan)
        if not np.isfinite(min_obs):
            candidates = np.concatenate([
                min_ultras[np.isfinite(min_ultras)],
                min_depths[np.isfinite(min_depths)],
            ])
            min_obs = float(np.min(candidates)) if candidates.size else np.nan
        clearances.append(min_obs)

        ttc_min = safe_float(summary.get("ttc_min"), default=np.nan, allow_inf=True)
        if np.isnan(ttc_min):
            finite_ttcs = ttc_arr[np.isfinite(ttc_arr)]
            if finite_ttcs.size:
                ttc_min = float(np.min(finite_ttcs))
            elif ttc_arr.size and np.all(np.isinf(ttc_arr)):
                ttc_min = np.inf
        ttcs.append(ttc_min)

        shield_states = raw.get("shield_active", [])
        if isinstance(shield_states, list):
            shield_states = [str(s).lower() for s in shield_states]
        else:
            shield_states = []

        caution = int(safe_float(
            summary.get("shield_count_caution"),
            default=shield_states.count("caution")
        ))
        danger = int(safe_float(
            summary.get("shield_count_danger"),
            default=shield_states.count("danger")
        ))
        critical = int(safe_float(
            summary.get("shield_count_critical"),
            default=shield_states.count("critical")
        ))

        shield_total = caution + danger + critical
        shield_totals.append(float(shield_total))

        shield_rate = safe_float(summary.get("shield_intervention_rate"), default=np.nan)
        if not np.isfinite(shield_rate):
            shield_rate = shield_total / len(shield_states) if shield_states else np.nan
        shield_rates.append(shield_rate)

        duration_sec = safe_float(summary.get("duration_sec"), default=np.nan)
        durations.append(duration_sec)

        solver_fails = numeric_array(raw.get("solver_fails", []))
        solver_count = safe_float(summary.get("solver_fail_count"), default=np.nan)
        if not np.isfinite(solver_count):
            finite_solver = solver_fails[np.isfinite(solver_fails)]
            solver_count = float(np.sum(finite_solver > 0.0)) if finite_solver.size else 0.0
        solver_counts.append(solver_count)

    n = len(reasons)

    goal_count = reasons.count("goal")
    timeout_count = reasons.count("timeout")
    collision_count = reasons.count("collision")
    other_count = max(0, n - goal_count - timeout_count - collision_count)

    finite_ttc_for_median = []
    for value in ttcs:
        if np.isfinite(value):
            finite_ttc_for_median.append(value)
        elif np.isinf(value):
            finite_ttc_for_median.append(50.0)

    lengths_arr = np.asarray(lengths, dtype=float)
    durations_arr = np.asarray(durations, dtype=float)

    valid_wall = np.isfinite(lengths_arr) & (lengths_arr > 0) & np.isfinite(durations_arr)
    wall_ms = np.full_like(lengths_arr, np.nan, dtype=float)
    wall_ms[valid_wall] = 1000.0 * durations_arr[valid_wall] / lengths_arr[valid_wall]

    total_steps = float(np.nansum(lengths_arr[np.isfinite(lengths_arr)]))
    total_solver_fails = float(np.nansum(solver_counts))
    solver_fallback_rate = (
        100.0 * total_solver_fails / total_steps
        if total_steps > 0
        else np.nan
    )

    return RunSummary(
        test=int(mode["test"]),
        kind=str(mode["kind"]),
        filename=str(mode["filename"]),
        label=str(mode["label"]),
        short=str(mode["short"]),
        model=str(mode["model"]),
        observation=str(mode["observation"]),
        shield=str(mode["shield"]),

        n=n,
        goal_count=goal_count,
        timeout_count=timeout_count,
        collision_count=collision_count,
        other_count=other_count,

        success_rate=percent(goal_count, n),
        timeout_rate=percent(timeout_count, n),
        collision_rate=percent(collision_count, n),
        other_rate=percent(other_count, n),

        mean_return=none_if_nan(finite_mean(returns)),
        mean_length=none_if_nan(finite_mean(lengths)),
        median_clearance=none_if_nan(finite_median(clearances)),
        median_ttc=none_if_nan(finite_median(finite_ttc_for_median)),

        shield_rate=none_if_nan(100.0 * finite_mean(shield_rates)),
        total_shield_activations=none_if_nan(float(np.nansum(shield_totals))),

        median_wall_ms=none_if_nan(finite_median(wall_ms)),
        solver_fallback_rate=none_if_nan(solver_fallback_rate),
    )


def to_embeddable_js(data: Any) -> str:
    text = json.dumps(data, ensure_ascii=False, allow_nan=False)
    text = text.replace("</script", "<\\/script").replace("<!--", "<\\!--")
    return text


def render_html(runs: List[RunSummary]) -> str:
    run_dicts = [asdict(r) for r in runs]
    embedded = to_embeddable_js(run_dicts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Stage-9 Final Comparison</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>

<style>
:root {{
  --paper: #FAF6F0;
  --panel: #FFFFFF;
  --ink: #3A2E22;
  --muted: #8C7B6E;
  --grid: #E4DCD0;
  --teal: #7D9D9C;
  --teal-dark: #5F7C7B;
  --amber: #C9A27E;
  --terracotta: #C2703D;
  --goal: #5A8A5A;
  --timeout: #B87820;
  --collision: #B83820;
  --fallback: #A09080;
  --shadow: 0 1px 2px rgba(58,46,34,0.06), 0 4px 14px rgba(58,46,34,0.06);
}}

* {{
  box-sizing: border-box;
}}

html, body {{
  margin: 0;
  padding: 0;
  background:
    radial-gradient(circle at 50% 0%, rgba(125,157,156,0.12), transparent 34%),
    linear-gradient(145deg, #FAF6F0 0%, #F2E8DE 60%, #EDE0D2 100%);
  color: var(--ink);
  font-family: 'Vazirmatn', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}

body {{
  padding: 28px clamp(18px, 4vw, 54px) 44px;
}}

header.top {{
  margin-bottom: 18px;
}}

.eyebrow {{
  margin: 0 0 4px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}}

h1 {{
  margin: 0;
  font-size: clamp(24px, 3vw, 38px);
  line-height: 1.2;
  font-weight: 800;
  color: var(--ink);
}}

.subtitle {{
  margin: 6px 0 0;
  color: var(--muted);
  font-size: 14px;
}}

.toolbar {{
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin: 18px 0;
}}

.filter-btn {{
  border: 1px solid var(--grid);
  background: rgba(255,255,255,0.65);
  color: var(--ink);
  border-radius: 999px;
  padding: 8px 14px;
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 700;
  cursor: pointer;
  box-shadow: var(--shadow);
}}

.filter-btn.active {{
  background: var(--teal);
  border-color: var(--teal);
  color: white;
}}

.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(160px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}}

.kpi-card {{
  background: rgba(255,255,255,0.78);
  border: 1px solid var(--grid);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 14px 16px;
}}

.kpi-card .label {{
  color: var(--muted);
  font-size: 11.5px;
  font-weight: 700;
}}

.kpi-card .value {{
  margin-top: 6px;
  color: var(--ink);
  font-size: 22px;
  font-weight: 800;
}}

.kpi-card .sub {{
  margin-top: 4px;
  color: var(--muted);
  font-size: 11.5px;
}}

.dashboard-grid {{
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 14px;
  align-items: stretch;
}}

.panel {{
  background: rgba(255,255,255,0.78);
  border: 1px solid var(--grid);
  border-radius: 18px;
  box-shadow: var(--shadow);
  padding: 12px 12px 8px;
  min-width: 0;
}}

.panel-head {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  padding: 2px 4px 8px;
}}

.panel-title {{
  font-size: 14px;
  font-weight: 800;
  color: var(--ink);
}}

.panel-sub {{
  font-size: 11px;
  color: var(--muted);
}}

.plot {{
  width: 100%;
  height: 520px;
}}

.side-stack {{
  display: grid;
  grid-template-rows: 1fr auto;
  gap: 14px;
}}

.tradeoff-plot {{
  height: 360px;
}}

.detail {{
  background: rgba(255,255,255,0.78);
  border: 1px solid var(--grid);
  border-radius: 18px;
  box-shadow: var(--shadow);
  padding: 16px;
}}

.detail .badge {{
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 5px 11px;
  background: #F1ECE4;
  color: var(--muted);
  font-size: 11px;
  font-weight: 800;
}}

.detail h2 {{
  margin: 12px 0 6px;
  font-size: 20px;
  line-height: 1.25;
}}

.detail p {{
  margin: 0 0 12px;
  color: var(--muted);
  font-size: 12.5px;
}}

.metric-table {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}}

.metric {{
  border: 1px solid var(--grid);
  background: rgba(250,246,240,0.72);
  border-radius: 12px;
  padding: 10px 12px;
}}

.metric .k {{
  color: var(--muted);
  font-size: 10.5px;
  font-weight: 700;
}}

.metric .v {{
  margin-top: 3px;
  color: var(--ink);
  font-size: 17px;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}}

.footnote {{
  margin-top: 18px;
  text-align: center;
  color: var(--muted);
  font-size: 11.5px;
}}

@media (max-width: 1000px) {{
  .kpi-grid {{
    grid-template-columns: repeat(2, 1fr);
  }}

  .dashboard-grid {{
    grid-template-columns: 1fr;
  }}

  .plot {{
    height: 460px;
  }}
}}
</style>
</head>

<body>

<header class="top">
  <p class="eyebrow">Safe DRL Scooter Navigation — Final Evaluation</p>
  <h1>Stage-9 final comparison</h1>
  <p class="subtitle">
    Six final tests under the same Stage-9 scenario: 3 static obstacles, 4 moving obstacles, and a fast moving goal.
  </p>
</header>

<div class="toolbar">
  <button class="filter-btn active" data-filter="all">All tests</button>
  <button class="filter-btn" data-filter="RL">RL policies</button>
  <button class="filter-btn" data-filter="MPC">MPC baselines</button>
  <button class="filter-btn" data-filter="Sensor">Sensor-based only</button>
</div>

<section class="kpi-grid" id="kpiGrid"></section>

<section class="dashboard-grid">
  <section class="panel">
    <div class="panel-head">
      <span class="panel-title">Terminal outcomes</span>
      <span class="panel-sub">Goal / Timeout / Collision composition</span>
    </div>
    <div id="outcomePlot" class="plot"></div>
  </section>

  <section class="side-stack">
    <section class="panel">
      <div class="panel-head">
        <span class="panel-title">Success–collision trade-off</span>
        <span class="panel-sub">Best region: high success, low collision</span>
      </div>
      <div id="tradeoffPlot" class="tradeoff-plot"></div>
    </section>

    <section class="detail" id="detailPanel"></section>
  </section>
</section>

<p class="footnote">
  Click any bar or point to inspect the corresponding test. Hover over charts for exact values.
</p>

<script>
const RUNS = {embedded};

const COLORS = {{
  paper: "#FAF6F0",
  panel: "#FFFFFF",
  bg: "#F8F4EE",
  ink: "#3A2E22",
  muted: "#8C7B6E",
  grid: "#E4DCD0",
  goal: "#5A8A5A",
  timeout: "#B87820",
  collision: "#B83820",
  other: "#A09080",
  rl: "#5A8A5A",
  mpc: "#B83820",
  amber: "#C9A27E",
  teal: "#7D9D9C"
}};

const state = {{
  filter: "all",
  selectedTest: null
}};

function fmtPct(v) {{
  return Number.isFinite(v) ? `${{v.toFixed(1)}}%` : "—";
}}

function fmtNum(v, digits = 2, suffix = "") {{
  return Number.isFinite(v) ? `${{v.toFixed(digits)}}${{suffix}}` : "—";
}}

function visibleRuns() {{
  if (state.filter === "all") return RUNS.slice();
  if (state.filter === "RL") return RUNS.filter(r => r.kind === "RL");
  if (state.filter === "MPC") return RUNS.filter(r => r.kind === "MPC");
  if (state.filter === "Sensor") return RUNS.filter(r => r.observation === "Sensor");
  return RUNS.slice();
}}

function commonLayout() {{
  return {{
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: COLORS.bg,
    font: {{
      family: "Vazirmatn, system-ui, sans-serif",
      color: COLORS.ink,
      size: 11
    }},
    margin: {{ l: 150, r: 36, t: 12, b: 58 }},
    xaxis: {{
      gridcolor: COLORS.grid,
      zeroline: false,
      tickfont: {{ color: COLORS.ink }}
    }},
    yaxis: {{
      gridcolor: COLORS.grid,
      zeroline: false,
      tickfont: {{ color: COLORS.ink }}
    }},
    legend: {{
      orientation: "h",
      y: -0.16,
      x: 0.5,
      xanchor: "center",
      font: {{ size: 10 }}
    }},
    hoverlabel: {{
      bgcolor: "#FFFFFF",
      bordercolor: COLORS.grid,
      font: {{ family: "Vazirmatn, system-ui, sans-serif", color: COLORS.ink }}
    }}
  }};
}}

const config = {{
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"]
}};

function renderKpis(runs) {{
  const bestSuccess = runs.reduce((a, b) => b.success_rate > a.success_rate ? b : a, runs[0]);
  const safest = runs.reduce((a, b) => b.collision_rate < a.collision_rate ? b : a, runs[0]);

  const sensorRL = runs
    .filter(r => r.kind === "RL" && r.observation === "Sensor")
    .sort((a, b) => b.success_rate - a.success_rate)[0];

  const bestMPC = runs
    .filter(r => r.kind === "MPC")
    .sort((a, b) => b.success_rate - a.success_rate)[0];

  let deltaText = "—";
  let deltaSub = "Need RL and MPC runs";
  if (sensorRL && bestMPC) {{
    deltaText = `+${{(sensorRL.success_rate - bestMPC.success_rate).toFixed(1)}} pp`;
    deltaSub = `Best sensor RL success advantage over best MPC`;
  }}

  const items = [
    {{
      label: "Highest success",
      value: bestSuccess ? fmtPct(bestSuccess.success_rate) : "—",
      sub: bestSuccess ? `T${{bestSuccess.test}} · ${{bestSuccess.model}}` : "—"
    }},
    {{
      label: "Lowest collision",
      value: safest ? fmtPct(safest.collision_rate) : "—",
      sub: safest ? `T${{safest.test}} · ${{safest.model}}` : "—"
    }},
    {{
      label: "Best sensor RL vs MPC",
      value: deltaText,
      sub: deltaSub
    }},
    {{
      label: "Selected set",
      value: `${{runs.length}} tests`,
      sub: state.filter === "all" ? "All final evaluations" : state.filter
    }}
  ];

  document.getElementById("kpiGrid").innerHTML = items.map(item => `
    <div class="kpi-card">
      <div class="label">${{item.label}}</div>
      <div class="value">${{item.value}}</div>
      <div class="sub">${{item.sub}}</div>
    </div>
  `).join("");
}}

function renderOutcomes(runs) {{
  const y = runs.map(r => r.short);
  const custom = runs.map(r => r.test);

  const selected = state.selectedTest;
  const opacities = runs.map(r => selected === null || r.test === selected ? 0.92 : 0.34);

  const traces = [
    {{
      type: "bar",
      orientation: "h",
      name: "Goal / Success",
      y,
      x: runs.map(r => r.success_rate),
      customdata: custom,
      marker: {{ color: COLORS.goal, opacity: opacities }},
      hovertemplate: "%{{y}}<br>Success: %{{x:.1f}}%<extra></extra>"
    }},
    {{
      type: "bar",
      orientation: "h",
      name: "Timeout",
      y,
      x: runs.map(r => r.timeout_rate),
      customdata: custom,
      marker: {{ color: COLORS.timeout, opacity: opacities }},
      hovertemplate: "%{{y}}<br>Timeout: %{{x:.1f}}%<extra></extra>"
    }},
    {{
      type: "bar",
      orientation: "h",
      name: "Collision",
      y,
      x: runs.map(r => r.collision_rate),
      customdata: custom,
      marker: {{ color: COLORS.collision, opacity: opacities }},
      hovertemplate: "%{{y}}<br>Collision: %{{x:.1f}}%<extra></extra>"
    }},
    {{
      type: "bar",
      orientation: "h",
      name: "Other",
      y,
      x: runs.map(r => r.other_rate),
      customdata: custom,
      marker: {{ color: COLORS.other, opacity: opacities }},
      hovertemplate: "%{{y}}<br>Other: %{{x:.1f}}%<extra></extra>"
    }}
  ];

  const layout = commonLayout();
  layout.barmode = "stack";
  layout.xaxis = Object.assign(layout.xaxis, {{
    title: "Episodes (%)",
    range: [0, 100]
  }});
  layout.yaxis = Object.assign(layout.yaxis, {{
    autorange: "reversed"
  }});

  Plotly.react("outcomePlot", traces, layout, config);

  const el = document.getElementById("outcomePlot");
  el.on("plotly_click", evt => {{
    const pt = evt.points && evt.points[0];
    if (!pt) return;
    selectTest(pt.customdata);
  }});
}}

function renderTradeoff(runs) {{
  const rl = runs.filter(r => r.kind === "RL");
  const mpc = runs.filter(r => r.kind === "MPC");

  function traceFor(group, name, color, symbol) {{
    return {{
      type: "scatter",
      mode: "markers+text",
      name,
      x: group.map(r => r.collision_rate),
      y: group.map(r => r.success_rate),
      text: group.map(r => `T${{r.test}}`),
      textposition: "top center",
      customdata: group.map(r => r.test),
      marker: {{
        color,
        symbol,
        size: group.map(r => state.selectedTest === null || r.test === state.selectedTest ? 15 : 10),
        opacity: group.map(r => state.selectedTest === null || r.test === state.selectedTest ? 0.92 : 0.36),
        line: {{ color: "#FFFFFF", width: 1.5 }}
      }},
      hovertemplate:
        "%{{text}}<br>Collision: %{{x:.1f}}%<br>Success: %{{y:.1f}}%<extra>" + name + "</extra>"
    }};
  }}

  const layout = commonLayout();
  layout.margin = {{ l: 58, r: 22, t: 12, b: 52 }};
  layout.xaxis = Object.assign(layout.xaxis, {{
    title: "Collision rate (%)",
    range: [-4, 78]
  }});
  layout.yaxis = Object.assign(layout.yaxis, {{
    title: "Success rate (%)",
    range: [20, 104]
  }});
  layout.shapes = [
    {{
      type: "rect",
      xref: "x",
      yref: "y",
      x0: 0,
      x1: 18,
      y0: 85,
      y1: 100,
      fillcolor: "rgba(90,138,90,0.08)",
      line: {{ width: 0 }}
    }}
  ];
  layout.annotations = [
    {{
      x: 2,
      y: 101,
      text: "desired region",
      showarrow: false,
      xanchor: "left",
      yanchor: "top",
      font: {{ size: 10, color: COLORS.muted }}
    }}
  ];

  Plotly.react(
    "tradeoffPlot",
    [
      traceFor(rl, "RL policy", COLORS.rl, "circle"),
      traceFor(mpc, "MPC", COLORS.mpc, "square")
    ],
    layout,
    config
  );

  const el = document.getElementById("tradeoffPlot");
  el.on("plotly_click", evt => {{
    const pt = evt.points && evt.points[0];
    if (!pt) return;
    selectTest(pt.customdata);
  }});
}}

function renderDetail() {{
  const runs = visibleRuns();
  let run = RUNS.find(r => r.test === state.selectedTest);

  if (!run || !runs.some(r => r.test === run.test)) {{
    run = runs[0] || RUNS[0];
    state.selectedTest = run ? run.test : null;
  }}

  if (!run) {{
    document.getElementById("detailPanel").innerHTML = "<p>No run selected.</p>";
    return;
  }}

  const kindColor = run.kind === "MPC" ? COLORS.collision : COLORS.goal;

  document.getElementById("detailPanel").innerHTML = `
    <span class="badge" style="background:${{kindColor}}22;color:${{kindColor}}">
      T${{run.test}} · ${{run.kind}}
    </span>
    <h2>${{run.model}} — ${{run.observation}} — ${{run.shield}}</h2>
    <p>${{run.label}}</p>

    <div class="metric-table">
      <div class="metric">
        <div class="k">Success</div>
        <div class="v">${{fmtPct(run.success_rate)}}</div>
      </div>
      <div class="metric">
        <div class="k">Collision</div>
        <div class="v">${{fmtPct(run.collision_rate)}}</div>
      </div>
      <div class="metric">
        <div class="k">Timeout</div>
        <div class="v">${{fmtPct(run.timeout_rate)}}</div>
      </div>
      <div class="metric">
        <div class="k">Episodes</div>
        <div class="v">${{run.n}}</div>
      </div>
      <div class="metric">
        <div class="k">Median clearance</div>
        <div class="v">${{fmtNum(run.median_clearance, 2, " m")}}</div>
      </div>
      <div class="metric">
        <div class="k">Median TTC</div>
        <div class="v">${{fmtNum(run.median_ttc, 2, " s")}}</div>
      </div>
      <div class="metric">
        <div class="k">Shield rate</div>
        <div class="v">${{fmtPct(run.shield_rate)}}</div>
      </div>
      <div class="metric">
        <div class="k">Wall time / step</div>
        <div class="v">${{fmtNum(run.median_wall_ms, 0, " ms")}}</div>
      </div>
    </div>
  `;
}}

function selectTest(test) {{
  if (state.selectedTest === test) {{
    state.selectedTest = null;
  }} else {{
    state.selectedTest = test;
  }}
  renderAll();
}}

function renderAll() {{
  const runs = visibleRuns();
  renderKpis(runs);
  renderOutcomes(runs);
  renderTradeoff(runs);
  renderDetail();
}}

document.querySelectorAll(".filter-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.filter = btn.dataset.filter;
    state.selectedTest = null;
    renderAll();
  }});
}});

renderAll();
</script>

</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interactive Stage-9 final comparison HTML dashboard."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Folder containing the six renamed JSON files.",
    )
    parser.add_argument(
        "--out",
        default="stage9_final_comparison_interactive.html",
        help="Output HTML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    folder = Path(args.dir)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    missing = []
    for mode in EVAL_MODES:
        if not (folder / mode["filename"]).exists():
            missing.append(mode["filename"])

    if missing:
        print("\n[ERROR] Missing required files:\n")
        for name in missing:
            print(f"  - {name}")

        print("\nExpected folder structure:\n")
        print(f"{folder}/")
        for mode in EVAL_MODES:
            print(f"├── {mode['filename']}")
        raise FileNotFoundError("Some required JSON files are missing.")

    summaries = [summarize_run(mode, folder) for mode in EVAL_MODES]

    html = render_html(summaries)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    print(f"[OK] Loaded {len(summaries)} final evaluation file(s)")
    print(f"[OK] Saved interactive dashboard: {output.resolve()}")

    print("\nSummary:")
    for s in summaries:
        print(
            f"  T{s.test} | {s.model:7s} | {s.observation:6s} | {s.shield:7s} | "
            f"success={s.success_rate:5.1f}% | "
            f"collision={s.collision_rate:5.1f}% | "
            f"timeout={s.timeout_rate:5.1f}%"
        )


if __name__ == "__main__":
    main()
