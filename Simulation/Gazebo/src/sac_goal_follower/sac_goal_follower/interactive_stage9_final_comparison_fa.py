#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ساخت داشبورد تعاملی فارسی برای مقایسه نهایی مرحله نهم.

این کد شش فایل JSON ارزیابی نهایی را می‌خواند و یک فایل HTML مستقل می‌سازد.
داده‌ها داخل خود HTML قرار می‌گیرند؛ بنابراین بعد از ساخت خروجی، دیگر نیازی
به همراه داشتن فایل‌های JSON نیست.

ساختار پوشه ورودی:

final_eval_jsons/
├── test01_student_sensor_active_shield.json
├── test02_student_sensor_passive_shield.json
├── test03_teacher_gt_active_shield.json
├── test04_teacher_sensor_active_shield.json
├── test05_mpc_sensor_shield_off.json
└── test06_mpc_sensor_shield_on.json

نحوه اجرا:

python3 interactive_stage9_final_comparison_fa.py \
  --dir final_eval_jsons \
  --out stage9_final_comparison_interactive_fa.html
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


EVAL_MODES = [
    {
        "test": 1,
        "kind": "RL",
        "filename": "test01_student_sensor_active_shield.json",
        "label": "آزمون ۱ — دانشجو / داده‌های حسگری / سپر فعال",
        "short": "آزمون ۱ | دانشجو | حسگری + سپر فعال",
        "model": "دانشجو",
        "observation": "حسگری",
        "shield": "سپر فعال",
    },
    {
        "test": 2,
        "kind": "RL",
        "filename": "test02_student_sensor_passive_shield.json",
        "label": "آزمون ۲ — دانشجو / داده‌های حسگری / بدون مداخله فعال سپر",
        "short": "آزمون ۲ | دانشجو | حسگری + بدون مداخله سپر",
        "model": "دانشجو",
        "observation": "حسگری",
        "shield": "بدون مداخله فعال",
    },
    {
        "test": 3,
        "kind": "RL",
        "filename": "test03_teacher_gt_active_shield.json",
        "label": "آزمون ۳ — معلم / داده‌های ایده‌آل / سپر فعال",
        "short": "آزمون ۳ | معلم | ایده‌آل + سپر فعال",
        "model": "معلم",
        "observation": "ایده‌آل",
        "shield": "سپر فعال",
    },
    {
        "test": 4,
        "kind": "RL",
        "filename": "test04_teacher_sensor_active_shield.json",
        "label": "آزمون ۴ — معلم / داده‌های حسگری / سپر فعال",
        "short": "آزمون ۴ | معلم | حسگری + سپر فعال",
        "model": "معلم",
        "observation": "حسگری",
        "shield": "سپر فعال",
    },
    {
        "test": 5,
        "kind": "MPC",
        "filename": "test05_mpc_sensor_shield_off.json",
        "label": "آزمون ۵ — MPC / داده‌های حسگری / سپر غیرفعال",
        "short": "آزمون ۵ | MPC | سپر غیرفعال",
        "model": "MPC",
        "observation": "حسگری",
        "shield": "غیرفعال",
    },
    {
        "test": 6,
        "kind": "MPC",
        "filename": "test06_mpc_sensor_shield_on.json",
        "label": "آزمون ۶ — MPC / داده‌های حسگری / سپر فعال",
        "short": "آزمون ۶ | MPC | سپر فعال",
        "model": "MPC",
        "observation": "حسگری",
        "shield": "سپر فعال",
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

    mean_return: Optional[float]
    mean_length: Optional[float]
    median_clearance: Optional[float]
    median_ttc: Optional[float]

    shield_rate: Optional[float]
    total_shield_activations: Optional[float]

    median_wall_ms: Optional[float]
    solver_fallback_rate: Optional[float]


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
                raise ValueError("ساختار براکت‌های JSON ناسازگار است و قابل ترمیم امن نیست.")

    closers = {"{": "}", "[": "]"}
    return text + "".join(closers[item] for item in reversed(stack))


def parse_constant(token: str) -> float:
    if token == "Infinity":
        return math.inf
    if token == "-Infinity":
        return -math.inf
    if token == "NaN":
        return math.nan
    return math.nan


def load_eval_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig").strip()

    if not text:
        raise ValueError(f"فایل JSON خالی است: {path}")

    def parse(candidate: str) -> Dict[str, Any]:
        result = json.loads(candidate, parse_constant=parse_constant)
        if not isinstance(result, dict):
            raise ValueError("مقدار سطح اول JSON باید object باشد.")
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


def safe_float(value: Any, default: float = math.nan, allow_inf: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if math.isfinite(number):
        return number

    if allow_inf and math.isinf(number):
        return number

    return default


def numeric_list(values: Any, allow_inf: bool = False) -> List[float]:
    if not isinstance(values, (list, tuple)):
        return []

    return [safe_float(v, default=math.nan, allow_inf=allow_inf) for v in values]


def finite_values(values: List[float]) -> List[float]:
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def finite_mean(values: List[float], default: float = math.nan) -> float:
    vals = finite_values(values)
    return float(statistics.mean(vals)) if vals else default


def finite_median(values: List[float], default: float = math.nan) -> float:
    vals = finite_values(values)
    return float(statistics.median(vals)) if vals else default


def finite_min(values: List[float], default: float = math.nan) -> float:
    vals = finite_values(values)
    return float(min(vals)) if vals else default


def clean_number(value: float) -> Optional[float]:
    try:
        value = float(value)
    except Exception:
        return None

    if math.isfinite(value):
        return value

    return None


def percent(count: int, total: int) -> float:
    return 100.0 * count / total if total > 0 else 0.0


def normalize_reason(reason: Any) -> str:
    text = str(reason).strip().lower()

    if "goal" in text or "success" in text or "reached" in text:
        return "goal"

    if "collision" in text or "crash" in text:
        return "collision"

    if (
        "timeout" in text
        or "time_out" in text
        or "time-limit" in text
        or "time limit" in text
        or "hard_cap" in text
    ):
        return "timeout"

    return text or "unknown"


def get_records(data: Dict[str, Any], path: Path) -> List[Dict[str, Any]]:
    records = data.get("episodes", [])

    if not isinstance(records, list) or not records:
        raise ValueError(f"در این فایل episode پیدا نشد: {path}")

    records = [rec for rec in records if isinstance(rec, dict)]

    if not records:
        raise ValueError(f"episodeها موجودند، اما قابل خواندن نیستند: {path}")

    return records


def summarize_run(mode: Dict[str, Any], folder: Path) -> RunSummary:
    path = folder / mode["filename"]
    data = load_eval_json(path)
    records = get_records(data, path)

    reasons: List[str] = []
    returns: List[float] = []
    lengths: List[float] = []
    clearances: List[float] = []
    ttcs_for_median: List[float] = []
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

        rewards = numeric_list(raw.get("rewards", []))
        v_cmds = numeric_list(raw.get("v_cmds", []))
        w_cmds = numeric_list(raw.get("w_cmds", []))
        goal_dists = numeric_list(raw.get("goal_dists", raw.get("dists", [])))
        min_ultras = numeric_list(raw.get("min_ultras", []))
        min_depths = numeric_list(raw.get("min_obs_depths", []))
        ttcs = numeric_list(raw.get("ttcs", []), allow_inf=True)

        return_sum = safe_float(summary.get("return_sum"), default=math.nan)
        if not math.isfinite(return_sum) and rewards:
            return_sum = sum(v for v in rewards if math.isfinite(v))
        returns.append(return_sum)

        length_steps = safe_float(summary.get("length_steps"), default=math.nan)
        if not math.isfinite(length_steps):
            length_steps = float(
                len(rewards)
                or len(v_cmds)
                or len(w_cmds)
                or len(goal_dists)
                or len(ttcs)
            )
        lengths.append(length_steps)

        min_obstacle = safe_float(summary.get("min_obstacle_distance"), default=math.nan)
        if not math.isfinite(min_obstacle):
            obstacle_candidates = finite_values(min_ultras) + finite_values(min_depths)
            min_obstacle = min(obstacle_candidates) if obstacle_candidates else math.nan
        clearances.append(min_obstacle)

        ttc_min = safe_float(summary.get("ttc_min"), default=math.nan, allow_inf=True)
        if math.isnan(ttc_min):
            finite_ttcs = finite_values(ttcs)
            if finite_ttcs:
                ttc_min = min(finite_ttcs)
            elif ttcs and all(math.isinf(v) for v in ttcs if isinstance(v, (int, float))):
                ttc_min = math.inf

        if math.isfinite(ttc_min):
            ttcs_for_median.append(ttc_min)
        elif math.isinf(ttc_min):
            ttcs_for_median.append(50.0)

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

        shield_rate = safe_float(summary.get("shield_intervention_rate"), default=math.nan)

        if not math.isfinite(shield_rate):
            shield_rate = shield_total / len(shield_states) if shield_states else math.nan

        shield_rates.append(shield_rate)

        duration_sec = safe_float(summary.get("duration_sec"), default=math.nan)
        durations.append(duration_sec)

        solver_fails = numeric_list(raw.get("solver_fails", []))
        solver_count = safe_float(summary.get("solver_fail_count"), default=math.nan)

        if not math.isfinite(solver_count):
            solver_count = float(sum(1 for v in solver_fails if math.isfinite(v) and v > 0.0))

        solver_counts.append(solver_count)

    n = len(reasons)

    goal_count = reasons.count("goal")
    timeout_count = reasons.count("timeout")
    collision_count = reasons.count("collision")
    other_count = max(0, n - goal_count - timeout_count - collision_count)

    wall_ms_values: List[float] = []
    for length, duration in zip(lengths, durations):
        if math.isfinite(length) and length > 0 and math.isfinite(duration):
            wall_ms_values.append(1000.0 * duration / length)

    total_steps = sum(v for v in lengths if math.isfinite(v))
    total_solver_fails = sum(v for v in solver_counts if math.isfinite(v))

    solver_fallback_rate = (
        100.0 * total_solver_fails / total_steps
        if total_steps > 0
        else math.nan
    )

    mean_shield_rate = finite_mean(shield_rates)
    if math.isfinite(mean_shield_rate):
        if mean_shield_rate <= 1.0:
            shield_rate_percent = 100.0 * mean_shield_rate
        else:
            shield_rate_percent = mean_shield_rate
    else:
        shield_rate_percent = math.nan

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

        mean_return=clean_number(finite_mean(returns)),
        mean_length=clean_number(finite_mean(lengths)),
        median_clearance=clean_number(finite_median(clearances)),
        median_ttc=clean_number(finite_median(ttcs_for_median)),

        shield_rate=clean_number(shield_rate_percent),
        total_shield_activations=clean_number(sum(v for v in shield_totals if math.isfinite(v))),

        median_wall_ms=clean_number(finite_median(wall_ms_values)),
        solver_fallback_rate=clean_number(solver_fallback_rate),
    )


def to_embeddable_js(data: Any) -> str:
    text = json.dumps(data, ensure_ascii=False, allow_nan=False)
    text = text.replace("</script", "<\\/script")
    text = text.replace("<!--", "<\\!--")
    return text


TEMPLATE_HTML = r"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>مقایسه نهایی مرحله نهم</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>

<style>
:root {
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
}

* {
  box-sizing: border-box;
}

html,
body {
  margin: 0;
  padding: 0;
  background:
    radial-gradient(circle at 50% 0%, rgba(125,157,156,0.12), transparent 34%),
    linear-gradient(145deg, #FAF6F0 0%, #F2E8DE 60%, #EDE0D2 100%);
  color: var(--ink);
  font-family: 'Vazirmatn', Tahoma, system-ui, -apple-system, 'Segoe UI', sans-serif;
  direction: rtl;
  text-align: right;
}

body {
  padding: 28px clamp(18px, 4vw, 54px) 44px;
}

header.top {
  margin-bottom: 18px;
}

.eyebrow {
  margin: 0 0 4px;
  font-size: 12px;
  font-weight: 800;
  color: var(--muted);
}

h1 {
  margin: 0;
  font-size: clamp(24px, 3vw, 38px);
  line-height: 1.2;
  font-weight: 800;
  color: var(--ink);
}

.subtitle {
  margin: 6px 0 0;
  color: var(--muted);
  font-size: 14px;
  line-height: 1.8;
}

.toolbar {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 8px;
  flex-wrap: wrap;
  margin: 18px 0;
}

.filter-btn {
  border: 1px solid var(--grid);
  background: rgba(255,255,255,0.65);
  color: var(--ink);
  border-radius: 999px;
  padding: 8px 14px;
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 800;
  cursor: pointer;
  box-shadow: var(--shadow);
}

.filter-btn.active {
  background: var(--teal);
  border-color: var(--teal);
  color: white;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(160px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.kpi-card {
  background: rgba(255,255,255,0.78);
  border: 1px solid var(--grid);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 14px 16px;
}

.kpi-card .label {
  color: var(--muted);
  font-size: 11.5px;
  font-weight: 800;
}

.kpi-card .value {
  margin-top: 6px;
  color: var(--ink);
  font-size: 22px;
  font-weight: 800;
  direction: ltr;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.kpi-card .sub {
  margin-top: 4px;
  color: var(--muted);
  font-size: 11.5px;
  line-height: 1.6;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 14px;
  align-items: stretch;
}

.panel {
  background: rgba(255,255,255,0.78);
  border: 1px solid var(--grid);
  border-radius: 18px;
  box-shadow: var(--shadow);
  padding: 12px 12px 8px;
  min-width: 0;
}

.panel-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  padding: 2px 4px 8px;
}

.panel-title {
  font-size: 14px;
  font-weight: 800;
  color: var(--ink);
}

.panel-sub {
  font-size: 11px;
  color: var(--muted);
}

.plot {
  width: 100%;
  height: 520px;
  direction: ltr;
}

.side-stack {
  display: grid;
  grid-template-rows: 1fr auto;
  gap: 14px;
}

.tradeoff-plot {
  width: 100%;
  height: 360px;
  direction: ltr;
}

.detail {
  background: rgba(255,255,255,0.78);
  border: 1px solid var(--grid);
  border-radius: 18px;
  box-shadow: var(--shadow);
  padding: 16px;
}

.detail .badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 5px 11px;
  background: #F1ECE4;
  color: var(--muted);
  font-size: 11px;
  font-weight: 800;
}

.detail h2 {
  margin: 12px 0 6px;
  font-size: 20px;
  line-height: 1.35;
  font-weight: 800;
}

.detail p {
  margin: 0 0 12px;
  color: var(--muted);
  font-size: 12.5px;
  line-height: 1.7;
}

.metric-table {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.metric {
  border: 1px solid var(--grid);
  background: rgba(250,246,240,0.72);
  border-radius: 12px;
  padding: 10px 12px;
}

.metric .k {
  color: var(--muted);
  font-size: 10.5px;
  font-weight: 800;
}

.metric .v {
  margin-top: 3px;
  color: var(--ink);
  font-size: 17px;
  font-weight: 800;
  direction: ltr;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.footnote {
  margin-top: 18px;
  text-align: center;
  color: var(--muted);
  font-size: 11.5px;
}

@media (max-width: 1000px) {
  .kpi-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .dashboard-grid {
    grid-template-columns: 1fr;
  }

  .plot {
    height: 460px;
  }
}
</style>
</head>

<body>

<header class="top">
  <p class="eyebrow">ناوبری ایمن اسکوتر با یادگیری عمیق — ارزیابی نهایی</p>
  <h1>مقایسه نهایی در سناریوی مرحله نهم</h1>
  <p class="subtitle">
    مقایسه شش آزمون نهایی در سناریوی مرحله نهم شامل سه مانع ایستا، چهار مانع متحرک و هدف متحرک سریع.
  </p>
</header>

<div class="toolbar">
  <button class="filter-btn active" data-filter="all">همه آزمون‌ها</button>
  <button class="filter-btn" data-filter="RL">سیاست‌های یادگیری</button>
  <button class="filter-btn" data-filter="MPC">مبناهای MPC</button>
  <button class="filter-btn" data-filter="Sensor">فقط آزمون‌های حسگری</button>
</div>

<section class="kpi-grid" id="kpiGrid"></section>

<section class="dashboard-grid">
  <section class="panel">
    <div class="panel-head">
      <span class="panel-title">نتایج نهایی آزمون‌ها</span>
      <span class="panel-sub">ترکیب رسیدن به هدف، اتمام زمان و برخورد</span>
    </div>
    <div id="outcomePlot" class="plot"></div>
  </section>

  <section class="side-stack">
    <section class="panel">
      <div class="panel-head">
        <span class="panel-title">مصالحه موفقیت و برخورد</span>
        <span class="panel-sub">ناحیه مطلوب: موفقیت بالا و برخورد کم</span>
      </div>
      <div id="tradeoffPlot" class="tradeoff-plot"></div>
    </section>

    <section class="detail" id="detailPanel"></section>
  </section>
</section>

<p class="footnote">
  برای مشاهده جزئیات هر آزمون، روی میله یا نقطه مربوط به آن کلیک کنید. با قرار دادن نشانگر روی نمودارها، مقدار دقیق نمایش داده می‌شود.
</p>

<script>
const RUNS = __RUNS_JSON__;

const COLORS = {
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
};

const state = {
  filter: "all",
  selectedTest: null
};

function fmtPct(v) {
  return Number.isFinite(v) ? `${v.toFixed(1)}%` : "—";
}

function fmtNum(v, digits = 2, suffix = "") {
  return Number.isFinite(v) ? `${v.toFixed(digits)}${suffix}` : "—";
}

function visibleRuns() {
  if (state.filter === "all") return RUNS.slice();
  if (state.filter === "RL") return RUNS.filter(r => r.kind === "RL");
  if (state.filter === "MPC") return RUNS.filter(r => r.kind === "MPC");
  if (state.filter === "Sensor") return RUNS.filter(r => r.observation === "حسگری");
  return RUNS.slice();
}

function commonLayout() {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: COLORS.bg,
    font: {
      family: "Vazirmatn, Tahoma, system-ui, sans-serif",
      color: COLORS.ink,
      size: 11
    },
    margin: { l: 230, r: 24, t: 12, b: 58 },
    xaxis: {
      gridcolor: COLORS.grid,
      zeroline: false,
      tickfont: { color: COLORS.ink }
    },
    yaxis: {
      gridcolor: COLORS.grid,
      zeroline: false,
      tickfont: { color: COLORS.ink }
    },
    legend: {
      orientation: "h",
      y: -0.16,
      x: 0.5,
      xanchor: "center",
      font: { size: 10 }
    },
    hoverlabel: {
      bgcolor: "#FFFFFF",
      bordercolor: COLORS.grid,
      font: {
        family: "Vazirmatn, Tahoma, system-ui, sans-serif",
        color: COLORS.ink
      }
    }
  };
}

const config = {
  responsive: true,
  displaylogo: false,
  displayModeBar: false
};

function renderKpis(runs) {
  const bestSuccess = runs.reduce((a, b) => b.success_rate > a.success_rate ? b : a, runs[0]);
  const safest = runs.reduce((a, b) => b.collision_rate < a.collision_rate ? b : a, runs[0]);

  const sensorRL = runs
    .filter(r => r.kind === "RL" && r.observation === "حسگری")
    .sort((a, b) => b.success_rate - a.success_rate)[0];

  const bestMPC = runs
    .filter(r => r.kind === "MPC")
    .sort((a, b) => b.success_rate - a.success_rate)[0];

  let deltaText = "—";
  let deltaSub = "نیازمند وجود داده RL و MPC";

  if (sensorRL && bestMPC) {
    deltaText = `+${(sensorRL.success_rate - bestMPC.success_rate).toFixed(1)} pp`;
    deltaSub = "برتری بهترین RL حسگری نسبت به بهترین MPC";
  }

  const items = [
    {
      label: "بیشترین نرخ موفقیت",
      value: bestSuccess ? fmtPct(bestSuccess.success_rate) : "—",
      sub: bestSuccess ? `آزمون ${bestSuccess.test} · ${bestSuccess.model}` : "—"
    },
    {
      label: "کمترین نرخ برخورد",
      value: safest ? fmtPct(safest.collision_rate) : "—",
      sub: safest ? `آزمون ${safest.test} · ${safest.model}` : "—"
    },
    {
      label: "بهترین RL حسگری نسبت به MPC",
      value: deltaText,
      sub: deltaSub
    },
    {
      label: "مجموعه انتخاب‌شده",
      value: `${runs.length} آزمون`,
      sub: state.filter === "all" ? "همه ارزیابی‌های نهایی" : state.filter
    }
  ];

  document.getElementById("kpiGrid").innerHTML = items.map(item => `
    <div class="kpi-card">
      <div class="label">${item.label}</div>
      <div class="value">${item.value}</div>
      <div class="sub">${item.sub}</div>
    </div>
  `).join("");
}

function renderOutcomes(runs) {
  const y = runs.map(r => r.short);
  const custom = runs.map(r => r.test);

  const selected = state.selectedTest;
  const opacities = runs.map(r => selected === null || r.test === selected ? 0.92 : 0.34);

  const traces = [
    {
      type: "bar",
      orientation: "h",
      name: "رسیدن به هدف",
      y,
      x: runs.map(r => r.success_rate),
      customdata: custom,
      marker: { color: COLORS.goal, opacity: opacities },
      hovertemplate: "%{y}<br>موفقیت: %{x:.1f}%<extra></extra>"
    },
    {
      type: "bar",
      orientation: "h",
      name: "اتمام زمان",
      y,
      x: runs.map(r => r.timeout_rate),
      customdata: custom,
      marker: { color: COLORS.timeout, opacity: opacities },
      hovertemplate: "%{y}<br>اتمام زمان: %{x:.1f}%<extra></extra>"
    },
    {
      type: "bar",
      orientation: "h",
      name: "برخورد",
      y,
      x: runs.map(r => r.collision_rate),
      customdata: custom,
      marker: { color: COLORS.collision, opacity: opacities },
      hovertemplate: "%{y}<br>برخورد: %{x:.1f}%<extra></extra>"
    },
    {
      type: "bar",
      orientation: "h",
      name: "سایر",
      y,
      x: runs.map(r => r.other_rate),
      customdata: custom,
      marker: { color: COLORS.other, opacity: opacities },
      hovertemplate: "%{y}<br>سایر: %{x:.1f}%<extra></extra>"
    }
  ];

  const layout = commonLayout();
  layout.barmode = "stack";
  layout.xaxis = Object.assign(layout.xaxis, {
    title: "درصد اپیزودها",
    range: [0, 100]
  });
  layout.yaxis = Object.assign(layout.yaxis, {
    autorange: "reversed"
  });

  Plotly.react("outcomePlot", traces, layout, config);

  const el = document.getElementById("outcomePlot");
  el.on("plotly_click", evt => {
    const pt = evt.points && evt.points[0];
    if (!pt) return;
    selectTest(pt.customdata);
  });
}

function renderTradeoff(runs) {
  const rl = runs.filter(r => r.kind === "RL");
  const mpc = runs.filter(r => r.kind === "MPC");

  function traceFor(group, name, color, symbol) {
    return {
      type: "scatter",
      mode: "markers+text",
      name,
      x: group.map(r => r.collision_rate),
      y: group.map(r => r.success_rate),
      text: group.map(r => `آزمون ${r.test}`),
      textposition: "top center",
      customdata: group.map(r => r.test),
      marker: {
        color,
        symbol,
        size: group.map(r => state.selectedTest === null || r.test === state.selectedTest ? 15 : 10),
        opacity: group.map(r => state.selectedTest === null || r.test === state.selectedTest ? 0.92 : 0.36),
        line: { color: "#FFFFFF", width: 1.5 }
      },
      hovertemplate:
        "%{text}<br>نرخ برخورد: %{x:.1f}%<br>نرخ موفقیت: %{y:.1f}%<extra>" + name + "</extra>"
    };
  }

  const layout = commonLayout();

  layout.margin = { l: 62, r: 24, t: 12, b: 52 };

  layout.xaxis = Object.assign(layout.xaxis, {
    title: "نرخ برخورد (%)",
    range: [-4, 78]
  });

  layout.yaxis = Object.assign(layout.yaxis, {
    title: "نرخ موفقیت (%)",
    range: [20, 104]
  });

  layout.shapes = [
    {
      type: "rect",
      xref: "x",
      yref: "y",
      x0: 0,
      x1: 18,
      y0: 85,
      y1: 100,
      fillcolor: "rgba(90,138,90,0.08)",
      line: { width: 0 }
    }
  ];

  layout.annotations = [
    {
      x: 2,
      y: 101,
      text: "ناحیه مطلوب",
      showarrow: false,
      xanchor: "left",
      yanchor: "top",
      font: { size: 10, color: COLORS.muted }
    }
  ];

  Plotly.react(
    "tradeoffPlot",
    [
      traceFor(rl, "سیاست یادگیری", COLORS.rl, "circle"),
      traceFor(mpc, "MPC", COLORS.mpc, "square")
    ],
    layout,
    config
  );

  const el = document.getElementById("tradeoffPlot");
  el.on("plotly_click", evt => {
    const pt = evt.points && evt.points[0];
    if (!pt) return;
    selectTest(pt.customdata);
  });
}

function renderDetail() {
  const runs = visibleRuns();
  let run = RUNS.find(r => r.test === state.selectedTest);

  if (!run || !runs.some(r => r.test === run.test)) {
    run = runs[0] || RUNS[0];
    state.selectedTest = run ? run.test : null;
  }

  if (!run) {
    document.getElementById("detailPanel").innerHTML = "<p>آزمونی انتخاب نشده است.</p>";
    return;
  }

  const kindColor = run.kind === "MPC" ? COLORS.collision : COLORS.goal;
  const kindLabel = run.kind === "MPC" ? "MPC" : "یادگیری";

  document.getElementById("detailPanel").innerHTML = `
    <span class="badge" style="background:${kindColor}22;color:${kindColor}">
      آزمون ${run.test} · ${kindLabel}
    </span>

    <h2>${run.model} — ${run.observation} — ${run.shield}</h2>
    <p>${run.label}</p>

    <div class="metric-table">
      <div class="metric">
        <div class="k">موفقیت</div>
        <div class="v">${fmtPct(run.success_rate)}</div>
      </div>
      <div class="metric">
        <div class="k">برخورد</div>
        <div class="v">${fmtPct(run.collision_rate)}</div>
      </div>
      <div class="metric">
        <div class="k">اتمام زمان</div>
        <div class="v">${fmtPct(run.timeout_rate)}</div>
      </div>
      <div class="metric">
        <div class="k">تعداد اپیزودها</div>
        <div class="v">${run.n}</div>
      </div>
      <div class="metric">
        <div class="k">میانه فاصله ایمن</div>
        <div class="v">${fmtNum(run.median_clearance, 2, " m")}</div>
      </div>
      <div class="metric">
        <div class="k">میانه TTC</div>
        <div class="v">${fmtNum(run.median_ttc, 2, " s")}</div>
      </div>
      <div class="metric">
        <div class="k">نرخ مداخله سپر</div>
        <div class="v">${fmtPct(run.shield_rate)}</div>
      </div>
      <div class="metric">
        <div class="k">زمان هر گام</div>
        <div class="v">${fmtNum(run.median_wall_ms, 0, " ms")}</div>
      </div>
    </div>
  `;
}

function selectTest(test) {
  if (state.selectedTest === test) {
    state.selectedTest = null;
  } else {
    state.selectedTest = test;
  }

  renderAll();
}

function renderAll() {
  const runs = visibleRuns();

  if (!runs.length) return;

  renderKpis(runs);
  renderOutcomes(runs);
  renderTradeoff(runs);
  renderDetail();
}

document.querySelectorAll(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));

    btn.classList.add("active");
    state.filter = btn.dataset.filter;
    state.selectedTest = null;

    renderAll();
  });
});

renderAll();
</script>

</body>
</html>
"""


def render_html(summaries: List[RunSummary]) -> str:
    runs = [asdict(summary) for summary in summaries]
    runs_json = to_embeddable_js(runs)
    return TEMPLATE_HTML.replace("__RUNS_JSON__", runs_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ساخت داشبورد HTML تعاملی فارسی برای مقایسه نهایی مرحله نهم."
    )

    parser.add_argument(
        "--dir",
        required=True,
        help="مسیر پوشه‌ای که شش فایل JSON تغییرنام‌داده‌شده داخل آن قرار دارند.",
    )

    parser.add_argument(
        "--out",
        default="stage9_final_comparison_interactive_fa.html",
        help="مسیر خروجی HTML.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.dir)

    if not folder.exists():
        raise FileNotFoundError(f"پوشه پیدا نشد: {folder}")

    missing = []

    for mode in EVAL_MODES:
        path = folder / mode["filename"]
        if not path.exists():
            missing.append(mode["filename"])

    if missing:
        print("\n[ERROR] فایل‌های زیر پیدا نشدند:\n")
        for name in missing:
            print(f"  - {name}")

        print("\nساختار مورد انتظار:\n")
        print(f"{folder}/")
        for mode in EVAL_MODES:
            print(f"├── {mode['filename']}")

        raise FileNotFoundError("برخی از فایل‌های JSON موردنیاز وجود ندارند.")

    summaries = [summarize_run(mode, folder) for mode in EVAL_MODES]

    html = render_html(summaries)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    print(f"[OK] تعداد {len(summaries)} فایل ارزیابی خوانده شد.")
    print(f"[OK] داشبورد تعاملی فارسی ساخته شد: {output.resolve()}")

    print("\nخلاصه نتایج:")
    for summary in summaries:
        print(
            f"  آزمون {summary.test} | "
            f"{summary.model:8s} | "
            f"{summary.observation:7s} | "
            f"{summary.shield:18s} | "
            f"موفقیت={summary.success_rate:5.1f}% | "
            f"برخورد={summary.collision_rate:5.1f}% | "
            f"اتمام زمان={summary.timeout_rate:5.1f}%"
        )


if __name__ == "__main__":
    main()
