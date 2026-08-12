#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-9 final evaluation comparison dashboard.

This script expects exactly 6 JSON files in one folder:

final_eval_jsons/
├── test01_student_sensor_active_shield.json
├── test02_student_sensor_passive_shield.json
├── test03_teacher_gt_active_shield.json
├── test04_teacher_sensor_active_shield.json
├── test05_mpc_sensor_shield_off.json
└── test06_mpc_sensor_shield_on.json

Outputs:
1) slide14_stage9_final_comparison.png
2) slide14_stage9_final_comparison.csv

Run:
python3 plot_stage9_final_eval_comparison.py \
  --dir final_eval_jsons \
  --out slide14_stage9_final_comparison.png \
  --csv slide14_stage9_final_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EVAL_MODES = [
    {
        "test": 1,
        "kind": "RL",
        "filename": "test01_student_sensor_active_shield.json",
        "label": "T1 Student | Sensor | Active shield",
        "short": "T1\nStudent\nSensor + Active",
        "model": "Student",
        "observation": "Sensor",
        "shield": "Active",
    },
    {
        "test": 2,
        "kind": "RL",
        "filename": "test02_student_sensor_passive_shield.json",
        "label": "T2 Student | Sensor | Passive shield",
        "short": "T2\nStudent\nSensor + Passive",
        "model": "Student",
        "observation": "Sensor",
        "shield": "Passive",
    },
    {
        "test": 3,
        "kind": "RL",
        "filename": "test03_teacher_gt_active_shield.json",
        "label": "T3 Teacher | Ideal GT | Active shield",
        "short": "T3\nTeacher\nGT + Active",
        "model": "Teacher",
        "observation": "GT",
        "shield": "Active",
    },
    {
        "test": 4,
        "kind": "RL",
        "filename": "test04_teacher_sensor_active_shield.json",
        "label": "T4 Teacher | Sensor | Active shield",
        "short": "T4\nTeacher\nSensor + Active",
        "model": "Teacher",
        "observation": "Sensor",
        "shield": "Active",
    },
    {
        "test": 5,
        "kind": "MPC",
        "filename": "test05_mpc_sensor_shield_off.json",
        "label": "T5 MPC | Sensor | Shield OFF",
        "short": "T5\nMPC\nShield OFF",
        "model": "MPC",
        "observation": "Sensor",
        "shield": "Off",
    },
    {
        "test": 6,
        "kind": "MPC",
        "filename": "test06_mpc_sensor_shield_on.json",
        "label": "T6 MPC | Sensor | Shield ON",
        "short": "T6\nMPC\nShield ON",
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

    mean_return: float
    mean_length: float
    median_clearance: float
    median_ttc: float

    shield_rate: float
    total_shield_activations: float

    median_wall_ms: float
    solver_fallback_rate: float


def _append_missing_closers(text: str) -> str:
    stack: List[str] = []
    in_string = False
    escaped = False
    matching = {"}": "{", "]": "["}

    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and stack[-1] == matching[ch]:
                stack.pop()
            else:
                raise ValueError("JSON contains mismatched brackets.")

    closers = {"{": "}", "[": "]"}
    return text + "".join(closers[item] for item in reversed(stack))


def load_eval_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig").strip()

    if not text:
        raise ValueError(f"Empty JSON file: {path}")

    def parse(candidate: str) -> Dict[str, Any]:
        result = json.loads(candidate, parse_constant=lambda token: float(token))
        if not isinstance(result, dict):
            raise ValueError("Top-level JSON must be an object.")
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

    out = []
    for value in values:
        out.append(safe_float(value, default=np.nan, allow_inf=allow_inf))

    return np.asarray(out, dtype=float)


def finite_mean(values: List[float] | np.ndarray, default: float = np.nan) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else default


def finite_median(values: List[float] | np.ndarray, default: float = np.nan) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else default


def percent(count: int, n: int) -> float:
    return 100.0 * count / n if n > 0 else 0.0


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


def get_records(data: Dict[str, Any], path: Path) -> List[Dict[str, Any]]:
    records = data.get("episodes", [])

    if not isinstance(records, list) or not records:
        raise ValueError(f"No episodes found in: {path}")

    clean_records = [r for r in records if isinstance(r, dict)]

    if not clean_records:
        raise ValueError(f"Episode records exist but none are readable in: {path}")

    return clean_records


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

    for index, record in enumerate(records, start=1):
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

        # Return, mainly meaningful for RL
        return_sum = safe_float(summary.get("return_sum"), default=np.nan)
        if not np.isfinite(return_sum) and rewards.size:
            return_sum = float(np.nansum(rewards))
        returns.append(return_sum)

        # Episode length
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

        # Minimum obstacle clearance
        min_obs = safe_float(summary.get("min_obstacle_distance"), default=np.nan)
        if not np.isfinite(min_obs):
            candidates = np.concatenate([
                min_ultras[np.isfinite(min_ultras)],
                min_depths[np.isfinite(min_depths)],
            ])
            min_obs = float(np.min(candidates)) if candidates.size else np.nan
        clearances.append(min_obs)

        # Minimum TTC
        ttc_min = safe_float(summary.get("ttc_min"), default=np.nan, allow_inf=True)
        if np.isnan(ttc_min):
            finite_ttcs = ttc_arr[np.isfinite(ttc_arr)]
            if finite_ttcs.size:
                ttc_min = float(np.min(finite_ttcs))
            elif ttc_arr.size and np.all(np.isinf(ttc_arr)):
                ttc_min = np.inf
        ttcs.append(ttc_min)

        # Shield statistics
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
        shield_rates.append(float(shield_rate))

        # MPC computational statistics
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

        mean_return=finite_mean(returns),
        mean_length=finite_mean(lengths),
        median_clearance=finite_median(clearances),
        median_ttc=finite_median(finite_ttc_for_median),

        shield_rate=100.0 * finite_mean(shield_rates),
        total_shield_activations=float(np.nansum(shield_totals)),

        median_wall_ms=finite_median(wall_ms),
        solver_fallback_rate=solver_fallback_rate,
    )


def write_csv(path: Path, summaries: List[RunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "test",
        "filename",
        "kind",
        "model",
        "observation",
        "shield",
        "episodes",
        "success_rate_percent",
        "timeout_rate_percent",
        "collision_rate_percent",
        "other_rate_percent",
        "goal_count",
        "timeout_count",
        "collision_count",
        "other_count",
        "mean_return",
        "mean_length_steps",
        "median_clearance_m",
        "median_ttc_s",
        "shield_rate_percent",
        "total_shield_activations",
        "median_wall_ms",
        "solver_fallback_rate_percent",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for s in summaries:
            writer.writerow({
                "test": s.test,
                "filename": s.filename,
                "kind": s.kind,
                "model": s.model,
                "observation": s.observation,
                "shield": s.shield,
                "episodes": s.n,
                "success_rate_percent": f"{s.success_rate:.3f}",
                "timeout_rate_percent": f"{s.timeout_rate:.3f}",
                "collision_rate_percent": f"{s.collision_rate:.3f}",
                "other_rate_percent": f"{s.other_rate:.3f}",
                "goal_count": s.goal_count,
                "timeout_count": s.timeout_count,
                "collision_count": s.collision_count,
                "other_count": s.other_count,
                "mean_return": f"{s.mean_return:.3f}" if np.isfinite(s.mean_return) else "",
                "mean_length_steps": f"{s.mean_length:.3f}" if np.isfinite(s.mean_length) else "",
                "median_clearance_m": f"{s.median_clearance:.3f}" if np.isfinite(s.median_clearance) else "",
                "median_ttc_s": f"{s.median_ttc:.3f}" if np.isfinite(s.median_ttc) else "",
                "shield_rate_percent": f"{s.shield_rate:.3f}" if np.isfinite(s.shield_rate) else "",
                "total_shield_activations": f"{s.total_shield_activations:.0f}" if np.isfinite(s.total_shield_activations) else "",
                "median_wall_ms": f"{s.median_wall_ms:.3f}" if np.isfinite(s.median_wall_ms) else "",
                "solver_fallback_rate_percent": f"{s.solver_fallback_rate:.3f}" if np.isfinite(s.solver_fallback_rate) else "",
            })


def style_axis(axis, bg_ax: str, c_text: str, c_grid: str) -> None:
    axis.set_facecolor(bg_ax)
    axis.tick_params(colors=c_text, labelsize=9)
    axis.xaxis.label.set_color(c_text)
    axis.yaxis.label.set_color(c_text)
    axis.title.set_color(c_text)
    axis.grid(True, color=c_grid, linewidth=0.65, alpha=0.65)
    axis.set_axisbelow(True)

    for spine in axis.spines.values():
        spine.set_edgecolor(c_grid)


def add_card(axis, x: float, y: float, w: float, h: float, title: str, main: str, sub: str) -> None:
    rect = plt.Rectangle(
        (x, y),
        w,
        h,
        transform=axis.transAxes,
        facecolor="#FAF8F5",
        edgecolor="#DDD5CC",
        linewidth=1.0,
        zorder=1,
    )
    axis.add_patch(rect)

    axis.text(
        x + 0.035,
        y + h - 0.18,
        title,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=8.4,
        color="#88796D",
        fontweight="bold",
    )

    axis.text(
        x + 0.035,
        y + h - 0.40,
        main,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=10.2,
        color="#49372A",
        fontweight="bold",
    )

    axis.text(
        x + 0.035,
        y + 0.18,
        sub,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        color="#88796D",
    )


def save_dashboard(out_path: Path, summaries: List[RunSummary], dpi: int = 220) -> None:
    BG_FIG = "#FAF8F5"
    BG_AX = "#F3F0EB"
    C_TEXT = "#49372A"
    C_MUTED = "#88796D"
    C_GRID = "#DDD5CC"

    C_GOAL = "#5F8F60"
    C_TIMEOUT = "#C28727"
    C_COLLISION = "#BF3B22"
    C_OTHER = "#9B8B7F"

    C_RL = "#5F8F60"
    C_MPC = "#BF3B22"

    y = np.arange(len(summaries))

    labels = [s.short for s in summaries]
    goal = np.asarray([s.success_rate for s in summaries], dtype=float)
    timeout = np.asarray([s.timeout_rate for s in summaries], dtype=float)
    collision = np.asarray([s.collision_rate for s in summaries], dtype=float)
    other = np.asarray([s.other_rate for s in summaries], dtype=float)

    fig = plt.figure(figsize=(16, 9), facecolor=BG_FIG)

    gs = fig.add_gridspec(
        3,
        5,
        left=0.065,
        right=0.975,
        bottom=0.075,
        top=0.84,
        wspace=0.42,
        hspace=0.42,
    )

    ax_outcome = fig.add_subplot(gs[:, :3])
    ax_tradeoff = fig.add_subplot(gs[0:2, 3:])
    ax_cards = fig.add_subplot(gs[2, 3:])

    fig.suptitle(
        "Stage-9 final evaluation: proposed policy vs MPC",
        fontsize=19,
        fontweight="bold",
        color=C_TEXT,
        y=0.955,
    )

    fig.text(
        0.5,
        0.917,
        "Six final tests under the same Stage-9 scenario: 3 static + 4 moving obstacles, fast moving goal",
        ha="center",
        va="center",
        fontsize=10.5,
        color=C_MUTED,
    )

    # ========================================================
    # 1) Terminal outcomes
    # ========================================================
    style_axis(ax_outcome, BG_AX, C_TEXT, C_GRID)

    ax_outcome.barh(y, goal, color=C_GOAL, alpha=0.92, label="Goal / Success")
    ax_outcome.barh(y, timeout, left=goal, color=C_TIMEOUT, alpha=0.92, label="Timeout")
    ax_outcome.barh(y, collision, left=goal + timeout, color=C_COLLISION, alpha=0.92, label="Collision")
    ax_outcome.barh(y, other, left=goal + timeout + collision, color=C_OTHER, alpha=0.92, label="Other")

    ax_outcome.set_yticks(y)
    ax_outcome.set_yticklabels(labels, fontsize=8.7)
    ax_outcome.invert_yaxis()
    ax_outcome.set_xlim(0, 112)
    ax_outcome.set_xlabel("Episodes (%)")
    ax_outcome.set_title("Terminal outcomes", fontsize=13, fontweight="bold", pad=10)

    for i, s in enumerate(summaries):
        parts = [
            (goal[i], 0.0, f"{goal[i]:.0f}%"),
            (timeout[i], goal[i], f"{timeout[i]:.0f}%"),
            (collision[i], goal[i] + timeout[i], f"{collision[i]:.0f}%"),
            (other[i], goal[i] + timeout[i] + collision[i], f"{other[i]:.0f}%"),
        ]

        for width, left, txt in parts:
            if width >= 6.0:
                ax_outcome.text(
                    left + width / 2,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

        ax_outcome.text(
            101.5,
            i,
            f"{s.goal_count}/{s.n}",
            ha="left",
            va="center",
            fontsize=8,
            color=C_MUTED,
        )

    ax_outcome.text(
        101.5,
        -0.72,
        "success\ncount",
        ha="left",
        va="bottom",
        fontsize=8,
        color=C_MUTED,
    )

    ax_outcome.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=True,
        facecolor=BG_FIG,
        edgecolor=C_GRID,
        fontsize=8,
    )

    # ========================================================
    # 2) Success-collision trade-off
    # ========================================================
    style_axis(ax_tradeoff, BG_AX, C_TEXT, C_GRID)

    ax_tradeoff.set_title(
        "Success–collision trade-off",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax_tradeoff.set_xlabel("Collision rate (%)")
    ax_tradeoff.set_ylabel("Success rate (%)")
    ax_tradeoff.set_xlim(-4, 78)
    ax_tradeoff.set_ylim(20, 104)

    ax_tradeoff.axhspan(85, 100, xmin=0.0, xmax=0.28, color=C_GOAL, alpha=0.08)
    ax_tradeoff.text(
        2,
        101,
        "desired region",
        ha="left",
        va="top",
        fontsize=8.5,
        color=C_MUTED,
    )

    for s in summaries:
        if s.kind == "MPC":
            marker = "s"
            color = C_MPC
            size = 95
        else:
            marker = "o"
            color = C_RL
            size = 82

        ax_tradeoff.scatter(
            s.collision_rate,
            s.success_rate,
            s=size,
            color=color,
            marker=marker,
            alpha=0.88,
            edgecolors=BG_FIG,
            linewidths=1.2,
        )

        ax_tradeoff.annotate(
            f"T{s.test}",
            xy=(s.collision_rate, s.success_rate),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9,
            color=C_TEXT,
            fontweight="bold",
        )

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=C_RL,
            markeredgecolor=BG_FIG,
            markersize=8,
            label="RL policy",
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="none",
            markerfacecolor=C_MPC,
            markeredgecolor=BG_FIG,
            markersize=8,
            label="MPC",
        ),
    ]

    ax_tradeoff.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        facecolor=BG_FIG,
        edgecolor=C_GRID,
        fontsize=8,
    )

    # ========================================================
    # 3) Takeaway cards
    # ========================================================
    ax_cards.set_facecolor(BG_AX)
    ax_cards.axis("off")

    rl_runs = [s for s in summaries if s.kind == "RL"]
    mpc_runs = [s for s in summaries if s.kind == "MPC"]

    best_success = max(summaries, key=lambda s: s.success_rate)
    safest = min(summaries, key=lambda s: s.collision_rate)
    best_sensor_rl = max(
        [s for s in rl_runs if s.observation.lower() == "sensor"],
        key=lambda s: s.success_rate,
    ) if rl_runs else None
    best_mpc = max(mpc_runs, key=lambda s: s.success_rate) if mpc_runs else None

    card_items = [
        (
            "Highest success",
            f"T{best_success.test} — {best_success.model}",
            f"{best_success.success_rate:.0f}% success · {best_success.collision_rate:.0f}% collision",
        ),
        (
            "Lowest collision",
            f"T{safest.test} — {safest.model}",
            f"{safest.collision_rate:.0f}% collision · {safest.success_rate:.0f}% success",
        ),
        (
            "Best sensor RL vs MPC",
            f"T{best_sensor_rl.test} vs T{best_mpc.test}" if best_sensor_rl and best_mpc else "n/a",
            (
                f"{best_sensor_rl.success_rate:.0f}% vs {best_mpc.success_rate:.0f}% success"
                if best_sensor_rl and best_mpc
                else "n/a"
            ),
        ),
    ]

    for i, (title, main, sub) in enumerate(card_items):
        add_card(
            ax_cards,
            x=0.02 + i * 0.325,
            y=0.18,
            w=0.30,
            h=0.66,
            title=title,
            main=main,
            sub=sub,
        )

    # ========================================================
    # Footer interpretation
    # ========================================================
    fig.text(
        0.065,
        0.027,
        "Interpretation: the preferred region is high success with low collision. "
        "Sensor-based tests show the real perception challenge, while MPC provides the model-based baseline.",
        ha="left",
        va="center",
        fontsize=8.2,
        color=C_MUTED,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Stage-9 final comparison dashboard from exactly six evaluation JSON files."
    )

    parser.add_argument(
        "--dir",
        required=True,
        help="Folder containing the six renamed JSON files.",
    )

    parser.add_argument(
        "--out",
        default="slide14_stage9_final_comparison.png",
        help="Output PNG path.",
    )

    parser.add_argument(
        "--csv",
        default="slide14_stage9_final_comparison.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    folder = Path(args.dir)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    missing = []
    for mode in EVAL_MODES:
        path = folder / mode["filename"]
        if not path.exists():
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

    summaries: List[RunSummary] = []

    for mode in EVAL_MODES:
        summary = summarize_run(mode, folder)
        summaries.append(summary)

    save_dashboard(
        out_path=Path(args.out),
        summaries=summaries,
        dpi=args.dpi,
    )

    if args.csv:
        write_csv(Path(args.csv), summaries)

    print(f"[OK] Loaded {len(summaries)} final evaluation files.")
    print(f"[OK] Saved dashboard: {Path(args.out).resolve()}")

    if args.csv:
        print(f"[OK] Saved CSV: {Path(args.csv).resolve()}")

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
