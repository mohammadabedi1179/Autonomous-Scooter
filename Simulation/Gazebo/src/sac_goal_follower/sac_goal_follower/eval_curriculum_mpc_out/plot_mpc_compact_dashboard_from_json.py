#!/usr/bin/env python3
"""Create a compact MPC-only dashboard from ``eval_full.json``.

The dashboard intentionally excludes RL-oriented quantities such as reward,
return, shield intervention, policy correlation, and training-style jerk plots.
It focuses on four quantities that are directly useful for evaluating an MPC
run:

1. terminal outcomes;
2. clearance versus minimum time-to-collision;
3. actuator saturation / command usage;
4. wall-clock step time and IPOPT fallback health.

The script is standalone. It does not import ROS 2, CasADi, IPOPT,
Stable-Baselines3, the environment, or ``mpc_policy.py``.

Example
-------
python3 plot_mpc_compact_dashboard_from_json.py \
    --json ./eval_curriculum_mpc_out/eval_full.json \
    --out ./eval_curriculum_mpc_out/mpc_compact_dashboard.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np


@dataclass
class EpisodeMetrics:
    episode: int
    reason: str
    length_steps: int
    duration_sec: float
    min_obstacle_distance: float
    ttc_min: float
    v_cmd_mean: float
    w_cmd_mean_abs: float
    solver_fail_count: int
    solver_fail_rate: float
    v_cmds: np.ndarray
    w_cmds: np.ndarray


def _number(value: Any, default: float = np.nan, allow_inf: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if np.isfinite(number) or (allow_inf and np.isinf(number)):
        return number
    return default


def _integer(value: Any, default: int = 0) -> int:
    number = _number(value, default=np.nan)
    return int(number) if np.isfinite(number) else int(default)


def _array(values: Any, allow_inf: bool = False) -> np.ndarray:
    if not isinstance(values, (list, tuple)):
        return np.array([], dtype=float)
    output = [_number(item, default=np.nan, allow_inf=allow_inf) for item in values]
    return np.asarray(output, dtype=float)


def _finite_min(values: np.ndarray, default: float = np.nan) -> float:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else default


def _finite_mean(values: np.ndarray, default: float = np.nan) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else default


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
                raise ValueError("JSON contains mismatched brackets.")

    closers = {"{": "}", "[": "]"}
    return text + "".join(closers[item] for item in reversed(stack))


def load_eval_json(path: str) -> Tuple[Dict[str, Any], bool]:
    """Load the evaluation JSON, with conservative repair for pasted fragments."""
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"JSON file is empty: {path}")

    def parse(candidate: str) -> Dict[str, Any]:
        result = json.loads(candidate, parse_constant=lambda token: float(token))
        if not isinstance(result, dict):
            raise ValueError("Top-level JSON value must be an object.")
        return result

    try:
        return parse(text), False
    except (json.JSONDecodeError, ValueError):
        repaired = text
        if not repaired.lstrip().startswith("{"):
            repaired = "{" + repaired
        repaired = re.sub(r",\s*$", "", repaired)
        repaired = _append_missing_closers(repaired)
        try:
            return parse(repaired), True
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "The input is not a valid MPC eval_full.json file and could not "
                "be repaired safely."
            ) from exc


def build_episode_metrics(data: Dict[str, Any]) -> List[EpisodeMetrics]:
    records = data.get("episodes", [])
    if not isinstance(records, list) or not records:
        raise ValueError("No episode records were found under data['episodes'].")

    episodes: List[EpisodeMetrics] = []

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue

        summary = record.get("summary", record)
        raw = record.get("raw", {})
        summary = summary if isinstance(summary, dict) else {}
        raw = raw if isinstance(raw, dict) else {}

        v_cmds = _array(raw.get("v_cmds", []))
        w_cmds = _array(raw.get("w_cmds", []))
        rewards = _array(raw.get("rewards", []))
        dists = _array(raw.get("goal_dists", raw.get("dists", [])))
        ttcs = _array(raw.get("ttcs", []), allow_inf=True)
        min_ultras = _array(raw.get("min_ultras", []))
        min_depths = _array(raw.get("min_obs_depths", []))
        solver_fails = _array(raw.get("solver_fails", []))

        length_fallback = int(
            rewards.size or v_cmds.size or w_cmds.size or dists.size or solver_fails.size
        )
        length_steps = _integer(summary.get("length_steps"), length_fallback)

        obstacle_values = np.concatenate(
            [min_ultras[np.isfinite(min_ultras)], min_depths[np.isfinite(min_depths)]]
        )
        min_obstacle_fallback = (
            float(np.min(obstacle_values)) if obstacle_values.size else np.nan
        )

        finite_ttcs = ttcs[np.isfinite(ttcs)]
        if finite_ttcs.size:
            ttc_min_fallback = float(np.min(finite_ttcs))
        elif ttcs.size and np.all(np.isinf(ttcs)):
            ttc_min_fallback = np.inf
        else:
            ttc_min_fallback = np.nan

        summary_ttc = _number(summary.get("ttc_min"), default=np.nan, allow_inf=True)
        ttc_min = summary_ttc if not np.isnan(summary_ttc) else ttc_min_fallback

        finite_solver = solver_fails[np.isfinite(solver_fails)]
        solver_count_fallback = int(np.sum(finite_solver > 0.0)) if finite_solver.size else 0
        solver_rate_fallback = (
            float(np.mean(finite_solver > 0.0)) if finite_solver.size else 0.0
        )
        solver_fail_count = _integer(
            summary.get("solver_fail_count"), solver_count_fallback
        )
        solver_fail_rate = _number(
            summary.get("solver_fail_rate"), solver_rate_fallback
        )
        if not np.isfinite(solver_fail_rate) and length_steps > 0:
            solver_fail_rate = solver_fail_count / length_steps
        solver_fail_rate = float(np.clip(solver_fail_rate, 0.0, 1.0))

        episodes.append(
            EpisodeMetrics(
                episode=_integer(summary.get("episode"), index),
                reason=str(summary.get("reason", "unknown")).strip().lower() or "unknown",
                length_steps=max(0, length_steps),
                duration_sec=_number(summary.get("duration_sec")),
                min_obstacle_distance=_number(
                    summary.get("min_obstacle_distance"), min_obstacle_fallback
                ),
                ttc_min=ttc_min,
                v_cmd_mean=_number(summary.get("v_cmd_mean"), _finite_mean(v_cmds)),
                w_cmd_mean_abs=_number(
                    summary.get("w_cmd_mean_abs"), _finite_mean(np.abs(w_cmds))
                ),
                solver_fail_count=max(0, solver_fail_count),
                solver_fail_rate=solver_fail_rate,
                v_cmds=v_cmds,
                w_cmds=w_cmds,
            )
        )

    if not episodes:
        raise ValueError("Episode records exist, but none could be parsed.")
    return episodes


def _stage_name(data: Dict[str, Any]) -> str:
    stage = data.get("stage", {})
    if isinstance(stage, dict):
        return str(stage.get("name", "MPC evaluation"))
    if isinstance(stage, str) and stage.strip():
        return stage.strip()
    return "MPC evaluation"


def _controller_name(data: Dict[str, Any]) -> str:
    controller = str(data.get("controller", "MPC")).strip()
    return controller or "MPC"


def _percent_label(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * value:.1f}%"


def _value_label(value: float, unit: str = "", decimals: int = 2) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}{unit}"


def _add_kpi_card(
    fig: plt.Figure,
    x: float,
    width: float,
    label: str,
    value: str,
    facecolor: str,
    edgecolor: str,
    text_color: str,
    muted_color: str,
) -> None:
    y = 0.835
    height = 0.082
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        transform=fig.transFigure,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
    )
    fig.add_artist(patch)
    fig.text(
        x + width / 2,
        y + 0.053,
        value,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=text_color,
    )
    fig.text(
        x + width / 2,
        y + 0.022,
        label,
        ha="center",
        va="center",
        fontsize=7.5,
        color=muted_color,
    )


def save_dashboard(
    output_path: str,
    episodes: Sequence[EpisodeMetrics],
    stage_name: str,
    controller_name: str,
    dpi: int = 180,
    v_max: float = 1.0,
    w_max: float = 1.0,
    saturation_fraction: float = 0.95,
    near_stop_fraction: float = 0.05,
    safety_margin: float = 0.60,
    ttc_threshold: float = 2.0,
    ttc_plot_cap: float = 10.0,
    target_step_ms: float = 100.0,
) -> None:
    reasons = [episode.reason for episode in episodes]
    n_episodes = len(episodes)
    lengths = np.asarray([episode.length_steps for episode in episodes], dtype=float)
    durations = np.asarray([episode.duration_sec for episode in episodes], dtype=float)
    min_clearances = np.asarray(
        [episode.min_obstacle_distance for episode in episodes], dtype=float
    )
    min_ttcs = np.asarray([episode.ttc_min for episode in episodes], dtype=float)
    solver_counts = np.asarray(
        [episode.solver_fail_count for episode in episodes], dtype=float
    )

    valid_wall = np.isfinite(durations) & (lengths > 0)
    wall_ms_per_step = np.full(n_episodes, np.nan, dtype=float)
    wall_ms_per_step[valid_wall] = 1000.0 * durations[valid_wall] / lengths[valid_wall]

    total_steps = int(np.sum(lengths[np.isfinite(lengths)]))
    total_solver_fails = int(np.sum(solver_counts[np.isfinite(solver_counts)]))
    weighted_solver_rate = total_solver_fails / total_steps if total_steps > 0 else np.nan

    success_rate = reasons.count("goal") / n_episodes
    collision_rate = reasons.count("collision") / n_episodes
    median_clearance = (
        float(np.nanmedian(min_clearances))
        if np.any(np.isfinite(min_clearances))
        else np.nan
    )
    median_wall_ms = (
        float(np.nanmedian(wall_ms_per_step))
        if np.any(np.isfinite(wall_ms_per_step))
        else np.nan
    )

    all_v_parts = [episode.v_cmds[np.isfinite(episode.v_cmds)] for episode in episodes]
    all_w_parts = [episode.w_cmds[np.isfinite(episode.w_cmds)] for episode in episodes]
    all_v = np.concatenate([part for part in all_v_parts if part.size]) if any(
        part.size for part in all_v_parts
    ) else np.array([], dtype=float)
    all_w = np.concatenate([part for part in all_w_parts if part.size]) if any(
        part.size for part in all_w_parts
    ) else np.array([], dtype=float)

    v_limit = abs(v_max) * saturation_fraction
    w_limit = abs(w_max) * saturation_fraction
    near_stop_limit = abs(v_max) * near_stop_fraction

    actuation_labels = [
        f"Forward ≥ {saturation_fraction:.0%} vmax",
        "Reverse command",
        f"|Turn| ≥ {saturation_fraction:.0%} wmax",
        f"Near stop ≤ {near_stop_fraction:.0%} vmax",
    ]
    actuation_rates = np.asarray(
        [
            float(np.mean(all_v >= v_limit)) if all_v.size else np.nan,
            float(np.mean(all_v < 0.0)) if all_v.size else np.nan,
            float(np.mean(np.abs(all_w) >= w_limit)) if all_w.size else np.nan,
            float(np.mean(np.abs(all_v) <= near_stop_limit)) if all_v.size else np.nan,
        ],
        dtype=float,
    )

    BG_FIG = "#FAF8F5"
    BG_AX = "#F3F0EB"
    C_TEXT = "#49372A"
    C_MUTED = "#88796D"
    C_GRID = "#DDD5CC"
    C_REF = "#8B6E4E"
    C_BAR = "#C47B49"

    colors_map = {
        "goal": "#5F8F60",
        "collision": "#BF3B22",
        "timeout": "#C28727",
        "hard_cap": "#7B6C62",
        "unknown": "#9B8B7F",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    fig.patch.set_facecolor(BG_FIG)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.075, top=0.77, wspace=0.25, hspace=0.34)

    fig.suptitle(
        f"{controller_name} compact evaluation — {stage_name}",
        x=0.5,
        y=0.965,
        fontsize=15,
        fontweight="bold",
        color=C_TEXT,
    )
    fig.text(
        0.5,
        0.934,
        f"{n_episodes} episodes · {total_steps:,} evaluated control steps",
        ha="center",
        va="center",
        fontsize=8.5,
        color=C_MUTED,
    )

    card_left = 0.075
    card_gap = 0.012
    card_width = (0.90 - 4 * card_gap) / 5
    card_values = [
        ("Success rate", _percent_label(success_rate)),
        ("Collision rate", _percent_label(collision_rate)),
        ("Median clearance", _value_label(median_clearance, " m", 2)),
        ("Median wall time / step", _value_label(median_wall_ms, " ms", 0)),
        ("IPOPT fallback rate", _percent_label(weighted_solver_rate)),
    ]
    for i, (label, value) in enumerate(card_values):
        _add_kpi_card(
            fig,
            card_left + i * (card_width + card_gap),
            card_width,
            label,
            value,
            BG_AX,
            C_GRID,
            C_TEXT,
            C_MUTED,
        )

    for axis in axes.flat:
        axis.set_facecolor(BG_AX)
        axis.tick_params(colors=C_TEXT, labelsize=8)
        axis.xaxis.label.set_color(C_TEXT)
        axis.yaxis.label.set_color(C_TEXT)
        axis.title.set_color(C_TEXT)
        axis.grid(True, color=C_GRID, linewidth=0.55, alpha=0.65)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_edgecolor(C_GRID)

    # 1. Terminal outcomes
    axis = axes[0, 0]
    ordered = [key for key in ("goal", "collision", "timeout", "hard_cap") if key in reasons]
    ordered += sorted(set(reasons) - set(ordered))
    counts = np.asarray([reasons.count(key) for key in ordered], dtype=float)
    positions = np.arange(len(ordered))
    bar_colors = [colors_map.get(key, colors_map["unknown"]) for key in ordered]
    bars = axis.barh(positions, counts, color=bar_colors, alpha=0.9, height=0.58)
    axis.set_yticks(positions)
    axis.set_yticklabels([key.replace("_", " ") for key in ordered])
    axis.invert_yaxis()
    axis.set_xlabel("Episodes")
    axis.set_title("Terminal outcomes", fontsize=10.5, fontweight="bold")
    axis.set_xlim(0, max(counts.max() * 1.22, 1.0))
    for bar, count in zip(bars, counts):
        axis.text(
            bar.get_width() + max(counts.max() * 0.025, 0.25),
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)}  ({100.0 * count / n_episodes:.1f}%)",
            va="center",
            fontsize=8,
            color=C_TEXT,
        )

    # 2. Safety geometry
    axis = axes[0, 1]
    plot_ttc = np.full_like(min_ttcs, np.nan, dtype=float)
    finite_ttc = np.isfinite(min_ttcs)
    plot_ttc[finite_ttc] = np.minimum(min_ttcs[finite_ttc], ttc_plot_cap)
    inf_ttc = np.isinf(min_ttcs)
    plot_ttc[inf_ttc] = ttc_plot_cap

    for reason in ordered:
        reason_mask = np.asarray([episode.reason == reason for episode in episodes])
        base_mask = reason_mask & np.isfinite(min_clearances) & np.isfinite(plot_ttc)
        finite_mask = base_mask & ~inf_ttc
        infinite_mask = base_mask & inf_ttc
        color = colors_map.get(reason, colors_map["unknown"])
        if np.any(finite_mask):
            axis.scatter(
                min_clearances[finite_mask],
                plot_ttc[finite_mask],
                s=34,
                alpha=0.78,
                color=color,
                edgecolors="none",
                label=reason.replace("_", " "),
            )
        if np.any(infinite_mask):
            axis.scatter(
                min_clearances[infinite_mask],
                plot_ttc[infinite_mask],
                s=42,
                marker="^",
                alpha=0.85,
                color=color,
                edgecolors="none",
            )

    axis.axvline(safety_margin, color=C_REF, linestyle="--", linewidth=1.1, label=f"margin {safety_margin:g} m")
    axis.axhline(ttc_threshold, color="#6F86A1", linestyle="--", linewidth=1.1, label=f"TTC {ttc_threshold:g} s")
    axis.set_xlabel("Minimum obstacle clearance (m)")
    axis.set_ylabel(f"Minimum TTC (s; ≥{ttc_plot_cap:g} capped)")
    axis.set_title("Safety geometry by episode", fontsize=10.5, fontweight="bold")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(
        unique.values(),
        unique.keys(),
        fontsize=7,
        frameon=True,
        facecolor=BG_FIG,
        edgecolor=C_GRID,
        loc="upper right",
    )

    # 3. Actuation profile
    axis = axes[1, 0]
    y = np.arange(len(actuation_labels))
    valid_rates = np.nan_to_num(actuation_rates, nan=0.0)
    bars = axis.barh(y, 100.0 * valid_rates, color=C_BAR, alpha=0.88, height=0.56)
    axis.set_yticks(y)
    axis.set_yticklabels(actuation_labels)
    axis.invert_yaxis()
    axis.set_xlim(0, max(100.0, 100.0 * valid_rates.max() * 1.12 if valid_rates.size else 100.0))
    axis.set_xlabel("Share of all command steps (%)")
    axis.set_title("MPC actuation profile", fontsize=10.5, fontweight="bold")
    for bar, rate in zip(bars, actuation_rates):
        text = "n/a" if not np.isfinite(rate) else f"{100.0 * rate:.1f}%"
        axis.text(
            min(bar.get_width() + 1.3, axis.get_xlim()[1] - 5.5),
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            fontsize=8,
            color=C_TEXT,
        )
    axis.text(
        0.99,
        0.02,
        f"vmax={v_max:g} m/s · wmax={w_max:g} rad/s",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=C_MUTED,
    )

    # 4. Computational / solver health
    axis = axes[1, 1]
    episode_numbers = np.asarray([episode.episode for episode in episodes], dtype=float)
    for reason in ordered:
        mask = np.asarray([episode.reason == reason for episode in episodes])
        mask &= np.isfinite(wall_ms_per_step)
        if np.any(mask):
            axis.scatter(
                episode_numbers[mask],
                wall_ms_per_step[mask],
                s=28,
                alpha=0.78,
                color=colors_map.get(reason, colors_map["unknown"]),
                edgecolors="none",
            )
    if np.isfinite(target_step_ms) and target_step_ms > 0:
        axis.axhline(
            target_step_ms,
            color="#6F86A1",
            linestyle="--",
            linewidth=1.1,
            label=f"target {target_step_ms:g} ms",
        )
    if np.isfinite(median_wall_ms):
        axis.axhline(
            median_wall_ms,
            color=C_REF,
            linestyle=":",
            linewidth=1.2,
            label=f"median {median_wall_ms:.0f} ms",
        )
    axis.set_xlabel("Episode")
    axis.set_ylabel("Wall time per control step (ms)")
    axis.set_title("Computational and solver health", fontsize=10.5, fontweight="bold")
    axis.legend(
        fontsize=7,
        frameon=True,
        facecolor=BG_FIG,
        edgecolor=C_GRID,
        loc="upper left",
    )

    finite_wall_values = wall_ms_per_step[np.isfinite(wall_ms_per_step)]
    p95_wall = float(np.percentile(finite_wall_values, 95)) if finite_wall_values.size else np.nan
    annotation = (
        f"wall p95: {_value_label(p95_wall, ' ms', 0)}\n"
        f"fallbacks: {total_solver_fails:,}\n"
        f"fallback rate: {_percent_label(weighted_solver_rate)}"
    )
    axis.text(
        0.98,
        0.97,
        annotation,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=C_TEXT,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=BG_FIG,
            edgecolor=C_GRID,
            alpha=0.94,
        ),
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact MPC-only dashboard from eval_full.json."
    )
    parser.add_argument("--json", required=True, help="Path to MPC eval_full.json")
    parser.add_argument(
        "--out",
        default="mpc_compact_dashboard.png",
        help="Output image path",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--v-max", type=float, default=1.0)
    parser.add_argument("--w-max", type=float, default=1.0)
    parser.add_argument(
        "--saturation-fraction",
        type=float,
        default=0.95,
        help="Fraction of command limit treated as saturation",
    )
    parser.add_argument(
        "--near-stop-fraction",
        type=float,
        default=0.05,
        help="Fraction of vmax treated as near zero speed",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.60,
        help="Reference MPC obstacle margin drawn on the safety panel",
    )
    parser.add_argument(
        "--ttc-threshold",
        type=float,
        default=2.0,
        help="Reference TTC threshold drawn on the safety panel",
    )
    parser.add_argument(
        "--ttc-plot-cap",
        type=float,
        default=10.0,
        help="Upper plotting cap for infinite/large minimum TTC values",
    )
    parser.add_argument(
        "--target-step-ms",
        type=float,
        default=100.0,
        help="Reference control-step wall time; set <=0 to hide",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.v_max <= 0 or args.w_max <= 0:
        raise ValueError("--v-max and --w-max must be positive.")
    if not 0 < args.saturation_fraction <= 1:
        raise ValueError("--saturation-fraction must be in (0, 1].")
    if not 0 <= args.near_stop_fraction < 1:
        raise ValueError("--near-stop-fraction must be in [0, 1).")
    if args.ttc_plot_cap <= 0:
        raise ValueError("--ttc-plot-cap must be positive.")

    data, repaired = load_eval_json(args.json)
    episodes = build_episode_metrics(data)
    save_dashboard(
        args.out,
        episodes,
        stage_name=_stage_name(data),
        controller_name=_controller_name(data),
        dpi=args.dpi,
        v_max=args.v_max,
        w_max=args.w_max,
        saturation_fraction=args.saturation_fraction,
        near_stop_fraction=args.near_stop_fraction,
        safety_margin=args.safety_margin,
        ttc_threshold=args.ttc_threshold,
        ttc_plot_cap=args.ttc_plot_cap,
        target_step_ms=args.target_step_ms,
    )

    if repaired:
        print("[WARN] Input JSON was repaired in memory before plotting.")
    print(f"[OK] Loaded {len(episodes)} MPC episode(s)")
    print(f"[OK] Saved compact dashboard: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
