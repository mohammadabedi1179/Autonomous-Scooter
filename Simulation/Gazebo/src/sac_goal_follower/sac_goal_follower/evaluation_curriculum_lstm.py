#!/usr/bin/env python3
"""
Evaluate a trained Curriculum LSTM-SAC agent (Stage 6: full difficulty).

Configures the environment for the final curriculum stage:
  - 10 obstacles (7 walking + 3 standing)
  - Fast moving goal (0.5 m/s)
  - Extended time limit (120s)
  - Success radius 2.0m

Publishes obstacle config and goal mover config before evaluation starts,
then runs N episodes and produces analysis report + plots + JSON.

Usage:
  python3 evaluation_curriculum_lstm.py \
    --model ./models_lstm_curriculum/lstm_sac_curriculum_last.zip \
    --episodes 100 \
    --outdir ./eval_curriculum_out \
    --stage 6

  # Or evaluate at a specific stage:
  python3 evaluation_curriculum_lstm.py \
    --model ./models_lstm_curriculum/lstm_sac_stage3_ep800.zip \
    --episodes 100 \
    --stage 3
"""

import os
import json
import time
import argparse
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from goal_env_lstm import GoalFollowerLSTMEnv
from lstm_sac_policy import create_lstm_sac_policy


# ============================================================================
# CURRICULUM STAGE CONFIGS (must match training script)
# ============================================================================

STAGE_CONFIGS = {
    1: {"name": "1: 1s+4w, static goal",    "num_standing": 1, "num_walking": 4, "sector_half_angle_deg": 15.0, "goal_enabled": False, "goal_speed": 0.0, "time_limit": 80.0,  "success_radius": 1.5},
    2: {"name": "2: 2s+4w, static goal",    "num_standing": 2, "num_walking": 4, "sector_half_angle_deg": 30.0, "goal_enabled": False, "goal_speed": 0.0, "time_limit": 80.0,  "success_radius": 1.5},
    3: {"name": "3: 2s+6w, static goal",    "num_standing": 2, "num_walking": 6, "sector_half_angle_deg": 30.0, "goal_enabled": False, "goal_speed": 0.0, "time_limit": 80.0,  "success_radius": 1.5},
    4: {"name": "4: 3s+7w, static goal",    "num_standing": 3, "num_walking": 7, "sector_half_angle_deg": 45.0, "goal_enabled": False, "goal_speed": 0.0, "time_limit": 90.0,  "success_radius": 1.5},
    5: {"name": "5: 3s+7w, slow goal 0.2",  "num_standing": 3, "num_walking": 7, "sector_half_angle_deg": 45.0, "goal_enabled": True,  "goal_speed": 0.2, "time_limit": 100.0, "success_radius": 2.0},
    6: {"name": "6: 3s+7w, fast goal 0.5",  "num_standing": 3, "num_walking": 7, "sector_half_angle_deg": 60.0, "goal_enabled": True,  "goal_speed": 0.5, "time_limit": 120.0, "success_radius": 2.0},
    7: {"name": "7: 3s+7w, fast goal 0.5",  "num_standing": 3, "num_walking": 7, "sector_half_angle_deg": 60.0, "goal_enabled": True,  "goal_speed": 0.5, "time_limit": 80.0, "success_radius": 1.5},
}


# ============================================================================
# HELPERS
# ============================================================================

def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _mean_ci95(xs: np.ndarray) -> Tuple[float, float]:
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return np.nan, np.nan
    m = float(np.mean(xs))
    if xs.size == 1:
        return m, np.nan
    s = float(np.std(xs, ddof=1))
    hw = 1.96 * s / np.sqrt(xs.size)
    return m, hw


@dataclass
class EpisodeResult:
    episode: int
    seed: int

    return_sum: float
    length_steps: int
    duration_sec: float

    terminated: bool
    truncated: bool
    reason: str

    robot_start: Optional[Tuple[float, float]]
    goal_start: Optional[Tuple[float, float]]
    robot_final: Optional[Tuple[float, float]]
    goal_final: Optional[Tuple[float, float]]
    min_dist: Optional[float]

    v_cmd_mean: float
    v_cmd_max: float
    w_cmd_mean_abs: float

    ttc_min: float
    ttc_mean: float
    ttc_frac_below_2s: float

    min_ultra_min: float
    min_obs_depth_min: float

    progress_total: float
    dist_start: float
    dist_final: float

    num_valid_obstacles_mean: float
    shield_intervention_rate: float
    shield_count_danger: int
    shield_count_critical: int

    # Jerk metrics
    v_jerk_mean: float
    w_jerk_mean: float

    # TTC capped mean (inf -> 50)
    ttc_mean_capped: float

    # Min obstacle distance (min of depth and ultrasonic)
    min_obstacle_distance: float

    # Moving goal metrics
    goal_dist_mean: float
    goal_dist_min: float
    goal_rel_speed_mean: float

    robot_traj: Optional[List[List[float]]] = None
    goal_traj: Optional[List[List[float]]] = None
    gt_obstacles_snapshots: Optional[List[Dict]] = None


def _extract_obs_features(obs_dict):
    try:
        fixed = obs_dict['fixed']
        goal_dist = float(fixed[0]) * 15.0
        goal_bearing = float(fixed[1]) * np.pi
        goal_rel_vx = float(fixed[2]) * 2.0
        goal_rel_vy = float(fixed[3]) * 2.0
        return goal_dist, goal_bearing, goal_rel_vx, goal_rel_vy
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def _count_valid_obstacles(obs_dict):
    try:
        depths = obs_dict['obstacles'][:, 2]
        return int(np.sum(depths < 0.95))
    except Exception:
        return 0


# ============================================================================
# RUN ONE EPISODE
# ============================================================================

def run_one_episode(
    env: Monitor,
    model: SAC,
    ep_i: int,
    seed: int,
    deterministic: bool = True,
    max_steps_hard_cap: int = 5000,
    save_traj: bool = True,
) -> Tuple[EpisodeResult, Dict[str, Any]]:

    obs, _ = env.reset(seed=seed)
    t0 = time.time()

    rews, dists, bearings = [], [], []
    v_cmds, w_cmds, ttcs = [], [], []
    min_ultras, min_obs_depths = [], []
    valid_obstacles, shield_active = [], []
    goal_dists, goal_rel_speeds = [], []

    dist_start, _, _, _ = _extract_obs_features(obs)
    prev_dist = dist_start

    terminated, truncated, reason = False, False, ""
    robot_start = goal_start = robot_final = goal_final = min_dist = None
    robot_traj_last = None

    # For saving goal trajectory and obstacle snapshots
    goal_path_steps = []
    obs_snapshots = []

    steps = 0
    while True:
        steps += 1
        if steps > max_steps_hard_cap:
            truncated = True
            reason = "hard_cap"
            break

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, term, trunc, info = env.step(action)

        rews.append(float(reward))

        d, b, grvx, grvy = _extract_obs_features(obs)
        dists.append(d)
        bearings.append(b)
        goal_dists.append(d)
        goal_rel_speeds.append(np.sqrt(grvx**2 + grvy**2) if np.isfinite(grvx) else np.nan)

        valid_obs = _count_valid_obstacles(obs)
        valid_obstacles.append(valid_obs)

        v_cmds.append(_safe_float(info.get("v_cmd", np.nan)))
        w_cmds.append(_safe_float(info.get("w_cmd", np.nan)))
        ttcs.append(_safe_float(info.get("ttc", np.inf), default=np.inf))
        min_ultras.append(_safe_float(info.get("min_ultrasonic_distance", np.nan)))
        min_obs_depths.append(_safe_float(info.get("min_obstacle_depth", np.nan)))

        shield_level = info.get("shield_level", "none")
        shield_active.append(shield_level)  # store "none", "danger", or "critical"

        # Track goal position from GT
        gp_gt = info.get("step_goal_pose_gt") or info.get("step_goal_pose")
        if gp_gt is not None and len(gp_gt) == 2:
            goal_path_steps.append([float(gp_gt[0]), float(gp_gt[1])])

        # Snapshot obstacle world positions every 10 steps
        if save_traj and steps % 10 == 0:
            gt_obs = info.get("step_gt_obstacles_world")
            if gt_obs and isinstance(gt_obs, dict):
                snapshot = {}
                for k, v in gt_obs.items():
                    snapshot[str(k)] = [float(v[0]), float(v[1])] if v else None
                obs_snapshots.append(snapshot)

        if term or trunc:
            terminated = bool(term)
            truncated = bool(trunc)
            reason = str(info.get("reason", ""))
            robot_start = info.get("robot_start")
            goal_start = info.get("goal_start")
            robot_final = info.get("robot_final")
            goal_final = info.get("goal_final")
            min_dist = info.get("min_dist")
            robot_traj_last = info.get("robot_traj")
            break

        if np.isfinite(prev_dist) and np.isfinite(d):
            prev_dist = d

    duration = time.time() - t0
    ret = float(np.sum(rews)) if rews else 0.0
    length = len(rews)

    dists_arr = np.array(dists, dtype=float)
    v_arr = np.array(v_cmds, dtype=float)
    w_arr = np.array(w_cmds, dtype=float)
    ttc_arr = np.array(ttcs, dtype=float)
    min_ultra_arr = np.array(min_ultras, dtype=float)
    min_obs_arr = np.array(min_obs_depths, dtype=float)
    valid_obs_arr = np.array(valid_obstacles, dtype=float)
    shield_arr = np.array(shield_active, dtype=object)  # stores "none"/"danger"/"critical"
    shield_bool = shield_arr != "none"
    shield_count_danger = int(np.sum(shield_arr == "danger"))
    shield_count_critical = int(np.sum(shield_arr == "critical"))

    # Jerk: discrete derivative of commands
    v_jerk = np.abs(np.diff(v_arr)) if v_arr.size > 1 else np.array([np.nan])
    w_jerk = np.abs(np.diff(w_arr)) if w_arr.size > 1 else np.array([np.nan])

    # TTC capped: replace inf with 50 before computing mean
    ttc_capped = np.where(np.isinf(ttc_arr) | (ttc_arr > 50), 50.0, ttc_arr)

    # Min obstacle distance: min of depth sensor and ultrasonic
    min_obs_distance = np.nan
    if min_ultra_arr.size > 0 and min_obs_arr.size > 0:
        combined = np.nanmin([np.nanmin(min_ultra_arr), np.nanmin(min_obs_arr)])
        min_obs_distance = float(combined)
    elif min_ultra_arr.size > 0:
        min_obs_distance = float(np.nanmin(min_ultra_arr))
    elif min_obs_arr.size > 0:
        min_obs_distance = float(np.nanmin(min_obs_arr))
    goal_dists_arr = np.array(goal_dists, dtype=float)
    goal_rel_arr = np.array(goal_rel_speeds, dtype=float)

    dist_final = float(dists_arr[-1]) if dists_arr.size else np.nan
    progress_total = float(dist_start - dist_final) if (np.isfinite(dist_start) and np.isfinite(dist_final)) else np.nan

    ttc_below = np.isfinite(ttc_arr) & (ttc_arr < 2.0)
    ttc_frac = float(np.mean(ttc_below)) if ttc_arr.size else np.nan

    ep = EpisodeResult(
        episode=ep_i,
        seed=seed,
        return_sum=ret,
        length_steps=length,
        duration_sec=duration,
        terminated=terminated,
        truncated=truncated,
        reason=reason,
        robot_start=tuple(robot_start) if isinstance(robot_start, (list, tuple)) else None,
        goal_start=tuple(goal_start) if isinstance(goal_start, (list, tuple)) else None,
        robot_final=tuple(robot_final) if isinstance(robot_final, (list, tuple)) else None,
        goal_final=tuple(goal_final) if isinstance(goal_final, (list, tuple)) else None,
        min_dist=_safe_float(min_dist) if min_dist is not None else None,
        v_cmd_mean=float(np.nanmean(v_arr)) if v_arr.size else np.nan,
        v_cmd_max=float(np.nanmax(v_arr)) if v_arr.size else np.nan,
        w_cmd_mean_abs=float(np.nanmean(np.abs(w_arr))) if w_arr.size else np.nan,
        ttc_min=float(np.nanmin(ttc_arr)) if ttc_arr.size else np.nan,
        ttc_mean=float(np.nanmean(ttc_arr)) if ttc_arr.size else np.nan,
        ttc_frac_below_2s=ttc_frac,
        min_ultra_min=float(np.nanmin(min_ultra_arr)) if min_ultra_arr.size else np.nan,
        min_obs_depth_min=float(np.nanmin(min_obs_arr)) if min_obs_arr.size else np.nan,
        progress_total=progress_total,
        dist_start=dist_start,
        dist_final=dist_final,
        num_valid_obstacles_mean=float(np.mean(valid_obs_arr)) if valid_obs_arr.size else np.nan,
        shield_intervention_rate=float(np.mean(shield_bool)) if shield_bool.size else 0.0,
        shield_count_danger=shield_count_danger,
        shield_count_critical=shield_count_critical,
        v_jerk_mean=float(np.nanmean(v_jerk)) if v_jerk.size else np.nan,
        w_jerk_mean=float(np.nanmean(w_jerk)) if w_jerk.size else np.nan,
        ttc_mean_capped=float(np.mean(ttc_capped)) if ttc_capped.size else np.nan,
        min_obstacle_distance=min_obs_distance,
        goal_dist_mean=float(np.nanmean(goal_dists_arr)) if goal_dists_arr.size else np.nan,
        goal_dist_min=float(np.nanmin(goal_dists_arr)) if goal_dists_arr.size else np.nan,
        goal_rel_speed_mean=float(np.nanmean(goal_rel_arr)) if goal_rel_arr.size else np.nan,
        robot_traj=robot_traj_last if save_traj else None,
        goal_traj=goal_path_steps if save_traj else None,
        gt_obstacles_snapshots=obs_snapshots if save_traj else None,
    )

    raw = {
        "rewards": [float(x) for x in rews],
        "dists": [float(x) for x in dists],
        "bearings": [float(x) for x in bearings],
        "v_cmds": [float(x) for x in v_cmds],
        "w_cmds": [float(x) for x in w_cmds],
        "ttcs": [float(x) for x in ttcs],
        "min_ultras": [float(x) for x in min_ultras],
        "min_obs_depths": [float(x) for x in min_obs_depths],
        "valid_obstacles": [int(x) for x in valid_obstacles],
        "shield_active": [str(x) for x in shield_active],
        "goal_dists": [float(x) for x in goal_dists],
        "goal_rel_speeds": [float(x) for x in goal_rel_speeds],
    }

    return ep, raw


# ============================================================================
# PLOTS
# ============================================================================

def save_plots(outdir: str, episodes: List[EpisodeResult], stage_name: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    R  = np.array([e.return_sum for e in episodes])
    L  = np.array([e.length_steps for e in episodes])
    TTCmin = np.array([e.ttc_min for e in episodes])
    TTCmean_capped = np.array([e.ttc_mean_capped for e in episodes])
    Vmean = np.array([e.v_cmd_mean for e in episodes])
    Wmean = np.array([e.w_cmd_mean_abs for e in episodes])
    ShieldRate = np.array([e.shield_intervention_rate for e in episodes])
    ShieldDanger   = np.array([e.shield_count_danger for e in episodes])
    ShieldCritical = np.array([e.shield_count_critical for e in episodes])
    GoalDistMean   = np.array([e.goal_dist_mean for e in episodes])
    MinDist        = np.array([_safe_float(e.min_dist) for e in episodes])
    VJerkMean      = np.array([e.v_jerk_mean for e in episodes])
    WJerkMean      = np.array([e.w_jerk_mean for e in episodes])
    MinObsDist     = np.array([_safe_float(e.min_obstacle_distance) for e in episodes])
    ShieldTotal    = ShieldDanger + ShieldCritical

    # ── Warm Linen palette ───────────────────────────────────────────────────
    BG_FIG    = "#FAF8F5"   # figure background
    BG_AX     = "#F3F0EB"   # axes background
    C_TEXT    = "#4A3728"   # primary text / spines / ticks
    C_MUTED   = "#8C7B6E"   # secondary labels
    C_GRID    = "#DDD5CC"   # grid lines
    C_MEAN    = "#8B6E4E"   # mean reference line

    # Termination colours — warm but readable on linen
    colors_map = {"goal": "#5A8A5A", "timeout": "#B87820", "collision": "#B83820"}
    ep_colors = [colors_map.get(e.reason, "#A09080") for e in episodes]

    # Shield type colours
    C_CRITICAL = "#B83820"
    C_DANGER   = "#B87820"
    C_CAUTION  = "#C4A840"

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle(f"Evaluation: {stage_name}", fontsize=14, fontweight="bold", color=C_TEXT)

    for ax in axes.flat:
        ax.set_facecolor(BG_AX)
        ax.tick_params(colors=C_TEXT, labelsize=8)
        ax.xaxis.label.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT)
        ax.title.set_color(C_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)

    legend_kw = dict(fontsize=7, facecolor=BG_FIG, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── 1. Episode Returns ───────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.bar(range(len(R)), R, color=ep_colors, alpha=0.85, width=1.0)
    ax.axhline(np.nanmean(R), color=C_MEAN, ls="--", lw=1.2, alpha=0.9,
               label=f"mean={np.nanmean(R):.1f}")
    ax.set_xlabel("Episode"); ax.set_ylabel("Return")
    ax.set_title("Episode Returns")
    ax.grid(True, color=C_GRID, alpha=0.6, linewidth=0.5)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_map["goal"],      label="goal"),
                       Patch(facecolor=colors_map["timeout"],   label="timeout"),
                       Patch(facecolor=colors_map["collision"], label="collision")]
    ax.legend(handles=legend_elements, loc="upper right", **legend_kw)

    # ── 2. Episode Lengths ───────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.bar(range(len(L)), L, color=ep_colors, alpha=0.85, width=1.0)
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps")
    ax.set_title("Episode Lengths")
    ax.grid(True, color=C_GRID, alpha=0.6, linewidth=0.5)

    # ── 3. Outcome Donut ─────────────────────────────────────────────────────
    ax = axes[0, 2]
    reasons = [e.reason for e in episodes]
    ordered_keys = [k for k in ["goal", "timeout", "collision"] if k in set(reasons)]
    other_keys   = [k for k in set(reasons) if k not in ordered_keys]
    all_keys = ordered_keys + other_keys
    counts = [reasons.count(k) for k in all_keys]
    pie_colors = [colors_map.get(k, "#A09080") for k in all_keys]
    wedges, texts, autotexts = ax.pie(
        counts,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(width=0.55, edgecolor=BG_FIG, linewidth=1.5),
    )
    for t in texts:
        t.set_color(C_TEXT); t.set_fontsize(8)
    for t in autotexts:
        t.set_color("#FAF8F5"); t.set_fontsize(8); t.set_fontweight("bold")
    pie_labels = [f"{k} ({c})" for k, c in zip(all_keys, counts)]
    ax.legend(wedges, pie_labels, loc="lower center", fontsize=7,
              labelcolor=C_TEXT, facecolor=BG_FIG, edgecolor=C_GRID,
              bbox_to_anchor=(0.5, -0.08), ncol=len(all_keys))
    ax.set_title("Outcomes")

    # ── 4. TTC Distribution ──────────────────────────────────────────────────
    ax = axes[1, 0]
    finite_min  = TTCmin[np.isfinite(TTCmin) & (TTCmin < 50)]
    finite_mean = TTCmean_capped[np.isfinite(TTCmean_capped)]
    bins = np.linspace(0, min(50, max(finite_min.max() if finite_min.size else 50,
                                      finite_mean.max() if finite_mean.size else 50) + 2), 25)
    ax.hist(finite_min,  bins=bins, color="#C47840", alpha=0.75, label="min TTC")
    ax.hist(finite_mean, bins=bins, color="#6A8AB0", alpha=0.65, label="mean TTC (inf→50)")
    ax.axvline(2.0, color=C_MEAN, ls="--", lw=1.2, alpha=0.9, label="TTC=2s")
    ax.set_xlabel("TTC (s)"); ax.set_ylabel("Episodes")
    ax.set_title("TTC Distribution (min & mean)")
    ax.legend(**legend_kw)
    ax.grid(True, color=C_GRID, alpha=0.6, linewidth=0.5)

    # ── 5. Min Obstacle Distance histogram (no collision line) ───────────────
    ax = axes[1, 1]
    valid_mod = MinObsDist[np.isfinite(MinObsDist)]
    if valid_mod.size:
        ax.hist(valid_mod, bins=20, color="#C47840", alpha=0.8, edgecolor=BG_FIG, linewidth=0.4)
    ax.set_xlabel("Min Obstacle Distance (m)"); ax.set_ylabel("Episodes")
    ax.set_title("Min Obstacle Approach (depth+ultrasonic)")
    ax.grid(True, color=C_GRID, alpha=0.6, linewidth=0.5)

    # ── 6. Shield Interventions — critical / danger / caution stacked ────────
    ax = axes[1, 2]
    ep_idx = np.arange(len(episodes))
    # Derive caution counts: total shield steps minus danger and critical
    total_steps   = np.array([e.length_steps for e in episodes])
    shield_total_steps = np.round(total_steps * ShieldRate).astype(int)
    ShieldCaution = np.clip(shield_total_steps - ShieldDanger - ShieldCritical, 0, None)

    ax.bar(ep_idx, ShieldCritical, color=C_CRITICAL, alpha=0.9, width=1.0, label="critical")
    ax.bar(ep_idx, ShieldDanger,   color=C_DANGER,   alpha=0.9, width=1.0,
           bottom=ShieldCritical, label="danger")
    ax.bar(ep_idx, ShieldCaution,  color=C_CAUTION,  alpha=0.9, width=1.0,
           bottom=ShieldCritical + ShieldDanger, label="caution")
    ax.set_xlabel("Episode"); ax.set_ylabel("Shield Activations (steps)")
    ax.set_title("Safety Shield Interventions by Type")
    ax.legend(**legend_kw)
    ax.grid(True, color=C_GRID, alpha=0.6, linewidth=0.5)

    # ── 7. Mean velocity commands scatter ────────────────────────────────────
    ax = axes[2, 0]
    for reason, col in colors_map.items():
        mask = np.array([e.reason == reason for e in episodes])
        vv = Vmean[mask]; ww = Wmean[mask]
        if vv.size:
            ax.scatter(vv, ww, c=col, label=reason, alpha=0.75, s=30, edgecolors="none")
    ax.set_xlabel("Mean v_cmd (m/s)"); ax.set_ylabel("Mean |w_cmd| (rad/s)")
    ax.set_title("Mean Velocity Commands per Episode")
    ax.legend(**legend_kw)
    ax.grid(True, color=C_GRID, alpha=0.6, linewidth=0.5)

    # ── 8. Jerk profile scatter ───────────────────────────────────────────────
    ax = axes[2, 1]
    for reason, col in colors_map.items():
        mask = np.array([e.reason == reason for e in episodes])
        vj = VJerkMean[mask]; wj = WJerkMean[mask]
        if vj.size:
            ax.scatter(vj, wj, c=col, label=reason, alpha=0.75, s=30, edgecolors="none")
    ax.set_xlabel("Mean v_jerk (m/s²)"); ax.set_ylabel("Mean w_jerk (rad/s²)")
    ax.set_title("Jerk Profile per Episode")
    ax.legend(**legend_kw)
    ax.grid(True, color=C_GRID, alpha=0.6, linewidth=0.5)

    # ── 9. Radar: |corr| with reward ─────────────────────────────────────────
    ax = axes[2, 2]
    ax.set_visible(False)

    # Shrink the polar axes inward so labels don't bleed into adjacent plots
    pos = axes[2, 2].get_position()
    pad = 0.06  # fraction of figure to inset on each side
    radar_ax = fig.add_axes(
        [pos.x0 + pad, pos.y0 + pad, pos.width - 2 * pad, pos.height - 2 * pad],
        polar=True,
        facecolor=BG_AX,
    )
    radar_ax.tick_params(colors=C_TEXT)

    radar_labels = ["Ep Length", "Dist to Goal", "mean v_cmd", "mean Jerk", "mean TTC", "Shield #"]
    GoalDistAll = np.array([e.goal_dist_mean for e in episodes])
    MeanJerk    = (VJerkMean + WJerkMean) / 2.0
    ShieldN     = ShieldTotal.astype(float)

    def _corr(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3: return 0.0
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    corrs = np.abs(np.array([
        _corr(R, L.astype(float)),
        _corr(R, GoalDistAll),
        _corr(R, Vmean),
        _corr(R, MeanJerk),
        _corr(R, TTCmean_capped),
        _corr(R, ShieldN),
    ]))

    N_radar = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, N_radar, endpoint=False).tolist()
    corrs_plot = np.concatenate([corrs, [corrs[0]]])
    angles_plot = angles + [angles[0]]

    radar_ax.plot(angles_plot, corrs_plot, color="#8B6E4E", linewidth=2)
    radar_ax.fill(angles_plot, corrs_plot, color="#C4A882", alpha=0.30)
    radar_ax.set_xticks(angles)
    radar_ax.set_xticklabels(radar_labels, size=7, color=C_TEXT)
    radar_ax.set_ylim(0, 1)
    radar_ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    radar_ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=6, color=C_MUTED)
    radar_ax.set_title("|Corr| with Reward", color=C_TEXT, fontsize=9, pad=10)
    radar_ax.spines["polar"].set_color(C_GRID)
    radar_ax.grid(color=C_GRID, alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(outdir, "eval_dashboard.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOTS] Saved eval_dashboard.png")


# ============================================================================
# REPORT
# ============================================================================

def write_report(outdir: str, episodes: List[EpisodeResult], stage_cfg: Dict) -> None:
    N = len(episodes)
    reasons = [e.reason for e in episodes]
    success = sum(1 for r in reasons if r == "goal")
    collision = sum(1 for r in reasons if r == "collision")
    timeout = sum(1 for r in reasons if r == "timeout")

    R = np.array([e.return_sum for e in episodes])
    L = np.array([e.length_steps for e in episodes])
    TTCmin = np.array([e.ttc_min for e in episodes])
    TTCfrac = np.array([e.ttc_frac_below_2s for e in episodes])
    Vmean = np.array([e.v_cmd_mean for e in episodes])
    Wabs = np.array([e.w_cmd_mean_abs for e in episodes])
    ValidObs = np.array([e.num_valid_obstacles_mean for e in episodes])
    ShieldRate = np.array([e.shield_intervention_rate for e in episodes])
    GoalDistMean = np.array([e.goal_dist_mean for e in episodes])
    GoalDistMin = np.array([e.goal_dist_min for e in episodes])
    GoalRelSpeed = np.array([e.goal_rel_speed_mean for e in episodes])

    def _corr(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            return np.nan
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    lines = []
    lines.append(f"# Curriculum LSTM-SAC Evaluation Report\n\n")
    lines.append(f"**Stage**: {stage_cfg['name']}\n\n")
    lines.append(f"**Config**: {stage_cfg['num_walking']} walking + {stage_cfg['num_standing']} standing obstacles, "
                 f"goal {'moving at ' + str(stage_cfg['goal_speed']) + ' m/s' if stage_cfg['goal_enabled'] else 'static'}\n\n")
    lines.append(f"**Time limit**: {stage_cfg['time_limit']}s | **Success radius**: {stage_cfg['success_radius']}m\n\n")
    lines.append(f"**Episodes**: {N}\n\n")

    lines.append("## Outcome Summary\n\n")
    lines.append(f"| Outcome | Count | Percentage |\n")
    lines.append(f"|---------|------:|-----------:|\n")
    lines.append(f"| Success | {success} | {success/N*100:.1f}% |\n")
    lines.append(f"| Collision | {collision} | {collision/N*100:.1f}% |\n")
    lines.append(f"| Timeout | {timeout} | {timeout/N*100:.1f}% |\n\n")

    r_mean, r_ci = _mean_ci95(R)
    l_mean, l_ci = _mean_ci95(L)
    lines.append("## Performance\n\n")
    lines.append(f"- **Mean return**: {r_mean:.2f} ± {r_ci:.2f} (95% CI)\n")
    lines.append(f"- **Median return**: {float(np.nanmedian(R)):.2f}\n")
    lines.append(f"- **Mean episode length**: {l_mean:.1f} ± {l_ci:.1f} steps\n")
    lines.append(f"- **Return range**: [{float(np.nanmin(R)):.2f}, {float(np.nanmax(R)):.2f}]\n\n")

    lines.append("## Safety\n\n")
    lines.append(f"- **Collision rate**: {collision/N*100:.1f}%\n")
    lines.append(f"- **Mean min TTC**: {float(np.nanmean(TTCmin)):.2f}s\n")
    lines.append(f"- **Episodes with TTC < 1s**: {int(np.sum(TTCmin < 1.0))}\n")
    lines.append(f"- **Mean TTC<2s fraction**: {float(np.nanmean(TTCfrac)):.3f}\n")
    lines.append(f"- **Mean shield intervention rate**: {float(np.nanmean(ShieldRate))*100:.1f}%\n")
    lines.append(f"- **Episodes with >10% shield**: {int(np.sum(ShieldRate > 0.1))}\n")
    ShieldDanger   = np.array([e.shield_count_danger for e in episodes])
    ShieldCritical = np.array([e.shield_count_critical for e in episodes])
    total_danger   = int(np.sum(ShieldDanger))
    total_critical = int(np.sum(ShieldCritical))
    total_shields  = total_danger + total_critical
    lines.append(f"- **Total shield activations**: {total_shields} "
                 f"(danger={total_danger}, critical={total_critical})\n")
    lines.append(f"- **Mean danger activations/ep**: {float(np.mean(ShieldDanger)):.1f}\n")
    lines.append(f"- **Mean critical activations/ep**: {float(np.mean(ShieldCritical)):.1f}\n\n")

    lines.append("## Moving Goal Tracking\n\n")
    lines.append(f"- **Goal speed**: {stage_cfg['goal_speed']} m/s\n")
    lines.append(f"- **Mean goal distance during episode**: {float(np.nanmean(GoalDistMean)):.2f}m\n")
    lines.append(f"- **Mean closest approach to goal**: {float(np.nanmean(GoalDistMin)):.2f}m\n")
    lines.append(f"- **Mean relative speed to goal**: {float(np.nanmean(GoalRelSpeed)):.3f} m/s\n")
    if success > 0:
        succ_eps = [e for e in episodes if e.reason == "goal"]
        lines.append(f"- **Successful episodes mean steps**: {np.mean([e.length_steps for e in succ_eps]):.1f}\n")
        lines.append(f"- **Successful episodes mean min dist**: {np.mean([_safe_float(e.min_dist) for e in succ_eps]):.3f}m\n")
    lines.append("\n")

    lines.append("## Control Behavior\n\n")
    lines.append(f"- **Mean v_cmd**: {float(np.nanmean(Vmean)):.3f} m/s\n")
    lines.append(f"- **Mean |w_cmd|**: {float(np.nanmean(Wabs)):.3f} rad/s\n")
    lines.append(f"- **Mean valid obstacles**: {float(np.nanmean(ValidObs)):.2f}\n\n")

    lines.append("## Correlations\n\n")
    lines.append(f"- Return ↔ Speed: {_corr(R, Vmean):.3f}\n")
    lines.append(f"- Return ↔ Min TTC: {_corr(R, TTCmin):.3f}\n")
    lines.append(f"- Return ↔ Length: {_corr(R, L.astype(float)):.3f}\n")
    lines.append(f"- Return ↔ Valid Obs: {_corr(R, ValidObs):.3f}\n")
    lines.append(f"- Return ↔ Goal Dist: {_corr(R, GoalDistMean):.3f}\n\n")

    lines.append("## Per-Episode Table\n\n")
    lines.append("| Ep | Reason | Return | Steps | MinDist | MinTTC | v_mean | |w| | Shield% | Danger | Critical | GoalDist |\n")
    lines.append("|---:|:------:|-------:|------:|--------:|-------:|-------:|----:|--------:|-------:|---------:|---------:|\n")
    for e in episodes:
        lines.append(
            f"| {e.episode:>3d} | {e.reason or '-':^8s} | {e.return_sum:>7.1f} | {e.length_steps:>5d} "
            f"| {_safe_float(e.min_dist):>7.2f} "
            f"| {e.ttc_min:>6.2f} "
            f"| {e.v_cmd_mean:>6.3f} | {e.w_cmd_mean_abs:>5.3f} "
            f"| {e.shield_intervention_rate*100:>7.1f} "
            f"| {e.shield_count_danger:>6d} | {e.shield_count_critical:>8d} "
            f"| {e.goal_dist_mean:>8.2f} |\n"
        )

    # Breakdown by outcome
    for label, filter_reason in [("Successful", "goal"), ("Collision", "collision"), ("Timeout", "timeout")]:
        subset = [e for e in episodes if e.reason == filter_reason]
        if not subset:
            continue
        lines.append(f"\n### {label} Episodes (n={len(subset)})\n\n")
        lines.append(f"- Mean return: {np.mean([e.return_sum for e in subset]):.2f}\n")
        lines.append(f"- Mean steps: {np.mean([e.length_steps for e in subset]):.1f}\n")
        lines.append(f"- Mean v_cmd: {np.mean([e.v_cmd_mean for e in subset]):.3f}\n")
        lines.append(f"- Mean shield rate: {np.mean([e.shield_intervention_rate for e in subset])*100:.1f}%\n")
        lines.append(f"- Mean goal dist: {np.mean([e.goal_dist_mean for e in subset]):.2f}m\n")
        if filter_reason == "timeout":
            lines.append(f"- Mean progress: {np.mean([e.progress_total for e in subset]):.2f}m\n")

    report_path = os.path.join(outdir, "report.md")
    with open(report_path, "w") as f:
        f.writelines(lines)
    print(f"[REPORT] Saved to {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Evaluate Curriculum LSTM-SAC")
    ap.add_argument("--model", type=str, required=True, help="Path to .zip model")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--outdir", type=str, default="./eval_curriculum_out")
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=12345)
    ap.add_argument("--stage", type=int, default=6, help="Curriculum stage (1-6)")
    ap.add_argument("--lstm_hidden", type=int, default=64)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--net_arch", type=str, default="256,256")
    ap.add_argument("--max_obstacles", type=int, default=10)
    ap.add_argument("--save_traj", type=int, default=1, help="Save trajectories (0/1)")
    args = ap.parse_args()

    if args.stage not in STAGE_CONFIGS:
        raise ValueError(f"Invalid stage {args.stage}. Choose 1-6.")

    stage_cfg = STAGE_CONFIGS[args.stage]
    os.makedirs(args.outdir, exist_ok=True)

    print(f"{'='*70}")
    print(f"CURRICULUM LSTM-SAC EVALUATION")
    print(f"{'='*70}")
    print(f"  Model:      {args.model}")
    print(f"  Stage:      {stage_cfg['name']}")
    print(f"  Obstacles:  {stage_cfg['num_walking']}w + {stage_cfg['num_standing']}s + {stage_cfg['sector_half_angle_deg']}deg")
    print(f"  Goal:       {'moving at ' + str(stage_cfg['goal_speed']) + ' m/s' if stage_cfg['goal_enabled'] else 'static'}")
    print(f"  Time limit: {stage_cfg['time_limit']}s")
    print(f"  Success R:  {stage_cfg['success_radius']}m")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Output:     {args.outdir}")
    print(f"{'='*70}\n")

    rclpy.init()

    try:
        # Create a temporary ROS node for publishing configs
        config_node = rclpy.create_node("eval_config_publisher")
        obstacle_pub = config_node.create_publisher(String, "/obstacle_config", 10)
        goal_pub = config_node.create_publisher(String, "/goal_mover/config", 10)

        # Let publishers register
        time.sleep(1.0)

        # Publish obstacle config
        obs_msg = String()
        obs_msg.data = json.dumps({
            "num_walking": stage_cfg["num_walking"],
            "num_standing": stage_cfg["num_standing"],
            "sector_half_angle_deg": stage_cfg["sector_half_angle_deg"],
        })
        obstacle_pub.publish(obs_msg)
        print(f"[CONFIG] Published obstacle config: {obs_msg.data}")

        # Publish goal mover config
        goal_msg = String()
        goal_msg.data = json.dumps({
            "enabled": stage_cfg["goal_enabled"],
            "speed": stage_cfg["goal_speed"],
        })
        goal_pub.publish(goal_msg)
        print(f"[CONFIG] Published goal mover config: {goal_msg.data}")

        # Publish again after a delay to ensure delivery
        time.sleep(1.0)
        obstacle_pub.publish(obs_msg)
        goal_pub.publish(goal_msg)
        time.sleep(0.5)

        # Destroy config node (env will create its own)
        config_node.destroy_node()

        # Create environment
        env_raw = GoalFollowerLSTMEnv(
            cmd_topic="/follower_robot/cmd_vel",
            goal_state_topic="/follower_robot/depth_cam/goal_marker_state",
            goal_odom_topic="/goal_marker/odom",
            dt=0.1,
            success_radius=stage_cfg["success_radius"],
            time_limit=stage_cfg["time_limit"],
            R_goal=100.0,
            R_collision=-100.0,
            gamma_shaping=0.97,
            step_penalty=0.2,
            ttc_threshold=2.0,
            cam_x=0.4,
            cam_y=0.0,
            use_ground_truth_geometry=False,
            smooth_alpha=0.70,
            max_obstacles=args.max_obstacles,
            moving_goal=stage_cfg["goal_enabled"],
        )
        env = Monitor(env_raw)

        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model not found: {args.model}")

        net_arch = [int(x.strip()) for x in args.net_arch.split(",")]
        PolicyClass = create_lstm_sac_policy(
            lstm_hidden_size=args.lstm_hidden,
            lstm_num_layers=args.lstm_layers,
            net_arch=net_arch,
        )

        print(f"Loading model: {args.model}")
        model = SAC.load(args.model, env=env, device="auto",
                         custom_objects={'policy': PolicyClass})

        deterministic = bool(args.deterministic)
        save_traj = bool(args.save_traj)

        all_eps: List[EpisodeResult] = []
        all_raw = {"stage": stage_cfg, "episodes": []}

        for i in range(1, args.episodes + 1):
            seed = args.seed0 + i
            print(f"\n--- Eval episode {i}/{args.episodes} (seed={seed}) ---")

            ep, raw = run_one_episode(
                env, model, i, seed,
                deterministic=deterministic,
                save_traj=save_traj,
            )
            all_eps.append(ep)
            all_raw["episodes"].append({"summary": asdict(ep), "raw": raw})

            print(
                f"  [{ep.reason:>9s}] R={ep.return_sum:>7.1f} | steps={ep.length_steps:>4d} | "
                f"minDist={_safe_float(ep.min_dist):>5.2f}m | "
                f"v={ep.v_cmd_mean:.2f} | obs={ep.num_valid_obstacles_mean:.1f} | "
                f"shield={ep.shield_intervention_rate*100:.0f}% "
                f"[danger={ep.shield_count_danger}, critical={ep.shield_count_critical}] | "
                f"goalDist={ep.goal_dist_mean:.2f}m"
            )

        # Save outputs
        print(f"\nSaving to {args.outdir}...")

        with open(os.path.join(args.outdir, "eval_full.json"), "w") as f:
            json.dump(all_raw, f, indent=2, default=str)

        with open(os.path.join(args.outdir, "eval_summaries.json"), "w") as f:
            json.dump([asdict(e) for e in all_eps], f, indent=2, default=str)

        save_plots(args.outdir, all_eps, stage_cfg["name"])
        write_report(args.outdir, all_eps, stage_cfg)

        # Print final summary
        reasons = [e.reason for e in all_eps]
        success = sum(1 for r in reasons if r == "goal")
        collision = sum(1 for r in reasons if r == "collision")
        timeout = sum(1 for r in reasons if r == "timeout")

        print(f"\n{'='*70}")
        print(f"EVALUATION COMPLETE — {stage_cfg['name']}")
        print(f"{'='*70}")
        print(f"  Episodes:   {len(all_eps)}")
        print(f"  Success:    {success:>3d} ({success/len(all_eps)*100:5.1f}%)")
        print(f"  Collision:  {collision:>3d} ({collision/len(all_eps)*100:5.1f}%)")
        print(f"  Timeout:    {timeout:>3d} ({timeout/len(all_eps)*100:5.1f}%)")
        print(f"  Mean return: {np.mean([e.return_sum for e in all_eps]):.2f} ± {np.std([e.return_sum for e in all_eps]):.2f}")
        print(f"  Mean steps:  {np.mean([e.length_steps for e in all_eps]):.1f}")
        print(f"  Mean shield: {np.mean([e.shield_intervention_rate for e in all_eps])*100:.1f}%")
        if stage_cfg["goal_enabled"]:
            print(f"  Mean goal dist: {np.mean([e.goal_dist_mean for e in all_eps]):.2f}m")
        print(f"{'='*70}")
        print(f"\nOutputs: {os.path.abspath(args.outdir)}/")
        print(f"  report.md  eval_dashboard.png  eval_summaries.json  eval_full.json")

    finally:
        try:
            env_raw.close()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()