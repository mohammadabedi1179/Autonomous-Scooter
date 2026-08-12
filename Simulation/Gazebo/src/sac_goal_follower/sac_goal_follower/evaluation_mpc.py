#!/usr/bin/env python3
"""
Evaluation script for MPC baseline.
Structure mirrors evaluation_curriculum_lstm.py exactly.

Usage:
  python3 evaluation_mpc.py --episodes 100 --stage 6
  python3 evaluation_mpc.py --episodes 100 --stage 7 --horizon 20
"""

import os
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rclpy
from std_msgs.msg import String

from goal_env_lstm import GoalFollowerLSTMEnv
from mpc_policy import MPCController


# ============================================================================
# CURRICULUM STAGE CONFIGS (must match training script)
# ============================================================================

STAGE_CONFIGS = {
    1: {"name": "1: 1s+4w, static goal",    "num_standing": 1, "num_walking": 4, "sector_half_angle_deg": 15.0, "goal_enabled": False, "goal_speed": 0.0, "time_limit": 80.0,  "success_radius": 1.5},
    2: {"name": "2: 2s+4w, static goal",    "num_standing": 2, "num_walking": 4, "sector_half_angle_deg": 30.0, "goal_enabled": False, "goal_speed": 0.0, "time_limit": 80.0,  "success_radius": 1.5},
    3: {"name": "3: 2s+6w, static goal",    "num_standing": 2, "num_walking": 6, "sector_half_angle_deg": 30.0, "goal_enabled": False, "goal_speed": 0.0, "time_limit": 80.0,  "success_radius": 1.5},
    4: {"name": "4: 3s+7w, static goal",    "num_standing": 3, "num_walking": 7, "sector_half_angle_deg": 45.0, "goal_enabled": False, "goal_speed": 0.0, "time_limit": 90.0,  "success_radius": 1.5},
    5: {"name": "5: 3s+7w, slow goal 0.2",  "num_standing": 3, "num_walking": 7, "sector_half_angle_deg": 45.0, "goal_enabled": True,  "goal_speed": 0.2, "time_limit": 100.0, "success_radius": 2.0},
    6: {"name": "6: 3s+7w, fast goal 0.5",  "num_standing": 3, "num_walking": 7, "sector_half_angle_deg": 60.0, "goal_enabled": True,  "goal_speed": 0.5, "time_limit": 100.0, "success_radius": 2.0},
    7: {"name": "7: 3s+7w, fast goal tight", "num_standing": 3, "num_walking": 7, "sector_half_angle_deg": 60.0, "goal_enabled": True,  "goal_speed": 0.5, "time_limit": 80.0,  "success_radius": 1.5},
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


def _extract_obs_features(obs_dict):
    """Extract goal dist and bearing from observation (mirrors RL eval)."""
    try:
        fixed = obs_dict['fixed']
        goal_dist = float(fixed[0]) * 15.0      # denormalize
        goal_bearing = float(fixed[1]) * np.pi
        goal_rel_vx = float(fixed[2]) * 2.0
        goal_rel_vy = float(fixed[3]) * 2.0
        return goal_dist, goal_bearing, goal_rel_vx, goal_rel_vy
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def _count_valid_obstacles(obs_dict):
    """Count active obstacles using depth channel (mirrors RL eval)."""
    try:
        depths = obs_dict['obstacles'][:, 2]
        return int(np.sum(depths < 0.95))
    except Exception:
        return 0


@dataclass
class EpisodeResult:
    episode: int
    seed: int

    length_steps: int
    duration_sec: float

    terminated: bool
    truncated: bool
    reason: str          # "goal" | "collision" | "timeout"

    dist_start: float
    dist_final: float
    min_dist: Optional[float]
    progress_total: float

    goal_dist_mean: float
    goal_dist_min: float
    goal_rel_speed_mean: float

    num_valid_obstacles_mean: float
    shield_intervention_rate: float

    v_cmd_mean: float
    v_cmd_max: float
    w_cmd_mean_abs: float

    solver_fail_rate: float    # fraction of steps where MPC fell back to P-ctrl


# ============================================================================
# RUN ONE EPISODE
# ============================================================================

def run_one_episode(
    env: GoalFollowerLSTMEnv,
    mpc: MPCController,
    ep_i: int,
    seed: int,
    max_steps_hard_cap: int = 5000,
) -> Tuple[EpisodeResult, Dict[str, Any]]:

    # Clear MPC warm-start cache at episode boundary
    mpc.reset()

    obs, _ = env.reset(seed=seed)
    t0 = time.time()

    dists, goal_dists, goal_rel_speeds = [], [], []
    v_cmds, w_cmds = [], []
    valid_obstacles, shield_active = [], []
    solver_fails = []

    dist_start, _, _, _ = _extract_obs_features(obs)

    terminated, truncated, reason = False, False, ""
    min_dist = None

    steps = 0
    while True:
        steps += 1
        if steps > max_steps_hard_cap:
            truncated = True
            reason = "timeout"
            break

        # MPC step
        action, _ = mpc.predict(obs)
        # Read solver-fail flag directly from MPC (not from info dict)
        solver_fails.append(mpc.last_solver_failed)

        obs, _reward, term, trunc, info = env.step(action)

        # Metrics  
        d, _b, grvx, grvy = _extract_obs_features(obs)
        dists.append(d)
        goal_dists.append(d)
        goal_rel_speeds.append(np.sqrt(grvx**2 + grvy**2) if np.isfinite(grvx) else np.nan)

        valid_obstacles.append(_count_valid_obstacles(obs))

        shield_level = info.get("shield_level", "none")
        shield_active.append(shield_level != "none")

        # v/w from info (same keys as RL eval)
        v_cmds.append(_safe_float(info.get("v_cmd", np.nan)))
        w_cmds.append(_safe_float(info.get("w_cmd", np.nan)))

        if term or trunc:
            terminated = bool(term)
            truncated = bool(trunc)
            # Normalise reason string to match RL eval conventions
            raw_reason = str(info.get("reason", info.get("termination_reason", "")))
            if "goal" in raw_reason or info.get("success", False):
                reason = "goal"
            elif "collision" in raw_reason:
                reason = "collision"
            else:
                reason = "timeout"
            min_dist = info.get("min_dist")
            break

    duration = time.time() - t0

    dists_arr         = np.array(dists, dtype=float)
    v_arr             = np.array(v_cmds, dtype=float)
    w_arr             = np.array(w_cmds, dtype=float)
    valid_obs_arr     = np.array(valid_obstacles, dtype=float)
    shield_arr        = np.array(shield_active, dtype=bool)
    goal_dists_arr    = np.array(goal_dists, dtype=float)
    goal_rel_arr      = np.array(goal_rel_speeds, dtype=float)
    solver_fail_arr   = np.array(solver_fails, dtype=bool)

    dist_final     = float(dists_arr[-1]) if dists_arr.size else np.nan
    progress_total = float(dist_start - dist_final) if (np.isfinite(dist_start) and np.isfinite(dist_final)) else np.nan

    ep = EpisodeResult(
        episode=ep_i,
        seed=seed,
        length_steps=steps,
        duration_sec=duration,
        terminated=terminated,
        truncated=truncated,
        reason=reason,
        dist_start=dist_start,
        dist_final=dist_final,
        min_dist=_safe_float(min_dist) if min_dist is not None else None,
        progress_total=progress_total,
        goal_dist_mean=float(np.nanmean(goal_dists_arr)) if goal_dists_arr.size else np.nan,
        goal_dist_min=float(np.nanmin(goal_dists_arr)) if goal_dists_arr.size else np.nan,
        goal_rel_speed_mean=float(np.nanmean(goal_rel_arr)) if goal_rel_arr.size else np.nan,
        num_valid_obstacles_mean=float(np.mean(valid_obs_arr)) if valid_obs_arr.size else np.nan,
        shield_intervention_rate=float(np.mean(shield_arr)) if shield_arr.size else 0.0,
        v_cmd_mean=float(np.nanmean(v_arr)) if v_arr.size else np.nan,
        v_cmd_max=float(np.nanmax(v_arr)) if v_arr.size else np.nan,
        w_cmd_mean_abs=float(np.nanmean(np.abs(w_arr))) if w_arr.size else np.nan,
        solver_fail_rate=float(np.mean(solver_fail_arr)) if solver_fail_arr.size else 0.0,
    )

    raw = {
        "dists":          [float(x) for x in dists],
        "v_cmds":         [float(x) for x in v_cmds],
        "w_cmds":         [float(x) for x in w_cmds],
        "valid_obstacles":[int(x) for x in valid_obstacles],
        "shield_active":  [bool(x) for x in shield_active],
        "goal_dists":     [float(x) for x in goal_dists],
        "goal_rel_speeds":[float(x) for x in goal_rel_speeds],
        "solver_fails":   [bool(x) for x in solver_fails],
    }

    return ep, raw


# ============================================================================
# PLOTS  (mirrors RL eval dashboard layout)
# ============================================================================

def save_plots(outdir: str, episodes: List[EpisodeResult], stage_name: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    L             = np.array([e.length_steps for e in episodes])
    Vmean         = np.array([e.v_cmd_mean for e in episodes])
    ValidObs      = np.array([e.num_valid_obstacles_mean for e in episodes])
    ShieldRate    = np.array([e.shield_intervention_rate for e in episodes])
    GoalDistMean  = np.array([e.goal_dist_mean for e in episodes])
    MinDist       = np.array([_safe_float(e.min_dist) for e in episodes])
    SolverFail    = np.array([e.solver_fail_rate for e in episodes])
    Progress      = np.array([e.progress_total for e in episodes])

    colors_map = {"goal": "#00e676", "timeout": "#ff9100", "collision": "#ff1744"}
    ep_colors  = [colors_map.get(e.reason, "#888") for e in episodes]

    # Cumulative success rate
    successes = np.array([e.reason == "goal" for e in episodes], dtype=float)
    cumulative_sr = np.cumsum(successes) / np.arange(1, len(successes) + 1)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f"MPC Evaluation: {stage_name}", fontsize=14, fontweight="bold")

    # 1. Cumulative success rate
    ax = axes[0, 0]
    ax.plot(cumulative_sr, linewidth=2, color="#00e676")
    ax.axhline(float(np.mean(successes)), color="red", ls="--",
               label=f"Final: {float(np.mean(successes)):.2%}")
    ax.set_xlabel("Episode"); ax.set_ylabel("Success Rate")
    ax.set_title("Cumulative Success Rate")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Episode lengths
    ax = axes[0, 1]
    ax.bar(range(len(L)), L, color=ep_colors, alpha=0.8, width=1.0)
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps")
    ax.set_title("Episode Lengths"); ax.grid(True, alpha=0.3)

    # 3. Outcome pie
    ax = axes[0, 2]
    reasons = [e.reason for e in episodes]
    labels_counts = {}
    for r in reasons:
        labels_counts[r] = labels_counts.get(r, 0) + 1
    pie_colors = [colors_map.get(k, "#888") for k in labels_counts]
    pie_labels = [f"{k}\n({v})" for k, v in labels_counts.items()]
    ax.pie(labels_counts.values(), labels=pie_labels, colors=pie_colors,
           autopct="%1.1f%%", startangle=90)
    ax.set_title("Outcomes")

    # 4. Goal distance (moving goal tracking)
    ax = axes[1, 0]
    ax.plot(GoalDistMean, "o-", markersize=3, color="#2196f3", alpha=0.7, label="Mean dist")
    ax.plot(MinDist,      "o-", markersize=3, color="#00e676", alpha=0.5, label="Min dist")
    ax.set_xlabel("Episode"); ax.set_ylabel("Goal Distance (m)")
    ax.set_title("Goal Tracking"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. Progress per episode
    ax = axes[1, 1]
    ax.bar(range(len(Progress)), Progress, color=ep_colors, alpha=0.8, width=1.0)
    ax.axhline(float(np.nanmean(Progress)), color="white", ls="--", alpha=0.7,
               label=f"mean={float(np.nanmean(Progress)):.2f}m")
    ax.set_xlabel("Episode"); ax.set_ylabel("Progress (m)")
    ax.set_title("Progress Toward Goal"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6. Shield rate
    ax = axes[1, 2]
    ax.bar(range(len(ShieldRate)), ShieldRate * 100, color="#ff5722", alpha=0.7, width=1.0)
    ax.set_xlabel("Episode"); ax.set_ylabel("Shield Rate (%)")
    ax.set_title("Safety Shield Intervention Rate"); ax.grid(True, alpha=0.3)

    # 7. MPC solver fail rate
    ax = axes[2, 0]
    ax.bar(range(len(SolverFail)), SolverFail * 100, color="#9c27b0", alpha=0.7, width=1.0)
    ax.set_xlabel("Episode"); ax.set_ylabel("Solver Fail Rate (%)")
    ax.set_title("MPC Solver Fallback Rate"); ax.grid(True, alpha=0.3)

    # 8. Valid obstacles seen
    ax = axes[2, 1]
    ax.plot(ValidObs, "o-", markersize=3, color="#ffaa00", alpha=0.7)
    ax.set_xlabel("Episode"); ax.set_ylabel("Mean Valid Obstacles")
    ax.set_title("Obstacle Density"); ax.grid(True, alpha=0.3)

    # 9. Speed distribution
    ax = axes[2, 2]
    ax.scatter(Vmean, GoalDistMean, c=ep_colors, alpha=0.7, s=25)
    ax.set_xlabel("Mean v_cmd (m/s)"); ax.set_ylabel("Mean Goal Dist (m)")
    ax.set_title("Speed vs Goal Distance"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mpc_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved to {os.path.join(outdir, 'mpc_dashboard.png')}")


# ============================================================================
# REPORT
# ============================================================================

def write_report(outdir: str, episodes: List[EpisodeResult], stage_cfg: Dict) -> None:
    N       = len(episodes)
    reasons = [e.reason for e in episodes]
    success   = sum(1 for r in reasons if r == "goal")
    collision = sum(1 for r in reasons if r == "collision")
    timeout   = sum(1 for r in reasons if r == "timeout")

    L            = np.array([e.length_steps for e in episodes])
    ShieldRate   = np.array([e.shield_intervention_rate for e in episodes])
    GoalDistMean = np.array([e.goal_dist_mean for e in episodes])
    GoalDistMin  = np.array([e.goal_dist_min for e in episodes])
    GoalRelSpeed = np.array([e.goal_rel_speed_mean for e in episodes])
    SolverFail   = np.array([e.solver_fail_rate for e in episodes])
    Vmean        = np.array([e.v_cmd_mean for e in episodes])
    Wabs         = np.array([e.w_cmd_mean_abs for e in episodes])
    ValidObs     = np.array([e.num_valid_obstacles_mean for e in episodes])

    lines = []
    lines.append(f"# MPC Baseline Evaluation Report\n\n")
    lines.append(f"**Stage**: {stage_cfg['name']}\n\n")
    lines.append(f"**Config**: {stage_cfg['num_walking']} walking + {stage_cfg['num_standing']} standing, "
                 f"goal {'moving at ' + str(stage_cfg['goal_speed']) + ' m/s' if stage_cfg['goal_enabled'] else 'static'}\n\n")
    lines.append(f"**Time limit**: {stage_cfg['time_limit']}s | **Success radius**: {stage_cfg['success_radius']}m\n\n")
    lines.append(f"**Episodes**: {N}\n\n")

    lines.append("## Outcome Summary\n\n")
    lines.append("| Outcome | Count | Percentage |\n")
    lines.append("|---------|------:|-----------:|\n")
    lines.append(f"| Success   | {success}   | {success/N*100:.1f}% |\n")
    lines.append(f"| Collision | {collision} | {collision/N*100:.1f}% |\n")
    lines.append(f"| Timeout   | {timeout}   | {timeout/N*100:.1f}% |\n\n")

    l_mean, l_ci = _mean_ci95(L)
    lines.append("## Performance\n\n")
    lines.append(f"- **Mean episode length**: {l_mean:.1f} ± {l_ci:.1f} steps\n")
    lines.append(f"- **Mean v_cmd**: {float(np.nanmean(Vmean)):.3f} m/s\n")
    lines.append(f"- **Mean |w_cmd|**: {float(np.nanmean(Wabs)):.3f} rad/s\n")
    lines.append(f"- **Mean valid obstacles**: {float(np.nanmean(ValidObs)):.2f}\n\n")

    lines.append("## Safety\n\n")
    lines.append(f"- **Collision rate**: {collision/N*100:.1f}%\n")
    lines.append(f"- **Mean shield intervention rate**: {float(np.nanmean(ShieldRate))*100:.1f}%\n")
    lines.append(f"- **Mean MPC solver fail rate**: {float(np.nanmean(SolverFail))*100:.1f}%\n\n")

    lines.append("## Moving Goal Tracking\n\n")
    lines.append(f"- **Goal speed**: {stage_cfg['goal_speed']} m/s\n")
    lines.append(f"- **Mean goal distance during episode**: {float(np.nanmean(GoalDistMean)):.2f}m\n")
    lines.append(f"- **Mean closest approach to goal**: {float(np.nanmean(GoalDistMin)):.2f}m\n")
    lines.append(f"- **Mean relative speed to goal**: {float(np.nanmean(GoalRelSpeed)):.3f} m/s\n")
    if success > 0:
        succ_eps = [e for e in episodes if e.reason == "goal"]
        lines.append(f"- **Successful episodes mean steps**: {np.mean([e.length_steps for e in succ_eps]):.1f}\n")
    lines.append("\n")

    for label, fr in [("Successful", "goal"), ("Collision", "collision"), ("Timeout", "timeout")]:
        subset = [e for e in episodes if e.reason == fr]
        if not subset:
            continue
        lines.append(f"### {label} Episodes (n={len(subset)})\n\n")
        lines.append(f"- Mean steps: {np.mean([e.length_steps for e in subset]):.1f}\n")
        lines.append(f"- Mean shield rate: {np.mean([e.shield_intervention_rate for e in subset])*100:.1f}%\n")
        lines.append(f"- Mean goal dist: {np.mean([e.goal_dist_mean for e in subset]):.2f}m\n\n")

    report_path = os.path.join(outdir, "report.md")
    with open(report_path, "w") as f:
        f.writelines(lines)
    print(f"[REPORT] Saved to {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Evaluate MPC baseline")
    ap.add_argument("--episodes",         type=int,   default=100)
    ap.add_argument("--stage",            type=int,   default=6,     help="Curriculum stage (1-7)")
    ap.add_argument("--max_obstacles",    type=int,   default=10)
    ap.add_argument("--horizon",          type=int,   default=15,    help="MPC prediction horizon")
    ap.add_argument("--goal_weight",      type=float, default=10.0)
    ap.add_argument("--obstacle_weight",  type=float, default=100.0)
    ap.add_argument("--seed0",            type=int,   default=12345)
    ap.add_argument("--outdir",           type=str,   default="./mpc_eval_out")
    ap.add_argument("--verbose",          action="store_true")
    args = ap.parse_args()

    if args.stage not in STAGE_CONFIGS:
        raise ValueError(f"Invalid stage {args.stage}. Choose 1-7.")

    stage_cfg = STAGE_CONFIGS[args.stage]
    os.makedirs(args.outdir, exist_ok=True)

    print(f"{'='*70}")
    print(f"MPC BASELINE EVALUATION")
    print(f"{'='*70}")
    print(f"  Stage:      {stage_cfg['name']}")
    print(f"  Obstacles:  {stage_cfg['num_walking']}w + {stage_cfg['num_standing']}s, sector={stage_cfg['sector_half_angle_deg']}deg")
    print(f"  Goal:       {'moving at ' + str(stage_cfg['goal_speed']) + ' m/s' if stage_cfg['goal_enabled'] else 'static'}")
    print(f"  Time limit: {stage_cfg['time_limit']}s  |  Success R: {stage_cfg['success_radius']}m")
    print(f"  MPC:        horizon={args.horizon}, goal_w={args.goal_weight}, obs_w={args.obstacle_weight}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Output:     {args.outdir}")
    print(f"{'='*70}\n")

    rclpy.init()

    try:
        # ── 1. Temporary config publisher node (mirrors RL eval exactly) ──────
        config_node = rclpy.create_node("mpc_eval_config_publisher")
        obstacle_pub = config_node.create_publisher(String, "/obstacle_config", 10)
        goal_pub     = config_node.create_publisher(String, "/goal_mover/config", 10)

        # Let publishers register with the ROS graph
        time.sleep(1.0)

        # Build messages
        obs_msg = String()
        obs_msg.data = json.dumps({
            "num_walking":          stage_cfg["num_walking"],
            "num_standing":         stage_cfg["num_standing"],
            "sector_half_angle_deg": stage_cfg["sector_half_angle_deg"],
        })

        goal_msg = String()
        goal_msg.data = json.dumps({
            "enabled": stage_cfg["goal_enabled"],
            "speed":   stage_cfg["goal_speed"],
        })

        # First publish
        obstacle_pub.publish(obs_msg)
        print(f"[CONFIG] Published obstacle config: {obs_msg.data}")
        goal_pub.publish(goal_msg)
        print(f"[CONFIG] Published goal mover config: {goal_msg.data}")

        # Second publish after delay (guarantees delivery — same as RL eval)
        time.sleep(1.0)
        obstacle_pub.publish(obs_msg)
        goal_pub.publish(goal_msg)
        time.sleep(0.5)

        # Destroy config node before creating env (env creates its own node)
        config_node.destroy_node()

        # ── 2. Create environment ─────────────────────────────────────────────
        env = GoalFollowerLSTMEnv(
            max_obstacles=args.max_obstacles,
            time_limit=stage_cfg["time_limit"],
            success_radius=stage_cfg["success_radius"],
            moving_goal=stage_cfg["goal_enabled"],
        )

        # ── 3. Create MPC controller ──────────────────────────────────────────
        mpc = MPCController(
            horizon=args.horizon,
            dt=0.1,
            v_max=1.0,
            w_max=1.0,
            goal_weight=args.goal_weight,
            obstacle_weight=args.obstacle_weight,
            max_obstacles=args.max_obstacles,
            verbose=args.verbose,
        )

        # ── 4. Evaluation loop ────────────────────────────────────────────────
        all_eps: List[EpisodeResult] = []
        all_raw = {"stage": stage_cfg, "episodes": []}

        for i in range(1, args.episodes + 1):
            seed = args.seed0 + i
            print(f"\n--- MPC episode {i}/{args.episodes} (seed={seed}) ---")

            ep, raw = run_one_episode(env, mpc, i, seed)
            all_eps.append(ep)
            all_raw["episodes"].append({"summary": asdict(ep), "raw": raw})

            print(
                f"  [{ep.reason:>9s}] steps={ep.length_steps:>4d} | "
                f"goalDist={ep.goal_dist_mean:.2f}m | "
                f"progress={ep.progress_total:.2f}m | "
                f"shield={ep.shield_intervention_rate*100:.0f}% | "
                f"solverFail={ep.solver_fail_rate*100:.0f}%"
            )

        # ── 5. Save outputs ───────────────────────────────────────────────────
        print(f"\nSaving to {args.outdir}...")

        with open(os.path.join(args.outdir, "eval_full.json"), "w") as f:
            json.dump(all_raw, f, indent=2, default=str)

        with open(os.path.join(args.outdir, "eval_summaries.json"), "w") as f:
            json.dump([asdict(e) for e in all_eps], f, indent=2, default=str)

        save_plots(args.outdir, all_eps, stage_cfg["name"])
        write_report(args.outdir, all_eps, stage_cfg)

        # ── 6. Final summary ──────────────────────────────────────────────────
        reasons   = [e.reason for e in all_eps]
        success   = sum(1 for r in reasons if r == "goal")
        collision = sum(1 for r in reasons if r == "collision")
        timeout   = sum(1 for r in reasons if r == "timeout")

        print(f"\n{'='*70}")
        print(f"MPC EVALUATION COMPLETE — {stage_cfg['name']}")
        print(f"{'='*70}")
        print(f"  Episodes:       {len(all_eps)}")
        print(f"  Success:        {success:>3d} ({success/len(all_eps)*100:5.1f}%)")
        print(f"  Collision:      {collision:>3d} ({collision/len(all_eps)*100:5.1f}%)")
        print(f"  Timeout:        {timeout:>3d} ({timeout/len(all_eps)*100:5.1f}%)")
        print(f"  Mean steps:     {np.mean([e.length_steps for e in all_eps]):.1f}")
        print(f"  Mean shield:    {np.mean([e.shield_intervention_rate for e in all_eps])*100:.1f}%")
        print(f"  Mean solverFail:{np.mean([e.solver_fail_rate for e in all_eps])*100:.1f}%")
        if stage_cfg["goal_enabled"]:
            print(f"  Mean goal dist: {np.mean([e.goal_dist_mean for e in all_eps]):.2f}m")
        print(f"{'='*70}")
        print(f"\nOutputs: {os.path.abspath(args.outdir)}/")
        print(f"  report.md  mpc_dashboard.png  eval_summaries.json  eval_full.json")

    finally:
        try:
            env.close()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()