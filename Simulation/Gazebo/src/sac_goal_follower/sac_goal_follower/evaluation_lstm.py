#!/usr/bin/env python3
"""
Evaluate a trained LSTM-SAC agent on GoalFollowerLSTMEnv (ROS2 + Gazebo).

What it does:
- Runs N evaluation episodes (default 50)
- Logs per-step + per-episode metrics  
- Produces a deep analysis report (Markdown) + plots + JSON/CSV outputs

Usage example:
  python3 evaluation_lstm.py \
    --model ./models_lstm/lstm_sac_goal_follower_best.zip \
    --episodes 50 \
    --outdir ./eval_lstm_out \
    --deterministic 1

Notes:
- Requires ROS2 environment running + Gazebo running + your topics/services available.
- Must be executed in the same ROS2 workspace env where detectors_msgs etc are available.
- This version handles dict observations from the LSTM environment.
"""

import os
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import rclpy

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

# Import LSTM environment and policy
from goal_env_lstm import GoalFollowerLSTMEnv
from lstm_sac_policy import create_lstm_sac_policy


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _mean_ci95(xs: np.ndarray) -> Tuple[float, float]:
    """Return (mean, halfwidth_95CI) using normal approx; fine for quick reporting."""
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

    # Step-derived stats
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
    
    # New LSTM-specific metrics
    num_valid_obstacles_mean: float
    shield_intervention_rate: float

    # Paths (optional, can be large)
    robot_traj: Optional[List[List[float]]] = None


def _extract_obs_features(obs_dict: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """
    Extract goal distance and bearing from LSTM dict observation.
    
    obs_dict['fixed'] = [goal_dist, goal_bearing, goal_rel_vx, goal_rel_vy, v, w, 
                         ultra_fl, ultra_fr, ultra_ls, ultra_rs, prev_v, prev_w]
    """
    try:
        fixed = obs_dict['fixed']
        # Unnormalize: goal_dist is normalized by GOAL_DIST_MAX (15.0)
        goal_dist = float(fixed[0]) * 15.0  # Unnormalize
        goal_bearing = float(fixed[1]) * np.pi  # Unnormalize from [-1, 1] to [-π, π]
        return goal_dist, goal_bearing
    except Exception:
        return np.nan, np.nan


def _count_valid_obstacles(obs_dict: Dict[str, np.ndarray]) -> int:
    """
    Count number of valid (non-dummy) obstacles in observation.
    
    obs_dict['obstacles'] = [6×5] where each row is [x, y, depth, vx, vy]
    Dummy obstacles have depth ≈ 1.0 (max normalized value = max distance)
    """
    try:
        obstacles = obs_dict['obstacles']
        depths = obstacles[:, 2]  # Third column is depth (normalized)
        # Valid obstacles have depth < 0.95 (not at max distance)
        valid = np.sum(depths < 0.95)
        return int(valid)
    except Exception:
        return 0


def run_one_episode(
    env: Monitor,
    model: SAC,
    ep_i: int,
    seed: int,
    deterministic: bool = True,
    max_steps_hard_cap: int = 5000,
) -> Tuple[EpisodeResult, Dict[str, Any]]:
    """
    Runs a single episode and returns:
    - EpisodeResult (summary)
    - raw dict with per-step arrays (for deeper offline analysis)
    """
    obs, _ = env.reset(seed=seed)

    t0 = time.time()

    # Per-step logs
    rews: List[float] = []
    dists: List[float] = []
    bearings: List[float] = []
    v_cmds: List[float] = []
    w_cmds: List[float] = []
    ttcs: List[float] = []
    min_ultras: List[float] = []
    min_obs_depths: List[float] = []
    valid_obstacles: List[int] = []
    shield_active: List[bool] = []

    robot_traj_last: Optional[List[Tuple[float, float]]] = None

    # For progress (extract from dict observation)
    dist_start, _ = _extract_obs_features(obs)
    prev_dist = dist_start

    terminated = False
    truncated = False
    reason = ""

    # placeholders from info
    robot_start = None
    goal_start = None
    robot_final = None
    goal_final = None
    min_dist = None

    steps = 0
    while True:
        steps += 1
        if steps > max_steps_hard_cap:
            # hard safety stop (should not happen if time_limit works)
            truncated = True
            reason = "hard_cap"
            break

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, term, trunc, info = env.step(action)

        rews.append(float(reward))

        # Extract features from dict observation
        d, b = _extract_obs_features(obs)
        dists.append(d)
        bearings.append(b)
        
        # Count valid obstacles
        valid_obs = _count_valid_obstacles(obs)
        valid_obstacles.append(valid_obs)

        # From info (your env provides these)
        v_cmds.append(_safe_float(info.get("v_cmd", np.nan)))
        w_cmds.append(_safe_float(info.get("w_cmd", np.nan)))
        ttcs.append(_safe_float(info.get("ttc", np.inf), default=np.inf))

        min_ultras.append(_safe_float(info.get("min_ultrasonic_distance", np.nan)))
        min_obs_depths.append(_safe_float(info.get("min_obstacle_depth", np.nan)))
        
        # Shield status
        shield_level = info.get("shield_level", "none")
        shield_active.append(shield_level != "none")

        # Episode boundary
        if term or trunc:
            terminated = bool(term)
            truncated = bool(trunc)
            reason = str(info.get("reason", ""))

            robot_start = info.get("robot_start", None)
            goal_start = info.get("goal_start", None)
            robot_final = info.get("robot_final", None)
            goal_final = info.get("goal_final", None)
            min_dist = info.get("min_dist", None)

            # robot_traj is stored in env info at end as "robot_traj" (list of (x,y))
            robot_traj_last = info.get("robot_traj", None)
            break

        # progress tracking
        if np.isfinite(prev_dist) and np.isfinite(d):
            prev_dist = d

    duration = time.time() - t0
    ret = float(np.sum(rews)) if rews else 0.0
    length = int(len(rews))

    dists_arr = np.array(dists, dtype=float)
    v_arr = np.array(v_cmds, dtype=float)
    w_arr = np.array(w_cmds, dtype=float)
    ttc_arr = np.array(ttcs, dtype=float)
    min_ultra_arr = np.array(min_ultras, dtype=float)
    min_obs_arr = np.array(min_obs_depths, dtype=float)
    valid_obs_arr = np.array(valid_obstacles, dtype=float)
    shield_arr = np.array(shield_active, dtype=bool)

    # progress_total: dist_start - dist_final (positive is good)
    dist_final = float(dists_arr[-1]) if dists_arr.size else np.nan
    progress_total = float(dist_start - dist_final) if (np.isfinite(dist_start) and np.isfinite(dist_final)) else np.nan

    # TTC fraction below 2s (danger zone)
    ttc_below = np.isfinite(ttc_arr) & (ttc_arr < 2.0)
    ttc_frac = float(np.mean(ttc_below)) if ttc_arr.size else np.nan
    
    # Valid obstacles mean
    valid_obs_mean = float(np.mean(valid_obs_arr)) if valid_obs_arr.size else np.nan
    
    # Shield intervention rate
    shield_rate = float(np.mean(shield_arr)) if shield_arr.size else 0.0

    ep = EpisodeResult(
        episode=ep_i,
        seed=seed,
        return_sum=ret,
        length_steps=length,
        duration_sec=float(duration),
        terminated=terminated,
        truncated=truncated,
        reason=reason,

        robot_start=tuple(robot_start) if isinstance(robot_start, (list, tuple)) else None,
        goal_start=tuple(goal_start) if isinstance(goal_start, (list, tuple)) else None,
        robot_final=tuple(robot_final) if isinstance(robot_final, (list, tuple)) else None,
        goal_final=tuple(goal_final) if isinstance(goal_final, (list, tuple)) else None,
        min_dist=_safe_float(min_dist, default=np.nan) if min_dist is not None else None,

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
        
        num_valid_obstacles_mean=valid_obs_mean,
        shield_intervention_rate=shield_rate,

        robot_traj=robot_traj_last,
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
        "shield_active": [bool(x) for x in shield_active],
    }

    return ep, raw


def save_plots(outdir: str, episodes: List[EpisodeResult]) -> None:
    os.makedirs(outdir, exist_ok=True)

    R = np.array([e.return_sum for e in episodes])
    L = np.array([e.length_steps for e in episodes])
    TTCmin = np.array([e.ttc_min for e in episodes])
    Vmean = np.array([e.v_cmd_mean for e in episodes])
    ValidObs = np.array([e.num_valid_obstacles_mean for e in episodes])
    ShieldRate = np.array([e.shield_intervention_rate for e in episodes])

    # Return per episode
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(R, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode returns")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(R, bins=20, edgecolor="black")
    plt.xlabel("Return")
    plt.ylabel("Count")
    plt.title("Return distribution")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "returns.png"), dpi=150)
    plt.close()

    # Length per episode
    plt.figure()
    plt.plot(L, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode lengths")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lengths.png"), dpi=150)
    plt.close()

    # Safety: min TTC per episode
    plt.figure()
    plt.plot(TTCmin, marker="o")
    plt.axhline(2.0, linestyle="--", color="red", label="TTC threshold")
    plt.xlabel("Episode")
    plt.ylabel("Min TTC (s)")
    plt.title("Safety: minimum time-to-collision (lower is worse)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ttc_min.png"), dpi=150)
    plt.close()

    # Return vs mean speed
    plt.figure()
    plt.scatter(Vmean, R)
    plt.xlabel("Mean commanded v (m/s)")
    plt.ylabel("Return")
    plt.title("Return vs mean speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "return_vs_speed.png"), dpi=150)
    plt.close()
    
    # LSTM-specific: Valid obstacles over episodes
    plt.figure()
    plt.plot(ValidObs, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Mean # valid obstacles")
    plt.title("Average number of valid obstacles per episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "valid_obstacles.png"), dpi=150)
    plt.close()
    
    # Shield intervention rate
    plt.figure()
    plt.plot(ShieldRate * 100, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Shield intervention rate (%)")
    plt.title("Safety shield intervention rate per episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "shield_rate.png"), dpi=150)
    plt.close()


def write_report(outdir: str, episodes: List[EpisodeResult]) -> None:
    os.makedirs(outdir, exist_ok=True)

    reasons = [e.reason for e in episodes]
    success = sum(1 for r in reasons if r == "goal")
    collision = sum(1 for r in reasons if r == "collision")
    timeout = sum(1 for r in reasons if r == "timeout")
    hard_cap = sum(1 for r in reasons if r == "hard_cap")
    other = len(episodes) - (success + collision + timeout + hard_cap)

    R = np.array([e.return_sum for e in episodes], dtype=float)
    L = np.array([e.length_steps for e in episodes], dtype=float)
    P = np.array([e.progress_total for e in episodes], dtype=float)
    TTCmin = np.array([e.ttc_min for e in episodes], dtype=float)
    TTCfrac = np.array([e.ttc_frac_below_2s for e in episodes], dtype=float)
    Vmean = np.array([e.v_cmd_mean for e in episodes], dtype=float)
    Wabs = np.array([e.w_cmd_mean_abs for e in episodes], dtype=float)
    ValidObs = np.array([e.num_valid_obstacles_mean for e in episodes], dtype=float)
    ShieldRate = np.array([e.shield_intervention_rate for e in episodes], dtype=float)

    r_mean, r_ci = _mean_ci95(R)
    l_mean, l_ci = _mean_ci95(L)

    def corr(a, b) -> float:
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if np.sum(m) < 3:
            return np.nan
        return float(np.corrcoef(a[m], b[m])[0, 1])

    corr_R_L = corr(R, L)
    corr_R_TTC = corr(R, TTCmin)
    corr_R_V = corr(R, Vmean)
    corr_R_W = corr(R, Wabs)

    lines = []
    lines.append("# LSTM-SAC Evaluation Report\n")
    lines.append(f"**Evaluated on {len(episodes)} episodes**\n\n")
    
    lines.append("## Outcome summary\n")
    lines.append(f"- Total episodes: {len(episodes)}\n")
    lines.append(f"- **Success (goal)**: {success} ({success/len(episodes)*100:.1f}%)\n")
    lines.append(f"- **Collision**: {collision} ({collision/len(episodes)*100:.1f}%)\n")
    lines.append(f"- **Timeout**: {timeout} ({timeout/len(episodes)*100:.1f}%)\n")
    if hard_cap:
        lines.append(f"- Hard cap: {hard_cap} ({hard_cap/len(episodes)*100:.1f}%)\n")
    if other:
        lines.append(f"- Other: {other} ({other/len(episodes)*100:.1f}%)\n")

    lines.append("\n## Return & efficiency\n")
    lines.append(f"- **Mean return**: {r_mean:.3f} (± {r_ci:.3f} 95% CI)\n")
    lines.append(f"- **Mean length**: {l_mean:.1f} steps (± {l_ci:.1f} 95% CI)\n")
    lines.append(f"- Return ↔ Length correlation: {corr_R_L:.3f}\n")
    lines.append(f"- Best episode return: {float(np.nanmax(R)):.3f}\n")
    lines.append(f"- Worst episode return: {float(np.nanmin(R)):.3f}\n")

    lines.append("\n## Progress dynamics\n")
    lines.append(f"- **Mean total progress** (dist_start - dist_final): {float(np.nanmean(P)):.3f} m\n")
    lines.append(f"- Median total progress: {float(np.nanmedian(P)):.3f} m\n")
    lines.append(f"- Mean starting distance: {float(np.nanmean([e.dist_start for e in episodes])):.3f} m\n")
    lines.append(f"- Mean final distance: {float(np.nanmean([e.dist_final for e in episodes])):.3f} m\n")

    lines.append("\n## Safety (TTC + proximity)\n")
    lines.append(f"- **Mean(min TTC)**: {float(np.nanmean(TTCmin)):.3f} s\n")
    lines.append(f"- Median(min TTC): {float(np.nanmedian(TTCmin)):.3f} s\n")
    lines.append(f"- **Mean fraction of steps TTC<2s**: {float(np.nanmean(TTCfrac)):.3f}\n")
    lines.append(f"- Episodes with TTC<1s: {int(np.sum(TTCmin < 1.0))}\n")
    lines.append(f"- Return ↔ minTTC correlation: {corr_R_TTC:.3f} (positive = safer episodes score higher)\n")

    lines.append("\n## Control behavior\n")
    lines.append(f"- **Mean v_cmd**: {float(np.nanmean(Vmean)):.3f} m/s\n")
    lines.append(f"- Max v_cmd across all episodes: {float(np.nanmax([e.v_cmd_max for e in episodes])):.3f} m/s\n")
    lines.append(f"- **Mean |w_cmd|**: {float(np.nanmean(Wabs)):.3f} rad/s\n")
    lines.append(f"- Return ↔ mean speed correlation: {corr_R_V:.3f}\n")
    lines.append(f"- Return ↔ mean turning correlation: {corr_R_W:.3f}\n")
    
    lines.append("\n## LSTM-specific metrics\n")
    lines.append(f"- **Mean valid obstacles per episode**: {float(np.nanmean(ValidObs)):.2f}\n")
    lines.append(f"- Min valid obstacles (any episode): {float(np.nanmin(ValidObs)):.2f}\n")
    lines.append(f"- Max valid obstacles (any episode): {float(np.nanmax(ValidObs)):.2f}\n")
    lines.append(f"- **Mean shield intervention rate**: {float(np.nanmean(ShieldRate))*100:.2f}%\n")
    lines.append(f"- Episodes with shield interventions: {int(np.sum(ShieldRate > 0))}\n")
    lines.append(f"- Episodes with >10% shield rate: {int(np.sum(ShieldRate > 0.1))}\n")

    lines.append("\n## Per-episode table\n")
    lines.append("| Ep | Reason | Return | Steps | MinDist | MinTTC | TTC<2s | v_mean | |w|_mean | ValidObs | Shield% |\n")
    lines.append("|---:|:------:|-------:|------:|--------:|-------:|-------:|-------:|--------:|---------:|--------:|\n")
    for e in episodes:
        lines.append(
            f"| {e.episode:>2d} | {e.reason or '-':^6s} | {e.return_sum:>7.2f} | {e.length_steps:>5d} "
            f"| {(_safe_float(e.min_dist) if e.min_dist is not None else np.nan):>7.2f} "
            f"| {e.ttc_min:>6.2f} | {e.ttc_frac_below_2s:>6.3f} "
            f"| {e.v_cmd_mean:>6.3f} | {e.w_cmd_mean_abs:>7.3f} "
            f"| {e.num_valid_obstacles_mean:>8.2f} | {e.shield_intervention_rate*100:>7.1f} |\n"
        )

    lines.append("\n## Success/failure breakdown by outcome\n")
    if success > 0:
        success_eps = [e for e in episodes if e.reason == "goal"]
        lines.append(f"\n### Successful episodes (n={success})\n")
        lines.append(f"- Mean return: {np.mean([e.return_sum for e in success_eps]):.3f}\n")
        lines.append(f"- Mean steps: {np.mean([e.length_steps for e in success_eps]):.1f}\n")
        lines.append(f"- Mean shield rate: {np.mean([e.shield_intervention_rate for e in success_eps])*100:.1f}%\n")
    
    if collision > 0:
        collision_eps = [e for e in episodes if e.reason == "collision"]
        lines.append(f"\n### Collision episodes (n={collision})\n")
        lines.append(f"- Mean return: {np.mean([e.return_sum for e in collision_eps]):.3f}\n")
        lines.append(f"- Mean min TTC: {np.mean([e.ttc_min for e in collision_eps]):.3f}s\n")
        lines.append(f"- Mean shield rate: {np.mean([e.shield_intervention_rate for e in collision_eps])*100:.1f}%\n")
    
    if timeout > 0:
        timeout_eps = [e for e in episodes if e.reason == "timeout"]
        lines.append(f"\n### Timeout episodes (n={timeout})\n")
        lines.append(f"- Mean return: {np.mean([e.return_sum for e in timeout_eps]):.3f}\n")
        lines.append(f"- Mean progress: {np.mean([e.progress_total for e in timeout_eps]):.3f}m\n")
        lines.append(f"- Mean shield rate: {np.mean([e.shield_intervention_rate for e in timeout_eps])*100:.1f}%\n")

    lines.append("\n## Interpretation notes\n")
    lines.append(
        "- **High success rate with low shield rate** indicates the LSTM policy learned effective avoidance.\n"
        "- **High shield rate** suggests the policy is aggressive and relies on safety corrections.\n"
        "- **Low valid obstacles mean** (<3) indicates sparse obstacle scenarios.\n"
        "- **High valid obstacles mean** (>4) indicates the LSTM is processing near-capacity obstacle loads.\n"
        "- If **collisions have low shield rate**, the shield parameters may need tuning.\n"
        "- If **timeouts have high progress but fail**, the agent may be stuck in local minima near the goal.\n"
        "- **Correlation R ↔ TTC > 0**: Safer episodes get higher rewards (good reward design).\n"
        "- **Correlation R ↔ Length < 0**: Shorter episodes get higher rewards (efficient paths rewarded).\n"
    )

    report_path = os.path.join(outdir, "report.md")
    with open(report_path, "w") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to LSTM-SAC .zip model")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--outdir", type=str, default="./eval_lstm_out")
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=12345)
    ap.add_argument("--lstm_hidden", type=int, default=64, help="LSTM hidden size (must match training)")
    ap.add_argument("--lstm_layers", type=int, default=1, help="LSTM num layers (must match training)")
    ap.add_argument("--net_arch", type=str, default="256,256", help="Network architecture (must match training)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ROS2 init (required before env construction)
    rclpy.init()

    try:
        # Create LSTM environment
        env_raw = GoalFollowerLSTMEnv(
            cmd_topic="/follower_robot/cmd_vel",
            goal_state_topic="/follower_robot/depth_cam/goal_marker_state",
            goal_odom_topic="/goal_marker/odom",
            dt=0.1,
            success_radius=1.5,
            time_limit=80.0,
            R_goal=50.0,
            R_collision=-50.0,
            gamma_shaping=0.97,
            step_penalty=0.005,
            ttc_threshold=2.0,
            cam_x=0.4,
            cam_y=0.0,
            use_ground_truth_geometry=True,
            smooth_alpha=0.70,
        )
        env = Monitor(env_raw)

        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model not found: {args.model}")

        # Parse network architecture
        net_arch = [int(x.strip()) for x in args.net_arch.split(",")]

        # Create policy class with matching architecture
        PolicyClass = create_lstm_sac_policy(
            lstm_hidden_size=args.lstm_hidden,
            lstm_num_layers=args.lstm_layers,
            net_arch=net_arch,
        )

        # Load model
        print(f"Loading LSTM-SAC model from: {args.model}")
        print(f"  LSTM hidden: {args.lstm_hidden}")
        print(f"  LSTM layers: {args.lstm_layers}")
        print(f"  Net arch: {net_arch}")
        
        model = SAC.load(args.model, env=env, device="auto", custom_objects={'policy': PolicyClass})

        deterministic = bool(args.deterministic)

        all_eps: List[EpisodeResult] = []
        all_raw: Dict[str, Any] = {"episodes": []}

        for i in range(1, args.episodes + 1):
            seed = args.seed0 + i
            print(f"\n=== Eval episode {i}/{args.episodes} (seed={seed}) ===")
            ep, raw = run_one_episode(env, model, i, seed, deterministic=deterministic)
            all_eps.append(ep)

            all_raw["episodes"].append(
                {
                    "summary": asdict(ep),
                    "raw": raw,
                }
            )

            print(
                f"[EP {i}] reason={ep.reason} return={ep.return_sum:.2f} "
                f"steps={ep.length_steps} minTTC={ep.ttc_min:.2f} "
                f"progress={ep.progress_total:.2f}m validObs={ep.num_valid_obstacles_mean:.1f} "
                f"shield={ep.shield_intervention_rate*100:.1f}%"
            )

        # Save JSON outputs
        print(f"\nSaving results to {args.outdir}...")
        
        with open(os.path.join(args.outdir, "eval_full.json"), "w") as f:
            json.dump(all_raw, f, indent=2)

        with open(os.path.join(args.outdir, "eval_episode_summaries.json"), "w") as f:
            json.dump([asdict(e) for e in all_eps], f, indent=2)

        # Save plots + report
        save_plots(args.outdir, all_eps)
        write_report(args.outdir, all_eps)

        # Print summary statistics
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        reasons = [e.reason for e in all_eps]
        success = sum(1 for r in reasons if r == "goal")
        collision = sum(1 for r in reasons if r == "collision")
        timeout = sum(1 for r in reasons if r == "timeout")
        
        print(f"Episodes: {len(all_eps)}")
        print(f"Success:   {success:3d} ({success/len(all_eps)*100:5.1f}%)")
        print(f"Collision: {collision:3d} ({collision/len(all_eps)*100:5.1f}%)")
        print(f"Timeout:   {timeout:3d} ({timeout/len(all_eps)*100:5.1f}%)")
        print(f"\nMean return: {np.mean([e.return_sum for e in all_eps]):.2f} ± {np.std([e.return_sum for e in all_eps]):.2f}")
        print(f"Mean steps:  {np.mean([e.length_steps for e in all_eps]):.1f} ± {np.std([e.length_steps for e in all_eps]):.1f}")
        print(f"Mean shield intervention: {np.mean([e.shield_intervention_rate for e in all_eps])*100:.1f}%")
        print("="*70)

        print(f"\n[DONE] Full results saved to: {os.path.abspath(args.outdir)}")
        print("\nGenerated files:")
        print(" - report.md                      (comprehensive analysis)")
        print(" - eval_episode_summaries.json    (episode-level data)")
        print(" - eval_full.json                 (timestep-level data)")
        print(" - returns.png                    (return plots)")
        print(" - lengths.png                    (episode length)")
        print(" - ttc_min.png                    (safety metrics)")
        print(" - return_vs_speed.png            (behavior analysis)")
        print(" - valid_obstacles.png            (LSTM-specific)")
        print(" - shield_rate.png                (safety shield usage)")

    finally:
        try:
            env_raw.close()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
