#!/usr/bin/env python3
"""
Animate robot trajectories from episode_stats.json.

Shows:
- Robot trajectory trail
- Robot heading arrow
- Per-frame obstacle positions (updated each frame, no trail)
- All obstacle positions across all frames (static background scatter)
- Goal position (fixed if only goal_final exists; animated if goal path exists)
- Goal path trail (if goal path exists)
"""

import argparse
import json
import os
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# Helpers
# ============================================================

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def vec2(p):
    if p is None or not isinstance(p, (list, tuple)) or len(p) < 2:
        return (np.nan, np.nan)
    return (safe_float(p[0]), safe_float(p[1]))


# ============================================================
# Robot Path Loader
# ============================================================

def load_robot_path_xy_yaw(ep: Dict[str, Any]) -> Optional[np.ndarray]:
    rp = ep.get("robot_path", None)
    if rp is None:
        rp = ep.get("robot_traj", None)

    if not isinstance(rp, list) or len(rp) < 2:
        return None

    pts = []
    for p in rp:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        x, y = safe_float(p[0]), safe_float(p[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        yaw = safe_float(p[2]) if len(p) >= 3 else np.nan
        pts.append([x, y, yaw])

    if len(pts) < 2:
        return None

    arr = np.asarray(pts, dtype=float)

    if np.any(~np.isfinite(arr[:, 2])):
        dx = np.diff(arr[:, 0])
        dy = np.diff(arr[:, 1])
        est = np.arctan2(dy, dx)
        yaws = np.full(arr.shape[0], np.nan)
        yaws[1:] = est
        if len(est) > 0:
            yaws[0] = est[0]
        arr[:, 2] = np.where(np.isfinite(arr[:, 2]), arr[:, 2], yaws)

    return arr


# ============================================================
# Goal Loaders (NEW)
# ============================================================

def _parse_xy_list(obj: Any) -> Optional[np.ndarray]:
    """
    Accepts:
      - list of [x,y] or [x,y,...]
      - list of dicts containing x/y (rare)
    Returns Nx2 float array or None.
    """
    if not isinstance(obj, list) or len(obj) < 1:
        return None

    pts: List[Tuple[float, float]] = []
    for p in obj:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = safe_float(p[0]), safe_float(p[1])
        elif isinstance(p, dict):
            # try common dict forms
            x = safe_float(p.get("x", np.nan))
            y = safe_float(p.get("y", np.nan))
            if not (np.isfinite(x) and np.isfinite(y)):
                # maybe {"pos":[x,y]} etc.
                if "pos" in p and isinstance(p["pos"], (list, tuple)) and len(p["pos"]) >= 2:
                    x, y = safe_float(p["pos"][0]), safe_float(p["pos"][1])
        else:
            continue

        if np.isfinite(x) and np.isfinite(y):
            pts.append((x, y))

    if len(pts) < 1:
        return None
    return np.asarray(pts, dtype=float)


def load_goal_path(ep: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Tries to find a goal trajectory/history inside the episode dict.
    If found, returns Nx2 array. Else returns None.
    """
    candidate_keys = [
        "goal_path", "goal_traj", "goal_history", "goal_positions",
        "goal_xy", "goal_per_frame", "goal_track"
    ]

    for k in candidate_keys:
        if k in ep:
            arr = _parse_xy_list(ep.get(k))
            if arr is not None and arr.shape[0] >= 1:
                return arr

    # Sometimes it's nested, e.g. ep["info"]["goal_path"]
    info = ep.get("info", None)
    if isinstance(info, dict):
        for k in candidate_keys:
            if k in info:
                arr = _parse_xy_list(info.get(k))
                if arr is not None and arr.shape[0] >= 1:
                    return arr

    return None


# ============================================================
# Obstacle Loaders
# ============================================================

def load_obstacles_per_frame(ep: Dict[str, Any]) -> List[np.ndarray]:
    """
    Returns a list (one entry per frame) of Nx2 arrays with obstacle positions.
    Frames with no obstacles get an empty array.
    """
    gt = ep.get("gt_obstacles_path", None)
    frames = []

    if isinstance(gt, list):
        for frame in gt:
            pts = []
            if isinstance(frame, dict):
                for v in frame.values():
                    if v is None:
                        continue
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        x, y = safe_float(v[0]), safe_float(v[1])
                        if np.isfinite(x) and np.isfinite(y):
                            pts.append([x, y])
            frames.append(np.asarray(pts, dtype=float) if pts else np.zeros((0, 2)))

    return frames


def load_all_obstacle_positions(ep: Dict[str, Any]) -> np.ndarray:
    """
    Returns all obstacle positions across every frame as a single Nx2 array.
    """
    gt = ep.get("gt_obstacles_path", None)
    pts = []

    if isinstance(gt, list):
        for frame in gt:
            if isinstance(frame, dict):
                for v in frame.values():
                    if v is None:
                        continue
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        x, y = safe_float(v[0]), safe_float(v[1])
                        if np.isfinite(x) and np.isfinite(y):
                            pts.append([x, y])

    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2))


# ============================================================
# Utility
# ============================================================

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def set_equal_axes(ax, pts: np.ndarray, padding: float = 1.5):
    if pts.size == 0:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal", adjustable="box")
        return

    xmin, ymin = np.min(pts[:, 0]), np.min(pts[:, 1])
    xmax, ymax = np.max(pts[:, 0]), np.max(pts[:, 1])

    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin, 1.0) + 2 * padding

    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_aspect("equal", adjustable="box")


# ============================================================
# Animation
# ============================================================

def animate_episode(ep: Dict[str, Any], outdir: str, fps: int = 20):

    ep_id = int(safe_float(ep.get("episode"), -1))
    reason = str(ep.get("reason", ""))
    reward = safe_float(ep.get("reward"), 0.0)

    robot_xyz = load_robot_path_xy_yaw(ep)
    if robot_xyz is None:
        print(f"[skip] Episode {ep_id}: no trajectory")
        return

    robot_xy  = robot_xyz[:, :2]
    robot_yaw = robot_xyz[:, 2]
    n_robot   = robot_xy.shape[0]

    # Per-frame obstacles – align length to robot path length
    obs_per_frame = load_obstacles_per_frame(ep)
    n_obs_frames  = len(obs_per_frame)

    def get_obs_frame(i: int) -> np.ndarray:
        """Map robot frame index to closest obstacle frame."""
        if n_obs_frames == 0:
            return np.zeros((0, 2))
        idx = min(int(round(i * n_obs_frames / n_robot)), n_obs_frames - 1)
        return obs_per_frame[idx]

    # All obstacle positions (static background)
    all_obs = load_all_obstacle_positions(ep)

    # Goal: either animated (path exists) or fixed (goal_final)
    goal_path = load_goal_path(ep)  # Nx2 or None
    n_goal = int(goal_path.shape[0]) if isinstance(goal_path, np.ndarray) else 0

    def get_goal_xy(i: int) -> Tuple[float, float]:
        """Map robot frame index to closest goal frame if goal path exists."""
        if n_goal <= 0:
            gf = vec2(ep.get("goal_final"))
            return gf[0], gf[1]
        idx = min(int(round(i * n_goal / n_robot)), n_goal - 1)
        return float(goal_path[idx, 0]), float(goal_path[idx, 1])

    # For bounds: use goal_path if present; else goal_final
    goal_final = vec2(ep.get("goal_final"))
    goal_for_bounds = None
    if n_goal > 0:
        goal_for_bounds = goal_path
    elif np.isfinite(goal_final[0]) and np.isfinite(goal_final[1]):
        goal_for_bounds = np.array([[goal_final[0], goal_final[1]]], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Episode {ep_id} | R={reward:.2f} | {reason}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Compute bounding box from all relevant points
    bound_pts = [robot_xy]
    if all_obs.size:
        bound_pts.append(all_obs)
    if isinstance(goal_for_bounds, np.ndarray) and goal_for_bounds.size:
        bound_pts.append(goal_for_bounds)
    set_equal_axes(ax, np.vstack(bound_pts))

    # --- Static layers ---

    # All obstacle positions across entire episode (background reference)
    if all_obs.size:
        ax.scatter(all_obs[:, 0], all_obs[:, 1],
                   s=15, c="lightcoral", alpha=0.25, zorder=1,
                   label="All obstacle positions")

    # --- Goal static reference (optional): show final goal faintly even when animated ---
    if np.isfinite(goal_final[0]) and np.isfinite(goal_final[1]):
        ax.scatter([goal_final[0]], [goal_final[1]],
                   s=130, marker="*", c="gold", edgecolors="k",
                   linewidths=0.8, alpha=0.35,
                   label="Goal (final)", zorder=4)

    # --- Animated layers ---

    trail_line, = ax.plot([], [], lw=2, color="royalblue", label="Robot trail", zorder=3)
    head_line,  = ax.plot([], [], lw=2.5, color="navy", zorder=4)

    # Current-frame obstacles as a scatter
    live_obs_scatter = ax.scatter([], [], s=80, c="red", marker="o",
                                  edgecolors="darkred", linewidths=0.5,
                                  label="Obstacles (current frame)", zorder=2)

    # Robot marker
    robot_dot, = ax.plot([], [], "o", color="navy", ms=6, zorder=6)

    # Goal animated marker + goal trail (NEW)
    goal_trail_line, = ax.plot([], [], lw=2, linestyle="--",
                               color="goldenrod", label="Goal trail", zorder=3)
    goal_dot = ax.scatter([], [], s=200, marker="*",
                          c="gold", edgecolors="k", linewidths=0.8,
                          label="Goal (current)", zorder=7)

    ax.legend(loc="upper right", fontsize=8)

    def init():
        trail_line.set_data([], [])
        head_line.set_data([], [])
        robot_dot.set_data([], [])
        live_obs_scatter.set_offsets(np.zeros((0, 2)))

        goal_trail_line.set_data([], [])
        goal_dot.set_offsets(np.zeros((0, 2)))

        return (trail_line, head_line, robot_dot, live_obs_scatter,
                goal_trail_line, goal_dot)

    def update(i):
        # Robot trail
        trail_line.set_data(robot_xy[:i+1, 0], robot_xy[:i+1, 1])

        # Robot position & heading
        x, y   = robot_xy[i]
        yaw    = robot_yaw[i]
        hx     = x + 0.8 * math.cos(yaw)
        hy     = y + 0.8 * math.sin(yaw)
        head_line.set_data([x, hx], [y, hy])
        robot_dot.set_data([x], [y])

        # Current-frame obstacles
        obs = get_obs_frame(i)
        if obs.size:
            live_obs_scatter.set_offsets(obs)
        else:
            live_obs_scatter.set_offsets(np.zeros((0, 2)))

        # Goal current + goal trail
        gx, gy = get_goal_xy(i)
        if np.isfinite(gx) and np.isfinite(gy):
            goal_dot.set_offsets(np.array([[gx, gy]], dtype=float))
        else:
            goal_dot.set_offsets(np.zeros((0, 2)))

        if n_goal > 0:
            # map i to goal idx; show trail up to that idx
            gi = min(int(round(i * n_goal / n_robot)), n_goal - 1)
            goal_trail_line.set_data(goal_path[:gi+1, 0], goal_path[:gi+1, 1])
        else:
            goal_trail_line.set_data([], [])

        return (trail_line, head_line, robot_dot, live_obs_scatter,
                goal_trail_line, goal_dot)

    anim = FuncAnimation(fig, update, frames=n_robot,
                         init_func=init, blit=True)

    ensure_dir(outdir)
    base = os.path.join(outdir, f"episode_{ep_id:06d}")

    # Always try MP4 first, fall back to GIF
    try:
        path = base + ".mp4"
        anim.save(path, fps=fps, dpi=150)
        print(f"[ok] saved {path}")
    except Exception:
        print("[warn] ffmpeg not available, saving GIF instead.")
        path = base + ".gif"
        anim.save(path, writer="pillow", fps=fps)
        print(f"[ok] saved {path}")

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Animate robot episodes from episode_stats.json")
    ap.add_argument("--json", required=True,
                    help="Path to episode_stats.json")
    ap.add_argument("--out",  default="anim_out",
                    help="Output directory for animations")
    ap.add_argument("--step", type=int, default=100,
                    help="Animate every Nth episode (default: 100)")
    ap.add_argument("--episode", type=int, default=None,
                    help="Animate a single specific episode number")
    ap.add_argument("--fps",  type=int, default=20,
                    help="Frames per second (default: 20)")
    args = ap.parse_args()

    with open(args.json, "r") as f:
        data = json.load(f)

    ensure_dir(args.out)
    episodes = sorted(data, key=lambda e: e.get("episode", 0))

    for ep in episodes:
        ep_id = int(safe_float(ep.get("episode"), -1))

        if args.episode is not None:
            if ep_id != args.episode:
                continue
        else:
            if ep_id % args.step != 0:
                continue

        animate_episode(ep, args.out, fps=args.fps)


if __name__ == "__main__":
    main()