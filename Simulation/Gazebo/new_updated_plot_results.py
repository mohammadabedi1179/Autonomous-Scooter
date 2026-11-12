"""
Training Analysis Plotter — Scientific Edition
Clean, publication-oriented theme for a complete visualization suite.

Reads episode_stats.json and generates the full analysis suite.

Usage:
  python training_analysis_scientific.py --json episode_stats.json --out plots_out --rolling 20 --topk 12
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════
# SCIENTIFIC THEME CONFIGURATION
# ═══════════════════════════════════════════
BG        = "#ffffff"      # White background
BG_AXES   = "#ffffff"      # White axes
GRID_CLR  = "#d0d0d0"      # Light gray grid
TEXT_CLR  = "#000000"      # Black text
TICK_CLR  = "#000000"      # Black ticks
EDGE_CLR  = "#808080"      # Gray edges
PRIMARY   = "#1f77b4"      # Blue
SECONDARY = "#ff7f0e"      # Orange
ACCENT    = "#2ca02c"      # Green
PAPER_WHITE = "#ffffff"

# Reason / outcome palette
REASON_COLORS = {
    "goal":      "#2ca02c",   # Green
    "collision": "#d62728",   # Red
    "timeout":   "#9467bd",   # Purple
}

SHIELD_COLORS = {
    "none":     "#2ca02c",    # Green
    "caution":  "#ff7f0e",    # Orange
    "danger":   "#d62728",    # Red
    "critical": "#9467bd",    # Purple
}

LOSS_COLORS = {
    "actor":  "#1f77b4",      # Blue
    "critic": "#ff7f0e",      # Orange
}


def _apply_scientific_rc():
    """Set global rcParams for scientific theme."""
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG_AXES,
        "axes.edgecolor":    EDGE_CLR,
        "axes.grid":         True,
        "grid.color":        GRID_CLR,
        "grid.alpha":        0.7,
        "grid.linewidth":    0.6,
        "grid.linestyle":    "--",
        "axes.labelcolor":   TEXT_CLR,
        "text.color":        TEXT_CLR,
        "xtick.color":       TICK_CLR,
        "ytick.color":       TICK_CLR,
        "axes.spines.top":   True,
        "axes.spines.right": True,
        "axes.linewidth":    1.0,
        "font.size":         11,
        "font.family":       "sans-serif",
        "legend.facecolor":  PAPER_WHITE,
        "legend.edgecolor":  EDGE_CLR,
        "legend.labelcolor": TEXT_CLR,
        "legend.framealpha": 0.95,
        "savefig.facecolor": BG,
        "savefig.dpi":       150,
    })


def reason_color(r):
    return REASON_COLORS.get(r, PRIMARY)


def reason_palette(reasons):
    return {r: reason_color(r) for r in sorted(set(reasons))}


# ─────────────────────────────
# Plotting helpers
# ─────────────────────────────
def line_with_shadow(ax, x, y, color, lw=2.0, label=None, alpha=0.9):
    """Draw a line with a faint shadow for better readability."""
    ax.plot(x, y, linewidth=lw + 1.0, color="#bbbbbb", alpha=0.3, solid_capstyle="round")
    ax.plot(x, y, linewidth=lw, color=color, alpha=alpha, solid_capstyle="round", label=label)


def scatter_points(ax, x, y, color, s=20, label=None, alpha=0.8):
    """Standard scatter plot with clear edges."""
    ax.scatter(x, y, s=s, alpha=alpha, color=color,
               edgecolors=PAPER_WHITE, linewidths=0.8, label=label)


def reference_hline(ax, y, color, label=None, ls="-"):
    """Horizontal reference line."""
    ax.axhline(y=y, color=color, linewidth=2.0, alpha=0.3, linestyle=ls)
    ax.axhline(y=y, color=color, linewidth=1.0, alpha=0.8, linestyle=ls, label=label)


def add_legend(ax, loc="best", fontsize=9):
    """Standard legend formatting."""
    legend = ax.legend(loc=loc, fontsize=fontsize, facecolor=PAPER_WHITE,
                       edgecolor=EDGE_CLR, labelcolor=TEXT_CLR)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_linewidth(1.0)


def set_title(ax, title):
    """Standard title formatting."""
    ax.set_title(title, color=TEXT_CLR, fontsize=13, fontweight="bold", pad=12)


def scientific_boxplot(ax, data, tick_labels, colors):
    """Boxplot with publication-style aesthetic."""
    bp = ax.boxplot(
        data, labels=tick_labels, patch_artist=True, showfliers=True,
        flierprops=dict(marker="o", markersize=4, markerfacecolor=ACCENT,
                        markeredgecolor=PAPER_WHITE, markeredgewidth=0.8),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color=TEXT_CLR, linewidth=1.0),
        capprops=dict(color=TEXT_CLR, linewidth=1.0),
        boxprops=dict(linewidth=1.0)
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    return bp


# ─────────────────────────────
# Helper functions
# ─────────────────────────────
def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def rolling_mean(x, w):
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) == 0:
        return x
    out = np.full_like(x, np.nan, dtype=float)
    c = np.cumsum(np.where(np.isfinite(x), x, 0.0))
    n = np.cumsum(np.isfinite(x).astype(float))
    for i in range(len(x)):
        j0 = max(0, i - w + 1)
        s = c[i] - (c[j0 - 1] if j0 > 0 else 0.0)
        k = n[i] - (n[j0 - 1] if j0 > 0 else 0.0)
        out[i] = (s / k) if k > 0 else np.nan
    return out


def rolling_fraction(labels, w):
    labels = list(labels)
    uniq = sorted(set(l for l in labels if l is not None))
    n = len(labels)
    out = {u: np.full(n, np.nan, dtype=float) for u in uniq}
    for i in range(n):
        j0 = max(0, i - w + 1)
        window = labels[j0: i + 1]
        total = len(window)
        if total == 0:
            continue
        cnt = Counter(window)
        for u in uniq:
            out[u][i] = cnt.get(u, 0) / total
    return out


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def vec2(p):
    if p is None or len(p) < 2:
        return (np.nan, np.nan)
    return (safe_float(p[0]), safe_float(p[1]))


def dist2(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return math.sqrt(dx**2 + dy**2) if np.isfinite(dx) and np.isfinite(dy) else np.nan


def savefig(fname):
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  saved: {fname}")


# ═══════════════════════════════════════════
# PLOT FUNCTIONS
# ═══════════════════════════════════════════

def plot_reward(episodes, stage_transitions, outdir, w):
    """Episode reward over training."""
    ep = [e["episode"] for e in episodes]
    rew = [safe_float(e.get("reward")) for e in episodes]
    if all(np.isnan(r) for r in rew):
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Individual rewards
    scatter_points(ax, ep, rew, PRIMARY, s=16, alpha=0.5)

    # Rolling mean
    line_with_shadow(ax, ep, rolling_mean(rew, w), PRIMARY, lw=2.5,
                     label=f"Rolling mean (window={w})")

    # Reference lines
    reference_hline(ax, 0.0, EDGE_CLR, label="Reward = 0")
    if any(r > 0 for r in rew if np.isfinite(r)):
        reference_hline(ax, np.nanmax(rew), REASON_COLORS["goal"], label="Max reward")
    for transition_ep in stage_transitions:
        ax.axvline(x=transition_ep, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    set_title(ax, "Reward Progress Over Training")
    add_legend(ax)
    savefig(os.path.join(outdir, "01_reward.png"))


def plot_success_rate(episodes, stage_transitions, outdir, w):
    """Success rate (fraction of 'goal' outcomes) over training."""
    ep = [e["episode"] for e in episodes]
    reasons = [str(e.get("reason", "unknown")) for e in episodes]
    if not reasons:
        return

    success = [1 if r == "goal" else 0 for r in reasons]
    fig, ax = plt.subplots(figsize=(12, 6))

    # Rolling success rate
    frac = rolling_fraction(success, w)[1]
    line_with_shadow(ax, ep, frac, REASON_COLORS["goal"],
                     lw=2.5, label="Success rate (rolling)")

    # Individual successful episodes (at y=1.02)
    success_eps = [e for e, s in zip(ep, success) if s]
    if success_eps:
        scatter_points(ax, success_eps, [1.02] * len(success_eps),
                       REASON_COLORS["goal"], s=25, alpha=0.7,
                       label="Successful episodes")
    for transition_ep in stage_transitions:
        ax.axvline(x=transition_ep, color='red', linestyle='--', alpha=0.5, linewidth=1)                   

    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.05, 1.1)
    set_title(ax, "Success Rate Over Training")
    add_legend(ax)
    savefig(os.path.join(outdir, "02_success_rate.png"))


def plot_length(episodes, stage_transitions, outdir, w):
    """Episode length and wall-clock time."""
    ep = [e["episode"] for e in episodes]
    lengths = [safe_float(e.get("length")) for e in episodes]

    if not any(np.isfinite(l) for l in lengths):
        return

    fig, (ax) = plt.subplots(figsize=(12, 10))

    # Episode length (steps)
    if any(np.isfinite(l) for l in lengths):
        scatter_points(ax, ep, lengths, SECONDARY, s=12, alpha=0.4)
        line_with_shadow(ax, ep, rolling_mean(lengths, w), SECONDARY, lw=2.5,
                         label=f"Rolling mean (window={w})")
        set_title(ax, "Episode Length Over Training")
        add_legend(ax)
        for transition_ep in stage_transitions:
            ax.axvline(x=transition_ep, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Length")
        savefig(os.path.join(outdir, "03_episode_length_time.png"))


def plot_reason_distribution(episodes, stage_transitions, outdir, w):
    """Rolling distribution of termination reasons."""
    ep = [e["episode"] for e in episodes]
    reasons = [str(e.get("reason", "unknown")) for e in episodes]
    if not reasons:
        return

    fractions = rolling_fraction(reasons, w)
    uniq = sorted(fractions.keys())

    fig, ax = plt.subplots(figsize=(12, 7))

    # Stacked area plot
    bottom = np.zeros(len(ep))
    for r in uniq:
        color = reason_color(r)
        ax.fill_between(ep, bottom, bottom + fractions[r],
                        alpha=0.7, color=color,
                        label=f"Reason: {r}", linewidth=0)
        bottom += fractions[r]
    
    for transition_ep in stage_transitions:
        ax.axvline(x=transition_ep, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction of episodes (rolling)")
    ax.set_ylim(0, 1)
    set_title(ax, "Termination Reason Distribution Over Training")
    add_legend(ax)
    savefig(os.path.join(outdir, "04_reason_distribution.png"))

def plot_entropy_learning_rate(episodes, stage_transitions, outdir, w):
    """learning rate and entropy training."""
    ep = [e["episode"] for e in episodes]
    learning_rate = [safe_float(e.get("learning_rate")) for e in episodes]
    entropy = [safe_float(e.get("ent_coef")) for e in episodes]

    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Learning Rate
    if any(np.isfinite(a) for a in learning_rate):
        scatter_points(axes[0], ep, learning_rate, LOSS_COLORS["actor"], s=8, alpha=0.3)
        line_with_shadow(axes[0], ep, rolling_mean(learning_rate, w),
                         LOSS_COLORS["actor"], lw=2.5,
                         label="Learning Rate (rolling)")
        axes[0].set_ylabel("Learning Rate")
        set_title(axes[0], "Learning Rate Over Training")
        add_legend(axes[0])

    # Entropy
    if any(np.isfinite(e_val) for e_val in entropy):
        scatter_points(axes[1], ep, entropy, PRIMARY, s=8, alpha=0.3)
        line_with_shadow(axes[1], ep, rolling_mean(entropy, w),
                         PRIMARY, lw=2.5,
                         label="Entropy (rolling)")
        axes[1].set_ylabel("Entropy")
        set_title(axes[1], "Policy Entropy Over Training")
        add_legend(axes[1])
    for transition_ep in stage_transitions:
        for item in enumerate(axes):
            axes[item[0]].axvline(x=transition_ep, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].set_xlabel("Episode")
    savefig(os.path.join(outdir, "06_learning_rate_entropy.png"))


def plot_losses_and_entropy(episodes, stage_transitions, outdir, w):
    """Actor loss and critic loss over training."""
    ep = [e["episode"] for e in episodes]
    actor_loss = [safe_float(e.get("actor_loss")) for e in episodes]
    critic_loss = [safe_float(e.get("critic_loss")) for e in episodes]

    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Actor loss
    if any(np.isfinite(a) for a in actor_loss):
        scatter_points(axes[0], ep, actor_loss, LOSS_COLORS["actor"], s=8, alpha=0.3)
        line_with_shadow(axes[0], ep, rolling_mean(actor_loss, w),
                         LOSS_COLORS["actor"], lw=2.5,
                         label="Actor loss (rolling)")
        axes[0].set_ylabel("Actor loss")
        set_title(axes[0], "Actor Loss Over Training")
        add_legend(axes[0])

    # Critic loss
    if any(np.isfinite(c) for c in critic_loss):
        scatter_points(axes[1], ep, critic_loss, LOSS_COLORS["critic"], s=8, alpha=0.3)
        line_with_shadow(axes[1], ep, rolling_mean(critic_loss, w),
                         LOSS_COLORS["critic"], lw=2.5,
                         label="Critic loss (rolling)")
        axes[1].set_ylabel("Critic loss")
        set_title(axes[1], "Critic Loss Over Training")
        add_legend(axes[1])
    
    for transition_ep in stage_transitions:
        for item in enumerate(axes):
            axes[item[0]].axvline(x=transition_ep, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].set_xlabel("Episode")
    savefig(os.path.join(outdir, "05_losses_entropy.png"))


def plot_shield_stats(episodes, stage_transitions, outdir, w):
    """Shield interventions and activations per episode."""
    ep = [e["episode"] for e in episodes]
    interventions = [safe_float(e.get("shield_interventions_ep", 0)) for e in episodes]
    lengths = [safe_float(e.get("length")) for e in episodes]
    fractions = np.divide(np.array(interventions), np.array(lengths))


    if not any(np.isfinite(i) for i in interventions):
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Shield interventions
    if any(np.isfinite(i) for i in interventions):
        scatter_points(axes[0], ep, interventions, REASON_COLORS["collision"], s=12, alpha=0.4)
        line_with_shadow(axes[0], ep, rolling_mean(interventions, w),
                         REASON_COLORS["collision"], lw=2.5,
                         label=f"Interventions (rolling, w={w})")
        axes[0].set_ylabel("Shield interventions / episode")
        set_title(axes[0], "Shield Interventions Over Training")
        add_legend(axes[0])
    if any(np.isfinite(c) for c in fractions):
        scatter_points(axes[1], ep, fractions, LOSS_COLORS["critic"], s=8, alpha=0.3)
        line_with_shadow(axes[1], ep, rolling_mean(fractions, w),
                         LOSS_COLORS["critic"], lw=2.5,
                         label="Fraction of number of interventions over episode length per episode (rolling)")
        axes[1].set_ylabel("Fraction of number of interventions over episode length")
        set_title(axes[1], "Fraction of number of interventions over episode length per episode")
        add_legend(axes[1])
    for transition_ep in stage_transitions:
        axes[0].axvline(x=transition_ep, color='red', linestyle='--', alpha=0.5, linewidth=1)
        axes[1].axvline(x=transition_ep, color='red', linestyle='--', alpha=0.5, linewidth=1)

    axes[1].set_xlabel("Episode")
    savefig(os.path.join(outdir, "07_shield_stats.png"))


def pick_representative_episodes(episodes, topk=12):
    """Select episodes for detailed trajectory analysis."""
    if not episodes:
        return []
    by_reason = defaultdict(list)
    for e in episodes:
        by_reason[str(e.get("reason", "unknown"))].append(e)

    def sort_rew(es, rev=True):
        return sorted(es, key=lambda x: safe_float(x.get("reward"), -np.inf), reverse=rev)

    picks = []
    if "goal" in by_reason:
        picks += sort_rew(by_reason["goal"])[:max(1, topk // 3)]
    if "collision" in by_reason:
        picks += sort_rew(by_reason["collision"], rev=False)[:max(1, topk // 3)]
    if "timeout" in by_reason:
        picks += sort_rew(by_reason["timeout"])[:max(1, topk // 3)]

    if len(picks) < topk:
        rem = sorted([e for e in episodes if e not in picks],
                     key=lambda x: safe_float(x.get("reward"), 0.0))
        if rem:
            idxs = np.linspace(0, len(rem) - 1,
                               num=min(topk - len(picks), len(rem)), dtype=int)
            picks += [rem[i] for i in idxs]

    seen, uniq = set(), []
    for e in picks:
        eid = e.get("episode")
        if eid in seen:
            continue
        seen.add(eid)
        uniq.append(e)
        if len(uniq) >= topk:
            break
    return uniq


def plot_obstacle_world_overlay(episodes, outdir, topk=12):
    """Obstacle map with start/goal/obstacle locations for representative episodes."""
    reps = pick_representative_episodes(episodes, topk=topk)
    if not reps:
        return
    for e in reps:
        ep = e.get("episode")
        reason = str(e.get("reason", "unknown"))
        rsx, rsy = vec2(e.get("robot_start"))
        rfx, rfy = vec2(e.get("robot_final"))
        gsx, gsy = vec2(e.get("goal_start"))
        gfx, gfy = vec2(e.get("goal_final"))

        fig, ax = plt.subplots(figsize=(8, 8))
        if np.isfinite(rsx):
            scatter_points(ax, [rsx], [rsy], PRIMARY, s=120, label="Robot start")
        if np.isfinite(rfx):
            ax.scatter([rfx], [rfy], marker="x", s=120, color=PRIMARY,
                       linewidths=3, label="Robot end")
        if np.isfinite(gsx):
            ax.scatter([gsx], [gsy], marker="^", s=120, color=REASON_COLORS["goal"],
                       label="Goal start")
        if np.isfinite(gfx):
            ax.scatter([gfx], [gfy], marker="s", s=120, color=REASON_COLORS["goal"],
                       label="Goal end")

        # Obstacles
        slots = e.get("obstacles_world_slots", [])
        ox, oy = [], []
        for s in slots:
            if isinstance(s, dict) and s.get("valid", True):
                wx, wy = safe_float(s.get("wx")), safe_float(s.get("wy"))
                if np.isfinite(wx) and np.isfinite(wy):
                    ox.append(wx)
                    oy.append(wy)
        if ox:
            ax.scatter(ox, oy, s=140, marker="D", color=REASON_COLORS["collision"],
                       label="Obstacles", alpha=0.8)
            for x, y in zip(ox, oy):
                circle = plt.Circle((x, y), 0.7, color=REASON_COLORS["collision"],
                                    fill=False, linestyle="--", alpha=0.4, linewidth=2)
                ax.add_patch(circle)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        set_title(ax, f"Episode {ep}: {reason}")
        add_legend(ax)
        savefig(os.path.join(outdir, f"15_world_overlay_episode_{ep:05d}.png"))


def plot_trajectory_story(episodes, outdir, topk=12):
    """Representative robot and goal trajectories with obstacles."""
    candidates = [e for e in episodes
                  if isinstance(e.get("robot_path"), list) and len(e.get("robot_path")) > 1]
    if not candidates:
        return
    reps = pick_representative_episodes(candidates, topk=topk)
    for e in reps:
        ep = e.get("episode")
        reason = str(e.get("reason", "unknown"))
        shield_count = safe_float(e.get("shield_interventions_ep", 0), 0)
        rp = e.get("robot_path", [])
        rxy = np.array([[safe_float(p[0]), safe_float(p[1])]
                        for p in rp
                        if isinstance(p, (list, tuple)) and len(p) >= 2], dtype=float)
        if rxy.shape[0] < 2:
            continue

        fig, ax = plt.subplots(figsize=(9, 9))
        # Robot path
        ax.plot(rxy[:, 0], rxy[:, 1], linewidth=4, color=PRIMARY, alpha=0.3)
        ax.plot(rxy[:, 0], rxy[:, 1], linewidth=2.5, color=PRIMARY, alpha=0.8,
                label="Robot path")
        ax.scatter([rxy[0, 0]], [rxy[0, 1]], marker="o", s=140, color=PRIMARY,
                   edgecolors=PAPER_WHITE, linewidths=2)
        ax.scatter([rxy[-1, 0]], [rxy[-1, 1]], marker="x", s=140, color=PRIMARY,
                   linewidths=4)

        # Goal path (if available)
        gp = e.get("goal_path", None)
        if isinstance(gp, list) and len(gp) > 1:
            gxy = np.array([[safe_float(p[0]), safe_float(p[1])]
                            for p in gp
                            if isinstance(p, (list, tuple)) and len(p) >= 2], dtype=float)
            if gxy.shape[0] >= 2:
                ax.plot(gxy[:, 0], gxy[:, 1], linewidth=3,
                        color=REASON_COLORS["goal"], alpha=0.7,
                        label="Goal path")
                ax.scatter([gxy[0, 0]], [gxy[0, 1]], marker="^", s=140,
                           color=REASON_COLORS["goal"])

        # Obstacles
        slots = e.get("obstacles_world_slots", [])
        ox, oy = [], []
        for s in slots:
            if isinstance(s, dict) and s.get("valid", True):
                wx, wy = safe_float(s.get("wx")), safe_float(s.get("wy"))
                if np.isfinite(wx) and np.isfinite(wy):
                    ox.append(wx)
                    oy.append(wy)
        if ox:
            ax.scatter(ox, oy, s=160, marker="D", color=REASON_COLORS["collision"],
                       label="Obstacles", alpha=0.8)
            for x, y in zip(ox, oy):
                circle = plt.Circle((x, y), 0.7, color=REASON_COLORS["collision"],
                                    fill=False, linestyle="--", alpha=0.4, linewidth=2)
                ax.add_patch(circle)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        set_title(ax, f"Episode {ep}: {reason} | Shield interventions: {int(shield_count)}")
        add_legend(ax)
        savefig(os.path.join(outdir, f"16_trajectory_episode_{ep:05d}.png"))

# === ADD THESE FUNCTIONS BELOW YOUR EXISTING PLOT FUNCTIONS ===

def plot_mean_jerk(episodes, stage_transitions, outdir, w):
    ep = [e["episode"] for e in episodes]
    jerk_v = [safe_float(e.get("ep_mean_jerk_v")) for e in episodes]
    jerk_w = [safe_float(e.get("ep_mean_jerk_w")) for e in episodes]

    if not any(np.isfinite(j) for j in jerk_v + jerk_w):
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    scatter_points(ax, ep, jerk_v, PRIMARY, s=8, alpha=0.3)
    line_with_shadow(ax, ep, rolling_mean(jerk_v, w), PRIMARY,
                     label="Mean jerk v (rolling)")

    scatter_points(ax, ep, jerk_w, SECONDARY, s=8, alpha=0.3)
    line_with_shadow(ax, ep, rolling_mean(jerk_w, w), SECONDARY,
                     label="Mean jerk w (rolling)")

    for t in stage_transitions:
        ax.axvline(x=t, color='red', linestyle='--', alpha=0.4)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Jerk")
    set_title(ax, "Mean Jerk (v & w)")
    add_legend(ax)

    savefig(os.path.join(outdir, "08_mean_jerk.png"))


def plot_deadlock_escapes(episodes, stage_transitions, outdir, max_bars=400):
    ep = [e["episode"] for e in episodes]
    deadlocks = [safe_float(e.get("deadlock_escapes_ep", 0)) for e in episodes]

    if not any(np.isfinite(d) for d in deadlocks):
        return

    # limit for readability
    ep = ep[-max_bars:]
    deadlocks = deadlocks[-max_bars:]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(ep, deadlocks, color=ACCENT, alpha=0.8)

    for t in stage_transitions:
        if t >= ep[0]:
            ax.axvline(x=t, color='red', linestyle='--', alpha=0.4)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Deadlock Escapes")
    set_title(ax, f"Deadlock Escapes (last {max_bars} episodes)")

    savefig(os.path.join(outdir, "09_deadlock_escapes.png"))


def plot_min_distances(episodes, stage_transitions, outdir, w):
    ep = [e["episode"] for e in episodes]

    ultra = [safe_float(e.get("ep_min_ultrasonic_distance")) for e in episodes]
    depth = [safe_float(e.get("ep_min_obstacle_depth")) for e in episodes]

    if not any(np.isfinite(x) for x in ultra + depth):
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    scatter_points(ax, ep, ultra, PRIMARY, s=8, alpha=0.3)
    line_with_shadow(ax, ep, rolling_mean(ultra, w), PRIMARY,
                     label="Min Ultrasonic (rolling)")

    scatter_points(ax, ep, depth, SECONDARY, s=8, alpha=0.3)
    line_with_shadow(ax, ep, rolling_mean(depth, w), SECONDARY,
                     label="Min Obstacle Depth (rolling)")

    for t in stage_transitions:
        ax.axvline(x=t, color='red', linestyle='--', alpha=0.4)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Distance")
    set_title(ax, "Minimum Distances per Episode")
    add_legend(ax)

    savefig(os.path.join(outdir, "10_min_distances.png"))


def plot_shield_stacked(episodes, stage_transitions, outdir, max_bars=400):
    ep = [e["episode"] for e in episodes]

    caution = [safe_float(e.get("shield_interventions_caution_ep", 0)) for e in episodes]
    danger = [safe_float(e.get("shield_interventions_danger_ep", 0)) for e in episodes]
    critical = [safe_float(e.get("shield_interventions_critical_ep", 0)) for e in episodes]

    if not any(np.isfinite(x) for x in caution + danger + critical):
        return

    # limit for readability
    ep = ep[-max_bars:]
    caution = caution[-max_bars:]
    danger = danger[-max_bars:]
    critical = critical[-max_bars:]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(ep, caution, color="#ffd700", label="Caution")
    ax.bar(ep, danger, bottom=caution, color="#ff7f0e", label="Danger")

    bottom_cd = np.array(caution) + np.array(danger)
    ax.bar(ep, critical, bottom=bottom_cd, color="#d62728", label="Critical")

    for t in stage_transitions:
        if t >= ep[0]:
            ax.axvline(x=t, color='black', linestyle='--', alpha=0.4)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Intervention Steps")
    set_title(ax, f"Shield Interventions (last {max_bars} episodes)")
    add_legend(ax)

    savefig(os.path.join(outdir, "11_shield_stacked.png"))

def plot_shield_fraction_area(episodes, stage_transitions, outdir, w):
    ep = [e["episode"] for e in episodes]

    caution = np.array([safe_float(e.get("shield_interventions_caution_ep", 0)) for e in episodes])
    danger = np.array([safe_float(e.get("shield_interventions_danger_ep", 0)) for e in episodes])
    critical = np.array([safe_float(e.get("shield_interventions_critical_ep", 0)) for e in episodes])

    total = caution + danger + critical

    # Avoid division by zero
    total[total == 0] = np.nan

    frac_caution = caution / total
    frac_danger = danger / total
    frac_critical = critical / total

    # Smooth
    frac_caution = rolling_mean(frac_caution, w)
    frac_danger = rolling_mean(frac_danger, w)
    frac_critical = rolling_mean(frac_critical, w)

    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(ep))

    ax.fill_between(ep, bottom, bottom + frac_caution,
                    color="#ffd700", alpha=0.8, label="Caution")
    bottom += frac_caution

    ax.fill_between(ep, bottom, bottom + frac_danger,
                    color="#ff7f0e", alpha=0.8, label="Danger")
    bottom += frac_danger

    ax.fill_between(ep, bottom, bottom + frac_critical,
                    color="#d62728", alpha=0.8, label="Critical")

    for t in stage_transitions:
        ax.axvline(x=t, color='black', linestyle='--', alpha=0.4)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction of Shield Types")
    set_title(ax, "Shield Type Distribution Over Training (Fraction)")
    add_legend(ax)

    savefig(os.path.join(outdir, "12_shield_fraction_area.png"))



# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Plot training results — Scientific Edition"
    )
    parser.add_argument("--json", required=True,
                        help="Path to episode_stats.json")
    parser.add_argument("--curr", required=True,
                        help="Path to curricullum json log file")
    parser.add_argument("--out", default="plots_scientific",
                        help="Output directory for visualizations.")
    parser.add_argument("--rolling", type=int, default=20,
                        help="Rolling window size.")
    parser.add_argument("--topk", type=int, default=12,
                        help="Number of representative episodes for detailed analysis.")
    args = parser.parse_args()

    ensure_dir(args.out)
    _apply_scientific_rc()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("JSON must be a non-empty list of episode dicts.")
    
    # Add after loading episode_stats.json
    with open(args.curr, 'r', encoding="utf-8") as f:
        curriculum_log = json.load(f)

    # Extract stage boundaries
    stage_transitions = [
        entry['total_episodes'] 
        for entry in curriculum_log 
        if entry['event'] == 'stage_advanced'
    ]

    data = sorted(data, key=lambda e: safe_float(e.get("episode"), np.inf))
    print(f"Loaded {len(data)} episodes.")
    print("Generating analysis figures (scientific theme)...")
    # After loading data, add this diagnostic section:
    print("\n=== Data Availability Check ===")
    fields_to_check = [
        'length', 'episode_time', 'min_dist_to_obstacles',
        'min_ttc', 'v_policy', 'v_cmd', 'w_policy', 'w_cmd'
    ]
    for field in fields_to_check:
        values = [safe_float(e.get(field)) for e in data]
        valid_count = sum(1 for v in values if np.isfinite(v))
        print(f"{field}: {valid_count}/{len(data)} valid values")
    print("=" * 40 + "\n")

    # Visualization suite
    plot_reward(data, stage_transitions, args.out, args.rolling)
    plot_success_rate(data, stage_transitions, args.out, args.rolling)
    plot_length(data, stage_transitions, args.out, args.rolling)
    plot_reason_distribution(data, stage_transitions, args.out, args.rolling)
    plot_losses_and_entropy(data, stage_transitions, args.out, w=args.rolling)
    plot_entropy_learning_rate(data, stage_transitions, args.out, w=args.rolling)

    # OLD shield plot (keep or remove if you want)
    plot_shield_stats(data, stage_transitions, args.out, args.rolling)

    # NEW plots
    plot_mean_jerk(data, stage_transitions, args.out, args.rolling)
    plot_deadlock_escapes(data, stage_transitions, args.out)
    plot_min_distances(data, stage_transitions, args.out, args.rolling)
    plot_shield_stacked(data, stage_transitions, args.out)
    plot_shield_fraction_area(data, stage_transitions, args.out, args.rolling)

    plot_obstacle_world_overlay(data, args.out, topk=args.topk)
    plot_trajectory_story(data, args.out, topk=args.topk)

    print(f"Analysis complete. Figures saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
