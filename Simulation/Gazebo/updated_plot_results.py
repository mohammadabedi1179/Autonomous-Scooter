"""
Training Analysis Plotter — EUCALYPTUS SPA WELLNESS JOURNEY EDITION
Calming spa-inspired theme with sage greens and natural tones for complete visualization suite

Reads episode_stats.json and generates the full analysis suite.

Usage:
  python eucalyptus_wellness_journey.py --json episode_stats.json --out plots_out --rolling 20 --topk 12
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════
# EUCALYPTUS SPA WELLNESS JOURNEY THEME CONFIG
# ═══════════════════════════════════════════
BG        = "#f7f9f8"      # Soft mint white
BG_AXES   = "#f7f9f8"      # Soft mint white
GRID_CLR  = "#d1e7dd"      # Mint green grid
TEXT_CLR   = "#1b4332"     # Deep forest green
TICK_CLR   = "#2d5a3d"     # Medium green
EDGE_CLR   = "#a7c3b3"     # Sage green
PRIMARY   = "#52b788"      # Eucalyptus green
SECONDARY = "#2d6a4f"      # Forest green
ACCENT    = "#74c69d"      # Light sage
WELLNESS_WHITE = "#ffffff" # Pure healing white

# Wellness spa inspired palette
REASON_COLORS = {
    "goal":      "#2d6a4f",   # Deep forest
    "collision": "#c77dff",   # Lavender (soft warning)
    "timeout":   "#ffb3ba",   # Soft coral
}

SHIELD_COLORS = {
    "none":     "#2d6a4f",    # Deep forest
    "caution":  "#ffb3ba",    # Soft coral
    "danger":   "#c77dff",    # Lavender
    "critical": "#a663cc",    # Deep lavender
}

LOSS_COLORS = {
    "actor":  "#52b788",      # Eucalyptus
    "critic": "#74c69d",      # Light sage
}


def _apply_wellness_rc():
    """Set global rcParams for Eucalyptus Spa Wellness Journey theme."""
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG_AXES,
        "axes.edgecolor":    EDGE_CLR,
        "axes.grid":         True,
        "grid.color":        GRID_CLR,
        "grid.alpha":        0.5,
        "grid.linewidth":    0.6,
        "grid.linestyle":    "-",
        "axes.labelcolor":   TEXT_CLR,
        "text.color":        TEXT_CLR,
        "xtick.color":       TICK_CLR,
        "ytick.color":       TICK_CLR,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
        "font.size":         11,
        "font.family":       "sans-serif",
        "legend.facecolor":  WELLNESS_WHITE,
        "legend.edgecolor":  EDGE_CLR,
        "legend.labelcolor": TEXT_CLR,
        "legend.framealpha": 0.92,
        "savefig.facecolor": BG,
        "savefig.dpi":       150,
    })


def reason_color(r):
    return REASON_COLORS.get(r, PRIMARY)


def reason_palette(reasons):
    return {r: reason_color(r) for r in sorted(set(reasons))}


# ─────────────────────────────
# Spa-inspired effect helpers
# ─────────────────────────────
def eucalyptus_line(ax, x, y, color, lw=2.2, label=None, alpha=0.85):
    """Draw a line with calming eucalyptus essence."""
    # Gentle mist effect
    ax.plot(x, y, linewidth=lw + 1.0, color=ACCENT, alpha=0.2, solid_capstyle="round")
    # Main organic line
    ax.plot(x, y, linewidth=lw, color=color, alpha=alpha, solid_capstyle="round", label=label)


def spa_scatter(ax, x, y, color, s=20, label=None, alpha=0.8):
    """Scatter with therapeutic spa stone styling."""
    # Soft therapeutic glow
    ax.scatter(x, y, s=s * 1.8, alpha=0.15, color=PRIMARY, edgecolors="none")
    # Main healing points
    ax.scatter(x, y, s=s, alpha=alpha, color=color, edgecolors=WELLNESS_WHITE, 
               linewidths=1.0, label=label)


def wellness_hline(ax, y, color, label=None):
    """Horizontal line like calm water surface."""
    ax.axhline(y=y, color=color, linewidth=2.0, alpha=0.25, linestyle="-")
    ax.axhline(y=y, color=color, linewidth=1.0, alpha=0.6, linestyle="-", label=label)


def spa_legend(ax, loc="best", fontsize=9):
    """Tranquil legend with spa styling."""
    legend = ax.legend(loc=loc, fontsize=fontsize, facecolor=WELLNESS_WHITE, 
                      edgecolor=EDGE_CLR, labelcolor=TEXT_CLR)
    legend.get_frame().set_alpha(0.92)
    legend.get_frame().set_linewidth(1.0)


def wellness_title(ax, title):
    """Peaceful title styling like spa signage."""
    ax.set_title(title, color=PRIMARY, fontsize=13, fontweight="400", 
                pad=18, family="sans-serif")


def spa_boxplot(ax, data, tick_labels, colors):
    """Styled boxplot with wellness center aesthetic."""
    bp = ax.boxplot(
        data, labels=tick_labels, patch_artist=True, showfliers=True,
        flierprops=dict(marker="o", markersize=4, markerfacecolor=ACCENT, 
                       markeredgecolor=WELLNESS_WHITE, markeredgewidth=0.8),
        medianprops=dict(color=PRIMARY, linewidth=2.5),
        whiskerprops=dict(color=TEXT_CLR, linewidth=1.2),
        capprops=dict(color=TEXT_CLR, linewidth=1.2),
        boxprops=dict(linewidth=1.2)
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.2)
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
        window = labels[j0 : i + 1]
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
    print(f"  🌿 {fname}")


# ═══════════════════════════════════════════
# WELLNESS JOURNEY PLOT FUNCTIONS
# ═══════════════════════════════════════════

def plot_reward(episodes, outdir, w):
    """Reward progress as wellness journey."""
    ep = [e["episode"] for e in episodes]
    rew = [safe_float(e.get("reward")) for e in episodes]
    if all(np.isnan(r) for r in rew): return

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Individual rewards as therapeutic stones
    spa_scatter(ax, ep, rew, PRIMARY, s=16, alpha=0.5)
    
    # Rolling mean as flowing water
    eucalyptus_line(ax, ep, rolling_mean(rew, w), PRIMARY, lw=2.5, 
                   label=f"Healing Journey ({w} sessions)")
    
    # Reference lines like zen garden lines
    wellness_hline(ax, 0.0, EDGE_CLR, label="Balance Line")
    if any(r > 0 for r in rew if np.isfinite(r)):
        wellness_hline(ax, np.nanmax(rew), REASON_COLORS["goal"], label="Peak Wellness")
    
    ax.set_xlabel("Training Session", fontweight="400")
    ax.set_ylabel("Wellness Score", fontweight="400")
    wellness_title(ax, "Wellness Journey — Reward Progress")
    spa_legend(ax)
    savefig(os.path.join(outdir, "01_wellness_reward_journey.png"))


def plot_success_rate(episodes, outdir, w):
    """Success rate as healing progress."""
    ep = [e["episode"] for e in episodes]
    reasons = [str(e.get("reason", "unknown")) for e in episodes]
    if not reasons: return

    success = [1 if r == "goal" else 0 for r in reasons]
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Success rate as crystalline line
    eucalyptus_line(ax, ep, rolling_fraction(success, w)[1], REASON_COLORS["goal"], 
                lw=2.8, label="Healing Success Rate")
    
    # Individual successes as healing moments
    success_eps = [e for e, s in zip(ep, success) if s]
    if success_eps:
        spa_scatter(ax, success_eps, [1.02] * len(success_eps), 
                   REASON_COLORS["goal"], s=25, alpha=0.7, label="Healing Achieved")
    
    ax.set_xlabel("Training Session", fontweight="400")
    ax.set_ylabel("Healing Success Rate", fontweight="400")
    ax.set_ylim(-0.05, 1.1)
    wellness_title(ax, "Healing Success Over Wellness Journey")
    spa_legend(ax)
    savefig(os.path.join(outdir, "02_wellness_success_rate.png"))


def plot_length_and_time(episodes, outdir, w):
    """Episode duration as therapy session length."""
    ep = [e["episode"] for e in episodes]
    lengths = [safe_float(e.get("episode_length")) for e in episodes]
    times = [safe_float(e.get("episode_time")) for e in episodes]
    
    if not any(np.isfinite(l) for l in lengths): return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Session duration
    if any(np.isfinite(l) for l in lengths):
        spa_scatter(ax1, ep, lengths, SECONDARY, s=12, alpha=0.4)
        eucalyptus_line(ax1, ep, rolling_mean(lengths, w), SECONDARY, lw=2.5, 
                       label=f"Session Flow ({w})")
        ax1.set_ylabel("Therapy Session Length", fontweight="400")
        wellness_title(ax1, "Therapy Session Duration")
        spa_legend(ax1)
    
    # Session time
    if any(np.isfinite(t) for t in times):
        spa_scatter(ax2, ep, times, PRIMARY, s=12, alpha=0.4)
        eucalyptus_line(ax2, ep, rolling_mean(times, w), PRIMARY, lw=2.5, 
                       label=f"Time Flow ({w})")
        ax2.set_ylabel("Session Time (minutes)", fontweight="400")
        wellness_title(ax2, "Wellness Session Duration")
        spa_legend(ax2)
    
    ax2.set_xlabel("Training Session", fontweight="400")
    savefig(os.path.join(outdir, "03_wellness_session_duration.png"))


def plot_reason_distribution(episodes, outdir, w):
    """Outcome distribution as wellness states."""
    ep = [e["episode"] for e in episodes]
    reasons = [str(e.get("reason", "unknown")) for e in episodes]
    if not reasons: return

    fractions = rolling_fraction(reasons, w)
    uniq = sorted(fractions.keys())
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Stacked area plot with wellness colors
    bottom = np.zeros(len(ep))
    for r in uniq:
        color = reason_color(r)
        ax.fill_between(ep, bottom, bottom + fractions[r], 
                       alpha=0.7, color=color, label=f"State: {r.title()}", linewidth=0)
        bottom += fractions[r]
    
    ax.set_xlabel("Training Session", fontweight="400")
    ax.set_ylabel("Wellness State Distribution", fontweight="400")
    ax.set_ylim(0, 1)
    wellness_title(ax, "Wellness State Evolution")
    spa_legend(ax)
    savefig(os.path.join(outdir, "04_wellness_state_distribution.png"))


def plot_min_dist_vs_reward(episodes, outdir):
    """Safety vs performance as balance vs growth."""
    min_dists = [safe_float(e.get("min_dist_to_obstacles")) for e in episodes]
    rewards = [safe_float(e.get("reward")) for e in episodes]
    reasons = [str(e.get("reason", "unknown")) for e in episodes]
    
    valid = [(d, r, rs) for d, r, rs in zip(min_dists, rewards, reasons) 
             if np.isfinite(d) and np.isfinite(r)]
    if not valid: return

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color by wellness state
    for reason in set(rs for _, _, rs in valid):
        subset = [(d, r) for d, r, rs in valid if rs == reason]
        if subset:
            ds, rs_vals = zip(*subset)
            spa_scatter(ax, ds, rs_vals, reason_color(reason), s=30, alpha=0.6, 
                       label=f"State: {reason.title()}")
    
    ax.set_xlabel("Safety Margin (Balance)", fontweight="400")
    ax.set_ylabel("Growth Score", fontweight="400")
    wellness_title(ax, "Balance vs Growth — Wellness Harmony")
    spa_legend(ax)
    savefig(os.path.join(outdir, "05_wellness_balance_vs_growth.png"))


def plot_safety_margins_by_reason(episodes, outdir):
    """Safety margins as wellness balance by state."""
    by_reason = defaultdict(list)
    for e in episodes:
        md = safe_float(e.get("min_dist_to_obstacles"))
        if np.isfinite(md):
            by_reason[str(e.get("reason", "unknown"))].append(md)
    
    if not by_reason: return
    uniq = sorted(by_reason.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    spa_boxplot(ax, [by_reason[r] for r in uniq], 
                [f"State: {r.title()}" for r in uniq], 
                [reason_color(r) for r in uniq])
    
    ax.set_ylabel("Wellness Balance Margin", fontweight="400")
    wellness_title(ax, "Wellness Balance by State")
    savefig(os.path.join(outdir, "06_wellness_balance_by_state.png"))


def plot_losses_and_entropy(episodes, outdir, w):
    """Training metrics as wellness adjustments."""
    ep = [e["episode"] for e in episodes]
    actor_loss = [safe_float(e.get("actor_loss")) for e in episodes]
    critic_loss = [safe_float(e.get("critic_loss")) for e in episodes]
    entropy = [safe_float(e.get("entropy")) for e in episodes]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Mind adjustment (actor loss)
    if any(np.isfinite(a) for a in actor_loss):
        spa_scatter(axes[0], ep, actor_loss, LOSS_COLORS["actor"], s=8, alpha=0.3)
        eucalyptus_line(axes[0], ep, rolling_mean(actor_loss, w), LOSS_COLORS["actor"], 
                       lw=2.5, label="Mind Adjustment Flow")
        axes[0].set_ylabel("Mind Adjustment", fontweight="400")
        wellness_title(axes[0], "Mind Wellness Calibration")
        spa_legend(axes[0])
    
    # Body adjustment (critic loss)
    if any(np.isfinite(c) for c in critic_loss):
        spa_scatter(axes[1], ep, critic_loss, LOSS_COLORS["critic"], s=8, alpha=0.3)
        eucalyptus_line(axes[1], ep, rolling_mean(critic_loss, w), LOSS_COLORS["critic"], 
                       lw=2.5, label="Body Adjustment Flow")
        axes[1].set_ylabel("Body Adjustment", fontweight="400")
        wellness_title(axes[1], "Body Wellness Calibration")
        spa_legend(axes[1])
    
    # Spirit exploration (entropy)
    if any(np.isfinite(e_val) for e_val in entropy):
        spa_scatter(axes[2], ep, entropy, PRIMARY, s=8, alpha=0.3)
        eucalyptus_line(axes[2], ep, rolling_mean(entropy, w), PRIMARY, 
                       lw=2.5, label="Spirit Exploration Flow")
        axes[2].set_ylabel("Spirit Exploration", fontweight="400")
        wellness_title(axes[2], "Spiritual Wellness Journey")
        spa_legend(axes[2])
    
    axes[2].set_xlabel("Training Session", fontweight="400")
    savefig(os.path.join(outdir, "07_wellness_mind_body_spirit.png"))


def plot_start_goal_difficulty(episodes, outdir):
    """Challenge analysis as growth opportunities."""
    starts, goals, dists = [], [], []
    for e in episodes:
        rs, gs = vec2(e.get("robot_start")), vec2(e.get("goal_start"))
        if all(np.isfinite([rs[0], rs[1], gs[0], gs[1]])):
            starts.append(rs); goals.append(gs); dists.append(dist2(rs, gs))
    
    if not dists: return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Challenge distribution
    ax1.hist(dists, bins=30, alpha=0.7, color=PRIMARY, 
             edgecolor=WELLNESS_WHITE, linewidth=1)
    ax1.set_xlabel("Growth Challenge Distance", fontweight="400")
    ax1.set_ylabel("Frequency", fontweight="400")
    wellness_title(ax1, "Growth Challenge Distribution")
    
    # Growth map
    sx, sy = zip(*starts)
    gx, gy = zip(*goals)
    spa_scatter(ax2, sx, sy, SECONDARY, s=25, alpha=0.6, label="Starting Points")
    spa_scatter(ax2, gx, gy, REASON_COLORS["goal"], s=25, alpha=0.6, 
               label="Growth Destinations")
    ax2.set_xlabel("Wellness Space X", fontweight="400")
    ax2.set_ylabel("Wellness Space Y", fontweight="400")
    ax2.set_aspect("equal")
    wellness_title(ax2, "Wellness Growth Map")
    spa_legend(ax2)
    
    savefig(os.path.join(outdir, "08_wellness_growth_challenges.png"))


def plot_start_goal_maps(episodes, outdir):
    """Placeholder for heatmaps - keeping wellness theme simple."""
    pass


def plot_shield_stats(episodes, outdir, w):
    """Protection system as wellness guardians."""
    ep = [e["episode"] for e in episodes]
    interventions = [safe_float(e.get("shield_interventions_ep", 0)) for e in episodes]
    activations = [safe_float(e.get("shield_activations_ep", 0)) for e in episodes]
    
    if not any(np.isfinite(i) for i in interventions): return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Guardian interventions
    spa_scatter(ax1, ep, interventions, REASON_COLORS["collision"], s=12, alpha=0.4)
    eucalyptus_line(ax1, ep, rolling_mean(interventions, w), REASON_COLORS["collision"], 
                   lw=2.5, label=f"Guardian Protection Flow ({w})")
    ax1.set_ylabel("Guardian Interventions", fontweight="400")
    wellness_title(ax1, "Wellness Guardian Interventions")
    spa_legend(ax1)
    
    # Guardian activations
    if any(np.isfinite(a) for a in activations):
        spa_scatter(ax2, ep, activations, REASON_COLORS["timeout"], s=12, alpha=0.4)
        eucalyptus_line(ax2, ep, rolling_mean(activations, w), REASON_COLORS["timeout"], 
                       lw=2.5, label=f"Guardian Awareness Flow ({w})")
        ax2.set_ylabel("Guardian Activations", fontweight="400")
        wellness_title(ax2, "Wellness Guardian Awareness")
        spa_legend(ax2)
    
    ax2.set_xlabel("Training Session", fontweight="400")
    savefig(os.path.join(outdir, "09_wellness_guardian_protection.png"))


def plot_shield_level_distribution(episodes, outdir, w):
    """Protection levels as wellness guardian states."""
    ep = [e["episode"] for e in episodes]
    levels = [str(e.get("shield_level", "none")) for e in episodes]
    if not levels: return
    
    fractions = rolling_fraction(levels, w)
    uniq = sorted(fractions.keys())
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Stacked area with guardian colors
    bottom = np.zeros(len(ep))
    for level in uniq:
        color = SHIELD_COLORS.get(level, PRIMARY)
        label_map = {"none": "Peace", "caution": "Awareness", 
                    "danger": "Alert", "critical": "Protection"}
        label = label_map.get(level, level.title())
        ax.fill_between(ep, bottom, bottom + fractions[level], 
                       alpha=0.7, color=color, label=f"Guardian: {label}", linewidth=0)
        bottom += fractions[level]
    
    ax.set_xlabel("Training Session", fontweight="400")
    ax.set_ylabel("Guardian State Distribution", fontweight="400")
    ax.set_ylim(0, 1)
    wellness_title(ax, "Wellness Guardian State Evolution")
    spa_legend(ax)
    savefig(os.path.join(outdir, "10_wellness_guardian_states.png"))


def plot_ttc_analysis(episodes, outdir, w):
    """Time-to-collision as mindful awareness."""
    ep = [e["episode"] for e in episodes]
    ttc_data = [safe_float(e.get("min_ttc", np.nan)) for e in episodes]
    if not any(np.isfinite(t) for t in ttc_data): return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    spa_scatter(ax, ep, ttc_data, PRIMARY, s=12, alpha=0.4)
    eucalyptus_line(ax, ep, rolling_mean(ttc_data, w), PRIMARY, lw=2.5, 
                   label=f"Mindful Awareness Flow ({w})")
    wellness_hline(ax, 2.0, REASON_COLORS["collision"], label="Mindfulness Threshold")
    
    ax.set_xlabel("Training Session", fontweight="400")
    ax.set_ylabel("Mindful Awareness Time (s)", fontweight="400")
    wellness_title(ax, "Mindful Awareness Development")
    spa_legend(ax)
    savefig(os.path.join(outdir, "11_wellness_mindful_awareness.png"))


def plot_policy_vs_command(episodes, outdir, w):
    """Policy vs command as intention vs action alignment."""
    ep = [e["episode"] for e in episodes]
    v_pol = [safe_float(e.get("v_policy")) for e in episodes]
    v_cmd = [safe_float(e.get("v_cmd")) for e in episodes]
    w_pol = [safe_float(e.get("w_policy")) for e in episodes]
    w_cmd = [safe_float(e.get("w_cmd")) for e in episodes]
    if all(np.isnan(v) for v in v_pol): return

    for va, vb, la, lb, yl, ti, fn in [
        (v_pol, v_cmd, "Intended Movement", "Guided Movement", "Linear Flow (m/s)", 
         "Movement Intention vs Guidance — Linear Flow", "12_wellness_linear_movement_harmony.png"),
        (w_pol, w_cmd, "Intended Rotation", "Guided Rotation", "Rotational Flow (rad/s)", 
         "Movement Intention vs Guidance — Rotational Flow", "13_wellness_rotational_movement_harmony.png"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        spa_scatter(ax, ep, va, PRIMARY, s=8, alpha=0.3)
        spa_scatter(ax, ep, vb, REASON_COLORS["collision"], s=8, alpha=0.3)
        eucalyptus_line(ax, ep, rolling_mean(va, w), PRIMARY, lw=2.5, label=la)
        eucalyptus_line(ax, ep, rolling_mean(vb, w), REASON_COLORS["collision"], lw=2.5, label=lb)
        ax.set_xlabel("Training Session", fontweight="400")
        ax.set_ylabel(yl, fontweight="400")
        wellness_title(ax, ti)
        spa_legend(ax)
        savefig(os.path.join(outdir, fn))

    # Movement harmony analysis
    v_diff, w_diff, ep_v = [], [], []
    for i in range(len(episodes)):
        vp, vc = safe_float(v_pol[i]), safe_float(v_cmd[i])
        wp, wc = safe_float(w_pol[i]), safe_float(w_cmd[i])
        if np.isfinite(vp) and np.isfinite(vc) and np.isfinite(wp) and np.isfinite(wc):
            v_diff.append(abs(vp - vc)); w_diff.append(abs(wp - wc)); ep_v.append(ep[i])
    if v_diff:
        fig, ax = plt.subplots(figsize=(12, 6))
        spa_scatter(ax, ep_v, v_diff, REASON_COLORS["collision"], s=6, alpha=0.2)
        spa_scatter(ax, ep_v, w_diff, REASON_COLORS["timeout"], s=6, alpha=0.2)
        eucalyptus_line(ax, ep_v, rolling_mean(np.array(v_diff), w), REASON_COLORS["collision"], 
                       lw=2.5, label="Linear Adjustment")
        eucalyptus_line(ax, ep_v, rolling_mean(np.array(w_diff), w), REASON_COLORS["timeout"], 
                       lw=2.5, label="Rotational Adjustment")
        ax.set_xlabel("Training Session", fontweight="400")
        ax.set_ylabel("Guidance Adjustment Magnitude", fontweight="400")
        wellness_title(ax, "Movement Guidance Magnitude")
        spa_legend(ax)
        savefig(os.path.join(outdir, "14_wellness_movement_guidance_magnitude.png"))


def pick_representative_episodes(episodes, topk=12):
    """Select episodes for detailed wellness journey analysis."""
    if not episodes: return []
    by_reason = defaultdict(list)
    for e in episodes: by_reason[str(e.get("reason", "unknown"))].append(e)
    def sort_rew(es, rev=True): return sorted(es, key=lambda x: safe_float(x.get("reward"), -np.inf), reverse=rev)
    picks = []
    if "goal" in by_reason: picks += sort_rew(by_reason["goal"])[: max(1, topk // 3)]
    if "collision" in by_reason: picks += sort_rew(by_reason["collision"], rev=False)[: max(1, topk // 3)]
    if "timeout" in by_reason: picks += sort_rew(by_reason["timeout"])[: max(1, topk // 3)]
    if len(picks) < topk:
        rem = sorted([e for e in episodes if e not in picks], key=lambda x: safe_float(x.get("reward"), 0.0))
        if rem:
            idxs = np.linspace(0, len(rem)-1, num=min(topk-len(picks), len(rem)), dtype=int)
            picks += [rem[i] for i in idxs]
    seen, uniq = set(), []
    for e in picks:
        eid = e.get("episode")
        if eid in seen: continue
        seen.add(eid); uniq.append(e)
        if len(uniq) >= topk: break
    return uniq


def plot_obstacle_world_overlay(episodes, outdir, topk=12):
    """Wellness space analysis with obstacles as challenges."""
    reps = pick_representative_episodes(episodes, topk=topk)
    if not reps: return
    for e in reps:
        ep = e.get("episode"); reason = str(e.get("reason", "unknown"))
        rsx, rsy = vec2(e.get("robot_start")); rfx, rfy = vec2(e.get("robot_final"))
        gsx, gsy = vec2(e.get("goal_start")); gfx, gfy = vec2(e.get("goal_final"))
        fig, ax = plt.subplots(figsize=(8, 8))
        if np.isfinite(rsx): spa_scatter(ax, [rsx], [rsy], PRIMARY, s=120, label="Journey Start")
        if np.isfinite(rfx): ax.scatter([rfx], [rfy], marker="x", s=120, color=PRIMARY, 
                                       linewidths=3, label="Journey End")
        if np.isfinite(gsx): ax.scatter([gsx], [gsy], marker="^", s=120, color=REASON_COLORS["goal"], 
                                       label="Growth Target")
        if np.isfinite(gfx): ax.scatter([gfx], [gfy], marker="s", s=120, color=REASON_COLORS["goal"], 
                                       label="Growth Achievement")
        slots = e.get("obstacles_world_slots", []); ox, oy = [], []
        for s in slots:
            if isinstance(s, dict) and s.get("valid", True):
                wx, wy = safe_float(s.get("wx")), safe_float(s.get("wy"))
                if np.isfinite(wx) and np.isfinite(wy): ox.append(wx); oy.append(wy)
        if ox:
            ax.scatter(ox, oy, s=140, marker="D", color=REASON_COLORS["collision"], 
                      label="Growth Challenges", alpha=0.8)
            for x, y in zip(ox, oy):
                circle = plt.Circle((x, y), 0.7, color=REASON_COLORS["collision"], 
                                  fill=False, linestyle="--", alpha=0.4, linewidth=2)
                ax.add_patch(circle)
        ax.set_xlabel("Wellness Space X", fontweight="400")
        ax.set_ylabel("Wellness Space Y", fontweight="400")
        ax.set_aspect("equal")
        wellness_title(ax, f"Session {ep}: {reason.title()} — Wellness Space")
        spa_legend(ax)
        savefig(os.path.join(outdir, f"15_wellness_space_session_{ep:05d}.png"))


def plot_trajectory_story(episodes, outdir, topk=12):
    """Wellness journey path stories."""
    candidates = [e for e in episodes if isinstance(e.get("robot_path"), list) and len(e.get("robot_path")) > 1]
    if not candidates: return
    reps = pick_representative_episodes(candidates, topk=topk)
    for e in reps:
        ep = e.get("episode"); reason = str(e.get("reason", "unknown"))
        shield_count = safe_float(e.get("shield_interventions_ep", 0), 0)
        rp = e.get("robot_path", [])
        rxy = np.array([[safe_float(p[0]), safe_float(p[1])] for p in rp if isinstance(p, (list, tuple)) and len(p) >= 2], dtype=float)
        if rxy.shape[0] < 2: continue
        fig, ax = plt.subplots(figsize=(9, 9))
        # Wellness path with flowing effect
        ax.plot(rxy[:, 0], rxy[:, 1], linewidth=4, color=PRIMARY, alpha=0.3)
        ax.plot(rxy[:, 0], rxy[:, 1], linewidth=2.5, color=PRIMARY, alpha=0.8, label="Wellness Path")
        ax.scatter([rxy[0, 0]], [rxy[0, 1]], marker="o", s=140, color=PRIMARY, 
                  edgecolors=WELLNESS_WHITE, linewidths=2)
        ax.scatter([rxy[-1, 0]], [rxy[-1, 1]], marker="x", s=140, color=PRIMARY, 
                  linewidths=4)
        gp = e.get("goal_path", None)
        if isinstance(gp, list) and len(gp) > 1:
            gxy = np.array([[safe_float(p[0]), safe_float(p[1])] for p in gp if isinstance(p, (list, tuple)) and len(p) >= 2], dtype=float)
            if gxy.shape[0] >= 2:
                ax.plot(gxy[:, 0], gxy[:, 1], linewidth=3, color=REASON_COLORS["goal"], 
                       alpha=0.7, label="Growth Target Path")
                ax.scatter([gxy[0, 0]], [gxy[0, 1]], marker="^", s=140, color=REASON_COLORS["goal"])
        slots = e.get("obstacles_world_slots", []); ox, oy = [], []
        for s in slots:
            if isinstance(s, dict) and s.get("valid", True):
                wx, wy = safe_float(s.get("wx")), safe_float(s.get("wy"))
                if np.isfinite(wx) and np.isfinite(wy): ox.append(wx); oy.append(wy)
        if ox:
            ax.scatter(ox, oy, s=160, marker="D", color=REASON_COLORS["collision"], 
                      label="Growth Challenges", alpha=0.8)
            for x, y in zip(ox, oy):
                circle = plt.Circle((x, y), 0.7, color=REASON_COLORS["collision"], 
                                  fill=False, linestyle="--", alpha=0.4, linewidth=2)
                ax.add_patch(circle)
        ax.set_xlabel("Wellness Space X", fontweight="400")
        ax.set_ylabel("Wellness Space Y", fontweight="400")
        ax.set_aspect("equal")
        wellness_title(ax, f"Session {ep}: {reason.title()} | Guardian: {int(shield_count)} assists")
        spa_legend(ax)
        savefig(os.path.join(outdir, f"16_wellness_journey_story_session_{ep:05d}.png"))


# ═══════════════════════════════════════════
# MAIN WELLNESS JOURNEY
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Plot training results — Eucalyptus Spa Wellness Journey Edition")
    parser.add_argument("--json", required=True, help="Path to episode_stats.json")
    parser.add_argument("--out", default="wellness_plots", help="Output directory for wellness visualizations.")
    parser.add_argument("--rolling", type=int, default=20, help="Rolling window for wellness flow.")
    parser.add_argument("--topk", type=int, default=12, help="Representative wellness sessions for detailed analysis.")
    args = parser.parse_args()

    ensure_dir(args.out)
    _apply_wellness_rc()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("JSON must be a non-empty list of episode dicts.")

    data = sorted(data, key=lambda e: safe_float(e.get("episode"), np.inf))
    print(f"🌿 Welcome to your Wellness Journey! Loaded {len(data)} healing sessions...")
    print("🧘 Generating therapeutic visualizations with eucalyptus spa theme...")

    # Complete wellness journey visualization suite
    plot_reward(data, args.out, args.rolling)
    plot_success_rate(data, args.out, args.rolling)
    plot_length_and_time(data, args.out, args.rolling)
    plot_reason_distribution(data, args.out, args.rolling)
    plot_min_dist_vs_reward(data, args.out)
    plot_safety_margins_by_reason(data, args.out)
    plot_losses_and_entropy(data, args.out, w=args.rolling)
    plot_start_goal_difficulty(data, args.out)
    plot_start_goal_maps(data, args.out)
    plot_shield_stats(data, args.out, args.rolling)
    plot_shield_level_distribution(data, args.out, args.rolling)
    plot_ttc_analysis(data, args.out, args.rolling)
    plot_policy_vs_command(data, args.out, args.rolling)
    plot_obstacle_world_overlay(data, args.out, topk=args.topk)
    plot_trajectory_story(data, args.out, topk=args.topk)

    print(f"🌿 Wellness Journey Complete! Therapeutic visualizations saved to: {os.path.abspath(args.out)}")
    print("✨ Your data now flows with the healing energy of a eucalyptus spa retreat.")
    print("🧘 May these visualizations bring clarity and peace to your analysis journey.")


if __name__ == "__main__":
    main()
