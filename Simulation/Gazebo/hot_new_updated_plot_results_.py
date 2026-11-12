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
# THEMES
# ═══════════════════════════════════════════

THEMES = {

    # ─────────────────────────────
    # 1. MUJI / Japanese Minimal
    # ─────────────────────────────
    "muji": {

        "BG": "#f7f3ee",
        "BG_AXES": "#f7f3ee",
        "GRID_CLR": "#e6dfd5",
        "TEXT_CLR": "#5f5a55",
        "TICK_CLR": "#7a746e",
        "EDGE_CLR": "#d8d1c7",
        "PAPER_WHITE": "#f7f3ee",

        "PRIMARY": "#a8c5d1",
        "SECONDARY": "#d8b4a0",
        "ACCENT": "#b7c4b2",

        "REASON_COLORS": {
            "goal": "#a3b18a",
            "collision": "#bc6c25",
            "timeout": "#dda15e",
        },

        "SHIELD_COLORS": {
            "none": "#a3b18a",
            "caution": "#dda15e",
            "danger": "#bc6c25",
            "critical": "#8d99ae",
        },

        "LOSS_COLORS": {
            "actor": "#a8c5d1",
            "critic": "#d8b4a0",
        },

        "STAGE_BAND_COLORS": [
            "#f0ebe5",
            "#ebe6dd",
            "#e8e1d8",
            "#f2ede6",
        ]
    },

    # ─────────────────────────────
    # 2. Soft Linen
    # ─────────────────────────────
    "soft_linen": {

        "BG": "#F8F5F1",
        "BG_AXES": "#F8F5F1",
        "GRID_CLR": "#E7E0D8",
        "TEXT_CLR": "#5E5A56",
        "TICK_CLR": "#7C746D",
        "EDGE_CLR": "#D6CCC2",
        "PAPER_WHITE": "#F8F5F1",

        "PRIMARY": "#D8C3A5",
        "SECONDARY": "#C9ADA7",
        "ACCENT": "#B7B7A4",

        "REASON_COLORS": {
            "goal": "#A5A58D",
            "collision": "#C97C5D",
            "timeout": "#DDB892",
        },

        "SHIELD_COLORS": {
            "none": "#A5A58D",
            "caution": "#DDB892",
            "danger": "#C97C5D",
            "critical": "#9A8C98",
        },

        "LOSS_COLORS": {
            "actor": "#D8C3A5",
            "critic": "#C9ADA7",
        },

        "STAGE_BAND_COLORS": [
            "#f1ebe4",
            "#ece5dc",
            "#e8ddd1",
            "#f3eee8",
        ]
    },

    # ─────────────────────────────
    # 3. Nude Editorial
    # ─────────────────────────────
    "nude_editorial": {

        "BG": "#FAF7F2",
        "BG_AXES": "#FAF7F2",
        "GRID_CLR": "#ECE6DE",
        "TEXT_CLR": "#605A54",
        "TICK_CLR": "#7D756E",
        "EDGE_CLR": "#DDD2C7",
        "PAPER_WHITE": "#FAF7F2",

        "PRIMARY": "#B7A99A",
        "SECONDARY": "#C7A9A3",
        "ACCENT": "#A68A64",

        "REASON_COLORS": {
            "goal": "#A3B18A",
            "collision": "#B56576",
            "timeout": "#D4A373",
        },

        "SHIELD_COLORS": {
            "none": "#A3B18A",
            "caution": "#D4A373",
            "danger": "#B56576",
            "critical": "#6D6875",
        },

        "LOSS_COLORS": {
            "actor": "#B7A99A",
            "critic": "#C7A9A3",
        },

        "STAGE_BAND_COLORS": [
            "#f3eee7",
            "#eee5db",
            "#e9ddd2",
            "#f7f2eb",
        ]
    },

    # ─────────────────────────────
    # 4. Japanese Earth
    # ─────────────────────────────
    "japanese_earth": {

        "BG": "#F6F1EB",
        "BG_AXES": "#F6F1EB",
        "GRID_CLR": "#E4DBD0",
        "TEXT_CLR": "#5C5752",
        "TICK_CLR": "#756F69",
        "EDGE_CLR": "#D8CEC2",
        "PAPER_WHITE": "#F6F1EB",

        "PRIMARY": "#A3B18A",
        "SECONDARY": "#DDBEA9",
        "ACCENT": "#B08968",

        "REASON_COLORS": {
            "goal": "#A3B18A",
            "collision": "#9C6644",
            "timeout": "#D4A373",
        },

        "SHIELD_COLORS": {
            "none": "#A3B18A",
            "caution": "#D4A373",
            "danger": "#9C6644",
            "critical": "#7F5539",
        },

        "LOSS_COLORS": {
            "actor": "#A3B18A",
            "critic": "#DDBEA9",
        },

        "STAGE_BAND_COLORS": [
            "#efe6dc",
            "#e7ddd1",
            "#ddd1c3",
            "#f3ece3",
        ]
    },

    # ─────────────────────────────
    # 5. Warm Gray Ultra Minimal
    # ─────────────────────────────
    "warm_gray": {

        "BG": "#FBFAF8",
        "BG_AXES": "#FBFAF8",
        "GRID_CLR": "#ECE8E2",
        "TEXT_CLR": "#57534E",
        "TICK_CLR": "#78716C",
        "EDGE_CLR": "#D6D3D1",
        "PAPER_WHITE": "#FBFAF8",

        "PRIMARY": "#A8A29E",
        "SECONDARY": "#CBBBA0",
        "ACCENT": "#78716C",

        "REASON_COLORS": {
            "goal": "#A3A380",
            "collision": "#BC6C25",
            "timeout": "#D4A373",
        },

        "SHIELD_COLORS": {
            "none": "#A3A380",
            "caution": "#D4A373",
            "danger": "#BC6C25",
            "critical": "#6B705C",
        },

        "LOSS_COLORS": {
            "actor": "#A8A29E",
            "critic": "#CBBBA0",
        },

        "STAGE_BAND_COLORS": [
            "#f2f0ec",
            "#ece8e2",
            "#e6e1da",
            "#f7f5f1",
        ]
    },
    "porcelain_sage": {
    "BG": "#FAF9F6",
    "BG_AXES": "#FAF9F6",
    "GRID_CLR": "#E8E4DD",
    "TEXT_CLR": "#4F514A",
    "TICK_CLR": "#74766F",
    "EDGE_CLR": "#D9D5CC",
    "PAPER_WHITE": "#FAF9F6",

    "PRIMARY": "#8FA998",
    "SECONDARY": "#D7BFAE",
    "ACCENT": "#A7B8C2",

    "REASON_COLORS": {
        "goal": "#8FA998",
        "collision": "#B76E65",
        "timeout": "#D6A96F",
    },

    "SHIELD_COLORS": {
        "none": "#8FA998",
        "caution": "#D6A96F",
        "danger": "#B76E65",
        "critical": "#6F7480",
    },

    "LOSS_COLORS": {
        "actor": "#A7B8C2",
        "critic": "#D7BFAE",
    },

    "STAGE_BAND_COLORS": [
        "#F1EFEA",
        "#EBE8E1",
        "#E5E0D7",
        "#F5F2EC",
    ]
},
"sandstone_clay": {
    "BG": "#FBF6EF",
    "BG_AXES": "#FBF6EF",
    "GRID_CLR": "#EADFD2",
    "TEXT_CLR": "#574E45",
    "TICK_CLR": "#786D62",
    "EDGE_CLR": "#DDD0C0",
    "PAPER_WHITE": "#FBF6EF",

    "PRIMARY": "#A3A380",
    "SECONDARY": "#C8A27C",
    "ACCENT": "#9FB3B5",

    "REASON_COLORS": {
        "goal": "#8E9F76",
        "collision": "#A65F46",
        "timeout": "#D19A66",
    },

    "SHIELD_COLORS": {
        "none": "#8E9F76",
        "caution": "#D19A66",
        "danger": "#A65F46",
        "critical": "#6D5F58",
    },

    "LOSS_COLORS": {
        "actor": "#9FB3B5",
        "critic": "#C8A27C",
    },

    "STAGE_BAND_COLORS": [
        "#F1E8DD",
        "#EBDDCF",
        "#E4D2C1",
        "#F6EEE4",
    ]
},
"oat_blueprint": {
    "BG": "#FAF8F2",
    "BG_AXES": "#FAF8F2",
    "GRID_CLR": "#E3E0D8",
    "TEXT_CLR": "#50545A",
    "TICK_CLR": "#73777E",
    "EDGE_CLR": "#D5D2CA",
    "PAPER_WHITE": "#FAF8F2",

    "PRIMARY": "#8EA4B8",
    "SECONDARY": "#CDBBA7",
    "ACCENT": "#A9A58F",

    "REASON_COLORS": {
        "goal": "#9AA37A",
        "collision": "#B56B65",
        "timeout": "#D0A96D",
    },

    "SHIELD_COLORS": {
        "none": "#9AA37A",
        "caution": "#D0A96D",
        "danger": "#B56B65",
        "critical": "#687284",
    },

    "LOSS_COLORS": {
        "actor": "#8EA4B8",
        "critic": "#CDBBA7",
    },

    "STAGE_BAND_COLORS": [
        "#F0EEE8",
        "#E9E7E0",
        "#E2DFD6",
        "#F5F3ED",
    ]
},
"gallery_mist": {
    "BG": "#FCFBF8",
    "BG_AXES": "#FCFBF8",
    "GRID_CLR": "#ECE9E3",
    "TEXT_CLR": "#4D4A46",
    "TICK_CLR": "#706C66",
    "EDGE_CLR": "#DAD6CE",
    "PAPER_WHITE": "#FCFBF8",

    "PRIMARY": "#9CAFAA",
    "SECONDARY": "#D5BDAF",
    "ACCENT": "#B8B2A6",

    "REASON_COLORS": {
        "goal": "#9CAFAA",
        "collision": "#B58375",
        "timeout": "#D4A373",
    },

    "SHIELD_COLORS": {
        "none": "#9CAFAA",
        "caution": "#D4A373",
        "danger": "#B58375",
        "critical": "#7A7D7D",
    },

    "LOSS_COLORS": {
        "actor": "#9CAFAA",
        "critic": "#D5BDAF",
    },

    "STAGE_BAND_COLORS": [
        "#F3F1EC",
        "#EDEAE4",
        "#E6E2DA",
        "#F7F5F0",
    ]
},
"thesis_nude_contrast": {
    "BG": "#FAF6F0",
    "BG_AXES": "#FAF6F0",
    "GRID_CLR": "#E8DDD2",
    "TEXT_CLR": "#463F3A",
    "TICK_CLR": "#6B625B",
    "EDGE_CLR": "#D8CABD",
    "PAPER_WHITE": "#FAF6F0",

    "PRIMARY": "#7D9D9C",
    "SECONDARY": "#C9A27E",
    "ACCENT": "#B7A99A",

    "REASON_COLORS": {
        "goal": "#87986A",
        "collision": "#A44A3F",
        "timeout": "#C98245",
    },

    "SHIELD_COLORS": {
        "none": "#87986A",
        "caution": "#C98245",
        "danger": "#A44A3F",
        "critical": "#5E6472",
    },

    "LOSS_COLORS": {
        "actor": "#7D9D9C",
        "critic": "#C9A27E",
    },

    "STAGE_BAND_COLORS": [
        "#F1E8DE",
        "#EADBCB",
        "#E3D1BF",
        "#F6EEE6",
    ]
}
}

# Default theme
CURRENT_THEME = THEMES["muji"]


def apply_theme(theme_name="muji"):
    global CURRENT_THEME

    if theme_name not in THEMES:
        raise ValueError(
            f"Unknown theme '{theme_name}'. "
            f"Available themes: {list(THEMES.keys())}"
        )

    CURRENT_THEME = THEMES[theme_name]

    globals().update(CURRENT_THEME)


def _apply_scientific_rc(theme_name="muji"):
    apply_theme(theme_name)
    """Set global rcParams for caramel pastel theme."""
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG_AXES,
        "axes.edgecolor":    EDGE_CLR,
        "axes.grid":         True,
        "grid.color":        GRID_CLR,
        "grid.alpha":        0.8,
        "grid.linewidth":    0.4,
        "grid.linestyle":    "-",
        "axes.labelcolor":   TEXT_CLR,
        "text.color":        TEXT_CLR,
        "xtick.color":       TICK_CLR,
        "ytick.color":       TICK_CLR,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  True,
        "axes.spines.bottom": True,
        "axes.linewidth":    0.8,
        "font.size":         10,
        "font.family":       "sans-serif",
        "legend.facecolor":  PAPER_WHITE,
        "legend.edgecolor":  EDGE_CLR,
        "legend.labelcolor": TEXT_CLR,
        "legend.framealpha": 0.90,
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
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  saved: {fname}")


def add_stage_regions(ax, stage_transitions, all_episodes):
    """Shade stage regions and label them. Works for any y range."""
    boundaries = [min(all_episodes)] + list(stage_transitions) + [max(all_episodes)]
    ymin, ymax = ax.get_ylim()
    for i in range(len(boundaries) - 1):
        x0, x1 = boundaries[i], boundaries[i + 1]
        color = STAGE_BAND_COLORS[i % len(STAGE_BAND_COLORS)]
        ax.axvspan(x0, x1, alpha=0.55, color=color, zorder=0)
        x_mid = (x0 + x1) / 2
        ax.text(x_mid, ymax - (ymax - ymin) * 0.03,
                f"S{i + 1}", ha="center", va="top",
                fontsize=9, fontweight="bold", color=TICK_CLR,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=PAPER_WHITE,
                          edgecolor=EDGE_CLR, alpha=0.75, linewidth=0.7))
    ax.set_ylim(ymin, ymax)


def density_cloud(ax, x, y, color, label=None, n_layers=6):
    """Tableau-style density cloud: multiple scatter passes with decreasing size & alpha.
    Dense regions become saturated; sparse regions stay faint — no KDE required."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) == 0:
        return
    # Layer parameters: (point_size, alpha)
    layers = [
        (80, 0.04),
        (40, 0.06),
        (18, 0.10),
        (9,  0.14),
        (4,  0.18),
        (2,  0.22),
    ]
    for i, (s, a) in enumerate(layers[:n_layers]):
        kw = dict(s=s, alpha=a, color=color, linewidths=0, zorder=2)
        if i == n_layers - 1 and label:
            kw["label"] = label
        ax.scatter(x, y, **kw)


def add_stage_borders(ax, stage_transitions):
    """Draw only vertical border lines at stage transitions with S-number labels."""
    ymin, ymax = ax.get_ylim()
    for i, t in enumerate(stage_transitions):
        ax.axvline(x=t, color=TICK_CLR, linewidth=1.2,
                   linestyle="--", alpha=0.6, zorder=3)
        ax.text(t + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.005,
                ymax - (ymax - ymin) * 0.03,
                f"S{i + 2}", va="top", ha="left",
                fontsize=8, fontweight="bold", color=TICK_CLR,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=PAPER_WHITE,
                          edgecolor=EDGE_CLR, alpha=0.7, linewidth=0.6))
    ax.set_ylim(ymin, ymax)


# ═══════════════════════════════════════════
# PLOT FUNCTIONS
# ═══════════════════════════════════════════

def plot_reward(episodes, stage_transitions, outdir, w):
    """Episode reward — single axis, density clouds by termination reason, stage border lines."""
    ep = np.array([e["episode"] for e in episodes])
    rew = np.array([safe_float(e.get("reward")) for e in episodes])
    reasons = [str(e.get("reason", "unknown")) for e in episodes]
    if all(np.isnan(r) for r in rew):
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # ── Density clouds per termination reason ──
    for reason in ["collision", "timeout", "goal"]:
        mask = np.array([r == reason for r in reasons])
        if not mask.any():
            continue
        ex, ry = ep[mask], rew[mask]
        valid = np.isfinite(ry)
        ex, ry = ex[valid], ry[valid]
        if len(ex) == 0:
            continue
        density_cloud(ax, ex, ry, REASON_COLORS.get(reason, PRIMARY),
                      label=reason.capitalize())

    # ── Rolling mean ──
    line_with_shadow(ax, ep, rolling_mean(rew, w), PRIMARY, lw=2.5,
                     label=f"Rolling mean (w={w})")

    # ── Zero reference ──
    ax.axhline(0, color=EDGE_CLR, linewidth=1.0, alpha=0.6, linestyle="-")

    # ── Max reward annotation ──
    r_max = np.nanmax(rew)
    if r_max > 0:
        ax.axhline(y=r_max, color=REASON_COLORS["goal"],
                   linewidth=1.2, alpha=0.7, linestyle="--")
        ax.text(ep[-1], r_max, f"  max={r_max:.1f}", va="bottom", ha="right",
                fontsize=9, color=REASON_COLORS["goal"], fontweight="bold")

    # ── Stage border lines only (no shading) ──
    add_stage_borders(ax, stage_transitions)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    set_title(ax, "Reward Progress Over Training")
    add_legend(ax, loc="lower right")
    savefig(os.path.join(outdir, "01_reward.png"))


def plot_success_rate(episodes, stage_transitions, outdir, w):
    """Success rate (fraction of 'goal' outcomes) over training."""
    ep = [e["episode"] for e in episodes]
    reasons = [str(e.get("reason", "unknown")) for e in episodes]
    if not reasons:
        return

    success = [1 if r == "goal" else 0 for r in reasons]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_ylim(-0.05, 1.1)

    add_stage_borders(ax, stage_transitions)

    # Smoothed success rate — use a larger effective window for extra smoothness
    smooth_w = max(w, w * 2)
    frac_raw = rolling_fraction(success, w)[1]
    frac_smooth = rolling_mean(frac_raw, smooth_w)

    # Fill under the curve for clarity
    ax.fill_between(ep, 0, frac_smooth, alpha=0.15, color=REASON_COLORS["goal"])
    line_with_shadow(ax, ep, frac_smooth, REASON_COLORS["goal"],
                     lw=2.5, label=f"Success rate (rolling w={w}, smoothed)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate")
    set_title(ax, "Success Rate Over Training")
    add_legend(ax)
    savefig(os.path.join(outdir, "02_success_rate.png"))


def plot_length(episodes, stage_transitions, outdir, w):
    """Episode length, colored by termination reason as density clouds."""
    ep = [e["episode"] for e in episodes]
    lengths = [safe_float(e.get("length")) for e in episodes]
    reasons = [str(e.get("reason", "unknown")) for e in episodes]

    if not any(np.isfinite(l) for l in lengths):
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Density clouds by termination reason
    for reason in ["collision", "timeout", "goal"]:
        idx = [i for i, r in enumerate(reasons) if r == reason]
        if not idx:
            continue
        ex = [ep[i] for i in idx]
        ey = [lengths[i] for i in idx]
        density_cloud(ax, ex, ey, REASON_COLORS.get(reason, PRIMARY),
                      label=reason.capitalize())

    line_with_shadow(ax, ep, rolling_mean(lengths, w), PRIMARY, lw=2.5,
                     label=f"Rolling mean (w={w})")
    set_title(ax, "Episode Length Over Training")
    add_legend(ax)
    add_stage_borders(ax, stage_transitions)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length (steps)")
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
    ax.set_ylim(0, 1)

    add_stage_borders(ax, stage_transitions)

    # Stacked area plot
    bottom = np.zeros(len(ep))
    for r in uniq:
        color = reason_color(r)
        ax.fill_between(ep, bottom, bottom + fractions[r],
                        alpha=0.7, color=color,
                        label=f"Reason: {r}", linewidth=0)
        bottom += fractions[r]

    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction of episodes (rolling)")
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
        density_cloud(axes[0], ep, learning_rate, LOSS_COLORS["actor"])
        line_with_shadow(axes[0], ep, rolling_mean(learning_rate, w),
                         LOSS_COLORS["actor"], lw=2.5,
                         label="Learning Rate (rolling)")
        axes[0].set_ylabel("Learning Rate")
        set_title(axes[0], "Learning Rate Over Training")
        add_legend(axes[0])

    # Entropy
    if any(np.isfinite(e_val) for e_val in entropy):
        density_cloud(axes[1], ep, entropy, PRIMARY)
        line_with_shadow(axes[1], ep, rolling_mean(entropy, w),
                         PRIMARY, lw=2.5,
                         label="Entropy (rolling)")
        axes[1].set_ylabel("Entropy")
        set_title(axes[1], "Policy Entropy Over Training")
        add_legend(axes[1])
    add_stage_borders(axes[0], stage_transitions)
    add_stage_borders(axes[1], stage_transitions)
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
        density_cloud(axes[0], ep, actor_loss, LOSS_COLORS["actor"])
        line_with_shadow(axes[0], ep, rolling_mean(actor_loss, w),
                         LOSS_COLORS["actor"], lw=2.5,
                         label="Actor loss (rolling)")
        axes[0].set_ylabel("Actor loss")
        set_title(axes[0], "Actor Loss Over Training")
        add_legend(axes[0])

    # Critic loss
    if any(np.isfinite(c) for c in critic_loss):
        density_cloud(axes[1], ep, critic_loss, LOSS_COLORS["critic"])
        line_with_shadow(axes[1], ep, rolling_mean(critic_loss, w),
                         LOSS_COLORS["critic"], lw=2.5,
                         label="Critic loss (rolling)")
        axes[1].set_ylabel("Critic loss")
        set_title(axes[1], "Critic Loss Over Training")
        add_legend(axes[1])
    
    add_stage_borders(axes[0], stage_transitions)
    add_stage_borders(axes[1], stage_transitions)
    axes[1].set_xlabel("Episode")
    savefig(os.path.join(outdir, "05_losses_entropy.png"))


def plot_shield_stats(episodes, stage_transitions, outdir, w):
    """Shield interventions per episode — density clouds colored by termination reason."""
    ep = [e["episode"] for e in episodes]
    interventions = [safe_float(e.get("shield_interventions_ep", 0)) for e in episodes]
    lengths = [safe_float(e.get("length")) for e in episodes]
    reasons = [str(e.get("reason", "unknown")) for e in episodes]
    fractions = np.divide(np.array(interventions, dtype=float),
                          np.array(lengths, dtype=float))

    if not any(np.isfinite(i) for i in interventions):
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # ── Top panel: interventions per episode, colored by reason ──
    for reason in ["collision", "timeout", "goal"]:
        idx = [i for i, r in enumerate(reasons) if r == reason]
        if not idx:
            continue
        ex = [ep[i] for i in idx]
        ey_int = [interventions[i] for i in idx]
        ey_frac = [fractions[i] for i in idx]
        color = REASON_COLORS.get(reason, PRIMARY)
        density_cloud(axes[0], ex, ey_int, color, label=reason.capitalize())
        density_cloud(axes[1], ex, ey_frac, color)   # no duplicate legend labels

    line_with_shadow(axes[0], ep, rolling_mean(interventions, w),
                     SECONDARY, lw=2.5, label=f"Rolling mean (w={w})")
    axes[0].set_ylabel("Shield interventions / episode")
    set_title(axes[0], "Shield Interventions Over Training")
    add_legend(axes[0])

    line_with_shadow(axes[1], ep, rolling_mean(fractions, w),
                     SECONDARY, lw=2.5, label=f"Rolling mean (w={w})")
    axes[1].set_ylabel("Interventions / episode length")
    set_title(axes[1], "Shield Intervention Fraction per Episode")
    add_legend(axes[1])

    add_stage_borders(axes[0], stage_transitions)
    add_stage_borders(axes[1], stage_transitions)
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

    density_cloud(ax, ep, jerk_v, PRIMARY)
    line_with_shadow(ax, ep, rolling_mean(jerk_v, w), PRIMARY,
                     label="Mean jerk v (rolling)")

    density_cloud(ax, ep, jerk_w, SECONDARY)
    line_with_shadow(ax, ep, rolling_mean(jerk_w, w), SECONDARY,
                     label="Mean jerk w (rolling)")


    add_stage_borders(ax, stage_transitions)
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

    density_cloud(ax, ep, ultra, PRIMARY)
    line_with_shadow(ax, ep, rolling_mean(ultra, w), PRIMARY,
                     label="Min Ultrasonic (rolling)")

    density_cloud(ax, ep, depth, SECONDARY)
    line_with_shadow(ax, ep, rolling_mean(depth, w), SECONDARY,
                     label="Min Obstacle Depth (rolling)")


    add_stage_borders(ax, stage_transitions)
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

    ax.bar(ep, caution, color=SHIELD_COLORS["caution"], label="Caution")
    ax.bar(ep, danger, bottom=caution, color=SHIELD_COLORS["danger"], label="Danger")

    bottom_cd = np.array(caution) + np.array(danger)
    ax.bar(ep, critical, bottom=bottom_cd, color=SHIELD_COLORS["critical"], label="Critical")


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
                    color=SHIELD_COLORS["caution"], alpha=0.8, label="Caution")
    bottom += frac_caution

    ax.fill_between(ep, bottom, bottom + frac_danger,
                    color=SHIELD_COLORS["danger"], alpha=0.8, label="Danger")
    bottom += frac_danger

    ax.fill_between(ep, bottom, bottom + frac_critical,
                    color=SHIELD_COLORS["critical"], alpha=0.8, label="Critical")


    add_stage_borders(ax, stage_transitions)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction of Shield Types")
    set_title(ax, "Shield Type Distribution Over Training (Fraction)")
    add_legend(ax)

    savefig(os.path.join(outdir, "12_shield_fraction_area.png"))



def plot_termination_pie(episodes, outdir):
    """Pie chart of termination reasons showing counts and fractions."""
    reasons = [str(e.get("reason", "unknown")) for e in episodes]
    if not reasons:
        return

    counts = Counter(reasons)
    total = len(reasons)
    labels_order = ["goal", "timeout", "collision"] + \
                   [r for r in counts if r not in ("goal", "timeout", "collision")]
    present = [r for r in labels_order if r in counts]

    sizes = [counts[r] for r in present]
    colors = [reason_color(r) for r in present]
    labels = [
        f"{r.capitalize()}\n{counts[r]:,} ({counts[r]/total*100:.1f}%)"
        for r in present
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=2.0),
        textprops=dict(fontsize=11, color=TEXT_CLR),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color("white")

    ax.legend(wedges, labels, title="Termination Reason",
              loc="lower center", bbox_to_anchor=(0.5, -0.12),
              fontsize=10, ncol=len(present),
              facecolor=PAPER_WHITE, edgecolor=EDGE_CLR,
              labelcolor=TEXT_CLR, title_fontsize=10)

    ax.text(0, 0, f"n={total:,}", ha="center", va="center",
            fontsize=13, fontweight="bold", color=TEXT_CLR)

    set_title(ax, "Termination Reason Distribution")
    savefig(os.path.join(outdir, "13_termination_pie.png"))


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
    parser.add_argument("--theme", default="muji",
                        choices=list(THEMES.keys()), help="Visualization theme")
    args = parser.parse_args()

    ensure_dir(args.out)
    _apply_scientific_rc(args.theme)

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
    plot_termination_pie(data, args.out)

    plot_obstacle_world_overlay(data, args.out, topk=args.topk)
    plot_trajectory_story(data, args.out, topk=args.topk)

    print(f"Analysis complete. Figures saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
