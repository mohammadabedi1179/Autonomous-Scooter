"""
Training Analysis Plotter — Creative Scientific Edition
Light, nude, minimal, thesis-oriented visualization suite with optional interactive dashboard.

Reads episode_stats.json and generates the full analysis suite.

Usage:
  python training_analysis_creative_static.py --json episode_stats.json --out plots_out --rolling 20 --topk 12 --theme thesis_nude_contrast
  python training_analysis_creative_static.py --json episode_stats.json --curr curriculum_log.json --out plots_out --interactive
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

    # ─────────────────────────────
    # 6. Porcelain Sage — calm scientific safety theme
    # ─────────────────────────────
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

    # ─────────────────────────────
    # 7. Thesis Nude Contrast — minimal but clearer than beige-only palettes
    # ─────────────────────────────
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
    },

    # ─────────────────────────────
    # 8. Oat Blueprint — soft engineering notebook theme
    # ─────────────────────────────
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
    """Standard legend formatting with high-contrast scatter markers."""
    legend = ax.legend(
        loc=loc,
        fontsize=fontsize,
        facecolor=PAPER_WHITE,
        edgecolor=EDGE_CLR,
        labelcolor=TEXT_CLR,
        scatterpoints=1,
        handlelength=2.2,
        handletextpad=0.8,
        borderpad=0.7,
    )
    legend.get_frame().set_alpha(0.97)
    legend.get_frame().set_linewidth(1.0)

    # Matplotlib preserves the original scatter alpha and marker size in the
    # legend. For density clouds this can make the legend dots almost invisible
    # (for example, s=2 and alpha=0.22). Strengthen only the legend artists;
    # the plotted data remain unchanged.
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", [])

    for handle in handles:
        # PathCollection-like artists (scatter plots).
        if hasattr(handle, "get_sizes") and hasattr(handle, "set_sizes"):
            handle.set_sizes([58])
            handle.set_alpha(1.0)
            if hasattr(handle, "set_edgecolors"):
                handle.set_edgecolors(TEXT_CLR)
            if hasattr(handle, "set_linewidths"):
                handle.set_linewidths([0.85])

        # Patch-like artists (filled areas and bars).
        elif hasattr(handle, "set_facecolor") and hasattr(handle, "set_edgecolor"):
            handle.set_alpha(0.95)
            handle.set_edgecolor(TEXT_CLR)
            if hasattr(handle, "set_linewidth"):
                handle.set_linewidth(0.7)

        # Line artists.
        elif hasattr(handle, "set_alpha"):
            handle.set_alpha(1.0)


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


def rolling_quantile(x, w, q):
    """Rolling quantile that safely ignores NaN/Inf values."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    out = np.full(len(x), np.nan, dtype=float)
    for i in range(len(x)):
        j0 = max(0, i - w + 1)
        win = x[j0:i + 1]
        win = win[np.isfinite(win)]
        if len(win) > 0:
            out[i] = np.quantile(win, q)
    return out


def safe_divide(a, b, default=np.nan):
    """Safe scalar division for derived episode metrics."""
    try:
        a = float(a)
        b = float(b)
        if not np.isfinite(a) or not np.isfinite(b) or b == 0:
            return default
        return a / b
    except Exception:
        return default


def infer_stage_transitions_from_episodes(episodes):
    """Fallback stage-boundary detection when no curriculum log is provided."""
    if not episodes:
        return []

    eps = [int(safe_float(e.get("episode"), i + 1)) for i, e in enumerate(episodes)]
    stages = []
    for e in episodes:
        s = e.get("curriculum_stage", None)
        try:
            stages.append(int(float(s)))
        except Exception:
            stages.append(-1)

    transitions = []
    prev = stages[0]
    for i in range(1, len(stages)):
        if stages[i] != prev:
            transitions.append(eps[i])
            prev = stages[i]
    return transitions


def get_stage_groups(episodes):
    """Return ordered stage ids, compact labels, long labels, and grouped episodes."""
    stage_map = defaultdict(list)
    stage_names = {}

    for e in episodes:
        s = e.get("curriculum_stage", None)
        try:
            s = int(float(s))
        except Exception:
            s = -1

        name = e.get("curriculum_stage_name", None)
        if not name:
            name = f"Stage {s + 1}" if s >= 0 else "Unknown Stage"

        stage_map[s].append(e)
        stage_names[s] = str(name)

    ordered = sorted(stage_map.keys())
    short_labels = [f"S{s + 1}" if s >= 0 else "U" for s in ordered]
    long_labels = [stage_names[s] for s in ordered]
    return ordered, short_labels, long_labels, stage_map


def style_violinplot(vp, color):
    """Apply the active theme to a Matplotlib violin plot."""
    for body in vp.get("bodies", []):
        body.set_facecolor(color)
        body.set_edgecolor(TEXT_CLR)
        body.set_alpha(0.55)
        body.set_linewidth(0.8)

    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in vp:
            try:
                vp[key].set_edgecolor(TEXT_CLR)
                vp[key].set_linewidth(1.0)
            except Exception:
                pass


def extract_xy_path(e, key):
    """Extract an Nx2 path array from an episode dict, if the path exists."""
    p = e.get(key, None)
    if not isinstance(p, list) or len(p) < 2:
        return None

    arr = []
    for item in p:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = safe_float(item[0]), safe_float(item[1])
            if np.isfinite(x) and np.isfinite(y):
                arr.append([x, y])

    if len(arr) < 2:
        return None
    return np.asarray(arr, dtype=float)


def choose_storyboard_episodes(episodes):
    """Pick one representative collision, timeout, and goal episode if available."""
    by_reason = defaultdict(list)
    for e in episodes:
        by_reason[str(e.get("reason", "unknown"))].append(e)

    picks = []

    if by_reason.get("collision"):
        collision = sorted(
            by_reason["collision"],
            key=lambda x: safe_float(x.get("reward"), np.inf)
        )
        picks.append(collision[0])

    if by_reason.get("timeout"):
        timeout = sorted(
            by_reason["timeout"],
            key=lambda x: safe_float(x.get("reward"), np.inf)
        )
        picks.append(timeout[0])

    if by_reason.get("goal"):
        goal = sorted(
            by_reason["goal"],
            key=lambda x: safe_float(x.get("reward"), -np.inf),
            reverse=True
        )
        picks.append(goal[0])

    return picks


def finite_values(episodes, key):
    vals = [safe_float(e.get(key)) for e in episodes]
    return np.asarray([v for v in vals if np.isfinite(v)], dtype=float)


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
    """Tableau-style density cloud with a separate high-contrast legend proxy.

    Dense regions become saturated while sparse regions stay faint. The data
    layers intentionally remain subtle, but the legend marker is drawn as an
    independent, fully opaque proxy so that pale MUJI colors remain readable.
    """
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
    for s, a in layers[:n_layers]:
        ax.scatter(
            x,
            y,
            s=s,
            alpha=a,
            color=color,
            linewidths=0,
            zorder=2,
            label="_nolegend_",
        )

    # Do not use the smallest/faintest cloud layer as the legend handle.
    # An empty scatter creates a clean proxy without adding points to the axes.
    if label:
        ax.scatter(
            [],
            [],
            s=58,
            alpha=1.0,
            color=color,
            edgecolors=TEXT_CLR,
            linewidths=0.85,
            marker="o",
            label=label,
            zorder=4,
        )


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
# THESIS-GRADE ADDITIONAL PLOTS
# ═══════════════════════════════════════════

def plot_reward_quantile_band(episodes, stage_transitions, outdir, w):
    """Reward with raw points, rolling median, and rolling quantile bands."""
    ep = np.array([safe_float(e.get("episode")) for e in episodes], dtype=float)
    rew = np.array([safe_float(e.get("reward")) for e in episodes], dtype=float)

    valid = np.isfinite(ep) & np.isfinite(rew)
    if not valid.any():
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(ep[valid], rew[valid], s=10, alpha=0.16, color=EDGE_CLR,
               linewidths=0, label="Episode reward")

    q10 = rolling_quantile(rew, w, 0.10)
    q25 = rolling_quantile(rew, w, 0.25)
    q50 = rolling_quantile(rew, w, 0.50)
    q75 = rolling_quantile(rew, w, 0.75)
    q90 = rolling_quantile(rew, w, 0.90)

    ax.fill_between(ep, q10, q90, color=PRIMARY, alpha=0.08, label="10–90% band")
    ax.fill_between(ep, q25, q75, color=PRIMARY, alpha=0.18, label="25–75% band")
    line_with_shadow(ax, ep, q50, PRIMARY, lw=2.6, label=f"Rolling median (w={w})")

    add_stage_borders(ax, stage_transitions)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    set_title(ax, "Reward Distribution Over Training")
    add_legend(ax, loc="lower right")

    savefig(os.path.join(outdir, "17_reward_quantile_band.png"))


def plot_stagewise_distribution_summary(episodes, outdir):
    """Stage-wise distribution summary for key thesis metrics."""
    ordered, short_labels, long_labels, stage_map = get_stage_groups(episodes)
    if len(ordered) == 0:
        return

    rng = np.random.default_rng(0)

    metric_specs = [
        ("reward", "Reward by Curriculum Stage", "Reward", PRIMARY),
        ("length", "Episode Length by Curriculum Stage", "Episode length", SECONDARY),
        ("_shield_fraction", "Shield Fraction by Curriculum Stage", "Shield interventions / episode length", ACCENT),
        ("ep_min_obstacle_depth", "Minimum Obstacle Depth by Stage", "Min obstacle depth", REASON_COLORS["collision"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (key, title, ylabel, color) in zip(axes, metric_specs):
        data = []
        positions = []

        for i, s in enumerate(ordered, start=1):
            vals = []
            for e in stage_map[s]:
                if key == "_shield_fraction":
                    v = safe_divide(e.get("shield_interventions_ep", 0), e.get("length", np.nan))
                else:
                    v = safe_float(e.get(key))
                if np.isfinite(v):
                    vals.append(v)

            if len(vals) > 0:
                data.append(vals)
                positions.append(i)

        if len(data) == 0:
            ax.set_visible(False)
            continue

        vp = ax.violinplot(
            data,
            positions=positions,
            widths=0.8,
            showmeans=False,
            showmedians=True,
            showextrema=True
        )
        style_violinplot(vp, color)

        for pos, vals in zip(positions, data):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                s=8,
                alpha=0.20,
                color=TICK_CLR,
                linewidths=0
            )

        ax.set_xticks(range(1, len(short_labels) + 1))
        ax.set_xticklabels(short_labels)
        ax.set_xlabel("Curriculum stage")
        ax.set_ylabel(ylabel)
        set_title(ax, title)

    savefig(os.path.join(outdir, "18_stagewise_distribution_summary.png"))


def plot_stagewise_outcome_composition(episodes, outdir):
    """Stacked bar chart of termination reason composition by curriculum stage."""
    ordered, short_labels, long_labels, stage_map = get_stage_groups(episodes)
    if len(ordered) == 0:
        return

    all_reasons = {str(e.get("reason", "unknown")) for e in episodes}
    reason_order = ["goal", "timeout", "collision"] + sorted(
        [r for r in all_reasons if r not in ("goal", "timeout", "collision")]
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(ordered), dtype=float)
    x = np.arange(len(ordered))

    for r in reason_order:
        vals = []
        for s in ordered:
            subset = stage_map[s]
            total = len(subset)
            cnt = sum(1 for e in subset if str(e.get("reason", "unknown")) == r)
            vals.append(cnt / total if total > 0 else 0.0)

        vals = np.asarray(vals, dtype=float)
        if np.all(vals == 0):
            continue
        ax.bar(
            x, vals, bottom=bottom,
            color=reason_color(r),
            alpha=0.85,
            width=0.72,
            label=r.capitalize()
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Curriculum stage")
    ax.set_ylabel("Fraction of episodes")
    set_title(ax, "Termination Reason Composition by Curriculum Stage")
    add_legend(ax, loc="upper right")

    savefig(os.path.join(outdir, "19_stagewise_outcome_composition.png"))


def plot_safety_performance_tradeoff(episodes, outdir):
    """Reward vs shield fraction, separated by curriculum stage and colored by termination reason."""
    ordered, short_labels, long_labels, stage_map = get_stage_groups(episodes)
    if len(ordered) == 0:
        return

    n = len(ordered)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    reason_order = ["goal", "timeout", "collision"]

    for ax_idx, s in enumerate(ordered):
        ax = axes[ax_idx]
        subset = stage_map[s]
        if len(subset) == 0:
            ax.set_visible(False)
            continue

        x = np.array([safe_divide(e.get("shield_interventions_ep", 0), e.get("length", np.nan)) for e in subset], dtype=float)
        y = np.array([safe_float(e.get("reward")) for e in subset], dtype=float)
        l = np.array([safe_float(e.get("length")) for e in subset], dtype=float)
        reasons = np.array([str(e.get("reason", "unknown")) for e in subset])

        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            ax.set_visible(False)
            continue

        x = x[valid]
        y = y[valid]
        l = l[valid]
        reasons = reasons[valid]

        if np.isfinite(l).any():
            lmin, lmax = np.nanmin(l), np.nanmax(l)
            if lmax > lmin:
                sizes = 20 + 90 * (l - lmin) / (lmax - lmin)
            else:
                sizes = np.full_like(l, 45.0)
        else:
            sizes = np.full_like(y, 45.0)

        sorted_reasons = sorted(set(reasons), key=lambda rr: reason_order.index(rr) if rr in reason_order else 99)
        for r in sorted_reasons:
            mask = reasons == r
            ax.scatter(
                x[mask], y[mask],
                s=sizes[mask],
                alpha=0.36,
                color=reason_color(r),
                edgecolors=PAPER_WHITE,
                linewidths=0.5,
                label=r.capitalize()
            )

        x_med = np.nanmedian(x)
        y_med = np.nanmedian(y)
        ax.axvline(x_med, color=EDGE_CLR, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axhline(y_med, color=EDGE_CLR, linestyle="--", linewidth=1.0, alpha=0.8)

        ax.set_xlabel("Shield interventions / episode length")
        ax.set_ylabel("Reward")
        set_title(ax, f"Safety–Performance Tradeoff ({short_labels[ax_idx]})")

        if ax_idx == 0:
            add_legend(ax, loc="best")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    savefig(os.path.join(outdir, "20_safety_performance_tradeoff.png"))


def plot_reward_density_hexbin(episodes, stage_transitions, outdir, w):
    """Reward density map using hexagonal binning plus rolling median."""
    ep = np.array([safe_float(e.get("episode")) for e in episodes], dtype=float)
    rew = np.array([safe_float(e.get("reward")) for e in episodes], dtype=float)

    valid = np.isfinite(ep) & np.isfinite(rew)
    if not valid.any():
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    hb = ax.hexbin(
        ep[valid], rew[valid],
        gridsize=45,
        mincnt=1,
        cmap="Greys",
        linewidths=0,
        alpha=0.95
    )
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Count per bin")

    q50 = rolling_quantile(rew, w, 0.50)
    q25 = rolling_quantile(rew, w, 0.25)
    q75 = rolling_quantile(rew, w, 0.75)

    ax.fill_between(ep, q25, q75, color=PRIMARY, alpha=0.18, label="25–75% band")
    line_with_shadow(ax, ep, q50, PRIMARY, lw=2.4, label=f"Rolling median (w={w})")

    add_stage_borders(ax, stage_transitions)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    set_title(ax, "Reward Density Over Training")
    add_legend(ax, loc="lower right")

    savefig(os.path.join(outdir, "21_reward_density_hexbin.png"))


def plot_episode_storyboard(episodes, outdir):
    """One-page representative behavior storyboard for failure/transition/success episodes."""
    picks = choose_storyboard_episodes(episodes)
    if len(picks) == 0:
        return

    fig, axes = plt.subplots(1, len(picks), figsize=(6 * len(picks), 6), squeeze=False)
    axes = axes.flatten()

    for ax, e in zip(axes, picks):
        ep_id = int(safe_float(e.get("episode"), 0))
        reason = str(e.get("reason", "unknown"))
        reward = safe_float(e.get("reward"))
        length = safe_float(e.get("length"))
        shield_frac = safe_divide(e.get("shield_interventions_ep", 0), e.get("length", np.nan))
        min_depth = safe_float(e.get("ep_min_obstacle_depth"))

        rsx, rsy = vec2(e.get("robot_start"))
        rfx, rfy = vec2(e.get("robot_final"))
        gsx, gsy = vec2(e.get("goal_start"))
        gfx, gfy = vec2(e.get("goal_final"))

        rpath = extract_xy_path(e, "robot_path")
        gpath = extract_xy_path(e, "goal_path")

        if rpath is not None:
            ax.plot(rpath[:, 0], rpath[:, 1], color=PRIMARY, linewidth=2.4, alpha=0.85, label="Robot path")
        elif np.isfinite(rsx) and np.isfinite(rfx):
            ax.plot([rsx, rfx], [rsy, rfy], color=PRIMARY, linewidth=2.4, alpha=0.85, label="Robot path")

        if gpath is not None:
            ax.plot(gpath[:, 0], gpath[:, 1], color=REASON_COLORS["goal"], linewidth=2.0, alpha=0.75, label="Goal path")
        elif np.isfinite(gsx) and np.isfinite(gfx):
            ax.plot([gsx, gfx], [gsy, gfy], color=REASON_COLORS["goal"], linewidth=2.0, alpha=0.75, label="Goal path")

        if np.isfinite(rsx):
            ax.scatter([rsx], [rsy], s=90, color=PRIMARY, edgecolors=PAPER_WHITE, linewidths=1.2, label="Robot start")
        if np.isfinite(rfx):
            ax.scatter([rfx], [rfy], s=90, marker="x", color=PRIMARY, linewidths=2.4, label="Robot end")

        if np.isfinite(gsx):
            ax.scatter([gsx], [gsy], s=95, marker="^", color=REASON_COLORS["goal"], label="Goal start")
        if np.isfinite(gfx):
            ax.scatter([gfx], [gfy], s=95, marker="s", color=REASON_COLORS["goal"], label="Goal end")

        slots = e.get("obstacles_world_slots", [])
        ox, oy = [], []
        for slot in slots:
            if isinstance(slot, dict) and slot.get("valid", True):
                wx, wy = safe_float(slot.get("wx")), safe_float(slot.get("wy"))
                if np.isfinite(wx) and np.isfinite(wy):
                    ox.append(wx)
                    oy.append(wy)

        if ox:
            ax.scatter(ox, oy, s=90, marker="D", color=REASON_COLORS["collision"], alpha=0.75, label="Obstacles")

        txt = (
            f"Episode: {ep_id}\n"
            f"Reason: {reason}\n"
            f"Reward: {reward:.1f}\n"
            f"Length: {int(length) if np.isfinite(length) else 'NA'}\n"
            f"Shield frac: {shield_frac:.3f}\n"
            f"Min depth: {min_depth:.3f}"
        )
        ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=PAPER_WHITE,
                edgecolor=EDGE_CLR,
                alpha=0.90
            )
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        set_title(ax, f"{reason.capitalize()} representative")

    # Avoid repeating legend in every panel.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best", fontsize=8, facecolor=PAPER_WHITE,
                       edgecolor=EDGE_CLR, labelcolor=TEXT_CLR)

    savefig(os.path.join(outdir, "22_episode_storyboard.png"))


def plot_metric_correlation_matrix(episodes, outdir):
    """Compact correlation heatmap for the real scalar fields available in episode_stats."""
    metric_keys = [
        "reward", "length", "time", "min_dist", "ttc",
        "v_cmd", "w_cmd",
        "ep_min_obstacle_depth", "ep_min_ultrasonic_distance",
        "shield_interventions_ep",
        "shield_interventions_caution_ep", "shield_interventions_danger_ep", "shield_interventions_critical_ep",
        "ep_mean_jerk_v", "ep_mean_jerk_w", "ep_mean_ttc", "deadlock_escapes_ep"
    ]

    cols = []
    names = []
    for key in metric_keys:
        vals = np.asarray([safe_float(e.get(key)) for e in episodes], dtype=float)
        if np.isfinite(vals).sum() >= 3 and np.nanstd(vals[np.isfinite(vals)]) > 0:
            cols.append(vals)
            names.append(key)

    if len(cols) < 2:
        return

    X = np.vstack(cols).T
    # Pairwise finite correlation.
    C = np.full((len(names), len(names)), np.nan, dtype=float)
    for i in range(len(names)):
        for j in range(len(names)):
            xi, xj = X[:, i], X[:, j]
            m = np.isfinite(xi) & np.isfinite(xj)
            if m.sum() >= 3 and np.std(xi[m]) > 0 and np.std(xj[m]) > 0:
                C[i, j] = np.corrcoef(xi[m], xj[m])[0, 1]

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson correlation")

    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    set_title(ax, "Episode Metric Correlation Matrix")

    savefig(os.path.join(outdir, "23_metric_correlation_matrix.png"))


def create_interactive_plotly_dashboard(episodes, outdir):
    """Optional interactive HTML dashboard. Skips cleanly if Plotly is unavailable."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        print("Plotly is not installed; skipping interactive dashboard.")
        return

    ep = np.array([safe_float(e.get("episode")) for e in episodes], dtype=float)
    reward = np.array([safe_float(e.get("reward")) for e in episodes], dtype=float)
    length = np.array([safe_float(e.get("length")) for e in episodes], dtype=float)
    shield_frac = np.array([
        safe_divide(e.get("shield_interventions_ep", 0), e.get("length", np.nan))
        for e in episodes
    ], dtype=float)
    reasons = [str(e.get("reason", "unknown")) for e in episodes]
    stages = [str(e.get("curriculum_stage", "unknown")) for e in episodes]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Reward over episodes",
            "Safety–performance tradeoff",
            "Episode length over episodes",
            "Shield fraction over episodes"
        )
    )

    hover_text = [
        f"Episode: {int(ep[i]) if np.isfinite(ep[i]) else 'NA'}<br>"
        f"Reason: {reasons[i]}<br>"
        f"Stage: {stages[i]}<br>"
        f"Reward: {reward[i]:.3f}<br>"
        f"Length: {length[i]:.0f}<br>"
        f"Shield frac: {shield_frac[i]:.4f}"
        for i in range(len(episodes))
    ]

    fig.add_trace(
        go.Scatter(x=ep, y=reward, mode="markers+lines", name="Reward",
                   text=hover_text, hoverinfo="text", opacity=0.65),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=shield_frac, y=reward, mode="markers", name="Reward vs shield",
                   text=hover_text, hoverinfo="text", opacity=0.65),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=ep, y=length, mode="markers+lines", name="Length",
                   text=hover_text, hoverinfo="text", opacity=0.65),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=ep, y=shield_frac, mode="markers+lines", name="Shield fraction",
                   text=hover_text, hoverinfo="text", opacity=0.65),
        row=2, col=2
    )

    fig.update_layout(
        title="Interactive Training Dashboard",
        template="plotly_white",
        height=850,
        showlegend=False
    )
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_xaxes(title_text="Shield interventions / length", row=1, col=2)
    fig.update_yaxes(title_text="Reward", row=1, col=2)
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Length", row=2, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=2)
    fig.update_yaxes(title_text="Shield fraction", row=2, col=2)

    html_path = os.path.join(outdir, "interactive_training_dashboard.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"  saved: {html_path}")




# ═══════════════════════════════════════════
# EXTENDED DIAGNOSTIC PLOTS — MODEL PROGRESS EDITION
# Added without removing or changing the previous plot suite.
# ═══════════════════════════════════════════

def _episode_array(episodes):
    return np.array([safe_float(e.get("episode"), i + 1) for i, e in enumerate(episodes)], dtype=float)


def _reason_array(episodes):
    return np.array([str(e.get("reason", "unknown")) for e in episodes], dtype=object)


def _metric_array(episodes, key, default=np.nan):
    return np.array([safe_float(e.get(key), default) for e in episodes], dtype=float)


def _shield_fraction_array(episodes):
    vals = []
    for e in episodes:
        vals.append(safe_divide(e.get("shield_interventions_ep", 0), e.get("length", np.nan)))
    return np.array(vals, dtype=float)


def _shield_severity_array(episodes, normalize_by_length=True):
    vals = []
    for e in episodes:
        caution = safe_float(e.get("shield_interventions_caution_ep", 0), 0)
        danger = safe_float(e.get("shield_interventions_danger_ep", 0), 0)
        critical = safe_float(e.get("shield_interventions_critical_ep", 0), 0)
        score = caution + 2.0 * danger + 3.0 * critical
        if normalize_by_length:
            score = safe_divide(score, 3.0 * safe_float(e.get("length"), np.nan))
        vals.append(score)
    return np.array(vals, dtype=float)


def _total_jerk_array(episodes):
    jv = _metric_array(episodes, "ep_mean_jerk_v")
    jw = _metric_array(episodes, "ep_mean_jerk_w")
    return np.sqrt(jv ** 2 + jw ** 2)


def _clearance_array(episodes):
    """Use depth first; fall back to ultrasonic if depth is missing."""
    depth = _metric_array(episodes, "ep_min_obstacle_depth")
    ultra = _metric_array(episodes, "ep_min_ultrasonic_distance")
    out = depth.copy()
    mask = ~np.isfinite(out)
    out[mask] = ultra[mask]
    return out


def _finite_minmax_norm(x, invert=False):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    valid = np.isfinite(x)
    if not valid.any():
        return out
    lo = np.nanmin(x[valid])
    hi = np.nanmax(x[valid])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out[valid] = 0.5
    else:
        out[valid] = (x[valid] - lo) / (hi - lo)
    if invert:
        out[valid] = 1.0 - out[valid]
    return out


def _scaled_sizes(values, min_size=25, max_size=130):
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), (min_size + max_size) / 2.0, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        lo = np.nanmin(values[valid])
        hi = np.nanmax(values[valid])
        if hi > lo:
            out[valid] = min_size + (max_size - min_size) * (values[valid] - lo) / (hi - lo)
        else:
            out[valid] = (min_size + max_size) / 2.0
    return out


def _finite_ttc_array(episodes, cap_percentile=95):
    ttc = _metric_array(episodes, "ep_mean_ttc")
    # fall back to instantaneous ttc if ep_mean_ttc is missing
    fallback = _metric_array(episodes, "ttc")
    mask = ~np.isfinite(ttc)
    ttc[mask] = fallback[mask]
    finite = ttc[np.isfinite(ttc)]
    if len(finite) == 0:
        return ttc
    cap = np.nanpercentile(finite, cap_percentile)
    if not np.isfinite(cap) or cap <= 0:
        cap = np.nanmax(finite)
    ttc = np.where(np.isfinite(ttc), np.minimum(ttc, cap), np.nan)
    return ttc


def _draw_episode_map(ax, e, title=None):
    """Small map drawer for storyboard-style diagnostic panels."""
    reason = str(e.get("reason", "unknown"))
    rpath = extract_xy_path(e, "robot_path")
    gpath = extract_xy_path(e, "goal_path")
    rsx, rsy = vec2(e.get("robot_start"))
    rfx, rfy = vec2(e.get("robot_final"))
    gsx, gsy = vec2(e.get("goal_start"))
    gfx, gfy = vec2(e.get("goal_final"))

    if rpath is not None:
        ax.plot(rpath[:, 0], rpath[:, 1], color=PRIMARY, linewidth=2.2, alpha=0.85, label="Robot")
    elif np.isfinite(rsx) and np.isfinite(rfx):
        ax.plot([rsx, rfx], [rsy, rfy], color=PRIMARY, linewidth=2.2, alpha=0.85, label="Robot")

    if gpath is not None:
        ax.plot(gpath[:, 0], gpath[:, 1], color=REASON_COLORS.get("goal", ACCENT), linewidth=1.8, alpha=0.75, label="Goal")
    elif np.isfinite(gsx) and np.isfinite(gfx):
        ax.plot([gsx, gfx], [gsy, gfy], color=REASON_COLORS.get("goal", ACCENT), linewidth=1.8, alpha=0.75, label="Goal")

    if np.isfinite(rsx):
        ax.scatter([rsx], [rsy], s=70, marker="o", color=PRIMARY, edgecolors=PAPER_WHITE, linewidths=1.0)
    if np.isfinite(rfx):
        ax.scatter([rfx], [rfy], s=80, marker="x", color=PRIMARY, linewidths=2.2)
    if np.isfinite(gsx):
        ax.scatter([gsx], [gsy], s=70, marker="^", color=REASON_COLORS.get("goal", ACCENT))
    if np.isfinite(gfx):
        ax.scatter([gfx], [gfy], s=70, marker="s", color=REASON_COLORS.get("goal", ACCENT))

    slots = e.get("obstacles_world_slots", [])
    ox, oy = [], []
    for slot in slots:
        if isinstance(slot, dict) and slot.get("valid", True):
            wx, wy = safe_float(slot.get("wx")), safe_float(slot.get("wy"))
            if np.isfinite(wx) and np.isfinite(wy):
                ox.append(wx); oy.append(wy)
    if ox:
        ax.scatter(ox, oy, s=70, marker="D", color=REASON_COLORS.get("collision", SECONDARY), alpha=0.75, label="Obstacle")

    ep_id = safe_float(e.get("episode"), np.nan)
    reward = safe_float(e.get("reward"), np.nan)
    length = safe_float(e.get("length"), np.nan)
    shield_frac = safe_divide(e.get("shield_interventions_ep", 0), e.get("length", np.nan))
    info = f"Ep {int(ep_id) if np.isfinite(ep_id) else 'NA'}\n{reason}\nR={reward:.0f}\nL={int(length) if np.isfinite(length) else 'NA'}\nshield={shield_frac:.3f}"
    ax.text(0.02, 0.98, info, transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=PAPER_WHITE, edgecolor=EDGE_CLR, alpha=0.90))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    set_title(ax, title if title else reason.capitalize())


def plot_learning_efficiency(episodes, stage_transitions, outdir, w):
    """Cumulative learning efficiency: success accumulation and cumulative success rate."""
    ep = _episode_array(episodes)
    reasons = _reason_array(episodes)
    success = (reasons == "goal").astype(float)
    if len(ep) == 0:
        return

    cum_goals = np.cumsum(success)
    cum_rate = cum_goals / np.arange(1, len(success) + 1)
    roll_rate = rolling_mean(success, w)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()
    line_with_shadow(ax, ep, cum_rate, REASON_COLORS.get("goal", PRIMARY), lw=2.4, label="Cumulative success rate")
    line_with_shadow(ax, ep, roll_rate, PRIMARY, lw=2.0, label=f"Rolling success rate (w={w})")
    ax2.plot(ep, cum_goals, color=SECONDARY, linewidth=2.2, alpha=0.75, label="Cumulative goals")

    add_stage_borders(ax, stage_transitions)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate")
    ax2.set_ylabel("Cumulative goal count", color=TICK_CLR)
    set_title(ax, "Learning Efficiency: Success Accumulation")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", facecolor=PAPER_WHITE, edgecolor=EDGE_CLR, labelcolor=TEXT_CLR)
    savefig(os.path.join(outdir, "24_learning_efficiency.png"))


def plot_rolling_failure_probability(episodes, stage_transitions, outdir, w):
    """Separate rolling probabilities for goal, timeout, collision, and unknown outcomes."""
    ep = _episode_array(episodes)
    reasons = _reason_array(episodes)
    if len(ep) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ordered = ["goal", "timeout", "collision"] + sorted([r for r in set(reasons) if r not in ("goal", "timeout", "collision")])
    for r in ordered:
        y = rolling_mean((reasons == r).astype(float), w)
        if np.isfinite(y).any():
            line_with_shadow(ax, ep, y, reason_color(r), lw=2.2, label=f"P({r})")

    add_stage_borders(ax, stage_transitions)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling probability")
    set_title(ax, "Rolling Outcome Probability")
    add_legend(ax, loc="best")
    savefig(os.path.join(outdir, "25_rolling_failure_probability.png"))


def plot_policy_maturity_index(episodes, stage_transitions, outdir, w):
    """Composite diagnostic index: performance + safety + smoothness + shield independence."""
    ep = _episode_array(episodes)
    reasons = _reason_array(episodes)
    if len(ep) == 0:
        return

    success = rolling_mean((reasons == "goal").astype(float), w)
    collision = rolling_mean((reasons == "collision").astype(float), w)
    shield = rolling_mean(_shield_fraction_array(episodes), w)
    jerk = rolling_mean(_total_jerk_array(episodes), w)
    clearance = rolling_mean(_clearance_array(episodes), w)

    shield_independence = _finite_minmax_norm(shield, invert=True)
    smoothness = _finite_minmax_norm(jerk, invert=True)
    clearance_norm = _finite_minmax_norm(clearance, invert=False)
    safety = 1.0 - collision

    components = np.vstack([success, safety, shield_independence, smoothness, clearance_norm])
    maturity = np.nanmean(components, axis=0)
    if not np.isfinite(maturity).any():
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(ep, 0, maturity, color=PRIMARY, alpha=0.12)
    line_with_shadow(ax, ep, maturity, PRIMARY, lw=2.8, label="Policy maturity index")
    ax.plot(ep, success, color=REASON_COLORS.get("goal", PRIMARY), alpha=0.45, linewidth=1.2, label="Success component")
    ax.plot(ep, safety, color=REASON_COLORS.get("collision", SECONDARY), alpha=0.35, linewidth=1.2, label="Safety component")
    ax.plot(ep, shield_independence, color=SECONDARY, alpha=0.35, linewidth=1.2, label="Shield-independence component")

    add_stage_borders(ax, stage_transitions)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Index / normalized component")
    set_title(ax, "Policy Maturity Index")
    add_legend(ax, loc="lower right")
    savefig(os.path.join(outdir, "26_policy_maturity_index.png"))


def plot_shield_dependency_decay(episodes, stage_transitions, outdir, w):
    """Success rate and shield fraction on the same timeline."""
    ep = _episode_array(episodes)
    reasons = _reason_array(episodes)
    if len(ep) == 0:
        return

    success = rolling_mean((reasons == "goal").astype(float), w)
    shield = rolling_mean(_shield_fraction_array(episodes), w)
    if not (np.isfinite(success).any() or np.isfinite(shield).any()):
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()
    line_with_shadow(ax, ep, success, REASON_COLORS.get("goal", PRIMARY), lw=2.6, label=f"Success rate (w={w})")
    ax2.plot(ep, shield, color=SECONDARY, linewidth=2.4, alpha=0.85, label=f"Shield fraction (w={w})")

    add_stage_borders(ax, stage_transitions)
    ax.set_ylim(-0.03, 1.03)
    finite_shield = shield[np.isfinite(shield)]
    if len(finite_shield):
        ax2.set_ylim(0, max(0.05, np.nanmax(finite_shield) * 1.15))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate")
    ax2.set_ylabel("Shield interventions / episode length")
    set_title(ax, "Shield Dependency Decay")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", facecolor=PAPER_WHITE, edgecolor=EDGE_CLR, labelcolor=TEXT_CLR)
    savefig(os.path.join(outdir, "27_shield_dependency_decay.png"))


def plot_shield_severity_evolution(episodes, stage_transitions, outdir, w):
    """Rolling distribution of caution/danger/critical interventions plus severity score."""
    ep = _episode_array(episodes)
    if len(ep) == 0:
        return

    caution = _metric_array(episodes, "shield_interventions_caution_ep", default=0)
    danger = _metric_array(episodes, "shield_interventions_danger_ep", default=0)
    critical = _metric_array(episodes, "shield_interventions_critical_ep", default=0)
    total = caution + danger + critical
    if not np.isfinite(total).any():
        return

    with np.errstate(invalid="ignore", divide="ignore"):
        fc = np.where(total > 0, caution / total, np.nan)
        fd = np.where(total > 0, danger / total, np.nan)
        fk = np.where(total > 0, critical / total, np.nan)

    fc = rolling_mean(fc, w)
    fd = rolling_mean(fd, w)
    fk = rolling_mean(fk, w)
    severity = rolling_mean(_shield_severity_array(episodes, normalize_by_length=True), w)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    bottom = np.zeros(len(ep))
    for arr, label, color in [
        (fc, "Caution", SHIELD_COLORS.get("caution", ACCENT)),
        (fd, "Danger", SHIELD_COLORS.get("danger", SECONDARY)),
        (fk, "Critical", SHIELD_COLORS.get("critical", TICK_CLR)),
    ]:
        arr_clean = np.nan_to_num(arr, nan=0.0)
        axes[0].fill_between(ep, bottom, bottom + arr_clean, color=color, alpha=0.75, label=label)
        bottom += arr_clean

    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of shield interventions")
    set_title(axes[0], "Shield Severity Composition")
    add_legend(axes[0], loc="upper right")
    line_with_shadow(axes[1], ep, severity, SECONDARY, lw=2.5, label="Normalized severity score")
    axes[1].set_ylabel("Severity score / length")
    axes[1].set_xlabel("Episode")
    set_title(axes[1], "Shield Severity Score")
    add_legend(axes[1], loc="best")
    add_stage_borders(axes[0], stage_transitions)
    add_stage_borders(axes[1], stage_transitions)
    savefig(os.path.join(outdir, "28_shield_severity_evolution.png"))


def plot_success_without_shield(episodes, outdir, clean_threshold=0.01):
    """Stage-wise clean vs shield-assisted success/failure categories."""
    ordered, short_labels, long_labels, stage_map = get_stage_groups(episodes)
    if len(ordered) == 0:
        return

    categories = [
        ("Clean success", lambda e, sf: str(e.get("reason", "unknown")) == "goal" and sf < clean_threshold, REASON_COLORS.get("goal", PRIMARY)),
        ("Assisted success", lambda e, sf: str(e.get("reason", "unknown")) == "goal" and sf >= clean_threshold, ACCENT),
        ("Failed with shield", lambda e, sf: str(e.get("reason", "unknown")) != "goal" and sf >= clean_threshold, SECONDARY),
        ("Failed without shield", lambda e, sf: str(e.get("reason", "unknown")) != "goal" and sf < clean_threshold, EDGE_CLR),
    ]

    x = np.arange(len(ordered))
    bottom = np.zeros(len(ordered), dtype=float)
    fig, ax = plt.subplots(figsize=(12, 6))

    for label, pred, color in categories:
        vals = []
        for s in ordered:
            subset = stage_map[s]
            total = len(subset)
            cnt = 0
            for e in subset:
                sf = safe_divide(e.get("shield_interventions_ep", 0), e.get("length", np.nan), default=np.inf)
                if pred(e, sf):
                    cnt += 1
            vals.append(cnt / total if total > 0 else 0.0)
        vals = np.asarray(vals)
        ax.bar(x, vals, bottom=bottom, width=0.72, color=color, alpha=0.85, label=label)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Curriculum stage")
    ax.set_ylabel("Fraction of episodes")
    set_title(ax, f"Success Without Shield vs Shield-Assisted Behavior (threshold={clean_threshold:.0%})")
    add_legend(ax, loc="upper right")
    savefig(os.path.join(outdir, "29_success_without_shield.png"))


def plot_reward_safety_phase_portrait(episodes, outdir, w):
    """Phase portrait of rolling shield fraction vs rolling reward."""
    ep = _episode_array(episodes)
    reward = rolling_mean(_metric_array(episodes, "reward"), w)
    shield = rolling_mean(_shield_fraction_array(episodes), w)
    valid = np.isfinite(reward) & np.isfinite(shield) & np.isfinite(ep)
    if valid.sum() < 3:
        return

    x, y, c = shield[valid], reward[valid], ep[valid]
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(x, y, c=c, cmap="Greys", s=42, alpha=0.78, edgecolors=PAPER_WHITE, linewidths=0.4)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Episode")

    step = max(1, len(x) // 18)
    for i in range(0, len(x) - step, step):
        dx = x[i + step] - x[i]
        dy = y[i + step] - y[i]
        if np.isfinite(dx) and np.isfinite(dy):
            ax.arrow(x[i], y[i], dx, dy, length_includes_head=True,
                     head_width=max(0.002, 0.015 * max(np.nanmax(x) - np.nanmin(x), 1e-6)),
                     head_length=max(0.002, 0.025 * max(np.nanmax(x) - np.nanmin(x), 1e-6)),
                     color=PRIMARY, alpha=0.35, linewidth=0.7)

    ax.set_xlabel("Rolling shield fraction")
    ax.set_ylabel("Rolling reward")
    set_title(ax, "Reward–Safety Phase Portrait")
    savefig(os.path.join(outdir, "30_reward_safety_phase_portrait.png"))


def plot_distance_outcome_boundary(episodes, outdir):
    """Reward vs proximity metrics, colored by outcome."""
    reward = _metric_array(episodes, "reward")
    depth = _metric_array(episodes, "ep_min_obstacle_depth")
    ultra = _metric_array(episodes, "ep_min_ultrasonic_distance")
    reasons = _reason_array(episodes)

    specs = [(depth, "Minimum obstacle depth"), (ultra, "Minimum ultrasonic distance")]
    if not any(np.isfinite(x).any() for x, _ in specs):
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)
    axes = axes.flatten()
    for ax, (x, xlabel) in zip(axes, specs):
        valid = np.isfinite(x) & np.isfinite(reward)
        if not valid.any():
            ax.set_visible(False)
            continue
        for r in sorted(set(reasons[valid])):
            mask = valid & (reasons == r)
            ax.scatter(x[mask], reward[mask], s=36, alpha=0.45, color=reason_color(r),
                       edgecolors=PAPER_WHITE, linewidths=0.4, label=r.capitalize())
        med = np.nanmedian(x[valid])
        ax.axvline(med, color=EDGE_CLR, linestyle="--", linewidth=1.2, alpha=0.8, label="Median distance")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Reward")
        set_title(ax, f"Outcome Boundary: {xlabel}")
    add_legend(axes[0], loc="best")
    savefig(os.path.join(outdir, "31_distance_outcome_boundary.png"))


def plot_smoothness_vs_performance(episodes, outdir):
    """Mean jerk vs reward, colored by outcome and sized by shield fraction."""
    jerk = _total_jerk_array(episodes)
    reward = _metric_array(episodes, "reward")
    shield = _shield_fraction_array(episodes)
    reasons = _reason_array(episodes)
    valid = np.isfinite(jerk) & np.isfinite(reward)
    if not valid.any():
        return

    sizes = _scaled_sizes(shield, min_size=25, max_size=150)
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in sorted(set(reasons[valid])):
        mask = valid & (reasons == r)
        ax.scatter(jerk[mask], reward[mask], s=sizes[mask], alpha=0.42,
                   color=reason_color(r), edgecolors=PAPER_WHITE, linewidths=0.5, label=r.capitalize())
    ax.set_xlabel("Total mean jerk")
    ax.set_ylabel("Reward")
    set_title(ax, "Smoothness vs Performance")
    add_legend(ax, loc="best")
    savefig(os.path.join(outdir, "32_smoothness_vs_performance.png"))


def plot_curriculum_difficulty_response(episodes, stage_transitions, outdir, w, window=None):
    """Aligned response around curriculum transitions."""
    ep = _episode_array(episodes)
    if len(stage_transitions) == 0 or len(ep) == 0:
        return
    if window is None:
        window = max(30, 3 * w)

    reasons = _reason_array(episodes)
    reward = _metric_array(episodes, "reward")
    success = (reasons == "goal").astype(float)
    collision = (reasons == "collision").astype(float)
    shield = _shield_fraction_array(episodes)

    specs = [
        (reward, "Reward", False),
        (success, "Success probability", True),
        (collision, "Collision probability", True),
        (shield, "Shield fraction", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    axes = axes.flatten()
    any_plotted = False
    for ax, (arr, ylabel, is_binary) in zip(axes, specs):
        for k, t in enumerate(stage_transitions):
            t = safe_float(t)
            if not np.isfinite(t):
                continue
            mask = (ep >= t - window) & (ep <= t + window)
            if mask.sum() < 5:
                continue
            xrel = ep[mask] - t
            y = arr[mask]
            if is_binary:
                y = rolling_mean(y, max(3, min(w, len(y))))
            else:
                y = rolling_mean(y, max(3, min(w, len(y))))
            ax.plot(xrel, y, linewidth=1.7, alpha=0.65, label=f"Transition {k + 1}")
            any_plotted = True
        ax.axvline(0, color=TICK_CLR, linestyle="--", linewidth=1.2, alpha=0.75)
        ax.set_xlabel("Episodes relative to transition")
        ax.set_ylabel(ylabel)
        set_title(ax, f"Curriculum Response: {ylabel}")
    if not any_plotted:
        plt.close(fig)
        return
    add_legend(axes[0], loc="best")
    savefig(os.path.join(outdir, "33_curriculum_difficulty_response.png"))


def plot_recovery_time_after_transition(episodes, stage_transitions, outdir, w, window=None):
    """Episodes needed after each transition to recover pre-transition median reward."""
    ep = _episode_array(episodes)
    reward = _metric_array(episodes, "reward")
    if len(stage_transitions) == 0 or not np.isfinite(reward).any():
        return
    if window is None:
        window = max(40, 4 * w)

    labels, rec_times, targets = [], [], []
    reward_roll = rolling_mean(reward, max(3, w))
    for idx, t in enumerate(stage_transitions, start=1):
        t = safe_float(t)
        if not np.isfinite(t):
            continue
        pre_mask = (ep >= t - window) & (ep < t) & np.isfinite(reward)
        post_mask = (ep >= t) & (ep <= t + window) & np.isfinite(reward_roll)
        if pre_mask.sum() < 5 or post_mask.sum() < 5:
            continue
        target = np.nanmedian(reward[pre_mask])
        post_eps = ep[post_mask]
        post_vals = reward_roll[post_mask]
        recovered = post_eps[post_vals >= target]
        labels.append(f"T{idx}")
        targets.append(target)
        rec_times.append(recovered[0] - t if len(recovered) > 0 else np.nan)

    if len(labels) == 0:
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    vals = np.array(rec_times, dtype=float)
    display_vals = np.where(np.isfinite(vals), vals, window)
    bars = ax.bar(x, display_vals, color=PRIMARY, alpha=0.82)
    for bar, val in zip(bars, vals):
        txt = f"{int(val)}" if np.isfinite(val) else f">{window}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), txt,
                ha="center", va="bottom", fontsize=9, color=TEXT_CLR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Curriculum transition")
    ax.set_ylabel("Episodes to recover pre-transition median reward")
    set_title(ax, "Recovery Time After Curriculum Transition")
    savefig(os.path.join(outdir, "34_recovery_time_after_transition.png"))


def plot_stability_window(episodes, outdir, w, window=None):
    """Fixed-window stability: reward median/IQR plus success and collision rates."""
    ep = _episode_array(episodes)
    if len(ep) == 0:
        return
    if window is None:
        window = max(50, 5 * w)

    reward = _metric_array(episodes, "reward")
    reasons = _reason_array(episodes)
    success = (reasons == "goal").astype(float)
    collision = (reasons == "collision").astype(float)

    start_ep, end_ep = int(np.nanmin(ep)), int(np.nanmax(ep))
    centers, med, q25, q75, sr, cr = [], [], [], [], [], []
    for a in range(start_ep, end_ep + 1, window):
        b = a + window - 1
        mask = (ep >= a) & (ep <= b)
        if mask.sum() < 3:
            continue
        rw = reward[mask]
        if np.isfinite(rw).any():
            centers.append((a + b) / 2)
            med.append(np.nanmedian(rw))
            q25.append(np.nanquantile(rw[np.isfinite(rw)], 0.25))
            q75.append(np.nanquantile(rw[np.isfinite(rw)], 0.75))
            sr.append(np.nanmean(success[mask]))
            cr.append(np.nanmean(collision[mask]))

    if len(centers) == 0:
        return
    centers = np.array(centers); med = np.array(med); q25 = np.array(q25); q75 = np.array(q75)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].fill_between(centers, q25, q75, color=PRIMARY, alpha=0.18, label="Reward IQR")
    line_with_shadow(axes[0], centers, med, PRIMARY, lw=2.5, label="Median reward")
    axes[0].set_ylabel("Reward")
    set_title(axes[0], f"Training Stability by {window}-Episode Windows")
    add_legend(axes[0], loc="best")

    line_with_shadow(axes[1], centers, sr, REASON_COLORS.get("goal", PRIMARY), lw=2.3, label="Success rate")
    line_with_shadow(axes[1], centers, cr, REASON_COLORS.get("collision", SECONDARY), lw=2.3, label="Collision rate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Window rate")
    set_title(axes[1], "Outcome Stability")
    add_legend(axes[1], loc="best")
    savefig(os.path.join(outdir, "35_stability_window.png"))


def plot_best_worst_episode_contrast(episodes, outdir):
    """Compact contrast of strongest/weakest representative episodes."""
    if len(episodes) == 0:
        return
    by_reason = defaultdict(list)
    for e in episodes:
        by_reason[str(e.get("reason", "unknown"))].append(e)

    picks = []
    if by_reason.get("goal"):
        picks += sorted(by_reason["goal"], key=lambda e: safe_float(e.get("reward"), -np.inf), reverse=True)[:2]
    if by_reason.get("collision"):
        picks += sorted(by_reason["collision"], key=lambda e: safe_float(e.get("reward"), np.inf))[:2]
    if by_reason.get("timeout"):
        picks += sorted(by_reason["timeout"], key=lambda e: safe_float(e.get("reward"), np.inf))[:2]
    if len(picks) == 0:
        return
    picks = picks[:6]

    ncols = min(3, len(picks))
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax, e in zip(axes, picks):
        _draw_episode_map(ax, e, title=f"{str(e.get('reason', 'unknown')).capitalize()} contrast")
    for ax in axes[len(picks):]:
        ax.set_visible(False)
    savefig(os.path.join(outdir, "36_best_worst_episode_contrast.png"))


def plot_failure_signature_radar(episodes, outdir):
    """Radar chart of normalized outcome signatures."""
    reasons = _reason_array(episodes)
    present = [r for r in ["goal", "timeout", "collision"] if r in set(reasons)]
    present += sorted([r for r in set(reasons) if r not in present and r != "unknown"])
    if len(present) == 0:
        return

    metrics = {
        "Reward": _finite_minmax_norm(_metric_array(episodes, "reward"), invert=False),
        "Efficiency": _finite_minmax_norm(_metric_array(episodes, "length"), invert=True),
        "Shield-free": _finite_minmax_norm(_shield_fraction_array(episodes), invert=True),
        "Clearance": _finite_minmax_norm(_clearance_array(episodes), invert=False),
        "Smoothness": _finite_minmax_norm(_total_jerk_array(episodes), invert=True),
        "TTC": _finite_minmax_norm(_finite_ttc_array(episodes), invert=False),
    }
    labels = list(metrics.keys())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for r in present:
        vals = []
        mask = reasons == r
        for k in labels:
            arr = metrics[k]
            vals.append(np.nanmedian(arr[mask]) if np.isfinite(arr[mask]).any() else np.nan)
        if not np.isfinite(vals).any():
            continue
        vals = np.nan_to_num(vals, nan=0.0).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.0, color=reason_color(r), label=r.capitalize())
        ax.fill(angles, vals, color=reason_color(r), alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT_CLR)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    set_title(ax, "Outcome Signature Radar")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=max(1, min(3, len(present))),
              facecolor=PAPER_WHITE, edgecolor=EDGE_CLR, labelcolor=TEXT_CLR)
    savefig(os.path.join(outdir, "37_failure_signature_radar.png"))


def plot_outcome_condition_heatmap(episodes, outdir):
    """Median metric profile by outcome type."""
    reasons = _reason_array(episodes)
    present = [r for r in ["goal", "timeout", "collision"] if r in set(reasons)]
    present += sorted([r for r in set(reasons) if r not in present])
    if len(present) == 0:
        return

    metric_defs = [
        ("Reward", _metric_array(episodes, "reward")),
        ("Length", _metric_array(episodes, "length")),
        ("Shield frac", _shield_fraction_array(episodes)),
        ("Severity", _shield_severity_array(episodes, normalize_by_length=True)),
        ("Min depth", _metric_array(episodes, "ep_min_obstacle_depth")),
        ("Min ultra", _metric_array(episodes, "ep_min_ultrasonic_distance")),
        ("Jerk", _total_jerk_array(episodes)),
        ("TTC", _finite_ttc_array(episodes)),
        ("Deadlocks", _metric_array(episodes, "deadlock_escapes_ep", default=0)),
    ]

    raw = np.full((len(present), len(metric_defs)), np.nan)
    for i, r in enumerate(present):
        mask = reasons == r
        for j, (_, arr) in enumerate(metric_defs):
            vals = arr[mask]
            if np.isfinite(vals).any():
                raw[i, j] = np.nanmedian(vals)

    norm = np.full_like(raw, np.nan, dtype=float)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        valid = np.isfinite(col)
        if not valid.any():
            continue
        lo, hi = np.nanmin(col[valid]), np.nanmax(col[valid])
        if hi > lo:
            norm[valid, j] = (col[valid] - lo) / (hi - lo)
        else:
            norm[valid, j] = 0.5

    fig, ax = plt.subplots(figsize=(13, max(4, 0.75 * len(present) + 3)))
    im = ax.imshow(norm, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Column-normalized median")
    ax.set_xticks(np.arange(len(metric_defs)))
    ax.set_xticklabels([m[0] for m in metric_defs], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(present)))
    ax.set_yticklabels([r.capitalize() for r in present])

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            if np.isfinite(raw[i, j]):
                ax.text(j, i, f"{raw[i, j]:.2g}", ha="center", va="center",
                        fontsize=8, color=TEXT_CLR)
    set_title(ax, "Outcome–Condition Heatmap")
    savefig(os.path.join(outdir, "38_outcome_condition_heatmap.png"))


def plot_sensor_agreement(episodes, outdir):
    """Ultrasonic vs obstacle-depth agreement, colored by outcome."""
    ultra = _metric_array(episodes, "ep_min_ultrasonic_distance")
    depth = _metric_array(episodes, "ep_min_obstacle_depth")
    shield = _shield_fraction_array(episodes)
    reasons = _reason_array(episodes)
    valid = np.isfinite(ultra) & np.isfinite(depth)
    if not valid.any():
        return

    sizes = _scaled_sizes(shield, min_size=25, max_size=150)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for r in sorted(set(reasons[valid])):
        mask = valid & (reasons == r)
        ax.scatter(ultra[mask], depth[mask], s=sizes[mask], alpha=0.42,
                   color=reason_color(r), edgecolors=PAPER_WHITE, linewidths=0.5, label=r.capitalize())
    if valid.sum() > 2:
        corr = np.corrcoef(ultra[valid], depth[valid])[0, 1]
        ax.text(0.02, 0.98, f"r = {corr:.2f}", transform=ax.transAxes, ha="left", va="top",
                fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor=PAPER_WHITE, edgecolor=EDGE_CLR, alpha=0.85))
    ax.set_xlabel("Episode minimum ultrasonic distance")
    ax.set_ylabel("Episode minimum obstacle depth")
    set_title(ax, "Sensor Agreement: Ultrasonic vs Depth")
    add_legend(ax, loc="best")
    savefig(os.path.join(outdir, "39_sensor_agreement.png"))


def plot_ttc_risk_map(episodes, outdir):
    """TTC vs reward, sized by shield severity."""
    ttc = _finite_ttc_array(episodes)
    reward = _metric_array(episodes, "reward")
    severity = _shield_severity_array(episodes, normalize_by_length=True)
    reasons = _reason_array(episodes)
    valid = np.isfinite(ttc) & np.isfinite(reward)
    if not valid.any():
        return

    sizes = _scaled_sizes(severity, min_size=25, max_size=160)
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in sorted(set(reasons[valid])):
        mask = valid & (reasons == r)
        ax.scatter(ttc[mask], reward[mask], s=sizes[mask], alpha=0.42,
                   color=reason_color(r), edgecolors=PAPER_WHITE, linewidths=0.5, label=r.capitalize())
    med = np.nanmedian(ttc[valid])
    ax.axvline(med, color=EDGE_CLR, linestyle="--", linewidth=1.2, alpha=0.8, label="Median TTC")
    ax.set_xlabel("Episode mean TTC (capped)")
    ax.set_ylabel("Reward")
    set_title(ax, "TTC Risk Map")
    add_legend(ax, loc="best")
    savefig(os.path.join(outdir, "40_ttc_risk_map.png"))


def plot_deadlock_diagnosis(episodes, outdir):
    """Deadlock escapes vs length, colored by outcome and sized by shield fraction."""
    deadlocks = _metric_array(episodes, "deadlock_escapes_ep", default=0)
    length = _metric_array(episodes, "length")
    shield = _shield_fraction_array(episodes)
    reasons = _reason_array(episodes)
    valid = np.isfinite(deadlocks) & np.isfinite(length)
    if not valid.any():
        return

    sizes = _scaled_sizes(shield, min_size=25, max_size=150)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.04, 0.04, size=len(deadlocks))
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in sorted(set(reasons[valid])):
        mask = valid & (reasons == r)
        ax.scatter(deadlocks[mask] + jitter[mask], length[mask], s=sizes[mask], alpha=0.42,
                   color=reason_color(r), edgecolors=PAPER_WHITE, linewidths=0.5, label=r.capitalize())
    ax.set_xlabel("Deadlock escapes per episode")
    ax.set_ylabel("Episode length")
    set_title(ax, "Deadlock Diagnosis")
    add_legend(ax, loc="best")
    savefig(os.path.join(outdir, "41_deadlock_diagnosis.png"))


# ═══════════════════════════════════════════
# THESIS ADDITIONS — VERIFIED FIELDS ONLY
# All 8 functions use only fields present in
# the fields_to_check data-availability list.
# ═══════════════════════════════════════════

def plot_stagewise_convergence_speed(episodes, stage_transitions, outdir, w, threshold=0.95):
    """Bar chart: episodes needed within each stage to first reach a rolling success threshold."""
    ordered, short_labels, long_labels, stage_map = get_stage_groups(episodes)
    if not ordered:
        return

    speeds = []
    for s in ordered:
        eps_in_stage = sorted(stage_map[s], key=lambda e: safe_float(e.get("episode"), np.inf))
        reasons = [str(e.get("reason", "unknown")) for e in eps_in_stage]
        success = np.array([1.0 if r == "goal" else 0.0 for r in reasons])
        roll = rolling_mean(success, min(w, max(1, len(success))))
        crossed = np.where(np.isfinite(roll) & (roll >= threshold))[0]
        speeds.append(crossed[0] + 1 if len(crossed) > 0 else np.nan)

    x = np.arange(len(ordered))
    bar_colors = [REASON_COLORS.get("goal", PRIMARY) if np.isfinite(s) else EDGE_CLR
                  for s in speeds]

    fig, ax = plt.subplots(figsize=(max(6, len(ordered) * 1.6), 6))
    bars = ax.bar(x, [s if np.isfinite(s) else 0 for s in speeds],
                  color=bar_colors, alpha=0.78, edgecolor=PAPER_WHITE, linewidth=1.2)

    for i, (bar, s) in enumerate(zip(bars, speeds)):
        if np.isfinite(s):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(1, bar.get_height() * 0.02),
                    str(int(s)), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=TEXT_CLR)
        else:
            ax.text(x[i], 1, "never reached", ha="center", va="bottom",
                    fontsize=9, color=TICK_CLR, fontstyle="italic")

    valid_speeds = [s for s in speeds if np.isfinite(s)]
    if valid_speeds:
        ax.axhline(np.mean(valid_speeds), color=SECONDARY, linewidth=1.5,
                   linestyle="--", alpha=0.85, label=f"Mean = {np.mean(valid_speeds):.0f} ep")
        add_legend(ax)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_xlabel("Curriculum stage")
    ax.set_ylabel(f"Episodes to first reach {threshold * 100:.0f}% rolling success")
    set_title(ax, f"Stage-wise Convergence Speed  (threshold = {threshold * 100:.0f}%,  window = {w})")
    savefig(os.path.join(outdir, "42_stagewise_convergence_speed.png"))


def plot_reward_by_outcome(episodes, stage_transitions, outdir, w):
    """Rolling reward curves split by termination reason on the same axis."""
    ep = _episode_array(episodes)
    reward = _metric_array(episodes, "reward")
    reasons = _reason_array(episodes)

    if not np.isfinite(reward).any():
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    add_stage_borders(ax, stage_transitions)

    for reason in ["goal", "collision", "timeout"]:
        mask = reasons == reason
        if not mask.any():
            continue
        r_masked = np.where(mask, reward, np.nan)
        rolled = rolling_mean(r_masked, w)
        valid = np.isfinite(rolled)
        if valid.any():
            color = REASON_COLORS.get(reason, PRIMARY)
            line_with_shadow(ax, ep[valid], rolled[valid], color, lw=2.2,
                             label=f"{reason.capitalize()} (w={w})")

    ax.axhline(0, color=EDGE_CLR, linewidth=0.8, alpha=0.6, linestyle="-")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    set_title(ax, "Reward Conditioned on Outcome")
    add_legend(ax, loc="best")
    savefig(os.path.join(outdir, "43_reward_by_outcome.png"))


def plot_ttc_per_stage_violin(episodes, outdir):
    """Violin plot of ep_mean_ttc distribution per curriculum stage."""
    ordered, short_labels, long_labels, stage_map = get_stage_groups(episodes)
    if not ordered:
        return

    violin_data, valid_labels = [], []
    for s, label in zip(ordered, short_labels):
        vals = np.array([safe_float(e.get("ep_mean_ttc")) for e in stage_map[s]], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 3:
            violin_data.append(vals)
            valid_labels.append(label)

    if not violin_data:
        return

    colors = [PRIMARY, SECONDARY, ACCENT, REASON_COLORS.get("goal", PRIMARY),
              REASON_COLORS.get("timeout", PRIMARY)]
    x_pos = np.arange(1, len(violin_data) + 1)

    fig, ax = plt.subplots(figsize=(max(6, len(violin_data) * 1.8), 7))
    vp = ax.violinplot(violin_data, positions=x_pos, showmedians=True, showextrema=True)
    for i, body in enumerate(vp.get("bodies", [])):
        c = colors[i % len(colors)]
        body.set_facecolor(c)
        body.set_edgecolor(TEXT_CLR)
        body.set_alpha(0.55)
        body.set_linewidth(0.8)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in vp:
            vp[key].set_edgecolor(TEXT_CLR)
            vp[key].set_linewidth(1.0)

    for i, d in enumerate(violin_data):
        med = np.median(d)
        ax.text(x_pos[i] + 0.08, med, f"{med:.2f}", va="center", ha="left",
                fontsize=9, color=TEXT_CLR)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_labels)
    ax.set_xlabel("Curriculum stage")
    ax.set_ylabel("Mean TTC per episode")
    set_title(ax, "TTC Distribution per Curriculum Stage")
    savefig(os.path.join(outdir, "44_ttc_per_stage_violin.png"))


def plot_collision_proximity_histogram(episodes, outdir, bins=30):
    """Histogram of min obstacle depth for collision vs goal episodes."""
    reasons = _reason_array(episodes)
    depth = _metric_array(episodes, "ep_min_obstacle_depth")

    col_mask = (reasons == "collision") & np.isfinite(depth)
    goal_mask = (reasons == "goal") & np.isfinite(depth)

    if not col_mask.any():
        return

    col_vals = depth[col_mask]
    goal_vals = depth[goal_mask] if goal_mask.any() else np.array([])

    all_vals = np.concatenate([col_vals, goal_vals]) if len(goal_vals) else col_vals
    bin_edges = np.linspace(float(np.nanmin(all_vals)), float(np.nanmax(all_vals)), bins + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(col_vals, bins=bin_edges, color=REASON_COLORS["collision"],
            alpha=0.72, label=f"Collision (n={len(col_vals)})",
            edgecolor=PAPER_WHITE, linewidth=0.5)
    if len(goal_vals):
        ax.hist(goal_vals, bins=bin_edges, color=REASON_COLORS["goal"],
                alpha=0.55, label=f"Goal (n={len(goal_vals)})",
                edgecolor=PAPER_WHITE, linewidth=0.5)

    col_med = float(np.median(col_vals))
    ax.axvline(col_med, color=REASON_COLORS["collision"], linewidth=2.0,
               linestyle="--", alpha=0.9, label=f"Collision median = {col_med:.3f}")
    if len(goal_vals):
        goal_med = float(np.median(goal_vals))
        ax.axvline(goal_med, color=REASON_COLORS["goal"], linewidth=2.0,
                   linestyle="--", alpha=0.9, label=f"Goal median = {goal_med:.3f}")

    ax.set_xlabel("Minimum obstacle depth per episode")
    ax.set_ylabel("Count")
    set_title(ax, "Collision Proximity: Min Obstacle Depth by Outcome")
    add_legend(ax, loc="best")
    savefig(os.path.join(outdir, "45_collision_proximity_histogram.png"))


def plot_action_distribution_per_stage(episodes, outdir):
    """Grid of 2D histograms of (v_cmd, w_cmd) per curriculum stage."""
    ordered, short_labels, long_labels, stage_map = get_stage_groups(episodes)
    if not ordered:
        return

    valid_stages = []
    for s, label in zip(ordered, short_labels):
        eps = stage_map[s]
        v = np.array([safe_float(e.get("v_cmd")) for e in eps], dtype=float)
        w = np.array([safe_float(e.get("w_cmd")) for e in eps], dtype=float)
        mask = np.isfinite(v) & np.isfinite(w)
        if mask.sum() >= 5:
            valid_stages.append((label, v[mask], w[mask]))

    if not valid_stages:
        return

    n = len(valid_stages)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)

    all_v = np.concatenate([vs[1] for vs in valid_stages])
    all_w = np.concatenate([vs[2] for vs in valid_stages])
    v_lim = (float(np.nanpercentile(all_v, 1)), float(np.nanpercentile(all_v, 99)))
    w_lim = (float(np.nanpercentile(all_w, 1)), float(np.nanpercentile(all_w, 99)))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.2),
                             squeeze=False)

    for idx, (label, v, w) in enumerate(valid_stages):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        h = ax.hist2d(v, w, bins=32, range=[v_lim, w_lim], cmap="Greys", density=True)
        fig.colorbar(h[3], ax=ax, label="Density")
        ax.axhline(0, color=EDGE_CLR, linewidth=0.8, alpha=0.6)
        ax.axvline(0, color=EDGE_CLR, linewidth=0.8, alpha=0.6)
        ax.set_xlabel("v_cmd")
        ax.set_ylabel("w_cmd")
        set_title(ax, label)

    for idx in range(len(valid_stages), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Action Space Coverage per Curriculum Stage",
                 fontsize=13, fontweight="bold", color=TEXT_CLR, y=1.01)
    savefig(os.path.join(outdir, "46_action_distribution_per_stage.png"))


def plot_failure_free_streak(episodes, stage_transitions, outdir, w):
    """Current and rolling-max consecutive successful episodes over training."""
    ep = _episode_array(episodes)
    reasons = _reason_array(episodes)

    if len(ep) == 0:
        return

    # Build running streak (resets to 0 on any non-goal episode)
    streak = np.zeros(len(reasons), dtype=float)
    cur = 0
    for i, r in enumerate(reasons):
        cur = cur + 1 if r == "goal" else 0
        streak[i] = cur

    # Rolling maximum of streak
    roll_max = np.array([
        float(np.max(streak[max(0, i - w + 1): i + 1]))
        for i in range(len(streak))
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(ep, 0, streak, color=REASON_COLORS.get("goal", PRIMARY),
                    alpha=0.12, label="Streak (raw)")
    ax.plot(ep, streak, color=REASON_COLORS.get("goal", PRIMARY),
            linewidth=0.7, alpha=0.35)
    line_with_shadow(ax, ep, roll_max, PRIMARY, lw=2.5,
                     label=f"Rolling max streak (w={w})")

    best_idx = int(np.argmax(streak))
    best_val = streak[best_idx]
    if best_val > 0:
        ax.annotate(f"Best: {int(best_val)}",
                    xy=(ep[best_idx], best_val),
                    xytext=(ep[best_idx], best_val * 1.1),
                    ha="center", fontsize=9, fontweight="bold", color=TEXT_CLR,
                    arrowprops=dict(arrowstyle="-", color=EDGE_CLR, lw=0.8))

    add_stage_borders(ax, stage_transitions)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Consecutive successes")
    set_title(ax, "Failure-Free Streak Length")
    add_legend(ax, loc="upper left")
    savefig(os.path.join(outdir, "47_failure_free_streak.png"))


def plot_stagewise_bootstrap_ci(episodes, outdir, n_resamples=1000, ci=0.95):
    """Per-stage mean success rate with bootstrapped confidence intervals."""
    ordered, short_labels, long_labels, stage_map = get_stage_groups(episodes)
    if not ordered:
        return

    rng = np.random.default_rng(42)
    means, lows, highs = [], [], []

    for s in ordered:
        success = np.array(
            [1.0 if str(e.get("reason", "")) == "goal" else 0.0
             for e in stage_map[s]], dtype=float)
        if len(success) == 0:
            means.append(np.nan); lows.append(np.nan); highs.append(np.nan)
            continue
        obs = float(success.mean())
        boot = np.array([
            rng.choice(success, size=len(success), replace=True).mean()
            for _ in range(n_resamples)
        ])
        alpha = (1.0 - ci) / 2.0
        means.append(obs)
        lows.append(float(np.percentile(boot, alpha * 100)))
        highs.append(float(np.percentile(boot, (1 - alpha) * 100)))

    means = np.array(means, dtype=float)
    lows  = np.array(lows,  dtype=float)
    highs = np.array(highs, dtype=float)
    x = np.arange(len(ordered))
    valid = np.isfinite(means)

    fig, ax = plt.subplots(figsize=(max(6, len(ordered) * 1.6), 6))
    ax.bar(x[valid], means[valid],
           color=REASON_COLORS.get("goal", PRIMARY), alpha=0.75,
           edgecolor=PAPER_WHITE, linewidth=1.2, label="Mean success rate")
    ax.errorbar(x[valid], means[valid],
                yerr=[means[valid] - lows[valid], highs[valid] - means[valid]],
                fmt="none", color=TEXT_CLR, capsize=5, capthick=1.5,
                linewidth=1.5, label=f"{int(ci * 100)}% bootstrap CI")

    for i in x[valid]:
        ax.text(i, highs[i] + 0.025, f"{means[i]:.2f}",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=TEXT_CLR)

    ax.set_ylim(0, min(1.2, float(np.nanmax(highs)) + 0.12))
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_xlabel("Curriculum stage")
    ax.set_ylabel("Success rate")
    set_title(ax,
              f"Per-Stage Success Rate  —  {int(ci * 100)}% Bootstrap CI  (n={n_resamples})")
    add_legend(ax)
    savefig(os.path.join(outdir, "48_stagewise_bootstrap_ci.png"))


def plot_critic_loss_stability(episodes, stage_transitions, outdir, w):
    """Rolling mean and rolling std of critic loss — convergence quality diagnostic."""
    ep = _episode_array(episodes)
    critic = _metric_array(episodes, "critic_loss")

    if not np.isfinite(critic).any():
        return

    roll_mean = rolling_mean(critic, w)

    roll_std = np.array([
        float(np.nanstd(critic[max(0, i - w + 1): i + 1]))
        if np.isfinite(critic[max(0, i - w + 1): i + 1]).sum() > 1
        else np.nan
        for i in range(len(critic))
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()

    line_with_shadow(ax, ep, roll_mean, LOSS_COLORS.get("critic", SECONDARY),
                     lw=2.5, label=f"Rolling mean (w={w})")
    ax2.plot(ep, roll_std, color=ACCENT, linewidth=2.0, alpha=0.80,
             label=f"Rolling std (w={w})")

    # Annotate stabilization: first episode where std drops into the lowest 15th percentile
    valid_std = roll_std[np.isfinite(roll_std)]
    if len(valid_std) > 0:
        stab_thr = float(np.nanpercentile(valid_std, 15))
        candidates = np.where(np.isfinite(roll_std) & (roll_std <= stab_thr))[0]
        if len(candidates) > 0:
            stab_ep = ep[candidates[0]]
            ax2.axvline(stab_ep, color=ACCENT, linewidth=1.2,
                        linestyle=":", alpha=0.85)
            ax2.text(stab_ep, float(np.nanmax(valid_std)) * 0.92,
                     f"  stabilises\n  ep {int(stab_ep)}",
                     va="top", fontsize=8, color=ACCENT)

    add_stage_borders(ax, stage_transitions)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Critic loss — mean", color=TICK_CLR)
    ax2.set_ylabel("Critic loss — std", color=TICK_CLR)
    set_title(ax, "Critic Loss: Mean & Variance  (Convergence Quality)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
              facecolor=PAPER_WHITE, edgecolor=EDGE_CLR, labelcolor=TEXT_CLR)
    savefig(os.path.join(outdir, "49_critic_loss_stability.png"))


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Plot training results — Scientific Edition"
    )
    parser.add_argument("--json", required=True,
                        help="Path to episode_stats.json")
    parser.add_argument("--curr", default=None,
                        help="Optional path to curriculum json log file. If omitted, stage transitions are inferred from episode_stats.")
    parser.add_argument("--out", default="plots_scientific",
                        help="Output directory for visualizations.")
    parser.add_argument("--rolling", type=int, default=20,
                        help="Rolling window size.")
    parser.add_argument("--topk", type=int, default=12,
                        help="Number of representative episodes for detailed analysis.")
    parser.add_argument("--theme", default="thesis_nude_contrast",
                        choices=list(THEMES.keys()), help="Visualization theme")
    parser.add_argument("--interactive", action="store_true",
                        help="Also generate an optional Plotly HTML dashboard if Plotly is installed.")
    args = parser.parse_args()

    ensure_dir(args.out)
    _apply_scientific_rc(args.theme)

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("JSON must be a non-empty list of episode dicts.")
    
    # Stage transitions: use curriculum log if available, otherwise infer from episode_stats.
    if args.curr is not None and os.path.exists(args.curr):
        with open(args.curr, 'r', encoding="utf-8") as f:
            curriculum_log = json.load(f)

        stage_transitions = [
            entry.get('total_episodes')
            for entry in curriculum_log
            if entry.get('event') == 'stage_advanced'
        ]
        stage_transitions = [t for t in stage_transitions if t is not None]
    else:
        stage_transitions = infer_stage_transitions_from_episodes(data)

    data = sorted(data, key=lambda e: safe_float(e.get("episode"), np.inf))
    print(f"Loaded {len(data)} episodes.")
    print("Generating analysis figures (scientific theme)...")
    # After loading data, add this diagnostic section:
    print("\n=== Data Availability Check ===")
    fields_to_check = [
        'reward', 'length', 'time',
        'actor_loss', 'critic_loss', 'ent_coef', 'learning_rate',
        'curriculum_stage', 'min_dist', 'ttc',
        'v_cmd', 'w_cmd',
        'min_obstacle_depth', 'min_ultrasonic_distance',
        'shield_interventions_ep',
        'shield_interventions_caution_ep',
        'shield_interventions_danger_ep',
        'shield_interventions_critical_ep',
        'ep_min_obstacle_depth', 'ep_min_ultrasonic_distance',
        'ep_mean_jerk_v', 'ep_mean_jerk_w', 'ep_mean_ttc',
        'deadlock_escapes_ep'
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

    # Thesis-grade added plots
    plot_reward_quantile_band(data, stage_transitions, args.out, args.rolling)
    plot_stagewise_distribution_summary(data, args.out)
    plot_stagewise_outcome_composition(data, args.out)
    plot_safety_performance_tradeoff(data, args.out)
    plot_reward_density_hexbin(data, stage_transitions, args.out, args.rolling)
    plot_episode_storyboard(data, args.out)
    plot_metric_correlation_matrix(data, args.out)

    # Extended diagnostic plots — added after the previous suite, without removing anything.
    plot_learning_efficiency(data, stage_transitions, args.out, args.rolling)
    plot_rolling_failure_probability(data, stage_transitions, args.out, args.rolling)
    plot_policy_maturity_index(data, stage_transitions, args.out, args.rolling)
    plot_shield_dependency_decay(data, stage_transitions, args.out, args.rolling)
    plot_shield_severity_evolution(data, stage_transitions, args.out, args.rolling)
    plot_success_without_shield(data, args.out)
    plot_reward_safety_phase_portrait(data, args.out, args.rolling)
    plot_distance_outcome_boundary(data, args.out)
    plot_smoothness_vs_performance(data, args.out)
    plot_curriculum_difficulty_response(data, stage_transitions, args.out, args.rolling)
    plot_recovery_time_after_transition(data, stage_transitions, args.out, args.rolling)
    plot_stability_window(data, args.out, args.rolling)
    plot_best_worst_episode_contrast(data, args.out)
    plot_failure_signature_radar(data, args.out)
    plot_outcome_condition_heatmap(data, args.out)
    plot_sensor_agreement(data, args.out)
    plot_ttc_risk_map(data, args.out)
    plot_deadlock_diagnosis(data, args.out)

    # ── Thesis additions (verified fields only) ──────────────────────────────
    plot_stagewise_convergence_speed(data, stage_transitions, args.out, args.rolling)
    plot_reward_by_outcome(data, stage_transitions, args.out, args.rolling)
    plot_ttc_per_stage_violin(data, args.out)
    plot_collision_proximity_histogram(data, args.out)
    plot_action_distribution_per_stage(data, args.out)
    plot_failure_free_streak(data, stage_transitions, args.out, args.rolling)
    plot_stagewise_bootstrap_ci(data, args.out)
    plot_critic_loss_stability(data, stage_transitions, args.out, args.rolling)

    if args.interactive:
        create_interactive_plotly_dashboard(data, args.out)

    print(f"Analysis complete. Figures saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
