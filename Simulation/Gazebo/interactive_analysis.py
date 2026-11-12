import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────
# Helpers
# ─────────────────────────────
def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan


def rolling_mean(x, w):
    x = np.array(x, dtype=float)
    out = np.full_like(x, np.nan)

    for i in range(len(x)):
        j0 = max(0, i - w + 1)
        window = x[j0:i+1]
        window = window[np.isfinite(window)]
        if len(window) > 0:
            out[i] = np.mean(window)

    return out


# ─────────────────────────────
# Main
# ─────────────────────────────
def main(json_path, rolling=20):

    with open(json_path, "r") as f:
        data = json.load(f)

    data = sorted(data, key=lambda e: e["episode"])

    ep = [e["episode"] for e in data]

    # ===== Basic metrics =====
    reward = [safe_float(e.get("reward")) for e in data]

    jerk_v = [safe_float(e.get("ep_mean_jerk_v")) for e in data]
    jerk_w = [safe_float(e.get("ep_mean_jerk_w")) for e in data]

    deadlocks = [safe_float(e.get("deadlock_escapes_ep", 0)) for e in data]

    min_ultra = [safe_float(e.get("ep_min_ultrasonic_distance")) for e in data]
    min_depth = [safe_float(e.get("ep_min_obstacle_depth")) for e in data]

    # ===== Shield =====
    caution = np.array([safe_float(e.get("shield_interventions_caution_ep", 0)) for e in data])
    danger = np.array([safe_float(e.get("shield_interventions_danger_ep", 0)) for e in data])
    critical = np.array([safe_float(e.get("shield_interventions_critical_ep", 0)) for e in data])

    total = caution + danger + critical
    total[total == 0] = np.nan

    frac_caution = rolling_mean(caution / total, rolling)
    frac_danger = rolling_mean(danger / total, rolling)
    frac_critical = rolling_mean(critical / total, rolling)

    total_interventions = rolling_mean(total, rolling)

    # ─────────────────────────────
    # Create Subplots
    # ─────────────────────────────
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Reward",
            "Mean Jerk (v & w)",
            "Shield Type Fractions",
            "Total Shield Interventions",
            "Deadlocks & Min Distances"
        ]
    )

    # ===== 1. Reward =====
    fig.add_trace(go.Scatter(
        x=ep,
        y=rolling_mean(reward, rolling),
        mode='lines',
        name='Reward (rolling)',
    ), row=1, col=1)

    # ===== 2. Jerk =====
    fig.add_trace(go.Scatter(x=ep, y=rolling_mean(jerk_v, rolling),
                             name='Jerk v'), row=2, col=1)

    fig.add_trace(go.Scatter(x=ep, y=rolling_mean(jerk_w, rolling),
                             name='Jerk w'), row=2, col=1)

    # ===== 3. Shield fractions (STACKED AREA) =====
    fig.add_trace(go.Scatter(
        x=ep, y=frac_caution,
        stackgroup='one',
        name='Caution',
        line=dict(width=0),
        fillcolor='gold'
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=ep, y=frac_danger,
        stackgroup='one',
        name='Danger',
        line=dict(width=0),
        fillcolor='orange'
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=ep, y=frac_critical,
        stackgroup='one',
        name='Critical',
        line=dict(width=0),
        fillcolor='red'
    ), row=3, col=1)

    # ===== 4. Total interventions =====
    fig.add_trace(go.Scatter(
        x=ep,
        y=total_interventions,
        name='Total Interventions',
    ), row=4, col=1)

    # ===== 5. Deadlocks + distances =====
    fig.add_trace(go.Bar(
        x=ep,
        y=deadlocks,
        name='Deadlocks',
    ), row=5, col=1)

    fig.add_trace(go.Scatter(
        x=ep,
        y=rolling_mean(min_ultra, rolling),
        name='Min Ultrasonic',
    ), row=5, col=1)

    fig.add_trace(go.Scatter(
        x=ep,
        y=rolling_mean(min_depth, rolling),
        name='Min Depth',
    ), row=5, col=1)

    # ─────────────────────────────
    # Layout
    # ─────────────────────────────
    fig.update_layout(
        height=1400,
        title="RL Training Interactive Dashboard",
        hovermode='x unified'
    )

    fig.show()


if __name__ == "__main__":
    main("episode_stats.json", rolling=20)