import streamlit as st
import json
import numpy as np
import pandas as pd
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
# Load Data
# ─────────────────────────────
@st.cache_data
def load_data(file):
    data = json.load(file)
    data = sorted(data, key=lambda e: e["episode"])
    return data


# ─────────────────────────────
# UI
# ─────────────────────────────
st.set_page_config(layout="wide")
st.title("🚀 RL Training Dashboard (WandB-style)")

uploaded_file = st.file_uploader("Upload episode_stats.json", type="json")

if uploaded_file is None:
    st.stop()

data = load_data(uploaded_file)

episodes = [e["episode"] for e in data]

# ─────────────────────────────
# Sidebar Controls
# ─────────────────────────────
st.sidebar.header("Controls")

rolling = st.sidebar.slider("Smoothing window", 1, 100, 20)

ep_min, ep_max = st.sidebar.slider(
    "Episode Range",
    min(episodes),
    max(episodes),
    (min(episodes), max(episodes))
)

# Filter data
data = [e for e in data if ep_min <= e["episode"] <= ep_max]
ep = [e["episode"] for e in data]

# ─────────────────────────────
# Extract Metrics
# ─────────────────────────────
reward = [safe_float(e.get("reward")) for e in data]

jerk_v = [safe_float(e.get("ep_mean_jerk_v")) for e in data]
jerk_w = [safe_float(e.get("ep_mean_jerk_w")) for e in data]

deadlocks = [safe_float(e.get("deadlock_escapes_ep", 0)) for e in data]

min_ultra = [safe_float(e.get("ep_min_ultrasonic_distance")) for e in data]
min_depth = [safe_float(e.get("ep_min_obstacle_depth")) for e in data]

# Shield
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
# Layout: KPI Row
# ─────────────────────────────
col1, col2, col3, col4 = st.columns(4)

col1.metric("Episodes", len(ep))
col2.metric("Final Reward", f"{np.nanmean(reward[-50:]):.2f}")
col3.metric("Final Interventions", f"{np.nanmean(total_interventions[-50:]):.1f}")
col4.metric("Deadlocks (avg)", f"{np.nanmean(deadlocks[-50:]):.2f}")

# ─────────────────────────────
# Plot 1: Reward
# ─────────────────────────────
fig_reward = go.Figure()

fig_reward.add_trace(go.Scatter(
    x=ep,
    y=rolling_mean(reward, rolling),
    name="Reward",
))

fig_reward.update_layout(title="Reward")

st.plotly_chart(fig_reward, use_container_width=True)

# ─────────────────────────────
# Plot 2: Jerk
# ─────────────────────────────
fig_jerk = go.Figure()

fig_jerk.add_trace(go.Scatter(x=ep, y=rolling_mean(jerk_v, rolling), name="Jerk v"))
fig_jerk.add_trace(go.Scatter(x=ep, y=rolling_mean(jerk_w, rolling), name="Jerk w"))

fig_jerk.update_layout(title="Mean Jerk")

st.plotly_chart(fig_jerk, use_container_width=True)

# ─────────────────────────────
# Plot 3: Shield Fractions (STACKED)
# ─────────────────────────────
fig_shield = go.Figure()

fig_shield.add_trace(go.Scatter(
    x=ep, y=frac_caution,
    stackgroup='one',
    name='Caution',
    fillcolor='gold'
))

fig_shield.add_trace(go.Scatter(
    x=ep, y=frac_danger,
    stackgroup='one',
    name='Danger',
    fillcolor='orange'
))

fig_shield.add_trace(go.Scatter(
    x=ep, y=frac_critical,
    stackgroup='one',
    name='Critical',
    fillcolor='red'
))

fig_shield.update_layout(title="Shield Type Fractions", yaxis=dict(range=[0, 1]))

st.plotly_chart(fig_shield, use_container_width=True)

# ─────────────────────────────
# Plot 4: Total Interventions
# ─────────────────────────────
fig_total = go.Figure()

fig_total.add_trace(go.Scatter(
    x=ep,
    y=total_interventions,
    name="Total Interventions"
))

fig_total.update_layout(title="Total Shield Interventions")

st.plotly_chart(fig_total, use_container_width=True)

# ─────────────────────────────
# Plot 5: Deadlocks + Distances
# ─────────────────────────────
fig_misc = make_subplots(specs=[[{"secondary_y": True}]])

fig_misc.add_trace(go.Bar(
    x=ep,
    y=deadlocks,
    name="Deadlocks"
), secondary_y=False)

fig_misc.add_trace(go.Scatter(
    x=ep,
    y=rolling_mean(min_ultra, rolling),
    name="Min Ultrasonic"
), secondary_y=True)

fig_misc.add_trace(go.Scatter(
    x=ep,
    y=rolling_mean(min_depth, rolling),
    name="Min Depth"
), secondary_y=True)

fig_misc.update_layout(title="Deadlocks & Distances")

st.plotly_chart(fig_misc, use_container_width=True)