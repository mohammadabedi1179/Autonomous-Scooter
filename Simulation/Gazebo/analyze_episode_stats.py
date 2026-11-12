#!/usr/bin/env python3
"""
analyze_episode_stats.py
========================
Reads the `episode_stats.json` produced by the curriculum LSTM-SAC trainer and
prints a focused diagnostic report aimed at the exact failure modes we're chasing:

  1. Lagrangian lambda saturation  -> is the penalty dominating? (over-conservatism)
  2. Shield-cost budget calibration -> what should shield_cost_budget actually be?
  3. Over-conservatism signature    -> timeout vs goal vs collision, length, reward
  4. Shield behaviour               -> interventions per zone, severity, fraction
  5. Margins                        -> ep_min_obstacle_depth: widening or hugging?
  6. Curriculum progression         -> did it advance past S1, and when?
  7. Learning stability             -> actor/critic loss, entropy, early vs late

It is dependency-free (stdlib only) so it runs anywhere the trainer runs.

Usage:
    python3 analyze_episode_stats.py [path/to/episode_stats.json] \
        [--budget 3.0] [--lambda-max 50] [--window 20]

Outputs:
    - a human-readable report to stdout
    - stats_report.txt      (same report, saved)
    - stats_summary.json    (compact machine-readable summary -> paste this back)
"""

import argparse
import json
import math
import os
from collections import Counter, OrderedDict


# ----------------------------------------------------------------------------- helpers
def load_stats(path):
    """Load the stats file. Tolerates a bare list or a {'episodes': [...]} dict."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ("episodes", "stats", "data"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
        else:
            # dict keyed by episode index?
            data = list(data.values())
    if not isinstance(data, list):
        raise ValueError("Could not interpret stats file as a list of episodes.")
    return [e for e in data if isinstance(e, dict)]


def fnum(x):
    """Best-effort float; returns None if not a finite number."""
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def series(stats, key):
    """Ordered list of finite floats present under `key` (skips missing/NaN)."""
    out = []
    for e in stats:
        if key in e:
            v = fnum(e[key])
            if v is not None:
                out.append(v)
    return out


def present(stats, key):
    return any(key in e for e in stats)


def pct(values, p):
    """Pure-python percentile (linear interpolation). p in [0,100]."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def mean(values):
    return sum(values) / len(values) if values else None


def std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def rolling_success(reasons, w):
    """Rolling fraction of 'goal' over a window of width w."""
    flags = [1.0 if r == "goal" else 0.0 for r in reasons]
    out = []
    for i in range(len(flags)):
        lo = max(0, i - w + 1)
        chunk = flags[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def window_indices(n, frac=0.2):
    """Return (early_slice, late_slice) index ranges as (start,end)."""
    k = max(1, int(round(n * frac)))
    return (0, k), (n - k, n)


# ----------------------------------------------------------------------------- report
def build_report(stats, budget, lambda_max_hint, w):
    lines = []
    summary = OrderedDict()

    def p(line=""):
        lines.append(line)

    n = len(stats)
    summary["n_episodes"] = n
    p("=" * 78)
    p("EPISODE STATS DIAGNOSTIC REPORT")
    p("=" * 78)
    p(f"episodes loaded: {n}")
    if n == 0:
        return "\n".join(lines), summary

    # available keys (union)
    keys = set()
    for e in stats:
        keys.update(e.keys())
    summary["available_keys"] = sorted(k for k in keys if not isinstance(
        stats[0].get(k), (list, dict)))

    (e0, e1), (l0, l1) = window_indices(n, 0.2)
    p(f"early window = episodes [{e0}:{e1}]   late window = episodes [{l0}:{l1}]")
    reasons = [e.get("reason", "") for e in stats]

    # ---- 1. LAMBDA SATURATION ------------------------------------------------
    p("\n" + "-" * 78)
    p("1. LAGRANGIAN MULTIPLIER (lambda)  --  is the penalty dominating?")
    p("-" * 78)
    lam = series(stats, "shield_lambda")
    if lam:
        lam_max_obs = max(lam)
        lam_final = lam[-1]
        cap = lambda_max_hint if lambda_max_hint else lam_max_obs
        near_cap = [v for v in lam if v >= 0.95 * cap]
        frac_sat = len(near_cap) / len(lam)
        late_lam = lam[l0:l1] if len(lam) >= l1 else lam[-max(1, w):]
        p(f"  observed: min={fmt(min(lam))}  max={fmt(lam_max_obs)}  final={fmt(lam_final)}")
        p(f"  late-window mean lambda: {fmt(mean(late_lam))}")
        p(f"  fraction of episodes with lambda >= 0.95*cap({fmt(cap,1)}): {fmt(frac_sat,3)}")
        verdict = ("SATURATED -> penalty is dominating; raise budget / lower lambda_max"
                   if frac_sat > 0.5 or lam_final >= 0.95 * cap
                   else "not saturated")
        p(f"  >> VERDICT: {verdict}")
        summary["lambda"] = {
            "min": min(lam), "max": lam_max_obs, "final": lam_final,
            "late_mean": mean(late_lam), "frac_at_cap": frac_sat,
            "verdict": verdict,
        }
    else:
        p("  shield_lambda NOT in stats (older run, or logging not enabled).")
        summary["lambda"] = None

    # ---- 2. SHIELD-COST BUDGET CALIBRATION ----------------------------------
    p("\n" + "-" * 78)
    p("2. SHIELD COST (ep_shield_cost)  --  what should the budget be?")
    p("-" * 78)
    cost = series(stats, "ep_shield_cost")
    if cost:
        late_cost = cost[l0:l1]
        p(f"  overall: mean={fmt(mean(cost))}  median(p50)={fmt(pct(cost,50))}")
        p(f"           p10={fmt(pct(cost,10))}  p90={fmt(pct(cost,90))}  max={fmt(max(cost))}")
        p(f"  late window: mean={fmt(mean(late_cost))}  median={fmt(pct(late_cost,50))}")
        p(f"  current budget setting: {fmt(budget)}")
        med = pct(cost, 50) or 0.0
        suggest_lo, suggest_hi = 0.4 * med, 0.6 * med
        p(f"  >> SUGGESTED budget (40-60% of median natural cost): "
          f"{fmt(suggest_lo,2)} .. {fmt(suggest_hi,2)}")
        if budget and med and budget < 0.25 * med:
            p(f"  >> WARNING: budget {fmt(budget)} << median {fmt(med)} "
              f"-> lambda will ramp to max (this matches over-conservatism).")
        summary["shield_cost"] = {
            "mean": mean(cost), "median": pct(cost, 50),
            "p10": pct(cost, 10), "p90": pct(cost, 90), "max": max(cost),
            "late_mean": mean(late_cost),
            "current_budget": budget,
            "suggested_budget_range": [suggest_lo, suggest_hi],
        }
    else:
        p("  ep_shield_cost NOT in stats.")
        summary["shield_cost"] = None

    # ---- 3. OVER-CONSERVATISM SIGNATURE -------------------------------------
    p("\n" + "-" * 78)
    p("3. OUTCOME MIX & OVER-CONSERVATISM SIGNATURE")
    p("-" * 78)
    rc_all = Counter(reasons)
    rc_late = Counter(reasons[l0:l1])

    def frac(counter, key):
        tot = sum(counter.values())
        return counter.get(key, 0) / tot if tot else None

    for label, c in (("overall", rc_all), ("late", rc_late)):
        p(f"  [{label}] goal={fmt(frac(c,'goal'))}  timeout={fmt(frac(c,'timeout'))}  "
          f"collision={fmt(frac(c,'collision'))}")
    succ_roll = rolling_success(reasons, w)
    p(f"  rolling success (w={w}) final: {fmt(succ_roll[-1])}   peak: {fmt(max(succ_roll))}")

    length = series(stats, "length")
    reward = series(stats, "reward")
    timeout_lengths = [fnum(e.get("length")) for e in stats
                       if e.get("reason") == "timeout" and fnum(e.get("length"))]
    timeout_lengths = [v for v in timeout_lengths if v is not None]
    if length:
        p(f"  episode length: mean={fmt(mean(length),1)}  "
          f"late-mean={fmt(mean(length[l0:l1]),1)}")
    if timeout_lengths:
        p(f"  typical timeout length (median): {fmt(pct(timeout_lengths,50),1)} steps")
    if reward:
        p(f"  reward: mean={fmt(mean(reward),1)}  late-mean={fmt(mean(reward[l0:l1]),1)}  "
          f"max={fmt(max(reward),1)}")
    # signature flag
    f_to = frac(rc_late, "timeout")
    f_go = frac(rc_late, "goal")
    if f_to is not None and f_go is not None and f_to > 0.6 and f_go < 0.2:
        p("  >> SIGNATURE: high timeout + low goal + (see severity below) "
          "= OVER-CONSERVATISM / loitering.")
    summary["outcomes"] = {
        "overall": {k: frac(rc_all, k) for k in ("goal", "timeout", "collision")},
        "late": {k: frac(rc_late, k) for k in ("goal", "timeout", "collision")},
        "rolling_success_final": succ_roll[-1],
        "rolling_success_peak": max(succ_roll),
        "reward_late_mean": mean(reward[l0:l1]) if reward else None,
        "reward_max": max(reward) if reward else None,
        "length_late_mean": mean(length[l0:l1]) if length else None,
        "timeout_length_median": pct(timeout_lengths, 50) if timeout_lengths else None,
    }

    # ---- 4. SHIELD BEHAVIOUR -------------------------------------------------
    p("\n" + "-" * 78)
    p("4. SHIELD INTERVENTIONS & SEVERITY")
    p("-" * 78)
    interv = series(stats, "shield_interventions_ep")
    crit = series(stats, "shield_interventions_critical_ep") or series(stats, "ep_critical_steps")
    dang = series(stats, "shield_interventions_danger_ep")
    caut = series(stats, "shield_interventions_caution_ep")
    if interv:
        p(f"  interventions/episode: early-mean={fmt(mean(interv[e0:e1]),1)}  "
          f"late-mean={fmt(mean(interv[l0:l1]),1)}")
    if crit:
        p(f"  CRITICAL steps/episode: early-mean={fmt(mean(crit[e0:e1]),2)}  "
          f"late-mean={fmt(mean(crit[l0:l1]),2)}  max={fmt(max(crit),0)}")
    if dang:
        p(f"  danger/episode:  late-mean={fmt(mean(dang[l0:l1]),2)}")
    if caut:
        p(f"  caution/episode: late-mean={fmt(mean(caut[l0:l1]),2)}")
    # intervention fraction
    if interv and length:
        m = min(len(interv), len(length))
        fracs = [interv[i] / length[i] for i in range(m) if length[i] > 0]
        if fracs:
            p(f"  intervention fraction (interv/length): mean={fmt(mean(fracs),3)}  "
              f"late-mean={fmt(mean(fracs[l0:l1] if len(fracs)>=l1 else fracs),3)}")
    summary["shield"] = {
        "interv_early_mean": mean(interv[e0:e1]) if interv else None,
        "interv_late_mean": mean(interv[l0:l1]) if interv else None,
        "critical_late_mean": mean(crit[l0:l1]) if crit else None,
        "critical_max": max(crit) if crit else None,
        "danger_late_mean": mean(dang[l0:l1]) if dang else None,
        "caution_late_mean": mean(caut[l0:l1]) if caut else None,
    }

    # ---- 5. MARGINS ----------------------------------------------------------
    p("\n" + "-" * 78)
    p("5. CLEARANCE MARGINS (ep_min_obstacle_depth)  --  widening or hugging?")
    p("-" * 78)
    depth = series(stats, "ep_min_obstacle_depth")
    # drop the inf placeholder if it slipped through
    depth = [d for d in depth if d < 1e6]
    if depth:
        p(f"  early: mean={fmt(mean(depth[e0:e1]),3)}  p10={fmt(pct(depth[e0:e1],10),3)}")
        p(f"  late : mean={fmt(mean(depth[l0:l1]),3)}  p10={fmt(pct(depth[l0:l1],10),3)}")
        shift = mean(depth[l0:l1]) - mean(depth[e0:e1])
        p(f"  >> shift (late-early): {fmt(shift,3)}  "
          f"({'widening (good)' if shift > 0.05 else 'flat/hugging' if abs(shift)<=0.05 else 'narrowing (bad)'})")
        summary["margin"] = {
            "early_mean": mean(depth[e0:e1]), "late_mean": mean(depth[l0:l1]),
            "late_p10": pct(depth[l0:l1], 10), "shift": shift,
        }
    else:
        p("  ep_min_obstacle_depth NOT in stats.")
        summary["margin"] = None

    # ---- 6. CURRICULUM -------------------------------------------------------
    p("\n" + "-" * 78)
    p("6. CURRICULUM PROGRESSION")
    p("-" * 78)
    stage_key = "curriculum_stage" if present(stats, "curriculum_stage") else None
    if stage_key:
        starts = OrderedDict()
        prev = object()
        for i, e in enumerate(stats):
            s = e.get(stage_key)
            if s != prev:
                starts[str(s)] = i
                prev = s
        cur = stats[-1].get(stage_key)
        cur_name = stats[-1].get("curriculum_stage_name", "")
        p(f"  stages seen (stage -> first episode index): "
          f"{', '.join(f'{k}@{v}' for k, v in starts.items())}")
        p(f"  current stage: {cur}  ({cur_name})")
        if len(starts) == 1:
            p("  >> NEVER ADVANCED past the first stage seen "
              "(consistent with the success collapse).")
        summary["curriculum"] = {"stage_starts": starts, "current_stage": cur,
                                 "current_stage_name": cur_name}
    else:
        p("  curriculum_stage NOT in stats.")
        summary["curriculum"] = None

    # ---- 7. LEARNING STABILITY ----------------------------------------------
    p("\n" + "-" * 78)
    p("7. LEARNING STABILITY (early vs late)")
    p("-" * 78)
    for key, label in (("actor_loss", "actor loss"),
                       ("critic_loss", "critic loss"),
                       ("ent_coef", "entropy coef"),
                       ("learning_rate", "learning rate")):
        s = series(stats, key)
        if s:
            p(f"  {label:13s}: early={fmt(mean(s[e0:e1]),4)}+-{fmt(std(s[e0:e1]),4)}  "
              f"late={fmt(mean(s[l0:l1]),4)}+-{fmt(std(s[l0:l1]),4)}")
            summary.setdefault("learning", {})[key] = {
                "early_mean": mean(s[e0:e1]), "late_mean": mean(s[l0:l1]),
                "late_std": std(s[l0:l1]),
            }

    # ---- HEADLINE ------------------------------------------------------------
    p("\n" + "=" * 78)
    p("PASTE THIS BACK TO CLAUDE")
    p("=" * 78)
    head = OrderedDict()
    head["n_episodes"] = n
    if summary.get("lambda"):
        head["lambda_final"] = round(summary["lambda"]["final"], 3)
        head["lambda_frac_at_cap"] = round(summary["lambda"]["frac_at_cap"], 3)
    if summary.get("shield_cost"):
        head["ep_shield_cost_median"] = round(summary["shield_cost"]["median"], 2)
        head["suggested_budget"] = [round(x, 2) for x in
                                    summary["shield_cost"]["suggested_budget_range"]]
    head["late_goal"] = fnum(fmt(summary["outcomes"]["late"]["goal"]))
    head["late_timeout"] = fnum(fmt(summary["outcomes"]["late"]["timeout"]))
    head["late_collision"] = fnum(fmt(summary["outcomes"]["late"]["collision"]))
    head["rolling_success_final"] = round(summary["outcomes"]["rolling_success_final"], 3)
    if summary.get("margin"):
        head["margin_shift_late_minus_early"] = round(summary["margin"]["shift"], 3)
    if summary.get("curriculum"):
        head["current_stage"] = summary["curriculum"]["current_stage"]
        head["stages_reached"] = len(summary["curriculum"]["stage_starts"])
    p(json.dumps(head, indent=2))

    return "\n".join(lines), summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="episode_stats.json")
    ap.add_argument("--budget", type=float, default=3.0,
                    help="shield_cost_budget currently set in training")
    ap.add_argument("--lambda-max", type=float, default=50.0,
                    help="lambda_max currently set in training")
    ap.add_argument("--window", type=int, default=20,
                    help="rolling window for success rate")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        raise SystemExit(f"File not found: {args.path}")

    stats = load_stats(args.path)
    report, summary = build_report(stats, args.budget, args.lambda_max, args.window)
    print(report)

    rpt_path = os.path.join(args.out_dir, "stats_report.txt")
    sum_path = os.path.join(args.out_dir, "stats_summary.json")
    with open(rpt_path, "w") as f:
        f.write(report + "\n")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[saved] {rpt_path}\n[saved] {sum_path}")


if __name__ == "__main__":
    main()
