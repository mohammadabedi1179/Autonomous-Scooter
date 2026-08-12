#!/usr/bin/env python3
"""
Curriculum LSTM-SAC Training Script

Features:
- On-demand obstacle spawning via /obstacle_config topic
- Goal mover control via /goal_mover/config topic
- Resume-aware: restores curriculum stage from curriculum_log.json
- Single combined callback for reliability
- Trajectory saving every 100 episodes
"""

import os
import json
import re
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from stable_baselines3 import SAC
from sac_goal_follower.distilled_sac import DistilledSAC, DistillationDictReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from goal_env_lstm import GoalFollowerLSTMEnv
from lstm_sac_policy import create_lstm_sac_policy


# ============================================================================
# CURRICULUM STAGES v2 — decoupled task / observation axes (STUDENT ladder)
#
# Design: exactly ONE difficulty axis changes per stage transition.
#   Phase A (S0–S3): task skills acquired under GT-dominant observations.
#   Phase B (S4–S6): task FROZEN, observations ramp GT -> pure perception.
#   Phase C (S7–S9): observations FROZEN (pure perception), task scales up.
#
# Blend mechanics reminder (goal_env_lstm.start_gt_blend):
#   p_gt decays linearly 1 -> 0 over `gt_blend_episodes`, re-armed at every
#   stage change. Therefore:
#     * gt_blend_episodes >> expected stage length  => stage stays GT-DOMINANT
#       (p_gt never falls far before the next stage resets the schedule)
#     * gt_blend_episodes ~= expected stage length  => full RAMP inside stage
#     * gt_blend_episodes = 0                       => PURE PERCEPTION
#
# NOTE: obstacle load is capped at 4 movers — the sensor-realistic
# observability cap stated in the original Stage-6 comment. The old 5- and
# 7-mover stages exceeded what the sensors can observe and belong in an
# EVALUATION-ONLY stress suite, not in the training gate sequence.
# ============================================================================

CURRICULUM_STAGES = [

    # ── Phase A: task acquisition, GT-dominant ─────────────────────────────
    {
        # Pure goal-reaching. Perception goal marker (HSV) is reliable, and
        # there is no obstacle geometry to blend — no scaffold needed.
        "name": "Stage 0: goal reaching only (no obstacles)",
        "num_standing": 0,
        "num_walking": 0,
        "sector_half_angle_deg": 15.0,
        "moving_goal_enabled": False,
        "moving_goal_speed": 0.0,
        "time_limit": 80.0,
        "success_radius": 1.5,
        "advance_success_rate": 0.90,
        "advance_window": 50,
        "min_episodes": 100,
        "gt_blend_episodes": 0,
    },
    {
        # Static avoidance, learned on clean geometry. Merges old S1+S2 —
        # the teacher run showed static clutter is easy; no need for two
        # stages. Over-length blend keeps p_gt >= ~0.8 for a typical pass.
        "name": "Stage 1: 2 static, static goal (GT-dominant)",
        "num_standing": 2,
        "num_walking": 0,
        "sector_half_angle_deg": 20.0,
        "moving_goal_enabled": False,
        "moving_goal_speed": 0.0,
        "time_limit": 80.0,
        "success_radius": 1.5,
        "advance_success_rate": 0.88,
        "advance_window": 80,
        "min_episodes": 120,
        "gt_blend_episodes": 600,   # >> stage length => GT-dominant
    },
    {
        # First mover, still on clean geometry: isolate the anticipation /
        # velocity-reasoning skill from perception noise entirely.
        "name": "Stage 2: 1 static + 1 moving, static goal (GT-dominant)",
        "num_standing": 1,
        "num_walking": 1,
        "sector_half_angle_deg": 20.0,
        "moving_goal_enabled": False,
        "moving_goal_speed": 0.0,
        "time_limit": 80.0,
        "success_radius": 1.5,
        "advance_success_rate": 0.85,
        "advance_window": 100,
        "min_episodes": 150,
        "gt_blend_episodes": 800,   # >> stage length => GT-dominant
    },
    {
        # Multi-mover coordination on clean geometry. This is the task level
        # that Phase B will FREEZE while observations degrade.
        "name": "Stage 3: 2 static + 2 moving, static goal (GT-dominant)",
        "num_standing": 2,
        "num_walking": 2,
        "sector_half_angle_deg": 30.0,
        "moving_goal_enabled": False,
        "moving_goal_speed": 0.0,
        "time_limit": 85.0,
        "success_radius": 1.5,
        "advance_success_rate": 0.80,
        "advance_window": 100,
        "min_episodes": 150,
        "gt_blend_episodes": 800,   # >> stage length => GT-dominant
    },

    # ── Phase B: task frozen, observations ramp to pure perception ─────────
    # BC weight is HELD HIGH throughout this phase (BC_WEIGHT_BY_STAGE): the
    # teacher demonstrates correct behavior precisely where the student's
    # observations are least trustworthy.
    {
        # Statics-only perception ramp first: no track association, no
        # velocity estimation — the easiest perception regime to wean on.
        "name": "Stage 4: 2 static, perception RAMP (statics)",
        "num_standing": 2,
        "num_walking": 0,
        "sector_half_angle_deg": 30.0,
        "moving_goal_enabled": False,
        "moving_goal_speed": 0.0,
        "time_limit": 85.0,
        "success_radius": 1.5,
        "advance_success_rate": 0.82,
        "advance_window": 100,
        "min_episodes": 180,
        "gt_blend_episodes": 180,   # ~= stage length => full ramp in-stage
    },
    {
        # Dynamic perception ramp: Kalman tracks, association errors, late
        # detections now enter — at a task level mastered in Stage 3.
        "name": "Stage 5: 2 static + 2 moving, perception RAMP (movers)",
        "num_standing": 2,
        "num_walking": 2,
        "sector_half_angle_deg": 30.0,
        "moving_goal_enabled": False,
        "moving_goal_speed": 0.0,
        "time_limit": 85.0,
        "success_radius": 1.5,
        "advance_success_rate": 0.78,
        "advance_window": 120,
        "min_episodes": 250,
        "gt_blend_episodes": 250,   # ~= stage length => full ramp in-stage
    },
    {
        # Consolidation: SAME task, pure perception, no ramp. The replay
        # buffer fills with a stationary observation distribution here —
        # this is what lets the critic settle before task scaling begins.
        "name": "Stage 6: 2 static + 2 moving, PURE perception (consolidate)",
        "num_standing": 2,
        "num_walking": 2,
        "sector_half_angle_deg": 30.0,
        "moving_goal_enabled": False,
        "moving_goal_speed": 0.0,
        "time_limit": 85.0,
        "success_radius": 1.5,
        "advance_success_rate": 0.80,   # deliberately strict: master this
        "advance_window": 150,          # before ANY task scaling
        "min_episodes": 200,
        "gt_blend_episodes": 0,
    },

    # ── Phase C: pure perception frozen, task scales ────────────────────────
    # gt_blend_episodes = 0 for the rest of training: the observation
    # distribution never changes again.
    {
        "name": "Stage 7: 2 static + 3 moving, slow moving goal (pure perc.)",
        "num_standing": 2,
        "num_walking": 3,
        "sector_half_angle_deg": 30.0,
        "moving_goal_enabled": True,
        "moving_goal_speed": 0.2,
        "time_limit": 100.0,
        "success_radius": 2.0,
        "advance_success_rate": 0.75,
        "advance_window": 150,
        "min_episodes": 200,
        "gt_blend_episodes": 0,
    },
    {
        # Sensor-realistic crowd cap + fast goal.
        "name": "Stage 8: 2 static + 4 moving, fast moving goal (pure perc.)",
        "num_standing": 2,
        "num_walking": 4,
        "sector_half_angle_deg": 45.0,
        "moving_goal_enabled": True,
        "moving_goal_speed": 0.5,
        "time_limit": 100.0,
        "success_radius": 2.0,
        "advance_success_rate": 0.72,
        "advance_window": 150,
        "min_episodes": 250,
        "gt_blend_episodes": 0,
    },
    {
        # Final exam: widest goal geometry, tightest success radius, one
        # extra STATIC (statics are cheap to perceive; movers stay at the
        # sensor cap of 4). The old 5/7-mover configs move to eval-only.
        "name": "Stage 9: 3 static + 4 moving, fast moving goal (FINAL)",
        "num_standing": 3,
        "num_walking": 4,
        "sector_half_angle_deg": 60.0,
        "moving_goal_enabled": True,
        "moving_goal_speed": 0.5,
        "time_limit": 100.0,
        "success_radius": 1.5,
        "advance_success_rate": 0.80,   # terminal stage — aspirational gate
        "advance_window": 200,
        "min_episodes": 300,
        "gt_blend_episodes": 0,
    },
]

# ============================================================================
# Stage-keyed BC weight for DistilledSAC (student only). Replaces the
# timestep-based anneal: anchor hardest where observations are least
# trustworthy (Phase B ramps), release gradually as the student must
# outgrow the GT-trained teacher's blind spots under pure perception.
# Synced into the model by CurriculumEpisodeCallback._push_bc_weight().
# ============================================================================

BC_WEIGHT_BY_STAGE = {
    0: 0.5,    # trivial task, light anchor
    1: 1.0,
    2: 1.5,    # first mover: strong anchor
    3: 1.5,
    4: 2.0,    # perception ramps begin: maximum anchor
    5: 2.0,
    6: 1.0,    # pure perception consolidation: begin releasing
    7: 0.5,
    8: 0.3,
    9: 0.15,   # final: mostly self-directed, small residual anchor
}


# ============================================================================
# HELPERS
# ============================================================================

_MODEL_NAME_RE = re.compile(r".*?_ep_(\d+)_steps_(\d+)\.zip$")


def _parse_ep_steps_from_model_name(name: str):
    m = _MODEL_NAME_RE.match(os.path.basename(name))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _safe_load_episode_stats(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _safe_load_curriculum_log(path: str) -> List[Dict]:
    """Load curriculum log and return the list of events."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _restore_stage_from_log(curriculum_log: List[Dict]) -> Optional[int]:
    """
    Find the last curriculum stage from the log.
    Returns the stage index, or None if no stage info found.
    """
    last_stage = None
    for entry in curriculum_log:
        event = entry.get("event", "")
        if event == "stage_applied":
            last_stage = int(entry.get("stage_idx", 0))
        elif event == "stage_advanced":
            last_stage = int(entry.get("to_stage", 0))
    return last_stage


def _save_replay_buffer_if_possible(model: SAC, path: str, verbose: int = 1) -> None:
    try:
        model.save_replay_buffer(path)
        if verbose > 0:
            print(f"[ReplayBuffer] Saved to {path}")
    except AttributeError:
        if verbose > 0:
            print("[ReplayBuffer] save_replay_buffer not available; skipping.")
    except Exception as e:
        print(f"[ReplayBuffer] Error: {e}")


def _make_lr_schedule(initial_lr: float = 3e-4, min_fraction: float = 0.5):
    """
    Returns a learning-rate schedule callable for SB3.

    On a fresh run, progress_remaining starts at 1.0 and decays to 0.0.
    On resume, SB3 resets progress_remaining to 1.0 for the new learn() call,
    which would incorrectly jump the LR back up.  We therefore capture the
    *current* LR at resume time and pass it as initial_lr so the schedule
    continues from where it left off rather than restarting.
    """
    def schedule(progress_remaining: float) -> float:
        return initial_lr * max(min_fraction, progress_remaining)
    return schedule


def _get_current_lr(model: SAC) -> float:
    """Read the actual current LR from the actor optimizer."""
    try:
        return model.policy.actor.optimizer.param_groups[0]["lr"]
    except Exception:
        return 3e-4


# ============================================================================
# COMBINED CALLBACK
# ============================================================================

class CurriculumEpisodeCallback(BaseCallback):
    """
    Single callback handling curriculum management + episode stats + model saving.
    """

    def __init__(
        self,
        ros_node: Node,
        env: GoalFollowerLSTMEnv,
        stages: List[Dict],
        start_stage: int = 0,
        lr_reduction_on_advance: float = 0.5,
        save_path: str = "episode_stats.json",
        save_freq_episodes: int = 1,
        model_save_dir: str = "models_lstm",
        model_name_prefix: str = "lstm_sac_curriculum",
        best_model_path: str | None = None,
        best_metric_key: str = "reward",
        traj_save_every_episodes: int = 100,
        last_model_path: str | None = None,
        last_replay_buffer_path: str | None = None,
        curriculum_log_path: str = "curriculum_log.json",
        existing_stats: list | None = None,
        existing_curriculum_log: list | None = None,
        episode_start_offset: int = 0,
        is_resume: bool = False,
        teacher_mode: bool = True,
        teacher_advance_rate: float = 0.90,
        # ── PID-Lagrangian (shield cost constraint) ──
        shield_cost_budget: float = 6.0,   # target avg shield cost per episode (J_c)
        lambda_init: float = 0.0,          # FROM-SCRATCH: start with NO penalty
        # FROM-SCRATCH FIX: cap lambda LOW. The collapse was not lambda hitting
        # the cap (it settled ~12.9) but lambda*cost (~142) exceeding R_goal(100),
        # so loitering scored higher than reaching the goal. With lambda_max=2 and
        # a typical cost ~12, the worst-case penalty is ~24 << R_goal, so reaching
        # the goal is ALWAYS worth more than waiting. This is the key lever.
        lambda_max: float = 2.0,
        lambda_kp: float = 0.05,           # proportional gain
        lambda_ki: float = 0.01,           # integral gain (classic dual ascent)
        lambda_kd: float = 0.02,           # derivative gain (damps overshoot)
        lambda_ema: float = 0.1,           # smoothing on the measured cost
        # FROM-SCRATCH: learn the TASK before adding the safety penalty.
        #   - episodes <= warmup            : lambda forced to 0 (pure goal-reaching)
        #   - warmup < ep <= warmup+ramp    : lambda authority ramps linearly 0 -> full
        #   - ep > warmup+ramp              : full PID-Lagrangian control
        # Default warmup (~150) covers the no-obstacle Stage 0 plus the start of the
        # static stage, so the policy can find the goal before being penalized.
        lambda_warmup_episodes: int = 150,
        lambda_ramp_episodes: int = 150,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        # ---- PID-Lagrangian state ----
        self.shield_cost_budget = shield_cost_budget
        self.lambda_val = lambda_init
        self.lambda_max = lambda_max
        self.lambda_kp = lambda_kp
        self.lambda_ki = lambda_ki
        self.lambda_kd = lambda_kd
        self.lambda_ema = lambda_ema
        self.lambda_warmup_episodes = lambda_warmup_episodes
        self.lambda_ramp_episodes = lambda_ramp_episodes
        self._cost_ema = None          # smoothed J_c estimate
        self._lambda_integral = 0.0    # accumulated constraint error
        self._prev_cost_error = 0.0

        # ---- Curriculum ----
        self.ros_node = ros_node
        self.env = env
        self.stages = stages
        self.current_stage_idx = start_stage
        self.lr_reduction_on_advance = lr_reduction_on_advance
        self.teacher_mode = bool(teacher_mode)
        self.teacher_advance_rate = float(teacher_advance_rate)
        self.curriculum_log_path = curriculum_log_path

        self.obstacle_config_pub = self.ros_node.create_publisher(
            String, "/obstacle_config", 10
        )
        self.goal_mover_config_pub = self.ros_node.create_publisher(
            String, "/goal_mover/config", 10
        )

        self._advancing_stage = False  # guard against double-fire in multi-env loops

        # Restore existing curriculum log if resuming
        self._curriculum_log: List[Dict] = (
            existing_curriculum_log[:] if isinstance(existing_curriculum_log, list) else []
        )

        # On resume: reconstruct _stage_episode_count and _recent_outcomes from
        # the saved episode_stats so the advancement window is not reset to zero.
        # We replay only the episodes that belong to the current stage (i.e. those
        # recorded after the last stage_advanced event in the curriculum log).
        if is_resume and isinstance(existing_stats, list) and existing_stats:
            # Find the total_episodes value at the point the current stage started
            stage_start_total = 0
            for entry in self._curriculum_log:
                if entry.get("event") == "stage_advanced" and \
                        int(entry.get("to_stage", -1)) == start_stage:
                    stage_start_total = int(entry.get("total_episodes", 0))
                    break
                # Also handle the case where we never advanced (started at stage 0)
                # stage_start_total stays 0 in that case.

            # Filter episodes that belong to the current stage
            stage_eps = [
                e for e in existing_stats
                if int(e.get("episode", 0)) > stage_start_total
            ]
            self._stage_episode_count = len(stage_eps)
            self._recent_outcomes: deque = deque(maxlen=300)
            for e in stage_eps:
                reason = e.get("reason", "")
                self._recent_outcomes.append(reason)

            print(f"[RESUME] Restored stage state: "
                  f"stage_episode_count={self._stage_episode_count}, "
                  f"recent_outcomes={len(self._recent_outcomes)} "
                  f"(stage started at total_ep={stage_start_total})")
        else:
            self._stage_episode_count = 0
            self._recent_outcomes: deque = deque(maxlen=300)

        # ---- Stats ----
        self.save_path = save_path
        self.save_freq_episodes = int(save_freq_episodes)
        self.model_save_dir = model_save_dir
        self.model_name_prefix = model_name_prefix
        self.best_model_path = best_model_path
        self.best_metric_key = best_metric_key
        self.traj_save_every_episodes = int(traj_save_every_episodes)
        self.last_model_path = last_model_path
        self.last_replay_buffer_path = last_replay_buffer_path

        self.episode_stats = existing_stats[:] if isinstance(existing_stats, list) else []
        self._episode_counter = int(episode_start_offset)
        self.best_metric_value = -np.inf

        if self.episode_stats and self.best_metric_key:
            vals = [
                float(e.get(self.best_metric_key, float('-inf')))
                for e in self.episode_stats
                if isinstance(e.get(self.best_metric_key), (int, float))
                and np.isfinite(e.get(self.best_metric_key, float('nan')))
            ]
            if vals:
                self.best_metric_value = float(np.max(vals))

        self._step_buffers = None

        # Ensure directories
        for d in [self.model_save_dir,
                  os.path.dirname(self.save_path),
                  os.path.dirname(self.curriculum_log_path)]:
            if d:
                os.makedirs(d, exist_ok=True)
        if self.last_model_path:
            d = os.path.dirname(self.last_model_path)
            if d:
                os.makedirs(d, exist_ok=True)
        if self.best_model_path:
            d = os.path.dirname(self.best_model_path)
            if d:
                os.makedirs(d, exist_ok=True)

        print(f"[Callback] Initialized:\n"
              f"  stats_path={self.save_path}\n"
              f"  model_dir={self.model_save_dir}\n"
              f"  traj_save_every={self.traj_save_every_episodes}eps\n"
              f"  start_stage={self.current_stage_idx}\n"
              f"  episode_offset={self._episode_counter}\n"
              f"  existing_stats={len(self.episode_stats)}eps\n"
              f"  existing_curriculum_log={len(self._curriculum_log)} entries")

        # On resume: re-apply the stage to reconfigure env/ROS topics, but do NOT
        # append a new 'stage_applied' log entry — it was already logged originally.
        # On a fresh start: apply and log normally.
        self._apply_stage(self.current_stage_idx, log_event=not is_resume)

    # ========================
    # CURRICULUM
    # ========================

    @property
    def current_stage(self) -> Dict:
        return self.stages[self.current_stage_idx]

    def _publish_obstacle_config(self, num_walking: int, num_standing: int, sector_half_angle_deg: float):
        msg = String()
        msg.data = json.dumps({
            "num_walking": num_walking,
            "num_standing": num_standing,
            "sector_half_angle_deg": sector_half_angle_deg,
        })
        # Publish twice with a short delay: the obstacle node may not have
        # started yet (especially on resume), and ROS2 publisher latching is
        # not guaranteed.  A second publish after 1 s ensures the config lands
        # even if the subscriber was not yet ready on the first attempt.
        self.obstacle_config_pub.publish(msg)
        self.ros_node.get_logger().info(
            f"[CURRICULUM] Obstacle config (attempt 1/2): "
            f"{num_walking}w + {num_standing}s, sector={sector_half_angle_deg}°"
        )
        time.sleep(1.0)
        self.obstacle_config_pub.publish(msg)
        self.ros_node.get_logger().info(
            f"[CURRICULUM] Obstacle config (attempt 2/2): confirmed"
        )

    def _publish_goal_mover_config(self, enabled: bool, speed: float):
        msg = String()
        msg.data = json.dumps({"enabled": enabled, "speed": speed})
        self.goal_mover_config_pub.publish(msg)
        self.ros_node.get_logger().info(
            f"[CURRICULUM] Goal mover: enabled={enabled}, speed={speed}"
        )

    def _apply_stage(self, stage_idx: int, log_event: bool = True):
        stage = self.stages[stage_idx]
        if self.teacher_mode:
            self.env.use_ground_truth_geometry = True
            self.env._gt_static_mode = True
            self.env._gt_blend_episodes = 0
            self.env._gt_blend_counter = 0
        else:
            gt_blend = int(stage.get("gt_blend_episodes", 0))
            self.env.start_gt_blend(gt_blend, already_elapsed=self._stage_episode_count)

        print(f"\n{'='*70}")
        print(f"CURRICULUM: Applying {stage['name']}")
        print(f"  Obstacles: {stage['num_standing']}s + {stage['num_walking']}w")
        print(f"  Moving goal: {stage['moving_goal_enabled']} (speed={stage['moving_goal_speed']})")
        print(f"  Time limit: {stage['time_limit']}s, Success radius: {stage['success_radius']}m")
        print(f"  Advance: {stage['advance_success_rate']*100:.0f}% over "
              f"{stage['advance_window']} eps (min {stage['min_episodes']} eps)")
        print(f"{'='*70}\n")

        self.env.time_limit = stage['time_limit']
        self.env.success_radius = stage['success_radius']
        self.env.moving_goal = stage['moving_goal_enabled']

        self._publish_obstacle_config(
            num_walking=stage['num_walking'],
            num_standing=stage['num_standing'],
            sector_half_angle_deg=stage['sector_half_angle_deg'],
        )

        self._publish_goal_mover_config(
            enabled=stage['moving_goal_enabled'],
            speed=stage['moving_goal_speed'],
        )

        # ── Ground-truth -> perception blend for this stage ──────────────────
        # Each stage starts with clean GT geometry and weans to real detections
        # over `gt_blend_episodes`. Stages that introduce a NEW perception regime
        # (first static avoidance, first moving obstacles) get a longer blend;
        # continuations get a short refresh; Stage 0 (no obstacles) gets 0.
        # This is why "just the first stage" is NOT enough — see notes.
        # TEACHER: never blend to perception. STUDENT: honour per-stage weaning.
        if self.teacher_mode:
            print("  Geometry mode: TEACHER — pure ground-truth geometry")
        else:
            gt_blend = int(stage.get("gt_blend_episodes", 0))
            self.env.start_gt_blend(gt_blend, already_elapsed=self._stage_episode_count)
            _elapsed = self._stage_episode_count
            print(
                f"  GT->perception blend: {gt_blend} episodes "
                f"(elapsed={_elapsed}; "
                f"{'pure perception' if gt_blend == 0 or _elapsed >= gt_blend else 'clean start, weans to detections'})"
            )

        # ── Stage-keyed BC weight (student/DistilledSAC only) ────────────────
        # NOTE: the initial _apply_stage runs inside __init__, BEFORE SB3
        # attaches self.model — there this is a silent no-op, and the initial
        # weight is instead pushed from _on_training_start().
        self._push_bc_weight()

        if log_event:
            self._curriculum_log.append({
                "event": "stage_applied",
                "stage_idx": stage_idx,
                "stage_name": stage['name'],
                "total_episodes": self._episode_counter,
                "timestamp": time.time(),
            })
            self._save_curriculum_log()

    def _save_curriculum_log(self):
        try:
            with open(self.curriculum_log_path, "w") as f:
                json.dump(self._curriculum_log, f, indent=2)
        except Exception as e:
            print(f"[CURRICULUM] Error saving log: {e}")

    def _check_advancement(self) -> bool:
        stage = self.current_stage
        if self._stage_episode_count < stage['min_episodes']:
            return False
        window = stage['advance_window']
        if len(self._recent_outcomes) < window:
            return False
        recent = list(self._recent_outcomes)[-window:]
        success_rate = sum(1 for x in recent if x == 'goal') / len(recent)

        required = (max(stage['advance_success_rate'], self.teacher_advance_rate)
                    if self.teacher_mode else stage['advance_success_rate'])

        if self._stage_episode_count % 20 == 0:
            print(
                f"[CURRICULUM] Stage {self.current_stage_idx+1}/{len(self.stages)} | "
                f"Ep {self._stage_episode_count} (total {self._episode_counter}) | "
                f"Success (last {window}): {success_rate:.1%} "
                f"(need {required:.0%})"
            )
        return success_rate >= required

    def _advance_stage(self):
        if self.current_stage_idx >= len(self.stages) - 1:
            print("[CURRICULUM] Already at final stage.")
            return

        if self.model is not None:
            save_path = os.path.join(
                self.model_save_dir,
                f"lstm_sac_stage{self.current_stage_idx+1}_ep{self._episode_counter}.zip"
            )
            try:
                self.model.save(save_path)
                print(f"[CURRICULUM] Saved stage model: {save_path}")
                _save_replay_buffer_if_possible(
                    self.model, save_path.replace(".zip", "_replay.pkl"), verbose=1)
            except Exception as e:
                print(f"[CURRICULUM] Error saving stage model: {e}")

        if self.lr_reduction_on_advance < 1.0 and self.model is not None:
            # SAC has three separate optimizers: actor, critic, entropy coef.
            # We must reduce all of them; touching only policy.optimizer
            # (which is the actor) leaves the critic learning rate unchanged.
            _optimizers = []
            try:
                _optimizers.append(("actor",    self.model.policy.actor.optimizer))
            except AttributeError:
                pass
            try:
                _optimizers.append(("critic",   self.model.policy.critic.optimizer))
            except AttributeError:
                pass
            try:
                _optimizers.append(("ent_coef", self.model.policy.log_ent_coef_optimizer))
            except AttributeError:
                pass

            for name, opt in _optimizers:
                try:
                    for pg in opt.param_groups:
                        old = pg['lr']
                        pg['lr'] *= self.lr_reduction_on_advance
                        print(f"[CURRICULUM] LR [{name}]: {old:.2e} → {pg['lr']:.2e}")
                except Exception as e:
                    print(f"[CURRICULUM] LR reduction error [{name}]: {e}")

        self._curriculum_log.append({
            "event": "stage_advanced",
            "from_stage": self.current_stage_idx,
            "to_stage": self.current_stage_idx + 1,
            "episodes_in_stage": self._stage_episode_count,
            "total_episodes": self._episode_counter,
            "timestamp": time.time(),
        })

        self.current_stage_idx += 1
        self._stage_episode_count = 0
        self._recent_outcomes.clear()
        self._advancing_stage = False
        self._apply_stage(self.current_stage_idx)

    # ========================
    # STATS HELPERS
    # ========================

    @staticmethod
    def _to_jsonable_gt_obstacles(gt_dict):
        if gt_dict is None or not isinstance(gt_dict, dict):
            return None
        return {str(k): ([float(v[0]), float(v[1])] if v else None)
                for k, v in gt_dict.items()}

    def _clear_env_buffer(self, env_i: int):
        if self._step_buffers is None:
            return
        self._step_buffers[env_i] = {
            "robot_path": [], "goal_path": [], "gt_obstacles_path": [],
            "seen_obstacles_local": [], "seen_obstacles_world": [],
        }

    # ========================
    # SB3 HOOKS
    # ========================

    def _on_training_start(self) -> None:
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._step_buffers = [
            {"robot_path": [], "goal_path": [], "gt_obstacles_path": [],
             "seen_obstacles_local": [], "seen_obstacles_world": []}
            for _ in range(n_envs)
        ]
        print(f"[Callback] Training started. n_envs={n_envs}")

        # Sync the initial Lagrange multiplier into the env.
        self._push_lambda()

        # Sync the initial stage-keyed BC weight into the model. This MUST be
        # here (not only in _apply_stage): the initial _apply_stage runs in
        # __init__ before SB3 attaches self.model, so it can't do the push.
        # Also covers resumes, since current_stage_idx was restored from the
        # curriculum log.
        self._push_bc_weight()
        print(f"[Callback] Passive shield + PID-Lagrangian (FROM-SCRATCH config):\n"
              f"  shield levels: caution/danger/critical PASSIVE; no shield override\n"
              f"  lambda: init={self.lambda_val}, max={self.lambda_max} (capped LOW)\n"
              f"  warmup={self.lambda_warmup_episodes} ep (lambda=0), "
              f"ramp={self.lambda_ramp_episodes} ep\n"
              f"  budget(J_c)={self.shield_cost_budget}, "
              f"Kp={self.lambda_kp}, Ki={self.lambda_ki}, Kd={self.lambda_kd}")

    def _push_lambda(self) -> None:
        """Send the current lambda to the env (single-env; env_method also
        works for VecEnv)."""
        try:
            self.training_env.env_method("set_shield_lambda", self.lambda_val)
        except Exception:
            self.env.set_shield_lambda(self.lambda_val)

    def _push_bc_weight(self) -> None:
        """Sync the stage-keyed BC weight into the model (student only).

        Safe no-op for the teacher (plain SAC), for models without
        set_bc_weight, and when called before SB3 has attached self.model
        (i.e. from the initial _apply_stage inside __init__)."""
        if self.teacher_mode:
            return
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "set_bc_weight"):
            return
        weight = BC_WEIGHT_BY_STAGE.get(self.current_stage_idx)
        if weight is not None:
            model.set_bc_weight(weight)
            print(f"[BC] Stage-keyed BC weight: stage {self.current_stage_idx} "
                  f"-> {weight}")

    def _update_lambda(self, ep_shield_cost: float) -> None:
        """PID-Lagrangian update of the shield-cost multiplier (Stooke et al. 2020),
        with a FROM-SCRATCH warmup + ramp.

        Plain dual ascent (integral-only) on the constraint error is known to
        oscillate; the P and D terms damp that. The warmup keeps the penalty OFF
        until the policy has learned to reach the goal, then ramps lambda in so
        safety pressure is added gradually rather than fighting task learning.
        """
        # Always track the smoothed cost (for diagnostics + PID history).
        if self._cost_ema is None:
            self._cost_ema = ep_shield_cost
        else:
            self._cost_ema = ((1.0 - self.lambda_ema) * self._cost_ema
                              + self.lambda_ema * ep_shield_cost)

        ep = self._episode_counter

        # ── Phase 1: warmup — no penalty, learn the task ──
        if ep <= self.lambda_warmup_episodes:
            self.lambda_val = 0.0
            # keep error history fresh but DON'T accumulate the integral (anti-windup)
            self._prev_cost_error = self._cost_ema - self.shield_cost_budget
            self._push_lambda()
            return

        # ── PID target ──
        error = self._cost_ema - self.shield_cost_budget   # >0 => over budget
        self._lambda_integral = max(0.0, self._lambda_integral + error)  # anti-windup
        derivative = error - self._prev_cost_error
        self._prev_cost_error = error
        pid = (self.lambda_kp * error
               + self.lambda_ki * self._lambda_integral
               + self.lambda_kd * derivative)
        pid = min(self.lambda_max, max(0.0, pid))   # >=0 and capped LOW

        # ── Phase 2: linear ramp of lambda authority right after warmup ──
        ramp = 1.0
        if self.lambda_ramp_episodes > 0:
            prog = (ep - self.lambda_warmup_episodes) / float(self.lambda_ramp_episodes)
            ramp = min(1.0, max(0.0, prog))

        self.lambda_val = float(pid * ramp)
        self._push_lambda()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                print(f"[Callback] WARNING: dones/infos not in locals. "
                      f"Keys: {list(self.locals.keys())}")
            return True

        n_envs = len(dones)

        # Accumulate trajectory data
        if self._step_buffers is not None:
            for env_i in range(n_envs):
                info = infos[env_i]
                buf = self._step_buffers[env_i]
                rp = info.get("step_robot_pose")
                if rp is not None and len(rp) == 3:
                    buf["robot_path"].append([float(rp[0]), float(rp[1]), float(rp[2])])
                gp = info.get("step_goal_pose")
                if gp is not None and len(gp) == 2:
                    buf["goal_path"].append([float(gp[0]), float(gp[1])])
                gt_obs = self._to_jsonable_gt_obstacles(info.get("step_gt_obstacles_world"))
                if gt_obs is not None:
                    buf["gt_obstacles_path"].append(gt_obs)
                seen_local = info.get("step_seen_obstacles_local")
                if seen_local:
                    buf["seen_obstacles_local"].append(
                        [[float(a), float(b), float(c)] for a, b, c in seen_local])
                seen_world = info.get("step_seen_obstacles_world")
                if seen_world:
                    buf["seen_obstacles_world"].append(
                        [[float(a), float(b), float(c)] for a, b, c in seen_world])

        # Process episode ends
        try:
            log_vals = self.logger.name_to_value if self.logger else {}
        except Exception:
            log_vals = {}

        for env_i in range(n_envs):
            if not dones[env_i]:
                continue

            info = infos[env_i]
            self._episode_counter += 1

            reason = info.get("reason", "")
            self._stage_episode_count += 1
            self._recent_outcomes.append(reason)

            ep_data = {"episode": self._episode_counter}

            if "episode" in info:
                ep_info = info["episode"]
                ep_data["reward"] = float(ep_info.get("r", 0.0))
                ep_data["length"] = int(ep_info.get("l", 0))
                ep_data["time"] = float(ep_info.get("t", 0.0))

            for key in ["use_ground_truth_geometry", "shield_mode"]:
                if key in info:
                    ep_data[key] = info[key]

            if "obstacles_world_slots" in info:
                ep_data["valid_obstacle_slots"] = sum(
                    1 for s in info["obstacles_world_slots"] if s.get("valid", False)
                )

            if reason == "collision":
                obs_hit = float(info.get("min_obstacle_depth", 999.0)) < self.env.obstacle_collision_radius
                ultra_hit = float(info.get("min_ultrasonic_distance", 999.0)) < self.env.ultra_collision_radius

                if obs_hit and ultra_hit:
                    ep_data["collision_source"] = "both"
                elif obs_hit:
                    ep_data["collision_source"] = "obstacle_gt_or_depth"
                elif ultra_hit:
                    ep_data["collision_source"] = "ultrasonic"
                else:
                    ep_data["collision_source"] = "unknown"
            # ── PID-Lagrangian: update the shield-cost multiplier ──────────────
            # Constraint:  E[ ep_shield_cost ]  <=  shield_cost_budget
            ep_cost = float(info.get("ep_shield_cost", 0.0))
            self._update_lambda(ep_cost)
            ep_data["ep_shield_cost"] = ep_cost
            ep_data["ep_critical_steps"] = int(info.get("ep_critical_steps", 0))
            ep_data["shield_lambda"] = float(self.lambda_val)

            ep_data["actor_loss"] = float(log_vals.get("train/actor_loss", np.nan))
            ep_data["critic_loss"] = float(log_vals.get("train/critic_loss", np.nan))
            ep_data["ent_coef"] = float(log_vals.get("train/ent_coef", np.nan))
            ep_data["learning_rate"] = float(log_vals.get("train/learning_rate", np.nan))
            ep_data["curriculum_stage"] = self.current_stage_idx
            ep_data["curriculum_stage_name"] = self.current_stage['name']

            for key in ["reason", "robot_start", "goal_start", "robot_final",
                        "goal_final", "min_dist"]:
                if key in info:
                    ep_data[key] = info[key]
            for key in ["ttc", "v_cmd", "w_cmd", "min_obstacle_depth",
                        "min_ultrasonic_distance"]:
                if key in info:
                    ep_data[key] = float(info[key])
            for key in ["shield_level", "shield_interventions_ep",
                        "shield_interventions_danger_ep", "shield_interventions_caution_ep",
                        "shield_interventions_critical_ep"]:
                if key in info:
                    ep_data[key] = info[key]

            # Episode-level aggregate metrics
            for key in ["ep_min_obstacle_depth", "ep_min_ultrasonic_distance",
                        "ep_mean_jerk_v", "ep_mean_jerk_w", "ep_mean_ttc"]:
                if key in info:
                    ep_data[key] = float(info[key])

            # Deadlock escapes
            if "forced_escapes_ep" in info:
                ep_data["deadlock_escapes_ep"] = int(info["forced_escapes_ep"])

            # Trajectory every N episodes
            if (self.traj_save_every_episodes > 0 and
                    self._episode_counter % self.traj_save_every_episodes == 0):
                if self._step_buffers is not None and env_i < len(self._step_buffers):
                    buf = self._step_buffers[env_i]
                    for k in ["robot_path", "goal_path", "gt_obstacles_path",
                              "seen_obstacles_local", "seen_obstacles_world"]:
                        ep_data[k] = buf[k]

            self.episode_stats.append(ep_data)

            # Console
            ep_reward = ep_data.get("reward", 0.0)
            if self._episode_counter <= 5 or self._episode_counter % 10 == 0:
                print(f"[Ep {self._episode_counter}] reason={reason} R={ep_reward:.1f} "
                      f"stage={self.current_stage_idx+1} "
                      f"stage_ep={self._stage_episode_count}")

            # Save stats JSON
            if self._episode_counter % self.save_freq_episodes == 0:
                try:
                    with open(self.save_path, "w") as f:
                        json.dump(self.episode_stats, f, indent=2)
                except Exception as e:
                    print(f"[Callback] ERROR saving stats: {e}")
                    import traceback; traceback.print_exc()

            # Last model weights — cheap (a few MB), save every episode for
            # tight resume consistency.
            if self.last_model_path is not None:
                try:
                    self.model.save(self.last_model_path)
                except Exception as e:
                    print(f"[Callback] ERROR last model: {e}")
                    import traceback; traceback.print_exc()

            # Replay buffer — can be hundreds of MB; save every 10 episodes.
            # Worst-case resume gap is ~10 episodes of experience out of a
            # 1M-transition buffer, which is negligible for SAC.
            if (self.last_replay_buffer_path is not None and
                    self._episode_counter % 10 == 0):
                _save_replay_buffer_if_possible(
                    self.model, self.last_replay_buffer_path, verbose=0)

            # Best model
            if self.best_model_path and self.best_metric_key:
                val = ep_data.get(self.best_metric_key)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    if val > self.best_metric_value:
                        self.best_metric_value = val
                        try:
                            self.model.save(self.best_model_path)
                            print(f"[Callback] New best {self.best_metric_key}="
                                  f"{val:.3f} at ep {self._episode_counter}")
                        except Exception as e:
                            print(f"[Callback] ERROR best model: {e}")

            self._clear_env_buffer(env_i)

            # Curriculum check every 10 episodes
            if self._stage_episode_count % 10 == 0 and not self._advancing_stage:
                if self._check_advancement():
                    self._advancing_stage = True
                    self._advance_stage()

        return True


# ============================================================================
# TRAINER NODE
# ============================================================================

class LSTMSACCurriculumTrainer(Node):

    def __init__(self):
        super().__init__("lstm_sac_curriculum_trainer")

        self.declare_parameter("resume", False)
        self.declare_parameter("model_name", "")
        self.declare_parameter("lstm_hidden_size", 64)
        self.declare_parameter("lstm_num_layers", 1)
        self.declare_parameter("net_arch", "256,256")
        self.declare_parameter("start_stage", -1)  # -1 = auto-detect from log
        self.declare_parameter("max_obstacles", 10)
        # ── Teacher / Student mode ───────────────────────────────────────────
        #  teacher:=true   -> train the TEACHER on ground-truth geometry across
        #                     ALL stages (gt_blend forced to 0, high advance bar).
        #  teacher:=false  -> train the STUDENT on perception, warm-started from
        #                     a frozen teacher checkpoint and continuing SAC with
        #                     the replay buffer.
        self.declare_parameter("teacher", True)
        self.declare_parameter("teacher_model", "")
        self.declare_parameter("teacher_advance_rate", 0.90)
        # ── Student distillation / BC anchor ────────────────────────────────
        self.declare_parameter("bc_enabled", True)
        self.declare_parameter("bc_initial_weight", 2.0)
        self.declare_parameter("bc_final_weight", 0.05)
        self.declare_parameter("bc_anneal_steps", 500000)
        self.declare_parameter("bc_min_valid_fraction", 0.05)

        self.resume = bool(self.get_parameter("resume").value)
        self.model_name = str(self.get_parameter("model_name").value)
        self.lstm_hidden_size = int(self.get_parameter("lstm_hidden_size").value)
        self.lstm_num_layers = int(self.get_parameter("lstm_num_layers").value)
        self.start_stage = int(self.get_parameter("start_stage").value)
        self.max_obstacles = int(self.get_parameter("max_obstacles").value)
        self.teacher = bool(self.get_parameter("teacher").value)
        self.teacher_model = str(self.get_parameter("teacher_model").value)
        self.teacher_advance_rate = float(self.get_parameter("teacher_advance_rate").value)
        self.bc_enabled = bool(self.get_parameter("bc_enabled").value)
        self.bc_initial_weight = float(self.get_parameter("bc_initial_weight").value)
        self.bc_final_weight = float(self.get_parameter("bc_final_weight").value)
        self.bc_anneal_steps = int(self.get_parameter("bc_anneal_steps").value)
        self.bc_min_valid_fraction = float(self.get_parameter("bc_min_valid_fraction").value)
        self.role = "teacher" if self.teacher else "student"

        net_arch_str = str(self.get_parameter("net_arch").value)
        self.net_arch = [int(x.strip()) for x in net_arch_str.split(",")]

        self.env = GoalFollowerLSTMEnv(
            cmd_topic="/follower_robot/cmd_vel",
            goal_state_topic="/follower_robot/depth_cam/goal_marker_state",
            goal_odom_topic="/goal_marker/odom",
            dt=0.1,
            success_radius=1.5,
            time_limit=80.0,
            R_goal=100.0,
            R_collision=-100.0,
            gamma_shaping=0.97,   # MUST match SAC gamma below for potential-based shaping to be consistent
            step_penalty=0.2,
            ttc_threshold=2.0,
            cam_x=0.4,
            cam_y=0.0,
            use_ground_truth_geometry=self.teacher,   # GT in teacher mode, perception in student mode
            smooth_alpha=0.70,
            max_obstacles=self.max_obstacles,
        )

        # ── Passive shield + Lagrangian penalty ──────────────────────────────
        # Shield observes + classifies + penalizes only; it does NOT override the
        # policy's commands. The Lagrange multiplier (env.shield_lambda) is tuned
        # automatically by the PID-Lagrangian controller inside the callback.
        self.env.shield_mode = "passive"

        base_dir = os.path.dirname(__file__)
        # Mode-specific output dirs so a student run never overwrites the teacher.
        self.model_dir = os.path.join(base_dir, f"models_lstm_curriculum_{self.role}")
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_dir = os.path.join(base_dir, f"logs_lstm_curriculum_{self.role}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.teacher_model_dir = os.path.join(base_dir, "models_lstm_curriculum_teacher")

        self.get_logger().info(
            f"Curriculum LSTM-SAC Trainer:\n"
            f"  ROLE: {self.role.upper()}  (use_ground_truth_geometry={self.teacher})\n"
            f"  LSTM: hidden={self.lstm_hidden_size}, layers={self.lstm_num_layers}\n"
            f"  MLP: {self.net_arch}\n"
            f"  Max obstacles: {self.max_obstacles}\n"
            f"  Resume: {self.resume}\n"
            f"  BC distillation: enabled={self.bc_enabled and not self.teacher}, "
            f"w0={self.bc_initial_weight}, w_final={self.bc_final_weight}, "
            f"anneal_steps={self.bc_anneal_steps}\n"
            f"  model_dir: {self.model_dir}\n"
            f"  log_dir: {self.log_dir}"
        )

        self.sb3_logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])

    def train(self):
        env = Monitor(self.env)

        stats_path = os.path.join(self.log_dir, "episode_stats.json")
        curriculum_log_path = os.path.join(self.log_dir, "curriculum_log.json")
        best_model_path = os.path.join(self.model_dir, "lstm_sac_curriculum_best.zip")
        last_model_path = os.path.join(self.model_dir, "lstm_sac_curriculum_last.zip")
        last_replay_path = os.path.join(self.model_dir, "lstm_sac_curriculum_last_replay.pkl")

        existing_stats = []
        existing_curriculum_log = []
        episode_offset = 0
        resolved_start_stage = 0

        if self.resume:
            # Load existing stats
            existing_stats = _safe_load_episode_stats(stats_path)
            if existing_stats:
                try:
                    episode_offset = max(int(e.get("episode", 0)) for e in existing_stats)
                except Exception:
                    episode_offset = 0

            # Load existing curriculum log
            existing_curriculum_log = _safe_load_curriculum_log(curriculum_log_path)

            # Determine which stage to resume from
            if self.start_stage >= 0:
                # Explicit override
                resolved_start_stage = self.start_stage
                print(f"[RESUME] Using explicit start_stage={resolved_start_stage}")
            else:
                # Auto-detect from curriculum log
                detected = _restore_stage_from_log(existing_curriculum_log)
                if detected is not None:
                    resolved_start_stage = detected
                    print(f"[RESUME] Auto-detected stage {resolved_start_stage} "
                          f"from curriculum_log.json")
                else:
                    # Fallback: check last episode's curriculum_stage field
                    for ep in reversed(existing_stats):
                        if "curriculum_stage" in ep:
                            resolved_start_stage = int(ep["curriculum_stage"])
                            print(f"[RESUME] Detected stage {resolved_start_stage} "
                                  f"from episode_stats.json")
                            break

            # Clamp to valid range
            resolved_start_stage = max(0, min(
                resolved_start_stage, len(CURRICULUM_STAGES) - 1
            ))

            stage = CURRICULUM_STAGES[resolved_start_stage]
            print(f"[RESUME] Resuming at: {stage['name']}")
            print(f"[RESUME] Episode offset: {episode_offset}")
            print(f"[RESUME] Obstacles: {stage['num_walking']}w + {stage['num_standing']}s")
        else:
            if self.start_stage >= 0:
                resolved_start_stage = self.start_stage

        # Create policy
        PolicyClass = create_lstm_sac_policy(
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            net_arch=self.net_arch,
        )

        # Teacher training remains ordinary SAC. Student training uses a SAC
        # subclass that consumes replay-buffer teacher actions and adds an
        # annealed BC/distillation loss.
        AlgoClass = SAC if self.teacher else DistilledSAC

        if self.resume:
            if not self.model_name:
                # Try last model
                if os.path.exists(last_model_path):
                    model_path = last_model_path
                    print(f"[RESUME] No model_name given, using last model: {model_path}")
                else:
                    raise RuntimeError(
                        "resume:=true requires model_name or existing last model."
                    )
            else:
                model_path = self.model_name
                if not os.path.isabs(model_path):
                    model_path = os.path.join(self.model_dir, model_path)

            if self.teacher:
                model = SAC.load(model_path, env=env, device="auto")
            else:
                model = DistilledSAC.load(
                    model_path,
                    env=env,
                    device="auto",
                    custom_objects={"replay_buffer_class": DistillationDictReplayBuffer},
                )
            model.set_logger(self.sb3_logger)
            reset_num_timesteps = False

            # Fix: SB3 resets progress_remaining to 1.0 for the new learn() call,
            # which would jump the LR back to its initial value. We read the actual
            # current LR from the loaded optimizer and build a new schedule anchored
            # to that value so training continues smoothly from where it left off.
            resumed_lr = _get_current_lr(model)
            model.learning_rate = _make_lr_schedule(initial_lr=resumed_lr, min_fraction=0.5)
            print(f"[RESUME] LR schedule anchored to current LR: {resumed_lr:.2e}")

            for rp in [model_path.replace(".zip", "_replay.pkl"), last_replay_path]:
                if os.path.exists(rp):
                    try:
                        model.load_replay_buffer(rp)
                        self.get_logger().info(f"[RESUME] Loaded replay: {rp}")
                        break
                    except Exception as e:
                        self.get_logger().warn(f"[RESUME] Failed replay load: {e}")
        else:
            common_sac_kwargs = dict(
                policy=PolicyClass,
                env=env,
                learning_rate=_make_lr_schedule(initial_lr=3e-4, min_fraction=0.5),
                buffer_size=1_000_000,
                batch_size=256,
                tau=0.005,
                gamma=0.97,
                train_freq=1,
                gradient_steps=1,
                learning_starts=5000,
                ent_coef="auto",
                # FROM-SCRATCH: target_entropy raised above SB3's default (-dim(A)=-2)
                # to keep the policy stochastic for longer. Entropy collapsed to ~0.12
                # before the goal was ever found in the previous run, which locked the
                # policy into the loiter optimum. -1.0 keeps more exploration; relax
                # toward -2.0 once the policy is reliably reaching goals.
                target_entropy=-1.0,
                verbose=1,
                tensorboard_log=self.log_dir,
                device="auto",
            )

            if self.teacher:
                model = SAC(**common_sac_kwargs)
            else:
                model = DistilledSAC(
                    **common_sac_kwargs,
                    replay_buffer_class=DistillationDictReplayBuffer,
                    bc_initial_weight=self.bc_initial_weight,
                    bc_final_weight=self.bc_final_weight,
                    bc_anneal_steps=self.bc_anneal_steps,
                    bc_min_valid_fraction=self.bc_min_valid_fraction,
                )
            model.set_logger(self.sb3_logger)
            reset_num_timesteps = True

            # ── STUDENT warm-start from a frozen teacher ─────────────────────
            # Copy the teacher's actor + critic weights into the fresh student
            # policy. SAC then continues on PERCEPTION with the replay buffer,
            # and the env's per-stage GT->perception weaning eases the handoff.
            # NOTE: this is only the initial weight copy. The actual BC anchor
            # is attached below: the env queries a frozen teacher on GT obs and
            # DistilledSAC adds the annealed supervised actor loss.
            if (not self.teacher) and self.teacher_model:
                tpath = self.teacher_model
                if not os.path.isabs(tpath):
                    tpath = os.path.join(self.teacher_model_dir, tpath)
                if not os.path.exists(tpath):
                    raise RuntimeError(
                        f"[STUDENT] teacher_model not found: {tpath}. Train a teacher "
                        f"first (teacher:=true) and pass its .zip as teacher_model:=")
                try:
                    _teacher = SAC.load(tpath, device="auto")
                    model.policy.load_state_dict(_teacher.policy.state_dict())
                    del _teacher
                    print(f"[STUDENT] Warm-started actor+critic from teacher: {tpath}")
                except Exception as e:
                    raise RuntimeError(
                        "[STUDENT] Warm-start failed. Student and teacher must share "
                        "lstm_hidden_size / lstm_num_layers / net_arch / max_obstacles. "
                        f"Underlying error: {e}")
            elif not self.teacher:
                self.get_logger().warn(
                    "[STUDENT] No teacher_model given -> training student FROM SCRATCH "
                    "on perception. Pass teacher_model:=<path> to warm-start.")

        # Keep CLI BC parameters authoritative even when loading a saved student.
        if not self.teacher and isinstance(model, DistilledSAC):
            model.bc_initial_weight = self.bc_initial_weight
            model.bc_final_weight = self.bc_final_weight
            model.bc_anneal_steps = self.bc_anneal_steps
            model.bc_min_valid_fraction = self.bc_min_valid_fraction

        # ── Attach frozen teacher for BC/distillation targets ────────────────
        # Warm-start copies weights once. This attachment is different: it keeps
        # a frozen teacher alive so the env can query teacher.predict(GT_obs) at
        # every student step and put that action into the replay buffer.
        self._bc_teacher_model = None
        if not self.teacher:
            if self.teacher_model:
                tpath = self.teacher_model
                if not os.path.isabs(tpath):
                    tpath = os.path.join(self.teacher_model_dir, tpath)
                if not os.path.exists(tpath):
                    raise RuntimeError(f"[STUDENT/BC] teacher_model not found: {tpath}")
                try:
                    self._bc_teacher_model = SAC.load(tpath, device="auto")
                    self._bc_teacher_model.policy.set_training_mode(False)
                    for p in self._bc_teacher_model.policy.parameters():
                        p.requires_grad_(False)
                    self.env.set_bc_teacher(
                        self._bc_teacher_model,
                        enabled=self.bc_enabled,
                        deterministic=True,
                    )
                    print(
                        f"[STUDENT/BC] Frozen teacher attached for distillation: {tpath}\n"
                        f"             BC weight {self.bc_initial_weight} -> {self.bc_final_weight} "
                        f"over {self.bc_anneal_steps} gradient timesteps"
                    )
                except Exception as e:
                    raise RuntimeError(f"[STUDENT/BC] Failed to attach teacher for BC: {e}")
            else:
                self.env.set_bc_teacher(None, enabled=False)
                self.get_logger().warn(
                    "[STUDENT/BC] bc_enabled but no teacher_model was provided; "
                    "distillation loss will be inactive."
                )
        else:
            self.env.set_bc_teacher(None, enabled=False)

        # Single combined callback
        callback = CurriculumEpisodeCallback(
            ros_node=self,
            env=self.env,
            stages=CURRICULUM_STAGES,
            start_stage=resolved_start_stage,
            lr_reduction_on_advance=0.5,
            save_path=stats_path,
            save_freq_episodes=1,
            model_save_dir=self.model_dir,
            model_name_prefix="lstm_sac_curriculum",
            best_model_path=best_model_path,
            best_metric_key="reward",
            traj_save_every_episodes=100,
            last_model_path=last_model_path,
            last_replay_buffer_path=last_replay_path,
            curriculum_log_path=curriculum_log_path,
            existing_stats=existing_stats if self.resume else None,
            existing_curriculum_log=existing_curriculum_log if self.resume else None,
            episode_start_offset=episode_offset if self.resume else 0,
            is_resume=self.resume,
            teacher_mode=self.teacher,
            teacher_advance_rate=self.teacher_advance_rate,
            verbose=1,
        )

        obs_shape = env.observation_space.spaces
        self.get_logger().info(
            f"Starting training:\n"
            f"  Stats: {stats_path}\n"
            f"  Models: {self.model_dir}\n"
            f"  Fixed obs: {obs_shape['fixed'].shape}\n"
            f"  Obstacle obs: {obs_shape['obstacles'].shape}\n"
            f"  Start stage: {resolved_start_stage}"
        )

        model.learn(
            total_timesteps=3_000_000,
            callback=callback,
            log_interval=10,
            reset_num_timesteps=reset_num_timesteps,
        )

        final = os.path.join(self.model_dir, "lstm_sac_curriculum_final.zip")
        model.save(final)
        _save_replay_buffer_if_possible(model, final.replace(".zip", "_replay.pkl"))
        self.get_logger().info(f"Training complete. Final model: {final}")
        self.env.close()


def main(args=None):
    rclpy.init(args=args)
    trainer = LSTMSACCurriculumTrainer()
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.get_logger().info("Training interrupted.")
    finally:
        trainer.env.close()
        trainer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()