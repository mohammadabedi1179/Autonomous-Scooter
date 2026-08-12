#!/usr/bin/env python3
"""
export_onnx.py

Export a trained Curriculum LSTM-SAC checkpoint (Stable-Baselines3 .zip) to
ONNX so it can run client-side in a browser (onnxruntime-web) for the
scooter web-demo ("mini Gazebo").

Run this in the SAME environment you use for evaluation (the one with
stable_baselines3, torch, gymnasium, and lstm_sac_policy.py importable).
It does NOT need rclpy / ROS / Gazebo — pure PyTorch + SB3.

Usage:
    python3 export_onnx.py \
        --model ./models_lstm_curriculum_teacher/lstm_sac_curriculum_last.zip \
        --lstm_hidden 64 --lstm_layers 1 --net_arch 256,256 \
        --max_obstacles 10 --out policy.onnx

If you're exporting the distilled STUDENT model instead, pass the same
--lstm_hidden/--net_arch you trained it with, and add --student so the
DistilledSAC/custom replay buffer classes are registered before loading
(only needed for .load(), the exported graph is identical either way).

Produces:
    policy.onnx        - the exported inference graph
    policy_meta.json    - dims + normalization constants the JS side needs
"""

import argparse
import json
import os

import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import SAC
from lstm_sac_policy import create_lstm_sac_policy


class ONNXPolicyWrapper(nn.Module):
    """
    Exposes the actor as a pure, stateless-per-call function:

        (fixed, obstacles, h0, c0) -> (action, h1, c1)

    The browser owns the LSTM hidden/cell state between simulation steps
    (same role as the recurrent state carried across steps in your eval
    script's rollout loop) and feeds (h1, c1) back in as (h0, c0) next call.

    Reproduces SB3's deterministic SAC action exactly:
        features = concat(fixed, lstm_h_n[-1])
        latent   = latent_pi(features)
        action   = tanh(mu(latent))
    (No sampling, no clip_mean — clip_mean only applies to SDE, which this
    policy doesn't use.)
    """

    def __init__(self, actor):
        super().__init__()
        self.lstm = actor.features_extractor.lstm
        self.latent_pi = actor.latent_pi
        self.mu = actor.mu

    def forward(self, fixed, obstacles, h0, c0):
        # obstacles: [batch, max_obstacles, 5], h0/c0: [num_layers, batch, hidden]
        _, (h1, c1) = self.lstm(obstacles, (h0, c0))
        lstm_features = h1[-1]                       # [batch, hidden] (last layer)
        features = th.cat([fixed, lstm_features], dim=1)
        latent = self.latent_pi(features)
        action = th.tanh(self.mu(latent))
        return action, h1, c1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to SB3 .zip checkpoint")
    ap.add_argument("--lstm_hidden", type=int, default=64)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--net_arch", default="256,256")
    ap.add_argument("--max_obstacles", type=int, default=None,
                     help="Optional override; auto-detected from the checkpoint if omitted")
    ap.add_argument("--fixed_dim", type=int, default=None,
                     help="Optional override; auto-detected from the checkpoint if omitted")
    ap.add_argument("--student", action="store_true",
                     help="Set if this checkpoint is a DistilledSAC student model")
    ap.add_argument("--out", default="policy.onnx")
    args = ap.parse_args()

    net_arch = [int(x.strip()) for x in args.net_arch.split(",")]
    PolicyClass = create_lstm_sac_policy(
        lstm_hidden_size=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers,
        net_arch=net_arch,
    )

    print(f"Loading {args.model} ...")
    if args.student:
        try:
            from distilled_sac import DistilledSAC, DistillationDictReplayBuffer
        except ImportError:
            from sac_goal_follower.distilled_sac import DistilledSAC, DistillationDictReplayBuffer
        model = DistilledSAC.load(
            args.model, device="cpu",
            custom_objects={"policy": PolicyClass,
                             "replay_buffer_class": DistillationDictReplayBuffer},
        )
    else:
        model = SAC.load(args.model, device="cpu", custom_objects={"policy": PolicyClass})

    actor = model.policy.actor
    actor.eval()

    # --- Auto-detect real dims from the LOADED checkpoint itself, rather than
    # trusting docstrings/guesses. LSTMObstacleExtractor already stores these
    # as attributes, so we read them straight off the live module. ---
    fx = actor.features_extractor
    real_fixed_dim = int(fx.fixed_dim)
    real_max_obstacles = int(fx.num_obstacles)
    real_obstacle_feat_dim = int(fx.obstacle_feature_dim)
    real_lstm_hidden = int(fx.lstm_hidden_size)
    real_lstm_layers = int(fx.lstm_num_layers)

    print(f"Detected from checkpoint: fixed_dim={real_fixed_dim}, "
          f"max_obstacles={real_max_obstacles}, obstacle_feat_dim={real_obstacle_feat_dim}, "
          f"lstm_hidden={real_lstm_hidden}, lstm_layers={real_lstm_layers}")

    for name, cli_val, real_val in [
        ("fixed_dim", args.fixed_dim, real_fixed_dim),
        ("max_obstacles", args.max_obstacles, real_max_obstacles),
        ("lstm_hidden", args.lstm_hidden, real_lstm_hidden),
        ("lstm_layers", args.lstm_layers, real_lstm_layers),
    ]:
        if cli_val is not None and cli_val != real_val:
            print(f"WARNING: --{name}={cli_val} was passed but the checkpoint "
                  f"actually has {name}={real_val}. Using the checkpoint's real value.")

    wrapper = ONNXPolicyWrapper(actor)
    wrapper.eval()

    batch = 1
    dummy_fixed = th.zeros(batch, real_fixed_dim, dtype=th.float32)
    dummy_obstacles = th.zeros(batch, real_max_obstacles, real_obstacle_feat_dim, dtype=th.float32)
    dummy_h0 = th.zeros(real_lstm_layers, batch, real_lstm_hidden, dtype=th.float32)
    dummy_c0 = th.zeros(real_lstm_layers, batch, real_lstm_hidden, dtype=th.float32)

    with th.no_grad():
        ref_action, ref_h1, ref_c1 = wrapper(dummy_fixed, dummy_obstacles, dummy_h0, dummy_c0)
    print(f"Sanity forward pass OK. action shape={tuple(ref_action.shape)}")

    th.onnx.export(
        wrapper,
        (dummy_fixed, dummy_obstacles, dummy_h0, dummy_c0),
        args.out,
        input_names=["fixed", "obstacles", "h0", "c0"],
        output_names=["action", "h1", "c1"],
        dynamic_axes=None,   # fixed batch=1, fixed obstacle slot count -> simplest
        opset_version=17,    # LSTM op is well supported in onnxruntime-web at opset 17
    )
    print(f"Wrote {args.out}")

    # --- Verify the exported graph's math against SB3's own deterministic predict() ---
    obs = {
        "fixed": dummy_fixed.numpy()[0],
        "obstacles": dummy_obstacles.numpy()[0],
    }
    sb3_action, _ = model.predict(obs, deterministic=True)
    print(f"SB3 predict() action: {sb3_action}")
    print(f"Wrapper  action:      {ref_action.numpy()[0]}")
    print("(Both start from a zero LSTM state, so these should match closely.)")

    # --- Metadata the JS side needs: dims + every constant from goal_env_lstm.py ---
    # that affects EXECUTED behavior (kinematics, obs construction, shield gating).
    # Purely diagnostic constants (reward shaping, TTC stats, deadlock counters)
    # are intentionally left out -- they don't change what the robot does.
    meta = {
        "fixed_dim": real_fixed_dim,
        "max_obstacles": real_max_obstacles,
        "obstacle_feature_dim": real_obstacle_feat_dim,
        "lstm_hidden": real_lstm_hidden,
        "lstm_layers": real_lstm_layers,
        "net_arch": net_arch,

        "dt": 0.1,
        "v_max": 1.0,
        "w_max": 1.0,
        "smooth_alpha": 0.70,

        "GOAL_DIST_MAX": 15.0,
        "OBS_COORD_MAX": 10.0,
        "VEL_MAX": 2.0,
        "max_obstacle_depth": 10.0,
        "ultra_max_range": 1.0,

        "obstacle_collision_radius": 0.7,
        "ultra_collision_radius": 0.2,
        "success_radius": 1.5,

        "shield_enabled": True,
        "shield_mode": "passive",
        "shield_passive_levels": ["caution", "danger"],
        "shield_critical_dist": 0.25,
        "shield_danger_dist": 0.50,
        "shield_caution_dist": 0.80,
        "shield_obs_critical": 0.9,
        "shield_obs_danger": 1.5,
        "shield_obs_caution": 2.5,
        "shield_caution_max_v": 0.4,
        "shield_danger_max_v": 0.1,
        "shield_danger_steer_bias": 0.5,
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
