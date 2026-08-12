#!/usr/bin/env python3
"""
FrameStackWrapper — gym ObservationWrapper that stacks the last K `fixed`
observations into a single longer vector, giving the policy genuine temporal
context without an architectural rewrite.

## Why

The current LSTM in lstm_sac_policy.py is a *spatial* encoder over the
obstacle list (it reads obstacle 1, then 2, ..., then 10 and emits a summary).
It does NOT carry hidden state across environment steps.  The policy is
therefore reactive — it sees only the current observation and cannot remember
whether it has been stuck for 10 steps, whether a gap is opening or closing,
or what direction the goal was a moment ago.

For dynamic obstacles and moving goals, a single frame is not enough.

## What this wrapper does

Stacks the last K=3 `fixed` observation vectors (robot state, ultrasonics,
shield severity/duration, prev action) into a single concatenated vector of
length `fixed_dim * K`.  Obstacles are passed through unchanged because they
already carry velocity (vx, vy) features which give first-order temporal
information, and stacking would be ambiguous since obstacle identity is not
preserved across frames (the env sorts obstacles by distance every step, so
"closest obstacle at t-1" is not the same physical entity as "closest at t").

## Observation layout

Original:
    fixed:     [14]      single-step
    obstacles: [10, 5]   sorted by distance

After FrameStackWrapper(K=3):
    fixed:     [42]      = concat(fixed_{t-2}, fixed_{t-1}, fixed_t)
    obstacles: [10, 5]   unchanged
                         ^
                         oldest-first → newest-last ordering.
                         The most recent observation is at indices [-14:].

## Compatibility with LSTMObstacleExtractor

The extractor already reads `fixed_dim` dynamically from the observation
space (FIX Bug-4), so no policy changes are needed.  With K=3:
    features_dim = fixed_dim_stacked + lstm_hidden_size
                 = 42 + 64 = 106

## Reset behavior

On episode reset, the buffer is filled with copies of the first observation
rather than zeros.  Zeros would create a fake "everything was at origin"
history signal that the policy could spuriously learn.  Using copies makes
the early-episode history a constant rather than a misleading pattern.
"""

from typing import Any, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FrameStackWrapper(gym.ObservationWrapper):
    """
    Stack the last K `fixed` observations.  Obstacles pass through unchanged.

    Args:
        env:        a gym env whose observation_space is a spaces.Dict with
                    keys "fixed" (Box) and "obstacles" (Box).
        k:          number of frames to stack (default 3).
        fixed_key:  name of the fixed-features key (default "fixed").

    Raises:
        ValueError if the wrapped env's observation_space does not match.
    """

    def __init__(self, env: gym.Env, k: int = 3, fixed_key: str = "fixed"):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError(
                f"FrameStackWrapper requires Dict observation space, got "
                f"{type(env.observation_space)}"
            )
        if fixed_key not in env.observation_space.spaces:
            raise ValueError(
                f"FrameStackWrapper requires a '{fixed_key}' subspace; "
                f"available: {list(env.observation_space.spaces.keys())}"
            )
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.k = int(k)
        self.fixed_key = fixed_key

        old_fixed = env.observation_space.spaces[fixed_key]
        if not isinstance(old_fixed, spaces.Box):
            raise ValueError(
                f"'{fixed_key}' subspace must be Box, got {type(old_fixed)}"
            )
        if len(old_fixed.shape) != 1:
            raise ValueError(
                f"'{fixed_key}' subspace must be 1D, got shape {old_fixed.shape}"
            )

        self.fixed_dim_single = int(old_fixed.shape[0])
        self.fixed_dim_stacked = self.fixed_dim_single * self.k

        # Replicate low/high bounds K times so the new bounds remain valid.
        low_stacked  = np.tile(old_fixed.low,  self.k)
        high_stacked = np.tile(old_fixed.high, self.k)
        new_fixed_space = spaces.Box(
            low=low_stacked,
            high=high_stacked,
            shape=(self.fixed_dim_stacked,),
            dtype=old_fixed.dtype,
        )

        # Build the new Dict observation space, preserving all other keys.
        new_spaces = {fixed_key: new_fixed_space}
        for key, sub in env.observation_space.spaces.items():
            if key != fixed_key:
                new_spaces[key] = sub
        self.observation_space = spaces.Dict(new_spaces)

        # Rolling buffer holds the last k single-step fixed vectors.
        # Initialised in reset().
        self._buffer = None  # type: ignore

    # ------------------------------------------------------------------ #
    # gym ObservationWrapper API                                          #
    # ------------------------------------------------------------------ #

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        # Initialise buffer with K copies of the first observation.
        first_fixed = np.asarray(obs[self.fixed_key], dtype=np.float32).copy()
        self._buffer = [first_fixed.copy() for _ in range(self.k)]
        return self.observation(obs), info

    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Roll the buffer: drop oldest, append newest.
        new_fixed = np.asarray(observation[self.fixed_key], dtype=np.float32).copy()
        self._buffer.pop(0)
        self._buffer.append(new_fixed)
        # Concatenate oldest-first → newest-last so the most recent frame is at the tail.
        stacked = np.concatenate(self._buffer, axis=0).astype(np.float32, copy=False)

        # Build the new observation dict, preserving all other keys (e.g. obstacles).
        out = dict(observation)
        out[self.fixed_key] = stacked
        return out
