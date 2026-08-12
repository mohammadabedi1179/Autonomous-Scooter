#!/usr/bin/env python3
"""
Custom LSTM-SAC Policy for Stable-Baselines3

This module implements a custom SAC policy that uses LSTM to process
the obstacle sequence while keeping fixed features separate.

Architecture:
    Observation Dict:
        'fixed': [14D] - goal, robot state, ultrasonics, prev action,
                         shield_severity, shield_duration  (was 12D)
        'obstacles': [6×5] - obstacle sequence (sorted by distance)

    Network:
        obstacles → LSTM(hidden=64) → lstm_features[64]
        concat([fixed[14], lstm_features[64]]) → [78D]  (was 76D)
        → FC layers → actor/critic outputs

Changes vs original:
    - features_dim is now computed DYNAMICALLY as fixed_dim + lstm_hidden_size
      (FIX Bug-4: was hardcoded to 76, broke silently with non-default LSTM size)
    - fixed_dim is read from observation_space so it automatically adapts to
      the new 14D fixed observation (was 12D)
"""

from typing import Any, Dict, List, Optional, Tuple, Type

import torch as th
import torch.nn as nn
from torch.nn import functional as F

from gymnasium import spaces

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import SACPolicy, Actor, LOG_STD_MIN, LOG_STD_MAX


class LSTMObstacleExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses LSTM to process obstacle sequences.

    Input:
        observation (Dict):
            'fixed':     [batch, fixed_dim]          — fixed features (14D after fix)
            'obstacles': [batch, num_obs, obs_feat]  — obstacle sequence

    Output:
        features: [batch, fixed_dim + lstm_hidden_size]  — concatenated features

    FIX Bug-4: features_dim is now computed as fixed_dim + lstm_hidden_size
    instead of being hardcoded to 76.  The old hardcoded value silently broke
    the network whenever lstm_hidden_size != 64.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
    ):
        # FIX Bug-4: compute features_dim dynamically
        fixed_dim = int(observation_space.spaces['fixed'].shape[0])   # 14 after env fix
        computed_features_dim = fixed_dim + lstm_hidden_size          # e.g. 14+64=78
        super().__init__(observation_space, features_dim=computed_features_dim)

        # Extract spaces
        self.fixed_space     = observation_space.spaces['fixed']
        self.obstacles_space = observation_space.spaces['obstacles']

        # Fixed features dimension (auto-detected — works for both 12D and 14D)
        self.fixed_dim = fixed_dim

        # Obstacle features: [num_obstacles, feature_dim]
        self.num_obstacles       = int(self.obstacles_space.shape[0])  # e.g. 6 or 10
        self.obstacle_feature_dim = int(self.obstacles_space.shape[1])  # 5

        # LSTM for processing obstacle sequence
        self.lstm = nn.LSTM(
            input_size=self.obstacle_feature_dim,  # 5 (x, y, d, vx, vy)
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,                      # Input: [batch, seq, features]
        )

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers  = lstm_num_layers
        
    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Process observations through LSTM and concatenate with fixed features.

        Args:
            observations: Dict with
                'fixed'     [batch, fixed_dim]             e.g. [batch, 14]
                'obstacles' [batch, num_obstacles, 5]      e.g. [batch, 10, 5]

        Returns:
            features: [batch, fixed_dim + lstm_hidden_size]  e.g. [batch, 78]
        """
        fixed     = observations['fixed']        # [batch, fixed_dim]
        obstacles = observations['obstacles']    # [batch, num_obs, 5]

        # Process obstacle sequence through LSTM
        # lstm_out: [batch, seq_len, hidden_size]
        # h_n:      [num_layers, batch, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(obstacles)

        # Use the last layer's final hidden state as the obstacle representation
        lstm_features = h_n[-1, :, :]  # [batch, hidden_size]

        # Concatenate fixed features with LSTM output
        combined = th.cat([fixed, lstm_features], dim=1)  # [batch, fixed_dim+hidden_size]
        return combined


class LSTMSACActor(Actor):
    """
    Custom Actor network that uses LSTM feature extractor.
    
    This is nearly identical to the standard SAC Actor, but uses our
    custom LSTMObstacleExtractor instead of the default feature extractor.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images,
        )


class LSTMSACPolicy(SACPolicy):
    """
    Custom SAC Policy with LSTM obstacle processing.
    
    This policy uses:
    - LSTMObstacleExtractor for feature extraction
    - Standard SAC actor-critic architecture on top of extracted features
    
    The LSTM processes the obstacle sequence, then the output is combined
    with fixed features and passed through standard MLP layers.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = LSTMObstacleExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        # Default network architecture if not provided
        if net_arch is None:
            net_arch = [256, 256]
        
        # Default LSTM kwargs
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {
                'lstm_hidden_size': 64,
                'lstm_num_layers': 1,
            }
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        """Create the actor network."""
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return LSTMSACActor(**actor_kwargs).to(self.device)

    def forward(self, obs: Dict[str, th.Tensor], deterministic: bool = False) -> th.Tensor:
        """
        Forward pass through the policy.
        
        Args:
            obs: Dictionary observation
            deterministic: Whether to sample or return mean action
        
        Returns:
            action: Sampled or mean action
        """
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: Dict[str, th.Tensor], deterministic: bool = False) -> th.Tensor:
        """
        Get the action from the policy for a given observation.
        
        Args:
            observation: Dict observation
            deterministic: Whether to use stochastic or deterministic actions
        
        Returns:
            action: Taken action
        """
        # Get action from actor
        return self.actor(observation, deterministic)


# ============================================================================
# Helper function to create the policy
# ============================================================================

def create_lstm_sac_policy(
    lstm_hidden_size: int = 64,
    lstm_num_layers: int = 1,
    net_arch: Optional[List[int]] = None,
) -> Type[LSTMSACPolicy]:
    """
    Factory function to create an LSTM-SAC policy class with specific parameters.
    
    This is useful for passing custom configurations to Stable-Baselines3's SAC.
    
    Args:
        lstm_hidden_size: Hidden size of LSTM (default: 64)
        lstm_num_layers: Number of LSTM layers (default: 1)
        net_arch: Architecture of actor/critic networks (default: [256, 256])
    
    Returns:
        Custom policy class configured with the specified parameters
    
    Example:
        >>> policy_class = create_lstm_sac_policy(lstm_hidden_size=128, net_arch=[512, 512])
        >>> model = SAC(policy=policy_class, env=env, ...)
    """
    if net_arch is None:
        net_arch = [256, 256]
    
    class ConfiguredLSTMSACPolicy(LSTMSACPolicy):
        def __init__(self, *args, **kwargs):
            # Override the default features_extractor_kwargs
            kwargs['features_extractor_kwargs'] = {
                'lstm_hidden_size': lstm_hidden_size,
                'lstm_num_layers': lstm_num_layers,
            }
            kwargs['net_arch'] = net_arch
            super().__init__(*args, **kwargs)
    
    return ConfiguredLSTMSACPolicy


if __name__ == "__main__":
    """
    Test the LSTM policy with dummy data.
    Uses the updated 14D fixed observation (was 12D).
    """
    import numpy as np
    from gymnasium import spaces

    print("="*70)
    print("Testing LSTM-SAC Policy (fixed_dim=14, lstm_hidden=64 → features=78)")
    print("="*70)

    # Create dummy observation space — 14D fixed (matches updated env)
    obs_space = spaces.Dict({
        'fixed':     spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32),
        'obstacles': spaces.Box(low=-1, high=1, shape=(10, 5), dtype=np.float32),
    })

    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    # --- Test 1: default hidden size ---
    print("\n1. LSTMObstacleExtractor  lstm_hidden=64")
    extractor = LSTMObstacleExtractor(obs_space, lstm_hidden_size=64)
    assert extractor.features_dim == 78, f"Expected 78, got {extractor.features_dim}"
    assert extractor.fixed_dim    == 14, f"Expected 14, got {extractor.fixed_dim}"
    print(f"   features_dim={extractor.features_dim}  fixed_dim={extractor.fixed_dim}  ✓")

    # --- Test 2: non-default hidden size (was silently broken before fix) ---
    print("\n2. LSTMObstacleExtractor  lstm_hidden=128  (would have been wrong before fix)")
    extractor2 = LSTMObstacleExtractor(obs_space, lstm_hidden_size=128)
    assert extractor2.features_dim == 142, f"Expected 142, got {extractor2.features_dim}"
    print(f"   features_dim={extractor2.features_dim}  ✓")

    # --- Test 3: forward pass ---
    print("\n3. Forward pass")
    batch_size = 4
    dummy_obs = {
        'fixed':     th.randn(batch_size, 14),
        'obstacles': th.randn(batch_size, 10, 5),
    }
    features = extractor(dummy_obs)
    assert features.shape == (batch_size, 78), f"Expected (4,78), got {features.shape}"
    print(f"   Output shape: {features.shape}  ✓")

    # --- Test 4: full policy ---
    print("\n4. Full LSTM-SAC policy (lstm_hidden=64)")
    PolicyClass = create_lstm_sac_policy(lstm_hidden_size=64, net_arch=[256, 256])

    policy = PolicyClass(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )
    print("   Policy created  ✓")

    with th.no_grad():
        actions = policy._predict(dummy_obs, deterministic=True)
    assert actions.shape == (batch_size, 2), f"Expected (4,2), got {actions.shape}"
    print(f"   Actions shape: {actions.shape}  ✓")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
