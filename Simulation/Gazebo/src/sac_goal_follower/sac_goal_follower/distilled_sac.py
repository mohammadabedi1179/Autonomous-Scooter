#!/usr/bin/env python3
"""
Distilled SAC for perception-student training.

Adds a behavior-cloning/distillation anchor to SAC:
  - the environment places a frozen teacher action in info["bc_teacher_action"]
  - DistillationDictReplayBuffer stores that action beside the transition
  - DistilledSAC.train() adds an annealed MSE loss between the student's
    deterministic actor action and the teacher action

This file is intentionally small and independent from the env/policy files so
regular teacher SAC training remains unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional

import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.utils import polyak_update


class DistillationDictReplayBufferSamples(NamedTuple):
    observations: Dict[str, th.Tensor]
    actions: th.Tensor
    next_observations: Dict[str, th.Tensor]
    dones: th.Tensor
    rewards: th.Tensor
    bc_teacher_actions: th.Tensor
    bc_teacher_masks: th.Tensor


class DistillationDictReplayBuffer(DictReplayBuffer):
    """Dict replay buffer with one extra optional field: teacher action.

    The env is expected to put two fields in every info dict:
        info["bc_teacher_action"] -> action in the env/action-buffer scale
        info["bc_teacher_valid"]  -> bool mask
    Missing/invalid teacher actions are masked out and contribute zero BC loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bc_teacher_actions = np.zeros_like(self.actions, dtype=np.float32)
        self.bc_teacher_masks = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos) -> None:  # noqa: D401
        # Parent increments self.pos, so remember where this transition lands.
        insert_pos = self.pos
        super().add(obs, next_obs, action, reward, done, infos)

        if infos is None:
            self.bc_teacher_actions[insert_pos] = 0.0
            self.bc_teacher_masks[insert_pos] = 0.0
            return

        for env_idx in range(self.n_envs):
            info = infos[env_idx] if env_idx < len(infos) and infos[env_idx] is not None else {}
            valid = bool(info.get("bc_teacher_valid", False))
            teacher_action = info.get("bc_teacher_action", None)

            if valid and teacher_action is not None:
                teacher_action = np.asarray(teacher_action, dtype=np.float32).reshape(self.actions.shape[2:])
                if np.all(np.isfinite(teacher_action)):
                    self.bc_teacher_actions[insert_pos, env_idx] = teacher_action
                    self.bc_teacher_masks[insert_pos, env_idx] = 1.0
                    continue

            self.bc_teacher_actions[insert_pos, env_idx] = 0.0
            self.bc_teacher_masks[insert_pos, env_idx] = 0.0

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[Any] = None) -> DistillationDictReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        obs = self._normalize_obs(
            {key: obs_arr[batch_inds, env_indices, :] for key, obs_arr in self.observations.items()},
            env,
        )
        next_obs = self._normalize_obs(
            {key: obs_arr[batch_inds, env_indices, :] for key, obs_arr in self.next_observations.items()},
            env,
        )

        # Match SB3's timeout handling when available.
        timeouts = getattr(self, "timeouts", np.zeros_like(self.dones))
        dones = self.dones[batch_inds, env_indices] * (1 - timeouts[batch_inds, env_indices])

        return DistillationDictReplayBufferSamples(
            observations={key: self.to_torch(obs_arr) for key, obs_arr in obs.items()},
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations={key: self.to_torch(obs_arr) for key, obs_arr in next_obs.items()},
            dones=self.to_torch(dones.reshape(-1, 1)),
            rewards=self.to_torch(
                self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)
            ),
            bc_teacher_actions=self.to_torch(self.bc_teacher_actions[batch_inds, env_indices]),
            bc_teacher_masks=self.to_torch(self.bc_teacher_masks[batch_inds, env_indices].reshape(-1, 1)),
        )


class DistilledSAC(SAC):
    """SAC with an annealed behavior-cloning loss to a frozen teacher."""

    def __init__(
        self,
        *args,
        bc_initial_weight: float = 2.0,
        bc_final_weight: float = 0.05,
        bc_anneal_steps: int = 500_000,
        bc_min_valid_fraction: float = 0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bc_initial_weight = float(bc_initial_weight)
        self.bc_final_weight = float(bc_final_weight)
        self.bc_anneal_steps = int(max(0, bc_anneal_steps))
        self.bc_min_valid_fraction = float(np.clip(bc_min_valid_fraction, 0.0, 1.0))
        # Stage-keyed override: when set (via set_bc_weight), it takes
        # precedence over the timestep anneal in _current_bc_weight().
        # None = legacy timestep-anneal behavior.
        self._bc_stage_weight: Optional[float] = None

    def set_bc_weight(self, weight: Optional[float]) -> None:
        """Set a fixed BC weight (called by the curriculum callback at each
        stage change). Pass None to fall back to the timestep anneal."""
        self._bc_stage_weight = None if weight is None else float(max(0.0, weight))

    def _current_bc_weight(self) -> float:
        # Stage-keyed override wins. getattr guard: models deserialized by
        # SB3 .load() may predate this attribute.
        stage_weight = getattr(self, "_bc_stage_weight", None)
        if stage_weight is not None:
            return stage_weight

        if self.bc_initial_weight <= 0.0 and self.bc_final_weight <= 0.0:
            return 0.0
        if self.bc_anneal_steps <= 0:
            return self.bc_final_weight

        # Start annealing only after SAC actually starts gradient updates.
        t = max(0, int(self.num_timesteps) - int(self.learning_starts))
        frac = min(1.0, t / float(self.bc_anneal_steps))
        return float(self.bc_initial_weight + frac * (self.bc_final_weight - self.bc_initial_weight))

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:  # noqa: D401
        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, actor_sac_losses, critic_losses = [], [], []
        bc_losses, bc_weights, bc_valid_fracs = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(float(ent_coef.item()))

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(float(critic_loss.item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            sac_actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            bc_weight = self._current_bc_weight()
            bc_loss = th.zeros((), device=self.device)
            bc_valid_frac = 0.0

            teacher_actions = getattr(replay_data, "bc_teacher_actions", None)
            teacher_masks = getattr(replay_data, "bc_teacher_masks", None)
            if bc_weight > 0.0 and teacher_actions is not None and teacher_masks is not None:
                mask = teacher_masks.float()
                valid_count = th.clamp(mask.sum(), min=0.0)
                bc_valid_frac = float((valid_count / max(1, mask.shape[0])).detach().cpu().item())

                if bc_valid_frac >= self.bc_min_valid_fraction and valid_count.item() > 0:
                    # Use deterministic actor output for the supervised anchor;
                    # SAC's stochastic sampled action remains used in the entropy/Q term.
                    student_actions = self.actor(replay_data.observations, deterministic=True)
                    sq_err = (student_actions - teacher_actions.detach()).pow(2)
                    bc_loss = (sq_err * mask).sum() / (valid_count * sq_err.shape[1] + 1e-8)

            actor_loss = sac_actor_loss + bc_weight * bc_loss
            actor_losses.append(float(actor_loss.item()))
            actor_sac_losses.append(float(sac_actor_loss.item()))
            bc_losses.append(float(bc_loss.item()))
            bc_weights.append(float(bc_weight))
            bc_valid_fracs.append(float(bc_valid_frac))

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Keep SB3 batch-norm target statistics in sync when present.
                if hasattr(self, "batch_norm_stats") and hasattr(self, "batch_norm_stats_target"):
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", float(np.mean(ent_coef_losses)))
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        self.logger.record("train/actor_sac_loss", float(np.mean(actor_sac_losses)))
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        self.logger.record("train/bc_loss", float(np.mean(bc_losses)))
        self.logger.record("train/bc_weight", float(np.mean(bc_weights)))
        self.logger.record("train/bc_valid_frac", float(np.mean(bc_valid_fracs)))
