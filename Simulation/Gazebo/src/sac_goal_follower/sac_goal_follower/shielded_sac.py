#!/usr/bin/env python3
"""
ShieldedSAC — SAC variant that stores the *actually executed* action in the
replay buffer rather than the raw policy action.

## The bug this fixes

Standard SAC's collect_rollouts loop stores transitions of the form:

    (s, a_policy, r, s')

where a_policy is the action sampled from the actor.  But when a safety shield
rewrites the action between the policy output and the physics step, the action
that actually produced s' was a_shielded, not a_policy.  The critic therefore
learns a fictional dynamics model: "from state s, taking action a_policy leads
to state s' with reward r" — when in reality a_policy was discarded.

When the shield intervenes 30% of the time, 30% of all replay transitions have
wrong dynamics.  The critic eventually accumulates conflicting evidence and
its Q-estimates become unreliable for shielded actions; this propagates into
unstable actor gradients (rising actor loss, oscillating success rate).

## The fix

We override _store_transition to swap the buffer_action with the executed
action exposed by the environment in info["executed_action_normalized"].
The critic then learns the true dynamics:

    (s, a_shielded, r, s')

which is consistent with reality.  The actor still receives gradient signal
toward higher-Q actions, but now those Q-values reflect what actually happens
when those actions are executed.

## Compatibility

Drop-in replacement for stable_baselines3.SAC.  No API changes; the same
load/save/learn methods work identically.  Episodes that come from envs
without the info field (e.g. evaluation envs without the shield wrapper)
fall back to the original buffer_action — no breakage.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update


class ShieldedSAC(SAC):
    """
    SAC variant that stores the executed action (post-shield) in the
    replay buffer instead of the raw policy action, AND optionally adds
    a shield-imitation auxiliary loss to the actor's objective.

    Two upgrades over standard SAC:

    1. **Action-correction in the buffer** (always on): the action stored
       in each transition is the action that actually executed in the env
       (read from info["executed_action_normalized"]), not the raw policy
       output.  This eliminates the dynamics mismatch that occurs when a
       safety shield rewrites actions between policy and physics.

    2. **Shield imitation loss** (opt-in, off by default): when the shield
       intervenes on a transition, the actor is also trained to output an
       action closer to the executed (shielded) action.  This converts the
       shield from a passive corrector into an active teacher.  The loss
       is masked so it only applies to transitions where the shield was
       active — non-shielded transitions are unaffected.

           L_actor_total = L_actor_SAC + β * mean_{shielded}( ||π(s) - a_buf||² )

       β is exposed via `set_imitation_beta()` so the training script can
       schedule it via the curriculum (start at 0.0, raise as stages
       progress).

    Behaves identically to SAC when imitation_beta=0.0 and no info field.
    """

    INFO_KEY = "executed_action_normalized"

    def __init__(
        self,
        *args,
        log_action_replacement_stats: bool = True,
        imitation_beta: float = 0.0,
        imitation_severity_index: int = 12,
        imitation_severity_threshold: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Action-replacement diagnostic counters
        self._action_replace_count: int = 0
        self._action_total_count: int = 0
        self._action_replace_l2_sum: float = 0.0
        self._log_action_replacement_stats = bool(log_action_replacement_stats)

        # Shield imitation loss configuration.
        #   imitation_beta:                 multiplier on the imitation loss term.
        #                                   0.0 disables imitation entirely.
        #   imitation_severity_index:       index within the per-step `fixed`
        #                                   subvector where shield_severity lives.
        #                                   With the current env layout
        #                                   (14-dim fixed: ..., shield_severity[12],
        #                                   shield_duration_norm[13]) → 12.
        #                                   If FrameStackWrapper is used with K
        #                                   frames, the most recent frame is at
        #                                   the tail (newest-last ordering), so
        #                                   the runtime index becomes
        #                                   (K-1)*14 + 12.  Set this via
        #                                   `set_imitation_severity_index()`
        #                                   after wrapping.
        #   imitation_severity_threshold:   transitions with severity above this
        #                                   value are treated as "shield active".
        #                                   0.01 is just above the noise floor.
        self.imitation_beta = float(imitation_beta)
        self.imitation_severity_index = int(imitation_severity_index)
        self.imitation_severity_threshold = float(imitation_severity_threshold)

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Build a corrected action array.  Start from the original buffer_action
        # so that any env which does NOT publish the executed action falls
        # through unchanged.
        corrected_action = buffer_action.copy()

        n_envs = len(infos)
        for env_i in range(n_envs):
            info = infos[env_i]
            executed = info.get(self.INFO_KEY, None)
            if executed is None:
                continue

            # Defensive: ensure shape and dtype match buffer_action's row
            executed_np = np.asarray(executed, dtype=buffer_action.dtype)
            if executed_np.shape != buffer_action[env_i].shape:
                # Mismatched shape — skip this transition rather than crash
                continue

            # Compute the L2 delta between policy and executed for diagnostics
            delta = float(np.linalg.norm(executed_np - buffer_action[env_i]))
            self._action_total_count += 1
            if delta > 1e-6:
                self._action_replace_count += 1
                self._action_replace_l2_sum += delta

            corrected_action[env_i] = executed_np

        # Optionally surface the running stats into the SB3 logger so they
        # show up in TensorBoard alongside the standard losses.
        if (
            self._log_action_replacement_stats
            and self.logger is not None
            and self._action_total_count > 0
            and self._action_total_count % 1000 == 0
        ):
            replace_frac = self._action_replace_count / max(1, self._action_total_count)
            mean_delta = self._action_replace_l2_sum / max(1, self._action_replace_count)
            self.logger.record("shield/action_replace_fraction", replace_frac)
            self.logger.record("shield/action_replace_mean_l2", mean_delta)
            self.logger.record("shield/action_total_seen", self._action_total_count)

        # Delegate to the standard SAC store path with the corrected action.
        super()._store_transition(
            replay_buffer, corrected_action, new_obs, reward, dones, infos
        )

    # ------------------------------------------------------------------ #
    # Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def get_action_replacement_stats(self) -> Dict[str, float]:
        """Return a snapshot of how often actions were rewritten in the buffer."""
        total = max(1, self._action_total_count)
        replaced = self._action_replace_count
        mean_l2 = (self._action_replace_l2_sum / max(1, replaced))
        return {
            "total_transitions_seen": float(self._action_total_count),
            "transitions_with_replacement": float(replaced),
            "replacement_fraction": replaced / total,
            "mean_l2_delta_when_replaced": mean_l2,
        }

    def reset_action_replacement_stats(self) -> None:
        self._action_replace_count = 0
        self._action_total_count = 0
        self._action_replace_l2_sum = 0.0

    # ------------------------------------------------------------------ #
    # Shield imitation loss configuration                                 #
    # ------------------------------------------------------------------ #

    def set_imitation_beta(self, beta: float) -> None:
        """Update the imitation-loss multiplier (typically from a curriculum)."""
        beta = float(beta)
        if beta < 0.0:
            raise ValueError(f"imitation_beta must be >= 0, got {beta}")
        self.imitation_beta = beta

    def set_imitation_severity_index(self, index: int) -> None:
        """
        Set the index within next_obs['fixed'] where shield_severity lives.

        With the raw env, shield_severity is at index 12 of the 14-dim fixed
        vector.  With FrameStackWrapper(k=K) using newest-last ordering, the
        most recent frame is at the tail, so the index becomes (K-1)*14 + 12.
        """
        self.imitation_severity_index = int(index)

    # ------------------------------------------------------------------ #
    # Custom train() with shield imitation loss                           #
    # ------------------------------------------------------------------ #

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Modified SAC training step.  Identical to stable_baselines3.SAC.train()
        except the actor loss is augmented with a shield-imitation term when
        `imitation_beta > 0`.  When `imitation_beta == 0` this method
        produces the exact same updates as the parent class.
        """
        # Switch to training mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to the lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        imitation_losses: List[float] = []
        imitation_mask_fractions: List[float] = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            # ============== Shield imitation loss term ==============
            #
            # When the shield intervened on a stored transition, train the
            # actor to output something closer to the action that was
            # actually executed (which the replay buffer holds, thanks to
            # the action-correction fix above).  This converts the shield
            # from a passive corrector into an active teacher.
            #
            # Mask is derived from next_obs's shield_severity field — that
            # field reflects the shield level applied during the transition
            # that produced next_obs.
            if self.imitation_beta > 0.0:
                # Get deterministic mean action of current policy.  Pass
                # deterministic=True so the policy returns its mean (no
                # exploration noise).  Shape: [batch, action_dim].
                mean_actions = self.actor(
                    replay_data.observations, deterministic=True
                )

                # Build the shield-active mask from next_obs['fixed'].
                # If frame stacking is enabled, the index should point into
                # the most-recent frame (configured at startup).
                try:
                    next_fixed = replay_data.next_observations["fixed"]
                    shield_severity = next_fixed[:, self.imitation_severity_index]
                    shield_mask = (
                        shield_severity > self.imitation_severity_threshold
                    ).float().unsqueeze(-1)  # [batch, 1]

                    # Squared L2 per-transition
                    action_diff_sq = (
                        (mean_actions - replay_data.actions) ** 2
                    ).sum(dim=1, keepdim=True)  # [batch, 1]

                    # Masked mean (avoid div-by-zero when no shield activity
                    # in this batch)
                    mask_sum = shield_mask.sum()
                    if float(mask_sum.item()) > 0.5:
                        imitation_loss = (
                            self.imitation_beta
                            * (action_diff_sq * shield_mask).sum()
                            / mask_sum
                        )
                        actor_loss = actor_loss + imitation_loss
                        imitation_losses.append(float(imitation_loss.item()))
                        imitation_mask_fractions.append(
                            float(shield_mask.mean().item())
                        )
                    else:
                        imitation_mask_fractions.append(0.0)
                except (KeyError, IndexError) as e:
                    # Misconfigured index or missing 'fixed' key — skip
                    # imitation rather than crash the training step.
                    pass
            # ========================================================

            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                # Copy running stats, see GH-1063
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        # Imitation diagnostics — only logged when imitation is active.
        if self.imitation_beta > 0.0:
            self.logger.record("shield/imitation_beta", self.imitation_beta)
            if len(imitation_losses) > 0:
                self.logger.record(
                    "shield/imitation_loss_mean", float(np.mean(imitation_losses))
                )
            if len(imitation_mask_fractions) > 0:
                self.logger.record(
                    "shield/imitation_mask_fraction",
                    float(np.mean(imitation_mask_fractions)),
                )
