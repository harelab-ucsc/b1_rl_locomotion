"""PPO variant that ignores transitions flagged invalid via ``infos["valid"]``.

Used together with :class:`B1RlLocomotionEnv`, which sets ``valid=False`` for
the post-reset settle window. Settle transitions are still stored in memory
(memory shape is fixed) but contribute zero gradient to the policy/value
losses, are excluded from the entropy term, and are excluded from the running
state/value preprocessor statistics. GAE itself is computed over the full
buffer — that's harmless because settle advantages are masked to zero before
being used and don't propagate into agent-step advantages (GAE recurses
backward in time, and settle steps come *before* the next agent step).
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Optional, Union

import torch
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.ppo import PPO
from skrl.resources.schedulers.torch import KLAdaptiveLR


class MaskedPPO(PPO):
    """PPO with per-transition validity masking driven by ``infos["valid"]``."""

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        if self.memory is not None:
            self.memory.create_tensor(name="valid", size=1, dtype=torch.bool)
            for m in self.secondary_memories:
                m.create_tensor(name="valid", size=1, dtype=torch.bool)
            # sample_all in _update needs to return valid alongside the rest
            self._tensors_names = [*self._tensors_names, "valid"]

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        # extract the per-env valid flag from infos; default to all-valid
        if isinstance(infos, dict) and "valid" in infos:
            valid = infos["valid"]
            if not torch.is_tensor(valid):
                valid = torch.tensor(valid, device=states.device)
            valid = valid.to(dtype=torch.bool, device=states.device).view(-1, 1)
        else:
            valid = torch.ones((states.shape[0], 1), dtype=torch.bool, device=states.device)

        # write valid into the same memory slot the parent is about to fill,
        # then let the parent's add_samples advance the index for both
        if self.memory is not None:
            self.memory.tensors["valid"][self.memory.memory_index].copy_(valid)
            for m in self.secondary_memories:
                m.tensors["valid"][m.memory_index].copy_(valid)

        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

    def _update(self, timestep: int, timesteps: int) -> None:
        # ---- GAE (identical to PPO._update) -----------------------------------
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        # ---- value preprocessor: train on valid samples only ------------------
        # (PPO trains on all samples; we want running mean/var to reflect only
        #  the agent-controlled portion of the buffer.)
        valid_buf = self.memory.get_tensor_by_name("valid")  # (T, N, 1) bool
        valid_flat = valid_buf.reshape(-1, 1)
        if valid_flat.any():
            _ = self._value_preprocessor(values.reshape(-1, 1)[valid_flat.squeeze(-1)], train=True)
            _ = self._value_preprocessor(returns.reshape(-1, 1)[valid_flat.squeeze(-1)], train=True)
        # store normalized values/returns/advantages without further training
        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=False))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=False))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_valid,
            ) in sampled_batches:

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    valid_mask = sampled_valid.to(dtype=sampled_states.dtype)  # (B, 1)
                    valid_count = valid_mask.sum().clamp(min=1.0)

                    # state preprocessor: train on valid samples only, on first epoch
                    if not epoch and valid_mask.any():
                        _ = self._state_preprocessor(sampled_states[sampled_valid.squeeze(-1)], train=True)
                    sampled_states = self._state_preprocessor(sampled_states, train=False)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        # KL averaged over valid samples only
                        kl_per = (torch.exp(ratio) - 1) - ratio
                        kl_divergence = (kl_per * valid_mask).sum() / valid_count
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # entropy loss (mask out invalid samples)
                    if self._entropy_loss_scale:
                        entropy_per = self.policy.get_entropy(role="policy")
                        # entropy may be (B, action_dim) or (B,); reduce per-sample
                        entropy_per = entropy_per.view(entropy_per.shape[0], -1).mean(dim=-1, keepdim=True)
                        entropy_loss = -self._entropy_loss_scale * (entropy_per * valid_mask).sum() / valid_count
                    else:
                        entropy_loss = 0

                    # policy loss (mask out invalid samples)
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )
                    policy_per = -torch.min(surrogate, surrogate_clipped)
                    policy_loss = (policy_per * valid_mask).sum() / valid_count

                    # value loss (mask out invalid samples)
                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")
                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_per = F.mse_loss(sampled_returns, predicted_values, reduction="none")
                    value_loss = self._value_loss_scale * (value_per * valid_mask).sum() / valid_count

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
