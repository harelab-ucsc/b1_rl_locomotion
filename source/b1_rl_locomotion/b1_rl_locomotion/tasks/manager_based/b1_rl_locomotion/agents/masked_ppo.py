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
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo.ppo import compute_gae
from skrl.resources.schedulers.torch import KLAdaptiveLR


class MaskedPPO(PPO):
    """PPO with per-transition validity masking driven by ``infos["valid"]``."""

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        if self.memory is not None:
            self.memory.create_tensor(name="valid", size=1, dtype=torch.bool)
            self._tensors_names = [*self._tensors_names, "valid"]

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        # skrl 2.x logs scalars only through its own skrl.utils.tensorboard.SummaryWriter,
        # which wandb's sync_tensorboard monkeypatch never intercepts (it only patches the
        # tensorboard/torch/tensorboardX writer modules). The result is a wandb run with
        # nothing but system stats. Push the reduced scalars to wandb directly here,
        # mirroring the base reduction (min/max/mean), before super() clears the buffers.
        import wandb

        if wandb.run is not None and self.tracking_data:
            data = {}
            for k, v in self.tracking_data.items():
                if k.endswith("(min)"):
                    data[k] = np.min(v)
                elif k.endswith("(max)"):
                    data[k] = np.max(v)
                else:
                    data[k] = np.mean(v)
            wandb.log(data, step=timestep)

        super().write_tracking_data(timestep=timestep, timesteps=timesteps)

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
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
                valid = torch.tensor(valid, device=observations.device)
            valid = valid.to(dtype=torch.bool, device=observations.device).view(-1, 1)
        else:
            valid = torch.ones((observations.shape[0], 1), dtype=torch.bool, device=observations.device)

        # write valid into the same memory slot the parent is about to fill,
        # then let the parent's add_samples advance the index for both
        if self.memory is not None:
            self.memory.tensors["valid"][self.memory.memory_index].copy_(valid)

        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

    def update(self, *, timestep: int, timesteps: int) -> None:
        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor(self._current_next_observations),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
            time_limit_bootstrap=self.cfg.time_limit_bootstrap,
        )

        # value preprocessor: train on valid samples only
        valid_buf = self.memory.get_tensor_by_name("valid")  # (T, N, 1) bool
        valid_flat = valid_buf.reshape(-1, 1).squeeze(-1)
        if valid_flat.any():
            _ = self._value_preprocessor(values.reshape(-1, 1)[valid_flat], train=True)
            _ = self._value_preprocessor(returns.reshape(-1, 1)[valid_flat], train=True)
        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=False))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=False))
        self.memory.set_tensor_by_name("advantages", advantages)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_kl = 0
        cumulative_clip_fraction = 0
        num_minibatches = 0

        # learning epochs
        for epoch in range(self.cfg.learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_valid,
            ) in self.memory.sample(
                names=self._tensors_names, batch_size=len(self.memory), mini_batches=self.cfg.mini_batches
            ):

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    valid_mask = sampled_valid.to(dtype=sampled_observations.dtype)  # (B, 1)
                    valid_count = valid_mask.sum().clamp(min=1.0)
                    valid_idx = sampled_valid.squeeze(-1)

                    # preprocessors: train on valid samples only, on first epoch
                    if not epoch and valid_idx.any():
                        _ = self._observation_preprocessor(sampled_observations[valid_idx], train=True)
                        if sampled_states is not None:
                            _ = self._state_preprocessor(sampled_states[valid_idx], train=True)
                    inputs = {
                        "observations": self._observation_preprocessor(sampled_observations, train=False),
                        "states": self._state_preprocessor(sampled_states, train=False) if sampled_states is not None else None,
                    }

                    _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    # compute approximate KL divergence (over valid samples only)
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_per = (torch.exp(ratio) - 1) - ratio
                        kl_divergence = (kl_per * valid_mask).sum() / valid_count
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self.cfg.kl_threshold and kl_divergence > self.cfg.kl_threshold:
                        break

                    # entropy loss (mask out invalid samples)
                    if self.cfg.entropy_loss_scale:
                        entropy_per = self.policy.get_entropy(role="policy")
                        entropy_per = entropy_per.view(entropy_per.shape[0], -1).mean(dim=-1, keepdim=True)
                        entropy_loss = -self.cfg.entropy_loss_scale * (entropy_per * valid_mask).sum() / valid_count
                    else:
                        entropy_loss = 0

                    # policy loss (mask out invalid samples)
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    # fraction of valid samples whose ratio hit the clip bound
                    with torch.no_grad():
                        clipped = (torch.abs(ratio - 1.0) > self.cfg.ratio_clip).to(valid_mask.dtype)
                        clip_fraction = (clipped * valid_mask).sum() / valid_count
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    )
                    policy_per = -torch.min(surrogate, surrogate_clipped)
                    policy_loss = (policy_per * valid_mask).sum() / valid_count

                    # value loss (mask out invalid samples)
                    predicted_values, _ = self.value.act(inputs, role="value")
                    if self.cfg.value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self.cfg.value_clip, max=self.cfg.value_clip
                        )
                    value_per = F.mse_loss(sampled_returns, predicted_values, reduction="none")
                    value_loss = self.cfg.value_loss_scale * (value_per * valid_mask).sum() / valid_count

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()),
                            self.cfg.grad_norm_clip,
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                cumulative_kl += kl_divergence.item()
                cumulative_clip_fraction += clip_fraction.item()
                num_minibatches += 1

            # update learning rate
            if self.scheduler:
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
        self.track_data(
            "Loss / Policy loss", cumulative_policy_loss / (self.cfg.learning_epochs * self.cfg.mini_batches)
        )
        self.track_data("Loss / Value loss", cumulative_value_loss / (self.cfg.learning_epochs * self.cfg.mini_batches))
        if self.cfg.entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self.cfg.learning_epochs * self.cfg.mini_batches)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        # approximate KL divergence and policy clip ratio (averaged over minibatches)
        if num_minibatches:
            self.track_data("Policy / Approximate KL", cumulative_kl / num_minibatches)
            self.track_data("Policy / Clip ratio", cumulative_clip_fraction / num_minibatches)

        # explained variance of the value model: 1 - Var(returns - values) / Var(returns),
        # computed over valid samples (values/returns share the same value-preprocessor scale)
        with torch.no_grad():
            y_pred = self.memory.get_tensor_by_name("values").reshape(-1, 1)[valid_flat]
            y_true = self.memory.get_tensor_by_name("returns").reshape(-1, 1)[valid_flat]
            var_y = y_true.var()
            explained_variance = float("nan") if var_y == 0 else (1.0 - (y_true - y_pred).var() / var_y).item()
        self.track_data("Value / Explained variance", explained_variance)

        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
