# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def reset_joints_to_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    joint_pos_dict: dict[str, float],
    position_noise_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset joints to specific absolute positions, independent of default_joint_pos.

    Unlike reset_joints_by_offset, this ignores default_joint_pos entirely and writes
    the given positions directly to sim. This lets the robot start in a different pose
    (e.g. laying) while keeping the standing default for joint_pos_rel observations.

    Args:
        joint_pos_dict: Maps joint name patterns to target positions (radians).
        position_noise_range: Uniform noise added to every joint position.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()

    for name_pattern, target_pos in joint_pos_dict.items():
        joint_ids, _ = asset.find_joints(name_pattern)
        joint_pos[:, joint_ids] = target_pos

    lo, hi = position_noise_range
    if lo != hi or lo != 0.0:
        joint_pos += torch.empty_like(joint_pos).uniform_(lo, hi)

    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def scale_joint_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float] = (0.9, 1.1),
):
    """Scale each joint's static friction coefficient by a per-env uniform sample.

    Plain-function variant of `mdp.randomize_joint_parameters` (which is a class
    and runs into instantiation-timing issues at startup/reset). Samples one
    scale factor per (env, joint) and writes to sim.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=device)

    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or (isinstance(joint_ids, slice) and joint_ids == slice(None)):
        joint_ids_t = torch.arange(asset.num_joints, device=device)
        write_joint_ids = None
    else:
        joint_ids_t = torch.as_tensor(joint_ids, device=device, dtype=torch.long)
        write_joint_ids = joint_ids_t

    default_friction = asset.data.default_joint_friction_coeff
    base = default_friction[env_ids[:, None], joint_ids_t[None, :]]

    lo, hi = scale_range
    scale = torch.empty_like(base).uniform_(lo, hi)
    new_friction = (base * scale).clamp_(min=0.0)

    # PhysX requires static >= dynamic friction; clamp to ensure the constraint holds.
    dynamic_friction = asset.data.default_joint_dynamic_friction_coeff
    dyn_base = dynamic_friction[env_ids[:, None], joint_ids_t[None, :]]
    torch.maximum(new_friction, dyn_base, out=new_friction)

    asset.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=new_friction,
        joint_ids=write_joint_ids,
        env_ids=env_ids,
    )
