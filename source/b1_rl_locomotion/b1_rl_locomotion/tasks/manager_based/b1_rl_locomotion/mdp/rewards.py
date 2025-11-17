# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.envs.mdp.rewards import *
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_from_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_tanh: bool = False,  # when True, applies tanh to height error (becomes a reward instead of a penalty)
    tanh_scale: float = 0.1,  # d/dx f(x) = -1 at around x=18cm if scale=0.1, where f(x)=1-tanh(x/tanh_scale)
) -> torch.Tensor:
    """Penalize asset height deviation from command target.

    This function retrieves the target height from a pose command and computes
    the diff between the asset's current center of mass (CoM) height and the desired height.

    Args:
        env: The environment instance.
        command_name: Name of the command to retrieve the target height from.
        asset_cfg: Configuration for the asset to track.
        sensor_cfg: Optional sensor configuration for terrain adjustment.

    Returns:
        The per-environment height deviation as a tensor of shape [num_envs].
    """
    # Get pose command: [x, y, z, qx, qy, qz, qw]
    command = env.command_manager.get_command(command_name)
    robot: RigidObject = env.scene[asset_cfg.name]

    # Desired position in base (body) frame — shape [num_envs, 3]
    des_pos_b = command[:, :3]

    # Current CoM world position of the base (rigid body center of mass)
    curr_pos_w = robot.data.body_com_pose_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    # Compute per-env L2 height deviation (ignore xy)
    # pos_error = torch.norm(curr_pos_w - des_pose_w, dim=1)
    height_err = torch.abs(curr_pos_w[:, 2] - des_pos_b[:, 2])

    # # Debug prints
    # print("-------------------------------")
    # print("Desired base position (body):")
    # print(des_pos_b)
    # print("Current base CoM position (world):")
    # print(curr_pos_w)
    # print("Height error:")
    # print(height_err)

    if use_tanh:
        # Apply tanh to convert to a reward (higher is better)
        height_err = 1 - torch.tanh(height_err / tanh_scale)

    return height_err


def body_lin_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body linear velocity L2 norm (don't move). Reurns a positive value (the norm)."""
    robot: RigidObject = env.scene[asset_cfg.name]

    # Compute L2 norm of linear velocity
    lin_vel = robot.data.root_state_w[:, 7:9]  # ignore z velocity
    lin_vel_l2 = torch.norm(lin_vel, dim=1)

    return lin_vel_l2


def center_joints_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_hip_joint"]),
) -> torch.Tensor:
    """Penalize joint position deviation from center (0.0)."""
    robot: Articulation = env.scene[asset_cfg.name]

    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    pos_diff = torch.square(joint_pos)  # squared difference from zero

    # Return per-env scalar (sum over all joints)
    return torch.max(pos_diff, dim=1).values


def base_x_y_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base x and y position deviation from zero."""
    asset: RigidObject = env.scene[asset_cfg.name]

    # CoM pose in body frame: shape [N, 1, 7]
    body_com_pose_b = asset.data.body_com_pose_b  # local CoM offset (body frame)
    com_offset_b = body_com_pose_b[:, 0, :3]  # only (x, y, z) offset

    # Root link position in world frame
    root_pos_w = asset.data.root_link_pos_w  # shape [N, 3]

    # Compute XY squared difference between world pos and CoM offset (projected)
    diff_xy = torch.square(root_pos_w[:, :2] - com_offset_b[:, :2])

    # Return per-env scalar (sum of x² + y²)
    return torch.sum(diff_xy, dim=1)
