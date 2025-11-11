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


def base_height_l2_from_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height deviation from command target using L2 squared kernel.
    
    This function retrieves the target height from a pose command and uses it
    with the base_height_l2 reward function.
    
    Args:
        env: The environment instance.
        command_name: Name of the command to retrieve the target height from.
        asset_cfg: Configuration for the asset to track.
        sensor_cfg: Optional sensor configuration for terrain adjustment.
        
    Returns:
        The L2 squared penalty for height deviation.
    """
    # Get pose command: [x, y, z, qx, qy, qz, qw]
    command = env.command_manager.get_command(command_name)
    asset: RigidObject = env.scene[asset_cfg.name]

    # Desired position in base (body) frame — shape [num_envs, 3]
    des_pos_b = command[:, :3]

    # Transform desired pose from body → world frame
    des_pose_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3],   # world position of body origin
        asset.data.root_state_w[:, 3:7],  # world orientation
        des_pos_b,                        # desired pos (body frame)
    )

    # Current CoM world position of the base (rigid body center of mass)
    curr_pos_w = asset.data.body_com_pose_w[:, asset_cfg.body_ids[0], :3] # type: ignore
    # Compute per-env L2 position deviation
    pos_error = torch.norm(curr_pos_w - des_pose_w, dim=1)

    # Optionally, only height deviation (uncomment to isolate Z):
    # pos_error = torch.abs(curr_pos_w[:, 2] - des_pos_w[:, 2])
    return pos_error


def base_x_y_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base x and y position deviation from zero."""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # CoM pose in body frame: shape [N, 1, 7]
    body_com_pose_b = asset.data.body_com_pose_b  # local CoM offset (body frame)
    com_offset_b = body_com_pose_b[:, 0, :3]      # only (x, y, z) offset
    
    # Root link position in world frame
    root_pos_w = asset.data.root_link_pos_w  # shape [N, 3]

    # Compute XY squared difference between world pos and CoM offset (projected)
    diff_xy = torch.square(root_pos_w[:, :2] - com_offset_b[:, :2])

    # Return per-env scalar (sum of x² + y²)
    return torch.sum(diff_xy, dim=1)