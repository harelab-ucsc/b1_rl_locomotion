# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, quat_from_euler_xyz, euler_xyz_from_quat
from isaaclab.envs.mdp.rewards import *

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
    # Get the command from the command manager (returns a tensor)
    command = env.command_manager.get_command(command_name)
    # Extract the target height (z-coordinate) from the pose command
    # Pose command tensor shape is [num_envs, 7] with [x, y, z, qx, qy, qz, qw]
    # So index 2 is the z-coordinate (height)
    target_height = command[:, 2]  # z-coordinate for all environments
    
    # Use the base_height_l2 function with the target height
    # Note: base_height_l2 expects a scalar target_height, but we have a per-env tensor
    # So we need to compute it per environment
    from isaaclab.assets import RigidObject
    asset: RigidObject = env.scene[asset_cfg.name]
    
    if sensor_cfg is not None:
        from isaaclab.sensors import RayCaster
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    
    # Compute the L2 squared penalty per environment
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
