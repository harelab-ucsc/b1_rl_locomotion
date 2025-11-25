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
    tanh_scale: float = 0.18,  # d/dx f(x) = -1 at around x=18.5cm if scale=0.18, where f(x)=(1-tanh(x/tanh_scale))^2
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
    # Compute per-env height deviation (ignore xy)
    height_err = torch.square(torch.abs(curr_pos_w[:, 2] - des_pos_b[:, 2]))

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

    # print("[DEBUG] Body linear velocity L2 norm:", lin_vel_l2, lin_vel_l2.shape)

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


def strict_desired_contacts_penalty(
    env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Penalize if any of the desired contacts are non-present."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        .max(dim=1)[0]
        > threshold
    )
    all_contact = ~(contacts.all(dim=1))  # invert: True if any contact missing

    # print("[DEBUG] Strict desired contacts penalty:", all_contact, all_contact.shape)

    return all_contact.float()


def foot_slip_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,  # what counts as "in contact"
    slip_threshold: float = 0.5,  # m/s
) -> torch.Tensor:
    """Penalize foot slip based on contact sensor net force history."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
    force_magnitudes = net_forces.norm(dim=-1)
    max_force, _ = force_magnitudes.max(dim=1)

    # trigger mask
    mask = (max_force > contact_threshold).float()  # env x 4

    robot: Articulation = env.scene[asset_cfg.name]
    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :].norm(
        dim=2
    )  # env x 4

    # apply the slip threshold
    slip_mask = (foot_vel_w > slip_threshold).float()

    # final penalty (max over feet)
    penalty = (slip_mask * mask).max(dim=1).values

    # print("[DEBUG] Foot slip penalty:", penalty, penalty.shape)

    return penalty


def sparse_threshold_contact_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    max_threshold: float = 3.0,  # 3 N: above this is "too much" force
    trigger_threshold: float = 0.0,  # minimum force to start rewarding/penalizing
    value_clip: float = 1.0,
) -> torch.Tensor:
    """
    - max_force < max_threshold = positive reward (gentle contact)
    - max_force = max_threshold = zero
    - max_force > max_threshold = negative reward (too strong contact)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
    force_magnitudes = net_forces.norm(dim=-1)
    max_force, _ = force_magnitudes.max(dim=1)

    # trigger mask
    mask = (max_force > trigger_threshold).float()

    # cubic shape around max_threshold
    offset = max_force - max_threshold
    shaped = -(offset**3)

    # clamp
    shaped = torch.clamp(shaped, min=-value_clip, max=value_clip)

    # apply mask
    reward = shaped * mask  # envs x 1, need to squeeze
    reward = reward.squeeze(1)

    # print("[DEBUG] Sparse threshold contact reward:", reward, reward.shape)

    return reward
