# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import re

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
    height_err = torch.abs(curr_pos_w[:, 2] - des_pos_b[:, 2])
    height_err_square = torch.square(height_err)

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
        height_err_tanh = torch.square(1 - torch.tanh(height_err / tanh_scale))
        return height_err_tanh

    return height_err_square


def joint_pos_target_error_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target: dict[str, float],
    use_tanh: bool = False,  # when True, applies tanh to height error (becomes a reward instead of a penalty)
    tanh_scale: float = 0.5,  # d/dx f(x) = -1 at around x=20deg(~0.35rad) if scale=0.5, where f(x)= (1-tanh(err / scale)^2
) -> torch.Tensor:
    """
    Joint-position tracking penalty or reward.

    Computes the deviation between the robot's current joint positions and
    target joint positions specified by a regex-keyed dictionary.

    If use_tanh is False:
        Returns a per-environment penalty equal to the squared L2 norm:
            penalty = || q - q* ||₂²
                    = sum_j (q_j - q*_j)²

    If use_tanh is True:
        Computes a per-joint reward using the shaping:
            r_j = (1 - tanh(|q_j - q*_j| / tanh_scale))²
        and returns the per-environment reward obtained by averaging across joints:
            reward = mean_j r_j

    Args:
        asset_cfg: Configuration specifying which robot joints to evaluate.
        target: Dict mapping regex patterns → desired joint angles (radians).
        use_tanh: Whether to output a shaped reward instead of a penalty.
        tanh_scale: Scale parameter controlling tanh saturation.

    Returns:
        Tensor of shape [num_envs], one scalar penalty or reward per environment.
    """

    robot: Articulation = env.scene[asset_cfg.name]

    all_joint_names = robot.data.joint_names
    if isinstance(asset_cfg.joint_ids, slice):
        joint_names = list(all_joint_names)
    else:
        joint_names = [all_joint_names[i] for i in asset_cfg.joint_ids]

    # simple helper to find value from matching regex key in target
    def find_target_value(name: str) -> float:
        for key, value in target.items():
            if re.match(key, name):
                return value
        raise KeyError(
            f"Could not find target value for joint name '{name}' in target dict."
        )

    # build simple tensor a single robot's desired joint positions, and multiple envs time to get the whole thing
    desired_pos_single = torch.tensor(
        [find_target_value(n) for n in joint_names], device=env.device
    )
    desired_pos = desired_pos_single.unsqueeze(0).repeat(env.num_envs, 1)

    # print("[DEBUG] joint_pos_target_error_l2: desired_pos.shape =", desired_pos.shape)

    # print(
    #     "[DEBUG] joint_pos_target_error_l2: robot.data.joint_pos.shape =",
    #     robot.data.joint_pos[: len(joint_names)].shape,
    # )

    # compute squared L2 square error
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    diff = joint_pos - desired_pos

    error_l2_squared = (
        diff.square().sum(dim=1)
    )  # maximum joint's squared L2 norm

    if use_tanh:
        # per-joint absolute errors
        abs_diff = diff.abs()  # [N, J]

        # per-joint rewards from tanh shaping
        per_joint_reward = (1.0 - torch.tanh(abs_diff / tanh_scale)) ** 2  # [N, J]

        # take min tanh reward over joints
        error_l2_tanh = per_joint_reward.sum(dim=1)  # [N]
        return error_l2_tanh

    return error_l2_squared


def body_lin_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x: bool = True,
    y: bool = True,
    z: bool = True,
) -> torch.Tensor:
    """
    Penalize body linear velocity L2 norm (don't move).
    Can choose which axes to include.
    Returns a positive value (the norm).
    """
    robot: RigidObject = env.scene[asset_cfg.name]

    # Compute L2 norm of linear velocity
    indices = []
    if x:
        indices.append(7)
    if y:
        indices.append(8)
    if z:
        indices.append(9)

    lin_vel = robot.data.root_state_w[:, indices]  # envs x 2 or 3
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


def air_time_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize feet air time based on contact sensor net force history."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # type: ignore

    # print("[DEBUG] Air time penalty:", air_time, air_time.shape)
    return air_time.max(dim=1).values


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


def threshold_contact_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    no_contact_penalty: float = 0.2,
    max_thresholds_offset: float = 200.0,  # 200 N above trigger is too much, starts penalizing
    trigger_threshold: float = 588.0,  # minimum force at which we start scaling reward; below this we apply no_contact_penalty
) -> torch.Tensor:
    """
    Reward based on contact sensor net force with thresholds.
    Reward gentle contact, penalize too much contact, optionally penalize
    no/very weak contact with a fixed penalty.

    Let max_threshold = trigger_threshold + max_thresholds_offset.

    - max_force <= trigger_threshold: constant penalty = -no_contact_penalty
    - trigger_threshold < max_force < max_threshold: positive reward,
      decreasing linearly from 1 down to 0
    - max_force = max_threshold: zero reward
    - max_force > max_threshold: negative reward (too strong contact)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # N = num envs, H = history length, B = num bodies
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore # [N, H, B, 3]
    force_magnitudes = net_forces.norm(dim=-1)  # [N, H, B]
    max_force, _ = force_magnitudes.max(dim=1)  # [N, B]

    # print("[DEBUG] Threshold contact reward: max_force =", max_force, max_force.shape)

    mask = (max_force > trigger_threshold).float()  # [N, B]

    raw_rew = -(max_force - trigger_threshold) / max_thresholds_offset + 1.0
    raw_rew += no_contact_penalty  # [N, B]

    reward = (raw_rew * mask) - no_contact_penalty  # [N, B]

    # average over all bodies
    reward = reward.mean(dim=1)  # [N]

    # print("[DEBUG] Sparse threshold contact reward:", reward, reward.shape)

    return reward


def flat_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 norm.

    This is computed by penalizing the xy-component norm of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
