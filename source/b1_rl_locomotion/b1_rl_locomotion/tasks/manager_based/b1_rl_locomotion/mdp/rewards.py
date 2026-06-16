# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
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


def delayed(reward_func, delay_seconds: float):
    """Wrap a reward function so it returns 0 for the first ``delay_seconds`` after each env's reset.

    The wrapped function delegates to ``reward_func``, then multiplies the result by a
    per-env mask that is 1 once ``env.episode_length_buf * env.step_dt >= delay_seconds``.

    The wrapper is registered as an attribute on this module under a stable name
    (``_delayed__<reward_func.__name__>__<delay>s``) so Hydra's ``string_to_callable``
    can re-resolve it after serializing the cfg through a dict.

    Usage:
        my_reward = RewTerm(
            func=mdp.delayed(mdp.base_height_from_command, delay_seconds=1.0),
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
                    "command_name": "height"},
            weight=-0.15,
        )
    """
    inner_name = getattr(reward_func, "__name__", "reward")
    # sanitize delay into a stable identifier suffix (e.g. 1.5 -> "1p5", -0.1 -> "n0p1")
    delay_token = repr(delay_seconds).replace(".", "p").replace("-", "n")
    wrapper_name = f"_delayed__{inner_name}__{delay_token}s"

    module = sys.modules[__name__]
    existing = getattr(module, wrapper_name, None)
    if existing is not None:
        return existing

    def wrapped(env, **kwargs):
        value = reward_func(env, **kwargs)
        elapsed = env.episode_length_buf.to(value.dtype) * env.step_dt
        mask = (elapsed >= delay_seconds).to(value.dtype)
        return value * mask

    wrapped.__name__ = wrapper_name
    wrapped.__wrapped__ = reward_func
    setattr(module, wrapper_name, wrapped)
    return wrapped


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


def base_height_above_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sparse reward for the base CoM reaching the commanded height.

    Returns 1.0 for each env whose base center of mass is at or above the
    commanded target height, and 0.0 otherwise.

    Args:
        env: The environment instance.
        command_name: Name of the command to retrieve the target height from.
        asset_cfg: Configuration for the asset to track.

    Returns:
        A per-environment tensor of shape [num_envs] with values in {0.0, 1.0}.
    """
    # Get pose command: [x, y, z, qx, qy, qz, qw]
    command = env.command_manager.get_command(command_name)
    robot: RigidObject = env.scene[asset_cfg.name]

    # Desired height in base (body) frame
    des_height = command[:, 2]

    # Current CoM world height of the base (rigid body center of mass)
    curr_height = robot.data.body_com_pose_w[:, asset_cfg.body_ids[0], 2]  # type: ignore

    return (curr_height >= des_height).float()


def joint_pos_target_error_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target: dict[str, float],
    use_tanh: bool = False,  # when True, applies tanh to height error (becomes a reward instead of a penalty)
    tanh_scale: float = 0.5,  # d/dx f(x) = -1 at around x=20deg(~0.35rad) if scale=0.5, where f(x)= (1-tanh(err / scale)^2
    start_target: dict[str, float] | None = None,  # if set, ramp desired pose from this (prone) pose to `target`
    ramp_duration: float = 5.0,  # seconds over which the desired pose ramps start_target -> target
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

    # optionally ramp the desired pose from a start (prone) pose up to `target`
    # over `ramp_duration` seconds of episode time, so the policy tracks a slowly
    # changing joint setpoint instead of being rewarded for snapping to standing.
    if start_target is not None:
        start_pos_single = torch.tensor(
            [
                next(v for k, v in start_target.items() if re.match(k, n))
                for n in joint_names
            ],
            device=env.device,
        )
        phase = torch.clamp(
            env.episode_length_buf.float() * env.step_dt / ramp_duration, 0.0, 1.0
        )  # [N]
        # desired = start + phase * (end - start), broadcast over joints
        desired_pos = start_pos_single.unsqueeze(0) + phase.unsqueeze(1) * (
            desired_pos - start_pos_single.unsqueeze(0)
        )

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


def body_lin_vel_down(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize *downward* (negative) z linear velocity only; upward motion is free.
    Returns the squared downward speed (positive), zero when moving up or still.
    """
    robot: RigidObject = env.scene[asset_cfg.name]

    vel_z = robot.data.root_state_w[:, 9]  # z linear velocity
    down_speed = (-vel_z).clamp(min=0.0)  # downward magnitude, 0 if moving up

    return down_speed.square()


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
