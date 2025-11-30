from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast
import re

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.envs.mdp.rewards import *
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)


@generic_io_descriptor(
    observation_type="ContactSensorForce",
    on_inspect=[record_joint_names, record_dtype, record_shape],
    units="N",
)
def contact_sensor_force(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Observation: max net contact force magnitude per body over history.

    Shape: [num_envs, num_bodies], where num_bodies = len(sensor_cfg.body_ids).
    """
    # env.scene.sensors: dict[str, SensorBase] -> cast to ContactSensor
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])

    # net_forces_w_history: [N, H, B, 3]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore

    # magnitude of force vector -> [N, H, B]
    force_mag = net_forces.norm(dim=-1)

    # current over history dimension -> [N, B]
    current_force = force_mag[:, -1, :]
    return current_force
