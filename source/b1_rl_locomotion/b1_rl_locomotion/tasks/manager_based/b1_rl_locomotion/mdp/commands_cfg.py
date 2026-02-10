from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omegaconf import MISSING

from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.managers import CommandTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class SequentialHeightCommandCfg(CommandTermCfg):
    class_type: type = MISSING
    name: str = "height"
    asset_name: str = "robot"
    body_name: str = "base"

    # fixed pose components
    pos_x: float = 0.0
    pos_y: float = 0.0

    # sequential z
    z_range: tuple[float, float] = (0.2, 0.7)
    num_steps: int = 11
    wrap: bool = True
    random_start: bool = True

    resampling_time_range: tuple[float, float] = (0.5, 0.5)


@configclass
class UniformPoseCommandAbsoluteCfg(UniformPoseCommandCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = MISSING
