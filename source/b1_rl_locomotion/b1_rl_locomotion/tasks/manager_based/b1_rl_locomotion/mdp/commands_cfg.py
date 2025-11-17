from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from .commands import UniformPoseCommandAbsolute

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class UniformPoseCommandAbsoluteCfg(UniformPoseCommandCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformPoseCommandAbsolute
