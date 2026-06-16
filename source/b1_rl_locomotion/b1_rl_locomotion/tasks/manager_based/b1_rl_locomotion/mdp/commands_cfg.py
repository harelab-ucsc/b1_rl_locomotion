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

    ramp_duration: float = 0.0
    """Seconds over which the commanded height ramps from ``start_height`` up to the
    sampled target height. 0 disables ramping (constant target, original behavior).

    Ignored when ``keypoint_times``/``keypoint_heights`` are set."""

    start_height: float = 0.0
    """Height (m) the ramp starts from — should be ~ the prone CoM height after reset."""

    keypoint_times: list[float] | None = None
    """Episode-time breakpoints (s, ascending, starting at 0.0) for the commanded
    height trajectory. Paired element-wise with ``keypoint_heights``. The commanded
    height is piecewise-linearly interpolated between keypoints and held at the last
    value afterwards. Use unevenly-spaced keypoints to move slower at first and faster
    later (e.g. ``[0.0, 2.0, 3.0, 3.5]``). Overrides ``ramp_duration``/``start_height``."""

    keypoint_heights: list[float] | None = None
    """Commanded heights (m) at each entry of ``keypoint_times``. Must be the same
    length as ``keypoint_times``."""
