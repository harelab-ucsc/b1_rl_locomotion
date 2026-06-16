#   Custom Commands - UCSC HARE Lab - November 2025
#   author: oyoung@ucsc.edu Oliver Young
#
#   TerrainBasedPose3dCommandCfg: custom command for 3D position based off terrain
#

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.envs.mdp.rewards import *
from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

############################################################
# TODO: Impement TerrainBasedPose3dCommand
#
#
# def TerrainBasedPose3dCommand(


############################################################
#   TerrainBasedPose2dCommand from: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/commands/pose_2d_command.py
#
#   To be used as reference for implementing TerrainBasedPose3dCommand
#


class UniformPoseCommandAbsolute(UniformPoseCommand):
    """
    Command generator that generates absolute pose commands.
    This command generator samples the position commands from a uniform distribution
    within specified ranges. The heading commands are either set to point towards the target or are sampled uniformly.
    It expects the terrain to have a valid flat patches under the key 'target'.

    Height ramping: when ``cfg.ramp_duration > 0``, the commanded height (z) is not
    held at the sampled value for the whole episode. Instead it ramps linearly from
    ``cfg.start_height`` (the prone CoM height) up to the sampled target height over
    ``cfg.ramp_duration`` seconds of episode time, then holds. This turns "stand up"
    into tracking a slowly-rising setpoint so the policy cannot jump ahead of it.

    Keypoint ramping: when ``cfg.keypoint_times``/``cfg.keypoint_heights`` are set, the
    commanded height instead follows a piecewise-linear trajectory through those
    (time, height) keypoints (held at the last height afterwards). Unevenly-spaced
    keypoints let the setpoint rise slowly at first and faster later. This overrides
    ``ramp_duration``/``start_height``.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # cache keypoint trajectory tensors (if configured) for fast per-step interp
        self._kp_times = None
        if cfg.keypoint_times is not None and cfg.keypoint_heights is not None:
            assert len(cfg.keypoint_times) == len(cfg.keypoint_heights), (
                "keypoint_times and keypoint_heights must have the same length"
            )
            self._kp_times = torch.tensor(
                cfg.keypoint_times, dtype=torch.float, device=self.device
            )
            self._kp_heights = torch.tensor(
                cfg.keypoint_heights, dtype=torch.float, device=self.device
            )

    def _resample_command(self, env_ids):
        # sample the pose normally; this writes the FINAL standing height into
        # pose_command_b[:, 2]. Stash it so the per-step ramp can interpolate toward it.
        super()._resample_command(env_ids)
        if not hasattr(self, "_final_z"):
            self._final_z = torch.zeros(self.num_envs, device=self.device)
        self._final_z[env_ids] = self.pose_command_b[env_ids, 2].clone()

    def _update_command(self):
        elapsed = self._env.episode_length_buf.float() * self._env.step_dt
        if self._kp_times is not None:
            # follow a piecewise-linear height trajectory through the keypoints,
            # held at the last keypoint height afterwards.
            self.pose_command_b[:, 2] = self._interp_keypoints(elapsed)
            return
        # ramp the commanded height from start_height up to the sampled final height
        ramp_duration = getattr(self.cfg, "ramp_duration", 0.0)
        if ramp_duration and ramp_duration > 0.0:
            phase = torch.clamp(elapsed / ramp_duration, 0.0, 1.0)
            start_height = getattr(self.cfg, "start_height", 0.0)
            self.pose_command_b[:, 2] = start_height + phase * (self._final_z - start_height)

    def _interp_keypoints(self, t):
        # piecewise-linear interpolation of keypoint_heights over keypoint_times,
        # clamped to the endpoints (hold first height before t0, last height after tN).
        times, heights = self._kp_times, self._kp_heights
        # index of the segment each t falls into: i s.t. times[i] <= t < times[i+1]
        idx = torch.searchsorted(times, t, right=True) - 1
        idx = idx.clamp(0, times.numel() - 2)
        t0, t1 = times[idx], times[idx + 1]
        h0, h1 = heights[idx], heights[idx + 1]
        frac = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
        return h0 + frac * (h1 - h0)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose

        # combine body xy and world z
        goal_pos_position = self.robot.data.body_link_pose_w[:, self.body_idx]
        goal_pos_position[:, 2:] = self.pose_command_b[:, 2:]

        self.goal_pose_visualizer.visualize(
            goal_pos_position[:, :3], goal_pos_position[:, 3:]
        )
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(
            body_link_pose_w[:, :3], body_link_pose_w[:, 3:7]
        )
