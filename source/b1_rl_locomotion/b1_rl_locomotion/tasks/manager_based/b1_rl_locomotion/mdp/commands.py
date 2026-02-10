#   Custom Commands - UCSC HARE Lab - November 2025
#   author: oyoung@ucsc.edu Oliver Young
#
#   TerrainBasedPose3dCommandCfg: custom command for 3D position based off terrain
#

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, CommandTerm
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.envs.mdp.rewards import *
from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from .commands_cfg import SequentialHeightCommandCfg

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
    """

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

class SequentialHeightCommand(CommandTerm):
    cfg: SequentialHeightCommandCfg

    def __init__(self, cfg: SequentialHeightCommandCfg, env):
        super().__init__(cfg, env)

        self._asset = env.scene[cfg.asset_name]
        self._body_id = self._asset.find_bodies(cfg.body_name)[0]

        B = self.num_envs
        device = self.device

        self._z_seq = torch.linspace(cfg.z_range[0], cfg.z_range[1], steps=cfg.num_steps, device=device)
        self._S = self._z_seq.numel()

        self._idx = torch.randint(0, self._S, (B,), device=device) if cfg.random_start else torch.zeros((B,), dtype=torch.long, device=device)

        # CHANGE: 7D pose command buffer
        self._command = torch.zeros((B, 7), device=device)

        # init
        env_ids = torch.arange(B, device=device)
        self._write_pose(env_ids)
        self.time_left[:] = 0.0
        self._resample(env_ids)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _write_pose(self, env_ids: torch.Tensor):
        # fixed x,y; sequential z
        self._command[env_ids, 0] = self.cfg.pos_x
        self._command[env_ids, 1] = self.cfg.pos_y
        self._command[env_ids, 2] = self._z_seq[self._idx[env_ids]]
        print(f"DEBUGGING: Z COMMAND-env0: {self._z_seq[self._idx[0]]}")

        # identity quaternion (matches roll=pitch=yaw=0)
        self._command[env_ids, 3] = 1.0
        self._command[env_ids, 4] = 0.0
        self._command[env_ids, 5] = 0.0
        self._command[env_ids, 6] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        self._idx[env_ids] += 1
        if self.cfg.wrap:
            self._idx[env_ids] %= self._S
        else:
            self._idx[env_ids] = torch.clamp(self._idx[env_ids], max=self._S - 1)

        self._write_pose(env_ids)

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass
