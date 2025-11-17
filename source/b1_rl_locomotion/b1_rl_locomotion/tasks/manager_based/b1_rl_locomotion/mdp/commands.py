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
