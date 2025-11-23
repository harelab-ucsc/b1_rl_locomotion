# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg

from . import mdp

##
# Pre-defined configs
##

from b1_rl_locomotion.tasks.manager_based.b1_rl_locomotion.configs.b1 import B1_CFG

##
# Scene definition
##


@configclass
class B1RlLocomotionSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(500.0, 500.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robot
    robot: ArticulationCfg = B1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    contact_forces_body = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/b1_description/base",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["/World/ground"],
    )

    contact_forces_thighs = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/b1_description/.*_thigh",
        update_period=0.0,
        history_length=6,
        force_threshold=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["/World/ground"],
    )

    contact_forces_feet = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/b1_description/.*_foot",
        update_period=0.0,
        history_length=6,
        force_threshold=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["/World/ground"],
    )

    imu_sensor = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/b1_description/imu",
        update_period=0.0,
        debug_vis=True,
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=5000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.MirroredJointPositionActionCfg(
        asset_name="robot",
        joint_names=[  # FL -> FR -> RL -> RR  and  hip -> thigh -> calf
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ],
        use_default_offset=True,
        scale=1.0,
        preserve_order=True,  # keep on for model transfer
        debug_vis=True,
    )


@configclass
class CommandCfg:
    """Command specification"""

    # height command
    height = mdp.UniformPoseCommandAbsoluteCfg(
        asset_name="robot",  # type: ignore
        body_name="base",
        ranges=mdp.UniformPoseCommandAbsoluteCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.54, 0.54),  # ideal height is 0.54
            roll=(0.0, 0.0),
            pitch=(0, 0),
            yaw=(0, 0),
        ),
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
    )

    ############################################################
    # TODO: updated position command, (this may not be needed)

    ############################################################

    # velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     heading_command=True,  # use heading instead of angular vel
    #     rel_standing_envs=0.1,  # 10% of the time, stand still
    #     rel_heading_envs=0.7,  # 70% of the time, use heading instead of angular z
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1, 1),
    #         lin_vel_y=(-1, 1),
    #         ang_vel_z=(-math.pi / 4, math.pi / 4),  # 25 deg/s max
    #         heading=(-math.pi, math.pi),
    #     ),
    #     resampling_time_range=(5, 10),
    #     debug_vis=True,
    # )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        # relevant IMU data
        # imu_ang_vel = ObsTerm(
        #     func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}
        # )
        # imu_lin_acc = ObsTerm(
        #     func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("robot")}
        # )
        
        base_height = ObsTerm(
            func=mdp.base_pos_z, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        imu_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu_sensor")}
        )

        # command
        # velocity_cmd = ObsTerm(
        #     func=mdp.generated_commands, params={"command_name": "velocity"}
        # )

        # height command
        height_cmd = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "height"}
        )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False  # TODO: turn on later
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_all_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "position_range": (-0.015, 0.015),
            "velocity_range": (-0.0, 0.0),
        },
    )

    reset_robot_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Positional locality for feet, limit "skidding"


    # Track base height (CoM)
    base_com_height = RewTerm(
        func=mdp.base_height_from_command,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "command_name": "height",
        },
        weight=-0.3,
    )

    base_com_height_fine = RewTerm(
        func=mdp.base_height_from_command,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "command_name": "height",
            "use_tanh": True,
            "tanh_scale": 0.18,  # d/dx f(x) = -1 at around x=18.5cm if scale=0.18
        },
        weight=0.5,
    )

    # Track base velocity (CoM)
    base_lin_vel_xy = RewTerm(
        func=mdp.body_lin_vel_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
        },
        weight=-0.005,
    )

    # Combined tracking term (if desired)
    base_flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
        },
        weight=-0.15,
    )

    # base_x_y_diff = RewTerm(
    #     func=mdp.base_x_y_diff,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
    #     },
    #     weight=-1,
    # )

    # min_torque = RewTerm(
    #     func=mdp.joint_torques_l2,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"])},
    #     weight=0,
    # )

    # Center the hips
    center_hips = RewTerm(
        func=mdp.center_joints_pos,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
        weight=-0.1,
    )

    feet_contacting_ground = RewTerm(
        func=mdp.desired_contacts,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_feet")},
        weight=-0.5,
    )

    # penalize joint and action rate
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-0.0005,
    )

    action_rt = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.0005,
    )

    ############################################################

    # # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


@configclass
class CurriculumsCfg:
    """Curriculum settings for the MDP."""

    # set center_hips very high at the start, then lower it over time
    center_hips = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "center_hips",
            "w0": -0.5,
            "w1": -0.0,
            "t0": 0,
            "t1": 100000,
        },
    )

    # increase lin vel penalty over time
    base_lin_vel_xy = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "base_lin_vel_xy",
            "w0": -0.005,
            "w1": -0.05,
            "t0": 60000,
            "t1": 160000,
        },
    )

    # increase feet contacting ground penalty over time
    feet_contacting_ground = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "feet_contacting_ground",
            "w0": -0.1,
            "w1": -0.8,
            "t0": 60000,
            "t1": 280000,
        },
    )

    # increase joint position rate penalty over time
    joint_vel = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "joint_vel",
            "w0": -0.0005,
            "w1": -0.1,
            "t0": 0,
            "t1": 300000,
        },
    )
    # increase joint action rate penalty over time
    action_rt = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "action_rt",
            "w0": -0.0005,
            "w1": -0.1,
            "t0": 0,
            "t1": 300000,
        },
    )
    # incraese joint torque penalty over time
    min_torque = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "min_torque",
            "w0": 0.0,
            "w1": -2.5e-5,
            "t0": 0,
            "t1": 300000,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # # (2) Base touches the ground
    # falls_over = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "threshold": 400,
    #         "sensor_cfg": SceneEntityCfg("contact_forces_body"),
    #     },
    # )


##
# Environment configuration
##


@configclass
class B1RlLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: B1RlLocomotionSceneCfg = B1RlLocomotionSceneCfg(
        num_envs=1024, env_spacing=3.0
    )
    # Basic settings
    commands: CommandCfg = CommandCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculums: CurriculumsCfg = CurriculumsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 2.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.observations.policy.enable_corruption = True


@configclass
class B1RlLocomotionEnvCfg_PLAY(B1RlLocomotionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        """Post initialization."""
        # general settings
        self.scene.num_envs = 5
        self.scene.env_spacing = 3.0
        # disable noise
        self.observations.policy.enable_corruption = False
