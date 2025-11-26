# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np

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
from isaaclab.sensors import CameraCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg

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
        track_air_time=True,  # required for air time penalty
        update_period=0.0,
        history_length=6,
        force_threshold=0.0,
        debug_vis=True,
    )

    imu_sensor = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/b1_description/base",
        history_length=6,
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

    joint_effort = mdp.JointPositionActionCfg(
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
            pos_z=(0.0, 0.0),  # as low as possible
            roll=(0.0, 0.0),
            pitch=(0, 0),
            yaw=(0, 0),
        ),
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=AdditiveUniformNoiseCfg(n_min=-0.1, n_max=0.1),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=AdditiveUniformNoiseCfg(n_min=-0.1, n_max=0.1),
        )

        # relevant IMU data
        imu_orientation = ObsTerm(
            func=mdp.imu_orientation, params={"asset_cfg": SceneEntityCfg("imu_sensor")}
        )
        imu_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu_sensor")}
        )
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu_sensor")}
        )

        # command
        # velocity_cmd = ObsTerm(
        #     func=mdp.generated_commands, params={"command_name": "velocity"}
        # )

        def __post_init__(self) -> None:
            self.enable_corruption = True
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
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_robot_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.75),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )


@configclass
class EventCfg_PLAY(EventCfg):
    """Configuration for events."""

    # reset
    reset_all_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "position_range": (-0.25, 0.25),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_robot_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
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


class RewardConstants:
    """Constants for rewards."""

    sitting_joint_pos = {
        k: v * (np.pi / 180.0)  # convert to rad, the values below are in degrees
        for k, v in {
            "[F,R]R_hip_joint": -32.0,
            "[F,R]L_hip_joint": 32.0,
            ".*_thigh_joint": 57.0,
            ".*_calf_joint": -149.0,
        }.items()
    }


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # definition of reward terms (order preserved)

    # joint error to a target joint state (fixed)
    joint_error = RewTerm(  # normal L2 squared penalty
        func=mdp.joint_pos_target_error_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "target": RewardConstants.sitting_joint_pos,
        },
        weight=-0.01,
    )
    joint_error_fine = RewTerm(  # fine grained tanh reward
        func=mdp.joint_pos_target_error_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "target": RewardConstants.sitting_joint_pos,
            "use_tanh": True,
        },
        weight=0.01,
    )

    # base height tracking
    base_height = RewTerm(
        func=mdp.base_height_from_command,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "command_name": "height",
        },
        weight=-0.1,
    )
    # base height fine tracking
    base_height_fine = RewTerm(
        func=mdp.base_height_from_command,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "command_name": "height",
            "use_tanh": True,
        },
        weight=0.1,
    )

    # minimize base linear velocity in all directions
    base_lin_vel_xy = RewTerm(
        func=mdp.body_lin_vel_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
        },
        weight=-0.1,
    )

    # Track base orientation (CoM)
    base_flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
        },
        weight=-2.5,
    )

    # Minimize joint torques
    # min_torque = RewTerm(
    #     func=mdp.joint_torques_l2,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"])},
    #     weight=-5e-7,
    # )

    # feet shouldn't have any air time
    feet_air_time = RewTerm(
        func=mdp.air_time_penalty,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_feet"),
        },
        weight=-0.2,
    )

    # Feet must be in contact with the ground
    feet_contacting_ground = RewTerm(
        func=mdp.strict_desired_contacts_penalty,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_feet"),
            "threshold": 50.0,
        },  # at least 50N per foot
        weight=-0.2,  # penalize if any foot is not contacting ground
    )

    # minimize feet contact forces at all times
    sparse_feet_contact_forces = RewTerm(
        func=mdp.threshold_contact_reward,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_feet"),
            "no_contact_penalty": 0,  # no penalty for no contact (other terms take care of this)
            "max_thresholds_offset": 300.0,  # 300 N above trigger is too much, starts penalizing
            "trigger_threshold": 100.0,  # minimum force to start rewarding/penalizing
        },
        weight=-0.5,
    )

    # feet_not_slipping = RewTerm(
    #     func=mdp.foot_slip_penalty,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces_feet"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot"]),
    #         "contact_threshold": 1.0,  # any contact + movement = slip
    #         "slip_threshold": 0.5,
    #     },
    #     weight=-0.3,
    # )

    # # reward gentle body contact with the ground, but not too much
    # body_threshold_contact = RewTerm(
    #     func=mdp.threshold_contact_reward,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces_body"),
    #         "no_contact_penalty": 0.5,  # penalty 0.5 for no contact
    #         "max_thresholds_offset": 100.0,  # 100 N above trigger is too much, starts penalizing
    #         "trigger_threshold": 350.0,  # minimum force to start rewarding/penalizing
    #     },
    #     weight=0.05,
    # )

    # penalize joint and action rate
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-5e-4,
    )

    action_rt = RewTerm(
        func=mdp.action_rate_l2,
        weight=-5e-4,
    )

    ############################################################

    # # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


@configclass
class CurriculumsCfg:
    """Curriculum settings for the MDP."""

    ###########################################################
    # STEP 1: Make the robot learn to balance and have feet contact the ground
    ###########################################################

    # # increase lin vel penalty over time
    base_lin_vel_xy = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "base_lin_vel_xy",
            "w0": -0.2,
            "w1": -0.35,
            "t0": 0,
            "t1": 8000,
        },
    )

    # increase feet contacting ground penalty over time
    feet_contacting_ground = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "feet_contacting_ground",
            "w0": -0.3,
            "w1": -0.5,
            "t0": 1500,
            "t1": 15000,
        },
    )

    # increase air time penalty over time
    feet_air_time = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "feet_air_time",
            "w0": -0.5,
            "w1": -1.0,
            "t0": 1500,
            "t1": 15000,
        },
    )

    # # increase body contact threshold at the end to make the body touch the ground
    # body_threshold_contact = CurrTerm(
    #     func=mdp.lerp_reward_weight,
    #     params={
    #         "term_name": "body_threshold_contact",
    #         "w0": 0,
    #         "w1": 0.5,
    #         "t0": 20000,
    #         "t1": 25000,
    #     },
    # )

    # # increase feet slip penalty over time
    # feet_not_slipping = CurrTerm(
    #     func=mdp.lerp_reward_weight,
    #     params={
    #         "term_name": "feet_not_slipping",
    #         "w0": -0.05,
    #         "w1": -0.4,
    #         "t0": 5000,
    #         "t1": 15000,
    #     },
    # )

    # penalize joint torque more over time
    # min_torque = CurrTerm(
    #     func=mdp.lerp_reward_weight,
    #     params={
    #         "term_name": "min_torque",
    #         "w0": -5e-9,
    #         "w1": -5e-7,
    #         "t0": 5000,
    #         "t1": 10000,
    #     },
    # )

    # increase joint position rate penalty over time
    joint_vel = CurrTerm(  # part 1
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "joint_vel",
            "w0": -5e-4,
            "w1": -1e-2,
            "t0": 0,
            "t1": 10000,
        },
    )
    joint_vel = CurrTerm(  # part 2, after the robot has learned to balance
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "joint_vel",
            "w0": -1e-2,
            "w1": -0.05,
            "t0": 10001,
            "t1": 30000,
        },
    )
    # increase joint action rate penalty over time
    action_rt = CurrTerm(  # part 1
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "action_rt",
            "w0": -0.1,
            "w1": -0.5,
            "t0": 0,
            "t1": 15000,
        },
    )
    action_rt = CurrTerm(  # part 2, after the robot has learned to balance
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "action_rt",
            "w0": -0.5,
            "w1": -1.5,
            "t0": 15001,
            "t1": 25000,
        },
    )

    # PART 2: Make the robot learn to lay down and reach target height + desired pose
    ###########################################################
    joint_error = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "joint_error",
            "w0": -0.01,
            "w1": -0.5,
            "t0": 10000,
            "t1": 15000,
        },
    )
    joint_error_fine = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "joint_error_fine",
            "w0": 0.01,
            "w1": 1.5,
            "t0": 10000,
            "t1": 15000,
        },
    )
    base_height = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "base_height",
            "w0": -0.1,
            "w1": -0.3,
            "t0": 0,
            "t1": 10000,
        },
    )
    base_height_fine = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "base_height_fine",
            "w0": 0.01,
            "w1": 1.0,
            "t0": 0,
            "t1": 10000,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # # (2) Base touches the ground
    falls_over = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "threshold": 700,
            "sensor_cfg": SceneEntityCfg("contact_forces_body"),
        },
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "limit_angle": math.radians(120.0),
        },
    )


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
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandCfg = CommandCfg()
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
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 2.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.observations.policy.enable_corruption = True


@configclass
class B1RlLocomotionEnvCfg_PLAY(B1RlLocomotionEnvCfg):
    events: EventCfg = EventCfg_PLAY()

    def __post_init__(self) -> None:
        super().__post_init__()

        """Post initialization."""
        # general settings
        self.scene.num_envs = 5
        self.scene.env_spacing = 3.0
        # disable noise
        self.observations.policy.enable_corruption = False
