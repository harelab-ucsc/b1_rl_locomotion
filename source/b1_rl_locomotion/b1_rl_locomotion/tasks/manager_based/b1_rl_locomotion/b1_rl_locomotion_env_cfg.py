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
            noise=AdditiveUniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=AdditiveUniformNoiseCfg(n_min=-0.05, n_max=0.05),
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
            "position_range": (-0.15, 0.15),
            "velocity_range": (-0.1, 0.1),
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
                "z": (-0.05, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (0.0, 0.0),
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
            "position_range": (0.0, 0.0),
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


class RewardSettings:
    """Settings for curriculums."""

    # constants
    class constant:
        termination: float = -50.0
        # alive_bonus: float = 0.0

    # curriculum 1 settings (initial)
    class c1:
        # penalty / reward for laying down. Mostly turned off initially
        joint_error: float = -5e-4
        joint_error_fine: float = 5e-4
        base_height: float = -5e-4
        base_height_fine: float = 5e-4
        base_lin_vel_z: float = -1e-2

        # balancing rewards
        base_lin_vel_xy: float = -0.1
        base_flat_orientation: float = -2.0
        feet_air_time: float = -0.35
        feet_contacting_ground: float = -0.05
        # hip_centering: float = -0.1

        # smoothness rewards
        joint_vel: float = -1e-7
        action_rt: float = -1e-6
        # soft_landing: float = 1e-3

    # curriculum 2 settings (after C1 -> C2)
    class c2:
        # increase joint error and laying down rewards
        joint_error: float = -0.3
        joint_error_fine: float = 0.5
        base_height: float = -0.3
        base_height_fine: float = 0.5
        base_lin_vel_z: float = -0.15

        # balancing rewards
        base_lin_vel_xy: float = -0.5
        base_flat_orientation: float = -3.0
        feet_air_time: float = -2.0
        feet_contacting_ground: float = -0.7

        # hip_centering: float = 0.0  # turn off hip centering

    # longer curriculum 2 term, meant for more strict penalties
    class c2_1:
        joint_vel: float = -5e-4
        action_rt: float = -0.12

        # soft_landing: float = 0.1


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # definition of reward terms (order preserved)

    ###########################################################
    # REWARDS FOR SITTING DOWN
    ###########################################################

    # joint error to a target joint state (fixed)
    joint_error = RewTerm(  # normal L2 squared penalty
        func=mdp.joint_pos_target_error_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "target": RewardConstants.sitting_joint_pos,
        },
        weight=RewardSettings.c1.joint_error,
    )
    joint_error_fine = RewTerm(  # fine grained tanh reward
        func=mdp.joint_pos_target_error_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "target": RewardConstants.sitting_joint_pos,
            "use_tanh": True,
        },
        weight=RewardSettings.c1.joint_error_fine,
    )

    # base height tracking
    base_height = RewTerm(
        func=mdp.base_height_from_command,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "command_name": "height",
        },
        weight=RewardSettings.c1.base_height,
    )
    # base height fine tracking
    base_height_fine = RewTerm(
        func=mdp.base_height_from_command,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "command_name": "height",
            "use_tanh": True,
        },
        weight=RewardSettings.c1.base_height_fine,
    )

    # minimize base linear velocity in z direction
    base_lin_vel_z = RewTerm(
        func=mdp.body_lin_vel_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "x": False,
            "y": False,
            "z": True,
        },
        weight=RewardSettings.c1.base_lin_vel_z,
    )

    ###########################################################
    # REWARDS FOR BALANCING
    ###########################################################

    # Track base linear velocity in xy plane
    base_lin_vel_xy = RewTerm(
        func=mdp.body_lin_vel_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "z": False,
        },
        weight=RewardSettings.c1.base_lin_vel_xy,
    )

    # Track base orientation (CoM)
    base_flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
        },
        weight=RewardSettings.c1.base_flat_orientation,
    )

    # feet shouldn't have any air time
    feet_air_time = RewTerm(
        func=mdp.air_time_penalty,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_feet"),
        },
        weight=RewardSettings.c1.feet_air_time,
    )

    # Feet must be in contact with the ground
    feet_contacting_ground = RewTerm(
        func=mdp.strict_desired_contacts_penalty,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_feet"),
            "threshold": 100.0,
        },  # at least 100N per foot
        weight=RewardSettings.c1.feet_contacting_ground,
    )

    # # Slight penalty for hip centering (so it doesn't look like shit)
    # hip_centering = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
    #     },
    #     weight=RewardSettings.c1.hip_centering,
    # )

    # # minimize feet contact forces at all times
    # soft_landing = RewTerm(
    #     func=mdp.threshold_contact_reward,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces_feet"),
    #         "no_contact_penalty": 0,  # no penalty for no contact (other terms take care of this)
    #         "max_thresholds_offset": 150.0,  # 150 N above trigger is too much, starts penalizing
    #         "trigger_threshold": 120.0,  # minimum force to start rewarding/penalizing
    #     },
    #     weight=RewardSettings.c1.soft_landing,
    # )

    ###########################################################
    # REWARDS FOR SMOOTH MOTION
    ###########################################################

    # penalize joint and action rate
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=RewardSettings.c1.joint_vel,
    )

    action_rt = RewTerm(
        func=mdp.action_rate_l2,
        weight=RewardSettings.c1.action_rt,  # increase later
    )

    ###########################################################

    # # Alive bonus
    # alive_bonus = RewTerm(
    #     func=mdp.is_alive,
    #     weight=RewardSettings.constant.alive_bonus,
    # )

    # Failure penalty
    terminating = RewTerm(
        func=mdp.is_terminated, weight=RewardSettings.constant.termination
    )


class CurriculumSettings:
    """Settings for curriculums."""

    ###########################################################
    # C1 -> C2: Keep balancing rewards roughly the same,
    # but make laying down rewards more important.
    #
    # Activates: 500 steps
    # Activation Duration: 1.5k steps
    ###########################################################

    class c2:
        activation_step: int = 500
        end_step: int = 1500

    class c2_1(c2):
        activation_step: int = 1000
        end_step: int = 2500


@configclass
class CurriculumCfg:
    """Curriculum settings for the MDP."""

    ###########################################################
    # C1 -> C2 settings
    ###########################################################

    joint_error = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "joint_error",
            "w0": RewardSettings.c1.joint_error,
            "w1": RewardSettings.c2.joint_error,
            "t0": CurriculumSettings.c2.activation_step,
            "t1": CurriculumSettings.c2.end_step,
        },
    )
    joint_error_fine = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "joint_error_fine",
            "w0": RewardSettings.c1.joint_error_fine,
            "w1": RewardSettings.c2.joint_error_fine,
            "t0": CurriculumSettings.c2.activation_step,
            "t1": CurriculumSettings.c2.end_step,
        },
    )
    base_height = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "base_height",
            "w0": RewardSettings.c1.base_height,
            "w1": RewardSettings.c2.base_height,
            "t0": CurriculumSettings.c2.activation_step,
            "t1": CurriculumSettings.c2.end_step,
        },
    )
    base_height_fine = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "base_height_fine",
            "w0": RewardSettings.c1.base_height_fine,
            "w1": RewardSettings.c2.base_height_fine,
            "t0": CurriculumSettings.c2.activation_step,
            "t1": CurriculumSettings.c2.end_step,
        },
    )
    base_lin_vel_z = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "base_lin_vel_z",
            "w0": RewardSettings.c1.base_lin_vel_z,
            "w1": RewardSettings.c2.base_lin_vel_z,
            "t0": CurriculumSettings.c2.activation_step,
            "t1": CurriculumSettings.c2.end_step,
        },
    )

    base_lin_vel_xy = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "base_lin_vel_xy",
            "w0": RewardSettings.c1.base_lin_vel_xy,
            "w1": RewardSettings.c2.base_lin_vel_xy,
            "t0": CurriculumSettings.c2.activation_step,
            "t1": CurriculumSettings.c2.end_step,
        },
    )
    base_flat_orientation = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "base_flat_orientation",
            "w0": RewardSettings.c1.base_flat_orientation,
            "w1": RewardSettings.c2.base_flat_orientation,
            "t0": CurriculumSettings.c2.activation_step,
            "t1": CurriculumSettings.c2.end_step,
        },
    )
    feet_air_time = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "feet_air_time",
            "w0": RewardSettings.c1.feet_air_time,
            "w1": RewardSettings.c2.feet_air_time,
            "t0": CurriculumSettings.c2.activation_step,
            "t1": CurriculumSettings.c2.end_step,
        },
    )
    feet_contacting_ground = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "feet_contacting_ground",
            "w0": RewardSettings.c1.feet_contacting_ground,
            "w1": RewardSettings.c2.feet_contacting_ground,
            "t0": CurriculumSettings.c2.activation_step,
            "t1": CurriculumSettings.c2.end_step,
        },
    )
    # hip_centering = CurrTerm(
    #     func=mdp.lerp_reward_weight,
    #     params={
    #         "term_name": "hip_centering",
    #         "w0": RewardSettings.c1.hip_centering,
    #         "w1": RewardSettings.c2.hip_centering,
    #         "t0": CurriculumSettings.c2.activation_step,
    #         "t1": CurriculumSettings.c2.end_step,
    #     },
    # )
    # soft_landing = CurrTerm(
    #     func=mdp.lerp_reward_weight,
    #     params={
    #         "term_name": "soft_landing",
    #         "w0": RewardSettings.c1.soft_landing,
    #         "w1": RewardSettings.c2_1.soft_landing,
    #         "t0": CurriculumSettings.c2_1.activation_step,
    #         "t1": CurriculumSettings.c2_1.end_step,
    #     },
    # )

    joint_vel = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "joint_vel",
            "w0": RewardSettings.c1.joint_vel,
            "w1": RewardSettings.c2_1.joint_vel,
            "t0": CurriculumSettings.c2_1.activation_step,
            "t1": CurriculumSettings.c2_1.end_step,
        },
    )
    action_rt = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "action_rt",
            "w0": RewardSettings.c1.action_rt,
            "w1": RewardSettings.c2_1.action_rt,
            "t0": CurriculumSettings.c2_1.activation_step,
            "t1": CurriculumSettings.c2_1.end_step,
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
            "threshold": 800,
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
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 3
        # viewer settings
        self.viewer.eye = (4.0, 0.0, 1.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
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
        # disable noise
        self.observations.policy.enable_corruption = False
