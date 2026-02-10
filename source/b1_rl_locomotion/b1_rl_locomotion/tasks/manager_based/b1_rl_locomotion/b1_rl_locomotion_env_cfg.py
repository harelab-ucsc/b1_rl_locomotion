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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg

from . import mdp


##
# Pre-defined configs
##

from b1_rl_locomotion.tasks.manager_based.b1_rl_locomotion.configs.b1 import B1_CFG

##
# Constants
##

FOOT_GROUND_WEIGHT = -0.22
SLIPPING_WEIGHT = -0.005

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

    contact_forces_feet = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/b1_description/.*_foot",
        track_air_time=True,  # required for air time penalty
        update_period=0.0,
        history_length=6,
        force_threshold=0.0,
        debug_vis=True,
    )

    contact_forces_thighs = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/b1_description/.*_thigh",
        update_period=0.0,
        history_length=6,
        force_threshold=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["/World/ground"],
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
class CommandTrainCfg:
    """Command specification"""

    # height command
    height = mdp.UniformPoseCommandAbsoluteCfg(
        class_type = mdp.commands.UniformPoseCommandAbsolute,
        asset_name="robot",
        body_name="base",
        ranges=mdp.UniformPoseCommandAbsoluteCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.2, 0.7),  # 20cm to 70cm
            roll=(0.0, 0.0),
            pitch=(0, 0),
            yaw=(0, 0),
        ),
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
    )

@configclass
class CommandPlayCfg:
    """Command specification"""

    # height command
    height = mdp.SequentialHeightCommandCfg(
        class_type=mdp.commands.SequentialHeightCommand,
        name="height",          # command key used by generated_commands(...)
        asset_name="robot",        # must match env.scene asset key
        body_name="base",          # must match a rigid body name on that asset

        z_range=(0.2, 0.7),
        num_steps=22,              # linspace steps low->high inclusive
        wrap=True,                 # low->high->low->...

        resampling_time_range=(0.25, 0.25),
        random_start=True,         # de-sync parallel envs
    )


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
            "position_range": (-0.2, 0.2),
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
                "roll": (-0.1, 0.1),
                "pitch": (-.1, 0.1),
                "yaw": (0.0, 0.0),
            },
        },
    )

    apply_forces = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(3,10),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "force_range": (0,0),
            "torque_range": (0,0),  # Only apply forces, not torques
    
        }
    )

class RewardSettings:
    """Settings for curriculums."""

    # constants
    class constant:
        termination: float = -5.0
        # alive_bonus: float = 0.0

    # fixed settings (initial)
    class fixed:
        # penalty / reward for reaching height. Mostly turned off initially
        base_com_height: float = -0.6  # -0.5 -> -0.6
        base_com_height_fine: float = 0.4  # 0.3 - > 0.4

        # balancing rewards
        base_lin_vel_xy: float = -0.1
        base_flat_orientation: float = -4.0
        feet_air_time: float = -0.1  # -0.35
        feet_contacting_ground: float = -0.05  # -0.05
        soft_body_land: float = 0.4 # 0.3 -> 0.4
        soft_feet_land: float = 0.2 # 0.2 -> 0.3
        hip_centering: float = -0.5
        mirror_thighs: float = -0.1
        thigh_relaxed: float = -0.005

        # smoothness rewards
        joint_vel: float = -1e-5
        action_rt: float = -1e-3
        # soft_landing: float = 1e-3

    # curriculum 1 settings (after C1 -> C2)
    class c1:
        # increase command height rewards
        base_com_height: float = -0.6 # See adjustment notes in fixed
        base_com_height_fine: float = 0.4 # See adjustment notes in fixed

        # balancing rewards
        base_lin_vel_xy: float = -0.1
        base_flat_orientation: float = -4.0
        feet_air_time: float = -0.5
        feet_contacting_ground: float = -0.05 # -0.2 -> -0.05
        hip_centering: float = 0.0  # turn off hip centering
        thigh_relaxed: float = -0.005
        

    # longer curriculum 2 term, meant for more strict penalties
    class c2_1:
        joint_vel: float = -5e-4
        action_rt: float = -0.01

        # soft_landing: float = 0.1



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ################################################
    # Rewards for reaching target height
    ################################################

    # Track base height (CoM)
    base_com_height = RewTerm(
        func=mdp.base_height_from_command,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "command_name": "height",
        },
        weight=RewardSettings.fixed.base_com_height,
    )

    base_com_height_fine = RewTerm(
        func=mdp.base_height_from_command,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "command_name": "height",
            "use_tanh": True,
            "tanh_scale": 0.18,  # d/dx f(x) = -1 at around x=18.5cm if scale=0.18
        },
        weight=RewardSettings.fixed.base_com_height_fine,
    )

    ################################################
    # Rewards for balancing
    ################################################

    # Track base velocity (CoM)
    base_lin_vel_xy = RewTerm(
        func=mdp.body_lin_vel_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
        },
        weight=RewardSettings.fixed.base_lin_vel_xy,
    )

    # Combined tracking term (if desired)
    base_flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
        },
        weight=RewardSettings.fixed.base_flat_orientation,
    )

    # Feet should not have airtime
    feet_air_time = RewTerm(
        func=mdp.air_time_penalty,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_feet"),
        },
        weight=RewardSettings.fixed.feet_air_time,
    )

    # Feet must be in contact with the ground
    feet_contacting_ground = RewTerm(
        func=mdp.strict_desired_contacts_penalty,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_feet"),
            "threshold": 100.0,
        },  # at least 100N per foot
        weight=RewardSettings.fixed.feet_contacting_ground,
    )

    soft_body_landing = RewTerm(
        func=mdp.threshold_contact_reward,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_body"),
            "no_contact_penalty": 0,  # no penalty for no contact (other terms take care of this)
            "max_thresholds_offset": 800.0,  # 500 N above trigger is too much, starts penalizing
            "trigger_threshold": 700.0,  # minimum force to start rewarding/penalizing
        },
        weight=RewardSettings.fixed.soft_body_land,
    )

    soft_feet_landing = RewTerm(
        func=mdp.threshold_contact_reward,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_feet"),
            "no_contact_penalty": 0,  # no penalty for no contact (other terms take care of this)
            "max_thresholds_offset": 300.0,  # 300 N above trigger is too much, starts penalizing
            "trigger_threshold": 1.0,  # minimum force to start rewarding/penalizing
        },
        weight=RewardSettings.fixed.soft_feet_land,
    )

    # Center the hips
    center_hips = RewTerm(
        func=mdp.center_joints_pos,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
        weight=RewardSettings.fixed.hip_centering,
    )    

    front_mirror_thigh = RewTerm(
        func=mdp.joint_mirror_l1,
        params={"asset_cfg": SceneEntityCfg("robot",
                                            joint_names=["FR_thigh_joint", "FL_thigh_joint"])},
        weight=RewardSettings.fixed.mirror_thighs
    )
    
    rear_mirror_thigh = RewTerm(
        func=mdp.joint_mirror_l1,
        params={"asset_cfg": SceneEntityCfg("robot",
                                            joint_names=["RR_thigh_joint", "RL_thigh_joint"])},
        weight=RewardSettings.fixed.mirror_thighs
    )

    thigh_relaxed = RewTerm(
        func=mdp.thigh_relaxed_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint"])},
        weight=RewardSettings.fixed.thigh_relaxed,
    )

    ###########################################################
    # REWARDS FOR SMOOTH MOTION
    ###########################################################

    # penalize joint and action rate
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=RewardSettings.fixed.joint_vel,
    )

    action_rt = RewTerm(
        func=mdp.action_rate_l2,
        weight=RewardSettings.fixed.action_rt,  # increase later
    )

    ###########################################################
    
    # Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=RewardSettings.constant.termination)

class CurriculumSettings:
    """Settings for curriculums."""

    ###########################################################
    # C1 -> C2: Keep balancing rewards roughly the same,
    # but make laying down rewards more important.
    #
    # Activates: 500 steps
    # Activation Duration: 1.5k steps
    ###########################################################

    class c1:
        activation_step: int = 1500
        end_step: int = 8000

    class c2_1(c1):
        activation_step: int = 0
        end_step: int = 15000

    class forces_mild:
        activation_step: int = 8000

    class forces_natural:
        activation_step: int = 20000

# define a top level (hydra necessary) function to check time step for modification condition
def override_value(env,
                   env_ids,
                   data,
                   value,
                   num_steps):
    # if env.common_step_counter % 500 == 0:
        # print("[OVERRIDE DEBUG] step =", env.common_step_counter, "data =", data)
    if env.common_step_counter > num_steps:
        # print(">>> curriculum triggered at step =", env.common_step_counter)
        return value
    return mdp.modify_term_cfg.NO_CHANGE

@configclass
class CurriculumsCfg:

    #"""Curriculum settings for the MDP. Currently in testing"""
    #apply_forces_1 = CurrTerm(
    #    func=mdp.modify_term_cfg,
    #    params={
    #        "address": "events.apply_forces.params",   # note: `_manager.cfg` is omitted
    #        "modify_fn": override_value,
    #        "modify_params": {"value": {"asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
    #                                    "force_range": (0, 30),
    #                                    "torque_range": (0,0) # Only apply forces, not torques
    #                                    },  
    #                        "num_steps": CurriculumSettings.forces_mild.activation_step,
    #        }
    #    }                      
    #)

    #apply_forces_2 = CurrTerm(
    #    func=mdp.modify_term_cfg,
    #    params={
    #        "address": "events.apply_forces.params",   # note: `_manager.cfg` is omitted
    #        "modify_fn": override_value,
    #        "modify_params": {"value": {"asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
    #                                    "force_range": (20, 50),
    #                                    "torque_range": (0,0) # Only apply forces, not torques
    #                                    },  
    #                        "num_steps": CurriculumSettings.forces_natural.activation_step,
    #        }
    #    }                      
    #)

    ####################################################
    # TODO: Test force curriculums using above functions
    ####################################################


    ####################################################
    # Curriculum 1
    ####################################################
    base_com_height_c1 = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "base_com_height",
            "w0": RewardSettings.fixed.base_com_height,
            "w1": RewardSettings.c1.base_com_height,
            "t0": CurriculumSettings.c1.activation_step,
            "t1": CurriculumSettings.c1.end_step,
        }
    )

    base_com_height_fine_c1 = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "base_com_height_fine",
            "w0": RewardSettings.fixed.base_com_height_fine,
            "w1": RewardSettings.c1.base_com_height_fine,
            "t0": CurriculumSettings.c1.activation_step,
            "t1": CurriculumSettings.c1.end_step,
        }
    )

    center_hips_c1 = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "center_hips",
            "w0": RewardSettings.fixed.hip_centering,
            "w1": RewardSettings.c1.hip_centering,
            "t0": CurriculumSettings.c1.activation_step,
            "t1": CurriculumSettings.c1.end_step,
        }
    )

    base_lin_vel_xy_c1 = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "base_lin_vel_xy",
            "w0": RewardSettings.fixed.base_lin_vel_xy,
            "w1": RewardSettings.c1.base_lin_vel_xy,
            "t0": CurriculumSettings.c1.activation_step,
            "t1": CurriculumSettings.c1.end_step,
        }
    )

    feet_air_time_c1 = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "feet_air_time",
            "w0": RewardSettings.fixed.feet_air_time,
            "w1": RewardSettings.c1.feet_air_time,
            "t0": CurriculumSettings.c1.activation_step,
            "t1": CurriculumSettings.c1.end_step,
        }
    )

    feet_contacting_ground_c1 = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "feet_contacting_ground",
            "w0": RewardSettings.fixed.feet_contacting_ground,
            "w1": RewardSettings.c1.feet_contacting_ground,
            "t0": CurriculumSettings.c1.activation_step,
            "t1": CurriculumSettings.c1.end_step,
        }
    )

    # increase joint position rate penalty over time
    joint_vel_c1 = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "joint_vel",
            "w0": RewardSettings.fixed.joint_vel,
            "w1": RewardSettings.c2_1.joint_vel,
            "t0": CurriculumSettings.c2_1.activation_step,
            "t1": CurriculumSettings.c2_1.end_step,
        },
    )
    # increase joint action rate penalty over time
    action_rt_c1 = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "action_rt",
            "w0": RewardSettings.fixed.action_rt,
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

    # (2) Base touches the ground
    falls_over = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "threshold": 0.0,
            "sensor_cfg": SceneEntityCfg("contact_forces_body"),
        },
    )


##
# Environment configuration
##


@configclass
class B1RlLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: B1RlLocomotionSceneCfg = B1RlLocomotionSceneCfg(
        num_envs=4096, env_spacing=3.0
    )
    # Basic settings
    commands: CommandTrainCfg = CommandTrainCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumsCfg = CurriculumsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        super().__post_init__()
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
    commands: CommandPlayCfg = CommandPlayCfg()
    def __post_init__(self) -> None:
        super().__post_init__()

        """Post initialization."""
        # general settings
        self.scene.num_envs = 5
        self.scene.env_spacing = 3.0
        # disable noise
        self.observations.policy.enable_corruption = False

