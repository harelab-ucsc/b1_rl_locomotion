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
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from . import mdp

##
# Pre-defined configs
##

from b1_rl_locomotion.tasks.manager_based.b1_rl_locomotion.configs.b1 import B1_CFG 
from isaaclab.terrains import TerrainImporterCfg   # (or the exact IsaacLab module path)
from b1_rl_locomotion.tasks.manager_based.b1_rl_locomotion.terrains.config.rough import ROUGH_TERRAINS_CFG
##
# Scene definition
##


@configclass
class B1RlLocomotionSceneCfg(InteractiveSceneCfg):

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robot
    robot: ArticulationCfg = B1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/b1_description/.*",
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=None,
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
        debug_vis=False,
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class CommandCfg:
    """Command specification"""

    velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,  # use heading instead of angular vel
        rel_standing_envs=0.2,  # 20% of the time, stand still
        rel_heading_envs=1.0,  # 100% of the time, use heading instead of angular z
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1, 1),
            lin_vel_y=(-1, 1),
            ang_vel_z=(-math.pi / 4, math.pi / 4),  # 25 deg/s max
            heading=(-math.pi, math.pi),
        ),
        resampling_time_range=(10, 10),
        debug_vis=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # add noise later: noise=Unoise(n_min=-0.1, n_max=0.1)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)  # add noise later: noise=Unoise(n_min=-0.2, n_max=0.2)

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        # command
        velocity_cmd = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "velocity"}
        )
        actions = ObsTerm(func=mdp.last_action)
        gravity = ObsTerm(func=mdp.projected_gravity)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # TODO: startup events to randomize material, mass, and CoM
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    reset_all_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.25, 0.25),
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

    # -- Rewards

    # Command Tracking
    lin_vel_tracking = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1, params={"std": math.sqrt(0.25), "command_name": "velocity"})
    angle_vel_tracking = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.5, params={"std": math.sqrt(0.25), "command_name": "velocity"})

    # -- Penalties

    # Minismize up-down movement
    vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.35)

    # angular velocity xy (i.e. rotating sideways)
    angle_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)  # -0.05

    # Action rate
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.025)

    # thigh contact
    thigh_contact = RewTerm(func=mdp.undesired_contacts, weight=-1, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh")})

    # Flat Orientation
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-8)

    # Soft Joint Limits (prevent cross legs)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    # torques
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-6)

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # Feet Air time (pos weight but negative reward due to )
    feet_air_time = RewTerm(func=mdp.feet_air_time, weight=8.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "command_name": "velocity", "threshold": 0.125})

    # Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-45)

    # Base Height, works without sensors for FLAT TERRAIN ONLY
    base_height = RewTerm(func=mdp.base_height_l2, weight=-5, params={"target_height": 0.575})




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Base touches the ground
    falls_over = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
        },
    )


##
# Environment configuration
##


@configclass
class B1RlLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: B1RlLocomotionSceneCfg = B1RlLocomotionSceneCfg(
        num_envs=512, env_spacing=3.0
        # num_envs=4096, env_spacing=2.5
    )
    # Basic settings
    commands: CommandCfg = CommandCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4  # 4 sim steps per control step
        self.episode_length_s = 20.0    # 10s previously
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 2.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


@configclass
class B1RlLocomotionEnvCfg_PLAY(B1RlLocomotionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        """Post initialization."""
        # general settings
        self.scene.num_envs = 10
        self.scene.env_spacing = 3.0
        # disable noise
        self.observations.policy.enable_corruption = False
