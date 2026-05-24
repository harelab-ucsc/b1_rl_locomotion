
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
class B1StandSceneCfg(InteractiveSceneCfg):
    """Configuration for sitting to laying scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(500.0, 500.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robot — default joint positions stay as standing (for joint_pos_rel reference);
    # the reset event moves the robot into the laying-down pose each episode.
    robot: ArticulationCfg = B1_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    contact_forces_body = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/b1_description/base",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        # filter_prim_paths_expr=["/World/ground"],
    )

    contact_forces_thighs = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/b1_description/.*_thigh",
        update_period=0.0,
        history_length=6,
        force_threshold=0.0,
        debug_vis=False,
        # filter_prim_paths_expr=["/World/ground"],
    )

    contact_forces_feet = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/b1_description/.*_foot",
        track_air_time=True,  # required for air time penalty
        update_period=0.0,
        history_length=6,
        force_threshold=0.0,
        debug_vis=False,
    )

    imu_sensor = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/b1_description/base",
        history_length=6,
        update_period=0.0,
        debug_vis=False,
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
class CommandCfg:
    """Command specification"""

    # height command
    height = mdp.UniformPoseCommandAbsoluteCfg(
        asset_name="robot",  # type: ignore
        body_name="base",
        ranges=mdp.UniformPoseCommandAbsoluteCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.565, 0.565),  # ideal height is 0.54
            roll=(0.0, 0.0),
            pitch=(0, 0),
            yaw=(0, 0),
        ),
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
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
            # noise=AdditiveUniformNoiseCfg(n_min=-0.05, n_max=0.05),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
                    ],
                    preserve_order=True,  # keep on for model transfer
                )
            },
        )


        # relevant IMU data
        # imu_orientation = ObsTerm(
        #     func=mdp.imu_orientation, params={"asset_cfg": SceneEntityCfg("imu_sensor")}
        # )
        # imu_lin_acc = ObsTerm(
        #     func=mdp.imu_lin_acc,
        #     params={"asset_cfg": SceneEntityCfg("imu_sensor")}
        # )
        # imu_ang_vel = ObsTerm(
        #     func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu_sensor")}
        # )

        # base_height = ObsTerm(
        #     func=mdp.base_pos_z, params={"asset_cfg": SceneEntityCfg("robot")}
        # )

        # command
        # velocity_cmd = ObsTerm(
        #     func=mdp.generated_commands, params={"command_name": "velocity"}
        # )

        # # height command
        # height_cmd = ObsTerm(
        #     func=mdp.generated_commands, params={"command_name": "height"}
        # )

        # last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset joints to laying-down pose; default_joint_pos stays as standing so
    # joint_pos_rel is still measured relative to the standing reference.
    reset_all_joints = EventTerm(
        func=mdp.reset_joints_to_pose,
        mode="reset",
        params={
            "joint_pos_dict": {
                "FR_hip_joint": -0.558384,
                "FR_thigh_joint":  1.078270,
                "FR_calf_joint":  -2.751709,
                "FL_hip_joint":  0.534135,
                "FL_thigh_joint":  1.089023,
                "FL_calf_joint":  -2.739185,
                "RR_hip_joint": -0.584137,
                "RR_thigh_joint":  1.067164,
                "RR_calf_joint":  -2.622180,
                "RL_hip_joint":  0.544445,
                "RL_thigh_joint":  1.051190,
                "RL_calf_joint":  -2.626244,
            },
            "position_noise_range": (-0.015, 0.015),
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
                "z": (-0.5, -0.5),
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

    randomize_joint_friction = EventTerm(
        func=mdp.scale_joint_friction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["*_joint"]),
            "scale_range": (0.9, 1.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Track base height (CoM)
    base_com_height = RewTerm(
        func=mdp.base_height_from_command,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "command_name": "height",
        },
        weight=-0.15,
    )

    base_com_height_fine = RewTerm(
        func=mdp.base_height_from_command,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "command_name": "height",
            "use_tanh": True,
            "tanh_scale": 0.05,
        },
        weight=0.15,
    )

    # Track base velocity (CoM)
    base_lin_vel_xy = RewTerm(
        func=mdp.body_lin_vel_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
        },
        weight=-0.1,
    )

    base_flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
        },
        weight=-2.0,
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
        weight=-0.5,
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

    joint_error = RewTerm(
        func=mdp.joint_pos_target_error_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "target": {
                k: v * (np.pi / 180.0)  # convert to rad, the values below are in degrees
                for k, v in {
                    "[F,R]R_hip_joint": -1.5,
                    "[F,R]L_hip_joint": 1.5,
                    ".*_thigh_joint": 42.0,
                    ".*_calf_joint": -79.0,
                }.items()
            },
        },
        weight=-0.15,
    )

    joint_error_fine = RewTerm(
        func=mdp.joint_pos_target_error_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "target": {
                k: v * (np.pi / 180.0)  # convert to rad, the values below are in degrees
                for k, v in {
                    "[F,R]R_hip_joint": -1.5,
                    "[F,R]L_hip_joint": 1.5,
                    ".*_thigh_joint": 42.0,
                    ".*_calf_joint": -79.0,
                }.items()
            },
            "use_tanh": True,
        },
        weight=0.15,
    )

    # Feet must be in contact with the ground
    feet_contacting_ground = RewTerm(
        func=mdp.strict_desired_contacts_penalty,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_feet"),
            "threshold": 100.0,
        },  # at least 100N per foot
        weight=-0.1,
    )

    # penalize joint and action rate
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-0.00001,
    )

    action_rt = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.0015,
    )

    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)


@configclass
class CurriculumCfg:
    """Curriculum settings for the MDP."""


    # turn on joint_pos_rel observation noise after a warmup phase, so the policy
    # first learns on clean obs and then adapts to sensor noise
    joint_pos_rel_noise = CurrTerm(
        func=mdp.set_obs_term_noise,
        params={
            "group_name": "policy",
            "term_name": "joint_pos_rel",
            "noise_cfg": AdditiveUniformNoiseCfg(
                n_min=-0.01,
                n_max=0.01,
            ),
            "activation_step": 10000,
        },
    )

    feet_contacting_ground = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "feet_contacting_ground",
            "w0": -0.1,
            "w1": -0.5,
            "t0": 3000,
            "t1": 7000,
        },
    )

    action_rt = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "action_rt",
            "w0": -0.0015,
            "w1": -0.02,
            "t0": 15000,
            "t1": 20000,
        },
    )

    joint_vel = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "joint_vel",
            "w0": -0.00001,
            "w1": -0.001,
            "t0": 20000,
            "t1": 27000,
        },
    )

    base_lin_vel_z = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "base_lin_vel_z",
            "w0": -0.5,
            "w1": -1.0,
            "t0": 40000,
            "t1": 45000,
        },
    )

    feet_contacting_ground2 = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "feet_contacting_ground",
            "w0": -0.5,
            "w1": -1.5,
            "t0": 50000,
            "t1": 52000,
        },
    )


    joint_vel2 = CurrTerm(
        func=mdp.lerp_reward_weight,
        params={
            "term_name": "joint_vel",
            "w0": -0.001,
            "w1": -0.01,
            "t0": 54000,
            "t1": 55000,
        },
    )




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Base touches the ground
    # falls_over = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "threshold": 2000,
    #         "sensor_cfg": SceneEntityCfg("contact_forces_body"),
    #     },
    # )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "limit_angle": math.radians(60.0),
        },
    )


@configclass
class TerminationsCfg_PLAY:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Base touches the ground
    # falls_over = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "threshold": 2200,
    #         "sensor_cfg": SceneEntityCfg("contact_forces_body"),
    #     },
    # )


##
# Environment configuration
##
@configclass
class B1StandEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: B1StandSceneCfg = B1StandSceneCfg(num_envs=1024, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandCfg = CommandCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Number of physics decimation steps to run after each reset, with the
    # action manager holding default PD targets (no agent action). Lets the
    # robot settle from its randomized reset state before the agent takes over.
    num_reset_settle_steps: int = 10

    # Hold the sitting-down pose during settle steps (matches the reset pose set by
    # reset_all_joints), so PD doesn't fight gravity by targeting the standing default.
    settle_joint_pos: dict[str, float] = {
        "FR_hip_joint":   -0.558384,
        "FR_thigh_joint":  1.078270,
        "FR_calf_joint":  -2.751709,
        "FL_hip_joint":    0.534135,
        "FL_thigh_joint":  1.089023,
        "FL_calf_joint":  -2.739185,
        "RR_hip_joint":   -0.584137,
        "RR_thigh_joint":  1.067164,
        "RR_calf_joint":  -2.622180,
        "RL_hip_joint":    0.544445,
        "RL_thigh_joint":  1.051190,
        "RL_calf_joint":  -2.626244,
    }

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (4.0, 0.0, 1.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.observations.policy.enable_corruption = True

        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.min_velocity_iteration_count = 1

        # compensate episode length for settle steps (settle transitions are
        # masked out of training, so without this the agent loses effective
        # episode length)
        self.episode_length_s += self.num_reset_settle_steps * self.decimation * self.sim.dt


@configclass
class B1StandEnvCfg_PLAY(B1StandEnvCfg):
    terminations: TerminationsCfg_PLAY = TerminationsCfg_PLAY()

    num_reset_settle_steps = 0

    def __post_init__(self) -> None:
        super().__post_init__()

        """Post initialization."""
        self.viewer.origin_type = "world"
        self.viewer.env_index = 0

        self.episode_length_s = 5
        self.episode_length_s += self.num_reset_settle_steps * self.decimation * self.sim.dt

        # general settings
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
