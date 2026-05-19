"""Configuration for the B1 dog.

The following configurations are available:

* :obj:`UNITREE_B1_CFG`: Unitree B1 robot with DC motor model for the legs. Adapted from the A1 CFG.
TODO: match the DC motor actuator to the real specified values from Unitree

Reference:
    - https://github.com/unitreerobotics/unitree_ros
    - /pvc/isaac-sim/IsaacLab/scripts/lemon/main.py
"""

import isaaclab.sim as sim_utils

from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, IdealPDActuatorCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
import numpy as np

# Path to current config's directory (/home/.../b1_rl_locomotion/config)
BASE_DIR = os.path.dirname(__file__)
ASSET_DIR = os.path.join(BASE_DIR, "..", "assets")  # /home/.../b1_rl_locomotion/assets
B1_USD = os.path.join(ASSET_DIR, "b1.usd")  # /home/.../b1_rl_locomotion/assets/b1.usd

##
# Configuration
##

B1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(B1_USD),
        activate_contact_sensors=True,  # ADAPTED FROM A1 CFG
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            rigid_body_enabled=True,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.00, 0.01, 0.01))
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.565),
        # Option 1: Standing Joint pos (values in radians)
        joint_pos={
            k: v * (np.pi / 180.0)  # convert to rad, the values below are in degrees
            for k, v in {
                "[F,R]R_hip_joint": -1.5,
                "[F,R]L_hip_joint": 1.5,
                ".*_thigh_joint": 42.0,
                ".*_calf_joint": -79.0,
            }.items()
        },
        # Option 2: Lying down with thighs up (values in radians)
        # joint_pos={
        #     "FR_hip_joint": -0.578783,
        #     "FR_thigh_joint": 1.471383,
        #     "FR_calf_joint": -2.750345,
        #     "FL_hip_joint": 0.570375,
        #     "FL_thigh_joint": 1.471515,
        #     "FL_calf_joint": -2.738487,
        #     "RR_hip_joint": -0.590774,
        #     "RR_thigh_joint": 1.463904,
        #     "RR_calf_joint": -2.622220,
        #     "RL_hip_joint": 0.566924,
        #     "RL_thigh_joint": 1.467400,
        #     "RL_calf_joint": -2.626244,
        # },
        # Option 3: Lying down with thighs down (initial position) (values in radians)
        # joint_pos={
        #     "FR_hip_joint": -0.558384,
        #     "FR_thigh_joint": 1.078270,
        #     "FR_calf_joint": -2.751709,
        #     "FL_hip_joint": 0.534135,
        #     "FL_thigh_joint": 1.089023,
        #     "FL_calf_joint": -2.739185,
        #     "RR_hip_joint": -0.584137,
        #     "RR_thigh_joint": 1.067164,
        #     "RR_calf_joint": -2.622180,
        #     "RL_hip_joint": 0.544445,
        #     "RL_thigh_joint": 1.051190,
        #     "RL_calf_joint": -2.626244,
        # },
        # Option 4: Standing from lemon's config
        # joint_pos={
        #     k: v * (np.pi / 180.0)  # convert to rad, the values below are in degrees
        #     for k, v in {
        #         "FL_hip_joint": 2.5,
        #         "FR_hip_joint": -2.5,
        #         "RL_hip_joint": 6.0,
        #         "RR_hip_joint": -6.0,
        #         "F[L,R]_thigh_joint": 30.0,
        #         "R[L,R]_thigh_joint": 45.0,
        #         "F[L,R]_calf_joint": -80.0,
        #         "R[L,R]_calf_joint": -79.0,
        #     }.items()
        # },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1,

    actuators={  # taken from URDF
        "hips": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=140.0,
            saturation_effort=150.0,
            velocity_limit=20,
            friction=2.9,
            dynamic_friction=0.0,
            viscous_friction=7.6,
            armature=0.08,
            stiffness=300.0,
            damping=10.0,
        ),
        "thighs": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=140.0,
            saturation_effort=150.0,
            velocity_limit=20,
            friction=3.5,
            dynamic_friction=3.4,
            viscous_friction=11.0,
            armature=0.3,
            stiffness=430.0,
            damping=10.0,
        ),
        "calves": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=140.0,
            saturation_effort=150.0,
            velocity_limit=20,
            friction=3.5,
            dynamic_friction=3.4,
            viscous_friction=13.0,
            armature=0.3,
            stiffness=400.0,
            damping=11.0,
        ),
    },
)
