"""Configuration for the B1 dog.

The following configurations are available:

* :obj:`UNITREE_B1_CFG`: Unitree B1 robot with DC motor model for the legs. Adapted from the A1 CFG.
TODO: match the DC motor actuator to the real specified values from Unitree

Reference:
    - https://github.com/unitreerobotics/unitree_ros
    - /pvc/isaac-sim/IsaacLab/scripts/lemon/main.py
"""

import isaaclab.sim as sim_utils

from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg
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
        pos=(0.0, 0.0, 0.76258),
        joint_pos={
            k: v * (np.pi / 180.0)  # convert to rad, the values below are in degrees
            for k, v in {
                "FL_hip_joint": 2.5,
                "FR_hip_joint": -2.5,
                "RL_hip_joint": 6.0,
                "RR_hip_joint": -6.0,
                "F[L,R]_thigh_joint": 30.0,
                "R[L,R]_thigh_joint": 45.0,
                "F[L,R]_calf_joint": -80.0,
                "R[L,R]_calf_joint": -79.0,
            }.items()
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.75,  # 0.9
    # actuators={
    #     "base_legs": DCMotorCfg(
    #         joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    #         effort_limit= 1000, # 33.5
    #         saturation_effort=1000, # 33.5
    #         velocity_limit=21.0,
    #         stiffness=240, # 25.0
    #         damping=10, # 0.5
    #         friction=0.0,
    #     ),
    # },
    actuators={  # taken from URDF
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=None, damping=None
        )
        # "hips": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_hip_joint"],
        #     # effort_limit_sim=91.0,
        #     # saturation_effort=91.0,
        #     # velocity_limit_sim=20,
        #     stiffness=173.2,
        #     damping=17.32,
        # ),
        # "thighs": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_thigh_joint"],
        #     # effort_limit_sim=93.33,
        #     # saturation_effort=93.33,
        #     # velocity_limit_sim=20,
        #     stiffness=173.2,
        #     damping=17.32,
        # ),
        # "calves": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_calf_joint"],
        #     # effort_limit_sim=140.0,
        #     # saturation_effort=140.0,
        #     # velocity_limit_sim=20,
        #     stiffness=173.2,
        #     damping=17.32,
        # ),
    },
)

"""
Note: Check specifications from: https://www.trossenrobotics.com/a1-quadruped#specifications
"""
