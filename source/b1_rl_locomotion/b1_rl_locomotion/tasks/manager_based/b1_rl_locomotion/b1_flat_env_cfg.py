from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from b1_rl_locomotion.tasks.manager_based.b1_rl_locomotion.b1_rl_locomotion_env_cfg import B1RlLocomotionEnvCfg
from b1_rl_locomotion.tasks.manager_based.b1_rl_locomotion.b1_rough_env_cfg import B1RoughEnvCfg

# @configclass
# class PhysicsCfg(PresetCfg):
#     default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
#     newton = NewtonCfg(
#         solver_cfg=MJWarpSolverCfg(
#             njmax=60,
#             nconmax=30,
#             cone="pyramidal",
#             impratio=1,
#             integrator="implicitfast",
#         ),
#         num_substeps=1,
#         debug_mode=False,
#     )
#     physx = default


@configclass
class B1FlatEnvCfg(B1RoughEnvCfg):
    # sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # override rewards
        self.rewards.flat_orientation_l2.weight = -8.0
        self.rewards.dof_torques_l2.weight = -2.5e-6
        self.rewards.feet_air_time.weight = 8.0
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class B1FlatEnvCfg_PLAY(B1FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
