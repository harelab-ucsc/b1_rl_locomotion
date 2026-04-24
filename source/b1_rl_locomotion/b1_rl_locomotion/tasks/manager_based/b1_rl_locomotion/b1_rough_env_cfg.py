from isaaclab.utils import configclass

from b1_rl_locomotion.tasks.manager_based.b1_rl_locomotion.b1_rl_locomotion_env_cfg import B1RlLocomotionEnvCfg
from b1_rl_locomotion.tasks.manager_based.b1_rl_locomotion.configs.b1 import B1_CFG


@configclass
class B1RoughEnvCfg(B1RlLocomotionEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = B1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # rewards
        self.rewards.base_height.weight = 0.0
        self.rewards.vel_z.weight = -0.2
        self.rewards.flat_orientation_l2.weight = 0.0



@configclass
class B1RoughEnvCfg_PLAY(B1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None