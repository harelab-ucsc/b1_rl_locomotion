# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# All projects in this dir MUST be prefixed by "B1-"

gym.register(
    id="B1-Sit",
    entry_point=f"{__name__}.env:B1RlLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b1_rl_locomotion_env_cfg:B1RlLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B1PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="B1-Sit-Play",
    entry_point=f"{__name__}.env:B1RlLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b1_rl_locomotion_env_cfg:B1RlLocomotionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B1PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="B1-Stand",
    entry_point=f"{__name__}.env:B1RlLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_env_cfg:B1StandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B1PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_stand_cfg.yaml",
    },
)

gym.register(
    id="B1-Stand-Play",
    entry_point=f"{__name__}.env:B1RlLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_env_cfg:B1StandEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B1PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_stand_cfg.yaml",
    },
)
