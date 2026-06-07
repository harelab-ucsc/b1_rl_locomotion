# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to compare two RL checkpoints side by side.

Runs two models simultaneously in the same simulation, each controlling half
of the total environments. Both groups reset at the same time whenever any
episode terminates.

Green balls above model1, blue above model2
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Compare two skrl RL checkpoints side by side.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments per model (total will be 2x).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint1", type=str, required=True, help="Path to first model checkpoint.")
parser.add_argument("--checkpoint2", type=str, required=True, help="Path to second model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    help=(
        "Name of the RL algorithm to use (e.g. AMP, DDPG, IPPO, MAPPO, PPO, SAC, TD3, etc.) "
        "when several algorithms exist for the same task. For a more specific selection, use the argument --agent."
    ),
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "2.0.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from b1_rl_locomotion.tasks.manager_based.b1_rl_locomotion.agents.runner import B1Runner as Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import b1_rl_locomotion.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Compare two skrl agents side by side."""
    # per-model env count; total envs will be 2x
    num_envs_per_model = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.scene.num_envs = num_envs_per_model * 2
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    resume_path1 = os.path.abspath(args_cli.checkpoint1)
    resume_path2 = os.path.abspath(args_cli.checkpoint2)

    # use checkpoint1's directory for any env logging
    log_dir = os.path.dirname(os.path.dirname(resume_path1))
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "compare"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during comparison.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # configure and instantiate two runners sharing the same environment
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    experiment_cfg["agent"]["experiment"]["wandb"] = False  # don't log to wandb

    runner1 = Runner(env, experiment_cfg)
    runner2 = Runner(env, copy.deepcopy(experiment_cfg))

    print(f"[INFO] Loading model 1 from: {resume_path1}")
    runner1.agent.load(resume_path1)
    runner1.agent.enable_training_mode(False, apply_to_models=True)

    print(f"[INFO] Loading model 2 from: {resume_path2}")
    runner2.agent.load(resume_path2)
    runner2.agent.enable_training_mode(False, apply_to_models=True)

    # reset environment — both groups start fresh simultaneously
    obs, _ = env.reset()
    states = env.state()
    timestep = 0

    # colored sphere markers to identify each model group above the robots
    _marker_height = torch.tensor([0.0, 0.0, 1.2], device=env_cfg.sim.device)

    _model1_markers = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/model1_labels",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.12,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # green
                )
            },
        )
    )
    _model2_markers = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/model2_labels",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.12,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0)),  # blue
                )
            },
        )
    )

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # update label positions above each robot
            root_pos = env.unwrapped.scene["robot"].data.root_pos_w
            _model1_markers.visualize(translations=root_pos[:num_envs_per_model] + _marker_height)
            _model2_markers.visualize(translations=root_pos[num_envs_per_model:] + _marker_height)

            # split observations between the two models
            obs1 = obs[:num_envs_per_model]
            obs2 = obs[num_envs_per_model:]
            if states is not None:
                states1 = states[:num_envs_per_model]
                states2 = states[num_envs_per_model:]
            else:
                states1 = states2 = None

            # each agent acts on its half of the environments
            outputs1 = runner1.agent.act(obs1, states1, timestep=0, timesteps=0)
            outputs2 = runner2.agent.act(obs2, states2, timestep=0, timesteps=0)

            actions1 = outputs1[-1].get("mean_actions", outputs1[0])
            actions2 = outputs2[-1].get("mean_actions", outputs2[0])
            actions = torch.cat([actions1, actions2], dim=0)

            obs, _, terminated, truncated, _ = env.step(actions)
            states = env.state()

            # reset all envs when any episode ends so both groups stay in sync
            if (terminated | truncated).any():
                obs, _ = env.reset()
                states = env.state()

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
