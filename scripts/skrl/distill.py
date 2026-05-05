
EXPERT_MODELS = [
    # sit to stand
    {
        "model": "/home/afurman/dog_project/b1_rl_locomotion/scripts/skrl/logs/skrl/stand_sit/2026-04-29_23-40-51_ppo_torch_20/checkpoints/best_agent.pt",
        "yaml": "/home/afurman/dog_project/b1_rl_locomotion/scripts/skrl/logs/skrl/stand_sit/2026-04-29_23-40-51_ppo_torch_20/params/agent.yaml"
    },
    # stand to sit
    {
        "model": "/home/afurman/dog_project/convert_model/pt/std_sit_r26/best_agent.pt",
        "yaml": "/home/afurman/dog_project/b1_rl_locomotion/scripts/skrl/logs/skrl/stand_sit/2026-04-29_23-40-51_ppo_torch_20/params/agent.yaml"
    },
]


"""Launch Isaac Sim Simulator first."""

import argparse
import sys

import tqdm

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
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
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

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

import gymnasium as gym
import os
import random
from datetime import datetime

import omni
import torch
import torch.nn as nn
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import b1_rl_locomotion.tasks  # noqa: F401

from student import (
    StudentModelXS,
    StudentModelS,
    StudentModelM,
    StudentModelML,
    StudentModelL,
    StudentModelXL,
    StudentModelXLL,
    StudentModelXLLL,
    StudentModel,
    OBS_SIZE,
    ACT_SIZE,
)


# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    student = StudentModelL()

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        student.load_state_dict(torch.load(resume_path))

    experts = []
    for i, expert_files in enumerate(EXPERT_MODELS):
        experts.append(Runner(env, Runner.load_cfg_from_yaml(expert_files["yaml"])))
        experts[i].agent.load(expert_files["model"])
        # todo: check if this actually sets it to eval mode, though without dropout and batchnorm i'm not sure it matters
        experts[i].agent.set_running_mode("eval")


    # DAgger hyperparams
    T = 1500  # env steps per iter
    N = 80  # num iterations to go thru
    N_ACTIVE = N  # num iterations to sample from (disabled for now, don't need it)
    EPOCH = 30  # training epochs per iteration
    BATCH_SIZE = 256
    LR = 1e-3
    DECAY = 1e-2
    SCHED_RESTART_ITERS = 2  # dagger iterations per cosine restart
    ACT_VAR = 0.2  # action variance during data collection
    K = 0.85  # how much to prioritize choosing worst task (1 = max, 0 = uniform)
    T_EVAL = 2000  # eval steps per expert task

    # training settings
    CKPT_INT = 1  # how many iter before saving a checkpoint
    BEST_INT = 1  # how many iter before saving a best

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=DECAY)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=SCHED_RESTART_ITERS * EPOCH, eta_min=5e-4)

    D: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    bar = tqdm(total=(N * T) + (N * EPOCH), desc="Training", ascii=" ░▒█")
    
    # reset environment
    obs, _ = env.reset()

    # simulate environment
    while simulation_app.is_running():
        
        for n in range(N):
            Di: list[tuple[torch.Tensor, torch.Tensor, int]] = []

            # 1. collect trajectories under current student policy
            student.eval()
            
            time_alive = []
            with torch.no_grad():
                # Traceback (most recent call last):
                #     File "/home/shared/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 101, in hydra_main
                #         func(env_cfg, agent_cfg, *args, **kwargs)
                #     File "/home/afurman/dog_project/b1_rl_locomotion/scripts/skrl/distill.py", line 284, in main
                #         done = True
                #     TypeError: 'module' object is not callable

                #     Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
                done = True
                alive = 0

                for _ in range(T):

                    #TODO: make this reset per env
                    if done:
                        current_task = int(
                            torch.multinomial(torch.tensor([
                                1.0, 1.0
                            ]), env.num_envs, replacement=True) #TODO: implement task PMF
                        )
                        configureEnv(env, current_task)
                        obs, info = env.reset()
                        alive = 0

                    act_expert = experts[current_task].agent.act(obs, timestep=0, timesteps=0)
                    act_expert = act_expert[-1].get("mean_actions", act_expert[0])
                    # add zero-mean gaussian noise
                    act_expert += torch.normal(0, ACT_VAR**0.5, act_expert.shape)

                    obs_s = StudentModel.obs(obs, current_task)
                    act_student = student(obs_s)

                    alive += 1
                    obs, _, term, trunc, info = env.step(act_student)
                    done = term or trunc

                    if done:
                        time_alive.append(alive)

                    Di.append((obs_s, act_expert, current_task))
                    bar.update(1)

            # 2. aggregate dataset
            D += Di

            print(D)


            # 3. build training batch, prioritising recent data
            # obs_list, act_list, task_ids_all = zip(*D)  # type: ignore
            # if len(D) > N_ACTIVE * T:
            #     # linearly increasing weights: oldest sample is around 0, newest = highest
            #     w = np.arange(1, len(D) + 1, dtype=np.float64)
            #     w /= w.sum()
            #     idx = np.random.choice(len(D), size=N_ACTIVE * T, replace=False, p=w)
            #     obs_arr = np.array(obs_list)[idx]
            #     act_arr = np.array(act_list)[idx]
            #     task_ids_all = tuple(np.array(task_ids_all)[idx])
            # else:
            #     obs_arr = np.array(obs_list)
            #     act_arr = np.array(act_list)

            # x_full = torch.tensor(obs_arr, dtype=torch.float32)
            # y_full = torch.tensor(act_arr, dtype=torch.float32)

            # loader = DataLoader(
            #     TensorDataset(x_full, y_full), batch_size=BATCH_SIZE, shuffle=True
            # )

            # student.to(device)
            # student.train()

            # for epoch in range(EPOCH):
            #     epoch_loss = 0.0
            #     for obs_batch, act_batch in loader:
            #         obs_batch = obs_batch.to(device)
            #         act_batch = act_batch.to(device)

            #         pred = student(obs_batch)
            #         loss = loss_fn(pred, act_batch)

            #         optimizer.zero_grad()
            #         loss.backward()
            #         optimizer.step()
            #         epoch_loss += loss.item()

            #     writer.add_scalar("train/loss", epoch_loss / len(loader), n * EPOCH + epoch)
            #     # scheduler.step()
            #     bar.update(1)

            # # log per-iteration scalars
            # writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], n * EPOCH)
            # writer.add_scalar("train/dataset_size", len(D), n * EPOCH)
            # if time_alive:
            #     writer.add_scalar(
            #         "train/avg_time_alive", float(np.mean(time_alive)), n * EPOCH
            #     )

            # # per-expert training loss on full accumulated dataset
            # student.eval()
            # student.to("cpu")
            # with torch.no_grad():
            #     pred_all = student(x_full)
            #     total_D = len(task_ids_all)
            #     for task_id, task_name in enumerate(TASK_NAMES):
            #         indices = [i for i, t in enumerate(task_ids_all) if t == task_id]
            #         if indices:
            #             idx_t = torch.tensor(indices)
            #             writer.add_scalar(
            #                 f"train/loss_{task_name}",
            #                 F.mse_loss(pred_all[idx_t], y_full[idx_t]).item(),
            #                 n * EPOCH,
            #             )
            #         writer.add_scalar(
            #             f"train/task_pct_{task_name}",
            #             len(indices) / total_D,
            #             n * EPOCH,
            #         )

            # # 4. eval
            # eval_loss, _, task_live_time = evaluate(n * EPOCH)

            # # 5. save checkpoints
            # if n % CKPT_INT == 0 or n == N - 1:
            #     torch.save(
            #         {
            #             "policy": student.state_dict(),
            #             "optimizer": optimizer.state_dict(),
            #         },
            #         str(MODELS_DIR / EXPERIMENT_NAME / f"distill_{n}.pt"),
            #     )
            # if n % BEST_INT == 0:
            #     if eval_loss < best_loss:  # save best model
            #         torch.save(
            #             {
            #                 "policy": student.state_dict(),
            #                 "optimizer": optimizer.state_dict(),
            #             },
            #             str(MODELS_DIR / EXPERIMENT_NAME / f"best.pt"),
            #         )
            #         best_loss = eval_loss

            # iter_time = time.time() - iter_start
            # iter_per_s = 1.0 / iter_time
            # writer.add_scalar("train/iter_per_s", iter_per_s, n)
            # writer.add_scalar("train/iter_time_s", iter_time, n)
            # bar.set_postfix(iter_s=f"{iter_per_s:.3f}")


    # close the simulator
    env.close()



def configureEnv(env, task):
    print("new env: ", env, task)
    # TODO: see if tasks need specific environment configuration

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
