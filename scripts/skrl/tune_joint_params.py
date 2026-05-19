"""Tune joint friction parameters by comparing simulation step response to real-world CSV data.

Protocol:
  - CSV rows 0-499:   joint held at cmd=0 (static hold)
  - CSV rows 500-999: step command applied
  - CSV rows 1000-:   joint commanded back to 0

For each friction combination the script replays the CSV commands in simulation, records
joint positions, and computes RMSE vs. the recorded data.  Results are saved to a CSV and
three plots are generated: a top-5 trajectory overlay, a per-phase RMSE heatmap, and a
best-fit single-trial plot.

Parallelism (--num-envs N > 1):
  Grid: evaluates N parameter combinations simultaneously in one sim run.
  Anneal: runs N independent SA chains in parallel, one proposal per chain
          per step, each chain maintaining its own current state.
"""

import argparse

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Friction sweep for B1 joint step-response matching.")
parser.add_argument("--joint", type=str, default="hip", choices=["hip", "thigh", "calf"],
                    help="Which joint role to tune (CSV columns 0/1/2).")
parser.add_argument("--leg", type=str, default="FL",
                    choices=["FR", "FL", "RR", "RL"],
                    help="Which leg to test on in simulation.")
parser.add_argument("--csv-real", type=str, default="joint_data_aligned.csv",
                    help="Real recorded CSV (relative to this script's directory).")
parser.add_argument("--csv-out", type=str, default="tune_results.csv",
                    help="Sweep results CSV (relative to this script's directory).")
parser.add_argument("--plot-dir", type=str, default="tune_plots",
                    help="Directory (relative to this script) for output plots.")

parser.add_argument("--static-friction", type=str, default=",".join([f"{np.exp(i*0.3)-1:.1f}" for i in range(0,6,1)]),
                    help="Comma-separated static friction values to sweep.")
parser.add_argument("--dynamic-friction", type=str, default= ",".join([f"{np.exp(i*0.3)-1:.1f}" for i in range(0,6,1)]),
                    help="Comma-separated dynamic friction values to sweep.")
parser.add_argument("--viscous-friction", type=str, default=",".join([f"{np.exp(i*0.2)+2:.1f}" for i in range(0,13,1)]),
                    help="Comma-separated viscous friction values to sweep.")
parser.add_argument("--armature", type=str, default="0.0,0.05,0.1,0.3",
                    help="Comma-separated joint armature values to sweep.")
parser.add_argument("--inertia-scale", type=str, default="1.0",
                    help="Comma-separated scale factors applied to the default inertia tensor "
                         "of the child body of the active joint.")
parser.add_argument("--stiffness", type=str, default=",".join([f"{i:.1f}" for i in range(360,450,10)]),
                    help="Comma-separated joint stiffness (P-gain) values to sweep.")
parser.add_argument("--damping", type=str, default=",".join([f"{i:.1f}" for i in range(6,27,5)]),
                    help="Comma-separated joint damping (D-gain) values to sweep.")

parser.add_argument("--csv-hz", type=float, default=125.0,
                    help="Recording rate of the real CSV (Hz); sets the sim timestep.")
parser.add_argument("--static-rows", type=int, default=200,
                    help="Number of CSV rows that form the static hold phase.")
parser.add_argument("--step-rows", type=int, default=500,
                    help="Number of CSV rows that form the step-response phase.")
parser.add_argument("--no-gravity", action="store_true",
                    help="Disable gravity (default: on; robot base is fixed so it won't fall).")
parser.add_argument("--num-envs", type=int, default=1,
                    help="Number of parallel robot instances. "
                         "Grid: evaluates this many combos at once. "
                         "Anneal: runs this many independent SA chains in parallel. "
                         "Recommended: 64-100 for fast sweeps.")
# ── mode ──────────────────────────────────────────────────────────────────────
parser.add_argument("--mode", type=str, default="grid", choices=["grid", "anneal", "test-values"],
                    help="Optimisation mode: grid sweep, simulated annealing, or single-point test.")
parser.add_argument("--metric", type=str, default="total", choices=["step", "total", "max"],
                    help="RMSE metric to optimise: 'step' (step phase only), 'total' (all phases), or 'max' (peak error at any timestep).")
parser.add_argument("--live-plot", action="store_true",
                    help="[SA] Show a live trajectory plot updated each SA step (requires a display).")
# SA-specific args (ignored in grid mode)
parser.add_argument("--sa-steps", type=int, default=200,
                    help="[SA] Number of SA iterations.")
parser.add_argument("--sa-t0", type=float, default=0.05,
                    help="[SA] Initial temperature (in RMSE units).")
parser.add_argument("--sa-alpha", type=float, default=0.97,
                    help="[SA] Geometric cooling factor applied each iteration.")
parser.add_argument("--sa-sigma", type=str, default="0.1",
                    help="[SA] Std-dev of Gaussian neighbour proposal. Either a single float "
                         "(applied to all params) or 7 comma-separated floats: "
                         "static,dynamic,viscous,armature,stiffness,damping,inertia_scale.")
parser.add_argument("--sa-init", type=str, default=None,
                    help="[SA] Initial point as 'mu_s,mu_d,c_v,armature,stiffness,damping,inertia_scale'. "
                         "Defaults to the midpoint of each range.")
parser.add_argument("--sa-bounds-static", type=str, default="0.0,10.0",
                    help="[SA] lo,hi clamp for static friction (mu_s).")
parser.add_argument("--sa-bounds-dynamic", type=str, default="0.0,10.0",
                    help="[SA] lo,hi clamp for dynamic friction (mu_d).")
parser.add_argument("--sa-bounds-viscous", type=str, default="0.0,20.0",
                    help="[SA] lo,hi clamp for viscous friction (c_v).")
parser.add_argument("--sa-bounds-armature", type=str, default="0.07,0.09",
                    help="[SA] lo,hi clamp for armature.")
parser.add_argument("--sa-bounds-stiffness", type=str, default="1.0,500.0",
                    help="[SA] lo,hi clamp for stiffness (P-gain).")
parser.add_argument("--sa-bounds-damping", type=str, default="0.1,50.0",
                    help="[SA] lo,hi clamp for damping (D-gain).")
parser.add_argument("--sa-bounds-inertia", type=str, default="0.1,10.0",
                    help="[SA] lo,hi clamp for inertia scale factor.")
parser.add_argument("--sa-seed", type=int, default=0,
                    help="[SA] NumPy random seed.")
parser.add_argument("--test-values", type=str, default=None,
                    help="[test-values mode] 7 comma-separated parameter values: "
                         "static_friction,dynamic_friction,viscous_friction,armature,"
                         "stiffness,damping,inertia_scale. "
                         "Runs a single env with the original, 0.5×, and 1.5× command scales.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── everything below runs after the sim app is up ────────────────────────────

import csv
import itertools
import math
import os
import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from b1_rl_locomotion.tasks.manager_based.b1_rl_locomotion.configs.b1 import B1_CFG  # noqa: E402

JOINT_SUFFIX = {"hip": "hip_joint", "thigh": "thigh_joint", "calf": "calf_joint"}
CSV_JOINT_IDX = {"hip": 0, "thigh": 1, "calf": 2}
PHASE_STATIC = 0
PHASE_STEP = 1
PHASE_RETURN = 2


def load_real_csv(path: str, joint_idx: int):
    """Return (cmd, pos) as np.float32 arrays for the requested joint column."""
    cmds, positions = [], []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            cmds.append(float(row[f"cmd_{joint_idx}"]))
            positions.append(float(row[f"pos_{joint_idx}"]))
    return np.asarray(cmds, dtype=np.float32), np.asarray(positions, dtype=np.float32)


def phase_mask(n: int, static: int, step: int):
    """Return phase_ids int array of length n."""
    ids = np.full(n, PHASE_RETURN, dtype=np.int8)
    ids[:static] = PHASE_STATIC
    ids[static: static + step] = PHASE_STEP
    return ids


def run_batch(
    robot: Articulation,
    sim: SimulationContext,
    active_idx: int,
    active_actuator,
    active_local_idx: int,
    cmd_tensor: torch.Tensor,
    joint_offset: torch.Tensor,
    dt: float,
    params_list: list,
    num_envs: int,
    device: str,
    default_inertias: torch.Tensor,
    active_body_idx: int,
) -> torch.Tensor:
    """Run len(params_list) parameter sets in parallel across num_envs robots.

    cmd_tensor and returned positions are both in the relative frame (relative to
    the default standing pose). joint_offset is the default absolute position of
    the active joint, shape (num_envs,).

    Returns sim joint positions (relative) with shape (len(params_list), T).
    If the simulation stops early, the time dimension is truncated.
    """
    n_batch = len(params_list)
    # Pad to num_envs so every robot is doing something (excess results are discarded)
    padded = list(params_list)
    if n_batch < num_envs:
        padded += [params_list[0]] * (num_envs - n_batch)

    # Reset all joints to the default standing pose (not zero)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
    )
    robot.reset()

    # Scale per-env body inertia then set them all in one PhysX call (CPU tensors required)
    inertias_buf = default_inertias.clone()
    for i, params in enumerate(padded):
        inertias_buf[i, active_body_idx] = default_inertias[i, active_body_idx] * float(params[6])
    robot.root_physx_view.set_inertias(inertias_buf, torch.arange(num_envs, dtype=torch.long))

    # Write per-robot friction, armature, stiffness, and damping parameters
    for i, (mu_s, mu_d, c_v, arm, kp, kd, _) in enumerate(padded):
        env_ids = torch.tensor([i], device=device, dtype=torch.long)
        robot.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff=float(mu_s),
            joint_dynamic_friction_coeff=float(mu_d),
            joint_viscous_friction_coeff=float(c_v),
            joint_ids=[active_idx],
            env_ids=env_ids,
        )
        robot.write_joint_armature_to_sim(float(arm), joint_ids=[active_idx], env_ids=env_ids)
        # For explicit actuators (DCMotor), stiffness/damping live on the actuator object,
        # not on the PhysX drive. write_joint_stiffness_to_sim would enable a second
        # position drive on top of the actuator torques, causing double control.
        if active_actuator.is_implicit_model:
            robot.write_joint_stiffness_to_sim(float(kp), joint_ids=[active_idx], env_ids=env_ids)
            robot.write_joint_damping_to_sim(float(kd), joint_ids=[active_idx], env_ids=env_ids)
        else:
            active_actuator.stiffness[i, active_local_idx] = kp
            active_actuator.damping[i, active_local_idx] = kd

    n_steps = len(cmd_tensor)
    # Start target at default pose; only the active joint changes each step
    target = robot.data.default_joint_pos.clone()
    sim_pos = torch.empty((num_envs, n_steps), device=device)

    for t in range(n_steps):
        # CSV command is relative → convert to absolute before sending
        target[:, active_idx] = joint_offset + cmd_tensor[t]
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        # Record position in the same relative frame as the CSV
        sim_pos[:, t] = robot.data.joint_pos[:, active_idx] - joint_offset
        if not simulation_app.is_running():
            return sim_pos[:n_batch, :t]

    return sim_pos[:n_batch]


def compute_metrics(sim_pos: np.ndarray, real_pos: np.ndarray, phase_ids: np.ndarray):
    """Return dict of RMSE/MAE/max-err per phase and total."""
    out = {}
    for label, code in [("static", PHASE_STATIC), ("step", PHASE_STEP),
                        ("return", PHASE_RETURN), ("total", -1)]:
        if code == -1:
            mask = np.ones(len(sim_pos), dtype=bool)
        else:
            mask = phase_ids == code
        err = sim_pos[mask] - real_pos[mask]
        out[f"rmse_{label}"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"mae_{label}"] = float(np.mean(np.abs(err)))
        out[f"max_err_{label}"] = float(np.max(np.abs(err)))
    return out


TOP_K_TRAJ = 50  # max trajectories held in memory for plotting
PLOT_METRIC_KEYS = ["rmse_total", "rmse_step", "max_err_total"]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_in = os.path.join(script_dir, args_cli.csv_real)
    csv_out_path = os.path.join(script_dir, args_cli.csv_out)
    plot_dir = os.path.join(script_dir, args_cli.plot_dir)

    # Open the CSV immediately so the file exists on disk before sim init.
    # A crash during robot setup would otherwise leave no output at all.
    os.makedirs(plot_dir, exist_ok=True)
    metric_keys = [
        "rmse_static", "mae_static", "max_err_static",
        "rmse_step",   "mae_step",   "max_err_step",
        "rmse_return", "mae_return", "max_err_return",
        "rmse_total",  "mae_total",  "max_err_total",
    ]
    extra_cols = ["iteration", "temperature", "chain", "accepted"] if args_cli.mode == "anneal" else []
    if args_cli.mode != "test-values":
        results_file = open(csv_out_path, "w", newline="")
        writer = csv.writer(results_file)
        writer.writerow(extra_cols + ["static_friction", "dynamic_friction", "viscous_friction",
                                      "armature", "stiffness", "damping", "inertia_scale"] + metric_keys)
        results_file.flush()

    joint_role = args_cli.joint
    csv_idx = CSV_JOINT_IDX[joint_role]
    print(f"[INFO] Loading real data from {csv_in} (joint {joint_role}, CSV col {csv_idx})")
    cmd_real, pos_real = load_real_csv(csv_in, csv_idx)
    n_steps = len(cmd_real)
    phases = phase_mask(n_steps, args_cli.static_rows, args_cli.step_rows)
    dt = 1.0 / args_cli.csv_hz
    num_envs = args_cli.num_envs
    if args_cli.mode == "test-values":
        num_envs = 1
    print(f"[INFO] {n_steps} CSV rows at {args_cli.csv_hz} Hz "
          f"-> {n_steps * dt:.2f} s replay, dt = {dt:.5f} s")
    print(f"[INFO] Parallel environments: {num_envs}")

    # ── sim setup ─────────────────────────────────────────────────────────────
    gravity = (0.0, 0.0, 0.0) if args_cli.no_gravity else (0.0, 0.0, -9.81)
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=args_cli.device or "cuda:0", gravity=gravity)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 1.0], target=[0.0, 0.0, 0.5])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    # ── robot(s) ──────────────────────────────────────────────────────────────
    # Fix the root link so the trunk is pinned in place (matches real hang-test setup).
    fixed_spawn_cfg = B1_CFG.spawn.replace(
        articulation_props=B1_CFG.spawn.articulation_props.replace(fix_root_link=True)
    )

    if num_envs > 1:
        # Spawn N robots in a grid, then wrap them with one batched Articulation.
        # asset_base.py skips spawning when cfg.spawn is None, so we pre-spawn manually.
        cols = math.ceil(math.sqrt(num_envs))
        spacing = 2.5  # metres between robots
        for i in range(num_envs):
            x = float(i % cols) * spacing
            y = float(i // cols) * spacing
            fixed_spawn_cfg.func(
                f"/World/envs/env_{i}/Robot",
                fixed_spawn_cfg,
                translation=(x, y, 0.565),
                orientation=B1_CFG.init_state.rot,
            )
        robot_cfg = B1_CFG.copy()
        robot_cfg.prim_path = "/World/envs/env_.*/Robot"
        robot_cfg.spawn = None  # prims already spawned above
        robot = Articulation(robot_cfg)
        print(f"[INFO] Spawned {num_envs} robots in a {cols}×{math.ceil(num_envs/cols)} grid.")
    else:
        robot_cfg = B1_CFG.copy()
        robot_cfg.prim_path = "/World/Robot"
        robot_cfg.spawn = fixed_spawn_cfg
        robot = Articulation(robot_cfg)

    sim.reset()
    print("[INFO] Sim reset complete.")

    joint_names = robot.data.joint_names
    target_joint_name = f"{args_cli.leg}_{JOINT_SUFFIX[joint_role]}"
    if target_joint_name not in joint_names:
        raise ValueError(
            f"Joint '{target_joint_name}' not found in articulation. "
            f"Available: {joint_names}"
        )
    active_idx = joint_names.index(target_joint_name)
    n_joints = len(joint_names)
    device = str(sim.device)
    # Default absolute position of the active joint — used to convert between
    # relative (CSV) and absolute (sim) frames.
    joint_offset = robot.data.default_joint_pos[:, active_idx].clone()
    print(f"[INFO] Active joint: '{target_joint_name}' (idx {active_idx}), "
          f"default offset = {joint_offset[0].item():.4f} rad")

    # Find which actuator controls this joint (needed to set kp/kd correctly).
    active_actuator = None
    active_local_idx = None
    for act in robot.actuators.values():
        joint_indices_list = list(act.joint_indices)
        if active_idx in joint_indices_list:
            active_local_idx = joint_indices_list.index(active_idx)
            active_actuator = act
            break
    if active_actuator is None:
        raise ValueError(f"No actuator found for joint index {active_idx}")
    print(f"[INFO] Actuator: {'implicit' if active_actuator.is_implicit_model else 'explicit'}, "
          f"default kp={active_actuator.stiffness[0, active_local_idx].item():.1f}, "
          f"kd={active_actuator.damping[0, active_local_idx].item():.1f}")

    # Find the child body driven by the active joint (name = joint name minus "_joint" suffix).
    active_body_name = target_joint_name.replace("_joint", "")
    if active_body_name not in robot.body_names:
        raise ValueError(
            f"Child body '{active_body_name}' not found. Available: {robot.body_names}"
        )
    active_body_idx = robot.body_names.index(active_body_name)
    # Store default inertias on CPU — PhysX view tensors are CPU-resident.
    default_inertias = robot.root_physx_view.get_inertias().clone().cpu()
    print(f"[INFO] Active body: '{active_body_name}' (idx {active_body_idx}), "
          f"default inertia diag = [{default_inertias[0, active_body_idx, 0]:.5g}, "
          f"{default_inertias[0, active_body_idx, 4]:.5g}, "
          f"{default_inertias[0, active_body_idx, 8]:.5g}]")

    cost_key = "max_err_total" if args_cli.metric == "max" else f"rmse_{args_cli.metric}"
    cmd_tensor = torch.tensor(cmd_real, dtype=torch.float32, device=device)
    t_axis = np.arange(n_steps) * dt

    phase_ticks = [args_cli.static_rows * dt,
                   (args_cli.static_rows + args_cli.step_rows) * dt]

    def add_phase_lines(ax):
        for x in phase_ticks:
            ax.axvline(x, color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.axvspan(0, phase_ticks[0], alpha=0.04, color="blue", label="_static region")
        ax.axvspan(phase_ticks[0], phase_ticks[1], alpha=0.04, color="green", label="_step region")

    # ── matplotlib setup ──────────────────────────────────────────────────────
    try:
        import matplotlib
        if args_cli.live_plot and args_cli.mode == "anneal":
            matplotlib.use("TkAgg")
        else:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _mpl_ok = True
    except Exception:
        plt = None
        _mpl_ok = False

    statics = [float(x) for x in args_cli.static_friction.split(",")]
    dynamics = [float(x) for x in args_cli.dynamic_friction.split(",")]
    viscouses = [float(x) for x in args_cli.viscous_friction.split(",")]
    armatures = [float(x) for x in args_cli.armature.split(",")]
    stiffnesses = [float(x) for x in args_cli.stiffness.split(",")]
    dampings = [float(x) for x in args_cli.damping.split(",")]
    inertia_scales = [float(x) for x in args_cli.inertia_scale.split(",")]



    # top-K trajectories for plotting; all_param_metrics stores params+metrics for every trial
    all_results = []       # (params_tuple, sim_trace_np, metrics_dict) — capped at TOP_K_TRAJ, sorted by cost_key
    all_param_metrics = [] # (params_tuple, metrics_dict) — unbounded, no trajectory data
    top_results = {mk: [] for mk in PLOT_METRIC_KEYS}  # per-metric top-K lists for grid plots

    def evaluate_batch(params_list: list):
        """Run a batch of (mu_s, mu_d, c_v, arm) tuples in parallel.

        Returns list of (sim_pos_np, metrics_dict) or (None, None) on early stop.
        """
        sim_pos_batch = run_batch(robot, sim, active_idx, active_actuator, active_local_idx,
                                  cmd_tensor, joint_offset, dt, params_list, num_envs, device,
                                  default_inertias, active_body_idx)
        results = []
        for i, params in enumerate(params_list):
            traj = sim_pos_batch[i]
            if traj.shape[0] < n_steps:
                results.append((None, None))
            else:
                sp_np = traj.cpu().numpy()
                metrics = compute_metrics(sp_np, pos_real, phases)
                params_t = tuple(params)
                all_param_metrics.append((params_t, metrics))
                # Only keep top-K trajectories in memory to avoid OOM on large grids.
                cost = metrics[cost_key]
                if len(all_results) < TOP_K_TRAJ or cost < all_results[-1][2][cost_key]:
                    all_results.append((params_t, sp_np, metrics))
                    all_results.sort(key=lambda x: x[2][cost_key])
                    del all_results[TOP_K_TRAJ:]
                for mk in PLOT_METRIC_KEYS:
                    mk_cost = metrics[mk]
                    if len(top_results[mk]) < TOP_K_TRAJ or mk_cost < top_results[mk][-1][2][mk]:
                        top_results[mk].append((params_t, sp_np, metrics))
                        top_results[mk].sort(key=lambda x, k=mk: x[2][k])
                        del top_results[mk][TOP_K_TRAJ:]
                results.append((sp_np, metrics))
        return results

    t_start = time.monotonic()

    # ── grid mode ─────────────────────────────────────────────────────────────
    if args_cli.mode == "grid":
        combos = [(s, d, v, a, kp, kd, iscale)
                  for s, d, v, a, kp, kd, iscale in itertools.product(
                      statics, dynamics, viscouses, armatures, stiffnesses, dampings, inertia_scales)
                  if d <= s]
        n_batches = math.ceil(len(combos) / num_envs)
        print(f"[INFO] Grid sweep: {len(combos)} combinations in {n_batches} batches of up to {num_envs}.")

        done = False
        total_done = 0
        for batch_start in range(0, len(combos), num_envs):
            chunk = combos[batch_start: batch_start + num_envs]
            batch_results = evaluate_batch(list(chunk))

            for params, (_, metrics) in zip(chunk, batch_results):
                mu_s, mu_d, c_v, arm, kp, kd, iscale = params
                total_done += 1
                if metrics is None:
                    print("[WARN] Trial cut short, stopping sweep.")
                    done = True
                    break
                writer.writerow([mu_s, mu_d, c_v, arm, kp, kd, iscale] + [metrics[k] for k in metric_keys])
                results_file.flush()
                elapsed = time.monotonic() - t_start
                eta = elapsed / total_done * (len(combos) - total_done)
                print(f"[{total_done:3d}/{len(combos)}] mu_s={mu_s:.3g}  mu_d={mu_d:.3g}  c_v={c_v:.3g}  "
                      f"arm={arm:.3g}  kp={kp:.3g}  kd={kd:.3g}  iscale={iscale:.3g}  "
                      f"rmse_step={metrics['rmse_step']:.4f}  "
                      f"rmse_total={metrics['rmse_total']:.4f}  [{args_cli.metric}={metrics[cost_key]:.4f}]  "
                      f"[{elapsed:.0f}s elapsed  ETA {eta:.0f}s]")
            if done:
                break

    # ── simulated annealing mode ───────────────────────────────────────────────
    elif args_cli.mode == "anneal":
        rng = np.random.default_rng(args_cli.sa_seed)
        lo = np.array([float(x.split(",")[0]) for x in [
            args_cli.sa_bounds_static, args_cli.sa_bounds_dynamic,
            args_cli.sa_bounds_viscous, args_cli.sa_bounds_armature,
            args_cli.sa_bounds_stiffness, args_cli.sa_bounds_damping,
            args_cli.sa_bounds_inertia]])
        hi = np.array([float(x.split(",")[1]) for x in [
            args_cli.sa_bounds_static, args_cli.sa_bounds_dynamic,
            args_cli.sa_bounds_viscous, args_cli.sa_bounds_armature,
            args_cli.sa_bounds_stiffness, args_cli.sa_bounds_damping,
            args_cli.sa_bounds_inertia]])

        if args_cli.sa_init is not None:
            init_point = np.array([float(x) for x in args_cli.sa_init.split(",")], dtype=np.float64)
            init_point[1] = min(init_point[1], init_point[0])  # mu_d <= mu_s
        else:
            init_point = np.array([
                np.median(statics),
                np.median(dynamics),
                np.median(viscouses),
                np.median(armatures),
                np.median(stiffnesses),
                np.median(dampings),
                np.median(inertia_scales),
            ], dtype=np.float64)

        n_sa = args_cli.sa_steps
        T = args_cli.sa_t0
        sigma_vals = [float(x) for x in args_cli.sa_sigma.split(",")]
        if len(sigma_vals) == 1:
            sigma_vec = np.full(7, sigma_vals[0])
        elif len(sigma_vals) == 7:
            sigma_vec = np.array(sigma_vals)
        else:
            raise ValueError(f"--sa-sigma must be 1 or 7 comma-separated values, got {len(sigma_vals)}")

        print(f"[INFO] Simulated annealing: {n_sa} steps, T0={T:.4f}, alpha={args_cli.sa_alpha}, "
              f"sigma={sigma_vec.tolist()}, init={init_point.tolist()}, {num_envs} parallel chains")

        # All chains start at the same initial point
        cur_arr = np.tile(init_point, (num_envs, 1))  # shape (num_envs, 7)
        cur_cost_arr = np.full(num_envs, np.inf)

        # Initial evaluation (shared starting point — only one sim run needed)
        init_results = evaluate_batch([tuple(init_point)])
        if init_results[0][1] is None:
            print("[ERROR] Initial SA trial cut short.")
            results_file.close()
            return
        init_traj_np, init_metrics = init_results[0]
        init_cost = init_metrics[cost_key]
        cur_cost_arr[:] = init_cost
        writer.writerow([0, T, -1, True] + list(init_point) + [init_metrics[k] for k in metric_keys])
        results_file.flush()

        best_params = tuple(init_point)
        best_cost = init_cost
        best_metrics = init_metrics
        best_traj_np = init_traj_np

        sa_history = [(0, T, init_cost, init_cost)]

        print(f"[SA   0/{n_sa}] init  mu_s={init_point[0]:.4f}  mu_d={init_point[1]:.4f}  c_v={init_point[2]:.4f}  "
              f"arm={init_point[3]:.4f}  kp={init_point[4]:.4f}  kd={init_point[5]:.4f}  iscale={init_point[6]:.4f}  "
              f"cost={init_cost:.4f}")

        # ── live plot setup ───────────────────────────────────────────────────
        live_line_sim = live_title = None
        if args_cli.live_plot and _mpl_ok:
            try:
                plt.ion()
                live_fig, live_ax = plt.subplots(figsize=(10, 5))
                live_ax.plot(t_axis, pos_real, "k-", lw=2, label="real")
                live_ax.plot(t_axis, cmd_real, "k:", lw=1, alpha=0.4, label="cmd")
                live_line_sim, = live_ax.plot(t_axis, best_traj_np, "r-", lw=1.5, label="sim (best)")
                live_ax.set_xlabel("time (s)")
                live_ax.set_ylabel("joint position (rad)")
                live_ax.legend(fontsize=8)
                live_ax.grid(True, alpha=0.3)
                live_title = live_ax.set_title(
                    f"SA 0/{n_sa} | T={T:.5f} | {cost_key}={init_cost:.4f}")
                live_fig.tight_layout()
                live_fig.canvas.draw()
                plt.pause(0.005)
                print("[INFO] Live plot window opened.")
            except Exception as e:
                print(f"[WARN] Live plot unavailable: {e}")
                live_line_sim = live_title = None

        try:
            for i in range(1, n_sa + 1):
                if not simulation_app.is_running():
                    break
                # Each chain independently proposes one neighbour
                proposals = []
                for k in range(num_envs):
                    p = np.clip(cur_arr[k] + rng.normal(0.0, sigma_vec), lo, hi)
                    p[1] = min(p[1], p[0])  # mu_d <= mu_s
                    proposals.append(tuple(p))

                batch_results = evaluate_batch(proposals)

                # Each chain applies SA acceptance to its own proposal independently
                n_accepted = 0
                cut_short = False
                for k, (proposal, (sp, metrics)) in enumerate(zip(proposals, batch_results)):
                    if metrics is None:
                        print("[WARN] SA trial cut short, stopping.")
                        cut_short = True
                        break
                    prop_cost = metrics[cost_key]
                    delta = prop_cost - cur_cost_arr[k]
                    accept = delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12))
                    writer.writerow([i, T, k, accept] + list(proposal) + [metrics[mk] for mk in metric_keys])
                    if accept:
                        cur_arr[k] = np.array(proposal)
                        cur_cost_arr[k] = prop_cost
                        n_accepted += 1
                        if prop_cost < best_cost:
                            best_cost = prop_cost
                            best_params = tuple(proposal)
                            best_metrics = metrics
                            best_traj_np = sp

                if cut_short:
                    break

                results_file.flush()

                min_cur_cost = float(np.min(cur_cost_arr))
                sa_history.append((i, T, min_cur_cost, best_cost))
                T *= args_cli.sa_alpha

                elapsed = time.monotonic() - t_start
                best_p = np.array(best_params)
                print(f"[SA {i:3d}/{n_sa}] acc {n_accepted}/{num_envs}  "
                      f"mu_s={best_p[0]:.4f}  mu_d={best_p[1]:.4f}  c_v={best_p[2]:.4f}  "
                      f"arm={best_p[3]:.4f}  kp={best_p[4]:.4f}  kd={best_p[5]:.4f}  iscale={best_p[6]:.4f}  "
                      f"best={best_cost:.4f}  T={T:.5f}  [{elapsed:.0f}s]")

                if live_line_sim is not None:
                    live_line_sim.set_ydata(best_traj_np)
                    live_title.set_text(
                        f"SA {i}/{n_sa} | T={T:.5f} | {cost_key}={best_cost:.4f} (best)")
                    live_fig.canvas.draw_idle()
                    plt.pause(0.001)

        except (KeyboardInterrupt, SystemExit):
            print(f"\n[INFO] SA interrupted — generating plots from {len(all_results)} trials.")

        best_params = tuple(float(x) for x in best_params)

    # ── test-values mode ──────────────────────────────────────────────────────
    elif args_cli.mode == "test-values":
        if args_cli.test_values is None:
            raise ValueError("--test-values must be provided in test-values mode.")
        tv = [float(x) for x in args_cli.test_values.split(",")]
        if len(tv) != 7:
            raise ValueError(f"--test-values must have exactly 7 values, got {len(tv)}")
        params = [tuple(tv)]

        cmd_scales = [
            ("original (1×)", 1.0,  "#1f77b4"),
            ("half (0.5×)",   0.5,  "#ff7f0e"),
            ("1.5× scale",    1.5,  "#2ca02c"),
        ]

        # Run simulation for each command scale
        scale_results = []  # (label, scale, color, scaled_cmd_np, sim_traj_np)
        for label, scale, color in cmd_scales:
            scaled_cmd = cmd_real * scale
            scaled_tensor = torch.tensor(scaled_cmd, dtype=torch.float32, device=device)
            sim_pos_batch = run_batch(
                robot, sim, active_idx, active_actuator, active_local_idx,
                scaled_tensor, joint_offset, dt, params, num_envs, device,
                default_inertias, active_body_idx)
            traj = sim_pos_batch[0].cpu().numpy()
            scale_results.append((label, scale, color, scaled_cmd, traj))
            print(f"[INFO] {label}: simulated {min(len(traj), n_steps)} steps")

        if _mpl_ok:
            # 1. Combined overlay: all three traces on one figure
            fig, (ax_pos, ax_cmd) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                                  gridspec_kw={"height_ratios": [3, 1]})
            ax_pos.plot(t_axis, pos_real, "k-", lw=2.0, label="real (original cmd)", zorder=5)
            for label, scale, color, scmd, traj in scale_results:
                T_len = min(len(traj), n_steps)
                ax_pos.plot(t_axis[:T_len], traj[:T_len], lw=1.8, color=color,
                            label=f"sim — {label}")
                ax_cmd.plot(t_axis, scmd, lw=1.2, color=color, ls="--", label=f"cmd — {label}")
            add_phase_lines(ax_pos)
            add_phase_lines(ax_cmd)
            ax_pos.set_ylabel("joint position (rad)")
            ax_pos.set_title(
                f"Test values — {target_joint_name}\n"
                f"μs={tv[0]}  μd={tv[1]}  cv={tv[2]}  arm={tv[3]}  kp={tv[4]}  kd={tv[5]}  iscale={tv[6]}"
            )
            ax_pos.legend(fontsize=8)
            ax_pos.grid(True, alpha=0.3)
            ax_cmd.set_xlabel("time (s)")
            ax_cmd.set_ylabel("command (rad)")
            ax_cmd.legend(fontsize=8)
            ax_cmd.grid(True, alpha=0.3)
            fig.tight_layout()
            fname = f"test_values_overlay_{joint_role}.png"
            fig.savefig(os.path.join(plot_dir, fname), dpi=120)
            plt.close(fig)
            print(f"[INFO] Saved {fname}")

            # 2. Individual plot per command scale
            for label, scale, color, scmd, traj in scale_results:
                T_len = min(len(traj), n_steps)
                fig2, (ax2_top, ax2_bot) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                                         gridspec_kw={"height_ratios": [3, 1]})
                if scale == 1.0:
                    ax2_top.plot(t_axis, pos_real, "k-", lw=2.0, label="real")
                ax2_top.plot(t_axis, scmd, "k:", lw=1.0, alpha=0.4, label="command")
                ax2_top.plot(t_axis[:T_len], traj[:T_len], color=color, lw=1.8, label="sim")
                add_phase_lines(ax2_top)
                ax2_top.set_ylabel("joint position (rad)")
                ax2_top.set_title(
                    f"Test values ({label}) — {target_joint_name}\n"
                    f"μs={tv[0]}  μd={tv[1]}  cv={tv[2]}  arm={tv[3]}  kp={tv[4]}  kd={tv[5]}  iscale={tv[6]}"
                )
                ax2_top.legend(fontsize=8)
                ax2_top.grid(True, alpha=0.3)

                if scale == 1.0:
                    err = traj[:T_len] - pos_real[:T_len]
                    ax2_bot.plot(t_axis[:T_len], err, color="darkorange", lw=1.0, label="sim − real")
                    ax2_bot.axhline(0, color="k", lw=0.6)
                    ax2_bot.set_ylabel("error (rad)")
                else:
                    ax2_bot.plot(t_axis, scmd, color=color, lw=1.2, label="command")
                    ax2_bot.set_ylabel("command (rad)")
                add_phase_lines(ax2_bot)
                ax2_bot.set_xlabel("time (s)")
                ax2_bot.legend(fontsize=8)
                ax2_bot.grid(True, alpha=0.3)

                fig2.tight_layout()
                if scale == 1.0:
                    scale_str = "orig"
                elif scale < 1.0:
                    scale_str = "half"
                else:
                    scale_str = "1p5x"
                fname2 = f"test_values_{scale_str}_{joint_role}.png"
                fig2.savefig(os.path.join(plot_dir, fname2), dpi=120)
                plt.close(fig2)
                print(f"[INFO] Saved {fname2}")

        total_elapsed = time.monotonic() - t_start
        print(f"\n[INFO] Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        return

    results_file.close()
    print(f"[INFO] Results written to {csv_out_path}")

    # ── plots ─────────────────────────────────────────────────────────────────
    if not _mpl_ok:
        print("[WARN] matplotlib not available — skipping plots.")
        return

    sorted_by_step = sorted(all_results, key=lambda x: x[2][cost_key])

    # ── 1. Top-5 trajectory overlays ─────────────────────────────────────────
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    _plot_metric_info = [
        ("rmse_total",   "total RMSE",  "total"),
        ("rmse_step",    "step RMSE",   "step"),
        ("max_err_total","max error",   "max"),
    ]
    if args_cli.mode == "grid":
        # One plot per metric
        for mk, mk_label, mk_suffix in _plot_metric_info:
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(t_axis, pos_real, "k-", lw=2.0, label="real", zorder=5)
            ax.plot(t_axis, cmd_real, "k:", lw=1.0, alpha=0.45, label="command")
            add_phase_lines(ax)
            for rank, (params, trace, metrics) in enumerate(top_results[mk][:5]):
                ax.plot(t_axis, trace, lw=1.2, alpha=0.85, color=colors[rank],
                        label=(f"#{rank+1}  μs={params[0]:.2g}  μd={params[1]:.2g}  "
                               f"cv={params[2]:.2g}  arm={params[3]:.2g}  "
                               f"kp={params[4]:.2g}  kd={params[5]:.2g}  iscale={params[6]:.2g}  "
                               f"{mk}={metrics[mk]:.3f}"))
            ax.set_xlabel("time (s)")
            ax.set_ylabel("joint position (rad)")
            ax.set_title(f"Top-5 by {mk_label} — {target_joint_name}")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fname = f"top5_{mk_suffix}_{joint_role}.png"
            fig.savefig(os.path.join(plot_dir, fname), dpi=120)
            plt.close(fig)
            print(f"[INFO] Saved {fname}")
    else:
        # SA mode: single top-5 by selected metric
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(t_axis, pos_real, "k-", lw=2.0, label="real", zorder=5)
        ax.plot(t_axis, cmd_real, "k:", lw=1.0, alpha=0.45, label="command")
        add_phase_lines(ax)
        for rank, (params, trace, metrics) in enumerate(sorted_by_step[:5]):
            ax.plot(t_axis, trace, lw=1.2, alpha=0.85, color=colors[rank],
                    label=(f"#{rank+1}  μs={params[0]:.2g}  μd={params[1]:.2g}  "
                           f"cv={params[2]:.2g}  arm={params[3]:.2g}  "
                           f"kp={params[4]:.2g}  kd={params[5]:.2g}  iscale={params[6]:.2g}  "
                           f"{cost_key}={metrics[cost_key]:.3f}"))
        ax.set_xlabel("time (s)")
        ax.set_ylabel("joint position (rad)")
        ax.set_title(f"Top-5 friction combinations — {target_joint_name}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"top5_{joint_role}.png"), dpi=120)
        plt.close(fig)
        print(f"[INFO] Saved top5_{joint_role}.png")

    # ── 2. RMSE-step heatmap — grid mode only ────────────────────────────────
    if args_cli.mode == "grid":
        rmse_grid = np.full((len(statics), len(dynamics)), np.inf)
        best_cv_grid = np.zeros_like(rmse_grid)
        for params, metrics in all_param_metrics:
            i = statics.index(params[0])
            j = dynamics.index(params[1])
            if metrics[cost_key] < rmse_grid[i, j]:
                rmse_grid[i, j] = metrics[cost_key]
                best_cv_grid[i, j] = params[2]

        fig, ax = plt.subplots(figsize=(max(6, len(dynamics) * 1.3), max(5, len(statics) * 1.1)))
        im = ax.imshow(rmse_grid, origin="lower", aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(dynamics)))
        ax.set_xticklabels([f"{v:.3g}" for v in dynamics])
        ax.set_yticks(range(len(statics)))
        ax.set_yticklabels([f"{v:.3g}" for v in statics])
        ax.set_xlabel("dynamic friction μd")
        ax.set_ylabel("static friction μs")
        ax.set_title(f"{cost_key} (best viscous shown) — {target_joint_name}")
        for i in range(len(statics)):
            for j in range(len(dynamics)):
                cell_color = "w" if rmse_grid[i, j] < (rmse_grid.max() + rmse_grid.min()) / 2 else "k"
                ax.text(j, i,
                        f"{rmse_grid[i, j]:.3f}\ncv={best_cv_grid[i,j]:.2g}",
                        ha="center", va="center", color=cell_color, fontsize=7)
        fig.colorbar(im, ax=ax, label="RMSE (rad)")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"heatmap_{joint_role}.png"), dpi=120)
        plt.close(fig)
        print(f"[INFO] Saved heatmap_{joint_role}.png")

    # ── 3. Best-fit detailed trace ────────────────────────────────────────────
    if args_cli.mode == "grid":
        best_params, best_trace, best_metrics = sorted_by_step[0]
    else:
        _, best_trace, best_metrics = sorted_by_step[0]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax_pos, ax_err = axes

    ax_pos.plot(t_axis, pos_real, "k-", lw=2.0, label="real")
    ax_pos.plot(t_axis, cmd_real, "k:", lw=1.0, alpha=0.4, label="command")
    ax_pos.plot(t_axis, best_trace, "r-", lw=1.5,
                label=(f"sim  μs={best_params[0]:.3g}  μd={best_params[1]:.3g}  "
                       f"cv={best_params[2]:.3g}  arm={best_params[3]:.3g}  "
                       f"kp={best_params[4]:.3g}  kd={best_params[5]:.3g}  "
                       f"iscale={best_params[6]:.3g}"))
    add_phase_lines(ax_pos)
    ax_pos.set_ylabel("position (rad)")
    ax_pos.set_title(f"Best fit — {target_joint_name}  "
                     f"(rmse_step={best_metrics['rmse_step']:.4f}  "
                     f"rmse_total={best_metrics['rmse_total']:.4f})")
    ax_pos.legend(fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    err_trace = best_trace - pos_real
    ax_err.plot(t_axis, err_trace, color="darkorange", lw=1.0, label="sim − real")
    ax_err.axhline(0, color="k", lw=0.6)
    add_phase_lines(ax_err)
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("error (rad)")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"best_fit_{joint_role}.png"), dpi=120)
    plt.close(fig)
    print(f"[INFO] Saved best_fit_{joint_role}.png")

    # ── 4. SA convergence plot ───────────────────────────────────────────────
    if args_cli.mode == "anneal" and len(sa_history) > 1:
        iters, temps, cur_costs, best_costs = zip(*sa_history)
        fig, (ax_cost, ax_temp) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                               gridspec_kw={"height_ratios": [3, 1]})
        ax_cost.plot(iters, cur_costs, color="steelblue", lw=1.0, alpha=0.7, label="min chain RMSE")
        ax_cost.plot(iters, best_costs, color="crimson", lw=1.5, label="best RMSE")
        ax_cost.set_ylabel("step-phase RMSE (rad)")
        ax_cost.set_title(f"SA convergence — {target_joint_name}")
        ax_cost.legend(fontsize=8)
        ax_cost.grid(True, alpha=0.3)
        ax_temp.plot(iters, temps, color="darkorange", lw=1.0)
        ax_temp.set_xlabel("SA iteration")
        ax_temp.set_ylabel("temperature")
        ax_temp.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"sa_convergence_{joint_role}.png"), dpi=120)
        plt.close(fig)
        print(f"[INFO] Saved sa_convergence_{joint_role}.png")

    # ── grid-only: viscous-friction sensitivity for best (static, dynamic) ──
    best_s, best_d = best_params[0], best_params[1]
    visc_results = [(p[2], m["rmse_step"])
                    for p, m in all_param_metrics if p[0] == best_s and p[1] == best_d]
    if args_cli.mode == "grid" and len(visc_results) > 1:
        visc_cv, visc_rmse = zip(*sorted(visc_results))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(visc_cv, visc_rmse, "o-", color="steelblue")
        ax.set_xlabel("viscous friction cv")
        ax.set_ylabel("RMSE step phase (rad)")
        ax.set_title(f"Viscous sweep at μs={best_s:.3g} μd={best_d:.3g} — {target_joint_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"viscous_sweep_{joint_role}.png"), dpi=120)
        plt.close(fig)
        print(f"[INFO] Saved viscous_sweep_{joint_role}.png")

    total_elapsed = time.monotonic() - t_start
    print(f"\n[INFO] Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"\n[INFO] Best combination by {args_cli.metric} RMSE:")
    print(f"  static_friction  = {best_params[0]}")
    print(f"  dynamic_friction = {best_params[1]}")
    print(f"  viscous_friction = {best_params[2]}")
    print(f"  armature         = {best_params[3]}")
    print(f"  stiffness (kp)   = {best_params[4]}")
    print(f"  damping   (kd)   = {best_params[5]}")
    print(f"  inertia_scale    = {best_params[6]}")
    print(f"  rmse_step  = {best_metrics['rmse_step']:.4f}")
    print(f"  rmse_total = {best_metrics['rmse_total']:.4f}")
    if args_cli.mode == "grid":
        print(f"\n[INFO] Swept values:")
        print(f"  static_friction:  {statics}")
        print(f"  dynamic_friction: {dynamics}")
        print(f"  viscous_friction: {viscouses}")
        print(f"  armature:         {armatures}")
        print(f"  stiffness (kp):   {stiffnesses}")
        print(f"  damping   (kd):   {dampings}")
        print(f"  inertia_scale:    {inertia_scales}")


if __name__ == "__main__":
    main()
    simulation_app.close()
