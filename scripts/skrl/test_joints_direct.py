"""Minimal joint-position test for the B1 robot.

Bypasses gym, the action manager, the policy, observation manager, and any
domain randomization. Spawns one B1 in a scene with gravity disabled and
directly commands joint position targets through the articulation API.

Each joint is exercised in turn for `HOLD_SECONDS`, commanded to
`default_joint_pos + OFFSET_RAD`, then back to default. All other joints
are held at default the entire time.

Joint positions are logged every step to `recorded_data_direct.csv` along
with the commanded target. The header uses the *real* joint name from the
articulation, so you don't have to guess at index ordering.

Usage (from inside the IsaacLab Python env, same way you launch play_whatever.py):
    python test_joints_direct.py
    python test_joints_direct.py --offset 0.5 --hold 3.0
    python test_joints_direct.py --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Direct joint PD test for B1.")
parser.add_argument("--offset", type=float, default=0.3,
                    help="Position offset (radians) applied to the active joint, relative to default.")
parser.add_argument("--hold", type=float, default=2.0,
                    help="Seconds to hold each commanded position before moving on.")
parser.add_argument("--joints", type=str, default=None,
                    help="Comma-separated joint indices to test. Default: all 12.")
parser.add_argument("--csv", type=str, default="recorded_data_direct.csv",
                    help="Output CSV path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force GPU/CPU sim choice through AppLauncher exactly like the play script does.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Everything below runs only after the sim app is up. -----------------------

import csv
import os
import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from b1_rl_locomotion.tasks.manager_based.b1_rl_locomotion.configs.b1 import B1_CFG  # noqa: E402


def main():
    # ------------------------------------------------------------------ sim
    # dt = 1/200 s is the IsaacLab default for locomotion; match your env if it differs.
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 200.0, device=args_cli.device or "cuda:0",
                                      gravity=(0.0, 0.0, 0.0))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 1.0], target=[0.0, 0.0, 0.5])

    # Flat ground so the robot has something to rest against if gravity gets re-enabled later.
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # A single dome light so we can see what's going on in non-headless mode.
    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light)

    # ------------------------------------------------------------------ robot
    # Use the *exact* config from your project. The B1_CFG already has
    # disable_gravity=True on the rigid bodies, so the robot will float.
    robot_cfg = B1_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot = Articulation(robot_cfg)

    sim.reset()  # this is where actuators and PD controllers get instantiated
    print("[INFO] Sim reset complete.")

    # ------------------------------------------------------------------ inspect
    # This is the part you can't easily see from inside the gym env: the *actual*
    # joint order PhysX is using, and the *actual* default positions.
    joint_names = robot.data.joint_names
    default_qpos = robot.data.default_joint_pos.clone()  # shape (1, num_joints)
    num_joints = len(joint_names)

    print("\n========== Articulation introspection ==========")
    print(f"num_joints = {num_joints}")
    for i, name in enumerate(joint_names):
        print(f"  [{i:2d}] {name:25s}  default = {default_qpos[0, i].item():+.4f} rad "
              f"({default_qpos[0, i].item() * 180.0 / 3.14159265:+.2f} deg)")

    # Also dump the actuator stiffness/damping/friction so we know exactly
    # what's being applied. These come straight from the articulation, not
    # the cfg — so if something's getting overridden, we see the truth.
    print("\n========== Actuator gains (from articulation) ==========")
    print(f"stiffness:        {robot.data.joint_stiffness[0].cpu().numpy()}")
    print(f"damping:          {robot.data.joint_damping[0].cpu().numpy()}")
    try:
        print(f"joint_friction:   {robot.data.joint_friction[0].cpu().numpy()}")
    except AttributeError:
        # Older IsaacLab versions expose it differently.
        pass
    try:
        print(f"effort_limit:     {robot.data.joint_effort_limits[0].cpu().numpy()}")
    except AttributeError:
        pass
    try:
        print(f"velocity_limit:   {robot.data.joint_velocity_limits[0].cpu().numpy()}")
    except AttributeError:
        pass
    print("=================================================\n")

    # Decide which joints to test
    if args_cli.joints is None:
        joints_to_test = list(range(num_joints))
    else:
        joints_to_test = [int(x) for x in args_cli.joints.split(",")]

    # ------------------------------------------------------------------ csv
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args_cli.csv)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    header = ["sim_time", "active_joint_idx", "active_joint_name"]
    header += [f"qpos__{n}" for n in joint_names]
    header += [f"target__{n}" for n in joint_names]
    writer.writerow(header)
    print(f"[INFO] Logging to {csv_path}")

    # ------------------------------------------------------------------ test loop
    dt = sim.get_physics_dt()
    steps_per_hold = max(1, int(args_cli.hold / dt))
    print(f"[INFO] dt = {dt:.5f} s, holding each command for {steps_per_hold} steps "
          f"(~{args_cli.hold} s)")

    sim_time = 0.0

    # Helper to drive the articulation to a given target tensor for N steps.
    def hold_target(target_qpos: torch.Tensor, n_steps: int, active_idx: int):
        nonlocal sim_time
        for _ in range(n_steps):
            robot.set_joint_position_target(target_qpos)
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)

            qpos = robot.data.joint_pos[0].cpu().tolist()
            tgt = target_qpos[0].cpu().tolist()
            writer.writerow([sim_time, active_idx, joint_names[active_idx] if active_idx >= 0 else ""]
                            + qpos + tgt)
            sim_time += dt

            if not simulation_app.is_running():
                return False
        return True

    # 1) Settle at default for one hold period so the initial transient doesn't
    #    contaminate the first joint's plot.
    print("[INFO] Settling at default pose...")
    target = default_qpos.clone()
    if not hold_target(target, steps_per_hold, active_idx=-1):
        csv_file.close()
        return

    # 2) For each joint: offset by +OFFSET, then back to default.
    for j in joints_to_test:
        name = joint_names[j]
        print(f"[INFO] Testing joint [{j:2d}] {name}: default {default_qpos[0, j].item():+.3f} "
              f"-> {default_qpos[0, j].item() + args_cli.offset:+.3f}")

        # Command: default + offset on joint j
        target = default_qpos.clone()
        target[0, j] = default_qpos[0, j] + args_cli.offset
        if not hold_target(target, steps_per_hold, active_idx=j):
            break

        # Return to default before moving to the next joint, so we can see
        # whether the joint actually comes back (and whether commanding j
        # disturbs other joints during the return).
        target = default_qpos.clone()
        if not hold_target(target, steps_per_hold, active_idx=j):
            break

    csv_file.close()
    print(f"[INFO] Done. Data written to {csv_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
