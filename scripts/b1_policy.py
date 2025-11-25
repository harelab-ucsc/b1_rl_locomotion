import os

from typing import Optional

import numpy as np

from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController


class B1StandPolicy(PolicyController):
    """The Spot quadruped"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "b1",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """
        if usd_path == None:
            # Path to current config's directory (/home/.../b1_rl_locomotion/config)
            BASE_DIR = os.path.dirname(__file__)
            ASSET_DIR = os.path.join(BASE_DIR, "..", "source/b1_rl_locomotion/b1_rl_locomotion/tasks/manager_based/b1_rl_locomotion/assets")  # /home/.../b1_rl_locomotion/assets
            usd_path = os.path.join(ASSET_DIR, "b1.usd")  # /home/.../b1_rl_locomotion/assets/b1.usd

            # usd_path = assets_root_path + "/Isaac/Robots/BostonDynamics/spot/spot.usd"

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(
            "/../logs/skrl/stand_sit/2025-11-20_00-49-35_ppo_torch_run_8" + "/checkpoints/best_agent.pt",
            "/../logs/skrl/stand_sit/2025-11-20_00-49-35_ppo_torch_run_8" + "/params/env.yaml",
        )
        self._previous_action = np.zeros(12)
        self._policy_counter = 0

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        current_joint_pos = self.robot.get_joint_positions()

        """
        [INFO] Observation Manager: <ObservationManager> contains 1 groups.
        +------------------------------------------------------+
        | Active Observation Terms in Group: 'policy' (shape: (44,)) |
        +------------+----------------------------+------------+
        |   Index    | Name                       |   Shape    |
        +------------+----------------------------+------------+
        |     0      | joint_pos_rel              |   (12,)    |
        |     1      | joint_vel_rel              |   (12,)    |
        |     2      | base_height                |    (1,)    |
        |     3      | height_cmd                 |    (7,)    |
        |     4      | actions                    |   (12,)    |
        +------------------------------------------------------+

        joint_pos_rel,
        joint_vel_rel,
        base_height,
        height_cmd,
        actions (last action)
        """
        obs = np.zeros(44)
        # joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[:12] = current_joint_pos
        obs[12:24] = current_joint_vel
        # base height
        obs[24:25] = self.get_height()
        # command
        obs[25:32] = command
        # previous action
        obs[32:44] = self._previous_action

        return obs

    def stand(self, command):
        """
        Docstring for stand
        
        :param self: Description
        :param command: Description
        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()
        
        action = ArticulationAction(joint_positions=self.default_pos + (self.action))
        self.robot.apply_action(action)

        self._policy_counter += 1

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (p_x, p_y, p_z, qw, qx, qy, qz)
        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        action = ArticulationAction(joint_positions=self.default_pos + (self.action))
        self.robot.apply_action(action)

        self._policy_counter += 1
