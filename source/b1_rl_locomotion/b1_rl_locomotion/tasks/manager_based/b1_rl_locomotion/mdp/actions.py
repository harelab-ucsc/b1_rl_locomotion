from __future__ import annotations

import logging
import torch
from collections.abc import Sequence


import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class MirroredJointPositionAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # map name → index
        self.name_to_id = {name: idx for idx, name in enumerate(self._joint_names)}

        # define left leg sources
        self.src = {
            "hip": self.name_to_id["FL_hip_joint"],
            "thigh": self.name_to_id["FL_thigh_joint"],
            "calf": self.name_to_id["FL_calf_joint"],
        }

        # define targets for each leg
        self.targets = {
            "FR": +1,   # thigh/calf same sign
            "RL": +1,
            "RR": +1
        }

        self.hip_sign = {
            "FR": -1,
            "RL": +1,
            "RR": -1,
        }

        self.groups = {
            "hip": ["FR_hip_joint", "RL_hip_joint", "RR_hip_joint"],
            "thigh": ["FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
            "calf": ["FR_calf_joint", "RL_calf_joint", "RR_calf_joint"],
        }

    def apply_actions(self):

        mirrored = torch.zeros_like(self.processed_actions)

        # copy FL → FL
        for j in ["hip", "thigh", "calf"]:
            fl_idx = self.src[j]
            mirrored[:, fl_idx] = self.processed_actions[:, fl_idx]

        # mirror remaining legs
        for joint_type, names in self.groups.items():
            src_idx = self.src[joint_type]

            for name in names:
                tgt_idx = self.name_to_id[name]
                leg = name.split("_")[0]  # "FR", "RL", "RR"

                if joint_type == "hip":
                    sign = self.hip_sign[leg]
                else:
                    sign = +1

                mirrored[:, tgt_idx] = sign * self.processed_actions[:, src_idx]

        self._asset.set_joint_position_target(mirrored, joint_ids=self._joint_ids)

