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
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # Build a name → id dictionary
        self.name_to_id = {name: idx for idx, name in enumerate(self._joint_names)}

        # Define left→right mappings
        self.pairs = [
            ("FL_hip_joint", "FR_hip_joint"),
            ("FL_thigh_joint", "FR_thigh_joint"),
            ("FL_calf_joint", "FR_calf_joint"),
            ("RL_hip_joint", "RR_hip_joint"),
            ("RL_thigh_joint", "RR_thigh_joint"),
            ("RL_calf_joint", "RR_calf_joint"),
        ]

        # Convert names → indices
        self.pairs_idx = [
            (self.name_to_id[src], self.name_to_id[tgt]) for src, tgt in self.pairs
        ]

    def apply_actions(self):

        # Start with a copy (so FL/RL keep their own values)
        mirrored = self.processed_actions.clone()

        # Apply pair-based mirroring
        for src_idx, tgt_idx in self.pairs_idx:
            sign = -1.0 if "hip" in self._joint_names[tgt_idx] else 1.0
            mirrored[:, tgt_idx] = sign * self.processed_actions[:, src_idx]

        # Send to simulator
        self._asset.set_joint_position_target(mirrored, joint_ids=self._joint_ids)


# class MirroredJointPositionAction(JointPositionAction):

#     def __init__(self, cfg, env):
#         super().__init__(cfg, env)

#         # map name → index
#         self.name_to_id = {name: idx for idx, name in enumerate(self._joint_names)}

#         # define left leg sources
#         self.src = {
#             "hip":   self.name_to_id["FL_hip_joint"],
#             "thigh": self.name_to_id["FL_thigh_joint"],
#             "calf":  self.name_to_id["FL_calf_joint"],
#         }

#         # define targets for each leg
#         self.targets = {
#             "FR": +1,   # thigh/calf same sign
#             "RL": +1,
#             "RR": +1
#         }

#         self.hip_sign = {
#             "FR": -1,
#             "RL": +1,
#             "RR": -1,
#         }

#         self.groups = {
#             "hip":   ["FR_hip_joint", "RL_hip_joint", "RR_hip_joint"],
#             "thigh": ["FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
#             "calf":  ["FR_calf_joint", "RL_calf_joint", "RR_calf_joint"],
#         }

#     def apply_actions(self):

#         mirrored = torch.zeros_like(self.processed_actions)

#         # copy FL → FL
#         for j in ["hip", "thigh", "calf"]:
#             fl_idx = self.src[j]
#             mirrored[:, fl_idx] = self.processed_actions[:, fl_idx]

#         # mirror remaining legs
#         for joint_type, names in self.groups.items():
#             src_idx = self.src[joint_type]

#             for name in names:
#                 tgt_idx = self.name_to_id[name]
#                 leg = name.split("_")[0]  # "FR", "RL", "RR"

#                 if joint_type == "hip":
#                     sign = self.hip_sign[leg]
#                 else:
#                     sign = +1

#                 mirrored[:, tgt_idx] = sign * self.processed_actions[:, src_idx]

#         self._asset.set_joint_position_target(mirrored, joint_ids=self._joint_ids)
