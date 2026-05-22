import torch
from typing import Sequence

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class B1RlLocomotionEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with a per-env settle period after each reset.

    For the first `cfg.num_reset_settle_steps` steps after an env resets, that
    env's action is overridden with a settle action (which defaults to zeros,
    holding the robot at its default pose when use_default_offset=True). If
    cfg.settle_joint_pos is set, the settle action targets those absolute joint
    positions instead (useful when the robot resets into a non-default pose and
    PD should hold that pose during settling). Each step's transition validity
    is exposed in `extras["valid"]` (False during settle); a mask-aware agent
    (see `MaskedPPO`) zeros out gradient contributions and preprocessor updates
    from invalid transitions so settle steps don't influence learning at all.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._settle_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._settle_action = self._build_settle_action()

    def _build_settle_action(self) -> torch.Tensor:
        """Build a [1, action_dim] settle action tensor from cfg.settle_joint_pos.

        Returns zeros (= hold at default pose) if settle_joint_pos is not set.
        For each joint listed in settle_joint_pos, computes the raw action offset
        needed for JointPositionAction (with use_default_offset=True, scale=1) to
        target that absolute joint position: offset = target - default_joint_pos.
        """
        settle = torch.zeros(self.action_manager.total_action_dim, device=self.device)
        settle_joint_pos = getattr(self.cfg, "settle_joint_pos", None)
        if not settle_joint_pos:
            return settle

        robot = self.scene["robot"]
        default_pos = robot.data.default_joint_pos[0]  # [num_robot_joints]

        action_offset = 0
        for term in self.action_manager._terms.values():
            if not hasattr(term, "_joint_ids"):
                action_offset += term.action_dim
                continue

            joint_ids = term._joint_ids
            if isinstance(joint_ids, slice):
                joint_ids = list(range(robot.num_joints))

            robot_to_action_col = {rjid: action_offset + aidx for aidx, rjid in enumerate(joint_ids)}

            for pattern, target_pos in settle_joint_pos.items():
                matched_ids, _ = robot.find_joints(pattern)
                for rjid in matched_ids:
                    if rjid in robot_to_action_col:
                        col = robot_to_action_col[rjid]
                        settle[col] = target_pos - default_pos[rjid].item()

            action_offset += term.action_dim

        return settle

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self._settle_remaining[env_ids] = self.cfg.num_reset_settle_steps

    def step(self, action: torch.Tensor):
        n = action.shape[0]
        settling = self._settle_remaining[:n] > 0
        if settling.any():
            action = action.clone()
            action[settling] = self._settle_action.to(action.device)
            self._settle_remaining[:n] -= settling.long()

        obs, reward, terminated, truncated, extras = super().step(action)
        extras["valid"] = (~settling).view(-1, 1)
        return obs, reward, terminated, truncated, extras
