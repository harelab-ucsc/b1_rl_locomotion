"""Functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lerp_reward_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    w0,
    w1: float,
    t0: int,
    t1: int,
):
    """Curriculum that linearly interpolates a reward term's weight over a specified step range.

    The weight of the reward term transitions from an initial value ``w0`` to a final value ``w1``
    between steps ``t0`` and ``t1``. Before ``t0``, the weight remains at ``w0``. After ``t1``,
    the weight remains at ``w1``. If ``t1 <= t0``, the weight jumps instantly to ``w1`` at
    ``t0``. The interpolation is global across all environments; ``env_ids`` are ignored.

    Args:
        env (ManagerBasedRLEnv): The learning environment.
        env_ids (Sequence[int]): Not used; kept for API compatibility.
        term_name (str): The name of the reward term to modify.
        w0 (float): Initial weight of the reward term.
        w1 (float): Final weight of the reward term.
        t0 (int): Step at which interpolation begins.
        t1 (int): Step at which interpolation ends. If ``t1 <= t0``, the change is instantaneous.

    Returns:
        None
    """
    # sanity check
    if t0 < 0 or t1 < 0:
        return

    curr = int(env.common_step_counter)

    # handle degenerate window (instant jump at/after start_step)
    if t1 <= t0:
        new_w = w1 if curr >= t0 else w0
    else:
        # alpha in [0, 1]
        num = curr - t0
        den = t1 - t0
        alpha = num / den
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0
        new_w = (1.0 - alpha) * float(w0) + alpha * float(w1)

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    if getattr(term_cfg, "weight", None) == new_w:
        return  # no-op
    term_cfg.weight = new_w
    env.reward_manager.set_term_cfg(term_name, term_cfg)
