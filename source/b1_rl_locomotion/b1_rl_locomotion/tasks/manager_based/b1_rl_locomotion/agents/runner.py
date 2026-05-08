"""skrl Runner override that wires :class:`MaskedPPO` in place of stock PPO.

The stock skrl Runner._component is a hardcoded if/elif on lowercase class
names — there's no registry to inject into. So we subclass and intercept
``"ppo"`` only; everything else (including ``"ppo_default_config"``) falls
through to the parent.
"""

from __future__ import annotations

from typing import Type

from skrl.utils.runner.torch import Runner

from .masked_ppo import MaskedPPO


class B1Runner(Runner):
    def _component(self, name: str) -> Type:
        if name.lower() == "ppo":
            return MaskedPPO
        return super()._component(name)
