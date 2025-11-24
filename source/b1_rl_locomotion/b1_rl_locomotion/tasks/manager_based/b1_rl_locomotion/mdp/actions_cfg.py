
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions import JointPositionActionCfg

from . import actions

from .actions import MirroredJointPositionAction

@configclass
class MirroredJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = actions.MirroredJointPositionAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """
