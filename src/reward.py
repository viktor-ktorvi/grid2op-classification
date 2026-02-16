from logging import Logger

from grid2op import Action, Environment
from grid2op.Reward import BaseReward


class MaxCurrentReward(BaseReward):

    def __init__(self, logger: Logger | None = None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = -2.0

    def __call__(
        self, action: Action, env: Environment, has_error: bool, is_done: bool, is_illegal: bool, is_ambiguous: bool
    ) -> float:
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        obs = env.get_obs(_do_copy=False)
        return 1.0 - float(obs.rho.max())
