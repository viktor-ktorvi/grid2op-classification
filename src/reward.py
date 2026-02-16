from logging import Logger
from typing import Optional

import numpy as np
from grid2op import Action, Environment
from grid2op.dtypes import dt_float
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


class OverloadReward(BaseReward):
    def __init__(self, logger: Optional[Logger] = None, constrained: bool = False):
        super().__init__(logger=logger)
        self.penalty: float = dt_float(-1.0 if not constrained else 0.0)
        self.min_reward: float = dt_float(-5.0)

    def __call__(
        self, action: np.ndarray, env: Environment, has_error: bool, is_done: bool, is_illegal: bool, is_ambiguous: bool
    ) -> float:
        if has_error or is_illegal or is_ambiguous:
            return self.min_reward

        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        margin = np.divide(thermal_limits - ampere_flows, thermal_limits + 1e-10)
        penalty_disconnection = self.penalty * sum(~env.current_obs.line_status) / (env.current_obs.n_line * 0.1)
        penalty_overload = margin[margin < 0].sum() / (env.current_obs.n_line * 0.1)

        reward: float = penalty_overload + penalty_disconnection
        return reward
