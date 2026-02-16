from grid2op.Reward import BaseReward


class MaxCurrentReward(BaseReward):

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = -2.0

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        obs = env.get_obs(_do_copy=False)
        return 1.0 - obs.rho.max()
