import random
from typing import Callable

import grid2op
import matplotlib
from grid2op.Agent import DoNothingAgent, TopologyGreedy
from grid2op.Reward import (  # noqa: F401
    AlarmReward,
    AlertReward,
    BaseReward,
    BridgeReward,
    CloseToOverflowReward,
    ConstantReward,
    DistanceReward,
    EconomicReward,
    EpisodeDurationReward,
    FlatReward,
    GameplayReward,
    IncreasingFlatReward,
    L2RPNReward,
    LinesCapacityReward,
    LinesReconnectedReward,
    N1Reward,
    RedispReward,
)
from lightsim2grid import LightSimBackend
from matplotlib import pyplot as plt

from src.reward import MaxCurrentReward


def simulate(
    agent_name: str,
    env_name: str,
    MAX_STEPS: int,
    reward_class: Callable = LinesCapacityReward,
) -> list[float]:
    env = grid2op.make(env_name, reward_class=reward_class, backend=LightSimBackend())

    max_timeseries_id = len(env.chronics_handler.subpaths) - 1
    if agent_name == "do nothing":
        agent = DoNothingAgent(env.action_space)
    elif agent_name == "topology greedy":
        agent = TopologyGreedy(env.action_space)
    else:
        raise ValueError(f"{agent_name=}")

    obs = env.reset(
        options={
            "time serie id": random.choice(range(max_timeseries_id)),
            "max step": MAX_STEPS,
        }
    )
    reward = env.reward_range[0]
    done = False

    max_currents = []
    step_counter = 0
    while not done:
        max_current = obs.rho.max()
        action = agent.act(obs, reward, done)

        obs, reward, done, info = env.step(action)

        max_currents.append(max_current)
        print(f"{agent_name=}, {step_counter=}, {max_current=:2.3f}")
        step_counter += 1

    return max_currents


def main() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 18,
            "figure.autolayout": True,
        }
    )
    RANDOM_SEED = 0
    MAX_STEPS = 200
    random.seed(RANDOM_SEED)
    env_name = "l2rpn_case14_sandbox"
    # env_name = "l2rpn_icaps_2021_small"
    # env_name = "l2rpn_idf_2023"

    do_nothing_currents = simulate(agent_name="do nothing", env_name=env_name, MAX_STEPS=MAX_STEPS)

    fig, axs = plt.subplots(1, 1)
    axs.plot(do_nothing_currents, label="do nothing", linewidth=3)

    for reward_class in [
        LinesCapacityReward,
        MaxCurrentReward,
        # # AlarmReward,
        # AlertReward,
        # # BridgeReward,
        # CloseToOverflowReward,
        # ConstantReward,
        # DistanceReward,
        # EconomicReward,
        # EpisodeDurationReward,
        FlatReward,
        # GameplayReward,
        # IncreasingFlatReward,
        # L2RPNReward,
        # LinesReconnectedReward,
        # # N1Reward,
        # RedispReward
    ]:
        axs.plot(
            simulate(
                agent_name="topology greedy",
                env_name=env_name,
                MAX_STEPS=MAX_STEPS,
                reward_class=reward_class,
            ),
            label=reward_class.__name__,
            linestyle=random.choice(["dotted", "dashed", "dashdot"]),
        )

    axs.legend()
    axs.set_title("DoingNothing VS TopologyGreedy for different rewards")
    axs.set_xlabel("step")
    axs.set_ylabel("max current (lower is better)")
    plt.show()


if __name__ == "__main__":
    main()
