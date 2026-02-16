import lzma
import pickle
import random
import shutil

import grid2op
import matplotlib.pyplot as plt
import numpy as np
from grid2op.Agent import DoNothingAgent, TopologyGreedy
from lightsim2grid import LightSimBackend

from src import DATA_PATH
from src.reward import MaxCurrentReward


def main() -> None:
    RANDOM_SEED = 0
    MAX_STEPS = 100000
    CURRENT_THRESHOLD = 0.8
    random.seed(RANDOM_SEED)
    agent_name = "topology greedy"
    env_name = "l2rpn_case14_sandbox"
    # env_name = "l2rpn_icaps_2021_small"
    # env_name = "l2rpn_idf_2023"

    DATASET_PATH = DATA_PATH / "my_dataset"

    # delete contents
    if DATASET_PATH.is_dir():
        shutil.rmtree(DATASET_PATH)

    DATASET_PATH.mkdir(exist_ok=True, parents=True)
    reward_class = MaxCurrentReward
    env = grid2op.make(env_name, reward_class=reward_class, backend=LightSimBackend())
    env.seed(RANDOM_SEED)

    if agent_name == "do nothing":
        agent = DoNothingAgent(env.action_space)
    elif agent_name == "topology greedy":
        agent = TopologyGreedy(env.action_space)
    else:
        raise ValueError(f"{agent_name=}")
    agent.seed(RANDOM_SEED)

    default_do_nothing_agent = DoNothingAgent(env.action_space)

    step_counter = 0
    max_currents = []
    substations = []
    while step_counter < MAX_STEPS:
        max_timeseries_id = len(env.chronics_handler.subpaths) - 1
        obs = env.reset(
            options={
                "time serie id": random.choice(range(max_timeseries_id)),
                "max step": MAX_STEPS,
            }
        )
        reward = env.reward_range[0]
        done = False
        while not done and step_counter < MAX_STEPS:
            max_current = obs.rho.max()

            if max_current > CURRENT_THRESHOLD:
                action = agent.act(obs, reward, done)
                substation = {
                    bus_assignment["substation"].item()
                    for bus_assignment in action.impact_on_objects()["topology"]["assigned_bus"]
                }

                if len(substation) > 1:
                    raise ValueError("Agent acting on more than one substation.")

                if len(substation) > 0:
                    substation = substation.pop()
                    substations.append(substation)

                    # lzma for compression
                    with lzma.open(
                        DATASET_PATH / f"{str(step_counter).zfill(len(str(MAX_STEPS)))}.xz",
                        "wb",
                    ) as f:
                        # NOTE using pickle is bad practice because it will execute arbitrary code when loading;
                        #  never share a dataset in pickle format
                        obs_dump = {
                            k: v for k, v in obs.__dict__.items() if isinstance(v, (np.ndarray, int, float, str))
                        }
                        pickle.dump({"observation": obs_dump, "substation": substation}, f)

            else:
                action = default_do_nothing_agent.act(obs, reward, done)

            obs, reward, done, info = env.step(action)

            max_currents.append(max_current)
            print(f"{step_counter=}, {agent_name=}, {reward_class.__name__=}, {max_current=:2.3f}")
            step_counter += 1

    fig, axs = plt.subplots(1, 1)
    axs.hist(substations, bins=env.n_sub)
    axs.set_xlabel("substation")
    axs.set_ylabel("count")
    axs.set_title("Chosen substations")

    plt.show()


if __name__ == "__main__":
    main()
