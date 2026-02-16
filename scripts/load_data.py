import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx

from src import DATA_PATH
from src.dataset import load_data_sample, load_env, to_pyg_data


def main() -> None:
    np.set_printoptions(legacy="1.25")

    DATASET_PATH = DATA_PATH / "my_dataset"

    env = load_env(DATASET_PATH)
    filenames = DATASET_PATH.glob("*.xz")

    filename = next(filenames)

    data_sample = load_data_sample(filename)

    data = to_pyg_data(data_sample, env)

    G = to_networkx(data)

    pos = nx.nx_pydot.pydot_layout(G, prog="neato")

    node_colors = []
    for node in range(data.num_nodes):
        if data.substation_indicator[node]:
            node_colors.append("lightblue")

        if data.gen_indicator[node]:
            node_colors.append("lime")

        if data.load_indicator[node]:
            node_colors.append("orange")

    fig, axs = plt.subplots(1, 1)

    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, ax=axs)
    plt.show()

    print(data)


if __name__ == "__main__":
    main()
