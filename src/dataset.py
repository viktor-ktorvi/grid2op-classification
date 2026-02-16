import lzma
import pickle
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


def load_env(dataset_path: Path) -> dict:
    with lzma.open(dataset_path / "environment.xz", "rb") as f:
        env: dict = pickle.load(f)

    return env


def load_data_sample(filename: Path) -> dict:
    with lzma.open(filename, "rb") as f:
        data_sample: dict = pickle.load(f)
    return data_sample


class NodeX(IntEnum):
    gen_p = 0
    gen_q = 1
    gen_bus = 2
    gen_v = 3
    gen_theta = 4
    load_p = 5
    load_q = 6
    load_bus = 7
    load_v = 8
    load_theta = 9
    shunt_p = 10
    shunt_q = 11
    shunt_v = 12


class EdgeX(IntEnum):
    p = 0
    q = 1
    a = 2
    v = 3
    rho = 4
    theta = 5
    direction = 6


def to_pyg_data(data_sample: dict, env: dict) -> Data:
    obs = data_sample["observation"]
    n_sub = env["n_sub"]

    x_list = []
    for sub in range(n_sub):
        x_list.append(np.zeros((len(NodeX),)))

    n_line = env["n_line"]
    line_edge_index_or = np.vstack((env["line_or_to_subid"], env["line_ex_to_subid"]))
    line_edge_index_ex = np.vstack((env["line_ex_to_subid"], env["line_or_to_subid"]))

    line_edge_index = np.hstack((line_edge_index_or, line_edge_index_ex))
    line_edge_attr = np.zeros((n_line * 2, len(EdgeX)))
    for line in range(n_line):
        backwards_edge_idx = line + n_line

        line_edge_attr[line][EdgeX.p] = obs["p_or"][line]
        line_edge_attr[backwards_edge_idx][EdgeX.p] = obs["p_ex"][line]

        line_edge_attr[line][EdgeX.q] = obs["q_or"][line]
        line_edge_attr[backwards_edge_idx][EdgeX.q] = obs["q_ex"][line]

        line_edge_attr[line][EdgeX.a] = obs["a_or"][line]
        line_edge_attr[backwards_edge_idx][EdgeX.a] = obs["a_ex"][line]

        line_edge_attr[line][EdgeX.v] = obs["v_or"][line]
        line_edge_attr[backwards_edge_idx][EdgeX.v] = obs["v_ex"][line]

        line_edge_attr[line][EdgeX.rho] = obs["rho"][line]
        line_edge_attr[backwards_edge_idx][EdgeX.rho] = obs["rho"][line]

        line_edge_attr[line][EdgeX.theta] = obs["theta_or"][line]
        line_edge_attr[backwards_edge_idx][EdgeX.theta] = obs["theta_ex"][line]

        line_edge_attr[line][EdgeX.direction] = 0
        line_edge_attr[backwards_edge_idx][EdgeX.direction] = 1

    n_shunt = env["n_shunt"]
    for shunt in range(n_shunt):
        shunt_sub = env["shunt_to_subid"][shunt]
        x_list[shunt_sub][NodeX.shunt_p] = obs["_shunt_p"][shunt]
        x_list[shunt_sub][NodeX.shunt_q] = obs["_shunt_q"][shunt]
        x_list[shunt_sub][NodeX.shunt_v] = obs["_shunt_v"][shunt]

    n_gen = env["n_gen"]
    gen_edge_index = np.zeros((2, n_gen))
    gen_edge_attr = np.zeros((n_gen, len(EdgeX)))
    for gen in range(n_gen):
        x_idx = len(x_list)

        # directed edge from gen to sub, no need for backwards connection
        gen_edge_index[0][gen] = x_idx
        gen_edge_index[1][gen] = env["gen_to_subid"][gen]

        x = np.zeros((len(NodeX),))
        x[NodeX.gen_p] = obs["gen_p"][gen]
        x[NodeX.gen_q] = obs["gen_q"][gen]
        x[NodeX.gen_bus] = obs["gen_bus"][gen]
        x[NodeX.gen_v] = obs["gen_v"][gen]
        x[NodeX.gen_theta] = obs["gen_theta"][gen]
        x_list.append(x)

    n_load = env["n_load"]
    load_edge_index = np.zeros((2, n_load))
    load_edge_attr = np.zeros((n_load, len(EdgeX)))
    for load in range(n_load):
        x_idx = len(x_list)

        load_edge_index[0][load] = x_idx
        load_edge_index[1][load] = env["load_to_subid"][load]

        x = np.zeros((len(NodeX),))
        x[NodeX.load_p] = obs["load_p"][load]
        x[NodeX.load_q] = obs["load_q"][load]
        x[NodeX.load_bus] = obs["load_bus"][load]
        x[NodeX.load_v] = obs["load_v"][load]
        x[NodeX.load_theta] = obs["load_theta"][load]
        x_list.append(x)

    x_matrix = np.vstack(x_list)
    edge_index = np.hstack((line_edge_index, gen_edge_index, load_edge_index))
    edge_attr = np.vstack((line_edge_attr, gen_edge_attr, load_edge_attr))

    n_nodes = x_matrix.shape[0]

    substation_indicator = np.zeros((n_nodes, 1))
    substation_indicator[:n_sub] = 1

    gen_indicator = np.zeros((n_nodes, 1))
    gen_indicator[n_sub : n_sub + n_gen] = 1

    load_indicator = np.zeros((n_nodes, 1))
    load_indicator[n_sub + n_gen : n_sub + n_gen + n_load] = 1

    if np.sum(substation_indicator) + np.sum(gen_indicator) + np.sum(load_indicator) != n_nodes:
        raise RuntimeError("Indicators don't sum up to the number of nodes.")

    y = np.zeros((n_nodes, 1))
    y[data_sample["substation"]] = 1

    data = Data(
        x=torch.tensor(x_matrix),
        y=torch.tensor(y),
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.tensor(edge_attr),
        substation_indicator=torch.tensor(substation_indicator, dtype=torch.bool),
        gen_indicator=torch.tensor(gen_indicator, dtype=torch.bool),
        load_indicator=torch.tensor(load_indicator, dtype=torch.bool),
    )

    return data
