import numpy as np

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

    print(data)


if __name__ == "__main__":
    main()
