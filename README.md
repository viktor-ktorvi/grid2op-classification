# grid2op-classification

Classify which substation the topology greedy agent will act upon.

## Installation

Install uv

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Get the repo
```shell
git clone https://github.com/viktor-ktorvi/grid2op-classification.git
cd grid2op-classification
```

Install dependencies
```shell
uv sync --python 3.13
```

## Usage
### Generate data
```shell
uv run python -m scripts.generate_data
```

### Train a GNN classifier
```shell
uv run python -m scripts.train
```
