# TODO
#  for now just print metrics in progress bar
#  see which metrics are the best for this unbalanced stuff
#  accuracy, F1, at the end a confusion matrix to be sure
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule, Trainer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT, MLP
from torchmetrics import Accuracy

from src import DATA_PATH
from src.dataset import EdgeX, NodeX, load_data_sample, load_env, to_pyg_data


def load_data_list(dataset_path: Path) -> tuple[list[Data], int]:
    env = load_env(dataset_path)
    num_classes = env["n_sub"]

    filenames = dataset_path.glob("*.xz")

    data_list = []
    for filename in filenames:
        if filename.name == "environment.xz":
            continue

        data_sample = load_data_sample(filename)

        data = to_pyg_data(data_sample, env)
        data_list.append(data)

    return data_list, num_classes


def calculate_statistics(x: Tensor) -> tuple[Tensor, Tensor]:
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0)
    x_std[x_std < 1e-5] = 1.0

    return x_mean, x_std


@dataclass
class Statistics:
    mean: Tensor
    std: Tensor


class InputStatistics:
    x: Statistics
    edge_attr: Statistics

    def __init__(self, train_batch: Data, *, type_specific: bool = False):
        if type_specific:
            gen_mean, gen_std = calculate_statistics(train_batch.x[train_batch.gen_indicator.flatten(), : NodeX.load_p])
            load_mean, load_std = calculate_statistics(
                train_batch.x[train_batch.load_indicator.flatten(), NodeX.load_p : NodeX.shunt_p]
            )
            shunt_mean, shunt_std = calculate_statistics(
                train_batch.x[train_batch.substation_indicator.flatten(), NodeX.shunt_p :]
            )

            x_mean = torch.hstack((gen_mean, load_mean, shunt_mean))
            x_std = torch.hstack((gen_std, load_std, shunt_std))
        else:
            x_mean, x_std = calculate_statistics(train_batch.x)
        edge_attr_mean, edge_attr_std = calculate_statistics(train_batch.edge_attr)

        self.x = Statistics(mean=x_mean, std=x_std)
        self.edge_attr = Statistics(mean=edge_attr_mean, std=edge_attr_std)


class Scaler(nn.Module):
    mean: Tensor
    std: Tensor

    def __init__(self, mean: Tensor, std: Tensor):
        super().__init__()

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class Model(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_channels: int,
        num_mlp_layers: int,
        num_gnn_layers: int,
        dropout: float,
        input_statistics: InputStatistics,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.x_scaler = Scaler(mean=input_statistics.x.mean, std=input_statistics.x.std)
        self.edge_attr_scaler = Scaler(mean=input_statistics.edge_attr.mean, std=input_statistics.edge_attr.std)

        self.input_mlp_x = MLP(
            in_channels=len(NodeX),
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_mlp_layers,
            dropout=dropout,
        )

        self.input_mlp_edge = MLP(
            in_channels=len(EdgeX),
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_mlp_layers,
            dropout=dropout,
        )

        # TODO graph transformer
        self.gnn = GAT(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_gnn_layers,
            dropout=dropout,
            jk="cat",
            v2=True,
            edge_dim=hidden_channels,
            heads=16,
            residual=True,
        )

        self.out_mlp = MLP(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            num_layers=num_mlp_layers,
            dropout=dropout,
        )

    def forward(self, batch: Data) -> Tensor:
        x_normalized = self.x_scaler(batch.x)
        edge_attr_normalized = self.edge_attr_scaler(batch.edge_attr)

        x_hidden = self.input_mlp_x(x_normalized)
        edge_attr_hidden = self.input_mlp_edge(edge_attr_normalized)

        x_hidden = self.gnn(x=x_hidden, edge_index=batch.edge_index, edge_attr=edge_attr_hidden)

        x_hidden = x_hidden[batch.substation_indicator.flatten()]
        out: Tensor = self.out_mlp(x_hidden)

        out = out.reshape(-1, self.num_classes)
        return out


class LightningModel(LightningModule):
    def __init__(self, model: Model, learning_rate: float):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.accuracy_train = Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.accuracy_val = Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.accuracy = {"train": self.accuracy_train, "val": self.accuracy_val}

    def inference(self, batch: Data, dataset_type: Literal["train", "val", "test"]) -> Tensor:
        out = self.model(batch)
        loss: Tensor = self.criterion(input=out, target=batch.y)
        self.log(f"{dataset_type}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.accuracy[dataset_type](out.argmax(dim=1), batch.y)
        self.log(f"{dataset_type}/accuracy", self.accuracy[dataset_type], on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        return self.inference(batch, "train")

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        self.inference(batch, "val")

    def predict_step(self, batch: Data, batch_idx: int) -> tuple[Tensor, Tensor]:
        return self.model(batch).argmax(dim=1), batch.y

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def main() -> None:
    np.set_printoptions(legacy="1.25")
    torch.set_default_dtype(torch.float32)

    MAX_DATA_SAMPLES = 20000
    NUM_EPOCHS = 500
    HIDDEN_CHANNELS = 256
    NUM_MLP_LAYERS = 2
    NUM_GNN_LAYERS = 4
    DROPOUT = 0.2
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 256

    DATASET_PATH = DATA_PATH / "case14_20k"

    data_list, num_classes = load_data_list(DATASET_PATH)
    random.shuffle(data_list)
    data_list = data_list[:MAX_DATA_SAMPLES]

    training_dataset, validation_dataset = train_test_split(data_list, test_size=0.3)

    train_batch = next(iter(DataLoader(training_dataset, batch_size=len(training_dataset))))
    input_statistics = InputStatistics(train_batch, type_specific=True)

    model = Model(
        num_classes=num_classes,
        hidden_channels=HIDDEN_CHANNELS,
        num_mlp_layers=NUM_MLP_LAYERS,
        num_gnn_layers=NUM_GNN_LAYERS,
        input_statistics=input_statistics,
        dropout=DROPOUT,
    )
    lightning_model = LightningModel(model, learning_rate=LEARNING_RATE)

    train_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

    trainer = Trainer(max_epochs=NUM_EPOCHS, check_val_every_n_epoch=25)
    trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    predict_out = trainer.predict(lightning_model, DataLoader(validation_dataset, batch_size=len(validation_dataset)))
    if not predict_out:
        return

    predictions, targets = predict_out[0]

    fig, axs = plt.subplots(1, 1)
    ConfusionMatrixDisplay.from_predictions(y_true=targets.cpu().numpy(), y_pred=predictions.cpu().numpy(), ax=axs)
    plt.show()


if __name__ == "__main__":
    main()
