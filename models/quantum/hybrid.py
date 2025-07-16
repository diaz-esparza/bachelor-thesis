from collections import OrderedDict, abc as container_abcs
from pennylane import qml
from torch.nn import ModuleDict, Module
from torch._jit_internal import _copy_to_script_wrapper
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Metric
from torchvision.models import resnet18

import base64
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Tuple,
)

from ..module_dict import RobustModuleDict

__all__ = ["HybridClassifier"]

STAGE = Literal["train", "valid", "test"]


class HybridClassifier(L.LightningModule):

    def __init__(
        self,
        n_qbits: int,
        n_layers: int,
        lr: float = 1e-4,
        optim_hyper_params: dict[str, Any] = {},
        freeze: bool = False,
    ):
        super(HybridClassifier, self).__init__()
        self.save_hyperparameters()

        self.num_labels = 2
        self.lr = lr

        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = RobustModuleDict(
            {
                stage: nn.ModuleDict(
                    {
                        "accuracy": Accuracy(
                            task="multiclass",
                            num_classes=self.num_labels,
                            top_k=1,
                        ).to(self.device),
                        "accuracy_macro": Accuracy(
                            task="multiclass",
                            num_classes=self.num_labels,
                            top_k=1,
                            average="macro",
                        ).to(self.device),
                        "f1": F1Score(
                            task="multiclass",
                            num_classes=self.num_labels,
                            average="macro",
                            top_k=1,
                        ).to(self.device),
                    }
                )
                for stage in ("train", "valid", "test")
            }
        )

        self.optim_hyper_params = optim_hyper_params

        backbone = resnet18(pretrained="DEFAULT")
        self.model = nn.Sequential(*list(backbone.children())[:-2])

        ## Los encoders dan outputs con mucha dimensionalidad
        self.model.append(nn.LazyLinear(out_features=n_qbits))
        self.model.append(nn.ReLU(inplace=True))
        self.DEV = qml.device("default.qubit", wires=n_qbits)

        @qml.qnode(self.DEV)
        def qnode(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qbits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qbits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qbits)]

        weight_shapes = {"weights": (n_layers, n_qbits, 3)}
        self.model.append(qml.qnn.TorchLayer(qnode, weight_shapes))
        ## A clasificación binaria. CrossEntropy requiere no tener sigmoide.
        self.model.append(nn.Linear(n_qbits, 1))

        if freeze:
            pars = self.model.stages.parameters()
            for par in pars:
                par.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):  ##type:ignore
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            **self.optim_hyper_params,
        )

        return {"optimizer": optimizer}

    def shared_step(
        self,
        stage: STAGE,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log(f"{stage}_loss", loss)

        probs = logits.softmax(-1)
        preds = probs.argmax(-1)

        for fn in self.metrics[stage].values():
            fn(preds, y)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step("valid", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_step("test", batch, batch_idx)

    def shared_epoch_end(self, stage: STAGE) -> None:
        for name, fn in self.metrics[stage].items():
            self.log(f"{stage}_{name}", fn.compute())
            fn.reset()

    def on_train_epoch_end(self, *args, **kwargs):
        self.shared_epoch_end("train")
        super().on_train_epoch_end(*args, **kwargs)

    def on_validation_epoch_end(self, *args, **kwargs):
        self.shared_epoch_end("valid")
        super().on_validation_epoch_end(*args, **kwargs)

    def on_test_epoch_end(self, *args, **kwargs):
        self.shared_epoch_end("test")
        super().on_test_epoch_end(*args, **kwargs)

    def predict_step(self, batch, batch_idx):
        logits = self(batch)
        probs = logits.softmax(-1)
        max_probs, preds = probs.max(-1)

        return preds, max_probs, probs
