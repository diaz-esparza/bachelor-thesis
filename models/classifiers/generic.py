import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy, AUROC, ConfusionMatrix, F1Score, Metric

from data import ModelRegularizer

## Custom code
## Same as ``torch.nn.ModuleDict`` but uses base64 encoding to prevent
## possible attribute name clashes and allow the use of ".", mainly.
from ..module_dict import RobustModuleDict

## Real/Fake classifier
## With GRL, it forces the feature extractor to only get features common
## within the real and fake samples, effectively closing the real/fake
## domain gap.
from .rf_addon import DAWrapper, MultiDAWrapper

from typing import Any, cast, Literal

__all__ = ["GenericClassifier"]

STAGE = Literal["train", "valid", "test"]

N_SOURCES = 6


class GenericClassifier(L.LightningModule):
    DATASET_POLICY: str = "fake_source"

    def __init__(
        self,
        num_domains: int,
        model: str | nn.Module,
        ckpt_path: str = "",
        *,
        lr: float = 1e-4,
        l2: float = 1e-4,
        optim_hyper_params: dict[str, Any] = {},
        has_fake: bool = False,
        source_da: bool = False,
        disable_source_da: bool = False,
        freeze: bool = False,
        model_regularizer: ModelRegularizer | None = None,
        other_notes: dict[str, Any] | None = None,
    ):
        super(GenericClassifier, self).__init__()
        self.save_hyperparameters()

        self.num_labels = num_domains
        self.lr = lr
        self.l2 = l2
        self.model_regularizer = model_regularizer

        self.disable_source_da = disable_source_da

        if not isinstance(model, nn.Module):
            ## It's a string, get Timm to handle it
            model = timm.create_model(  ##type:ignore
                model,
                num_classes=self.num_labels,
                checkpoint_path=ckpt_path,
            )
        assert isinstance(model, nn.Module)

        n_domains_per_da: dict[str, int] = {}
        if has_fake:
            n_domains_per_da["fake"] = 2
        if source_da:
            n_domains_per_da["source"] = 6
        self.model = MultiDAWrapper(model, n_domains_per_da)

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
                        "confmat": ConfusionMatrix(
                            task="multiclass",
                            num_classes=self.num_labels,
                        ),
                    }
                )
                for stage in ("train", "valid", "test")
            }
        )
        self.p_metrics = RobustModuleDict(
            {
                stage: nn.ModuleDict(
                    {
                        "auroc": AUROC(
                            task="multiclass",
                            num_classes=self.num_labels,
                            average="macro",
                        ).to(self.device)
                    }
                )
                for stage in ("train", "valid", "test")
            }
        )
        # self.confusion = ConfusionMatrix(task="multiclass", num_classes=self.num_labels)
        # self.metrics = {
        #     "Accuracy": Accuracy(task="multiclass", num_classes=18, top_k=1),
        #     "ConfusionMatrix": ConfusionMatrix(task="multiclass", num_classes=18)
        # }

        # Training setup
        self.optim_hyper_params = optim_hyper_params

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
            weight_decay=self.l2,
            **self.optim_hyper_params,
        )  # , momentum=0.9)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=0.5,
                    mode="max",
                    patience=4,
                    verbose=True,  ##type:ignore
                    # min_lr=1e-5,
                ),
                "monitor": "valid_f1",
                "frequency": self.trainer.check_val_every_n_epoch,
            },
        }

    def shared_step(
        self,
        stage: STAGE,
        batch: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y, fake, source = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log(f"{stage}_loss", loss)

        da_values = self.model.da_values
        if not self.disable_source_da:
            if "fake" in da_values:
                rf_logits = da_values["fake"].squeeze()
                rf_loss = F.binary_cross_entropy_with_logits(rf_logits, fake.float())
                loss += rf_loss
                self.log(f"{stage}_rf", rf_loss)
            if "source" in da_values:
                source_logits = da_values["source"].squeeze()
                source_loss = F.cross_entropy(source_logits, source)
                loss += source_loss
                self.log(f"{stage}_source", source_loss)

        probs = logits.softmax(-1)
        for fn in self.p_metrics[stage].values():
            fn(probs, y)

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
        metrics = nn.ModuleDict()
        metrics.update(self.metrics[stage])
        metrics.update(self.p_metrics[stage])

        for name, fn in metrics.items():
            value = cast(torch.Tensor, fn.compute())
            assert value.numel() > 0
            if value.numel() > 1:
                value = value.squeeze()
                indices = [0] * value.ndim
                while indices[0] < value.shape[0]:
                    self.log(
                        f"{stage}_{name}_({','.join(map(str,indices))})",
                        value[*indices].to(torch.float32),
                    )
                    ## Counter with carry
                    indices[-1] += 1
                    for i in range(value.ndim - 1, 0, -1):
                        if indices[i] >= value.shape[i]:
                            indices[i] = 0
                            indices[i - 1] += 1

            else:
                self.log(f"{stage}_{name}", fn.compute().to(torch.float32))

            fn.reset()

        # Handle Confusion Matrix
        """
        pred_values = torch.flatten(torch.cat(self.validation_step_preds).cpu())
        y_values = torch.flatten(torch.cat(self.validation_step_y).cpu())
        self.loggers[0].experiment.log_sklearn_plot( ##type:ignore
            "confusion_matrix",
            np.array(y_values),
            np.array(pred_values),
            normalized=False,
            name="Confusion Matrix",
            title="Confusion Matrix",
        )
        """

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

    @classmethod
    def batch_regularizer(cls, batch, device, regularizer) -> Any:
        data = batch[0]
        if isinstance(data, torch.Tensor):
            output = torch.empty(
                (
                    len(data),  ## batch
                    data[0].shape[0],  ## channels
                    *regularizer.image_size,  ## image_size
                )
            ).to(device)
            for i, x in enumerate(data):
                x = x.to(device)
                output[i] = regularizer.pre_resize(x)
        else:
            output = data.to(device)
        return regularizer.post_resize(output), *[t.to(device) for t in batch[1:]]

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.model_regularizer is not None:
            return self.batch_regularizer(batch, device, self.model_regularizer)
        else:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
