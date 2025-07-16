from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import ModelRegularizer

from typing import Any, Literal

__all__ = ["VAE"]

STAGE = Literal["train", "valid", "test"]
METRIC = dict[STAGE, list[float | torch.Tensor]]


class VAE(L.LightningModule):
    DATASET_POLICY: str = "basic"

    def __init__(
        self,
        input_height: int,
        enc_type: Literal["resnet18", "resnet50"] = "resnet18",
        first_conv: bool = False,
        maxpool1: bool = False,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        l2: float = 1e-4,
        optim_hyper_params: dict[str, Any] = {},
        model_regularizer: ModelRegularizer | None = None,
    ) -> None:
        r"""
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """
        super(VAE, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.l2 = l2
        self.optim_hyper_params = optim_hyper_params
        self.kl_coeff = kl_coeff
        self.latent_dim = latent_dim
        self.input_height = input_height

        self.model_regularizer = model_regularizer

        match enc_type:
            case "resnet18":
                self.encoder = resnet18_encoder(first_conv, maxpool1)
                self.decoder = resnet18_decoder(
                    self.latent_dim,
                    self.input_height,
                    first_conv,
                    maxpool1,
                )
                self.enc_out_dim = 512
            case "resnet50":
                self.encoder = resnet50_encoder(first_conv, maxpool1)
                self.decoder = resnet50_decoder(
                    self.latent_dim,
                    self.input_height,
                    first_conv,
                    maxpool1,
                )
                self.enc_out_dim = 2048
            case _:
                raise ValueError(enc_type)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

        self.data_bus: dict[str, METRIC] = {}

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def infer(self, x):
        x = self.encoder(x)
        return self.fc_mu(x)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.l2,
            **self.optim_hyper_params,
        )

    def shared_step(
        self,
        stage: STAGE,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        losses = {
            "recon": recon_loss,
            "kl": kl,
            "loss": loss,
        }

        for name, v in losses.items():
            if name not in self.data_bus.keys():
                self.data_bus[name] = {"train": [], "valid": [], "test": []}
            self.data_bus[name][stage].append(v.detach().cpu())

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step("valid", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_step("test", batch, batch_idx)

    def shared_epoch_end(self, stage: STAGE) -> None:
        for name in self.data_bus.keys():
            self.log(
                f"{stage}_{name}",
                torch.nanmean(torch.tensor(self.data_bus[name][stage])).item(),
            )
            self.data_bus[name][stage] = []

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
        return regularizer.post_resize(output), *[t.to(device) for t in batch[1:]]

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.model_regularizer is not None:
            return self.batch_regularizer(batch, device, self.model_regularizer)
        else:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
