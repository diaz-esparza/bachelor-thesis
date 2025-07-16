from os.path import isfile, join as jo
from lightning.pytorch.callbacks import early_stopping
from torch.utils.data import DataLoader

import hashlib
import json
import lightning as L
import lightning.pytorch.callbacks as callbacks
import lightning.pytorch.loggers as loggers
import numpy as np
import numpy.typing as npt
import os
import os.path as osp
import torch
import torchvision.transforms.v2 as v2

from data import (
    BasicIndexer,
    StratifiedIndexer,
    IndexerCore,
    CoreHAM,
    CoreMED,
    CorePAD,
    Core2020,
    Core2024,
    GaussianNoiseInt,
    GenericRegularizer,
    GenericModelRegularizer,
    ModularDataset,
    ModelRegularizer,
    RandomRotationCrop,
    get_num_domains,
    STAGE,
)

from models import VAE

from typing import Any, Literal, Mapping

ROOT = jo(osp.dirname(__file__), "..", "..")
OUTPUTS = jo(ROOT, "storage", "outputs")
CKPT = jo(OUTPUTS, "ckpt")
torch.set_float32_matmul_precision("high")


def raw_collate(data: list[tuple[npt.NDArray[np.uint8], int, bool]]):
    output = list(zip(*data, strict=True))
    return output[0], torch.tensor(output[1]), torch.tensor(output[2])


def train_vae(
    core: IndexerCore = CoreHAM(),
    train_id: str = "vae",
    max_epochs: int = 32,
    batch_size: int = 32,
    enc_type: Literal["resnet18", "resnet50"] = "resnet18",
    first_conv: bool = False,
    maxpool1: bool = False,
    kl_coeff: float = 0.1,
    latent_dim: int = 256,
    lr: float = 1e-4,
    l2: float = 1e-4,
    image_size: tuple[int, int] = (224, 224),
    gpu_preprocessing: Literal[0, 1, 2] = 0,
) -> Mapping[str, Any] | None:

    ## Typical dataset schema
    regularizer_kwargs: dict[str, Any] = dict(
        image_size=image_size,
        norm_type="imagenet",
        flip_h=True,
        flip_v=True,
        crop_prob=0.25,
        post_resize_transforms=v2.ColorJitter(),
    )

    ## We may offload more preprocessing to the GPU,
    ## might add levels to manage that later
    collate_fn = None
    if gpu_preprocessing >= 1:
        model_regularizer = GenericModelRegularizer(device="cuda", **regularizer_kwargs)
        if gpu_preprocessing >= 2:
            regularizer = model_regularizer.compatible_reg(image_size=None)
            collate_fn = raw_collate
        else:
            regularizer = model_regularizer.compatible_reg(
                image_size=model_regularizer.image_size
            )
    else:
        regularizer = GenericRegularizer(**regularizer_kwargs)
        model_regularizer = None

    datasets: dict[STAGE, ModularDataset] = {
        stage: ModularDataset(
            stage,
            color_mode="RGB",
            indexer=StratifiedIndexer,
            indexer_core=core,
            regularizer=(
                regularizer
                if stage == "train"
                else GenericRegularizer(
                    image_size=regularizer_kwargs.get("image_size", (128, 128)),
                    norm_type=regularizer_kwargs.get("norm_type", "imagenet"),
                    scale_tensors=regularizer_kwargs.get("scale_tensors", True),
                )
            ),
            policy=VAE.DATASET_POLICY,
        )
        for stage in ("train", "valid", "test")
    }

    data_workers = max((os.cpu_count() or 1) - 1, 1)
    dataloaders: dict[STAGE, DataLoader] = {
        stage: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=stage == "train",
            num_workers=data_workers,
            collate_fn=collate_fn,
        )
        for stage, dataset in datasets.items()
    }

    model_args = dict(
        input_height=image_size[1],
        enc_type=enc_type,
        first_conv=first_conv,
        maxpool1=maxpool1,
        kl_coeff=kl_coeff,
        latent_dim=latent_dim,
        lr=lr,
        l2=l2,
    )
    all_args = {
        "model": model_args,
        "core": core.__class__.__name__,
    }

    exp_hash = hashlib.sha256(json.dumps(all_args).encode()).hexdigest()
    ckpt_path = jo(
        CKPT,
        "unsupe",
        train_id,
        f"VAE_{exp_hash}.ckpt",
    )
    lock_path = ckpt_path + ".worker"
    arg_path = ckpt_path + ".args"

    if not osp.exists(osp.dirname(ckpt_path)):
        os.makedirs(osp.dirname(ckpt_path))
    print(ckpt_path, "\n" * 5)

    if osp.isfile(ckpt_path):
        print("Skipping (done)...")
        return
    if osp.isfile(lock_path):
        with open(lock_path, "rt") as f:
            xana_name = f.read()
        if xana_name != os.environ["XANA_NAME"]:
            print(f"Skipping (busy by {xana_name})...")
            return
    with open(lock_path, "wt") as f:
        f.write(os.environ["XANA_NAME"])
    with open(arg_path, "wt") as f:
        json.dump(all_args, f)

    model = VAE(
        **model_args,  ##type:ignore
        model_regularizer=model_regularizer,
    ).to("cuda")

    output = jo(
        OUTPUTS,
        "unsupe",
        train_id,
        core.__class__.__name__.replace("Core", ""),
        enc_type,
    )
    if not osp.exists(output):
        os.makedirs(output)
    logger = loggers.CSVLogger(
        output,
        name=None,
    )

    early_stopping = callbacks.EarlyStopping(
        "train_loss",
        min_delta=2e-3,
        patience=5,
        verbose=True,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stopping],
        logger=logger,
        log_every_n_steps=0,
        enable_checkpointing=False,
    )

    trainer.fit(
        model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["valid"],
    )
    results = trainer.validate(model, dataloaders=dataloaders["valid"])

    trainer.save_checkpoint(ckpt_path)
    return results[0]


def main():
    for core in (CorePAD(), CoreMED(), CoreHAM()):
        for name in ("resnet18", "resnet50"):
            ## Changed implementation for xana ditributed
            """
            match os.environ["XANA_NAME"]:
                case "neo":
                    latent_dim_set = (128, 256)
                case "og":
                    latent_dim_set = (32, 64)
                case _:
                    raise NotImplementedError(
                        f"Not implemented for system '{os.environ['XANA_NAME']}'"
                    )
            """
            latent_dim_set = (32, 64, 128, 256)

            for latent_dim in latent_dim_set:
                print(f"TRAINING {core.__class__.__name__} {name} @ {latent_dim}...")
                match name:
                    case "resnet18":
                        batch_size = 6
                    case "resnet50":
                        batch_size = 1
                try:
                    train_vae(
                        core,
                        train_id="vae",
                        max_epochs=32,
                        batch_size=batch_size,
                        enc_type=name,
                        first_conv=False,
                        maxpool1=False,
                        kl_coeff=0.1,
                        latent_dim=latent_dim,
                        lr=1e-4,
                        l2=1e-4,
                        image_size=(256, 256),
                        gpu_preprocessing=0,
                    )
                except Exception as e:
                    print(f"Got exception '{e.__class__.__name__}': ``{e}``")


if __name__ == "__main__":
    main()
