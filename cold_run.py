from os.path import join as jo

import numpy as np
import numpy.typing as npt
import os.path as osp
import torch
import torchvision.transforms.v2 as v2

from data import (
    StratifiedIndexer,
    IndexerCore,
    CoreHAM,
    CorePAD,
    Core2020,
    Core2024,
    GaussianNoiseInt,
    GenericRegularizer,
    ModularDataset,
    RandomRotationCrop,
    STAGE,
)

from models.classifiers import GenericClassifier


from typing import Any

ROOT = osp.dirname(__file__)
torch.set_float32_matmul_precision("high")


def cold_run(
    core: IndexerCore = CoreHAM(),
    image_size: tuple[int, int] = (224, 224),
) -> None:

    ## Typical dataset schema
    regularizer_kwargs: dict[str, Any] = dict(
        image_size=image_size,
        norm_type=None,
        flip_h=True,
        flip_v=True,
        crop_prob=0.0,
        pre_resize_transforms=[
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 4.0))],
                p=0.3,
            ),
            v2.RandomApply([RandomRotationCrop(20)], p=0.4),  ##type:ignore
            # Perspective might need testing, as the black background is concerning
            v2.RandomApply(
                [
                    v2.RandomPerspective(distortion_scale=0.175, p=1),
                    v2.RandomAffine(0, scale=(1.25, 1.25)),  ##type:ignore
                ],
                p=0.25,
            ),
        ],
        post_resize_transforms=[
            v2.ColorJitter(
                brightness=0.3,
                contrast=0.2,
                saturation=0.3,
                hue=0.05,
            ),
            v2.RandomApply([GaussianNoiseInt(sigma=0.02)], p=0.3),
        ],
    )

    ## We may offload more preprocessing to the GPU,
    ## might add levels to manage that later
    regularizer = GenericRegularizer(**regularizer_kwargs)

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
            policy=GenericClassifier.DATASET_POLICY,
        )
        for stage in ("train", "valid", "test")
    }

    i = 0
    for x in datasets["train"]:
        print(i)
        print(x[0].shape)
        y = v2.ToPILImage()(x[0])

        print(np.asarray(y).min(), np.asarray(y).max())
        y.save(f"{i}.png")
        if i >= 130:
            break
        i += 1


def main():
    core = Core2024()
    cold_run(core, image_size=(224, 224))


if __name__ == "__main__":
    main()
