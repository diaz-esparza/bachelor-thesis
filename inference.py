from os.path import join as jo
from tqdm import tqdm

import json
import numpy as np
import os
import os.path as osp
import random
import torch

from data import (
    CoreHAM,
    CoreMED,
    CorePAD,
    GenericRegularizer,
    ModularDataset,
    BasicIndexer,
)
from models.autoencoders import VAE

from typing import Literal

ROOT = osp.dirname(__file__)
OUTPUTS = jo(ROOT, "storage", "outputs")
EMBEDDINGS = jo(OUTPUTS, "datasets", "embeddings")
CKPT = jo(OUTPUTS, "ckpt")

STAGE = Literal["train", "valid", "test"]


@torch.no_grad()
def vae_inference(ckpt_path: str, n_flips: int = 30):
    with open(ckpt_path + ".args", "rt") as f:
        args = json.load(f)
    core = None
    for x in (CoreHAM, CoreMED, CorePAD):
        if x.__name__ == args["core"]:
            core = x()
            break
    if core is None:
        raise ValueError(f"Unknown indexer core '{args['core']}'")
    model = VAE.load_from_checkpoint(ckpt_path)

    input_height = args["model"]["input_height"]
    datasets: dict[STAGE, ModularDataset] = {
        stage: ModularDataset(
            stage,
            color_mode="RGB",
            indexer=BasicIndexer,
            indexer_core=core,
            regularizer=(
                GenericRegularizer(
                    image_size=(input_height,) * 2,
                    norm_type="imagenet",
                    scale_tensors=True,
                )
            ),
            policy="path",
        )
        for stage in ("train", "valid")
    }
    for stage, d in datasets.items():
        for x in tqdm(d):  ##type:ignore
            filename = osp.splitext(osp.basename(x[2]))[0]
            filepath = jo(EMBEDDINGS, "quantum", stage, filename)
            os.makedirs(osp.dirname(filepath), exist_ok=True)

            z = model.infer(x[0].unsqueeze(0)).squeeze().detach().cpu().numpy()
            np.save(filepath, z)

            with open(filepath + ".txt", "wt") as f:
                f.write(str(x[1]))

    R = random.Random(33)
    flip_opts = range(3)
    dupli_set = R.sample([i for i in datasets["train"]], n_flips)  ##type:ignore
    for x in tqdm(dupli_set):
        filename = osp.splitext(osp.basename(x[2]))[0]

        opt = R.choice(flip_opts)
        im = x[0].unsqueeze(0)
        match opt:
            case 0:
                im = torch.flip(im, (-1,))
                filename += "_h"
            case 1:
                im = torch.flip(im, (-2,))
                filename += "_v"
            case 2:
                im = torch.flip(im, (-1, -2))
                filename += "_hv"
        filepath = jo(EMBEDDINGS, "quantum", "train", filename)

        z = model.infer(im).squeeze().detach().cpu().numpy()
        np.save(filepath, z)

        with open(filepath + ".txt", "wt") as f:
            f.write(str(x[1]))


def main_quantum():
    vae_inference(
        jo(
            CKPT,
            "unsupe",
            "vae_quantum",
            "VAE_f47018d2c588acec28711fea3ad7cf498139e8071dfbbfbba18fe2478516f0cc.ckpt",
        )
    )


def main():
    main_quantum()


if __name__ == "__main__":
    main()
