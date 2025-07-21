from argparse import ArgumentParser

PARSER = ArgumentParser(
    description="Realiza un proceso de entrenamiento y guarda los resultados. "
    "La configuración del entorno de entrenamiento depende de los parámetros de entrada.",
)

PARSER.add_argument(
    "-d",
    "--datasets",
    choices=["ham", "siim", "slice", "med"],
    nargs="+",
    required=True,
    help="Conjuntos de datos con los que se evaluará el modelo. "
    "Para cada arquitectura seleccionada, se entrenará una instancia "
    "por cada dataset seleccionado.",
)
PARSER.add_argument(
    "-m",
    "--models",
    choices=["vit", "densenet", "inception"],
    nargs="+",
    required=True,
    help="Arquitecturas de clasificación a ser entrenadas (una o más). "
    "Se entrenará cada conjunto de datos con cada arquitectura.",
)
PARSER.add_argument(
    "-e",
    "--experiment",
    choices=["base", "global", "dann", "pretrain"],
    required=True,
    help="Cada opción corresponde a un experimento ejecutado en el trabajo original. "
    "'base' corresponde al experimento de referencia, "
    "'global' se refiere a la primera alteración con un conjunto de datos global, "
    "'dann' se trata de la segunda alteración con una DANN, y "
    "'pretrain' ejecutará la rutina compuesta de pre-entrenamiento y ajuste fino detallada "
    "en la sección 5.3 del informe. Únicamente se elegirá una opción cada vez.",
)

## Early arg parsing for convenience
if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    print("Cargando experimento...")
else:
    ARGS: Any = object()


from os.path import join as jo
from lightning.pytorch.callbacks import early_stopping
from torch.utils.data import DataLoader

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
    Indexer,
    StratifiedIndexer,
    GlobalStratifiedIndexer,
    ExcludingGlobalStratifiedIndexer,
    IndexerCore,
    CoreHAM,
    CoreMED,
    Core2020,
    Core2024,
    GaussianNoiseInt,
    GenericRegularizer,
    GenericModelRegularizer,
    ModularDataset,
    RandomRotationCrop,
    get_num_domains,
    STAGE,
)

from models.classifiers import GenericClassifier

from typing import Any, Iterable, Literal, Mapping, Sequence

ROOT = osp.dirname(__file__)
OUTPUTS = jo(ROOT, "storage", "outputs")
CKPT = jo(OUTPUTS, "ckpt")
torch.set_float32_matmul_precision("high")


def raw_collate(data: list[tuple[npt.NDArray[np.uint8], int, bool]]):
    output = list(zip(*data, strict=True))
    return output[0], torch.tensor(output[1]), torch.tensor(output[2])


def show_proportions(datasets: dict[STAGE, ModularDataset]):
    print("train")
    counts = {}
    counts_unique = {}
    paths = set()
    for k in datasets["train"].indexer.list_items:  ##type:ignore
        v = datasets["train"].indexer.items[k]  ##type:ignore
        counts[v] = counts.get(v, 0) + 1
        if k not in paths:
            counts_unique[v] = counts_unique.get(v, 0) + 1
            paths.add(k)
    print(f"{counts=}")
    print(f"{counts_unique=}")

    print("valid")
    counts = {}
    counts_unique = {}
    paths = set()
    for k in datasets["valid"].indexer.list_items:  ##type:ignore
        v = datasets["valid"].indexer.items[k]  ##type:ignore
        counts[v] = counts.get(v, 0) + 1
        if k not in paths:
            counts_unique[v] = counts_unique.get(v, 0) + 1
            paths.add(k)
    print(f"{counts=}")
    print(f"{counts_unique=}")


def train_timm(
    model_name: str,
    core: IndexerCore = CoreHAM(),
    train_id: str = "untitled",
    *,
    max_epochs: int = 32,
    batch_size: int = 32,
    lr: float = 1e-4,
    l2: float = 1e-4,
    eps: float = 1e-8,
    source_da: bool = False,
    image_size: tuple[int, int] = (224, 224),
    indexer: type[Indexer] = StratifiedIndexer,
    n_items_per_class: int | None = None,
    early_stopping: bool = True,
    gpu_preprocessing: Literal[0, 1, 2] = 0,
    cv_id: int | None = None,
    split: Mapping[STAGE, slice | Sequence[slice]] | None = None,
    **indexer_kwargs,
) -> Mapping[str, Any]:

    pre_resize_transforms = [
        v2.RandomApply([RandomRotationCrop(15)], p=0.3),  ##type:ignore
    ]
    post_resize_transforms = [
        v2.ColorJitter(
            brightness=0.25,
            contrast=0.2,
            saturation=0.2,
            hue=0.03,
        ),
        v2.RandomApply(
            [
                v2.RandomChoice(
                    (
                        GaussianNoiseInt(sigma=0.025),
                        GaussianNoiseInt(sigma=0.05),
                    )
                )
            ],
            p=0.4,
        ),
    ]

    ## Typical dataset schema
    regularizer_kwargs: dict[str, Any] = dict(
        image_size=image_size,
        norm_type="imagenet",
        flip_h=True,
        flip_v=True,
        rotate=True,
        crop_prob=0.25,
        pre_resize_transforms=pre_resize_transforms,
        post_resize_transforms=post_resize_transforms,
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
                image_size=regularizer_kwargs["image_size"],
                pre_resize_transforms=regularizer_kwargs["pre_resize_transforms"],
            )
    else:
        regularizer = GenericRegularizer(**regularizer_kwargs)
        model_regularizer = None

    datasets: dict[STAGE, ModularDataset] = {
        stage: ModularDataset(
            stage,
            color_mode="RGB",
            indexer=indexer,
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
            split=split,
            n_items=n_items_per_class,
            **indexer_kwargs,
        )
        for stage in ("train", "valid")  # , "test")
    }

    show_proportions(datasets)

    data_workers = max((os.cpu_count() or 1) - 1, 1)
    dataloaders: dict[STAGE, DataLoader] = {
        stage: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=stage == "train",
            num_workers=data_workers,
            collate_fn=collate_fn,
            pin_memory=False,  ## Should investigate further
        )
        for stage, dataset in datasets.items()
    }

    num_domains = get_num_domains(datasets["train"])
    model = GenericClassifier(
        num_domains,
        model_name,
        lr=lr,
        l2=l2,
        optim_hyper_params={"eps": eps},
        source_da=source_da,
        model_regularizer=model_regularizer,
        other_notes={"cv_id": cv_id, "indexer_kwargs": indexer_kwargs},
    ).to("cuda")

    log_path = jo(
        OUTPUTS,
        "classifiers",
        train_id,
        core.__class__.__name__.replace("Core", ""),
        osp.basename(model_name),
    )
    if not osp.exists(log_path):
        os.makedirs(log_path)
    logger = loggers.CSVLogger(
        log_path,
        name=None,
    )

    trainer_callbacks = []
    if early_stopping:
        trainer_callbacks.append(
            callbacks.EarlyStopping(
                "valid_accuracy_macro",
                patience=3,
                verbose=True,
                mode="max",  ## FIXED
            )
        )

    cv_id_path = "" if cv_id is None else f"{cv_id}:"
    ckpt_path = jo(
        CKPT,
        "supe",
        train_id,
        f"{model_name}_{core.__class__.__name__}@{cv_id_path}.ckpt",
    )
    os.makedirs(osp.dirname(ckpt_path), exist_ok=True)
    count = 0
    path_comps = ckpt_path.split(".")
    while osp.isfile(".".join(path_comps[:-1] + [f"v{count}"] + path_comps[-1:])):
        count += 1

    actual_ckpt_path = ".".join(path_comps[:-1] + [f"v{count}"] + path_comps[-1:])

    trainer_callbacks.append(
        callbacks.ModelCheckpoint(
            dirpath=osp.dirname(actual_ckpt_path),
            # It appends to add the extension no matter what...
            filename=osp.splitext(osp.basename(actual_ckpt_path))[0],
            monitor="valid_f1",
            mode="max",
            verbose=True,
            save_last=False,
            save_top_k=1,
            enable_version_counter=False,
        )
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=trainer_callbacks,
        logger=logger,
        log_every_n_steps=0,
        enable_checkpointing=True,
    )

    trainer.fit(
        model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["valid"],
    )

    results = trainer.validate(model, dataloaders=dataloaders["valid"])
    output: dict[str, Any] = dict(results[0])
    output["actual_ckpt_path"] = actual_ckpt_path

    return output


def timm_xval(n_folds: int, **train_kwargs):
    if not (n_folds >= 2):
        raise ValueError(f"'{n_folds=}' (must be >= 2).")

    fold_set = tuple(slice(i / n_folds, (i + 1) / n_folds) for i in range(n_folds))
    for idx, fold in enumerate(fold_set):
        print(f"HOLDOUT: fold {idx}")
        metrics = train_timm(
            **train_kwargs,
            cv_id=idx,
            split={
                "train": tuple(x for i, x in enumerate(fold_set) if idx != i),
                "valid": fold,
            },
        )
        print(metrics)


def pretrain_xval(
    model_name: str,
    core: IndexerCore,
    train_id: str,
    n_folds: int,
    **pretrain_kwargs,
) -> None:

    if not (n_folds >= 2):
        raise ValueError(f"'{n_folds=}' (must be >= 2).")

    fold_set = tuple(slice(i / n_folds, (i + 1) / n_folds) for i in range(n_folds))
    for cv_id, fold in enumerate(fold_set):
        split = {
            "train": tuple(x for i, x in enumerate(fold_set) if cv_id != i),
            "valid": fold,
        }

        ## Init params
        image_size = (299, 299) if "inception" in model_name else (224, 224)
        batch_size = 32
        lr_pretrain = 5e-4
        lr_finetune = 5e-5
        l2 = 0
        eps = 1e-8

        ## Dict on FS
        ckpt_notation_dir = jo(OUTPUTS, "pretrain_paths")
        os.makedirs(ckpt_notation_dir, exist_ok=True)
        ckpt_notation_path = jo(
            ckpt_notation_dir,
            f"{train_id}:{model_name.replace('timm/','')}_"
            f"{core.__class__.__name__.replace('Core','')}.txt",
        )

        if osp.isfile(ckpt_notation_path):
            with open(ckpt_notation_path, "r", encoding="utf-8") as f:
                ckpt_path = f.read().strip()
        else:
            print(f"PRETRAINING {core.__class__.__name__} {model_name}...")
            metrics = train_timm(
                model_name=model_name,
                core=core,
                **pretrain_kwargs,
                image_size=image_size,
                batch_size=batch_size,
                indexer=ExcludingGlobalStratifiedIndexer,
                gpu_preprocessing=0,
                source_da=True,
                #
                n_items_per_class=None,
                lr=lr_pretrain,
                l2=l2,
                eps=eps,
                train_id="pre",
            )
            ckpt_path = str(metrics["actual_ckpt_path"])
            with open(ckpt_notation_path, "w", encoding="utf-8") as f:
                f.write(ckpt_path)

        ## TRAINING
        print(f"TRAINING {core.__class__.__name__} {model_name} @ HOLDOUT {cv_id}...")
        pre_resize_transforms = [
            v2.RandomApply([RandomRotationCrop(15)], p=0.3),  ##type:ignore
        ]
        post_resize_transforms = [
            v2.ColorJitter(
                brightness=0.25,
                contrast=0.2,
                saturation=0.2,
                hue=0.03,
            ),
            v2.RandomApply(
                [
                    v2.RandomChoice(
                        (
                            GaussianNoiseInt(sigma=0.025),
                            GaussianNoiseInt(sigma=0.05),
                        )
                    )
                ],
                p=0.4,
            ),
        ]

        ## Typical dataset schema
        regularizer_kwargs: dict[str, Any] = dict(
            image_size=image_size,
            norm_type="imagenet",
            flip_h=True,
            flip_v=True,
            rotate=True,
            crop_prob=0.25,
            pre_resize_transforms=pre_resize_transforms,
            post_resize_transforms=post_resize_transforms,
        )
        ## We may offload more preprocessing to the GPU,
        ## might add levels to manage that later
        regularizer = GenericRegularizer(**regularizer_kwargs)

        datasets: dict[STAGE, ModularDataset] = {
            stage: ModularDataset(
                stage,
                color_mode="RGB",
                indexer=GlobalStratifiedIndexer,
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
                split=split,
                n_items=(
                    17_000 if isinstance(core, Core2024) and stage == "train" else None
                ),
                non_global=True,
            )
            for stage in ("train", "valid")  # , "test")
        }

        show_proportions(datasets)

        data_workers = max((os.cpu_count() or 1) - 1, 1)
        dataloaders: dict[STAGE, DataLoader] = {
            stage: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=stage == "train",
                num_workers=data_workers,
                pin_memory=False,  ## Should investigate further
            )
            for stage, dataset in datasets.items()
        }

        model = GenericClassifier.load_from_checkpoint(
            ckpt_path,
            lr=lr_finetune,
            disable_source_da=True,
        ).to("cuda")

        log_path = jo(
            OUTPUTS,
            "classifiers",
            train_id,
            core.__class__.__name__.replace("Core", ""),
            osp.basename(model_name),
        )
        if not osp.exists(log_path):
            os.makedirs(log_path)
        logger = loggers.CSVLogger(
            log_path,
            name=None,
        )

        trainer_callbacks = []
        if early_stopping:
            trainer_callbacks.append(
                callbacks.EarlyStopping(
                    "valid_f1",
                    patience=8,
                    min_delta=0.01,
                    verbose=True,
                    mode="max",  ## FIXED
                )
            )

        cv_id_path = "" if cv_id is None else f"{cv_id}:"
        ckpt_path = jo(
            CKPT,
            "supe",
            train_id + "",
            f"{model_name}_{core.__class__.__name__}@{cv_id_path}_FINETUNE.ckpt",
        )
        os.makedirs(osp.dirname(ckpt_path), exist_ok=True)
        count = 0
        path_comps = ckpt_path.split(".")
        while osp.isfile(".".join(path_comps[:-1] + [f"v{count}"] + path_comps[-1:])):
            count += 1

        actual_ckpt_path = ".".join(path_comps[:-1] + [f"v{count}"] + path_comps[-1:])

        trainer_callbacks.append(
            callbacks.ModelCheckpoint(
                dirpath=osp.dirname(actual_ckpt_path),
                # It appends it extension no matter what...
                filename=osp.splitext(osp.basename(actual_ckpt_path))[0],
                monitor="valid_f1",
                mode="max",
                verbose=True,
                save_last=False,
                save_top_k=1,
                enable_version_counter=False,
            )
        )

        trainer = L.Trainer(
            max_epochs=128,
            callbacks=trainer_callbacks,
            logger=logger,
            log_every_n_steps=0,
            enable_checkpointing=True,
        )

        trainer.fit(
            model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["valid"],
        )

        results = trainer.validate(model, dataloaders=dataloaders["valid"])
        output: dict[str, Any] = dict(results[0])

        output["actual_ckpt_path"] = actual_ckpt_path


def standard_entrypoint(
    cores: Iterable[IndexerCore],
    model_names: Iterable[str],
    train_id: str,
    non_global: bool = True,
    source_da: bool = False,
) -> None:
    for c in cores:
        kwargs = {
            "train_id": train_id,
            "lr": 5e-4,
            "l2": 0,
            "eps": 1e-8,
            "non_global": non_global,
            "source_da": source_da,
            "n_items_per_class": 340 if isinstance(c, CoreMED) else 17_000,
        }

        for name in model_names:
            print(f"TRAINING {c.__class__.__name__} {name}...")
            timm_xval(
                n_folds=6,
                #
                model_name=name,
                core=c,
                **kwargs,
                # 'inception' can work with a larger image size
                image_size=(299, 299) if "inception" in name else (224, 224),
                batch_size=32,
                indexer=GlobalStratifiedIndexer,
                gpu_preprocessing=0,
            )


def finetune_entrypoint(
    cores: Iterable[IndexerCore],
    model_names: Iterable[str],
):
    for name in model_names:
        for c in cores:
            pretrain_xval(name, c, train_id="pretrain", n_folds=6)


if __name__ == "__main__":
    model_translations = {
        "inception": "timm/inception_v3.tv_in1k",
        "vit": "timm/vit_small_patch16_224.augreg_in21k_ft_in1k",
        "densenet": "timm/densenet201.tv_in1k",
    }
    core_translations = {
        "ham": CoreHAM(),
        "siim": Core2020(extended=True),
        "slice": Core2024(extended=True),
        "med": CoreMED(),
    }

    model_names = map(model_translations.__getitem__, ARGS.models)
    cores = map(core_translations.__getitem__, ARGS.datasets)

    if ARGS.experiment == "pretrain":
        finetune_entrypoint(cores, model_names)
    else:
        non_global, source_da = {
            "base": (True, False),
            "global": (False, False),
            "dann": (False, True),
        }[ARGS.experiment]
        standard_entrypoint(
            cores,
            model_names,
            ARGS.experiment,
            non_global,
            source_da,
        )
