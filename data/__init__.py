from os.path import join as jo
from torch.utils.data import Dataset

import cv2
import numpy as np
import numpy.typing as npt
import os.path as osp
import random
import torch

from typing import Any, Callable, Literal, Self, TypeVarTuple

from . import components
from .components import *
from . import skin
from .skin import *

Ts = TypeVarTuple("Ts")

__all__ = [
    "generate_image",
    "ModularDataset",
]

__all__.extend(components.__all__)
__all__.extend(skin.__all__)

FILE = jo(osp.dirname(__file__))
ROOT = jo(FILE, "..")


def get_num_domains(dataset: "ModularDataset") -> int:
    if dataset.classes is None:
        max_class = 0
        for i in dataset.indexer:
            if i[1] > max_class:
                max_class = i[1]
        return max_class + 1
    else:
        return len(dataset.classes)


def generate_image(
    path: str,
    format: Literal["RGB", "L"] = "L",
) -> npt.NDArray[np.uint8]:
    x = cv2.imread(path)
    match format:
        case "RGB":
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        case "L":
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    return x  ## type:ignore


class ModularDataset(Dataset[Any]):
    r"""
    Typical dataset implementation, except most of the
    functionality is configurable via the objects defined
    in the 'components' sub-module that are passed to the
    ``__init__`` args.

    There's also the 'dataset policy' standard, which mainly
    allows the quick swapping of models with different data
    formats.
    """

    R = random.Random(33)
    DATASET_SOURCES: tuple[str, ...] = (
        "bcn2000",
        "ham10000",
        "isic-2020",
        "isic-2024",
        "med-node",
        "pad-ufes-20",
    )

    def __init__(
        self,
        stage: STAGE,
        color_mode: Literal["RGB", "L"] = "RGB",
        indexer: type[Indexer] = BasicIndexer,
        indexer_core: IndexerCore = CoreHAM(),
        regularizer: Regularizer = GenericRegularizer(
            image_size=(128, 128),
            flip_h=True,
            flip_v=True,
            crop_prob=0.5,
        ),
        policy: str = "basic",
        class_translations: dict[str, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.color_mode = color_mode
        self.indexer = indexer(stage, indexer_core, **kwargs)
        self.policy = policy
        self.regularizer = regularizer

        if class_translations is not None:
            self.classes = class_translations

    def __len__(self):
        return self.indexer.__len__()

    def __getitem__(self, idx: int):
        return self.SUPPORTED_POLICIES[self.policy](self, idx)

    @property
    def classes(self) -> dict[str, int] | None:
        return getattr(self.indexer, "classes", None)

    @classes.setter
    def classes(self, x: dict[str, int]) -> None:
        if self.classes is None:
            raise RuntimeError("No ``classes`` defined on indexer")
        else:
            setattr(self.indexer, "classes", x)

    def _generate_image(self, path: str) -> npt.NDArray[np.uint8]:
        return generate_image(path, self.color_mode)  ##type:ignore

    def generation_pipeline(self, path: str):
        return self.regularizer(self._generate_image(path))

    r"""Dataset policy implementation ahead.
    A dataset's policy is dictated by the model in question,
    as different models may require different data structures
    (i.e. two samples at once). The data policy a model requires
    is stored as a str in its ``SUPPORTED_POLICIES`` class
    attribute, ans should be passed to the dataset through the
    'policy' ``__init__`` arg.

    Each data policy has its respective method defined here.
    """

    @property
    def policy(self) -> str:
        return self.__policy

    @policy.setter
    def policy(self, x: str):
        assert (
            x in self.SUPPORTED_POLICIES.keys()
        ), "Policy not supported or implemented."
        self.__policy = x

    def basic_policy(self, idx: int) -> tuple[torch.Tensor, Any]:
        path, label = self.indexer[idx]
        item = self.generation_pipeline(path)
        return item, label

    def add_path(
        self,
        policy: Callable[[Self, int], tuple[*Ts]],
    ) -> Callable[[Self, int], tuple[*Ts, str]]:
        def path_wrapper(
            self: Self,
            idx: int,
        ) -> tuple[*Ts, str]:
            path, _ = self.indexer[idx]
            return (*policy(self, idx), path)

        return path_wrapper

    def double_policy(self, idx: int) -> tuple[tuple[torch.Tensor, Any], ...]:
        output = []
        path, label = self.indexer[idx]
        item = self.generation_pipeline(path)
        output.append((item, label))

        for _ in range(1):
            path, label = self.R.choice(self.indexer)
            item = self.generation_pipeline(path)
            output.append((item, label))

        return tuple(output)

    def get_node(self, idx: int | None = None) -> tuple[torch.Tensor, Any]:
        if idx is None:
            node_label = None
        else:
            node_path, node_label = self.indexer.__getitem__(idx)
        while node_label != self.classes[ANCHOR_DOMAIN]:  ##type:ignore
            node_path, node_label = self.R.choice(self.indexer)
        node_item = self.generation_pipeline(node_path)  ##type:ignore
        return node_item, node_label

    def get_non_node(self, idx: int | None = None) -> tuple[torch.Tensor, Any]:
        if idx is None:
            node_label = self.classes[ANCHOR_DOMAIN]  ##type:ignore
        else:
            node_path, node_label = self.indexer.__getitem__(idx)
        while node_label == self.classes[ANCHOR_DOMAIN]:  ##type:ignore
            node_path, node_label = self.R.choice(self.indexer)
        node_item = self.generation_pipeline(node_path)  ##type:ignore
        return node_item, node_label

    def basic_node_policy(self, idx: int) -> tuple[tuple[torch.Tensor, Any], ...]:
        return (self.basic_policy(idx), self.get_node())

    def double_node_policy(self, idx: int) -> tuple[tuple[torch.Tensor, Any], ...]:
        return (*self.double_policy(idx), self.get_node())

    def binary_policy(self, idx: int) -> tuple[tuple[torch.Tensor, Any], ...]:
        return (self.get_non_node(idx), self.get_node(idx))

    def non_node_policy(self, idx: int) -> tuple[tuple[torch.Tensor, Any], ...]:
        return (self.get_non_node(), self.basic_policy(idx))

    def fake_policy(self, idx: int) -> tuple[torch.Tensor, Any, bool]:
        path, label = self.indexer[idx]
        item = self.generation_pipeline(path)
        return item, label, "fake" in osp.basename(path)

    def source_fake_policy(self, idx: int) -> tuple[torch.Tensor, Any, bool, int]:
        path, label = self.indexer[idx]
        item = self.generation_pipeline(path)
        source = None
        for i, domain in enumerate(self.DATASET_SOURCES):
            if domain in path:
                if source is not None:
                    raise RuntimeError(
                        f"Domains '{domain}' and '{self.DATASET_SOURCES[source]}' "
                        f"both in path '{path}'."
                    )
                source = i

        if source is None:
            raise RuntimeError(f"No domain found in path '{path}'.")

        return item, label, "fake" in osp.basename(path), source

    @property
    def SUPPORTED_POLICIES(self) -> dict[str, Callable[[Self, int], Any]]:
        return {
            "basic": type(self).basic_policy,
            "path": self.add_path(type(self).basic_policy),
            "double": type(self).double_policy,
            "basic_node": type(self).basic_node_policy,
            "double_node": type(self).double_node_policy,
            "binary": type(self).binary_policy,
            "binary_path": self.add_path(type(self).binary_policy),
            "non_node": type(self).non_node_policy,
            "non_node_path": self.add_path(type(self).non_node_policy),
            "fake": type(self).fake_policy,
            "fake_path": self.add_path(type(self).fake_policy),
            "fake_source": type(self).source_fake_policy,
            "fake_source_path": self.add_path(type(self).source_fake_policy),
        }
