r"""
Dataset architecture components.

The dataset components do the following:

    - Indexer: We offload __len__ and the ``(int) -> path``
    application in __getitem__ to this component, mainly
    responsible for sampling strategies, train-val-test
    splits, and the like.

    - IndexerCore: Responsible for handling the total set of
    elements to work with, plugs directly into the Indexer.

    - Regularizer: Mainly manages image (and possibly
    label) augmentations. If there were more than one
    image source (i.e. multiple sensors), it would also
    handle said data's concatenation.

To make these components interchangeable, we detach them from
the Dataset class and place them as ``__init__`` args.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from tqdm import tqdm
from multiprocessing import Pool
from torchvision.transforms import v2

import math
import numpy as np
import numpy.typing as npt
import os
import os.path as osp
import random
import torch
import torchvision.transforms.functional as F
import warnings

from typing import (
    Any,
    Callable,
    cast,
    Literal,
    Mapping,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
    TypeVarTuple,
)

T = TypeVar("T")
TS = TypeVarTuple("TS")

__all__ = [
    "IMAGE_SIZE",
    "STAGE",
    "FLOAT_SLICE",
    "IndexerCore",
    "CoreFilter",
    "CoreMerge",
    "CoreSpecs",
    "Indexer",
    "BasicIndexer",
    "StratifiedIndexer",
    "Regularizer",
    "GenericRegularizer",
    "ModelRegularizer",
    "GenericModelRegularizer",
    "GaussianNoiseInt",
    "RandomRotationCrop",
]

FILE = osp.dirname(__file__)
IMAGE_SIZE = (224, 224)

STAGE = Literal["train", "valid", "test"]
if TYPE_CHECKING:
    FLOAT_SLICE = slice[float | None, float | None, None]
else:
    FLOAT_SLICE = slice


class IndexerCore(ABC):
    """
    Returns the total set of images by class to be
    available for the sampling policy to manage.
    """

    @abstractmethod
    def __call__(self) -> dict[str, list[str]]: ...


@dataclass
class CoreSpecs:
    class_idx: int
    class_total: int
    global_idx: int
    global_total: int


class CoreFilter(IndexerCore):
    r"""Applies 'filter' function to input IndexerCore."""

    def __init__(
        self,
        base_core: IndexerCore,
        fn: Callable[[str, str, CoreSpecs], bool],
        progress_bar: bool = False,
    ) -> None:
        super().__init__()
        self.base_core = base_core
        self.fn = fn
        self.progress_bar = progress_bar
        self.cache: dict[str, list[str]] | None = None

    @staticmethod
    def evaluate_and_send(
        fn: Callable[[T, *TS], bool],
        x: T,
        *args: *TS,
    ) -> tuple[T, bool]:
        return x, fn(x, *args)

    def __call__(self) -> dict[str, list[str]]:
        if self.cache is not None:
            return self.cache

        pool = Pool(os.cpu_count() or 1)
        wrapper = tqdm if self.progress_bar else lambda x: x

        core = self.base_core()
        output_raw = {
            k: pool.starmap(
                self.evaluate_and_send,
                (
                    (
                        self.fn,
                        x,
                        k,
                        CoreSpecs(
                            class_i,
                            len(v),
                            global_i,
                            sum(len(set_imgs) for set_imgs in core.values()),
                        ),
                    )
                    for (class_i, x) in enumerate(v)
                ),
            )
            for global_i, (k, v) in enumerate(wrapper(core.items()))
        }
        output = {k: [i for (i, c) in v if c] for k, v in output_raw.items()}
        self.cache = {k: v for k, v in output.items() if v}
        return self.cache


class CoreMerge(IndexerCore):
    r"""Merges IndexerCore(s). An optional array of sets can be provided,
    functioning as an indicator of equivalent class annotations."""

    def __init__(
        self,
        *cores: IndexerCore,
        equiv_sets: Sequence[tuple[str, set[str]]] | None = None,
        include_stray_classes: bool = False,
        n_items: int | None = None,
    ) -> None:
        super().__init__()
        if equiv_sets is None:
            equiv_sets = tuple()

        ## Intersection of any two sets must always be empty
        global_set: set[str] = set()
        sum_set_length = 0
        for idx, x in equiv_sets:
            global_set.update(x)
            sum_set_length += len(x)
        if len(global_set) < sum_set_length:
            raise ValueError(
                "Intersection of any two sets of the "
                "provided array must always be empty."
            )

        self.cores: list[IndexerCore] = list(cores)
        self.equiv_sets = equiv_sets
        self.include_stray_classes = include_stray_classes
        self.n_items = n_items
        self.cache: dict[str, list[str]] | None = None

    @staticmethod
    def find_class_in_equiv_set(
        class_: str,
        equiv_sets: Sequence[tuple[str, set[str]]],
    ) -> str | None:
        for set_idx, eq_set in equiv_sets:
            if class_ in eq_set:
                return set_idx
                break
        else:
            ## If class not mentioned in equivalence set
            return None

    def __call__(self) -> dict[str, list[str]]:
        if self.cache is not None:
            return self.cache

        output: defaultdict[str, list[str]] = defaultdict(list)
        for core in self.cores:
            for class_, samples in core().items():
                set_alias = self.find_class_in_equiv_set(class_, self.equiv_sets)

                r = random.Random(33)
                r.shuffle(samples)
                if self.n_items is not None:
                    idx = class_ if set_alias is None else set_alias
                    if idx in output.keys():
                        upper_bound = max(self.n_items - len(output[idx]), 0)
                    else:
                        upper_bound = self.n_items
                else:
                    upper_bound = None

                if set_alias is not None:
                    output[set_alias].extend(samples[:upper_bound])
                elif self.include_stray_classes:
                    output[class_].extend(samples[:upper_bound])

        self.cache = dict(output)
        return self.cache


class Indexer(Sequence, ABC):
    """
    Manages the (idx:int) -> (datapath:str,...) correspondence.
    In charge of defining sampling methodology, etc.
    """

    @abstractmethod
    def __init__(self, stage: STAGE, core: IndexerCore, **kwargs) -> None: ...
    @abstractmethod
    def __len__(self) -> int: ...

    ## One item or multiple items
    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[str, int]: ...  ##type:ignore
    @staticmethod
    def float_slicing(
        array: Sequence[T],
        idx: FLOAT_SLICE | Sequence[FLOAT_SLICE],
    ) -> list[T]:
        ## Supports both slices and sequences of slices
        if isinstance(idx, Sequence):
            union_splits_raw = idx
        else:
            union_splits_raw = (idx,)

        ## Any float is understood as a fraction of the array,
        union_splits = [
            {i: getattr(split, i) for i in ("start", "stop", "step")}
            for split in union_splits_raw
        ]
        for split in union_splits:
            for k, v in split.items():
                if isinstance(v, (float, int)) and v >= 0 and v <= 1:
                    split[k] = round(v * len(array))

        return [
            i
            for split in union_splits
            for i in array[split["start"] : split["stop"] : split["step"]]
        ]


class BasicIndexer(Indexer):
    STAGE_SPLIT: dict[STAGE, FLOAT_SLICE] = {
        "train": slice(0.85),
        "valid": slice(0.85, 1),
        "test": slice(1),
    }

    def __init__(
        self,
        stage: STAGE,
        core: IndexerCore,
        split: Mapping[STAGE, FLOAT_SLICE | Sequence[FLOAT_SLICE]] | None = None,
        classes_idx: dict[str, int] | None = None,
    ) -> None:
        self.stage = stage
        core_content = core()
        if split is None:
            split = copy(self.STAGE_SPLIT)

        ## Class name to int
        if classes_idx is None:
            self.classes: dict[str, int] = {}
            for i, k in enumerate(core_content.keys()):
                self.classes[k] = i
        else:
            self.classes = classes_idx
            all_class_idx = list(classes_idx.values())
            not_seen_class_idx = set(range(max(all_class_idx))).difference(
                all_class_idx
            )
            if not_seen_class_idx:
                warnings.warn(
                    RuntimeWarning(
                        f"Some class idxs assigned but not seen: {not_seen_class_idx}"
                    )
                )

        ## Stores classes per item
        self.items: dict[str, str] = {}
        ## Stores items
        self.list_items: list[str] = []
        for class_, folder in core_content.items():
            real_folder = [i for i in folder if "fake" not in i]
            fake_folder = [i for i in folder if "fake" in i]
            r = random.Random(1234)
            r.shuffle(real_folder)
            r.shuffle(fake_folder)
            for i in self.float_slicing(real_folder, split[self.stage]):
                self.items[i] = class_
                self.list_items.append(i)
            if self.stage == "train":
                for i in fake_folder:
                    self.items[i] = class_
                    self.list_items.append(i)
        r = random.Random(1234)
        r.shuffle(self.list_items)

    def __len__(self) -> int:
        return self.list_items.__len__()

    def __getitem__(self, idx: int) -> tuple[str, int]:
        item = self.list_items[idx]
        return item, self.classes[self.items[item]]


class StratifiedIndexer(Indexer):
    STAGE_SPLIT: dict[STAGE, FLOAT_SLICE] = {
        "train": slice(0.85),
        "valid": slice(0.85, 1),
        "test": slice(1),
    }

    def __init__(
        self,
        stage: STAGE,
        core: IndexerCore,
        n_items: int | None = None,
        split: Mapping[STAGE, FLOAT_SLICE | tuple[FLOAT_SLICE]] | None = None,
        classes_idx: dict[str, int] | None = None,
    ) -> None:
        self.stage = stage
        core_content = core()
        if split is None:
            split = copy(self.STAGE_SPLIT)

        ## Class name to int
        if classes_idx is None:
            self.classes: dict[str, int] = {}
            for i, k in enumerate(core_content.keys()):
                self.classes[k] = i
        else:
            self.classes = classes_idx
            all_class_idx = list(classes_idx.values())
            not_seen_class_idx = set(range(max(all_class_idx))).difference(
                all_class_idx
            )
            if not_seen_class_idx:
                warnings.warn(
                    RuntimeWarning(
                        f"Some class idxs assigned but not seen: {not_seen_class_idx}"
                    )
                )
            for k in core_content.keys():
                if k not in self.classes:
                    raise ValueError(f"Class '{k}' not defined in given classes_idx.")

        ## Stores classes per item
        self.items: dict[str, str] = {}
        self.real_items: dict[int, list[str]] = {i: [] for i in self.classes.values()}
        self.fake_items: dict[int, list[str]] = {i: [] for i in self.classes.values()}

        for class_, folder in core_content.items():
            real_folder = [i for i in folder if "fake" not in i]
            fake_folder = [i for i in folder if "fake" in i]
            r = random.Random(1234)
            r.shuffle(real_folder)
            r.shuffle(fake_folder)
            for i in self.float_slicing(real_folder, split[self.stage]):
                self.items[i] = class_
                self.real_items[self.classes[class_]].append(i)
            if self.stage == "train":
                for i in fake_folder:
                    self.items[i] = class_
                    self.fake_items[self.classes[class_]].append(i)

        if n_items is None:
            n_items = max(*[len(i) for i in self.real_items.values()])

        ## Stores items
        self.list_items: list[str] = []

        if stage == "train":
            for c, i in self.classes.items():
                r = random.Random(1234)
                r.shuffle(self.real_items[i])
                self.list_items.extend(self.real_items[i][:n_items])

                left_items = n_items - len(self.real_items[i])
                if left_items > 0:
                    r = random.Random(1234)
                    r.shuffle(self.fake_items[i])
                    self.list_items.extend(self.fake_items[i][:left_items])
                    left_items -= len(self.fake_items[i])

                ## Oversampling
                if left_items > 0:
                    warnings.warn(
                        f"Not enough items for class '{c}', oversampling...",
                        RuntimeWarning,
                    )

                while left_items > 0:
                    self.list_items.extend(self.real_items[i][:left_items])
                    left_items -= len(self.real_items[i])
                    self.list_items.extend(self.fake_items[i][:left_items])
                    left_items -= len(self.fake_items[i])
                    if len(self.real_items[i]) + len(self.fake_items[i]) == 0:
                        warnings.warn(f"No items for class '{c}'!")
                        break
        else:
            for i in self.classes.values():
                ## Only store real items in validation/test
                self.list_items.extend(self.real_items[i])

        r = random.Random(1234)
        r.shuffle(self.list_items)
        print(self.classes)

    def __len__(self) -> int:
        return self.list_items.__len__()

    def __getitem__(self, idx: int) -> tuple[str, int]:
        item = self.list_items[idx]
        return item, self.classes[self.items[item]]


class Regularizer(ABC):
    r"""
    Takes raw input (e.g. PIL images, numpy arrays)
    and transforms them into torch data, along with
    standardization and augmentation techniques.
    """

    @abstractmethod
    def __call__(
        self,
        array: npt.NDArray[np.uint8],
    ) -> torch.Tensor: ...
    @property
    @abstractmethod
    def image_size(self) -> tuple[int, int] | None: ...

    @property
    @abstractmethod
    def scale_tensors(self) -> bool: ...


class GaussianNoiseInt(v2.GaussianNoise):
    @staticmethod
    def gaussian_int(
        transform: v2.GaussianNoise,
        x: torch.Tensor,
    ) -> torch.Tensor:
        r"""Fine, I'll do it myself.
        Simple adaptation of the v2 function to work with ints through the power
        of approximation. Props to the original implementation.
        """
        assert x.dtype == torch.uint8
        ## 16 bits should be enough
        noise = (
            torch.randn_like(
                x,
                dtype=torch.float32,
                device=x.device,
            )
            * (transform.sigma * 255)
        ).to(torch.int16) + round(transform.mean * 255)
        output = x + noise
        if transform.clip:
            output = output.clamp(0, 255)
        return output.to(torch.uint8)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        assert isinstance(inpt, torch.Tensor)
        if inpt.dtype == torch.uint8:
            return GaussianNoiseInt.gaussian_int(self, inpt)
        else:
            return super()._transform(inpt, params)


class RandomRotationCrop(v2.RandomRotation):
    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        assert isinstance(inpt, torch.Tensor)
        if self.expand:
            raise ValueError(
                "'expand' argument being True does not make sense in this context."
            )
        output = super()._transform(inpt, params)
        radians = math.radians(params["angle"])
        cos_a = abs(math.cos(radians))
        sin_a = abs(math.sin(radians))

        h, w = inpt.shape[-2:]
        bb_h = w * sin_a + h * cos_a
        bb_w = w * cos_a + h * sin_a
        scale = max(bb_h / h, bb_w / w)

        return F.center_crop(output, [int(h / scale), int(w / scale)])


class GenericRegularizer(Regularizer):
    r"""
    Previously there were a plethora of Regularizer
    sub-classes for different purposes, but ended up
    deciding it would just be easier to fit all those
    use cases on one class by the use of __init__
    parameters.
    """

    TO_TENSOR = v2.ToImage()

    def __init__(
        self,
        image_size: tuple[int, int] | None = IMAGE_SIZE,
        norm_type: Literal["flat", "imagenet"] | None = "imagenet",
        *,
        ## [0,255] uint8 -> [0,1] float32
        scale_tensors: bool = True,
        ## "Prebuilt" transforms
        flip_h: bool = False,
        flip_v: bool = False,
        rotate: bool = False,
        crop_prob: float = 0.0,
        pre_resize_transforms: v2.Transform | Sequence[v2.Transform] = [],
        post_resize_transforms: v2.Transform | Sequence[v2.Transform] = [],
    ) -> None:
        super().__init__()
        self.__image_size = image_size
        I = v2.Identity()

        self.resize = v2.Resize(image_size) if image_size is not None else I
        self.normalize = self.get_normalize(norm_type) if norm_type is not None else I
        self.scale_fn = v2.ToDtype(torch.float32, scale=True) if scale_tensors else I

        if isinstance(pre_resize_transforms, Sequence):
            if not pre_resize_transforms:
                pre_resize_transforms = [v2.Identity()]
            self.pre_resize_transforms = v2.Compose(pre_resize_transforms)
        else:
            self.pre_resize_transforms = v2.Compose([pre_resize_transforms])

        if isinstance(post_resize_transforms, Sequence):
            if not post_resize_transforms:
                post_resize_transforms = [v2.Identity()]
            self.post_resize_transforms = v2.Compose(post_resize_transforms)
        else:
            self.post_resize_transforms = v2.Compose([post_resize_transforms])

        if flip_h:
            self.post_resize_transforms.add_module("flip_h", v2.RandomHorizontalFlip())
        if flip_v:
            self.post_resize_transforms.add_module("flip_v", v2.RandomVerticalFlip())
        if rotate:
            ## Please note, "expand" should do nothing here, but just in case,
            self.post_resize_transforms.add_module(
                "rotate",
                v2.RandomChoice(
                    [
                        I,
                        (
                            v2.RandomRotation(90 * i, expand=True)  ##type:ignore
                            for i in range(1, 4)
                        ),
                    ]
                ),
            )
        if crop_prob > 0:
            if self.image_size is None:
                raise ValueError(
                    "image_size must be defined for ``crop`` aug to be applied"
                )
            self.post_resize_transforms.add_module(
                "crop",
                self.get_crop_transform(
                    self.image_size,
                    scale=(0.8, 1),
                    ratio=(0.9, 1.1),
                    p=crop_prob,
                ),
            )

    @property
    def image_size(self) -> tuple[int, int] | None:
        return self.__image_size

    @property
    def scale_tensors(self) -> bool:
        return not isinstance(self.scale_fn, v2.Identity)

    @staticmethod
    def get_normalize(norm_type: Literal["flat", "imagenet"]) -> v2.Lambda:
        match norm_type:
            case "flat":
                mean = std = [0.5] * 3
            case "imagenet":
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
        normalize = [
            v2.Normalize(
                mean=[np.mean(mean).__float__()], std=[np.mean(std).__float__()]
            ),
            v2.Normalize(mean=mean, std=std),
        ]
        return v2.Lambda(
            lambda x: normalize[0](x) if x.shape[0] == 1 else normalize[1](x)
        )

    @staticmethod
    def get_crop_transform(
        size: int | Sequence[int],
        scale: tuple[float, float] = (0.8, 1.0),
        ratio: tuple[float, float] = (0.9, 1.1),
        p: float = 0.5,
        **kwargs,
    ) -> v2.RandomApply:
        ## size, scale, ratio, interpolation, antialias
        return v2.RandomApply([v2.RandomResizedCrop(size, scale, ratio, **kwargs)], p)

    @staticmethod
    def compose(
        transforms: Sequence[v2.Transform],
        x: torch.Tensor,
    ) -> torch.Tensor:
        assert transforms
        for transform in transforms:
            x = transform(x)
        return x

    def pre_resize(self, x: torch.Tensor) -> torch.Tensor:
        return self.compose(
            [
                self.pre_resize_transforms,
                self.resize,
            ],
            x,
        )

    def post_resize(self, x: torch.Tensor) -> torch.Tensor:
        return self.compose(
            [
                self.post_resize_transforms,
                self.scale_fn,
                self.normalize,
            ],
            x,
        )

    def __call__(self, x: npt.NDArray[np.uint8] | torch.Tensor) -> torch.Tensor:
        x = cast(torch.Tensor, self.TO_TENSOR(x))
        x = self.pre_resize(x)
        x = self.post_resize(x)
        return x


class ModelRegularizer(ABC):
    r"""
    Adaptation of the ``Regularizer`` component to
    work on the model's device (after performing
    Tensor.to(device)), thus taking advantage of GPU
    compute and/or minimizing possible CPU -> GPU
    transfer bottlenecks (mostly because the ``int8``
    dtype is very much lighter than ``float32``).

    Planned to be used to facilitate running tests
    on cpu while gpu is training, but ended up being
    unused.
    """

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    @property
    def device(self) -> torch.device | str | int:
        return self.__device

    @device.setter
    def device(self, x: torch.device | str | int):
        self.__device = x
        for _, v in vars(self).items():
            if isinstance(v, torch.nn.Module):
                v.to(x)

    @staticmethod
    def compatible_reg(
        image_size: tuple[int, int] | None,
        pre_resize_transforms: v2.Transform | Sequence[v2.Transform] = [],
    ) -> GenericRegularizer:
        return GenericRegularizer(
            image_size=image_size,
            norm_type=None,
            scale_tensors=False,
            pre_resize_transforms=pre_resize_transforms,
        )


class GenericModelRegularizer(ModelRegularizer, GenericRegularizer):

    def __init__(
        self,
        device: torch.device | str | int,
        image_size: tuple[int, int] | None = IMAGE_SIZE,
        norm_type: Literal["flat", "imagenet"] | None = "flat",
        **kwargs,
    ) -> None:
        super(ModelRegularizer, self).__init__(image_size, norm_type, **kwargs)
        self.device = device

    def __call__(self, x: npt.NDArray[np.uint8] | torch.Tensor) -> torch.Tensor:
        x = cast(torch.Tensor, self.TO_TENSOR(x))
        x = x.to(self.device)
        return super(ModelRegularizer, self).__call__(x)
