from collections import defaultdict
from copy import copy
from os.path import join as jo

import numpy as np
import os.path as osp
import os
import pandas as pd
import random
import sys

from typing import Callable, Mapping, Sequence, Union

FILE = osp.dirname(__file__)
ROOT = jo(FILE, "..")
sys.path.append(ROOT)
DATA = jo(ROOT, "storage", "large", "datasets")


from .components import (
    IndexerCore,
    CoreFilter,
    CoreMerge,
    CoreSpecs,
    FLOAT_SLICE,
    Indexer,
    STAGE,
    StratifiedIndexer,
)

__all__ = [
    "CoreHAM",
    "CoreMED",
    "CorePAD",
    "CorePH2",
    "Core2020",
    "Core2024",
    "CoreBCN",
    "GlobalStratifiedIndexer",
    "ExcludingGlobalStratifiedIndexer",
]


def readlines(path: str) -> list[str]:
    with open(path, "rt") as f:
        return list(map(str.strip, f))


class CoreHAM(IndexerCore):
    ## Some metadata
    METADATA = jo(DATA, "ham10000", "HAM10000_metadata.csv")
    DATAFOLDERS = {
        jo(DATA, "ham10000", "ham10000_images_part_" + str(i)) for i in {1, 2}
    }

    def __init__(self) -> None:
        super().__init__()
        self.df = pd.read_csv(self.METADATA)
        self.files = {
            i.name[:-4]: i.path
            for folder in self.DATAFOLDERS
            for i in os.scandir(folder)
        }

    def __call__(self) -> dict[str, list[str]]:
        core: dict[str, list[str]] = {}
        for _, row in self.df.iterrows():
            name = row["image_id"]
            label = row["dx"]
            assert isinstance(name, str)
            assert isinstance(label, str)

            if label in core.keys():
                core[label].append(self.files[name])
            else:
                core[label] = [self.files[name]]
        return core


class Core2020(IndexerCore):
    METADATA = jo(DATA, "isic-2020", "train.csv")
    DATAFOLDER = jo(DATA, "isic-2020", "image", "train")

    def __init__(
        self,
        extended: bool = False,
        max_items_per_class: int | None = None,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(self.METADATA)
        self.files = {
            i.name[:-4]: i.path
            for i in os.scandir(
                self.DATAFOLDER,
            )
        }
        self.extended = extended
        self.max_items_per_class = max_items_per_class

    @classmethod
    def get_image(cls, id_: str) -> str:
        return jo(cls.DATAFOLDER, id_ + ".jpg")

    def __call__(self) -> dict[str, list[str]]:
        label_idx = "diagnosis" if self.extended else "benign_malignant"
        core: dict[str, list[str]] = {}
        for _, row in self.df.iterrows():
            name = row["image_name"]
            label = row[label_idx]
            assert isinstance(name, str)
            assert isinstance(label, str)

            if label in core.keys():
                core[label].append(self.get_image(name))
            else:
                core[label] = [self.get_image(name)]
        for label in core.keys():
            ## If it's None it does not do anything
            core[label] = core[label][: self.max_items_per_class]
        return core


class Core2024(IndexerCore):
    METADATA = jo(DATA, "isic-2024", "train-metadata.csv")
    DATAFOLDER = jo(DATA, "isic-2024", "image")
    CACHE: dict[bool, dict[str, list[str]] | None] = {
        True: None,  ## extended
        False: None,  ## default
    }

    def __init__(
        self,
        extended: bool = False,
        max_items_per_class: int | None = None,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(
            self.METADATA,
            # This just shuts a warning up
            dtype={"iddx_5": str, "mel_mitotic_index": str},
        )
        self.extended = extended
        self.max_items_per_class = max_items_per_class

    @classmethod
    def get_image(cls, id_: str) -> str:
        return jo(cls.DATAFOLDER, id_ + ".jpg")

    def __call__(self) -> dict[str, list[str]]:
        possible_cache = self.CACHE[self.extended]
        if possible_cache is not None:
            return possible_cache

        label_idx = "iddx_full" if self.extended else "target"
        core: dict[str, list[str]] = {}
        for _, row in self.df.iterrows():
            name = row["isic_id"]
            assert isinstance(name, str)
            label = str(row[label_idx])

            if label in core.keys():
                core[label].append(self.get_image(name))
            else:
                core[label] = [self.get_image(name)]
        for label in core.keys():
            ## If it's None it does not do anything
            core[label] = core[label][: self.max_items_per_class]

        self.CACHE[self.extended] = core
        return core


class CoreMED(IndexerCore):
    ## No metadata
    DATAFOLDERS = {jo(DATA, "med-node", i) for i in {"melanoma", "naevus"}}

    def __init__(self) -> None:
        super().__init__()
        self.core = {
            osp.basename(folder): [i.path for i in os.scandir(folder)]
            for folder in self.DATAFOLDERS
        }

    def __call__(self) -> dict[str, list[str]]:
        return self.core


class CorePAD(IndexerCore):
    ## A lot of metadata
    METADATA = jo(DATA, "pad-ufes-20", "metadata.csv")
    DATAFOLDERS = {jo(DATA, "pad-ufes-20", "imgs_part_" + str(i)) for i in (1, 2, 3)}

    def __init__(self) -> None:
        super().__init__()
        self.df = pd.read_csv(self.METADATA)
        self.files = {
            i.name: i.path for folder in self.DATAFOLDERS for i in os.scandir(folder)
        }

    def __call__(self) -> dict[str, list[str]]:
        core: dict[str, list[str]] = {}
        for _, row in self.df.iterrows():
            name = row["img_id"]
            label = row["diagnostic"]
            assert isinstance(name, str)
            assert isinstance(label, str)

            if label in core.keys():
                core[label].append(self.files[name])
            else:
                core[label] = [self.files[name]]
        return core


class CoreBCN(IndexerCore):
    ## Some metadata, very large
    METADATA = jo(DATA, "bcn20000", "bcn_20k_train.csv")
    DATAFOLDER = jo(DATA, "bcn20000", "bcn_20k_train")

    LEGACY_TRANSLATIONS = {
        "Nevus": "NV",
        "Melanoma, NOS": "MEL",
        "Melanoma metastasis": "MEL",
        "Seborrheic keratosis": "BKL",
        "Solar lentigo": "BKL",
        "Basal cell carcinoma": "BCC",
        "Solar or actinic keratosis": "AK",
        "Squamous cell carcinoma, NOS": "SCC",
        "Dermatofibroma": "DF",
        "Scar": "DF",
    }

    def __init__(self) -> None:
        super().__init__()
        self.df = pd.read_csv(self.METADATA)
        for v in list(self.LEGACY_TRANSLATIONS.values()):
            self.LEGACY_TRANSLATIONS[v] = v

    @classmethod
    def get_image(cls, id_: str) -> str:
        return jo(cls.DATAFOLDER, id_)

    def __call__(self) -> dict[str, list[str]]:
        core: dict[str, list[str]] = {}
        for _, row in self.df.iterrows():
            try:
                name = row["isic_id"] + ".jpg"
            except KeyError:
                name = row["bcn_filename"]

            assert isinstance(name, str)
            raw_label = str(row["diagnosis_3"])

            ## That means it's a test sample
            if raw_label == "nan":
                continue

            label = self.LEGACY_TRANSLATIONS[raw_label]

            if label in core.keys():
                core[label].append(self.get_image(name))
            else:
                core[label] = [self.get_image(name)]
        return core


class CorePH2(IndexerCore):
    ## Segmentarion-based, no metadata
    ## TBD: NOT IMPLEMENTED
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("TBD")


class SplitCriteria:
    def __init__(self, split: FLOAT_SLICE | Sequence[FLOAT_SLICE]) -> None:
        self.split = split

    def __call__(self, x: str, class_: str, specs: CoreSpecs) -> bool:
        ## Deterministic position shuffling
        ## Iss slow so we cache on ramdisk
        cache_path = f"/dev/shm/xana_indices_{specs.class_total}_seed33.bin"
        if osp.isfile(cache_path):
            ## No need to move to list as we indexing
            new_mapping = np.fromfile(cache_path, dtype=np.int32)
        else:
            new_mapping = list(range(specs.class_total))
            r = random.Random(33)
            r.shuffle(new_mapping)
            if specs.class_total >= 50_000:
                to_save = np.array(new_mapping, dtype=np.int32)
                with open(cache_path + ".tmp", "wb") as tmp:
                    to_save.tofile(tmp)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                del to_save
                os.rename(cache_path + ".tmp", cache_path)

        new_idx = new_mapping[specs.class_idx]

        if isinstance(self.split, Sequence):
            union_splits = self.split
        else:
            union_splits = (self.split,)

        for single_split in union_splits:
            start = single_split.start or 0
            stop = single_split.stop or 1
            assert start <= stop

            if start <= (new_idx / specs.class_total) < stop:
                return True
        return False


class GlobalStratifiedIndexer(Indexer):
    STAGE_SPLIT: dict[STAGE, FLOAT_SLICE] = {
        "train": slice(0.85),
        "valid": slice(0.85, 1),
        "test": slice(1),
    }
    GLOBAL_CORES: tuple[IndexerCore, ...] = (
        CoreHAM(),
        CoreMED(),
        CorePAD(),
        Core2020(extended=True, max_items_per_class=1000),
        Core2024(extended=True, max_items_per_class=1000),
        CoreBCN(),
    )
    ## TODO: UNKNOWN ON ISIC-2020
    EQUIV_SETS = (
        (
            "NAEVI",
            {
                "nv",  # HAM10000 6705
                "naevus",  # MED-NODE 103
                "NEV",  # PAD-UFES 244
                "unknown",  # ISIC2020 1000 (cropped)
                "nevus",  # ISIC2020 1000 (cropped)
                "NV",  # BCN20000 4206
                # ISIC2024 1518 (cropped)
                *readlines(jo(FILE, "core2024_alias", "nev")),
            },
        ),
        (
            "MELANOMA",
            {
                "mel",  # HAM10000 1113
                "melanoma",  # MED-NODE 73
                "MEL",  # PAD-UFES 52
                "melanoma",  # ISIC2020 584
                "MEL",  # BCN20000 2857
                # ISIC2024 147
                *readlines(jo(FILE, "core2024_alias", "mel")),
            },
        ),
        (
            "BENIGN_KERATOSIS",
            {
                "bkl",  # HAM10000 1099
                # MED-NODE 0
                "SEK",  # PAD-UFES 235 (subset)
                "seborrheic keratosis",  # ISIC2020 135 (subset)
                "lentigo NOS",  # ISIC2020 44 (subset)
                "lichenoid keratosis",  # ISIC2020 37 (subset)
                "solar lentigo",  # ISIC2020 7 (subset)
                "BKL",  # BCN20000 1138
                # ISIC2024 97
                *readlines(jo(FILE, "core2024_alias", "bkl")),
            },
        ),
        (
            "BASAL_CELL_CARCINOMA",
            {
                "bcc",  # HAM10000 514
                # MED-NODE 0
                "BCC",  # PAD-UFES 845
                # ISIC2020 0
                "BCC",  # BCN20000 2809
                # ISIC2024 163
                *readlines(jo(FILE, "core2024_alias", "bcc")),
            },
        ),
        (
            "ACTINIC_KERATOSES",
            {
                "akiec",  # HAM10000 327
                # MED-NODE 0
                "ACK",  # PAD-UFES 730 (subset)
                "SCC",  # PAD-UFES 192 (subset)
                # ISIC2020 0
                "AK",  # BCN20000 737 (subset)
                "SCC",  # BCN20000 431 (subset)
                # ISIC2024 112
                *readlines(jo(FILE, "core2024_alias", "ack")),
            },
        ),
        (
            "VACULAR",
            {
                "vasc",  # HAM10000 142
                # MED-NODE 0
                # PAD-UFES 0
                # ISIC2020 0
                "VASC",  # BCN20000 111
                # ISIC2024 3
                *readlines(jo(FILE, "core2024_alias", "vas")),
            },
        ),
        (
            "DERMATOFIBROMA",
            {
                "df",  # HAM10000 115
                # MED-NODE 0
                # PAD-UFES 0
                # ISIC2020 0
                "DF",  # BCN20000 124
                # ISIC2024 15
                *readlines(jo(FILE, "core2024_alias", "def")),
            },
        ),
    )
    ## CONSISTENT_IDX_MAPPING: dict[str, int] | None = None

    def __init__(
        self,
        stage: STAGE,
        core: IndexerCore,
        non_global: bool = False,
        n_items: int | None = None,
        split: Mapping[STAGE, FLOAT_SLICE | Sequence[FLOAT_SLICE]] | None = None,
    ) -> None:
        self.stage = stage
        if split is None:
            split = copy(self.STAGE_SPLIT)

        ## Merged, global core.
        ## We substitute the core that is the same as the argument.
        if non_global:
            support_cores = tuple()
        else:
            support_cores = tuple(i for i in self.GLOBAL_CORES if type(i) != type(core))

        core_types = [type(i) for i in self.GLOBAL_CORES]
        if not isinstance(core, Union[*core_types]):
            raise ValueError(
                f"Supported cores are {[i.__name__ for i in core_types]}. "
                f"Got {type(core)}."
            )
        if (isinstance(core, Core2020 | Core2024)) and (not core.extended):
            raise ValueError(
                f"For Core2020 and Core2024, the 'extended' arg must be set to 'True'."
            )
        ## 'n_items' limits the number of items per class, but only on train sets.
        ## The goal here is to undersample to make training more convenient through
        ## earlier and more frequent end-of-epoch events (and thus, more granular
        ## validation performance monitoring).
        if n_items is not None:
            max_support_length: dict[str, int] = defaultdict(int)
            for c in support_cores:
                for k, v in c().items():
                    max_support_length[k] += len(v)

            if max_support_length:
                n_support_items = max(max_support_length.values())
            else:
                n_support_items = 0
            n_items_left = n_items - n_support_items

            if n_items_left < 0:
                raise ValueError(
                    f"'{n_items=}' is too low, (must be >= {n_support_items})"
                )

        ## The idea is to perform the slicing before feeding
        ## it to the submodule.
        self.filtered_core = self.core_slicing(core, split[stage])
        ## We define a consistent class_idx to maintain consistency between str <-> int
        ## mappings in train and valid indexers. Useful when the studied dataset lacks
        ## a certain class.
        classes_idx = {alias: i for i, (alias, eq_set) in enumerate(self.EQUIV_SETS)}
        global_split: dict[STAGE, FLOAT_SLICE] = {stage: slice(None)}

        if stage == "train":
            self.internal_core = CoreMerge(
                *(support_cores + (self.filtered_core,)),
                equiv_sets=self.EQUIV_SETS,
                include_stray_classes=False,
                n_items=n_items,
            )

            ## Reuse StratifiedIndexer functionality, bit hacky.
            ## We want consistent idx mappings for real valid metrics on the model.
            ## We generate one on the fly and cache it for future use.

            self._internal_indexer = StratifiedIndexer(
                stage="train",
                core=self.internal_core,
                n_items=n_items,
                split=global_split,
                classes_idx=classes_idx,
            )

        else:
            # Step is still necessary because we need to translate and filter classes.
            self.internal_core = CoreMerge(
                self.filtered_core,
                equiv_sets=self.EQUIV_SETS,
                include_stray_classes=False,
            )
            self._internal_indexer = StratifiedIndexer(
                stage=stage,
                core=self.internal_core,
                split=global_split,
                classes_idx=classes_idx,
            )

        self.list_items = self._internal_indexer.list_items
        self.items = self._internal_indexer.items

    @staticmethod
    def core_slicing(
        core: IndexerCore,
        split: FLOAT_SLICE | Sequence[FLOAT_SLICE],
    ) -> CoreFilter:
        return CoreFilter(
            core,
            SplitCriteria(split),
            progress_bar=False,
        )

    def __len__(self) -> int:
        return self._internal_indexer.__len__()

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self._internal_indexer.__getitem__(idx)

    ## Complex functionality, so unit tests are convenient
    @staticmethod
    def unit_test_suite() -> dict[str, bool]:
        r"""Quick unit test suite, returns dict with format {test_name:passed}"""
        output: dict[str, bool] = {}

        ## Setup
        GSInd = GlobalStratifiedIndexer
        instance_train = GSInd("train", CoreMED(), split={"train": slice(1)})
        instance_val = GSInd("valid", CoreMED(), split={"valid": slice(1)})
        raw_core = CoreMED()

        merged_core = instance_train.internal_core
        assert merged_core is not None

        class_translations = {
            class_: (
                CoreMerge.find_class_in_equiv_set(class_, merged_core.equiv_sets)
                or class_
            )
            for class_ in raw_core().keys()
        }

        all_samples: list[tuple[str, str]] = [
            (x, class_translations[class_])
            for (class_, sample_subset) in CoreMED()().items()
            for x in sample_subset
        ]

        ## Part 1: consistent mappings between indexers
        idx_to_classes_train: dict[int, str] = {
            v: k for (k, v) in instance_train._internal_indexer.classes.items()
        }
        idx_to_classes_val: dict[int, str] = {
            v: k for (k, v) in instance_val._internal_indexer.classes.items()
        }
        output["consistent_indexer_assignments"] = (
            idx_to_classes_train == idx_to_classes_val
        )

        ## Part 2: valid indexer has every sample
        all_samples_indexer_val: list[tuple[str, str]] = [
            (sample, idx_to_classes_val[class_idx])
            for (sample, class_idx) in instance_val
        ]
        output["all_val_samples"] = set(all_samples) == set(all_samples_indexer_val)

        ## Part 3: train indexer contains every sample
        all_samples_indexer_train: list[tuple[str, str]] = [
            (sample, idx_to_classes_train[class_idx])
            for (sample, class_idx) in instance_train
        ]
        output["all_given_train_samples"] = set(all_samples_indexer_val).issubset(
            all_samples_indexer_train
        )

        ## Part 4: slices actually working (general proportions and disjointness)
        instance_val_sliced1 = GSInd(
            "valid",
            CoreMED(),
            split={"valid": (slice(0.8, 0.85), slice(0.85, 0.95))},
        )
        instance_val_sliced2 = GSInd(
            "valid",
            CoreMED(),
            split={"valid": (slice(0.8), slice(0.95, 1))},
        )
        all_samples_set: set[str] = set(x for (x, class_) in all_samples)
        all_samples_val_sliced1: list[str] = [
            sample for (sample, class_idx) in instance_val_sliced1
        ]
        all_samples_val_sliced2: list[str] = [
            sample for (sample, class_idx) in instance_val_sliced2
        ]

        output["slice_proportions"] = (
            len(set(all_samples_val_sliced1))
            < (len(all_samples_set) // 2)
            < len(set(all_samples_val_sliced2))
            < len(all_samples_set)
        )
        output["slice_disjoint"] = set(all_samples_val_sliced1).isdisjoint(
            all_samples_val_sliced2
        )
        output["slice_whole"] = (
            set(all_samples_val_sliced1).union(all_samples_val_sliced2)
            == all_samples_set
        )

        ## Part 5: limit on number of items per class
        instance_train_limited = GSInd(
            "train",
            CoreHAM(),
            split={"train": slice(1)},
            n_items=14_773 - 6705 + 3102,
        )
        ham_naevi = 0
        for k, v in instance_train_limited.internal_core().items():
            print(f"{k}: {len(v)}")
            for i in v:
                if "ham" in i and k == "NAEVI":
                    ham_naevi += 1
        print(f"{ham_naevi = }")
        for k, v in instance_train_limited.filtered_core().items():
            print(f"{k}: {len(v)}")

        return output


class ExcludingGlobalStratifiedIndexer(Indexer):
    STAGE_SPLIT = GlobalStratifiedIndexer.STAGE_SPLIT
    GLOBAL_CORES = GlobalStratifiedIndexer.GLOBAL_CORES
    EQUIV_SETS = GlobalStratifiedIndexer.EQUIV_SETS

    def __init__(
        self,
        stage: STAGE,
        core: IndexerCore,
        n_items: int | None = None,
        split: Mapping[STAGE, FLOAT_SLICE | Sequence[FLOAT_SLICE]] | None = None,
    ) -> None:
        self.stage = stage
        if split is None:
            split = copy(self.STAGE_SPLIT)

        core_types = [type(i) for i in self.GLOBAL_CORES]
        if not isinstance(core, Union[*core_types]):
            raise ValueError(
                f"Supported cores are {[i.__name__ for i in core_types]}. "
                f"Got {type(core)}."
            )
        if (isinstance(core, Core2020 | Core2024)) and (not core.extended):
            raise ValueError(
                f"For Core2020 and Core2024, the 'extended' arg must be set to 'True'."
            )

        ## Merged, global core.
        ## We remove the core that is the same as the argument.
        support_cores = tuple(i for i in self.GLOBAL_CORES if type(i) != type(core))
        del core

        ## 'n_items' limits the number of items per class, but only on train sets.
        ## The goal here is to undersample to make training more convenient through
        ## earlier and more frequent end-of-epoch events (and thus, more granular
        ## validation performance monitoring).
        if n_items is not None:
            max_support_length: dict[str, int] = defaultdict(int)
            for c in support_cores:
                for k, v in c().items():
                    max_support_length[k] += len(v)

            if max_support_length:
                n_support_items = max(max_support_length.values())
            else:
                n_support_items = 0
            n_items_left = n_items - n_support_items

            if n_items_left < 0:
                raise ValueError(
                    f"'{n_items=}' is too low, (must be >= {n_support_items})"
                )

        ## The idea is to perform the slicing before feeding
        ## it to the submodule.
        self.filtered_support_cores = tuple(
            GlobalStratifiedIndexer.core_slicing(c, split[stage]) for c in support_cores
        )
        ## We define a consistent class_idx to maintain consistency between str <-> int
        ## mappings in train and valid indexers. Useful when the studied dataset lacks
        ## a certain class.
        classes_idx = {alias: i for i, (alias, eq_set) in enumerate(self.EQUIV_SETS)}
        global_split: dict[STAGE, FLOAT_SLICE] = {stage: slice(None)}

        if stage == "train":
            self.internal_core = CoreMerge(
                *self.filtered_support_cores,
                equiv_sets=self.EQUIV_SETS,
                include_stray_classes=False,
                n_items=n_items,
            )

            ## Reuse StratifiedIndexer functionality, bit hacky.
            ## We want consistent idx mappings for real valid metrics on the model.
            ## We generate one on the fly and cache it for future use.

            self._internal_indexer = StratifiedIndexer(
                stage="train",
                core=self.internal_core,
                n_items=n_items,
                split=global_split,
                classes_idx=classes_idx,
            )

        else:
            # Step is still necessary because we need to translate and filter classes.
            self.internal_core = CoreMerge(
                *self.filtered_support_cores,
                equiv_sets=self.EQUIV_SETS,
                include_stray_classes=False,
            )
            self._internal_indexer = StratifiedIndexer(
                stage=stage,
                core=self.internal_core,
                split=global_split,
                classes_idx=classes_idx,
            )

        self.list_items = self._internal_indexer.list_items
        self.items = self._internal_indexer.items

    def __len__(self) -> int:
        return self._internal_indexer.__len__()

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self._internal_indexer.__getitem__(idx)

    ## Complex functionality, so unit tests are convenient here too!
    @staticmethod
    def unit_test_suite() -> dict[str, bool]:
        output: dict[str, bool] = {}

        ## Setup
        EGSInd = ExcludingGlobalStratifiedIndexer
        instance_train = EGSInd("train", CoreHAM())
        instance_val = EGSInd("valid", CoreHAM())

        all_samples_train: set[str] = set(
            sample for (sample, class_idx) in instance_train
        )
        all_samples_val: set[str] = set(sample for (sample, class_idx) in instance_val)

        output["train_val_disjointness"] = all_samples_train.isdisjoint(all_samples_val)
        output["core_disjointness"] = all_samples_train.union(
            all_samples_val
        ).isdisjoint(i for (_, array) in CoreHAM()().items() for i in array)

        return output


def test_run():
    ## Core2020's image dimenstion distribution
    core = Core2020()()
    from time import time
    import cv2
    import numpy as np

    time_ = time()

    x = []
    y = []
    for j in core.values():
        for i in j:
            im = cv2.imread(i)
            if im is None:
                print(i)
            else:
                x.append(im.shape[1])
                y.append(im.shape[0])

    print(core.keys())
    print(f"{(np.mean(x))=}; {(np.std(x))=}")
    print(f"{(np.mean(y))=}; {(np.std(y))=}")
    print(f"{len(x)=}")
    print(f"{(time()-time_)=}")

    ## Core2024's class-size-cropping tools
    for k, v in Core2024(extended=True, max_items_per_class=1000)().items():
        print(k, len(v))


def __main():
    print(GlobalStratifiedIndexer.unit_test_suite())
    print(ExcludingGlobalStratifiedIndexer.unit_test_suite())


if __name__ == "__main__":
    __main()
