# These are just some preprocsessing tools
from time import sleep
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult, Pool as PoolType
from os.path import dirname, join as jo
from warnings import warn

import cv2
import numpy as np
import numpy.typing as npt
import os
import shutil

from typing import Any, Callable, cast

N_CPU = os.cpu_count() or 1
MIN_DIM = 32 * 12
ROOT = dirname(__file__)
RESULTS: list[AsyncResult] = []
RESULTS_DONE = 0


class NotAnImageError(Exception): ...


def resize_logic(rows: int, cols: int) -> tuple[int, int]:
    rate = MIN_DIM / min(rows, cols)
    if rate >= 1:
        return rows, cols
    else:
        return round(rows * rate), round(cols * rate)


def read_resize(src_path: str, dst_path: str) -> None:
    x = cv2.imread(src_path)
    if x is None:
        shutil.copy(src_path, dst_path)
    rows, cols = x.shape[0], x.shape[1]
    n_rows, n_cols = resize_logic(rows, cols)
    x = cv2.resize(
        x,
        (n_cols, n_rows),
        interpolation=cv2.INTER_LINEAR,
    )
    cv2.imwrite(dst_path, x)


def report_queue_and_wait(
    max_size: int = 100,
    wait_time: float = 0.5,
) -> None:
    global RESULTS_DONE
    i = 0
    while i < len(RESULTS):
        if RESULTS[i].ready():
            del RESULTS[i]
            RESULTS_DONE += 1
        else:
            i += 1

        if not (i < len(RESULTS)):
            if len(RESULTS) > max_size:
                sleep(wait_time)
                i = 0

    print(
        f"\rTasks:{len(RESULTS)+ RESULTS_DONE}, Pending:{len(RESULTS)}",
        end="",
        flush=True,
    )


def dir_enqueue(
    src_folder: str,
    dst_folder: str,
    pool: PoolType,
    f: Callable[[str, str], Any],
    _root: bool = True,
) -> None:
    try:
        os.mkdir(dst_folder)
    except FileExistsError:
        ...

    for i in os.scandir(src_folder):
        if i.is_dir():
            dir_enqueue(i.path, jo(dst_folder, i.name), pool, f, _root=False)
        elif i.is_symlink():
            shutil.copy(i.path, jo(dst_folder, i.name), follow_symlinks=False)
        elif i.is_file():
            RESULTS.append(pool.apply_async(f, args=(i.path, jo(dst_folder, i.name))))
            report_queue_and_wait()
        else:
            warn(f"Unsure of the nature of the item '{i.path}'", RuntimeWarning)

    if _root:
        print("\nDone!")


def resize_2020() -> None:
    SRC = jo(ROOT, *("storage/large/datasets/isic-2020/image_raw".split("/")))
    DST = jo(ROOT, *("storage/large/datasets/isic-2020/image".split("/")))
    pool = Pool(processes=N_CPU)
    dir_enqueue(SRC, DST, pool, read_resize)
    pool.close()
    pool.join()


def read_crop_vignette(src_path: str, dst_path: str) -> None:
    ## Tolerance (rows/cols) in proportion to smallest dims
    TOLERANCE_PROPORTION = 10 / 100
    ## Condition: not vignette if nth quantile > threshold
    THRESHOLD = 50
    PERCENTILE = 90

    def idx_row_vignette(
        x: npt.NDArray[np.uint8],
        threshold: float,
        percentile: float,
        tolerance: float,
        negative: bool = False,
    ) -> int:
        if x.ndim != 2:
            raise ValueError("Only implemented for 2 dim array.")
        tolerance_absolute = round(max(x.shape) * tolerance)
        rows = np.percentile(x, percentile, axis=1)

        i = 0
        tolerance_counter = 0
        idx = latest_idx = (not negative) * i + negative * (len(rows) - 1 - i)
        while tolerance_counter <= tolerance_absolute:
            idx = (not negative) * i + negative * (len(rows) - 1 - i)
            if idx not in range(len(rows)):
                warn("Whole image seems to be dark!", RuntimeWarning)
                return (not negative) * 0 + negative * (len(rows) - 1)

            if rows[idx] > threshold:
                tolerance_counter += 1
            else:
                tolerance_counter = 0
                latest_idx = idx

            i += 1
        return latest_idx

    im_bgr = cv2.imread(src_path)
    if im_bgr is None:
        raise FileNotFoundError(src_path)
    im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    row_start = idx_row_vignette(
        im,
        THRESHOLD,
        PERCENTILE,
        TOLERANCE_PROPORTION,
        negative=False,
    )
    row_end = (
        idx_row_vignette(
            im,
            THRESHOLD,
            PERCENTILE,
            TOLERANCE_PROPORTION,
            negative=True,
        )
        + 1
    )
    col_start = idx_row_vignette(
        im.T,
        THRESHOLD,
        PERCENTILE,
        TOLERANCE_PROPORTION,
        negative=False,
    )
    col_end = (
        idx_row_vignette(
            im.T,
            THRESHOLD,
            PERCENTILE,
            TOLERANCE_PROPORTION,
            negative=True,
        )
        + 1
    )
    cv2.imwrite(dst_path, im_bgr[row_start:row_end, col_start:col_end, :])


def vignette_bcn() -> None:
    SRC = jo(
        ROOT,
        *(
            "storage/large/hdd_datasets/bcn20000_raw/bcn_20k_train_uncropped/".split(
                "/"
            )
        ),
    )
    DST = jo(ROOT, *("storage/large/datasets/bcn20000/bcn_20k_train/".split("/")))
    pool = Pool(processes=N_CPU)
    dir_enqueue(SRC, DST, pool, read_crop_vignette)
    pool.close()
    pool.join()


def main():
    vignette_bcn()


if __name__ == "__main__":
    main()
