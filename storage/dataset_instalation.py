# Installs the datasets and applies preprocessing
from argparse import ArgumentParser
from time import sleep
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult, Pool as PoolType
from os.path import dirname, join as jo
from warnings import warn

import cv2
import numpy as np
import numpy.typing as npt
import os
import os.path as osp
import shutil
import subprocess

from typing import Any, Callable

N_CPU = os.cpu_count() or 1
MIN_DIM = 32 * 12
RESULTS: list[AsyncResult] = []
RESULTS_DONE = 0

ROOT = dirname(__file__)
PARSER = ArgumentParser(
    description="Extrae los archivos '.zip' del conjunto de datos "
    "y realiza todos los pasos necesarios de preprocesamiento."
)


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
    else:
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


def siim_callback(x: str) -> None:
    print("Aplicando preprocesamiento: redimensionamiento SIIM-ISIC...")
    shutil.rmtree(jo(x, "image", "test"), ignore_errors=True)

    train_path = jo(x, "image", "train")
    tmp_path = jo(x, "image", "raw_train")
    os.rename(train_path, tmp_path)

    pool = Pool(processes=N_CPU)
    dir_enqueue(tmp_path, train_path, pool, read_resize)
    pool.close()
    pool.join()

    shutil.rmtree(tmp_path)


def bcn20000_callback(x: str) -> None:
    print("Aplicando preprocesamiento: recorte BCN-20000...")

    shutil.rmtree(jo(x, "bcn_20k_test"), ignore_errors=True)

    train_path = jo(x, "bcn_20k_train")
    tmp_path = jo(x, "raw_bcn_20k_train")
    os.rename(train_path, tmp_path)

    pool = Pool(processes=N_CPU)
    dir_enqueue(tmp_path, train_path, pool, read_crop_vignette)
    pool.close()
    pool.join()

    shutil.rmtree(tmp_path)


def extract_and_prepare(
    zip_path: str,
    dir_path: str,
    dir_callback: Callable[[str], None] | None = None,
) -> None:
    tmp_zip_dir = jo(osp.dirname(dir_path), ".tmp_zip_dir")
    os.mkdir(tmp_zip_dir)

    try:

        ## Packaging is hard...
        match osp.basename(zip_path):
            case "siim.zip":
                subprocess.run(["unzip", "-q", zip_path, "-d", tmp_zip_dir])
                subdirs = list(os.scandir(tmp_zip_dir))

                os.mkdir(jo(tmp_zip_dir, "image"))
                os.rename(subdirs[0].path, jo(tmp_zip_dir, "image", "train"))
                shutil.copy(
                    jo(ROOT, "aux", "siim.csv"),
                    jo(tmp_zip_dir, "train.csv"),
                )

            case "bcn.zip":
                os.mkdir(jo(tmp_zip_dir, "bcn_20k_train"))
                subprocess.run(
                    [
                        "unzip",
                        "-q",
                        zip_path,
                        "-d",
                        jo(tmp_zip_dir, "bcn_20k_train"),
                    ]
                )
                shutil.copy(
                    jo(ROOT, "aux", "bcn.csv"),
                    jo(tmp_zip_dir, "bcn_20k_train.csv"),
                )

            case "pad.zip":
                subprocess.run(["unzip", "-q", zip_path, "-d", tmp_zip_dir])
                for i in list(os.scandir(tmp_zip_dir)):
                    if i.is_dir() and i.name.startswith("imgs_part"):
                        tmp_inter_dir = jo(tmp_zip_dir, ".tmp_imgs")
                        os.rename(i.path, tmp_inter_dir)
                        os.rename(jo(tmp_inter_dir, i.name), i.path)
                        os.rmdir(tmp_inter_dir)

            case "slice.zip":
                subprocess.run(["unzip", "-q", zip_path, "-d", tmp_zip_dir])
                os.rename(
                    jo(tmp_zip_dir, "train_image", "image"),
                    jo(tmp_zip_dir, "image"),
                )
                os.rmdir(jo(tmp_zip_dir, "train_image"))
                os.remove(jo(tmp_zip_dir, "train-image.hdf5"))
                os.remove(jo(tmp_zip_dir, "test-image.hdf5"))

            case _:
                subprocess.run(["unzip", "-q", zip_path, "-d", tmp_zip_dir])

        subdirs = list(os.scandir(tmp_zip_dir))
        if len(subdirs) == 1:
            target_dir = subdirs[0].path
        else:
            target_dir = tmp_zip_dir
        if dir_callback is not None:
            dir_callback(target_dir)

    except BaseException as e:
        match e:
            case KeyboardInterrupt():
                print("Se ha cancelado la operación, eliminando directorio temporal...")
            case _:
                print("Se ha detectado un error, eliminando directorio temporal...")
        shutil.rmtree(tmp_zip_dir)
        raise e

    else:
        os.rename(target_dir, dir_path)
        try:
            os.rmdir(tmp_zip_dir)
        except FileNotFoundError:
            ...
        os.remove(zip_path)


def main():
    _ = PARSER.parse_args()

    basedir = jo(ROOT, "large")
    if not osp.isdir(basedir):
        raise NotADirectoryError(basedir)

    datadir = jo(basedir, "datasets")
    os.makedirs(datadir, exist_ok=True)

    zip_dir_translations = [
        ("ham.zip", "ham10000"),
        ("med.zip", "med-node"),
        ("siim.zip", "isic-2020"),
        ("slice.zip", "isic-2024"),
        ("pad.zip", "pad-ufes-20"),
        ("bcn.zip", "bcn20000"),
    ]

    for zip_name, dir_name in zip_dir_translations:
        zip_path = jo(ROOT, zip_name)
        dir_path = jo(datadir, dir_name)
        if osp.isdir(jo(datadir, dir_path)):
            print(f"Directorio {dir_path} ya existe, saltando...")
        elif osp.isfile(jo(ROOT, zip_path)):
            print(f"Extrayendo archivo {zip_name}...")
            match zip_name:
                case "siim.zip":
                    callback = siim_callback
                case "bcn.zip":
                    callback = bcn20000_callback
                case _:
                    callback = None
            extract_and_prepare(zip_path, dir_path, callback)
        else:
            raise FileNotFoundError(zip_path)


if __name__ == "__main__":
    main()
