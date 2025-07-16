r"""Misc analysis script."""

from hashlib import sha256
from os.path import join as jo

import numpy as np
import os
import os.path as osp
import pandas as pd

ROOT = osp.dirname(__file__)
DATA = jo(ROOT, "storage", "outputs", "classifiers")
OUTPUTS = jo(ROOT, "storage", "outputs", "analysis")
METRICS = {"f1": "valid_f1", "acc": "valid_accuracy", "auroc": "valid_auroc"}
REF_METRIC = "valid_f1"


def main() -> None:
    r"""
    Logs follow the following structure:
    <experiment_name>/<dataset>/<model_id>/<version_number>
    """

    for exp in os.scandir(DATA):
        for dataset in os.scandir(exp.path):
            save_path = jo(OUTPUTS, exp.name, dataset.name) + ".csv"
            columns = ["model_id", "config"]
            for name in METRICS.keys():
                columns.extend([f"{name}_best", f"{name}_avg", f"{name}_sd"])
            columns.append("n")
            df = pd.DataFrame(columns=columns)  ##type:ignore

            for model in sorted(os.scandir(dataset.path), key=lambda x: x.name):
                data: dict[bytes, dict[str, list[float]]] = {}
                for x in os.scandir(model.path):
                    with open(jo(x.path, "hparams.yaml"), "rb") as f:
                        config = sha256(f.read()).digest()  ## type:ignore

                    for name, val in METRICS.items():
                        try:
                            with open(jo(x.path, "metrics.csv"), "rt") as f:
                                try:
                                    serie = pd.read_csv(f)[val]
                                    metric = np.nanmax(serie)
                                    ## Either take the latest value, or the highest
                                    """
                                        i = -1
                                        metric = np.nan
                                        while i >= -len(serie) and pd.isna(metric):
                                            metric = serie.iloc[i]
                                            i -= 1
                                    """
                                except KeyError:
                                    metric = np.nan
                        except FileNotFoundError:
                            continue

                        try:
                            try:
                                data[config][name].append(metric)
                            except KeyError:
                                data[config][name] = [metric]
                        except KeyError:
                            data[config] = {name: [metric]}

                for i, xs in enumerate(data.values()):
                    newline = [model.name, i]
                    for name in METRICS.keys():
                        newline.extend(
                            [
                                np.round(np.nanmax(xs[name]), decimals=3),
                                np.round(np.nanmean(xs[name]), decimals=3),
                                np.round(np.nanstd(xs[name]), decimals=3),
                            ]
                        )
                    name = list(METRICS.keys())[-1]
                    newline.append(np.sum(np.logical_not(np.isnan(xs[name]))))

                    df.loc[-1] = newline
                    df.index = df.index + 1
                    df = df.sort_index()

            ## Padding
            pad = lambda x: x.str.pad(x.str.len().max())
            if not osp.exists(osp.dirname(save_path)):
                os.makedirs(osp.dirname(save_path))
            df.astype(str).apply(pad).to_csv(save_path)


if __name__ == "__main__":
    main()
