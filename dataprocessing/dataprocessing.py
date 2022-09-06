import json
import csv
import pickle

import pandas
import pandas as pd
import xlsxwriter

from skfuzzy import cluster as fc
import numpy as np
import re
from fuzzysets import FUZZY_SETS
from writer import write_csv, write_fuzzy
from cleaner import is_number
from reader import AttributeReader
from abc import ABC, abstractmethod


class Processor(ABC):
    @abstractmethod
    def __call__(self, inp: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class Fuzzyfier:
    def get_extended_fuzzy_data(self, features: pd.DataFrame, labels):
        inflated_data = dict()
        names = []
        for col in features.columns:
            print(col)
            values = features.loc[:, col]
            fltr = values.notnull()
            filtered = values[fltr].astype(float)
            if not np.max(filtered) <= 1:
                fs = FUZZY_SETS.get(col[1])
                if fs is not None:
                    fuzzy_sets = list(fs.items())
                    new_values = np.zeros((values.shape[0], len(fuzzy_sets)))
                    new_values[fltr] = np.array(
                        [[fs(v) if v is not None else None for _, fs in fuzzy_sets] for v in filtered]
                    )
                    new_names = [(f"{col[1]} ({x})", f"{col[1]} ({x})") for x in (l for l, _ in fuzzy_sets)]
                else:
                    n_centroids = 5
                    exfl = np.expand_dims(filtered, axis=-1).T
                    cntr, u, u0, d, jm, p, fpc = fc.cmeans(
                        exfl, n_centroids, 2, error=0.005, maxiter=1000, init=None
                    )
                    u = u[np.squeeze(cntr, axis=-1).argsort()]
            else:
                new_values = np.expand_dims(values, axis=-1)
                new_names = [col]
            for n, v in zip(new_names, new_values.T):
                inflated_data[n] = v
            names += new_names
        # assert 0 <= np.min(inflated_data) and np.max(inflated_data) <= 1, (np.min(inflated_data), np.max(inflated_data))
        new_features = pd.DataFrame(inflated_data, index=features.index, columns=names)
        return new_features, labels


def main():
    reader = AttributeReader()
    ds = reader.read()
    print("Build fuzzy dataset")

    labels = ds[6451791]
    feature_columns = [c[0] not in (6451791, 6080518) for c in ds.columns]
    features = ds.loc[:, feature_columns]

    write_csv(features)

    f = Fuzzyfier()
    features, labels = f.get_extended_fuzzy_data(features, labels)
    write_fuzzy(features.astype(float), labels.astype(float))


if __name__ == "__main__":
    main()
