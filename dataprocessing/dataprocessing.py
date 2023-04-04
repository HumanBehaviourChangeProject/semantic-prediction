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
from collections import Counter

class Processor(ABC):
    @abstractmethod
    def __call__(self, inp: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class Fuzzyfier:
    def get_extended_fuzzy_data(self, features: pd.DataFrame, labels):
        inflated_data = dict()
        names = []
        for col in features.columns:
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

def _at_least_10(collection):
    c = Counter(collection)
    if len(c.keys()) == 2:
        return all(vs >= 10 for vs in c.values())
    return True

def filter_and_cluster(features, labels):

    with open("data/all_feature_info.csv") as fin:
        reader = csv.DictReader(fin)
        feature_list = list(reader)
    clusters = {d["Aggregate Feature"] for d in feature_list if d["Aggregate Feature"]}
    features_maps = {d["Feature ID"]:[c for c in features.columns if str(c[0]) == d["Feature ID"]][0] for d in feature_list if d["Aggregate Feature"]}
    cluster_mappings = {f:[features_maps[d["Feature ID"]] for d in feature_list if d["Aggregate Feature"] == f] for f in clusters}
    print(clusters)

    for f in clusters:
        features[(f,f)] = features[[*cluster_mappings[f]]].any(axis=1)

    threshold = 30

    feature_filter = features.sum(axis=0) > threshold
    sufficiently_represented_features = features.columns[feature_filter]

    print("\n".join(f"Dropping feature \"{x}\" due to low evidence ({y} instances)" for x, y in
                    zip((x[1] for x in features.columns[~feature_filter]),
                        features.sum(axis=0)[~feature_filter])))

    has_outcome = (~labels.isna()).values
    features = features[sufficiently_represented_features][has_outcome].astype(
        float)
    labels = labels[has_outcome].astype(float)

    return features, labels


def filter_pregnancy_trials(features, labels):
    filter_trials = features[[("Relapse Prevention Trial","Relapse Prevention Trial"),
                   ("Relapse Prevention Trial(Mixed)", "Relapse Prevention Trial(Mixed)"),
                   ("Pregnancy trial", "Pregnancy trial"),
                   ("Pregnancy trial (Mixed)", "Pregnancy trial (Mixed)")]].any(axis=1)

    pregnancy_documents = [i[0] for i in features[filter_trials].index]
    documents_to_drop = pd.Series({i:i[0] in pregnancy_documents for i in features.index})

    print(f"Dropping the following {len(features[documents_to_drop].index)} pregnancy or relapse trials:")
    print("\t* " + ("\n\t* ".join({c[2] for c in features[documents_to_drop].index})))
    return features[~documents_to_drop], labels[~documents_to_drop]


def main():
    reader = AttributeReader()
    ds = reader.read()
    print("Build fuzzy dataset")

    ds[ds.isna()] = None

    # Sort columns
    ds = ds[sorted(ds.columns, key=lambda x: str(x[1]))]

    write_csv(ds)

    labels = ds[6451791]
    feature_columns = [c[0] not in (6451791, 6080518) for c in ds.columns]
    features = ds.loc[:, feature_columns]

    print(f"After initial loading, dataset has {len(features.columns)} columns and {ds.shape}")

    female = (6080485, "Proportion identifying as female gender")
    male = (6080486, "Proportion identifying as male gender")

    features[female][features[female].isnull()] = 100*features[("Pregnancy trial", "Pregnancy trial")][features[female].isnull()]

    features[female][features[female].isnull()] = 100 - features[male][features[female].isnull()]
    features[male][features[male].isnull()] = 100 - features[female][features[male].isnull()]

    with open("data/analysed.csv", "wb") as fout:
        features[(6080719, "Individual-level analysed")].to_csv(fout)

    f = Fuzzyfier()
    features, labels = f.get_extended_fuzzy_data(features, labels)

    print(f"After extending, dataset has {len(features.columns)} columns and {features.shape}")

    features, labels = filter_pregnancy_trials(features, labels)

    print(f"After filtering pregnancy trials, dataset has {len(features.columns)} columns and {features.shape}")

    features, labels = filter_and_cluster(features, labels)

    print(f"After filtering and clustering, dataset has {len(features.columns)} columns and {features.shape}")
    write_fuzzy(features, labels)


if __name__ == "__main__":
    main()
