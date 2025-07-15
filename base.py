import csv
import os

import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import typing
import abc
import random
from dataprocessing.filters import feature_filter
import math
import pickle

T = typing.TypeVar("T")

class BaseModel(typing.Generic[T]):

    def __init__(self, variables: typing.List[typing.AnyStr]):
        self.variables = variables

    def _train(self, train_features: T, train_labels: T, val_features: T, val_labels: T, train_docs, val_docs, verbose=True, weights=None):
        raise NotImplementedError

    def train(self, train_features: np.ndarray, train_labels: np.ndarray, val_features: np.ndarray, val_labels: np.ndarray, train_docs, val_docs, verbose=True, weights=None, delay_val=True):
        return self._train(
            self._prepare_single(train_features),
            self._prepare_single(train_labels),
            self._prepare_single(val_features),
            self._prepare_single(val_labels),
            train_docs, val_docs, verbose=verbose, weights=self._prepare_single(weights) if weights is not None else None,
            delay_val=delay_val
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._predict(
            self._prepare_single(features)
        )

    @abc.abstractmethod
    def _predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractclassmethod
    def name(cls):
        raise NotImplementedError

    def _prepare_single(self, data: np.ndarray) -> T:
        return data

    @abc.abstractmethod
    def save(self, path):
        raise NotImplementedError

    @abc.abstractclassmethod
    def load(self, path):
        raise NotImplementedError

def filter_features(features):
    headers = [i for i in features.columns if i[0] in feature_filter]
    return features[headers]


def cross_val(model_classes, raw_features: np.ndarray, raw_labels: np.ndarray, variables: typing.List[typing.AnyStr], output_path: str, no_test):

    outs = {c: np.empty(0) for c in model_classes}

    chunks = get_cross_split(raw_features.index, 5)
    for i in tqdm.tqdm(list(range(len(chunks)))):
        train_index = [
            c for j in range(len(chunks)) for c in chunks[j] if i != j and i != j - 1
        ]
        val_index = chunks[i]
        test_index = chunks[(i + 1) % len(chunks)]

        for model_cls in model_classes:
            model = model_cls(variables)
            model.train(raw_features.iloc[train_index].values, raw_labels[train_index],
                        raw_features.iloc[val_index].values, raw_labels[val_index],
                        raw_features.iloc[train_index].index, raw_features.iloc[val_index].index,
                        verbose=False
                        )
            y_pred = model.predict(raw_features.iloc[test_index].values)
            outs[model_cls] = np.concatenate((outs[model_cls], y_pred - raw_labels[test_index]))

    for model_cls, values in outs.items():
        os.makedirs(os.path.join(output_path, model_cls.name()), exist_ok=True)
        outfile = os.path.join(output_path, model_cls.name(), "crossval.txt")
        with open(outfile, "w", encoding="utf8") as fout:
            fout.write("\n".join(map(str, values)))



def _load_data(path, filters, weighted=False, drop=None):
    with open(path, "rb", encoding="utf8") as fin:
        features, labels = pickle.load(fin)

    features[np.isnan(features)] = 0
    if weighted:
        copy_features = pd.DataFrame()
        with open("data/analysed.csv") as fin:
            reader = csv.reader(fin)
            weights = {(int(a), int(b), c, d): (max(1, int(math.log2(float(v)))) if v != "" else 1) for a, b, c, d, v in reader}
            weights = [(k, v) for k, v in weights.items() if k in features.index]
            features = pd.DataFrame(y for x in [[features.loc[key]]*value for key, value in weights] for y in x)
            labels = np.array([y for x in [[labels[i]] * value for i, (key, value) in enumerate(weights)] for y in x])

    if filters is not None:
        features = filter_features(features)

    if drop is not None:
        col = features.columns[int(drop)]
        print("Exclude column:", col)
        features.drop(columns=[col], inplace=True)

    return features, labels

def single_run(model_cls, raw_features: pd.DataFrame, raw_labels: np.ndarray, variables: typing.List[typing.AnyStr], no_test, output_path: str, seed=None, weights=None):
    model = model_cls(variables)
    output_path = os.path.join(output_path, model_cls.name())
    _single_run(model, raw_features, raw_labels, no_test, output_path, seed = seed, weights = weights)


def _single_run(model, raw_features: pd.DataFrame, raw_labels: np.ndarray, no_test, output_path: str, seed=None, weights=None, delay_val=True):
    os.makedirs(output_path, exist_ok=True)
    index = raw_features.index
    train_index, test_index, val_index = get_data_split(raw_features.index, test=not no_test, seed=seed)
    model.train(
        raw_features.iloc[train_index].values, raw_labels[train_index],
        raw_features.iloc[val_index].values, raw_labels[val_index],
        raw_features.iloc[train_index].index, raw_features.iloc[val_index].index, weights=weights.iloc[train_index].values if weights is not None else None, delay_val=delay_val
    )

    if not no_test:
        with open(os.path.join(output_path, "predictions_test.csv",), "w") as fout:
            y_pred = model.predict(raw_features.iloc[test_index].values)
            fout.write(",".join(("doc,arm", "prediction", "target")) + "\n")
            for t in zip(index[test_index].values, y_pred, raw_labels[test_index]):
                fout.write(",".join((str(t[0][0]), str(t[0][1]), *map(str, t[1:]))) + "\n")
            fout.flush()

    with open(os.path.join(output_path, "predictions_full.csv"), "w") as fout:
        y_pred = model.predict(raw_features.values)
        fout.write(",".join(("set", "doc,arm", "prediction", "target")) + "\n")
        for i, t in enumerate(zip(index, y_pred, raw_labels)):
            fout.write(",".join(("train" if i in train_index else ("test" if test_index and i in test_index else "val"), str(t[0][0]),
                                 str(t[0][1]), *map(str, t[1:]))) + "\n")
        fout.flush()

    model.save(output_path)


def get_data_split(index, seed=None, test=False):
    documents = list({i[0] for i in index})
    random.shuffle(documents)
    if test:
        split = 0.8
    else:
        split = 0.9

    train_doc_index, val_doc_index = train_test_split(list(documents), random_state=seed, train_size=split)
    if test:
        test_doc_index, val_doc_index = train_test_split(val_doc_index, random_state=seed, train_size=0.5)
        test_index = [i for i, t in enumerate(index) if t[0] in test_doc_index]
    else:
        test_index = None

    train_index = [i for i, t in enumerate(index) if t[0] in train_doc_index]
    val_index = [i for i, t in enumerate(index) if t[0] in val_doc_index]

    return train_index, test_index, val_index


def get_cross_split(index, num_splits=3):
    documents = list({i[0] for i in index})
    random.shuffle(documents)
    splits = np.array_split(documents, num_splits)
    return [[i for i, t in enumerate(index) if t[0] in split] for split in splits]
