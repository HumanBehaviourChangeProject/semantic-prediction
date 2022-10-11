import csv
import os

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import typing
import abc
import random
import multiprocessing as mp

T = typing.TypeVar("T")

class BaseModel(typing.Generic[T]):
    def _train(self, train_features: T, train_labels: T, val_features: T, val_labels: T, variables:typing.List[typing.AnyStr], train_docs, val_docs, verbose=True):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, train_features: np.ndarray, train_labels: np.ndarray, val_features: np.ndarray, val_labels: np.ndarray, variables:typing.List[typing.AnyStr], train_docs, val_docs, verbose=True):
        return self._train(
            self._prepare_single(train_features),
            self._prepare_single(train_labels),
            self._prepare_single(val_features),
            self._prepare_single(val_labels),
            variables, train_docs, val_docs, verbose=verbose
        )

    @abc.abstractmethod
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


def cross_val(model_classes, raw_features: np.ndarray, raw_labels: np.ndarray, variables: typing.List[typing.AnyStr], output_path: str):

    outs = {c: np.empty(0) for c in model_classes}

    chunks = get_cross_split(raw_features.index, 5)
    for i in range(len(chunks)):
        train_index = [
            c for j in range(len(chunks)) for c in chunks[j] if i != j and i != j - 1
        ]
        val_index = chunks[i]
        test_index = chunks[(i + 1) % len(chunks)]

        for model_cls in model_classes:
            model = model_cls()
            model.train(raw_features.iloc[train_index].values, raw_labels[train_index],
                        raw_features.iloc[val_index].values, raw_labels[val_index], variables,
                        raw_features.iloc[train_index].index, raw_features.iloc[val_index].index,
                        verbose=False
                        )
            y_pred = model.predict(raw_features.iloc[test_index].values)
            outs[model_cls] = np.concatenate((outs[model_cls], y_pred - raw_labels[test_index]))

    for model_cls, values in outs.items():
        os.makedirs(os.path.join(output_path, model_cls.name()), exist_ok=True)
        outfile = os.path.join(output_path, model_cls.name(), "crossval.txt")
        with open(outfile, "w") as fout:
            fout.write("\n".join(map(str, values)))


def single_run(model_cls, raw_features: pd.DataFrame, raw_labels: np.ndarray, variables: typing.List[typing.AnyStr], output_path: str, seed=None):
    os.makedirs(os.path.join(output_path, model_cls.name()), exist_ok=True)
    index = raw_features.index
    train_index, test_index, val_index = get_data_split(raw_features.index)
    model = model_cls()

    model.train(
        raw_features.iloc[train_index].values, raw_labels[train_index],
        raw_features.iloc[val_index].values, raw_labels[val_index],
        variables, raw_features.iloc[train_index].index, raw_features.iloc[val_index].index
    )

    with open(os.path.join(output_path, model_cls.name(), "predictions_test.csv",), "w") as fout:
        y_pred = model.predict(raw_features.iloc[test_index].values)
        fout.write(",".join(("doc,arm", "prediction", "target")) + "\n")
        for t in zip(index[test_index].values, y_pred, raw_labels[test_index]):
            fout.write(",".join((str(t[0][0]), str(t[0][1]), *map(str, t[1:]))) + "\n")
        fout.flush()

    with open(os.path.join(output_path, model_cls.name(), "predictions_full.csv"), "w") as fout:
        y_pred = model.predict(raw_features.values)
        fout.write(",".join(("set", "doc,arm", "prediction", "target")) + "\n")
        for i, t in enumerate(zip(index, y_pred, raw_labels)):
            fout.write(",".join(("train" if i in train_index else ("test" if i in test_index else "val"), str(t[0][0]),
                                 str(t[0][1]), *map(str, t[1:]))) + "\n")
        fout.flush()

    model.save(os.path.join(output_path, model_cls.name()))

def get_data_split(index, seed=None):
    documents = list({i[0] for i in index})
    random.shuffle(documents)

    train_doc_index, val_doc_index = train_test_split(list(documents), random_state=seed, train_size=0.8)
    test_doc_index, val_doc_index = train_test_split(val_doc_index, random_state=seed, train_size=0.5)

    train_index = [i for i, t in enumerate(index) if t[0] in train_doc_index]
    test_index = [i for i, t in enumerate(index) if t[0] in test_doc_index]
    val_index = [i for i, t in enumerate(index) if t[0] in val_doc_index]

    return train_index, test_index, val_index


def get_cross_split(index, num_splits=3):
    documents = list({i[0] for i in index})
    random.shuffle(documents)
    splits = np.array_split(documents, num_splits)
    return [[i for i, t in enumerate(index) if t[0] in split] for split in splits]


def get_feature_set(included_values):
    with open("data/feature_filter.csv", "rt") as fin:
        reader = csv.reader(fin)
        return {i: (v in included_values) for i, _, v, _ in reader if v in included_values}
