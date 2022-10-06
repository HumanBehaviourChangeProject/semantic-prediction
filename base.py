import csv
import os

from sklearn.model_selection import train_test_split
import numpy as np
import typing
import abc
import random


class BaseModel:
    @abc.abstractmethod
    def train(self, train_features: np.ndarray, train_labels: np.ndarray, val_features: np.ndarray, val_labels: np.ndarray, variables:typing.List[typing.AnyStr]):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, features):
        raise NotImplementedError

    @abc.abstractclassmethod
    def name(cls):
        raise NotImplementedError

    @classmethod
    def prepare_data(cls, features, labels):
        return features, labels

    @abc.abstractmethod
    def save(self, path):
        raise NotImplementedError


def cross_val(model_cls, raw_features: np.ndarray, raw_labels: np.ndarray, variables: typing.List[typing.AnyStr], output_path: str):
    os.makedirs(os.path.join(output_path, model_cls.name()), exist_ok=True)
    with open(os.path.join(output_path, model_cls.name(), "crossval.txt"), "w") as fout:
        for _ in range(10):
            chunks = get_cross_split(raw_features.index, 5)
            for i in range(len(chunks) - 1):
                train_index = [
                    c for j in range(len(chunks)) for c in chunks[j] if i != j and i != j - 1
                ]
                val_index = chunks[i]
                test_index = chunks[i + 1]
                model = model_cls()
                model.train(
                    *model_cls.prepare_data(raw_features.iloc[train_index],  raw_labels[train_index]),
                    *model_cls.prepare_data(raw_features.iloc[val_index],  raw_labels[val_index]),
                    variables
                )
                test_features, test_labels = model_cls.prepare_data(raw_features.iloc[test_index], raw_labels[test_index])
                y_pred = model.predict(test_features)

                for pred, targ in zip(y_pred.tolist(), test_labels.squeeze(-1).tolist()):
                    fout.write(str(pred - targ) + "\n")


def single_run(model_cls, raw_features: np.ndarray, raw_labels: np.ndarray, variables: typing.List[typing.AnyStr], output_path: str, seed=None):
    os.makedirs(os.path.join(output_path, model_cls.name()), exist_ok=True)
    index = raw_features.index
    train_index, test_index, val_index = get_data_split(raw_features.index)
    model = model_cls()

    model.train(
        *model_cls.prepare_data(raw_features.iloc[train_index], raw_labels[train_index]),
        *model_cls.prepare_data(raw_features.iloc[val_index], raw_labels[val_index]),
        variables
    )

    test_features, test_labels = model_cls.prepare_data(raw_features.iloc[test_index], raw_labels[test_index])

    with open(os.path.join(output_path, model_cls.name(), "predictions_test.csv",), "w") as fout:
        y_pred = model.predict(test_features)
        fout.write(",".join(("doc,arm", "prediction", "target")) + "\n")
        for t in zip(index[test_index].values, y_pred, test_labels):
            fout.write(",".join((str(t[0][0]), str(t[0][1]), *map(str, t[1:]))) + "\n")
        fout.flush()

    all_features, all_labels = model_cls.prepare_data(raw_features, raw_labels)
    with open(os.path.join(output_path, model_cls.name(), "predictions_full.csv"), "w") as fout:
        y_pred = model.predict(all_features)
        fout.write(",".join(("set", "doc,arm", "prediction", "target")) + "\n")
        for i, t in enumerate(zip(index, y_pred, all_labels)):
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
    p = 1/num_splits
    splits = []
    remainder = documents
    for i in range(num_splits):
        split, remainder = train_test_split(remainder, train_size=p)
        splits.append(split)

    return [[i for i, t in enumerate(index) if t[0] in split] for split in splits]


def get_feature_set(included_values):
    with open("data/feature_filter.csv", "rt") as fin:
        reader = csv.reader(fin)
        return {i: (v in included_values) for i, _, v, _ in reader if v in included_values}
