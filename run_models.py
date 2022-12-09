import csv
import random

import pandas as pd
import tqdm

from base import cross_val, single_run, filter_features, _single_run
from regression.mle import MLEModel
from regression.random_forest import RFModel
from rulenn.rule_nn import RuleNNModel
from rulenn.apply_rules import print_rules, apply_rules
from dl import DeepLearningModel
import numpy as np
import pickle
import math
import click
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

model_classes = [
    RFModel,
    DeepLearningModel,
    MLEModel,
    RuleNNModel
]

@click.group()
def cli():
    pass


@cli.command()
@click.argument('path')
@click.option('--select', default=None, help="Available options: " + ", ".join(m.name() for m in model_classes))
@click.option('--filters', is_flag=True, default=None)
@click.option('--no-test', is_flag=True, default=False)
@click.option('--weighted', is_flag=True, default=False)
@click.option('--out', default="out")
def single(*args, **kwargs):
    _single(*args, **kwargs)


def _load_data(path, filters, weighted=False, drop=None):
    with open(path, "rb") as fin:
        features, labels = pickle.load(fin)

    features[np.isnan(features)] = 0
    if weighted:
        copy_features = pd.DataFrame()
        with open("data/analysed.csv") as fin:
            reader = csv.reader(fin)
            weights = {(int(a), int(b), c, d): (max(1, int(math.log(float(v),5))) if v != "" else 1) for a, b, c, d, v in reader}
            weights = [(k, v) for k, v in weights.items() if k in features.index]
            #keys, values = zip(*weights)
            #weights = pd.DataFrame(values, index=keys)
            #weights = weights.astype(float)
            features = pd.DataFrame(y for x in [[features.loc[key]]*value for key, value in weights] for y in x)
            labels = np.array([y for x in [[labels[i]] * value for i, (key, value) in enumerate(weights)] for y in x])
            print(copy_features)

    if filters is not None:
        features = filter_features(features)

    if drop is not None:
        col = features.columns[int(drop)]
        print("Exclude column:", col)
        features.drop(columns=[col], inplace=True)

    return features, labels

def _single(path, select, filters, no_test, weighted, seed=None, out="out", **kwargs):
    if select is not None:
        models_to_run = [m for m in model_classes if m.name() == select]
    else:
        models_to_run = model_classes

    features, labels = _load_data(path, filters, weighted)

    variables = [x[1] for x in features.columns]
    for model_cls in models_to_run:
        single_run(
            model_cls,
            features,
            labels[:, 0],
            variables,
            no_test,
            out,
            #weights=weights,
            seed=seed
        )


@cli.command()
@click.argument('path')
@click.option('--out', default="out")
@click.option('--filters', is_flag=True, default=False)
@click.option('--select', default=None, help="Available options: " + ", ".join(m.name() for m in model_classes))
@click.option('--no-test', is_flag=True, default=False)
@click.option('--weighted', is_flag=True, default=False)
@click.option('--drop-feature', default=None)
def cross(*args, **kwargs):
    _cross(*args, **kwargs)

def _cross(path, out, filters, select, no_test, weighted, drop_feature, **kwargs):
    features, labels = _load_data(path, filters, weighted, drop_feature)

    if select is not None:
        models_to_run = [m for m in model_classes if m.name() == select]
    else:
        models_to_run = model_classes

    variables = [x[1] for x in features.columns]
    cross_val(
        models_to_run,
        features,
        labels[:, 0],
        variables,
        out,
        no_test
    )

@cli.command()
@click.argument('path')
@click.option('--select', default=None, help="Available options: " + ", ".join(m.name() for m in model_classes))
def cross_full(path, select):
    for filters in (True, False):
        for weighted in (True, False):
            out = "out"
            out += "_" + ("filtered" if filters else "unfiltered")
            out += "_" + ("weighted" if weighted else "unweighted")
            _cross(path=path,out=out,filters=filters,select=select,no_test=False,weighted=weighted)


@cli.command()
@click.argument('path')
def print_rules(path):
    model = RuleNNModel.load(path)
    model.print_rules()


@cli.command()
@click.argument('path')
@click.argument('checkpoint')
@click.option('--filters', is_flag=True, default=None)
@click.option('-v', default=False, count=True)
@click.option('--threshold', type=float, default = 0.1)
def apply(path, checkpoint, filters, v, threshold):
    model = RuleNNModel.load(checkpoint, fix_conjunctions=False)
    model.model.eval()
    features, labels = _load_data(path, filters, False)

    for row in features.values:
        applied_rules, result = apply_rules(model, row, model.variables)
        print("The following rules were applied:")
        for conjunction, impact, fit in applied_rules:
            if impact > 0:
                impstr = f"raise predicted outcome by {impact:.2f}"
            else:
                impstr = f"lower predicted outcome by {-impact:.2f}"
            if v>0:
                impstr += f" (fit: {fit.item():.2f})"
            if fit > threshold:
                print(" & ".join(name + ("" if v <= 0 else f"[{weight:.2f}]") for name, weight in conjunction) + " => " + impstr)
        print(f"The application of these rules resulted in the following prediction: {result:.2f}")
        print("\n---\n")


@cli.command()
@click.argument('path')
@click.argument('checkpoint')
@click.option('--filters', is_flag=True, default=None)
@click.option('--weighted', is_flag=True, default=False)
def fine_tune(path, checkpoint, filters, weighted):
    model = RuleNNModel.load(checkpoint)
    features, labels = _load_data(path, filters, weighted)
    _single_run(model, features, labels, True, "/tmp/out", delay_val=False)

@cli.command()
@click.argument('n')
@click.argument('path')
@click.option('--select', default=None, help="Available options: " + ", ".join(m.name() for m in model_classes))
@click.option('--filters', is_flag=True, default=None)
@click.option('--no-test', is_flag=True, default=False)
@click.option('--weighted', is_flag=True, default=False)
def runcopy(n, *args, **kwargs):
    import shutil
    for i in tqdm.tqdm(list(range(int(n)))):
        _single(*args, seed=random.randint(1,1000), **kwargs)
        shutil.copyfile("out/rulenn/model.json", f"out/rulenn/model.{i}.json")

if __name__ == '__main__':
    cli()
