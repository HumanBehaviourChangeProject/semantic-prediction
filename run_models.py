import csv
import random

import pandas as pd
import tqdm

from base import cross_val, single_run, filter_features
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
@click.option('--filters', is_flag=True, default=False)
@click.option('--no-test', is_flag=True, default=False)
@click.option('--weighted', is_flag=True, default=False)
def single(*args, **kwargs):
    _single(*args, **kwargs)


def _single(path, select, filters, no_test, weighted, seed=None, **kwargs):
    with open(path, "rb") as fin:
        features, labels = pickle.load(fin)

    if select is not None:
        models_to_run = [m for m in model_classes if m.name() == select]
    else:
        models_to_run = model_classes

    features[np.isnan(features)] = 0

    weights = None
    if weighted:
        copy_features = pd.DataFrame()
        with open("data/analysed.csv") as fin:
            reader = csv.reader(fin)
            weights = {(int(a), int(b), c, d): (max(1, int(math.log2(float(v)))) if v != "" else 1) for a, b, c, d, v in reader}
            weights = [(k, v) for k, v in weights.items() if k in features.index]
            #keys, values = zip(*weights)
            #weights = pd.DataFrame(values, index=keys)
            #weights = weights.astype(float)
            features = pd.DataFrame(y for x in [[features.loc[key]]*value for key, value in weights] for y in x)
            labels = np.array([y for x in [[labels[i]] * value for i, (key, value) in enumerate(weights)] for y in x])
            print(copy_features)


    if filters is not None:
        features = filter_features(features)

    variables = [x[1] for x in features.columns]
    for model_cls in models_to_run:
        single_run(
            model_cls,
            features,
            labels[:, 0],
            variables,
            no_test,
            "out",
            #weights=weights,
            seed=seed
        )


@cli.command()
@click.argument('path')
@click.option('--out', default="out")
@click.option('--filters', is_flag=True, default=False)
def cross(path, out, filters):
    with open(path, "rb") as fin:
        features, labels = pickle.load(fin)

    features[np.isnan(features)] = 0

    if filters is not None:
        features = filter_features(features)

    variables = [x[1] for x in features.columns]
    cross_val(
        model_classes,
        features,
        labels[:, 0],
        variables,
        out
    )

@cli.command()
@click.argument('path')
def printrules(path):
    model = RuleNNModel.load(path)
    model.print_rules()


@cli.command()
@click.argument('path')
@click.argument('checkpoint')
@click.option('--filters', is_flag=True, default=False)
@click.option('-v', default=False, count=True)
@click.option('--threshold', type=float, default = 0.1)
def apply(path, checkpoint, filters, v, threshold):
    model = RuleNNModel.load(checkpoint)
    with open(path, "rb") as fin:
        raw_features, raw_labels = pickle.load(fin)
    raw_features[np.isnan(raw_features)] = 0

    if filters:
        features = filter_features(raw_features)
    else:
        features = raw_features

    for row in features.values:
        applied_rules, result = apply_rules(model, [row], features.columns)
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
@click.argument('n')
@click.argument('path')
@click.option('--select', default=None, help="Available options: " + ", ".join(m.name() for m in model_classes))
@click.option('--filters', is_flag=True, default=False)
@click.option('--no-test', is_flag=True, default=False)
@click.option('--weighted', is_flag=True, default=False)
def runcopy(n, *args, **kwargs):
    import shutil
    for i in tqdm.tqdm(list(range(int(n)))):
        _single(*args, seed=random.randint(1,1000), **kwargs)
        shutil.copyfile("out/rulenn/model.json", f"out/rulenn/model.{i+69}.json")

if __name__ == '__main__':
    cli()
