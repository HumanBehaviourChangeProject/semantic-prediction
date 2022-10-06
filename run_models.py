from base import cross_val, single_run
from regression.mle import MLEModel
from rulenn.rule_nn import RuleNNModel
from dl import DeepLearningModel
import sys
import numpy as np
import pickle

import click

@click.group()
def cli():
    pass

@cli.command()
@click.argument('path')
@click.option('--select', default=None)
def single(path, select):
    with open(path, "rb") as fin:
        features, labels = pickle.load(fin)

    if select is not None:
        models_to_run = [m for m in model_classes if m.name() == select]
    else:
        models_to_run = model_classes

    features[np.isnan(features)] = 0
    variables = [x[1] for x in features.columns]
    for model_cls in models_to_run:
        single_run(
            model_cls,
            features,
            labels[:, 0],
            variables,
            "out"
        )

@cli.command()
@click.argument('path')
def cross(path):
    with open(path, "rb") as fin:
        features, labels = pickle.load(fin)

    features[np.isnan(features)] = 0
    variables = [x[1] for x in features.columns]
    for model_cls in model_classes:
        cross_val(
            model_cls,
            features,
            labels[:, 0],
            variables,
            "out"
        )

model_classes = [
    DeepLearningModel,
    MLEModel,
    RuleNNModel
]

if __name__ == '__main__':
    cli()
