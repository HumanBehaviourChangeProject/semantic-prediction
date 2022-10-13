from base import cross_val, single_run, filter_features
from regression.mle import MLEModel
from regression.random_forest import RFModel
from rulenn.rule_nn import RuleNNModel
from dl import DeepLearningModel
import numpy as np
import pickle

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
def single(path, select, filters, no_test):
    with open(path, "rb") as fin:
        features, labels = pickle.load(fin)

    if select is not None:
        models_to_run = [m for m in model_classes if m.name() == select]
    else:
        models_to_run = model_classes

    features[np.isnan(features)] = 0

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
            "out"
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
def runcopy(path):
    import shutil
    select = "rulenn"
    filters = True
    for i in range(10):
        with open(path, "rb") as fin:
            features, labels = pickle.load(fin)

        if select is not None:
            models_to_run = [m for m in model_classes if m.name() == select]
        else:
            models_to_run = model_classes

        features[np.isnan(features)] = 0

        if filters is not None:
            features = filter_features(features)

        variables = [x[1] for x in features.columns]
        for model_cls in models_to_run:
            single_run(
                model_cls,
                features,
                labels[:, 0],
                variables,
                "out"
            )
        shutil.copyfile("out/rulenn/model.json", f"out/rulenn/model.{i}.json")

if __name__ == '__main__':
    cli()
