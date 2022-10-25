import torch
import pickle
import os
import numpy as np
import pandas as pd
from rulenn.rule_nn import RuleNNModel

VERBOSE = False


def print_rules(df: pd.DataFrame):
    for row in df.iterrows():
        print(" & ".join(df.columns[:-1][row[1][:-1]>0.3]) + " => " + str(row[1][-1]))


def apply_rules(container:RuleNNModel, x:np.ndarray, features, feature_threshold=0.1):
    """
    Args:
        container: A trained instance of RuleNNModel
        x: A 1d-array of features
        features: names of the features
        feature_threshold: weight threshold above which the fit of a feature is considered part of a rule

    Returns: (conjunction_list, fit)
        conjunction_list: A list of tuples (lhs, impact, con_fit) where each element represents the application of
            a rule to `x`. `lhs` is a list of tuples (f, w) of those features whole weight w was above the
            `feature_threshold` in this rule. `fit` is the degree to which this rule fits the feature vector `x` and
            `con_fit` represents the product of this rule's fit with it's rule weight.
        fit: The overal prediction of the system

    """
    rules = []
    conjunctions = container.model.non_lin(container.model.conjunctions)
    fits = container.model.calculate_fit(container._prepare_single([x]))
    for row, weight, fit in zip(conjunctions, container.model.rule_weights, fits[0]):
        conjunctions = [(features[i][1], row[i].item()) for i in range(len(features)-1) if row[i] > feature_threshold]
        impact = fit.item()*weight.item()
        rules.append((conjunctions, impact, fit))
    return sorted(rules, key=lambda x:-abs(x[1])), container.model.apply_fit(fits).item()


def get_feature_row_str(row, names,threshold=0.1):
    present_features = []
    for i in range(len(row[1])):
        if row[1][i] > threshold:
            s = names[i]
            if VERBOSE:
                s += f" [{row[1][i]}]"
            present_features.append(s)
    return " & ".join(present_features)


def logit(mu0):
    mu = mu0*0.999+0.0001
    return torch.log(mu/(1-mu))


