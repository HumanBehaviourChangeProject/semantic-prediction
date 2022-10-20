import torch
import pickle
import os
import numpy as np
import pandas as pd
import tqdm

VERBOSE = False

def print_rules(df: pd.DataFrame):
    for row in df.iterrows():
        print(" & ".join(df.columns[:-1][row[1][:-1]>0.3]) + " => " + str(row[1][-1]))


def print_applied_rules(conjunctions, rule_weights, fits, features, fit_threshold=0.1, feature_threshold=0.1):
    rules = []
    for row, weight, fit in zip(conjunctions, rule_weights, fits[0]):
        conjunction = [f"{features[i][1]}"+ (f"[{row[i]}]" if VERBOSE else "") for i in range(len(features)-1) if row[i] > feature_threshold]
        impact = fit.item()*weight.item()
        rules.append((conjunction, impact, fit))
    rules = sorted(rules, key=lambda x:-abs(x[1]))
    for conjunction, impact, fit in rules:
        if impact > 0:
            impstr = f"raise predicted outcome by {impact}"
        else:
            impstr = f"lower predicted outcome by {-impact}"

        if VERBOSE:
            impstr += f" (fit: {fit.item()})"
        if fit > fit_threshold:
            print(" & ".join(conjunction) + " => " + impstr)


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


