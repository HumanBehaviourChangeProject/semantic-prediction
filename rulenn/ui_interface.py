import sys
import pathlib
import inspect

import pickle
import pandas as pd
import numpy as np
import torch

#src_file_path = inspect.getfile(lambda: None)

#PACKAGE_PARENT = pathlib.Path(src_file_path).parent
#SCRIPT_DIR = PACKAGE_PARENT / "rulenn"
#sys.path.append(str(SCRIPT_DIR))

sys.path.append("/Users/hastingj/Work/Python/semantic-prediction/rulenn")

from rule_nn import RuleNet
from apply_rules import print_rules, print_applied_rules, get_feature_row_str, logit

with open('data/hbcp_gen.pkl', "rb") as fin:
    features, labels, names = pickle.load(fin)

VERBOSE = True

model = None
df = None


def initialise_model():
    with open("results/rules.txt", "r") as fin:
        for line in fin:
            rules = []
            global df
            global model

            for rule in line.split(";"):
                *conjunctions, weight = rule.split("&")
                rules.append([float(c) for c in conjunctions] + [float(weight)])
            df = pd.DataFrame(rules)
            df.columns = names + ["not " + n for n in names] + ["weight"]
            #print_rules(df)
            #print("---")
            config = dict(conjunctions=logit(torch.tensor(df.iloc[:, :-1].values)),
                          rule_weights=torch.tensor(df.iloc[:, -1].values))
            model = RuleNet(len(names), len(rules), 1, config=config)
            model.eval()


def get_prediction(features):
    if model is None:
        return("Cannot predict, model not initialised")
    print("Current features: ", get_feature_row_str(features, df.columns))
    fits = model.calculate_fit(torch.tensor(features).unsqueeze(0))
    print("The following rules were applied:")
    reasons = print_applied_rules(df, fits, features)
    prediction = model.apply_fit(fits).item()
    print(
        f"The application of these rules resulted in the following prediction: {prediction}")
    print("\n---\n")
    return((prediction,reasons))
