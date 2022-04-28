from rule_nn import RuleNet
import torch
import pickle
import os
import numpy as np
import pandas as pd

def print_rules(df: pd.DataFrame):
    for row in df.iterrows():
        print(" & ".join(df.columns[:-1][row[1][:-1]>0.1]) + " => " + str(row[1][-1]))


def print_applied_rules(df: pd.DataFrame, fits, threshold=0.1):
    for row, fit in zip(df.iterrows(), fits[0]):
        if fit>threshold:
            print(" & ".join(df.columns[:-1][row[1][:-1]>0.1]) + " => " + f"{fit.item()}*{row[1][-1]}")

def get_feature_row_str(row, threshold=0.1):
    return " & ".join(row[1].index[i] for i in range(len(row[1])) if row[1][i] > threshold)

def logit(mu0):
    mu = mu0*0.999+0.0001
    return torch.log(mu/(1-mu))

if __name__=="__main__":
    import sys

    with open(sys.argv[1], "rb") as fin:
        features, labels, names = pickle.load(fin)

    with open("results/rules.txt", "r") as fin:
        rulessets = []
        for line in fin:
            rules = []
            weights = []

            for rule in line.split(";"):
                *conjunctions, weight = rule.split("&")
                rules.append([float(c) for c in conjunctions]+[float(weight)])
            df = pd.DataFrame(rules)
            df.columns = names + ["not "+n for n in names] + ["weight"]
            print_rules(df)
            print("---")
            config = dict(conjunctions=logit(torch.tensor(df.iloc[:,:-1].values)), rule_weights = torch.tensor(df.iloc[:,-1].values))
            model = RuleNet(len(names), len(rules), 1, config=config)
            model.eval()
            for row in features.iterrows():
                print("Current features: ", get_feature_row_str(row))
                fits = model.calculate_fit(torch.tensor(row[1]).unsqueeze(0))
                print("The following rules were applied:")
                print_applied_rules(df, fits)
                print("The application of these rules resulted in the following prediction:")
                print(model.apply_fit(fits))
                print("---")


