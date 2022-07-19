from rule_nn import RuleNet
import torch
import pickle
import os
import numpy as np
import pandas as pd

VERBOSE = True

def print_rules(df: pd.DataFrame):
    for row in df.iterrows():
        print(" & ".join(df.columns[:-1][row[1][:-1]>0.3]) + " => " + str(row[1][-1]))


def print_applied_rules(df: pd.DataFrame, fits, features, fit_threshold=0.1, feature_threshold=0.1):
    rules = []
    for row, fit in zip(df.iterrows(), fits[0]):
        conjunction = [f"{df.columns[i]}"+ (f"[{row[1][i]}]" if VERBOSE else "") for i in range(len(df.columns)-1) if row[1][i]>feature_threshold]
        impact = fit.item()*row[1][-1]
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
    return rules


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


#if __name__=="__main__":
#    import sys
#
#    with open(sys.argv[1], "rb") as fin:
#        features, labels, names = pickle.load(fin)
#
#    for i in range(2, len(sys.argv)):
#        if sys.argv[i] == "-v":
#            VERBOSE=True
#
#    with open("results/rules.txt", "r") as fin:
#        rulessets = []
#        for line in fin:
#            rules = []
#            weights = []
#
#            for rule in line.split(";"):
#                *conjunctions, weight = rule.split("&")
#                rules.append([float(c) for c in conjunctions]+[float(weight)])
#            df = pd.DataFrame(rules)
#            df.columns = names + ["not "+n for n in names] + ["weight"]
#            print_rules(df)
#            print("---")
#            config = dict(conjunctions=logit(torch.tensor(df.iloc[:,:-1].values)), rule_weights = torch.tensor(df.iloc[:,-1].values))
#            model = RuleNet(len(names), len(rules), 1, config=config)
#            model.eval()
#            for row in features.iterrows():
#                print("Current features: ", get_feature_row_str(row, df.columns))
#                fits = model.calculate_fit(torch.tensor(row[1]).unsqueeze(0))
#                print("The following rules were applied:")
#                print_applied_rules(df, fits, row[1])
#                print(f"The application of these rules resulted in the following prediction: {model.apply_fit(fits).item()}")
#                print("\n---\n")


