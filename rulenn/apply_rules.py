from rule_nn import CYK
import torch
import pickle
import os
import numpy as np
import pandas as pd

def print_rules(df: pd.DataFrame):
    for row in df.iterrows():
        print(" & ".join(df.columns[:-1][row[1][:-1]>0]) + " => " + str(row[1][-1]))

with open("../data/hbcp_gen.pkl", "rb") as fin:
    _, _, names = pickle.load(fin)


with open("../results/rules.txt", "r") as fin:
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
        config = dict(conjunctions=torch.tensor(df.iloc[:,:-1].values), rule_weights = torch.tensor(df.iloc[:,-1].values))
        model = CYK(len(names),len(names),1,config=config)
        print(model)
