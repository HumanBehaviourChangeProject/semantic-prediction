import os
import sys
import json
import math
from rulenn.rule_nn import RuleNNModel
from base import _load_data
import torch
import tqdm
import numpy as np

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def logit(mu0):
    mu = mu0*0.999+0.0001
    return math.log(mu/(1-mu))

def consolidate(path):
    rulessets = []
    runs = 0
    features, labels = _load_data("data/hbcp_gen.pkl", None, False)
    features = torch.cat((torch.tensor(features.values), torch.tensor(features.values)), dim=1)
    for f in tqdm.tqdm([f for f in os.listdir(path) if f.startswith("out_")]):
        runs += 1
        with open(os.path.join(path,f,"rulenn","model.json"), "r") as fin:
            checkpoint = os.path.join(path, f)
            container = RuleNNModel.load(checkpoint, fix_conjunctions=False)
            fit = container.model.calculate_fit(features)
            d = json.load(fin)
            items = d["variables"] + [f"not {v}" for v in d["variables"]]
            rulessets.append([({v for c,v in zip(conjunction, items) if sigmoid(float(c)) > 0.2}, float(weight)) for conjunction, weight in zip(d["conjunctions"], d["weights"])])

    transactions = [con for rule in rulessets for con, _ in rule]
    tres = runs*0.80
    current = [(set(), transactions)]
    print(tres)
    frequents = []
    while current:
        new_current = []
        for itemset, contained_in in current:
            missing = sorted({c for cs in contained_in for c in cs if not itemset or c > max(itemset)})
            some_frequent = False
            for n in missing:
                new_item_set = itemset.union({n})
                new_contained_in = [t for t in contained_in if n in t]
                if len(new_contained_in) > tres:
                    new_current.append((new_item_set, new_contained_in))
                    some_frequent = True
            frequents.append((itemset, len(contained_in)))
        current = new_current
    with open(os.path.join(path, "frequent_model.json"), "w") as fout:
        conjunctions = []
        weights = []

        for itms, contained_in in frequents:
            conjunctions.append([100 if c in itms else -100 for c in items])
            match_weights = [w for rule in rulessets for con, w in rule if itms.issubset(con)]
            avg = np.mean(match_weights)
            std = np.std(match_weights)
            print(contained_in," & ".join(itms), "->", avg, std)
            weights.append(avg)
        json.dump(dict(
            variables=items,
            conjunctions=conjunctions,
            weights=weights,
            base=10,
        ), fout)
            #fout.write("&".join(map(str, itms)) + ";" + str(frequency) + "\n")

if __name__ == "__main__":
    consolidate(sys.argv[1])
