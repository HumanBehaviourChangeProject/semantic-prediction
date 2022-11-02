import os
import sys
import json
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def logit(mu0):
    mu = mu0*0.999+0.0001
    return math.log(mu/(1-mu))

def consolidate(path):
    rulessets = []
    runs = 0
    for f in os.listdir(path):
        if f.startswith("model."):
            runs += 1
            with open(os.path.join(path, f), "r") as fin:
                d = json.load(fin)
                for conjunctions, weight in zip(d["conjunctions"], d["weights"]):
                    rules = []
                    items = d["variables"] + [f"not {v}" for v in d["variables"]]
                    conjunctions = {v for c,v in zip(conjunctions, items) if sigmoid(float(c)) > 0.2}
                    rules.append((conjunctions, float(weight)))
                    rulessets.append(rules)



    transactions = [con for rule in rulessets for con, _ in rule]
    #items = {x for r in transactions for x in r}
    initial_item_sets = [frozenset({x}) for x in items]

    tres = runs*0.1
    current = [i for i,l in ((i, sum(1 for t in transactions if i.issubset(t))) for i in initial_item_sets) if l > tres]
    print(tres)
    frequents = []
    while current:
        nxt = [(i,l) for i,l in ((i, sum(1 for t in transactions if i.issubset(t))) for i in current) if l > tres]
        frequents += list(nxt)
        current = [frozenset({n}.union(i)) for i,l in nxt for n in items if n > max(i)]

    with open(os.path.join(path, "frequent_model.json"), "w") as fout:
        conjunctions = []
        weights = []

        for itms, frequency in frequents:
            conjunctions.append([100 if c in itms else -100 for c in items])
            match_weights = [w for rule in rulessets for con, w in rule if itms.issubset(con)]
            avg = sum(match_weights)/len(match_weights)
            print(" & ".join(itms), "->", avg)
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
