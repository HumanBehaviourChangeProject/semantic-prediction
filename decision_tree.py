import torch
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz, DecisionTreeRegressor

def is_number(x):
    try:
        float(x)
    except:
        return False
    else:
        return True

def cross_val(patience, features, labels, variables):
    results = []
    features[features == None] = 0
    not_bcts = [not is_number(v[0]) for v in variables]
    features[:, not_bcts] = 0
    with open("results/decision_tree_bct.txt", "w") as fout:
        for _ in range(100):
            index = list(np.array(range(features.shape[0])))
            random.shuffle(index)
            step = len(index) // 10
            chunks = [index[i:i + step] for i in range(0, len(index), step)]
            for i in range(len(chunks)):
                train_index = [c for j in range(len(chunks)) for c in chunks[j] if i != j ]
                val_index = chunks[i]
                best = main(patience, features, labels, train_index, val_index, variables)
                fout.writelines(str(x.item()) + "\n" for x in best)


def get_rule(tree: DecisionTreeRegressor, root:int, variables):
    if tree.tree_.children_left[root] != tree.tree_.children_right[root]:
        f = tree.tree_.children_left[root]
        yield [variables[f]] + get_rule(tree, tree.tree_.children_left[root], variables)
        yield [variables[f]] + get_rule(tree, tree.tree_.children_right[root], variables)
    else:
        yield tree.tree_.value[root]


def main(p, features, labels, train_index, val_index, variables):
    regr_1 = RandomForestRegressor(100, max_depth=4, max_leaf_nodes=7)
    regr_1.fit(X=features[train_index], y=labels[train_index,0])
    pred = regr_1.predict(X=features[val_index])
    return pred - labels[val_index, 0]

if __name__ == "__main__":
    import pickle
    import sys

    with open(sys.argv[2], "rb") as fin:
        features, labels, rename = pickle.load(fin)
    cross_val(int(sys.argv[1]), features.astype(float).values,
         labels.astype(float), rename)
