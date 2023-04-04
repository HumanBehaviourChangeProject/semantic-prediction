from seaborn import boxplot, kdeplot, violinplot
import pandas as pd
from matplotlib import pyplot as plt
from math import fabs
buckets = []

folder = "results/runs/5-final"

with open(f"{folder}/random_forest/crossval.txt", "r") as fin:
    buckets.append(("decision tree", [float(l) for l in fin]))

with open(f"{folder}/rulenn/crossval.txt", "r") as fin:
    buckets.append(("rule", [float(l) for l in fin]))

with open(f"{folder}/mle/crossval.txt", "r") as fin:
    buckets.append(("mixed-effect", [-float(l) for l in fin]))

with open(f"{folder}/deep/crossval.txt", "r") as fin:
    buckets.append(("deep", [float(l) for l in fin]))

df = pd.DataFrame([{"model":n,"error":v} for n, l in buckets for v in l])
df['abserror'] = df['error'].abs()

#boxplot(x=[name for name, l in buckets for _ in l],y=[abs(x) for _, l in buckets for x in l], showfliers=False)
boxplot(data=df,x="model",y="abserror", showfliers=False)
plt.savefig("box.png")
plt.clf()
plt.close()

kdeplot(data=df, x="error", hue="model", common_norm=False)
plt.savefig("kde.png")
plt.clf()
plt.close()

df['model'].unique()

meanabserrors = {}

for modelname in df['model'].unique():
    meanabserrors[modelname] = df.query(f"model=='{modelname}'")['error'].abs().mean()




### Get the grand mean of the dataset as an additional comparison
import inspect
import pathlib
import sys
import numpy as np
import pickle
src_file_path = inspect.getfile(lambda: None)

PACKAGE_PARENT = pathlib.Path(src_file_path).parent
PACKAGE_PARENT2 = PACKAGE_PARENT.parent
sys.path.append(str(PACKAGE_PARENT2))

checkpoint = 'examples/model_unweighted.json'
path = 'data/hbcp_gen.pkl'
filters = False
with open(path, "rb") as fin:
    raw_features, raw_labels = pickle.load(fin)
raw_features[np.isnan(raw_features)] = 0



errorscrossvalmean = []
for i in range(0,5):
    start = int(i * len(raw_labels)/5)
    end = int(start + (i+1 * len(raw_labels)/5))
    testdata = [x[0] for x in raw_labels[start:end].tolist()]
    traindata = [x[0] for x in   raw_labels[0:start].tolist() +  raw_labels[end:len(raw_labels)].tolist() ]
    meantraindata = np.mean(traindata)
    for val in testdata:
        err = val - meantraindata
        errorscrossvalmean.append(err)

np.mean(np.abs(errorscrossvalmean))  # 9.98039626526184



## Features treatment

feature_df = pd.read_csv("data/model_input_data.csv")
cleaned_feature_df = pd.read_csv("data/hbcp_gen.csv")