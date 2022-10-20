from seaborn import boxplot, kdeplot
import pandas as pd
from matplotlib import pyplot as plt
from math import fabs
buckets = []

folder = "out"

with open(f"{folder}/random_forest/crossval.txt", "r") as fin:
    buckets.append(("decision tree", [float(l) for l in fin]))

with open(f"{folder}/rulenn/crossval.txt", "r") as fin:
    buckets.append(("rule", [float(l) for l in fin]))

with open(f"{folder}/mle/crossval.txt", "r") as fin:
    buckets.append(("mixed-effect", [-float(l) for l in fin]))

with open(f"{folder}/deep/crossval.txt", "r") as fin:
    buckets.append(("deep", [float(l) for l in fin]))

df = pd.DataFrame([{"model":n,"error":v} for n, l in buckets for v in l])

boxplot(x=[name for name, l in buckets for _ in l],y=[abs(x) for _, l in buckets for x in l], showfliers=False)
plt.savefig("box.png")
plt.clf()
plt.close()

kdeplot(data=df, x="error", hue="model", common_norm=False)
plt.savefig("kde.png")
plt.clf()
plt.close()