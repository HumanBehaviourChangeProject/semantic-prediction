import inspect
import pathlib
import pickle
import sys
from os import listdir

import numpy as np
import pandas as pd
import statistics

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

### Handle local imports

src_file_path = inspect.getfile(lambda: None)

PACKAGE_PARENT = pathlib.Path(src_file_path).parent
PACKAGE_PARENT2 = PACKAGE_PARENT.parent
#SCRIPT_DIR = PACKAGE_PARENT2 / "rulenn"
#DATAPR_DIR = PACKAGE_PARENT2 / "dataprocessing"
sys.path.append(str(PACKAGE_PARENT2))
#sys.path.append(str(SCRIPT_DIR))
#sys.path.append(str(DATAPR_DIR))

from base import filter_features
from rulenn.rule_nn import RuleNNModel

###  Server state

#checkpoint = 'examples/model_consolidated.json'
path = 'data/hbcp_gen.pkl'
filters = False

#model = RuleNNModel.load(checkpoint)
with open(path, "rb") as fin:
    raw_features, raw_labels = pickle.load(fin)
raw_features[np.isnan(raw_features)] = 0

# In the raw feature table we have 153 columns

featurecounts = raw_features.astype(bool).sum(axis=0)
featurenames = [x[1] for x in featurecounts.index]
featurecountvals = featurecounts.values
plt.bar(featurenames,featurecountvals)
plt.xticks(fontsize=8, rotation=90)
plt.subplots_adjust(bottom=0.4)

#features_less10 = [ (x,y) for x,y in zip(featurenames,featurecountvals) if y<10 ]
#print("Features with <= 10 values:", features_less10)
#plt.bar([x for (x,y) in features_less10],[y for (x,y) in features_less10])
#plt.xticks(fontsize=8, rotation=90)
#plt.subplots_adjust(bottom=0.4)


# Process results of ablation study, threshold study
#outdir = 'out_features'
outdir = 'out 4'
allvals = {}
for f in listdir(outdir):
    if 'out' in f:
        datafile = outdir + '/' + f + '/rulenn/crossval.txt'
        with open(datafile,'r') as infile:
            data = infile.read().split('\n')
            data = [abs(float(x)) for x in data]
            allvals[f] = data

df = pd.DataFrame(allvals)

import re

dfcols = list(df.columns.values)
dfcols.sort(key=lambda col_string :  int(col_string.replace('out_','')))

ax = sns.boxplot(df[dfcols])
ax.tick_params(axis='x', rotation=90)

plt.savefig('feathresholds2.png')


