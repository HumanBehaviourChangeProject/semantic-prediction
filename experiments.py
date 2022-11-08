import inspect
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
from shiny import App, render, ui

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

features_less10 = [ x for i,x in enumerate(featurenames) if featurecountvals[i]<10 ]

print("Features with <= 10 values:")