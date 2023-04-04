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

checkpoint = 'examples/model_unweighted.json'
path = 'data/hbcp_gen.pkl'
filters = False

model = RuleNNModel.load(checkpoint)
with open(path, "rb") as fin:
    raw_features, raw_labels = pickle.load(fin)
raw_features[np.isnan(raw_features)] = 0

model.print_rules()


### Plot some features of the dataset for the paper

column_names = [x[1] for x in raw_features.columns]

face_to_face = raw_features.iloc[:,[column_names.index("Face to face")]]
df = pd.DataFrame("facetoface"=face_to_face.values)
df['outcome'] = raw_labels
df.boxplot(by=0)




# Process results of ablation study, threshold study
outdir = 'out_features3'
#outdir = 'out 6'
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

plt.savefig('features3.png')


# Processing the features
# In the raw feature table we have 153 columns

featurecounts = raw_features.astype(bool).sum(axis=0)
featurenames = [x[1] for x in featurecounts.index]
featurecountvals = featurecounts.values

#features_less10 = [ (x,y) for x,y in zip(featurenames,featurecountvals) if y<10 ]
#print("Features with <= 10 values:", features_less10)
#plt.bar([x for (x,y) in features_less10],[y for (x,y) in features_less10])
#plt.xticks(fontsize=8, rotation=90)
#plt.subplots_adjust(bottom=0.4)

# How many features have >X values?

featurenumbercounts = { str(x): len(featurecountvals[featurecountvals>x]) for x in range(0,121,10)}
sns.scatterplot(x=list(featurenumbercounts.keys()),y=list(featurenumbercounts.values()))

feature_info = pd.read_csv("data/feature_filter.csv")

def getNumUsed(row):
    count = 0
    name = row['Feature name']
    if name in featurenames:
        count = featurecountvals[featurenames.index(name)]
    return(count)


allvals = {}
for featurerun in ['2','3']:
    outdir = 'out_features'+featurerun
    meanvals = {}
    for f in listdir(outdir):
        if 'out' in f:
            datafile = outdir + '/' + f + '/rulenn/crossval.txt'
            with open(datafile,'r') as infile:
                data = infile.read().split('\n')
                data = [abs(float(x)) for x in data]
                meanvals[f] = statistics.mean(data)
    allvals[featurerun] = meanvals

def getMeanPredWithout(row,ind_run):
    mean_pred = 0
    name = row['Feature name']
    if name in featurenames:
        index_feature_name = 'out_'+str(featurenames.index(name))
        if index_feature_name in allvals[ind_run].keys():
            mean_pred = allvals[ind_run][index_feature_name]
    return (mean_pred)

feature_info['Count used'] = feature_info.apply(lambda row: getNumUsed(row), axis=1)
feature_info['Mean pred without 1'] = feature_info.apply(lambda row: getMeanPredWithout(row,'2'), axis=1)
feature_info['Mean pred without 2'] = feature_info.apply(lambda row: getMeanPredWithout(row,'3'), axis=1)

feature_info.to_csv("data/all_feature_info.csv")


# Start from a threshold of 30. So only features with featurecount > 30 are included.
# Then,
from itertools import compress

basefeaturecountvals = featurecountvals[featurecountvals > 30]
basefeaturenames = list(compress(featurenames,featurecountvals > 30))

# Process results of ablation study, threshold study
outdir = 'out 8'
allvals = {}
for f in listdir(outdir):
    if 'out' in f:
        datafile = outdir + '/' + f + '/rulenn/crossval.txt'
        with open(datafile,'r') as infile:
            data = infile.read().split('\n')
            data = [abs(float(x)) for x in data]
            allvals[f] = data

df = pd.DataFrame(allvals)

dfcols = list(df.columns.values)
dfcols.sort(key=lambda col_string :  int(col_string.replace('out_','')))

ax = sns.boxplot(df[dfcols])
ax.tick_params(axis='x', rotation=90)

plt.savefig('combined2.png')

basefeatures = pd.DataFrame({"Feature name":basefeaturenames,
                             "Count":basefeaturecountvals})

basefeatures3050 = basefeatures[(basefeatures['Count']>30) & (basefeatures['Count']<50)]

ax = sns.barplot(data=basefeatures3050,x='Feature name',y='Count',color='grey')
ax.tick_params(axis='x', rotation=90)
plt.subplots_adjust(bottom=0.4)

