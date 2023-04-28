import seaborn
from seaborn import boxplot, kdeplot, barplot
import pandas as pd
from matplotlib import pyplot as plt
from math import fabs
buckets = []

folder = "results/runs/5-final"

with open(f"{folder}/random_forest/crossval.txt", "r") as fin:
    buckets.append(("random forest", [float(l) for l in fin]))

with open(f"{folder}/rulenn/crossval.txt", "r") as fin:
    buckets.append(("rule", [float(l) for l in fin]))

with open(f"{folder}/mle/crossval.txt", "r") as fin:
    buckets.append(("mixed-effect", [-float(l) for l in fin]))

with open(f"{folder}/deep/crossval.txt", "r") as fin:
    buckets.append(("deep", [float(l) for l in fin]))

df = pd.DataFrame([{"model":n,"error":v} for n, l in buckets for v in l])
df['abserror'] = df['error'].abs()

#boxplot(x=[name for name, l in buckets for _ in l],y=[abs(x) for _, l in buckets for x in l], showfliers=False)


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

from base import filter_features
from rulenn.rule_nn import RuleNNModel

checkpoint = 'examples/model_unweighted.json'
path = 'data/hbcp_gen.pkl'
filters = False
with open(path, "rb") as fin:
    raw_features, raw_labels = pickle.load(fin)
raw_features[np.isnan(raw_features)] = 0

model = RuleNNModel.load(checkpoint)


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

buckets.append(("grand mean", [float(l) for l in errorscrossvalmean]))

df = pd.DataFrame([{"model":n,"error":v} for n, l in buckets for v in l])
df['abserror'] = df['error'].abs()

boxplot(data=df,x="model",y="abserror")
plt.savefig("box.png")
plt.clf()
plt.close()

kdeplot(data=df, x="abserror", hue="model")
plt.savefig("kde.png")
plt.clf()
plt.close()


## Features treatment

feature_df = pd.read_csv("data/model_input_data.csv")
cleaned_feature_df = pd.read_csv("data/hbcp_gen.csv")

cleaned_feature_df.shape
cleaned_feature_df.head()

## How many study arms are there for specific features?
sums = cleaned_feature_df.sum()[5:98]
len(sums)
featurenames = [x[1] for x in raw_features.columns]
print("We have ",len(featurenames)," features.")

for (i, item) in enumerate(featurenames):
    print(i, item)

# BCTs 0-23; 51
# Abstinence 24-25
# Biochemical verification 26
# Follow-up 27-31
# Delivery type and pharmacological 33-36; 52; 55-56; 66-67; 70; 73; 77-82
# Individual-level 37-41
# Mean age 42-45
# Mean no. of times tobacco used 46-50
# Pharma competing interest 53-54
# Proportion female + male 57-65
# Patient role 68
# Setting 74
# BCT groups 83-92

featuresemantics = pd.read_csv('data/feature-semantics.csv')

intervention = featuresemantics.query('group == "intervention"')['featurename'].values.tolist()
intervention = [x for x in intervention if x in featurenames]
delivery = featuresemantics.query('group == "deliverymode"')['featurename'].values.tolist()
delivery = [x for x in delivery if x in featurenames]
source = featuresemantics.query('group == "deliverysource"')['featurename'].values.tolist()
source = [x for x in source if x in featurenames]
pharmacological = featuresemantics.query('group == "pharmacological"')['featurename'].values.tolist()
pharmacological = [x for x in pharmacological if x in featurenames]
outcome = featuresemantics.query('group == "outcome"')['featurename'].values.tolist()
outcome = [x for x in outcome if x in featurenames]



fig, axs = plt.subplot_mosaic("A;B", figsize=(30,10))

for n, (key, ax) in enumerate(axs.items()):
    ax.imshow(np.random.randn(10,10), interpolation='none')
    ax.text(-0.1, 1.1, key, transform=ax.transAxes,
            size=20, weight='bold')

barpl = barplot(x=intervention, y=sums[ [featurenames.index(a) for a in intervention] ])
loc, labels = plt.xticks()
barpl.set_xticklabels(labels, rotation=90)
plt.xticks(fontsize=6)
plt.tight_layout()

barpl = barplot(x=source, y=sums[ [featurenames.index(a) for a in source] ])
loc, labels = plt.xticks()
barpl.set_xticklabels(labels, rotation=90)
plt.xticks(fontsize=14)
plt.tight_layout()

model.print_rules()





model.model.eval()  # Run in production mode

with open(path, "rb") as fin:
    raw_features, raw_labels = pickle.load(fin)
raw_features[np.isnan(raw_features)] = 0

if filters:
    features = filter_features(raw_features)
else:
    features = raw_features

#Additional filter based on the loaded model. Maybe the only one really needed?
retainedfeatures = [x for x in features.columns if x[1] in model.variables]
features = features[retainedfeatures]

featurenames = [x[1] for x in features.columns]

from rulenn.apply_rules import apply_rules
from dataprocessing.fuzzysets import FUZZY_SETS

print("We have ",len(featurenames)," features.")
test = features.iloc[0].values
# Baseline
test[0: len(test)] = 0
fuzzynames = ['Mean age',
              'Proportion identifying as female gender',
              'Proportion identifying as male gender',
              'Mean number of times tobacco used',
              'Combined follow up']
fuzzyvalues = [40, # mean age
               50, # proportion female
               50, # proportion male
               10,  #input.meantobacco(),
               26 #input.followup()
               ]
for fname, fvalue in zip(fuzzynames, fuzzyvalues):
    fs = FUZZY_SETS.get(fname)
    for valname, valfs in list(fs.items()):
        colname = f"{fname} ({valname})"
        if colname in featurenames:
            test[featurenames.index(colname)] = valfs(fvalue)
test[featurenames.index('aggregate patient role')] = 0 #input.patientrole()
test[featurenames.index('Biochemical verification')] = 1 #input.verification()
#if input.outcome() is not None:
#    test[featurenames.index(input.outcome())] = True

# Shared attributes have been set, copy this to the control
control = [i for i in test]  # deep copy
control[featurenames.index('control')] = 1

# Set intervention-specific attributes
#for x in input.intervention():
#    test[featurenames.index(x)] = True
#for x in input.delivery():
#    test[featurenames.index(x)] = True
#for x in input.source():
#    test[featurenames.index(x)] = True
#if '11.1 Pharmacological support' in input.intervention():
#    if input.pharmacological() is not None:
#        test[featurenames.index(input.pharmacological())] = True

# run prediction
extendednames = featurenames + ["not " + n for n in featurenames]
(testrls, testfit) = apply_rules(model, test, extendednames)
(ctrlrls, ctrlfit) = apply_rules(model, control, extendednames)

allinterventions = {}

import random

for i in range(1,21):
    resultsforinterventions = []
    for j in range(1,10):
        selectindexes = random.choices(range(0,len(intervention)), k=i)
        intervtest = [i for i in test]
        for intervindex in selectindexes:
            intervtest[intervindex]=True
            interv = featurenames[intervindex]
            if interv == '11.1 Pharmacological support':
                intervtest[featurenames.index('nrt')]=True
                intervtest[featurenames.index('Pill')] = True
                intervtest[featurenames.index('bupropion')] = True
        (testrls, testfit) = apply_rules(model, intervtest, extendednames)
        resultsforinterventions.append(testfit)
    allinterventions[i] = resultsforinterventions

from seaborn import stripplot
import seaborn as sns
#sns.set_palette("Greys")

chart = stripplot(allinterventions)
chart.set_xlabel("Number of intervention content features")
chart.set_ylabel("Predicted percentage cessation")

# Mean population age

intervtest = [i for i in test]
resultsforages = {}
for i in range(15,70,5):
    fuzzyvalues = [i,  # mean age
                   50,  # proportion female
                   50,  # proportion male
                   10,  # input.meantobacco(),
                   26  # input.followup()
                   ]
    for fname, fvalue in zip(fuzzynames, fuzzyvalues):
        fs = FUZZY_SETS.get(fname)
        for valname, valfs in list(fs.items()):
            colname = f"{fname} ({valname})"
            if colname in featurenames:
                intervtest[featurenames.index(colname)] = valfs(fvalue)
    (testrls, testfit) = apply_rules(model, intervtest, extendednames)
    resultsforages[i]=testfit

from seaborn import scatterplot

chart = scatterplot(resultsforages,color='b')
chart.set_xlabel("Mean population age")
chart.set_ylabel("Predicted percentage cessation")



### Mean number of times tobacco used

intervtest = [i for i in test]
resultsfortob = {}
for i in range(5,20,2):
    fuzzyvalues = [40,  # mean age
                   50,  # proportion female
                   50,  # proportion male
                   i,  # input.meantobacco(),
                   26  # input.followup()
                   ]
    for fname, fvalue in zip(fuzzynames, fuzzyvalues):
        fs = FUZZY_SETS.get(fname)
        for valname, valfs in list(fs.items()):
            colname = f"{fname} ({valname})"
            if colname in featurenames:
                intervtest[featurenames.index(colname)] = valfs(fvalue)
    (testrls, testfit) = apply_rules(model, intervtest, extendednames)
    resultsfortob[i]=testfit



chart = scatterplot(resultsfortob,color='b')
chart.set_xlabel("Mean number of times tobacco used daily")
chart.set_ylabel("Predicted percentage cessation")


### Percentage male/female

intervtest = [i for i in test]
resultsforgend = {}
for i in range(20,80,5):
    fuzzyvalues = [40,  # mean age
                   i,  # proportion female
                   (100-i),  # proportion male
                   10,  # input.meantobacco(),
                   26  # input.followup()
                   ]
    for fname, fvalue in zip(fuzzynames, fuzzyvalues):
        fs = FUZZY_SETS.get(fname)
        for valname, valfs in list(fs.items()):
            colname = f"{fname} ({valname})"
            if colname in featurenames:
                intervtest[featurenames.index(colname)] = valfs(fvalue)
    (testrls, testfit) = apply_rules(model, intervtest, extendednames)
    resultsforgend[i]=testfit



chart = scatterplot(resultsforgend,color='b')
chart.set_xlabel("Percentage population female")
chart.set_ylabel("Predicted percentage cessation")

# Source



resultsforsource = {}
for srctype in source:
    intervtest = [i for i in test]
    intervtest[featurenames.index(srctype)] = 1
    (testrls, testfit) = apply_rules(model, intervtest, extendednames)
    resultsforsource[srctype] = testfit


chart = scatterplot(resultsforsource,color='b')
chart.set_xlabel("Source")
chart.set_ylabel("Predicted percentage cessation")



resultsfordeliv = {}
for delivtype in delivery:
    intervtest = [i for i in test]
    intervtest[featurenames.index(delivtype)] = 1
    (testrls, testfit) = apply_rules(model, intervtest, extendednames)
    resultsfordeliv[delivtype] = testfit


chart = scatterplot(resultsfordeliv,color='b')
chart.set_xlabel("Delivery")
chart.set_xticklabels(
    chart.get_xticklabels(),
    rotation=90,
    horizontalalignment='right'
)
chart.set_ylabel("Predicted percentage cessation")
plt.tight_layout()


## Pharmacological

resultsforpharma = {}
for pharmatype in pharmacological:
    intervtest = [i for i in test]
    intervtest[featurenames.index('11.1 Pharmacological support')]=1
    intervtest[featurenames.index(pharmatype)] = 1
    (testrls, testfit) = apply_rules(model, intervtest, extendednames)
    resultsforpharma[pharmatype] = testfit


chart = scatterplot(resultsforpharma,color='b')
chart.set_xlabel("Pharmacological Intervention")
chart.set_xticklabels(
    chart.get_xticklabels(),
    rotation=90,
    horizontalalignment='right'
)
chart.set_ylabel("Predicted percentage cessation")
plt.tight_layout()


