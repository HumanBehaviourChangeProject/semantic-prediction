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
from rulenn.apply_rules import apply_rules
from dataprocessing.fuzzysets import FUZZY_SETS

###  Which model are we testing
checkpoint = 'examples/model_final.json'
path = 'data/hbcp_gen.pkl'

model = RuleNNModel.load(checkpoint)
with open(path, "rb") as fin:
    raw_features, raw_labels = pickle.load(fin)
raw_features[np.isnan(raw_features)] = 0
features=raw_features

#Additional filter based on the loaded model. Maybe the only one really needed?
retainedfeatures = [x for x in features.columns if x[1] in model.variables]
features = features[retainedfeatures]

featurenames = [x[1] for x in features.columns]
featuresemantics = pd.read_csv('data/feature-semantics.csv')

print("We have ",len(featurenames)," features.")

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


def predict(input):
    test = features.iloc[0].values
    # Baseline
    test[0: len(test)] = 0
    fuzzynames = ['Mean age',
                  'Proportion identifying as female gender',
                  'Proportion identifying as male gender',
                  'Mean number of times tobacco used',
                  'Combined follow up']
    fuzzyvalues = [input['meanage'],
                   input['proportionfemale'],
                   100 - input['proportionfemale'],
                   input['meantobacco'],
                   input['followup']]
    for fname, fvalue in zip(fuzzynames, fuzzyvalues):
        fs = FUZZY_SETS.get(fname)
        for valname, valfs in list(fs.items()):
            colname = f"{fname} ({valname})"
            if colname in featurenames:
                test[featurenames.index(colname)] = valfs(fvalue)
    test[featurenames.index('aggregate patient role')] = input['patientrole']
    test[featurenames.index('Biochemical verification')] = input['verification']
    if input['outcome'] is not None:
        test[featurenames.index(input['outcome'])] = True

    # Shared attributes have been set, copy this to the control
    control = [i for i in test]  # deep copy
    control[featurenames.index('control')] = 1

    # Set intervention-specific attributes
    for x in input['intervention']:
        test[featurenames.index(x)] = True
    for x in input['delivery']:
        test[featurenames.index(x)] = True
    for x in input['source']:
        test[featurenames.index(x)] = True
    if '11.1 Pharmacological support' in input['intervention']:
        if input['pharmacological'] is not None:
            test[featurenames.index(input['pharmacological'])] = True

    # run prediction
    extendednames = featurenames + ["not " + n for n in featurenames]
    (testrls, testfit) = apply_rules(model, test, extendednames)
    (ctrlrls, ctrlfit) = apply_rules(model, control, extendednames)

    testimpacts = [a[1] for a in testrls]
    ctrlimpacts = [b[1] for b in ctrlrls]
    testnames = [a[0] for a in testrls]
    ctrlnames = [b[0] for b in ctrlrls]
    testrulestrs = []
    ctrlrulestrs = []

    NO_RULES = 30

    for i, ruleslst in enumerate(ctrlnames):
        ruleslststr = [x + "(" + str(round(w, 1)) + ")" for (x, w) in ruleslst]
        impact = ctrlimpacts[i]
        rulestr = ' & '.join(ruleslststr)
        rulestr = rulestr + ": " + str(round(impact, 1))
        ctrlrulestrs.append(rulestr)

    for i, ruleslst in enumerate(testnames):
        ruleslststr = [x + "(" + str(round(w, 1)) + ")" for (x, w) in ruleslst]
        impact = testimpacts[i]
        rulestr = ' & '.join(ruleslststr)
        rulestr = rulestr + ": " + str(round(impact, 1))
        if rulestr not in ctrlrulestrs:
            testrulestrs.append(rulestr)

    return(testfit,testimpacts,testrulestrs,ctrlfit, ctrlimpacts,ctrlrulestrs)


# Set up test case
def createBaseInput():
    input = {}
    input['pharmacological'] = 'placebo'
    input['intervention'] = []
    input['meanage'] = 40
    input['proportionfemale'] = 60
    input['patientrole'] = False
    input['meantobacco'] = 5
    input['verification'] = True
    input['followup']=6
    input['outcome']='Abstinence: Continuous '
    input['delivery']=[]
    input['source']=[]

    return input


results={"intervention":[],"test":[],"control":[],"testrules":[],"ctrlrules":[]}
for interven in intervention:
    input = createBaseInput()
    input['intervention'].append(interven)
    print(input)
    (testfit, testimpacts, testrulestrs, ctrlfit, ctrlimpacts, ctrlrulestrs) = predict(input)
    results['intervention'].append(interven)
    results['test'].append(testfit)
    results['control'].append(ctrlfit)
    results['testrules'].append(len(testrulestrs))
    results['ctrlrules'].append(len(ctrlrulestrs))

interven_res = pd.DataFrame(results)
ax = sns.scatterplot(data=interven_res,x='intervention',y='test')
sns.scatterplot(data=interven_res,x='intervention',y='control')
ax.tick_params(axis='x', rotation=90)
plt.subplots_adjust(bottom=0.4)
ax.legend(labels=['test','ctrl'],loc='best')

results={"source":[],"test":[],"control":[],"testrules":[],"ctrlrules":[]}
for sour in source:
    input = createBaseInput()
    input['intervention'].append('brief advise')
    input['source'].append(sour)
    print(input)
    (testfit, testimpacts, testrulestrs, ctrlfit, ctrlimpacts, ctrlrulestrs) = predict(input)
    results['source'].append(sour)
    results['test'].append(testfit)
    results['control'].append(ctrlfit)
    results['testrules'].append(len(testrulestrs))
    results['ctrlrules'].append(len(ctrlrulestrs))

source_res = pd.DataFrame(results)
ax = sns.scatterplot(data=source_res,x='source',y='test')
sns.scatterplot(data=source_res,x='source',y='control')
ax.tick_params(axis='x', rotation=90)
plt.subplots_adjust(bottom=0.4)
ax.legend(labels=['test','ctrl'],loc='best')

ax2 = sns.scatterplot(data=source_res,x='source',y='ctrlrules')
sns.scatterplot(data=source_res,x='source',y='testrules')
ax.tick_params(axis='x', rotation=90)
plt.subplots_adjust(bottom=0.4)
ax.legend(labels=['test','ctrl'],loc='best')

results={"delivery":[],"test":[],"control":[],"testrules":[],"ctrlrules":[]}
for deliv in delivery:
    input = createBaseInput()
    input['intervention'].append('brief advise')
    input['delivery'].append(deliv)
    print(input)
    (testfit, testimpacts, testrulestrs, ctrlfit, ctrlimpacts, ctrlrulestrs) = predict(input)
    results['delivery'].append(deliv)
    results['test'].append(testfit)
    results['control'].append(ctrlfit)
    results['testrules'].append(len(testrulestrs))
    results['ctrlrules'].append(len(ctrlrulestrs))

deliv_res = pd.DataFrame(results)
ax = sns.scatterplot(data=deliv_res,x='delivery',y='test')
sns.scatterplot(data=deliv_res,x='delivery',y='control')
ax.tick_params(axis='x', rotation=90)
plt.subplots_adjust(bottom=0.4)
ax.legend(labels=['test','ctrl'],loc='best')
