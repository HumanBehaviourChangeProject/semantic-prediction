import inspect
import pathlib
import pickle
import sys
import os

import numpy as np
import pandas as pd
from shiny import App, render, ui

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

### Handle local imports


### Handle local imports
#os.chdir('/var/www/html/semantic-prediction')
#sys.path.append('/var/www/html/semantic-prediction')


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

###  Server state

checkpoint = 'examples/model_final.json'
path = 'data/hbcp_gen.pkl'
filters = False

model = RuleNNModel.load(checkpoint)
model.model.eval()  # Run in production mode

with open(path, "rb") as fin:
    raw_features, raw_labels = pd.read_pickle(fin)
raw_features[np.isnan(raw_features)] = 0

if filters:
    features = filter_features(raw_features)
else:
    features = raw_features

#Additional filter based on the loaded model. Maybe the only one really needed?
retainedfeatures = [x for x in features.columns if x[1] in model.variables]
features = features[retainedfeatures]

featurenames = [x[1] for x in features.columns]
featuresemantics = pd.read_csv('data/feature-semantics.csv')


# basically just reformats JSON to class format needed to conform to rest of code
class InputClass:

    def __init__(self, values):
        self.values = values


    def meanage(self):
        return self.values["meanage"]

    def proportionfemale(self):
        return self.values["proportionfemale"]

    def meantobacco(self):
        return self.values["meantobacco"]

    def followup(self):
        return self.values["followup"]

    def patientrole(self):
        return self.values["patientrole"]

    def verification(self):
        return self.values["verification"]

    def outcome(self):
        return self.values["outcome"]

    def intervention(self):
        if "intervention" not in self.values:
            return []
        return self.values["intervention"]

    def delivery(self):
        if "delivery" not in self.values:
            return []
        return self.values["delivery"]

    def source(self):
        if "source" not in self.values:
            return []
        return self.values["source"]

    def pharmacological(self):
        if "pharmacological" not in self.values:
            return "-"
        return self.values["pharmacological"]

# if two rules are the same, just the number is different, we want to remove one of them
# eg ["A (<= 12 months), A (<= 15 months)"] -> ["A (<= 12 months)"]
def cleanup_rule(rule: list):

    if len(rule) == 1:
        return rule

    terms_separated = []
    for term_tuple in rule[0]:
        term = term_tuple[0]
        elements = term.split(" (")
        feature = elements[0]
        unit = ""
        if len(elements) > 1:
            elements2 = elements[1].split(" ")
            if len(elements2) > 1:
                comparator = elements2[0]
                value = elements2[1].split(")")[0]

                if len(elements2) > 2:
                    unit = elements2[2].split(")")[0]

            else:
                comparator = ""
                value = elements[1].split(")")[0]

        else:
            value = ""
            comparator = ""

        terms_separated.append({"feature": feature, "comparator": comparator, "value": value, "unit": unit})

    # join terms with same feature
    cleaned_terms = []
    cleaned_features = []
    for term in terms_separated:
        # when this is the first time the feature is seen, add it
        if term["feature"] not in cleaned_features:
            cleaned_terms.append(term)
            cleaned_features.append(term["feature"])
        # if the feature is seen before, join them
        else:
            # find the term in the cleaned_terms
            feature_index = cleaned_features.index(term["feature"])
            prev_term = cleaned_terms[feature_index]

            # if both are <=, take the smaller one
            if term["comparator"] == "<=" and prev_term["comparator"] == "<=":
                if int(term["value"]) < int(prev_term["value"]):
                    cleaned_terms[feature_index]["value"] = term["value"]

            # if both are >=, take the larger one
            elif term["comparator"] == ">=" and prev_term["comparator"] == ">=":
                if int(term["value"]) > int(prev_term["value"]):
                    cleaned_terms[feature_index]["value"] = term["value"]

            # if one is <= and the other is >=, create new comparator -
            elif term["comparator"] == "<=" and prev_term["comparator"] == ">=":
                cleaned_terms[feature_index]["comparator"] = "-"
                cleaned_terms[feature_index]["value_prev"] = prev_term["comparator"]
                cleaned_terms[feature_index]["value"] = term["value"]
            elif term["comparator"] == ">=" and prev_term["comparator"] == "<=":
                cleaned_terms[feature_index]["comparator"] = "-"
                cleaned_terms[feature_index]["value_prev"] = term["comparator"]
                cleaned_terms[feature_index]["value"] = prev_term["value"]

            # if one is <= and the other is -, restrict range
            elif term["comparator"] == "<=" and prev_term["comparator"] == "-":
                if int(term["value"]) < int(prev_term["value"]):
                    cleaned_terms[feature_index]["value"] = term["value"]
                else:
                    cleaned_terms[feature_index]["value"] = prev_term["value"]
            elif term["comparator"] == "-" and prev_term["comparator"] == "<=":
                if int(term["value"]) > int(prev_term["value"]):
                    cleaned_terms[feature_index]["value"] = term["value"]
                else:
                    cleaned_terms[feature_index]["value"] = prev_term["value"]

            # if one is >= and the other is -, restrict range
            elif term["comparator"] == ">=" and prev_term["comparator"] == "-":
                if int(term["value"]) > int(prev_term["value"]):
                    cleaned_terms[feature_index]["value"] = term["value"]
                else:
                    cleaned_terms[feature_index]["value"] = prev_term["value"]
            elif term["comparator"] == "-" and prev_term["comparator"] == ">=":
                if int(term["value"]) < int(prev_term["value"]):
                    cleaned_terms[feature_index]["value"] = term["value"]
                else:
                    cleaned_terms[feature_index]["value"] = prev_term["value"]

    # create the cleaned rules
    cleaned_rule = []
    for term in cleaned_terms:
        if term["comparator"] == "":
            if term["value"] == "":
                cleaned_rule.append(term["feature"] + " " + term["unit"])
            else:
                cleaned_rule.append(term["feature"] + " (" + term["value"] + " " +term["unit"] + ")")
        elif term["comparator"] == "-":
            cleaned_rule.append(term["feature"] + " (" + term["value_prev"] + term["comparator"] + term["value"] + " " + term["unit"] + ")")
        else:
            cleaned_rule.append(term["feature"] + " (" + term["comparator"] + " " + term["value"] + " " + term["unit"] + ")")



    return [cleaned_rule, rule[1]]  # return the cleaned rule and the impact


def cleanup_rules(rules):
    # clean up individual rules
    cleaned_rules = list(map(cleanup_rule, rules))

    # join duplicate rules
    unique_rules = []
    unique_rule_strs = []
    for rule in cleaned_rules:
        rule_str = str(rule[0])
        if rule_str not in unique_rule_strs:
            unique_rules.append(rule)
            unique_rule_strs.append(rule_str)
        else:
            # if the rule is already in the list, add the values together
            index = unique_rule_strs.index(rule_str)
            unique_rules[index][1] += rule[1]

    return unique_rules


# predicts for the given input in InputClass format
def predict(input: InputClass):

    refined_input = features.iloc[0].values
    # Baseline
    refined_input[0: len(refined_input)] = 0
    fuzzynames = ['Mean age',
                  'Proportion identifying as female gender',
                  'Proportion identifying as male gender',
                  'Mean number of times tobacco used',
                  'Combined follow up']
    fuzzyvalues = [input.meanage(),
                   input.proportionfemale(),
                   100 -input.proportionfemale(),
                   input.meantobacco(),
                   input.followup()]
    for fname, fvalue in zip(fuzzynames, fuzzyvalues):
        fs = FUZZY_SETS.get(fname)
        for valname, valfs in list(fs.items()):
            colname = f"{fname} ({valname})"
            if colname in featurenames:
                refined_input[featurenames.index(colname)] = valfs(fvalue)
    refined_input[featurenames.index('aggregate patient role')] = input.patientrole()
    refined_input[featurenames.index('Biochemical verification')] = input.verification()
    if input.outcome() is not None:
        refined_input[featurenames.index(input.outcome())] = True



    has_interventions = False
    # Set intervention-specific attributes
    for x in input.intervention():
        refined_input[featurenames.index(x)] = True
        has_interventions = True
    for x in input.delivery():
        refined_input[featurenames.index(x)] = True
        has_interventions = True
    for x in input.source():
        refined_input[featurenames.index(x)] = True
        has_interventions = True
    if input.pharmacological() != "-":
        refined_input[featurenames.index(input.pharmacological())] = True
        refined_input[featurenames.index('11.1 Pharmacological support')] = True
        has_interventions = True

    # If no interventions are set, add the control group features
    if not has_interventions:
        # Shared attributes have been set, copy this to the control
        refined_input = [i for i in refined_input]  # deep copy
        refined_input[featurenames.index('control')] = 1

    # run prediction
    extendednames = featurenames + ["not " + n for n in featurenames]
    (rules ,fit) = apply_rules(model ,refined_input ,extendednames)

    # clean up rules
    rules = cleanup_rules(rules)

    return {"fit": fit, "rules": rules}

example_input = {
    "meanage": 20,
    "proportionfemale": 50,
    "meantobacco": 10,
    "followup": 26,
    "patientrole": 1,
    "verification": 1,
    "outcome": "Abstinence: Continuous ",
    "intervention": [],
    "pharmacological": "-",
    "delivery": [],
    "source": []
}

example_input_instance = InputClass(example_input)

result = predict(example_input_instance)

print(result)
