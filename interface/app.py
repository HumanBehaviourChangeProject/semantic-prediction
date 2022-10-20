import inspect
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
from shiny import App, render, ui

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
from rulenn.apply_rules import print_applied_rules
from rulenn.rule_nn import RuleNNModel
from dataprocessing.fuzzysets import FUZZY_SETS

###  Server state

checkpoint = 'out/rulenn/model_example.json'
path = 'data/hbcp_gen.pkl'
filters = True

model = RuleNNModel.load(checkpoint)
with open(path, "rb") as fin:
    raw_features, raw_labels = pickle.load(fin)
raw_features[np.isnan(raw_features)] = 0

if filters:
    features = filter_features(raw_features)
else:
    features = raw_features

featurenames = [x[1] for x in features.columns]

#for row in model._prepare_single(features.values):
#    fits = model.model.calculate_fit(row.unsqueeze(0))
#    print("The following rules were applied:")
#    print_applied_rules(model.model.non_lin(model.model.conjunctions), model.model.rule_weights, fits, features.columns)
#    print(f"The application of these rules resulted in the following prediction: {model.model.apply_fit(fits).item()}")
#    print("\n---\n")

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

app_ui = ui.page_fluid(
    ui.panel_title(ui.h2("HBCP predictions (prototype): Smoking Cessation")),

    ui.layout_sidebar(ui.panel_sidebar(
        ui.input_checkbox_group('intervention',
                                'Intervention',
                                intervention),
        ui.input_checkbox_group("delivery",
                                "Delivery",
                                delivery
                                ),
        ui.input_checkbox_group("source",
                                "Source",
                                source
                                ),
        ui.panel_conditional("input.intervention.indexOf('11.1 Pharmacological support') > -1",
                             ui.input_radio_buttons("pharmacological",
                                                    "Pharmacological",
                                                    pharmacological,
                                                    selected="placebo"
                                                    )
                             ),
    ),
        ui.panel_main(ui.row(
                             ui.column(4,
                                       ui.input_slider("meanage",
                                                       "Mean age",
                                                       15, # min (mean age)
                                                       80, # max (mean age)
                                                       25  # mean (age)
                                                       ),
                                       ui.input_slider("proportionfemale",
                                                       "Proportion female",
                                                       0,
                                                       100,
                                                       50  # mean (proportion female)
                                                       )
                                       ),
                             ui.column(4,
                                       ui.input_slider("meantobacco",
                                                       "Mean number of times tobacco used",
                                                       1, # min (mean number of times tobacco used)
                                                       30,
                                                       10 # median(df.clean$Mean.number.of.times.tobacco.used)
                                                       ),
                                       ui.input_checkbox("patientrole",
                                                         "Patient role?"
                                                        )
                                       ),
                             ui.column(4,
                                       ui.input_radio_buttons(
                                           "outcome",
                                           "Outcome",
                                           choices=outcome
                                       ),
                                       ui.input_checkbox(
                                           "verification",
                                           "Biochemical verification"
                                       ),
                                       ui.input_slider(
                                           "followup",
                                           "Follow up (weeks)",
                                           4, #min(df.clean$Combined.follow.up),
                                           60,   #as.integer(max(df.clean$Combined.follow.up)),
                                           12     #median(df.clean$Combined.follow.up)
                                       ),
                                       )
                      ),
            ui.row( ui.column(12,ui.output_text("predict") ) )
        )
    ),

)


def server(input, output, session):
    @output
    @render.text
    def predict():
        test = features.iloc[0].values
        # Baseline
        test[0: len(test)] = 0
        fuzzynames = ['Mean age',
                      'Proportion identifying as female gender',
                      'Proportion identifying as male gender',
                      'Mean number of times tobacco used',
                      'Combined follow up']
        fuzzyvalues = [input.meanage(),
                       input.proportionfemale(),
                       100-input.proportionfemale(),
                       input.meantobacco(),
                       input.followup()]
        for fname, fvalue in zip(fuzzynames, fuzzyvalues):
            fs = FUZZY_SETS.get(fname)
            for valname, valfs in list(fs.items()):
                colname = f"{fname} ({valname})"
                test[featurenames.index(colname)] = valfs(fvalue)
        test[featurenames.index('aggregate patient role')] = input.patientrole()
        control = [i for i in test]  # deep copy
        control[featurenames.index('control')] = 1
        # intervention attributes
        
        # run prediction
        testv = model._prepare_single(test)
        testf = model.model.calculate_fit(testv.unsqueeze(0))
        testp = model.model.apply_fit(testf).item()
        ctrlv = model._prepare_single(control)
        ctrlf = model.model.calculate_fit(ctrlv.unsqueeze(0))
        ctrlp = model.model.apply_fit(ctrlf).item()
        return [test,control,testp,ctrlp]


    @output
    @render.text
    def txt():
        return f"n*2 is {input.n() * 2}"

    @output
    @render.text
    def txtfeatures():
        return str(featurenames)


app = App(app_ui, server)
