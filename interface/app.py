from shiny import App, render, ui

import pickle
import pandas as pd
import numpy as np
import torch

import sys
import pathlib
import inspect
import statistics
import math

import itertools

src_file_path = inspect.getfile(lambda: None)

PACKAGE_PARENT = pathlib.Path(src_file_path).parent
PACKAGE_PARENT2 = PACKAGE_PARENT.parent
SCRIPT_DIR = PACKAGE_PARENT2 / "rulenn"
sys.path.append(str(PACKAGE_PARENT2))
sys.path.append(str(SCRIPT_DIR))

from rule_nn import RuleNet
from base import filter_features
from apply_rules import print_applied_rules, get_feature_row_str

with open('data/hbcp_gen.pkl', "rb") as fin:
    features, labels = pickle.load(fin)
    features = filter_features(features)

index = np.array(features.index)
featurenames = [x[1] for x in features.columns]

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
                      )
        )
    ),

)


def server(input, output, session):

    def preparePrediction():
        return ''


    @output
    @render.text
    def txt():
        return f"n*2 is {input.n() * 2}"

    @output
    @render.text
    def txtfeatures():
        return str(featurenames)


app = App(app_ui, server)
