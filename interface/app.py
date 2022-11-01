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
from rulenn.apply_rules import apply_rules
from dataprocessing.fuzzysets import FUZZY_SETS

###  Server state

checkpoint = 'out/rulenn/examples/model_example.json'
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
            ui.row( ui.column(12,ui.output_plot("predict") ) )
        )
    ),

)


def server(input, output, session):
    @output
    @render.plot
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
        test[featurenames.index('Biochemical verification')] = input.verification()

        # Shared attributes have been set, copy this to the control
        control = [i for i in test]  # deep copy
        control[featurenames.index('control')] = 1

        # Set intervention-specific attributes
        for x in input.intervention():
            test[featurenames.index(x)] = True
        for x in input.delivery():
            test[featurenames.index(x)] = True
        for x in input.source():
            test[featurenames.index(x)] = True
        if '11.1 Pharmacological support' in input.intervention():
            if input.pharmacological() is not None:
                test[featurenames.index(input.pharmacological())] = True

        # run prediction
        (testrls,testfit) = apply_rules(model,test,featurenames)
        (ctrlrls,ctrlfit) = apply_rules(model,control,featurenames)

        testimpacts = [a[1] for a in testrls]
        ctrlimpacts = [b[1] for b in ctrlrls]
        testnames = [a[0] for a  in testrls]
        ctrlnames = [b[0] for b in ctrlrls]

        f = plt.figure()

        gs = GridSpec(2, 6, figure=f)
        ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=1))
        #axarr = f.add_subplot(2, 2, 1)
        plt.barh(list(range(0,-len(testimpacts),-1)),testimpacts,color=['red' if a < 0 else 'green' for a in testimpacts])
        plt.title('Intervention: '+str(round(testfit,2))+'%')
        plt.xlabel('Rule impact (% cessation)')
        plt.yticks([])

        NO_RULES = 15

        #axarr = f.add_subplot(2, 2, 2)
        ax2 = plt.subplot(gs.new_subplotspec((0, 1), colspan=5))
        #plt.plot()
        plt.title('Intervention: Rules applied')
        plt.ylim(0,NO_RULES)
        plt.rc('font', size=6)
        for i, ruleslst in enumerate(testnames):
            if i+1 < NO_RULES:
                ruleslststr = [x+ "(" +str(round(w,1))+")" for (x,w) in ruleslst]
                impact = testimpacts[i]
                rulestr = ' & '.join(ruleslststr)
                rulestr = rulestr + ": " + str(round(impact,1))
                plt.text(0,NO_RULES-(i+1),rulestr)
        plt.xlabel('')
        plt.xticks([])
        plt.yticks([])

        #axarr = f.add_subplot(2, 2, 3)
        ax3 = plt.subplot(gs.new_subplotspec((1, 0), colspan=1))
        plt.barh(list(range(0,-len(ctrlimpacts),-1)),ctrlimpacts,color=['red' if a < 0 else 'green' for a in ctrlimpacts])
        plt.title('Control (' + str(ctrlfit) + ')')
        plt.xlabel('Rule impact (% cessation)')
        plt.yticks([])

        #axarr = f.add_subplot(2, 2, 4)
        ax4 = plt.subplot(gs.new_subplotspec((1, 1), colspan=5))
        plt.title('Control: Rules applied')
        plt.ylim(0, NO_RULES)
        plt.rc('font', size=6)
        for i, ruleslst in enumerate(ctrlnames):
            if i + 1 < NO_RULES:
                ruleslststr = [x + "(" + str(round(w, 1)) + ")" for (x, w) in ruleslst]
                impact = testimpacts[i]
                rulestr = ' & '.join(ruleslststr)
                rulestr = rulestr + ": " + str(round(impact, 1))
                plt.text(0, NO_RULES - (i + 1), rulestr)
        plt.xlabel('')
        plt.xticks([])
        plt.yticks([])

        f.tight_layout()
        return f


# Start the application
app = App(app_ui, server)
