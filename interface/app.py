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

app_ui = ui.page_fluid(
    ui.panel_title(ui.div(
        ui.img(src="logo.png",style="width: 400px"),
        ui.h2("Smoking Cessation"),
        ui.span(style="flex: 1"),
        ui.a("Contact", href="https://bciovis.hbcptools.org/contact"),
        style="display: flex; align-items: center; margin-right: 14px;")),

    ui.layout_sidebar(ui.sidebar(
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
        ui.row(
            ui.column(4,
                   ui.input_slider("meanage",
                                   "Mean age",
                                   15, # min (mean age)
                                   80, # max (mean age)
                                   50  # default value
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
                       26     #default value (6 months)
                   ),
                   )
            ),
        ui.row( ui.column(12,ui.output_plot("predict") ) )
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
                if colname in featurenames:
                    test[featurenames.index(colname)] = valfs(fvalue)
        test[featurenames.index('aggregate patient role')] = input.patientrole()
        test[featurenames.index('Biochemical verification')] = input.verification()
        if input.outcome() is not None:
            test[featurenames.index(input.outcome())] = True

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
        extendednames = featurenames + ["not " + n for n in featurenames]
        (testrls,testfit) = apply_rules(model,test,extendednames)
        (ctrlrls,ctrlfit) = apply_rules(model,control,extendednames)

        testimpacts = [a[1] for a in testrls]
        ctrlimpacts = [b[1] for b in ctrlrls]
        testonlyimpacts = [a[1] for a in testrls if a not in ctrlrls]
        testnames = [a[0] for a  in testrls]
        ctrlnames = [b[0] for b in ctrlrls]
        testrulestrs = []
        ctrlrulestrs = []

        NO_RULES = 30

        for i,ruleslst in enumerate(ctrlnames):
            ruleslststr = [x for (x,w) in ruleslst] #+"(" +str(round(w,1))+")"
            impact = ctrlimpacts[i]
            rulestr = ' & '.join(ruleslststr)
            rulestr = rulestr + ": " + str(round(impact,1))
            ctrlrulestrs.append(rulestr)

        for i,ruleslst in enumerate(testnames):
            ruleslststr = [x for (x,w) in ruleslst] # + "(" +str(round(w,1))+")"
            impact = testimpacts[i]
            rulestr = ' & '.join(ruleslststr)
            rulestr = rulestr + ": " + str(round(impact,1))
            if rulestr not in ctrlrulestrs:
                testrulestrs.append(rulestr)

        f = plt.figure(figsize=(30,100))

        gs = GridSpec(5, 6, figure=f)
        ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=1, rowspan=2))
        #axarr = f.add_subplot(2, 2, 1)
        plt.barh(list(range(0,-(len(testonlyimpacts)+1),-1)),testonlyimpacts+[0],color=['red' if a < 0 else 'green' for a in testonlyimpacts])
        ax1.set_title('Intervention: '+str(round(testfit,2))+'%')
        ax1.set_xlabel('')
        ax1.set_xlim(-5,5)
        ax1.set_xticks([])
        ax1.set_yticks([])

        #axarr = f.add_subplot(2, 2, 2)
        ax2 = plt.subplot(gs.new_subplotspec((0, 1), colspan=5, rowspan=2))
        #plt.plot()
        ax2.set_title('Intervention: Rules applied')
        ax2.set_ylim(0,NO_RULES/3)
        plt.rc('font', size=7)
        for i, rulestr in enumerate(testrulestrs):
            if i+1 < NO_RULES/3:
                plt.text(0,NO_RULES/3-(i+1)*3,rulestr)
        ax2.set_xlabel('')
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = plt.subplot(gs.new_subplotspec((1, 0), colspan=1,rowspan=3))
        plt.barh(list(range(0,-len(ctrlimpacts),-1)),ctrlimpacts,color=['red' if a < 0 else 'green' for a in ctrlimpacts])
        ax3.set_title('Control: ' + str(round(ctrlfit,2)) + '%')
        ax3.set_xlabel('Rule impact (% cessation)')
        ax3.set_yticks([])
        ax3.set_xlim(-5,5)

        ax4 = plt.subplot(gs.new_subplotspec((1, 1), colspan=5,rowspan=3))
        ax4.set_title('Control: Rules applied')
        ax4.set_ylim(0, NO_RULES)
        plt.rc('font', size=7)
        for i,rulestr in enumerate(ctrlrulestrs):
            if i+1 < NO_RULES:
                plt.text(0, NO_RULES - (i + 1), rulestr)
        ax4.set_xlabel('')
        ax4.set_xticks([])
        ax4.set_yticks([])

        #plt.subplots_adjust(left=0.12, bottom=0.08, right=0.85, top=0.92, wspace=0.01, hspace=0.08)
        #f.tight_layout()
        return f


# Start the application

www_dir = pathlib.Path(__file__).parent
print("Loading images from",www_dir)
app = App(app_ui, server,static_assets=www_dir)
