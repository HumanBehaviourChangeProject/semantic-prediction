from flask import Flask, request
from flask_cors import CORS
import numpy as np
import pandas as pd

import inspect
import pathlib
import pickle
import sys
import os

src_file_path = inspect.getfile(lambda: None)

PACKAGE_PARENT = pathlib.Path(src_file_path).parent
PACKAGE_PARENT2 = PACKAGE_PARENT.parent
#SCRIPT_DIR = PACKAGE_PARENT2 / "rulenn"
#DATAPR_DIR = PACKAGE_PARENT2 / "dataprocessing"
sys.path.append(str(PACKAGE_PARENT2))
#sys.path.append(str(SCRIPT_DIR))
#sys.path.append(str(DATAPR_DIR))

from rulenn.rule_nn import RuleNNModel

from predict import predict, InputClass

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=['POST'])
def predict_request():
    data = request.get_json()
    input = InputClass(data["input"])
    prediction = predict(input)

    # restrict range of prediction to 0-100
    prediction["fit"] = np.clip(prediction["fit"], 0, 100)

    return prediction

@app.route("/get_input_params")
def get_input_params():

    checkpoint = 'examples/model_final.json'
    path = 'data/hbcp_gen.pkl'

    model = RuleNNModel.load(checkpoint)
    model.model.eval()  # Run in production mode

    with open(path, "rb") as fin:
        raw_features, raw_labels = pd.read_pickle(fin)
    raw_features[np.isnan(raw_features)] = 0

    features = raw_features

    # Additional filter based on the loaded model. Maybe the only one really needed?
    retainedfeatures = [x for x in features.columns if x[1] in model.variables]
    features = features[retainedfeatures]

    featurenames = [x[1] for x in features.columns]
    featuresemantics = pd.read_csv('data/feature-semantics-labeled.csv')


    # remove "somatic mode of delivery" aka "Somatic"
    featurenames = [x for x in featurenames if x != "Somatic"]


    print("We have ", len(featurenames), " features.")

    intervention = featuresemantics.query('group == "intervention"')['featurename'].values.tolist()
    intervention = [x for x in intervention if x in featurenames and x != "11.1 Pharmacological support"]
    delivery = featuresemantics.query('group == "deliverymode"')['featurename'].values.tolist()
    delivery = [x for x in delivery if x in featurenames]
    source = featuresemantics.query('group == "deliverysource"')['featurename'].values.tolist()
    source = [x for x in source if x in featurenames]
    pharmacological = featuresemantics.query('group == "pharmacological"')['featurename'].values.tolist()
    pharmacological = [x for x in pharmacological if x in featurenames] + ["-"]
    outcome = featuresemantics.query('group == "outcome"')['featurename'].values.tolist()
    outcome = [x for x in outcome if x in featurenames]


    return {"params": [
        {"id": "meanage", "label": "Mean age of smokers", "type": "slider", "min": 15, "max": 80, "value": 50, "step": 1},
        {"id": "proportionfemale", "label": "Percentage who are female", "type": "slider", "min": 0, "max": 100, "value": 50, "step": 1},
        {"id": "meantobacco", "label": "Mean number of cigarettes smoked", "type": "slider", "min": 1, "max": 30, "value": 10, "step": 1},
        {"id":"followup", "label": "Follow-up point in weeks", "type": "slider", "min": 4, "max": 60, "value": 26, "step": 1},
        {"id": "patientrole", "label": "Smokers are patients", "type": "checkbox", "value": 0},
        {"id": "verification", "label": "Biochemical verification", "type": "checkbox", "value": 0},
        {"id": "outcome", "label": "Type of outcome", "type": "select", "choices": outcome, "value": "Abstinence: Continuous "},
        ],
        "interventions": [
        {"id": "intervention", "label": "Behavioral support", "type": "multiselect", "choices": intervention, "value": []},
        {"id": "pharmacological", "label": "Pharmacological support", "type": "select", "choices": pharmacological, "value": "-"},
        {"id": "delivery", "label": "Intervention mode of delivery", "type": "multiselect", "choices": delivery, "value": []},
        {"id": "source", "label": "Intervention provider", "type": "multiselect", "choices": source, "value": []}
    ]}

@app.route("/hello_world")
def hello_world():
    return {"text":"Hello, World!"}

@app.route("/get_labels")
def get_labels():
    label_csv = pd.read_csv('data/feature-semantics-labeled.csv')

    # replace nulls with empty strings
    label_csv = label_csv.fillna("")

    label_list = label_csv.to_dict(orient='records')

    # set the featurename as the key
    labels = {row['featurename']: row for row in label_list}

    return labels

