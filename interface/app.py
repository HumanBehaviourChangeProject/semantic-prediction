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


with open('data/hbcp_gen.pkl', "rb") as fin:
    features, labels = pickle.load(fin)
    #features = filter_features(features)

index = np.array(features.index)
featurenames = [x[1] for x in features.columns]

app_ui = ui.page_fluid(
    ui.h2("Hello Shiny!"),
    ui.input_slider("n", "N", 0, 100, 20),
    ui.output_text_verbatim("txt"),
    ui.output_text_verbatim("txtfeatures")
)


def server(input, output, session):
    @output
    @render.text
    def txt():
        return f"n*2 is {input.n() * 2}"

    @output
    @render.text
    def txtfeatures():
        return str(featurenames)


app = App(app_ui, server)
