import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.base._penalties import L2ConstraintsPenalty
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def mdf_fit(features, labels, rename):
    data = features
    feature_names = list(features.columns)
    data["Outcome"] = labels
    features[features.isnull()] = 0
    #data.columns = ["Outcome"] + rename
    data['document_id'] = [v[0] for v in data.index.values]
    #data = sm.datasets.get_rdataset("dietox", "geepack").data
    formula = "Outcome ~ " + " + ".join(feature_names)
    md = smf.mixedlm(formula, data=data, groups=["document_id"])
    mdf = md.fit()  # cov_pen=L2ConstraintsPenalty(data.iloc[:, -1]))
    print(mdf.summary())
    return mdf

if __name__ == "__main__":
    with open("/home/glauer/.config/JetBrains/PyCharm2021.3/scratches/hbcp/data/cleaned_dataset_13Feb2022_notes_removed.csv", "rb") as fin:
        features, labels, rename = pickle.load(fin)
    features[features == None] = 0

    train_index, test_index = train_test_split(range(features.shape[0]), test_size=0.1)
    mdf = mdf_fit(features.iloc[train_index], labels[train_index], rename)

    p = mdf.predict(features.iloc[test_index])
    rmse = np.average((p - labels[test_index])**2)**0.5
    print(rmse)