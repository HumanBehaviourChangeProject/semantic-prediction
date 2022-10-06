import statsmodels.api as sm

import numpy as np
from numpy import linalg

from base import BaseModel
import typing
import os

class MLEModel(BaseModel):

    @classmethod
    def name(cls):
        return "mle"

    def train(self, train_features: np.ndarray, train_labels: np.ndarray, val_features: np.ndarray,
              val_labels: np.ndarray, variables: typing.List[typing.AnyStr]):
        train_filter = np.any(train_features, axis=1)
        data = train_features[train_filter]
        Q, R = linalg.qr(data)
        self.independent = np.where(np.abs(R.diagonal()) > 1e-05)[0]
        data = data.iloc[:, self.independent]
        # data.columns = ["Outcome"] + rename
        documents = [v[0] for v in data.index.values]

        md = sm.MixedLM(train_labels[train_filter], data, groups=documents)
        mdf = md.fit()  # cov_pen=L2ConstraintsPenalty(data.iloc[:, -1]))
        print(mdf.summary())
        self.model = mdf

    def predict(self, features):
        return self.model.predict(features.iloc[:, self.independent])

    def save(self, path):
        self.model.save(os.path.join(path,"model.pkl"))
