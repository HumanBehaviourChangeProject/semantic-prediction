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

    def _train(self, train_features: np.ndarray, train_labels: np.ndarray, val_features: np.ndarray,
              val_labels: np.ndarray, train_docs, val_docs, verbose=True, weights=None, delay_val=True):
        train_filter = np.any(train_features, axis=1)
        data = train_features[train_filter]
        Q, R = linalg.qr(data)
        self.independent = np.where(np.abs(R.diagonal()) > 1e-05)[0]
        data = data[:, self.independent]
        # data.columns = ["Outcome"] + rename
        documents = [d[0] for d in train_docs]

        md = sm.MixedLM(train_labels[train_filter], data, groups=documents)
        mdf = md.fit()  # cov_pen=L2ConstraintsPenalty(data.iloc[:, -1]))
        if verbose:
            print(mdf.summary())
        self.model = mdf

    def predict(self, features):
        return self.model.predict(features[:, self.independent])

    def save(self, path):
        self.model.save(os.path.join(path,"model.pkl"))
