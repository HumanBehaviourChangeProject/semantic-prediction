from sklearn.ensemble import RandomForestRegressor

import numpy as np
from numpy import linalg
import pickle
from base import BaseModel
import typing
import os

class RFModel(BaseModel):

    @classmethod
    def name(cls):
        return "random_forest"

    def _train(self, train_features: np.ndarray, train_labels: np.ndarray, val_features: np.ndarray,
              val_labels: np.ndarray, *args, verbose=True, weights=None, delay_val=True):
        self.model = RandomForestRegressor(50, max_depth=3, max_leaf_nodes=5)
        self.model.fit(X=train_features, y=train_labels)
        pred = self.model.predict(X=val_features)
        return pred - val_labels

    def predict(self, features):
        return self.model.predict(X=features)

    def save(self, path):
        with open(os.path.join(path,"model.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)
