import random
import sys

from torch import nn
import torch
from torch.nn import functional as nnf
import torch.optim as optim
import copy
import numpy as np
import rulenn.feature_config as fc
from base import BaseModel
import math
import os
import json


class RuleNet(nn.Module):
    def __init__(
        self, num_variables, num_conjunctions, names, config=None, append=None, device="cpu"
    ):
        super().__init__()
        self.variables = names
        if config is None:
            conj = - 18 * torch.rand(
                (num_conjunctions, 2 * num_variables), requires_grad=True, device=device
            )
            rw = 10 - 20 * torch.rand((num_conjunctions,), requires_grad=True, device=device)
            base = 13 + 6 * torch.rand(1, requires_grad=True, device=device)
            if append is not None:
                conj = torch.cat((conj, append[0]))
                rw = torch.cat((rw, append[1]))
        else:
            conj = config["conjunctions"]
            rw = config["rule_weights"]
            base = config["base"]
        self.num_features = conj.shape[1]//2
        self.conjunctions = nn.Parameter(conj)
        self.num_rules = conj.shape[0]
        self.rule_weights = nn.Parameter(rw)
        self.base = nn.Parameter(base)
        self.non_lin = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

        self.implication_filter = self.load_implication_filters()
        self.disjoint_filter = self.load_disjoint_filters()

        d = torch.diag(torch.ones((len(names),), requires_grad=False, device=device))
        z = torch.zeros(len(names), len(names), requires_grad=False, device=device)
        self.pos = torch.cat([d, z])
        self.neg = torch.cat([z, d])

    def tnorm(self, x, dim):
        return torch.min(x, dim=dim)[0]

    def to(self, device):
        self.device = device
        super().to(device)

    def calculate_fit(self, x0):
        mu = self.non_lin(self.conjunctions)
        x = torch.cat((x0, 1 - x0), dim=1)
        x_pot = mu * (x[:,None,:] - 1) + 1  # torch.pow(x_exp, mu)
        return self.tnorm(x_pot, dim=-1)

    def forward(self, x0):
        return self.apply_fit(self.calculate_fit(x0))

    def apply_fit(self, fit):
        return 10 + torch.sum(self.rule_weights * fit, dim=-1).squeeze(-1)

    def print_rules(self, variables):
        weights = self.non_lin(self.conjunctions)
        for i in sorted(
            range(self.conjunctions.shape[0]), key=lambda i: self.rule_weights[i]
        ):
            yield " & ".join(
                v
                for v, c in zip(variables + [f"not {v}" for v in variables], weights[i])
                if c > 0.5
            ) + " => " + str(self.rule_weights[i].item())

    def calculate_penalties(self):
        w = self.non_lin(self.conjunctions)
        non_crips_penalty = 10 * torch.sum(
            torch.sum(w * (1 - w), dim=-1), dim=0
        )  # penalty for non-crisp rules
        m = torch.mean(
            torch.stack((torch.sum(w, dim=-1) - 3, torch.zeros(w.shape[:-1]))),
            dim=0,
        )
        long_rules_penalty = 10*torch.norm(m) # penalty for long rules

        contradiction_penalty = 0.5 * torch.sum(
            self.tnorm(w[:,:self.num_features] * w[:,self.num_features:] , dim=-1), dim=-1
        )

        disjoint_implied_penalty = 0.5 * torch.sum(
            torch.sum(self.calculate_pure_fit(w, self.disjoint_filter + self.implication_filter), dim=-1), dim=-1
        )

        penalties = 0.01*(
                non_crips_penalty
                + long_rules_penalty
                + contradiction_penalty
                + disjoint_implied_penalty
        )
        return penalties

    def load_disjoint_filters(self, device=None):
        filters = []
        for (lh, rhs) in fc.features_disjoint.items():
            lhi = self.get_index(lh)
            if lhi is not None:
                for rh in rhs:
                    rhi = self.get_index(rh)
                    if rhi is not None:
                        filters.append((lhi, rhi))
                        filters.append((lhi, len(self.variables) + rhi))
        return filters

    def get_index(self, v):
        try:
            return self.variables.index(v)
        except ValueError:
            print("Column not found:", v)
            return None

    def load_implication_filters(self, device=None):
        filters = []
        for (lh, rhs) in fc.features_implied.items():
            lhi = self.get_index(lh)
            if lhi is not None:
                for rh in rhs:
                    rhi = self.get_index(rh)
                    if rhi is not None:
                        filters.append((lhi, rhi))
                        filters.append((lhi, len(self.variables) + rhi))
        return filters

    def calculate_pure_fit(self, w, fltr):
        return self.tnorm(torch.stack([w[f] for f in fltr]), dim=-1)


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


class RuleNNModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super(RuleNNModel, self).__init__(*args, **kwargs)
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

    def _train(self, train_features, train_labels, val_features, val_labels, variables, *args, verbose=True, weights=None):

        pre = torch.tensor(
            np.linalg.lstsq(train_features.cpu().numpy(), train_labels.cpu(), rcond=None)[0],
            requires_grad=True, device=self.device
        )
        num_features = train_features.shape[1]
        presence_filter = torch.abs(pre) < 50
        conjs = (torch.diag(10 * torch.ones(num_features, requires_grad=True, device=self.device)) + -10*torch.rand(num_features,num_features, device=self.device))[presence_filter]
        conjs = torch.cat((conjs, -10*torch.rand(conjs.shape, device=self.device)), dim=-1)
        net = RuleNet(num_features, 200-conjs.shape[0], self.variables, append=(conjs, pre[presence_filter]), device=self.device)
        criterion = nn.MSELoss()
        val_criterion = nn.L1Loss()
        optimizer = optim.Adam(net.parameters(), lr=1e-2)
        keep_top = 5
        no_improvement = 0
        epoch = 0
        best = []
        j = 0
        running_loss = 0.0
        running_penalties = 0.0
        max_epoch = 400

        train_index = list(range(train_features.shape[0]))

        while epoch < max_epoch or no_improvement < max_epoch*0.2:
            # loop over the dataset multiple times
            # get the inputs; data is a list of [inputs, labels]
            # forward + backward + optimize

            e = min(epoch/(max_epoch/2),1)
            batch_size = 100
            random.shuffle((train_index))
            net.train()
            for i in list(range(0, len(train_index), batch_size)):
                optimizer.zero_grad()
                batch_index = train_index[i : i + batch_size]
                batch_labels = train_labels[batch_index].squeeze(-1)
                outputs = net(train_features[batch_index])

                base_loss = (outputs - batch_labels)**2
                if weights is not None:
                    base_loss = base_loss * weights[batch_index].squeeze(-1)

                base_loss = torch.sum(base_loss, dim=-1)**0.5

                #loss = base_loss + penalties
                penalties = e * net.calculate_penalties()

                loss = base_loss + penalties
                loss.backward()
                optimizer.step()

                # print(net.conjunctions.grad)
                running_loss += base_loss.detach().item()
                running_penalties += penalties.detach().item()
                j += 1
            net.eval()
            with torch.no_grad():
                val_out = net(val_features)
                val_loss = criterion(val_out, val_labels.squeeze(-1))
                val_rmse = val_criterion(val_out, val_labels.squeeze(-1))

            if epoch > max_epoch:
                if not best or val_loss - best[-1][0] < -0.05:
                    best.append(
                        (
                            val_loss,
                            copy.deepcopy(net),
                            val_out - val_labels.squeeze(-1),
                        )
                    )
                    best = sorted(best, key=lambda x: x[0])
                    if not len(best) <= keep_top:
                        del best[-1]
                    no_improvement = 0
                else:
                    no_improvement += 1
            # print statistics
            if verbose and epoch % 10 == 0:
                best_val_loss = best[0][0] if best else -1
                print(
                    f"{{epoch: {epoch},\tloss: {(running_loss / j):.2f},\tpenalties: {(running_penalties / j):.2f},\tval_loss: {val_loss.item():.2f},\t best_val_loss: {best_val_loss:0.2f},\tval_rmse: {val_loss.item()**0.5:.2f},\tval_mae: {val_rmse.item():.2f},\te: {e:.2f}}}"
                )
                running_loss = 0.0
                running_penalties = 0.0
                j = 0
            epoch += 1

        self.model = best[0][1]

    def _predict(self, features) -> np.ndarray:
        self.model.eval()
        return self.model(features).detach().numpy()


    @classmethod
    def _prepare_single(cls, data):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        return torch.tensor(data, device=device).float()


    @classmethod
    def name(cls):
        return "rulenn"

    def print_rules(self, threshold=0.2):
        names = self.model.variables + ["not " + n for n in self.model.variables]
        for (row, weight) in zip(self.model.non_lin(self.model.conjunctions), self.model.rule_weights):
            print(" & ".join([n for n, v in zip(names,row) if v > threshold]) + " -> " + str(weight.item()))

    def save(self, path:str):
        with open(os.path.join(path, "model.json"), "w") as fout:
            json.dump(
                dict(
                    variables=self.variables,
                    conjunctions=self.model.conjunctions.detach().numpy().tolist(),
                    weights=self.model.rule_weights.detach().numpy().tolist(),
                    base=self.model.base.detach().numpy().tolist()
                ), fout
            )

    @classmethod
    def load(cls, path):
        with open(os.path.join(path), "r") as fin:
            d = json.load(fin)
            x = cls(d["variables"])
            x.model = RuleNet(None, None, d["variables"], config=dict(conjunctions=torch.tensor(d["conjunctions"]),
                                                  rule_weights=torch.tensor(d["weights"]),
                                                  base=torch.tensor(d["base"])))
            return x