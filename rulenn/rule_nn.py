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

class RuleNet(nn.Module):
    def __init__(
        self, num_variables, num_conjunctions, names, config=None, append=None
    ):
        super().__init__()
        self.num_features = num_variables
        if config is None:
            conj = 3 - 6 * torch.rand(
                (num_conjunctions, 2 * num_variables), requires_grad=True
            )
            rw = 10 - 20 * torch.rand((num_conjunctions,), requires_grad=True)
            if append is not None:
                conj = torch.cat((conj, append[0]))
                rw = torch.cat((rw, append[1]))
        else:
            conj = config["conjunctions"]
            rw = config["rule_weights"]
        self.conjunctions = nn.Parameter(conj)
        self.num_rules = conj.shape[0]
        self.rule_weights = nn.Parameter(rw)
        self.base = nn.Parameter(torch.tensor(10.0, requires_grad=True), )
        self.non_lin = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

        self.implication_filter = self.load_implication_filters(names)
        self.disjoint_filter = self.load_disjoint_filters(names)

        d = torch.diag(torch.ones((len(names),), requires_grad=False))
        z = torch.zeros(len(names), len(names), requires_grad=False)
        self.pos = torch.cat([d, z])
        self.neg = torch.cat([z, d])

    def to(self, device):
        self.device = device
        super().to(device)

    def calculate_fit(self, x0):
        mu = self.dropout(self.non_lin(self.conjunctions))
        x = torch.cat((x0, 1 - x0), dim=1)
        x_exp = x.unsqueeze(1).expand((-1, self.conjunctions.shape[0], -1))
        x_pot = (mu * x_exp) + (1 - mu)  # torch.pow(x_exp, mu)
        return torch.prod(x_pot, dim=-1)

    def forward(self, x0):
        return self.apply_fit(self.calculate_fit(x0))

    def apply_fit(self, fit):
        return self.base + torch.sum(self.rule_weights * fit, dim=-1).squeeze(-1)

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
        non_crips_penalty = torch.sum(
            torch.sum(w * (1 - w), dim=-1), dim=0
        )  # penalty for non-crisp rules
        m = torch.max(
            torch.stack((torch.sum(w, dim=-1) - 3, torch.zeros(w.shape[:-1]))),
            dim=0,
        )
        long_rules_penalty = torch.sum(m[0]) # penalty for long rules

        contradiction_penalty = 0.5 * torch.sum(
            torch.prod(w[:,:self.num_features] * w[:,self.num_features:] , dim=-1), dim=-1
        )

        disjoint_implied_penalty = 0.5 * torch.sum(
            torch.sum(calculate_pure_fit(w, self.disjoint_filter + self.implication_filter), dim=-1), dim=-1
        )

        penalties = 0.1*(
                non_crips_penalty
                + long_rules_penalty
                + contradiction_penalty
                + disjoint_implied_penalty
        )
        return penalties

    def load_disjoint_filters(self, names, device=None):
        filters = []
        z = torch.zeros(len(names), requires_grad=False, device=device)
        for (lh, rhs) in fc.features_disjoint.items():
            lhi = self.get_index(lh, names)
            for rh in rhs:
                rhi = self.get_index(rh, names)
                filters.append((lhi, rhi))
                filters.append((lhi, len(names) + rhi))
        return filters

    @staticmethod
    def get_index(v, names):
        return names.index(v)

    def load_implication_filters(self, names, device=None):
        filters = []
        z = torch.zeros(len(names), requires_grad=False, device=device)
        for (lh, rhs) in fc.features_implied.items():
            lhi = self.get_index(lh, names)
            for rh in rhs:
                rhi = self.get_index(rh, names)
                filters.append((lhi, rhi))
                filters.append((lhi, len(names) + rhi))
        return filters

def calculate_pure_fit(w, fltr):
    return torch.prod(torch.stack([w[f] for f in fltr]), dim=-1)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


class RuleNNModel(BaseModel):

    def train(self, train_features, train_labels, val_features, val_labels, variables):

        pre = torch.tensor(
            np.linalg.lstsq(train_features.cpu().numpy(), train_labels.cpu(), rcond=None)[0],
            requires_grad=True,
        )
        num_features = train_features.shape[1]
        presence_filter = torch.abs(pre) < 50
        conjs = (torch.diag(10 * torch.ones(num_features, requires_grad=True)) + 10*torch.rand(num_features,num_features)-5)[presence_filter]
        conjs = torch.cat((conjs, 10*torch.rand(conjs.shape)-5), dim=-1)
        net = RuleNet(num_features, 200-conjs.shape[0], variables,append=(conjs, pre[presence_filter]))
        criterion = nn.MSELoss()
        val_criterion = nn.L1Loss()
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        keep_top = 5
        no_improvement = 0
        epoch = 0
        best = []
        j = 0
        running_loss = 0.0
        running_penalties = 0.0
        max_epoch = 200

        train_index = list(range(train_features.shape[0]))

        while epoch < max_epoch or no_improvement < max_epoch*0.2:
            # loop over the dataset multiple times
            # get the inputs; data is a list of [inputs, labels]
            # forward + backward + optimize

            e = sigmoid(10*min((epoch-50) / 100, 1)-8) if epoch > 50 else 0
            batch_size = 10
            random.shuffle((train_index))
            net.train()
            for i in list(range(0, len(train_index), batch_size)):
                optimizer.zero_grad()
                batch_index = train_index[i : i + batch_size]
                batch_labels = train_labels[batch_index].squeeze(-1)
                outputs = net(train_features[batch_index])

                base_loss = criterion(outputs, batch_labels)

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
            if epoch % 10 == 0:
                best_val_loss = best[0][0] if best else -1
                print(
                    f"{{epoch: {epoch},\tloss: {(running_loss / j):.2f},\tpenalties: {(running_penalties / j):.2f},\tval_loss: {val_loss.item():.2f},\t best_val_loss: {best_val_loss:0.2f},\tval_rmse: {val_loss.item()**0.5:.2f},\tval_mae: {val_rmse.item():.2f},\te: {e}}}"
                )
                print(net.base)
                running_loss = 0.0
                running_penalties = 0.0
                j = 0
            epoch += 1

        self.model = best[0][1]

    def predict(self, features):
        self.model.eval()
        return self.model(features)


    @classmethod
    def prepare_data(cls, features, labels):
        return torch.tensor(features.values).float(), torch.tensor(labels).float()


    @classmethod
    def name(cls):
        return "rulenn"


    def save(self, path:str):
        with open(os.path.join(path,"model.rules"), "w") as frules:
            rules = torch.cat(
                (
                    nnf.hardsigmoid(self.model.conjunctions),
                    self.model.rule_weights.unsqueeze(-1),
                ),
                dim=-1,
            )
            frules.write(";".join("&".join(str(v.item()) for v in row) for row in rules) + "\n")

        torch.save(self.model.state_dict(), os.path.join(path, "model.ckpt"))
