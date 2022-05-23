import random

from torch import nn
import torch
from torch.nn import functional as nnf
import torch.optim as optim
import copy
import numpy as np
import feature_config as fc


class RuleNet(nn.Module):
    def __init__(
        self, num_variables, num_conjunctions, layers, config=None, append=None
    ):
        assert layers > 0
        super().__init__()
        if config is None:
            conj = 1 - 2 * torch.rand(
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
        self.rule_weights = nn.Parameter(rw)
        # self.base = nn.Parameter(5 - 10 * torch.rand(1, requires_grad=True), )
        self.non_lin = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def calculate_fit(self, x0):
        mu = self.non_lin(self.conjunctions)
        x = torch.concat((x0, 1 - x0), dim=1)
        x_exp = x.unsqueeze(1).expand((-1, self.conjunctions.shape[0], -1))
        x_pot = (mu * x_exp) + (1 - mu)  # torch.pow(x_exp, mu)
        return torch.min(x_pot, dim=-1)[0]

    def forward(self, x0):
        return self.apply_fit(self.calculate_fit(x0))

    def apply_fit(self, fit):
        return 5 + torch.sum(self.rule_weights * fit, dim=-1).squeeze(-1)

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


def cross_val(patience, features, labels, variables):
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    features.to(device)
    labels.to(device)
    features[features.isnan()] = 0
    results = []
    with open("results/rule_nn.txt", "w") as fout:
        with open("results/rules.txt", "w") as frules:
            for _ in range(100):
                index = list(np.array(range(features.shape[0])))
                random.shuffle(index)
                step = len(index) // 5
                chunks = [index[i : i + step] for i in range(0, len(index), step)]
                for i in range(len(chunks)):
                    train_index = [
                        c for j in range(len(chunks)) for c in chunks[j] if i != j
                    ]
                    val_index = chunks[i]
                    best = main(
                        patience,
                        features,
                        labels,
                        train_index,
                        val_index,
                        variables,
                        frules,
                        device,
                    )
                    results.append(best)
                    for x in best[2]:
                        fout.write(str(x.item()) + "\n")
                    fout.flush()
                    exit()


def load_disjoint_filters(names):
    filters = []
    z = torch.zeros(features.shape[1], requires_grad=False)
    for (lh, rhs) in fc.features_disjoint.items():
        for rh in rhs:
            v = torch.zeros(features.shape[1], requires_grad=False)
            v[names.index[lh]] = 1
            v[names.index[rh]] = 1
            filters.append(torch.cat((v, z)))
            filters.append(torch.cat((z, v)))
    return torch.stack(filters)


def load_implication_filters(names):
    filters = []
    z = torch.zeros(features.shape[1], requires_grad=False)
    for (lh, rhs) in fc.features_implied.items():
        for rh in rhs:
            v = torch.zeros(features.shape[1], requires_grad=False)
            v[names.index(lh)] = 1
            v[names.index(rh)] = 1
            filters.append(torch.cat((v, z)))
            filters.append(torch.cat((z, v)))
    return torch.stack(filters)


def calculate_pure_fit(w, fltr):
    return torch.min(
        w.unsqueeze(1).repeat(1, fltr.shape[0], 1) * fltr + (1 - fltr), dim=-1
    )[0]


def main(epochs, features, labels, train_index, val_index, variables, frules, device):
    # test_index, val_index = train_test_split(test_index, test_size=0.25)
    pre = torch.tensor(
        np.linalg.lstsq(features.numpy(), labels[:, 0], rcond=None)[0],
        requires_grad=True,
    )
    conjs = torch.diag(10 * torch.ones(len(pre)))
    conjs = torch.cat((conjs, torch.zeros_like(conjs, requires_grad=True)), dim=-1)
    net = RuleNet(features.shape[1], 26, 3, append=(conjs, pre))
    net.to(device)
    criterion = nn.MSELoss()
    val_criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    postfix = ""
    keep_top = 5
    no_improvement = 0
    epoch = 0
    best = []
    j = 0
    running_loss = 0.0
    running_penalties = 0.0
    d = torch.diag(torch.ones((features.shape[1],), requires_grad=False))
    z = torch.zeros(features.shape[1], features.shape[1], requires_grad=False)
    pos = torch.cat([d, z])
    neg = torch.cat([z, d])
    implication_filter = load_implication_filters(variables)
    disjoint_filter = load_implication_filters(variables)

    while epoch < 200 or no_improvement < 50:  # loop over the dataset multiple times
        # get the inputs; data is a list of [inputs, labels]
        # forward + backward + optimize
        e = 0.2 * torch.sigmoid(
            torch.tensor(epoch / 25 - 6, device=device, requires_grad=False)
        )
        batch_size = 10
        random.shuffle((train_index))

        for i in list(range(0, len(train_index), batch_size)):
            optimizer.zero_grad()
            batch_index = train_index[i : i + batch_size]
            batch_labels = labels[batch_index].squeeze(-1)
            outputs = net(features[batch_index])
            w = net.non_lin(net.conjunctions)
            base_loss = criterion(outputs, batch_labels)
            non_crips_penalty = 2 * torch.sum(
                torch.sum(w * (1 - w), dim=-1), dim=0
            )  # penalty for non-crisp rules
            m = torch.max(
                torch.stack((torch.sum(w, dim=-1) - 3, torch.zeros(w.shape[:-1]))),
                dim=0,
            )
            long_rules_penalty = torch.sum(m[0], dim=0)  # penalty for long rules

            contradiction_penalty = 0.5 * torch.sum(
                torch.sum(torch.matmul(w, pos) * torch.matmul(w, neg), dim=-1), dim=-1
            )
            disjoint_penalty = 0.5 * torch.mean(
                torch.sum(calculate_pure_fit(w, disjoint_filter), dim=-1), dim=-1
            )
            implied_penalty = 0.5 * torch.mean(
                torch.sum(calculate_pure_fit(w, implication_filter), dim=-1), dim=-1
            )

            penalties = e * (
                non_crips_penalty
                + long_rules_penalty
                + contradiction_penalty
                + disjoint_penalty
                + implied_penalty
            )

            loss = base_loss + penalties
            loss.backward()
            optimizer.step()
            # print(net.conjunctions.grad)
            running_loss += base_loss.detach().item()
            running_penalties += penalties.detach().item()
            j += 1

        with torch.no_grad():
            val_out = net(features[val_index])
            val_loss = criterion(val_out, labels[val_index].squeeze(-1))
            val_rmse = val_criterion(val_out, labels[val_index].squeeze(-1)) ** 0.5

        if epoch > 300:
            if not best or val_loss - best[-1][0] < -0.05:
                best.append(
                    (
                        val_loss,
                        copy.deepcopy(net),
                        val_out - labels[val_index].squeeze(-1),
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
            print(
                f"epoch: {epoch},\tloss: {(running_loss / j)},penalties: {(running_penalties / j)},\tval_loss: {val_loss.item()},\tval_alt: {val_rmse.item()},\tno improvement since: {no_improvement}"
            )
            running_loss = 0.0
            running_penalties = 0.0
            j = 0
        epoch += 1
    best_model = best[0][1]
    rules = torch.cat(
        (
            nnf.hardsigmoid(best_model.conjunctions),
            best_model.rule_weights.unsqueeze(-1),
        ),
        dim=-1,
    )
    frules.write(";".join("&".join(str(v.item()) for v in row) for row in rules) + "\n")
    print("Finished Training")
    return best[0]


if __name__ == "__main__":
    import pickle
    import sys

    with open(sys.argv[2], "rb") as fin:
        features, labels, rename = pickle.load(fin)
    cross_val(
        int(sys.argv[1]),
        torch.tensor(features.astype(float).values).float(),
        torch.tensor(labels.astype(float)).float(),
        rename,
    )
