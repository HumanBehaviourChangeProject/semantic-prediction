import random

from torch import nn
import torch
from torch.nn import functional as nnf
import torch.optim as optim
import copy
import numpy as np
import feature_config as fc
from sklearn.model_selection import train_test_split
from torchmetrics import functional as tmf

RANDOM_SEED = None

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
        x = torch.cat((x0, 1 - x0), dim=1)
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


def cross_val(patience, raw_features, raw_labels, variables, index):
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    features = torch.tensor(raw_features.values, dtype=torch.float).to(device)
    labels = torch.tensor(raw_labels, dtype=torch.float).to(device)
    features[features.isnan()] = 0

    with open("results/rule_cross_nn.txt", "w") as fout:
        for _ in range(10):
            chunks = get_cross_split(raw_features, 5)
            for i in range(len(chunks)-1):
                train_index = [
                    c for j in range(len(chunks)) for c in chunks[j] if i != j and i != j-1
                ]
                val_index = chunks[i]
                test_index = chunks[i+1]
                best = main(
                    patience,
                    features,
                    labels,
                    train_index,
                    val_index,
                    variables,
                    device
                )

                model = best[1]
                model.eval()
                y_pred = model(features[test_index])

                for pred, targ in zip(y_pred.tolist(), labels[test_index].squeeze(-1).tolist()):
                    fout.write( str(pred-targ) + "\n")

def single_run(patience, features, labels, variables, index):
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    features = torch.tensor(features.values, dtype=torch.float).to(device)
    labels = torch.tensor(labels, dtype=torch.float).to(device)
    features[features.isnan()] = 0

    train_index, test_index, val_index = get_data_split(features, labels, seed=RANDOM_SEED)

    with open("results/rules.txt", "w") as frules:
        best = main(
            patience,
            features,
            labels,
            train_index,
            val_index,
            variables,
            device,
            frules=frules
        )
        model = best[1]
        model.eval()

    with open("results/rulenn_errors_test.csv", "w") as fout:
        y_pred = model(features[test_index])
        fout.write(",".join(("doc,arm","prediction","target")) + "\n")
        for t in zip(index[test_index], y_pred.tolist(), labels[test_index].squeeze(-1).tolist()):
            fout.write(",".join((t[0][0],t[0][1],*map(str,t[1:]))) + "\n")
        fout.flush()
        print("MSE:", tmf.mean_squared_error(y_pred, labels[test_index].squeeze(-1)))
    with open("results/rulenn_errors_all.csv", "w") as fout:
        y_pred = model(features)
        fout.write(",".join(("set", "doc,arm", "prediction", "target")) + "\n")
        for i, t in enumerate(zip(index, y_pred.tolist(), labels.squeeze(-1).tolist())):
            fout.write(",".join(("train" if i in train_index else ("test" if i in test_index else "val"), t[0][0], t[0][1], *map(str, t[1:]))) + "\n")
        fout.flush()


def load_disjoint_filters(names,device=None):
    filters = []
    z = torch.zeros(features.shape[1], requires_grad=False, device=device)
    for (lh, rhs) in fc.features_disjoint.items():
        for rh in rhs:
            v = torch.zeros(features.shape[1], requires_grad=False, device=device)
            v[names.index(lh)] = 1
            v[names.index(rh)] = 1
            filters.append(torch.cat((v, z)))
            filters.append(torch.cat((z, v)))
    return torch.stack(filters)


def load_implication_filters(names, device=None):
    filters = []
    z = torch.zeros(features.shape[1], requires_grad=False, device=device)
    for (lh, rhs) in fc.features_implied.items():
        for rh in rhs:
            v = torch.zeros(features.shape[1], requires_grad=False, device=device)
            v[names.index(lh)] = 1
            v[names.index(rh)] = 1
            filters.append(torch.cat((v, z)))
            filters.append(torch.cat((z, v)))
    return torch.stack(filters)


def calculate_pure_fit(w, fltr):
    return torch.min(
        w.unsqueeze(1).repeat(1, fltr.shape[0], 1) * fltr + (1 - fltr), dim=-1
    )[0]


def get_data_split(features, labels, seed=None):
    documents = list(i[0] for i in features.index)
    random.shuffle(documents)

    train_doc_index, val_doc_index = train_test_split(list(documents), random_state=seed, train_size=0.8)
    test_doc_index, val_doc_index = train_test_split(val_doc_index, random_state=seed, train_size=0.6)

    train_index = [i for i, t in enumerate(features.index) if t[0] in train_doc_index]
    test_index = [i for i, t in enumerate(features.index) if t[0] in test_doc_index]
    val_index = [i for i, t in enumerate(features.index) if t[0] in val_doc_index]

    return train_index, test_index, val_index


def get_cross_split(features, num_splits=3):
    documents = list(i[0] for i in features.index)
    random.shuffle(documents)

    p = 1/num_splits

    splits = []
    remainder = documents
    for i in range(num_splits):
        split, remainder = train_test_split(remainder, train_size=p)
        splits.append(split)

    return [[i for i, t in enumerate(features.index) if t[0] in split] for split in splits]



def main(epochs, features, labels, train_index, val_index, variables, device, frules=None):
    # test_index, val_index = train_test_split(test_index, test_size=0.25)
    pre = torch.tensor(
        np.linalg.lstsq(features[train_index].cpu().numpy(), labels[train_index, 0].cpu(), rcond=None)[0],
        requires_grad=True,
    )
    conjs = torch.diag(10 * torch.ones(len(pre)))
    conjs = torch.cat((conjs, torch.zeros_like(conjs, requires_grad=True)), dim=-1)
    net = RuleNet(features.shape[1], 26, 3, append=(conjs, pre))
    net.to(device)
    criterion = nn.MSELoss()
    val_criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=5e-4)
    postfix = ""
    keep_top = 5
    no_improvement = 0
    epoch = 0
    best = []
    j = 0
    running_loss = 0.0
    running_penalties = 0.0
    d = torch.diag(torch.ones((features.shape[1],), requires_grad=False, device=device))
    z = torch.zeros(features.shape[1], features.shape[1], requires_grad=False,device=device)
    pos = torch.cat([d, z])
    neg = torch.cat([z, d])
    implication_filter = load_implication_filters(variables,device=device)
    disjoint_filter = load_disjoint_filters(variables,device=device)

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
            non_crips_penalty = 10 * e * torch.sum(
                torch.sum(w * (1 - w), dim=-1), dim=0
            )  # penalty for non-crisp rules
            m = torch.max(
                torch.stack((torch.sum(w, dim=-1) - 3, torch.zeros(w.shape[:-1],device=device))),
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
        net.eval()
        with torch.no_grad():
            val_out = net(features[val_index])
            val_loss = criterion(val_out, labels[val_index].squeeze(-1))
            val_rmse = val_criterion(val_out, labels[val_index].squeeze(-1))
        net.train()

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
                f"epoch: {epoch},\tloss: {(running_loss / j)},\tpenalties: {(running_penalties / j)},\tval_loss: {val_loss.item()},\tval_alt: {val_rmse.item()},\tno improvement since: {no_improvement}"
            )
            running_loss = 0.0
            running_penalties = 0.0
            j = 0
        epoch += 1
    best_model = best[0][1]
    if frules is not None:
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
    index = np.array(features.index)
    cross_val(
        int(sys.argv[1]),
        features,
        labels,
        rename,
        index
    )
