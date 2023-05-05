from base import _load_data
from rulenn.rule_nn import RuleNNModel, sigmoid

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset, random_split
import csv

import ray
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import AsyncHyperBandScheduler

import sys

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 400

class Dataset(IterableDataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].squeeze(-1)


def get_data_loaders(path, seed=1):
    features, labels = _load_data(path, False, False, None)
    data = Dataset(torch.tensor(features.values), labels)
    train_size = int(0.8 * len(data))
    training_data, test_data = random_split(data, [train_size, len(data) - train_size], generator=torch.Generator().manual_seed(seed))
    feature_names = [b for _,b in features.columns.values]
    return DataLoader(training_data, batch_size=64, shuffle=True), DataLoader(test_data, batch_size=64, shuffle=True), feature_names




def train(model, optimizer, train_loader, device=None):
    device = device or torch.device("cpu")
    model.train()
    model.to(device)
    for epoch in range(EPOCH_SIZE):
        for batch_idx, (data, target) in enumerate(train_loader):
            epoch = batch_idx / len(data)

            e = sigmoid(
                -3 + 6 * epoch / (EPOCH_SIZE)) if epoch > EPOCH_SIZE / 4 else 0

            data, target = data.to(device), target.to(device)
            outputs = model(data)
            base_loss = F.mse_loss(outputs, target)
            if e > 0:
                penalties = model.calculate_penalties(e * 0.1)
            else:
                penalties = torch.tensor(0)
            loss = base_loss + penalties
            loss.backward()
            optimizer.step()


def test(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    running_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = F.l1_loss(outputs, target)
            running_loss += loss
            total += 1
    return running_loss / total

class Trainable(tune.Trainable):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.step_count = 1

    def step(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        train_loader, test_loader, feature_names = get_data_loaders(self.config["path"], seed=self.step_count)
        self.step_count += 1
        train_featurs, train_labels = zip(*train_loader)
        model = RuleNNModel(feature_names, device=device).prepare_model(torch.cat(train_featurs), torch.cat(train_labels), config=dict(
                non_crips =self.config["non_crips"],
                long_rules =self.config["long_rules"],
                weight =self.config["weight"],
                contradiction =self.config["contradiction"],
                negative_weight =self.config["negative_weight"],
                disjoint_implied = self.config["disjoint_implied"]
            ))
        optimizer = optim.Adamax(
            model.parameters(), lr=self.config["lr"]
        )

        train(model, optimizer, train_loader, device)
        acc = test(model, test_loader, device)
        # Report metrics (and possibly a checkpoint) to Tune
        return {"loss": acc}

def run(path):
    ray.init()

    # train_mnist(dict(lr=1e-3, momentum=1, non_crips =1, long_rules =1, weight =0.1, contradiction =0.5, negative_weight =0.1, disjoint_implied = 0.5))

    # for early stopping
    sched = AsyncHyperBandScheduler()

    resources_per_trial = {"cpu": 1}  # set this for GPUs
    tuner = tune.Tuner(
        Trainable,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            # scheduler=sched,
            num_samples=100,
        ),
        run_config=air.RunConfig(
            name="exp",
            stop={
                "training_iteration": 100,
            },
        ),
        param_space={
            "path": path,
            "lr": tune.loguniform(1e-4, 1e-2),
            "non_crips": tune.uniform(0.1, 10),
            "long_rules": tune.uniform(0.1, 10),
            "weight": tune.uniform(0.1, 10),
            "contradiction": tune.uniform(0.1, 10),
            "negative_weight": tune.uniform(0.1, 10),
            "disjoint_implied": tune.uniform(0.1, 10)
        },
    )
    results = tuner.fit()

    with open("raysults.csv", "wt") as fout:
        writer = csv.DictWriter(fout, ["mae", *results[0].config.keys()])
        writer.writeheader()
        writer.writerows(
            [dict(mae=r.metrics["loss"].item(), **r.config) for r in
             sorted(results, key=lambda x: x.metrics["loss"])])

if __name__ == "__main__":
    run(sys.argv[1])
