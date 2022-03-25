import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
import numpy as np
import copy

class Forward(nn.Module):

    def __init__(self, in_dim: int):
        super().__init__()
        self.fwd = nn.Sequential(nn.Linear(in_dim, 2*in_dim), nn.LeakyReLU(), nn.Dropout(0.1), nn.Linear(2*in_dim, in_dim), nn.LeakyReLU(), nn.Dropout(0.1), nn.Linear(in_dim, in_dim//2), nn.ReLU(), nn.Linear(in_dim//2, 1))

    def forward(self, data):
        return self.fwd(data)


def cross_val(patience, features, labels, variables):

    features[features.isnan()] = 0
    results = []
    with open("results/dl.txt", "w") as fout:
        for _ in range(100):
            index = list(np.array(range(features.shape[0])))
            random.shuffle(index)
            step = len(index) // 10
            chunks = [index[i:i + step] for i in range(0, len(index), step)]
            for i in range(len(chunks)):
                train_index = [c for j in range(len(chunks)) for c in chunks[j] if i != j ]
                val_index = chunks[i]
                best = main(patience, features, labels, train_index, val_index, variables)
                results.append(best)
                for x in best[2]:
                    fout.write(str(x.item()) + "\n")
                fout.flush()


def main(epochs, features, labels, train_index, val_index, variables):
    features[features.isnan()] = 0

    net = Forward(features.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    best = []
    keep_top = 5
    no_improvement = 0
    epoch = 0
    while epoch < epochs or no_improvement < 20:  # loop over the dataset multiple times
        running_loss = 0.0
        # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients


        # forward + backward + optimize
        j = 0
        batch_size = 10
        for i in range(0, len(train_index), batch_size):
            optimizer.zero_grad()
            batch_index = train_index[i:i + batch_size]
            outputs = net(features[batch_index])
            loss = criterion(outputs, labels[batch_index])
            running_loss += loss.item()
            j += 1
            loss.backward()
            optimizer.step()

        val_out = net(features[val_index])
        val_loss = criterion(val_out, labels[val_index])


        if not best or val_loss - best[-1][0] < -0.05:
            best.append((val_loss, copy.deepcopy(net), (val_out - labels[val_index]).squeeze(-1)))
            best = sorted(best, key=lambda x: x[0])
            if not len(best) <= keep_top:
                del best[-1]
            no_improvement = 0
        else:
            no_improvement += 1
        # print statistics
        if epoch % 10 == 0:
            print(f"epoch: {epoch},\tloss: {(running_loss / j)},\tval_loss: {val_loss.item()},\tno improvement since: {no_improvement}")
            running_loss = 0.0
            j = 0
        epoch += 1

    print('Finished Training')
    return best[0]

if __name__ == "__main__":
    import pickle
    import sys
    with open(sys.argv[2], "rb") as fin:
        features, labels, rename = pickle.load(fin)
    cross_val(int(sys.argv[1]), torch.tensor(features.astype(float).values).float(),
              torch.tensor(labels.astype(float)).float(), rename)