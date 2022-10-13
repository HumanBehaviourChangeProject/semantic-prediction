import os.path

import torch
from torch import nn
import torch.optim as optim
from base import BaseModel
import copy

class Forward(nn.Module):

    def __init__(self, in_dim: int):
        super().__init__()
        self.fwd = nn.Sequential(nn.Linear(in_dim, 2*in_dim), nn.LeakyReLU(), nn.Linear(2*in_dim, in_dim), nn.LeakyReLU(), nn.Dropout(0.1), nn.Linear(in_dim, in_dim//2), nn.ReLU(), nn.Linear(in_dim//2, 1))

    def forward(self, data):
        return self.fwd(data).squeeze(-1)


class DeepLearningModel(BaseModel):
    def _train(self, train_features: torch.Tensor, train_labels: torch.Tensor, val_features: torch.Tensor, val_labels: torch.Tensor, *args, verbose=True):
        epochs = 100
        net = Forward(train_features.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-4)
        best = []
        keep_top = 5
        no_improvement = 0
        epoch = 0
        train_index = list(range(train_features.shape[0]))
        running_loss = 0.0
        j = 0
        while epoch < epochs or no_improvement < 20:  # loop over the dataset multiple times
            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            # forward + backward + optimize

            batch_size = 10
            net.train()
            for i in range(0, len(train_index), batch_size):
                optimizer.zero_grad()
                batch_index = train_index[i:i + batch_size]
                outputs = net(train_features[batch_index])
                loss = criterion(outputs, train_labels[batch_index])
                running_loss += loss.item()
                j += 1
                loss.backward()
                optimizer.step()


            net.eval()
            val_out = net(val_features)
            val_loss = criterion(val_out, val_labels)


            if not best or val_loss - best[-1][0] < -0.05:
                best.append((val_loss, copy.deepcopy(net), (val_out - val_labels).squeeze(-1)))
                best = sorted(best, key=lambda x: x[0])
                if not len(best) <= keep_top:
                    del best[-1]
                no_improvement = 0
            else:
                no_improvement += 1
            # print statistics
            if verbose and epoch % 10 == 0:
                print(f"epoch: {epoch},\tloss: {(running_loss / j)},\tval_loss: {val_loss.item()},\tno improvement since: {no_improvement}")
                running_loss = 0.0
                j = 0
            epoch += 1
        self.model = best[0][1]

    @classmethod
    def name(cls):
        return "deep"

    def _predict(self, features: torch.Tensor):
        self.model.eval()
        return self.model(features).squeeze(-1).detach().numpy()

    @classmethod
    def _prepare_single(cls, data):
        return torch.tensor(data).float()

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, "model.ckpt"))