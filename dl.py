import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
class Forward(nn.Module):

    def __init__(self, in_dim: int):
        super().__init__()
        self.fwd = nn.Sequential(nn.Linear(in_dim, 2*in_dim), nn.LeakyReLU(), nn.Linear(2*in_dim, in_dim), nn.LeakyReLU(), nn.Dropout(0.1), nn.Linear(in_dim, in_dim//2), nn.ReLU(), nn.Linear(in_dim//2, 1))

    def forward(self, data):
        return self.fwd(data)

def main(epochs, features, labels):
    features[features.isnan()] = 0
    train_index, val_index = train_test_split(range(features.shape[0]), test_size=0.2)
    #test_index, val_index = train_test_split(test_index, test_size=0.25)

    net = Forward(features.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(epochs):  # loop over the dataset multiple times
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

        # print statistics
        print(running_loss/j, val_loss.item())
    print('Finished Training')


if __name__ == "__main__":
    import pickle
    import sys
    with open(sys.argv[2], "rb") as fin:
        features, labels, rename = pickle.load(fin)
    main(int(sys.argv[1]), torch.tensor(features.astype(float)).float(), torch.tensor(labels.astype(float)).float()
         )