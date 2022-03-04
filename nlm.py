from pathlib import Path
import sys
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
import tqdm

path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))
sys.path.append(os.path.join(path_root, "Jacinle"))
print(sys.path)

from neurallogic.difflogic.nn import neural_logic as nl

class RuleNet(nn.Module):

    def __init__(self, in_dim: int):
        super().__init__()
        self.logic_hidden_dim = in_dim
        self.logic_in_dim = in_dim
        depth = 3
        self.num_rules = 50
        self.logic_out_dim = self.num_rules
        self.logic_machine = nl.LogicMachine(depth, 1, self.logic_in_dim, self.logic_out_dim, self.logic_hidden_dim, residual=True)
        self.rule_weights_1 = nn.Linear(self.logic_in_dim+depth*self.num_rules, 1)
        self.rule_weights_2 = nn.Linear(self.logic_in_dim+depth*self.num_rules, 1)
        self.base = nn.Parameter(torch.rand(1)*10)
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)


    def forward(self, data):
        logic_output = self.logic_machine([torch.zeros(*data.shape),
                                           self.dropout_1(data).unsqueeze(-2)])

        l = self.rule_weights_2(self.dropout_2(logic_output[1].squeeze(1))) + self.base
        return l

def main(epochs, features, labels):
    features[features.isnan()] = 0
    train_index, val_index = train_test_split(range(features.shape[0]), test_size=0.1)
    #test_index, val_index = train_test_split(test_index, test_size=0.25)

    net = RuleNet(features.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    postfix = ""
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients


        # forward + backward + optimize
        j = 0
        batch_size = 10
        for i in tqdm.tqdm(list(range(0, len(train_index), batch_size)), postfix=postfix):
            optimizer.zero_grad()
            batch_index = train_index[i:i + batch_size]
            outputs = net(features[batch_index])
            loss = criterion(outputs, labels[batch_index])
            running_loss += loss.item()
            j += 1
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            net.eval()
            val_out = net(features[val_index])
            val_loss = criterion(val_out, labels[val_index])
            net.train()

        # print statistics
        postfix = f"epoch: {epoch}, loss: {(running_loss/j)**0.5}, val_loss: {val_loss.item()**0.5}"
    print('Finished Training')


if __name__ == "__main__":
    import pickle
    import sys
    with open(sys.argv[2], "rb") as fin:
        features, labels, rename = pickle.load(fin)
    main(int(sys.argv[1]), torch.tensor(features.astype(float).values).float(), torch.tensor(labels.astype(float)).float()
         )