from transformers import AutoModel
from torch import nn, optim
import csv
import pandas as pd
from collections import OrderedDict
import json
import copy
import numpy as np
import random

def is_number(x):
    try:
        float(x)
    except:
        return False
    else:
        return True

class FeaturePrediction(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.biobert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        self.discriminator = nn.Sequential(nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, n_classes))

    def forward(self, data):
        a = self.biobert(data)
        return self.discriminator(a)

def get_features():
    with open("/home/glauer/dev/hbcp/semantic_enhancements/data/cleaned_dataset_13Feb2022_notes_removed_control-2.csv") as fin:
        reader = csv.reader(fin)
        head = next(reader)
        data = []
        for line in reader:
            d = OrderedDict(zip(head, line))
            doc, doc_id, arm, arm_id, *_ = d.values()
            del d["Country of intervention"]
            del d["Abstinence type"]
            del d["Manually added follow-up duration value"]
            del d["Manually added follow-up duration units"]
            del d["document"]
            del d["document_id"]
            del d["arm"]
            data.append(d)
        df = pd.DataFrame(data)
        df.set_index("arm_id")
    return df

def cross_val(patience):
    features = get_features()
    results = []
    index = list(np.array(range(features.shape[0])))
    random.shuffle(index)
    step = len(index) // 10
    chunks = [index[i:i + step] for i in range(0, len(index), step)]
    texts = get_texts()
    for i in range(len(chunks)):
        train_index = [c for j in range(len(chunks)) for c in chunks[j] if i != j ]
        val_index = chunks[i]
        best = main(patience, texts.items(), train_index, val_index, features)

def get_texts():
    with open("/home/glauer/dev/hbcp/Info-extract/core/src/main/resources/data/jsons/All_annotations_512papers_05March20.json", "r") as fin:
        texts = dict()
        for ref in json.load(fin)["References"]:
            featured_arms = {c["ArmId"] for c in ref["Codes"]}
            texts.update({a: {c["AdditionalText"] for c in ref["Codes"] if c["ArmId"] in {0, a}} for a in featured_arms if a != 0})
    return texts



def main(epochs, features, train_index, val_index, labels):
    net = FeaturePrediction(labels.shape[1])
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
            outputs = net([features[i] for i in batch_index])
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

    cross_val(int(sys.argv[1]))

