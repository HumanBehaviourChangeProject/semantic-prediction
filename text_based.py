import tqdm
from transformers import AutoModel
from torch import nn, optim
import csv
import pandas as pd
from collections import OrderedDict
import json
import copy
import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

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
        self.biobert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2", num_labels=n_classes)
        self.output = nn.Sequential(nn.Linear(46, 512), nn.ReLU(), nn.Linear(512, n_classes))

    def forward(self, data):
        a = self.biobert(data)
        return self.output(a.logits)

def get_features():
    with open("data/cleaned_dataset_13Feb2022_notes_removed_control-2.csv") as fin:
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
            del d["Individual-level analysed"]
            del d["Mean age"]
            del d["Proportion identifying as female gender"]
            del d["Mean number of times tobacco used"]
            del d["Combined follow up"]
            del d["NEW Outcome value"]
            data.append(d)
        df = pd.DataFrame(data)
        df.set_index("arm_id")
    return df

def cross_val(patience, device):
    features = get_features()
    results = []
    index = list(np.array(range(features.shape[0])))
    random.shuffle(index)
    step = len(index) // 10
    chunks = [index[i:i + step] for i in range(0, len(index), step)]
    texts = get_texts()
    features[features == "-"] = 0
    for i in range(len(chunks)):
        train_index = [c for j in range(len(chunks)) for c in chunks[j] if i != j ]
        val_index = chunks[i]
        best = main(patience, list(texts.items()), train_index, val_index, features.astype(float), device)
        torch.save(best[1], "/tmp/text-hbcp.pt")

def get_texts():
    with open("data/All_annotations_512papers_05March20.json", "r") as fin:
        texts = dict()
        for ref in json.load(fin)["References"]:
            featured_arms = {c["ArmId"] for c in ref["Codes"]}
            texts.update({a: {c["AdditionalText"] for c in ref["Codes"] if c["ArmId"] in {0, a}} for a in featured_arms if a != 0})
    return texts



def main(epochs, features, train_index, val_index, labels, device):
    net = FeaturePrediction(labels.shape[1]-1)
    net.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "dmis-lab/biobert-base-cased-v1.2")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    best = []
    keep_top = 5
    no_improvement = 0
    epoch = 0

    inputs = [[torch.tensor(x, device=device) for x in tokenizer(list(y[1])).input_ids] for y in features]
    targed = torch.tensor(labels.iloc[:,1:].values, device=device).float()
    val_loss = None

    while epoch < epochs or no_improvement < 20:  # loop over the dataset multiple times
        running_loss = 0.0
        # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients


        # forward + backward + optimize
        j = 0
        batch_size = 10

        progress = tqdm.tqdm(range(0, len(train_index)))
        for i in progress:
            optimizer.zero_grad()
            batch_index = train_index[i]
            outputs = torch.max(torch.cat([net(sentence[:511].unsqueeze(0)) for sentence in inputs[i]]), dim=0).values
            loss = criterion(outputs, targed[batch_index])
            running_loss += loss.item()
            j += 1
            loss.backward()
            optimizer.step()
            progress.set_postfix_str(f"loss: {running_loss/j}, val_loss: {val_loss}")
        diffs = []
        with torch.no_grad():
            val_loss_sum = 0
            for i in list(val_index):
                outputs = torch.max(torch.cat([net(sentence[:511].unsqueeze(0)) for sentence in inputs[i]]), dim=0).values
                val_loss_sum += criterion(outputs, targed[i]).item()
                diffs.append((outputs-targed[i]).detach().cpu())
            val_loss = val_loss_sum/len(val_index)


        if not best or val_loss - best[-1][0] < -0.05:
            best.append((val_loss, copy.deepcopy(net), diffs))
            best = sorted(best, key=lambda x: x[0])
            if not len(best) <= keep_top:
                del best[-1]
            no_improvement = 0
        else:
            no_improvement += 1
        # print statistics
        if epoch % 10 == 0:
            print(f"epoch: {epoch},\tloss: {(running_loss / j)},\tval_loss: {val_loss},\tno improvement since: {no_improvement}")
            running_loss = 0.0
            j = 0
        epoch += 1

    print('Finished Training')
    return best[0]

if __name__ == "__main__":
    import pickle
    import sys

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    cross_val(int(sys.argv[1]), device)

