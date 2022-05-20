import json
import csv

class JSONReader:
    def load_prio(self):
        with open("data/prio.txt", "r") as fin:
            prio_names = [l.replace("\n", "") for l in fin]

        with open(
            "data/All_annotations_512papers_05March20.json",
            "r",
        ) as fin:
            d = json.load(fin)
            for cs in d["CodeSets"]:
                if cs["SetName"] == "New Prioritised Codeset":
                    prio = set(self.rec_attr(cs))

        return prio, prio_names

    def load_id_map(self, init=None):
        if init is None:
            id_map = dict()
        else:
            id_map = init

        with open(
            "data/EntityMapping_27May21.csv",
            "r",
        ) as fin:
            r = csv.reader(fin)
            header = next(r)
            for line in r:
                d = dict(zip(header, line))
                if d["Smoking"] != "N/A" and d["Old Codeset"] != "N/A":
                    id_map[int(d["Old Codeset"])] = int(d["Smoking"])
        return id_map

    def rec_attr(self, cs):
        for attr in cs["Attributes"]["AttributesList"]:
            yield attr["AttributeId"]
            if "Attributes" in attr:
                for r in self.rec_attr(attr):
                    yield r