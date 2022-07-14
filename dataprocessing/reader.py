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

def parse_int(textnum, numwords={}):
    # create our default word-lists
    if not numwords:

        # singles
        units = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]

        # tens
        tens = [
            "",
            "",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ]

        # larger scales
        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        # divisors
        numwords["and"] = (1, 0)

        # perform our loops and start the swap
        for idx, word in enumerate(units):
            numwords[word] = (1, idx)
        for idx, word in enumerate(tens):
            numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):
            numwords[word] = (10 ** (idx * 3 or 2), 0)

    # primary loop
    current = result = 0
    # loop while splitting to break into individual words
    for word in textnum.replace("-", " ").lower().split():
        # if problem then fail-safe
        if word not in numwords:
            raise ValueError("Illegal word: " + word)

        # use the index by the multiplier
        scale, increment = numwords[word]
        current = current * scale + increment

        # if larger than 100 then push for a round 2
        if scale > 100:
            result += current
            current = 0

    # return the result plus the current
    return result + current

def load_attributes(prio_names, id_map, combined_time_point=0):
    with open("data/trainAVs.txt", "r") as fin:
        d = json.load(fin)

    doc_attrs = dict()
    attributes_raw = dict()

    tries = 0
    successes = 0
    for av in d:
        try:
            arms = doc_attrs[av["docName"]]
        except KeyError:
            arms = dict()
            doc_attrs[av["docName"]] = arms
        try:
            attrs = arms[av["arm"]]
        except KeyError:
            attrs = dict()
            arms[av["arm"]] = attrs
        raw_id = int(av["attribute"].split("(")[0])
        at = id_map.get(raw_id, raw_id)
        if at not in attributes_raw:
            lpar = av["attribute"].index("(")
            rpar = av["attribute"].rindex(")")
            attributes_raw[at] = av["attribute"][lpar+1:rpar]
        try:
            a = attrs[at]
        except KeyError:
            a = set()
            attrs[at] = a
        v = av["value"]
        if at in NUMERIC_ATTRIBUTES:
            tries += 1
            successes += 1
            v = v.replace(",", "").replace("percent", "").replace("Þ", "fi")
            try:
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                if v.lower() in ("single", "once"):
                    v = 1
                elif v.lower() in ("twice",):
                    v = 2
                elif v.lower() in ("fıve", "fve"):
                    v = 2
                else:
                    try:
                        v = parse_int(v)
                    except ValueError:
                        successes -= 1
            v = str(v)
        a.add((v, av.get("context")))

    dropped_attrs = set()
    attributes = {combined_time_point: "Combined follow up"}
    for k, v in attributes_raw.items():
        if v.strip() in prio_names:
            attributes[k] = v.strip()
        else:
            dropped_attrs.add(v.strip())
    print(dropped_attrs)
    for a in prio_names:
        if a not in attributes.values():
            print("Missing:", a)

    return attributes, doc_attrs, prio_names

NUMERIC_ATTRIBUTES = (
    4099754,
    6451782,
    6451791,
    6080481,
    6080485,
    6080486,
    6080495,
    6080499,
    6080501,
    6080506,
    6080512,
    6080717,
    6080719,
    6080721,
    6080722,
    6080724,
    6099675,
    6823479,
    6823480,
    6823482,
)