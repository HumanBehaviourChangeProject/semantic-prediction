import json
import csv
import pandas as pd
from abc import ABC, abstractmethod
from cleaner import clean_row, get_id, get_name, Differ, Dropper
from thefuzz import fuzz


class Reader(ABC):
    @abstractmethod
    def read(self, *args) -> pd.DataFrame:
        raise NotImplementedError


class BaseReader(ABC):
    def load_id_map(self, init=None):
        """
        There has been a transition to a prioritised code set at some point.
        This caused some inconsistencies in the way attributes are identified.
        Some use the old code ids, some the new one. The purpose of this
        method is to create a mapping from 'old' ids to the ones used
        in the prioritised codeset.
        :param init: Intial mappings. Initial value for the mapping
        :return: A int -> int dictionary that maps ids
        """
        if init is None:
            id_map = dict()
        else:
            id_map = init

        with open(
            "data/EntityMapping_27May21.csv",
            "r", encoding="utf8"
        ) as fin:
            r = csv.reader(fin)
            header = next(r)
            for line in r:
                d = dict(zip(header, line))
                if d["Smoking"] != "N/A" and d["Old Codeset"] != "N/A":
                    id_map[int(d["Old Codeset"])] = int(d["Smoking"])
        return id_map

    def _read(self):
        with open("data/trainAVs.txt", "r", encoding="utf8") as fin:
            d = json.load(fin)

        doc_attrs = dict()
        attributes_raw = dict()

        tries = 0
        successes = 0

        id_map = self.load_id_map()
        prio_ids, prio_names = self.load_prio()

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
                attributes_raw[at] = av["attribute"][lpar + 1 : rpar]
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

        return doc_attrs, {k:v.strip() for k,v in attributes_raw.items()}

    def load_prio(self):
        with open("data/prio.txt", "r", encoding="utf8") as fin:
            prio_names = [l.replace("\n", "") for l in fin]

        with open(
            "data/All_annotations_512papers_05March20.json",
            "r", encoding="utf8"
        ) as fin:
            d = json.load(fin)
            for cs in d["CodeSets"]:
                if cs["SetName"] == "New Prioritised Codeset":
                    prio = set(self.rec_attr(cs))

        return prio, prio_names

class AttributeReader(BaseReader):
    def read(self, *args) -> pd.DataFrame:
        doc_attrs, attribute_names = self._read()
        differ = Differ()
        dropper = Dropper()
        d = {
            (get_id(a), get_id(b), get_name(a), get_name(b)): {
                (k, attribute_names.get(k, k)): v for k, v in clean_row(attrs, differ.get_diff(get_id(a),get_id(b)), get_name(b)).items()
            }
            for a, arms in doc_attrs.items()
            if not dropper.should_be_dropped(get_id(a))
            for b, attrs in arms.items()
        }
        return pd.DataFrame(d).T

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


def is_control(v):
    return (
        max(
            fuzz.partial_ratio(v, "control"),
            fuzz.partial_ratio(v, "standard treatment"),
            fuzz.partial_ratio(v, "usual care"),
            fuzz.partial_ratio(v, "standard care"),
            fuzz.partial_ratio(v, "normal care"),
            fuzz.partial_ratio(v, "comparison"),
        )
        > 80
    )


def is_placebo(v):
    return fuzz.partial_ratio(v, "placebo") > 80


def get_control(doc_attrs, arm_name_map):
    valids = 0
    num_arms = 0
    for doc_id, arms in sorted(doc_attrs.items()):
        control = []
        study = []
        for arm_id, arm_attributes in arms.items():
            num_arms += 1

            if arm_attributes["control"] == 1 or is_placebo(
                arm_name_map[arm_id].lower()
            ):
                control.append(arm_attributes)
            else:
                study.append((arm_id, arm_attributes))
        if not control:
            print("No control found", [arm_name_map[a] for a in arms])
        if len(control) <= 1:
            yield doc_id, study, control
            valids += 1
        else:
            print("Too many control groups:", [arm_name_map[a] for a in arms])

    print(f"Valid arms: {valids} out of {num_arms}")


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
