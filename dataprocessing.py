import json
import csv
import pickle
import random

import numpy
import pandas
import pandas as pd
import xlsxwriter
from thefuzz import process as fw_process, fuzz
from skfuzzy import cluster as fc
import numpy as np
import re

COMBINED_TIME_POINT_ID = 0

class FuzzySet:
    def __init__(self, l0, l, r, r0):
        self.l0 = l0
        self.l = l
        self.r = r
        self.r0 = r0

    def __call__(self, v):
        if v < self.l0:
            return 0
        elif v < self.l:
            return (v-self.l0)/(self.l-self.l0)
        elif v < self.r:
            return 1
        elif v < self.r0:
            return (v - self.r0) / (self.r - self.r0)
        else:
            return 0

def load_countries_and_cities():
    with open("data/worldcities.csv", "r") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        city_dict = {line[1]: line[4] for line in reader}

    return set(city_dict.values()).union({"USA", "UK", "England"}), city_dict


def use_rounded(ident, data):
    v = None
    if ident in data:
        for i in reversed(range(0, 3)):
            s = set(round(float(x), i) for x, _ in data[ident] if _clean(x) not in ("no", "none") and is_number(x))
            if len(s) == 1:
                return {ident: s.pop()}
    if v:
        return {ident: v}
    else:
        return dict()


def any_as_presence(ident, data):
    x = data.get(ident, False)
    if x:
        v = 1
    else:
        v = 0
    return {ident: v}


def _clean(x):
    return x.replace(",", "").replace(";", "").replace("-", "").replace(" ", "").lower()


def _process_with_key_list(ident, keys, data, initial_dictionary=None, threshold=90, negative=False):
    if initial_dictionary:
        d = dict(initial_dictionary)
    else:
        d = dict()
    for x, _ in data.get(ident, tuple()):
        for key, patterns in keys.items():
            match = fw_process.extract(x, patterns)
            if match[0][1] >= threshold:
                d[key] = 1 if not negative else 0
    return d


def process_motivational_interviewing(ident, data):
    v = 0
    brief = 0
    for x, _ in data.get(ident, tuple()):
        match = fw_process.extract(x.lower(), ("brief advice", "ba"))
        if match[0][1] >= 80:
            brief = 1
        else:
            v = 1

    return {ident: v, "brief advise": brief}


def process_digital_content(ident, data):
    v = 0
    text_message = 0
    for x, _ in data.get(ident, tuple()):
        match = fw_process.extract(x.lower(), ("text", "text message"))
        if match[0][1] >= 80:
            text_message = 1
        else:
            v = 1

    return {ident: v, "text messaging": text_message}


def process_distance(ident, data):
    keys = ("phone", "call", "telephone", "quitline", "hotline")
    phone = 0
    for x, _ in data.get(ident, tuple()):
        match = fw_process.extract(x, keys)
        if match[0][1] >= 90:
            phone = 1
    return {"phone": phone}


def process_somatic(ident, data):
    keys = {"gum": ["gum", "polacrilex"], "lozenge": ["lozenge"], "e_cigarette": ["ecig", "ecigarette"],
            "inhaler": ["inhaler", "inhal"], "placebo": ["placebo"],
            "varenicline": ["varenicline", "varen", "chantix", "champix"], "nasal_spray": ["nasal"],
            "rimonabant": ["rimonab"], "nrt": ["nicotine", "nrt"], "bupropion": ["bupropion"]}
    d = dict(gum=0, lozenge=0, e_cigarette=0, inhaler=0, placebo=0, nasal_spray=0, nrt=0)
    d.update(any_as_presence(ident, data))
    d = _process_with_key_list(ident, keys, data, initial_dictionary=d)
    patch = any_as_presence(6080694, data)[6080694]
    if d["gum"] or d["lozenge"] or d["e_cigarette"] or patch or d["inhaler"]:
        d["nrt"] = 1
    return d


def process_pill(ident, data):
    keys = {"bupropion": (
    ["bupropion"], True, any(True for x, _ in data.get(6080693, tuple()) if _clean(x) in ["bupropion"])),
            "nortriptyline": (["nortript"], True, False),
            "varenicline": (["varenicline", "varen", "chantix", "champix"], True, False)}
    d = any_as_presence(ident, data)
    for x, _ in data.get(ident, tuple()):
        for key, (patterns, ands, ors) in keys.items():
            match = fw_process.extract(x, patterns)
            if (match[0][1] > 90 or ors) and ands:
                d[key] = 1
    return d


def process_health_professional(ident, data):
    keys = {"nurse": ["nurse"],
            "doctor": ["physician", "doctor", "physician", "cardiologist", "pediatrician", "general pract", "GP",
                       "resident", "internal medicine"]}
    d = any_as_presence(ident, data)
    d = _process_with_key_list(ident, keys, data, initial_dictionary=d)
    return d


def process_pychologist(ident, data):
    keys = {"pychologist": ["pychologist", "psychol"]}
    d = _process_with_key_list(ident, keys, data)
    return d


def process_aggregate_patient_role(ident, data):
    keys = {"aggregate patient role": ["patient"]}
    d = {"aggregate patient role": 0}
    d = _process_with_key_list(ident, keys, data, initial_dictionary=d, threshold=80)
    return d


def process_healthcare_facility(ident, data):
    keys = {"healthcare facility": ["smok"]}
    d = _process_with_key_list(ident, keys, data,
                               initial_dictionary={"healthcare facility": 1 if data.get(ident, tuple()) else 0},
                               negative=True)
    return d


def is_number(x):
    if x is None:
        return False
    try:
        float(x)
        return True
    except ValueError:
        return False


def process_proportion_female(ident, data):
    value = None
    females = use_rounded(ident, {ident: {x for x in data.get(ident, set()) if is_number(x[0])}}).get(ident, None)
    if females:
        value = females
    else:
        males = {float(x[0]) for x in data.get(6080486, set()) if is_number(x[0])}
        if males:
            value = 100 - float(males.pop())
    if value:
        return {ident: value}
    else:
        return dict()


def process_pharmacological_interest(ident, data):
    return {ident: any_as_presence(ident, data).get(ident, 0) or any_as_presence(6830264, data).get(6830264, 0)}


def build_process_country():
    countries, city_dict = load_countries_and_cities()

    def inner(ident, data):
        countries_values = data.get(ident, set())
        if countries_values:
            countries_values = [x for x, _ in countries_values]
            if len(countries_values) > 1:
                value = "multinational"
            else:
                match, quality = fw_process.extract(countries_values[0], countries, scorer=fuzz.ratio)[0]
                if quality > 80:
                    value = match
                else:
                    value = countries_values[0]

        else:
            regions = data.get(6080519, set())
            value = [city_dict[x] for x, _ in regions if x in city_dict]
            if value:
                value = value[0]
            else:
                return dict()
        return {ident: value}

    return inner


mappings = {
    6451791: use_rounded,
    0: use_rounded,
    6451788: any_as_presence,
    6823485: any_as_presence,
    6823487: process_motivational_interviewing,
    6452745: any_as_presence, 6452746: any_as_presence, 6452748: any_as_presence, 6452756: any_as_presence,
    6452757: any_as_presence, 6452838: any_as_presence, 6452840: any_as_presence, 6452948: any_as_presence,
    6452949: any_as_presence, 6452763: any_as_presence, 6452831: any_as_presence, 6452836: any_as_presence,
    6080701: any_as_presence,
    6080686: any_as_presence,
    6080687: process_distance,
    6080692: any_as_presence,
    6080694: any_as_presence,
    6080693: process_somatic,
    6080688: any_as_presence,
    6080691: process_digital_content,
    6080695: process_pill,
    6080704: process_health_professional,
    6080706: process_pychologist,
    6080508: process_aggregate_patient_role,
    6080481: use_rounded,
    6080485: process_proportion_female,
    6080512: use_rounded,
    6080518: build_process_country(),  # Country of intervention
    6080629: process_healthcare_facility,
    6830264: any_as_presence,
    6830268: process_pharmacological_interest,
    6080719: use_rounded,
}


def rec_attr(cs):
    for attr in cs["Attributes"]["AttributesList"]:
        yield attr["AttributeId"]
        if "Attributes" in attr:
            for r in rec_attr(attr):
                yield r


def load_prio():
    with open(
            "data/prio.txt", "r"
    ) as fin:
        prio_names = [l.replace("\n", "") for l in fin]

    with open(
            "data/All_annotations_512papers_05March20.json",
            "r",
    ) as fin:
        d = json.load(fin)
        for cs in d["CodeSets"]:
            if cs["SetName"] == "New Prioritised Codeset":
                prio = set(rec_attr(cs))

    return prio, prio_names


def load_id_map(init=None):
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


numeric_attributes = (
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


def load_attributes(prio_names, id_map):
    with open(
            "data/trainAVs.txt", "r"
    ) as fin:
        d = json.load(fin)

    doc_attrs = dict()
    # attributes = {int(l.split("(")[0]): l.split("(")[1].split(")")[0] for l in open("/home/glauer/.PyCharm2019.3/config/scratches/hbcp/scratch_3.txt")}
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
            attributes_raw[at] = av["attribute"].split("(")[1].split(")")[0]
        try:
            a = attrs[at]
        except KeyError:
            a = set()
            attrs[at] = a
        v = av["value"]
        if at in numeric_attributes:
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

    attributes = {0: "Combined follow up"}
    for k, v in attributes_raw.items():
        if v.strip() in prio_names:
            attributes[k] = v.strip()

    for a in prio_names:
        if a not in attributes.values():
            print("Missing:", a)

    return attributes, doc_attrs


def write_attribute_values(attributes, doc_attrs, prio, id_map, id_map_reverse):
    workbook = xlsxwriter.Workbook("/tmp/attribute_values_na.xlsx")
    worksheet = workbook.add_worksheet()
    fixed_attributes = list(
        sorted(
            [a for a in attributes.keys() if id_map.get(a, a) in prio]
            + [COMBINED_TIME_POINT_ID],
            key=lambda x: id_map_reverse.get(x, x),
        )
    )
    for i, aid in enumerate(fixed_attributes):
        worksheet.write(0, i + 4, attributes[aid])

    for i, aid in enumerate(fixed_attributes):
        worksheet.write(1, i + 4, id_map_reverse.get(aid, aid))

    for i, aid in enumerate(fixed_attributes):
        worksheet.write(2, i + 4, id_map.get(aid, aid))

    row_counter = 3

    for doc, arms in sorted(doc_attrs.items()):
        # if len(arms) > 1:
        #    worksheet.merge_range(f'A{row_counter+1}:A{row_counter + len(arms)}', doc)
        # else:
        #    worksheet.write(row_counter, 0, doc)
        for arm, arm_attributes in arms.items():
            doc_name, doc_id = doc.split("___")
            arm_name, arm_id = arm.split("___")
            worksheet.write(row_counter, 0, doc_name)
            worksheet.write(row_counter, 1, doc_id)
            worksheet.write(row_counter, 2, arm_name)
            worksheet.write(row_counter, 3, arm_id)
            combined_time_point = arm_attributes.get(6451782) or arm_attributes.get(6451773)
            if combined_time_point:
                arm_attributes[COMBINED_TIME_POINT_ID] = combined_time_point

            for i, attribute_id in enumerate(fixed_attributes):
                worksheet.write(
                    row_counter, 4 + i, ";".join(set(x[0] for x in arm_attributes.get(attribute_id, {("-", None)})))
                )
            row_counter += 1

    workbook.close()


def clean_attributes(doc_attrs, attributes_to_clean):
    cleaned = dict()

    doc_name_map = dict()
    arm_name_map = dict()
    for doc, arms in sorted(doc_attrs.items()):
        doc_name, doc_id = doc.split("___")
        doc_name_map[doc_id] = doc_name
        for arm, arm_attributes in arms.items():
            arm_name, arm_id = arm.split("___")
            arm_name_map[arm_id] = arm_name
            values = {"bupropion": 0, "varenicline": 0, "pychologist": 0, "doctor": 0, "nurse": 0}
            combined_time_point = arm_attributes.get(6451782) or arm_attributes.get(6451773)
            if combined_time_point:
                arm_attributes[COMBINED_TIME_POINT_ID] = combined_time_point
            for attribute_id, mapping in mappings.items():
                mapped = mapping(attribute_id, arm_attributes)
                mapped_with_context = {k: mapped.get(k, 0) for k in mapped}
                values.update(mapped_with_context)
            try:
                doc_arms = cleaned[doc_id]
            except KeyError:
                cleaned[doc_id] = doc_arms = dict()
            doc_arms[arm_id] = values

    actual_cleaned_attributes = list(sorted({attribute_id for doc_id, arms in cleaned.items()
                                             for arm_id, attributes in arms.items()
                                             for attribute_id in attributes}, key=str))
    diff = set(actual_cleaned_attributes).difference(attributes_to_clean).union(
        set(attributes_to_clean).difference(actual_cleaned_attributes))
    assert not diff, diff

    return cleaned, doc_name_map, arm_name_map


def write_cleaned_attributes(cleaned, attributes_to_clean, doc_name_map, arm_name_map):
    with open("/tmp/cleaned_attributes.csv", "w") as fout:
        writer = csv.writer(fout)
        writer.writerow(["document", "document_id", "arm", "arm_id"] + [b for a in attributes_to_clean for b in
                                                                        [attributes_to_clean.get(a, a)]])
        # writer.writerow(["", ""] + [b for a in cleaned_attributes for b in [a]])
        for doc_id, arms in cleaned.items():
            for arm_id, arm in arms.items():
                writer.writerow([doc_name_map[doc_id], doc_id, arm_name_map[arm_id], arm_id] + [x for attribute_id in
                                                                                                attributes_to_clean for
                                                                                                vals in (
                                                                                                arm.get(attribute_id,
                                                                                                        "-"),) for x in
                                                                                                [vals]])

    with open("/tmp/cleaned_attributes.json", "w") as fout:
        json.dump(cleaned, fout, indent=2)


def write_bct_contexts(attributes, cleaned, doc_attrs, doc_name_map, arm_name_map):
    bcts = (6452745, 6452746, 6452748, 6452756, 6452757, 6452838, 6452840, 6452948, 6452949, 6452763, 6452831, 6452836)
    bct_contexts = {attributes[i]: dict() for i in bcts}

    for doc, arms in sorted(doc_attrs.items()):
        # if len(arms) > 1:
        #    worksheet.merge_range(f'A{row_counter+1}:A{row_counter + len(arms)}', doc)
        # else:
        #    worksheet.write(row_counter, 0, doc)
        for arm, arm_attributes in arms.items():
            doc_name, doc_id = doc.split("___")
            arm_name, arm_id = arm.split("___")
            for aid in bcts:
                try:
                    values = arm_attributes[aid]
                except KeyError:
                    pass
                else:
                    bct_contexts[attributes[aid]][(doc_name, doc_id, arm_name, arm_id)] = [context for (_, context) in
                                                                                           values]

    with open("/tmp/bct_context.csv", "wt") as fout:
        w = csv.writer(fout)
        for attr, doc_con in bct_contexts.items():
            w.writerow([attr])
            for (doc, doc_id, arm_name, arm_id), contexts in doc_con.items():
                w.writerow(["", doc, doc_id, arm_name, arm_id] + contexts)

    with open("/tmp/outcome_value_contexts.csv", "w") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            ["document", "document_id", "arm", "arm_id"] + [b for a in (6451791,) for b in [attributes.get(a, a)]] + [
                "context"])
        # writer.writerow(["", ""] + [b for a in cleaned_attributes for b in [a]])
        for doc_id, arms in cleaned.items():
            for arm_id, arm in arms.items():
                writer.writerow(
                    [doc_name_map[doc_id], doc_id, arm_name_map[arm_id], arm_id] + [x for attribute_id in (6451791,) for
                                                                                    vals in
                                                                                    (arm.get(attribute_id, "-"),) for x
                                                                                    in [vals]] + [list(
                        doc_attrs[doc_name_map[doc_id] + "___" + doc_id][arm_name_map[arm_id] + "___" + arm_id].get(
                            6451791, {(None, "-- no node in graph --")}))[0][1]])


def get_fuzzy_data(attributes, cleaned_attributes, cleaned):
    rename = [re.sub("[^A-z]", "_", attributes.get(attribute_id, attribute_id)) for attribute_id in cleaned_attributes
              if attribute_id not in (6451791, 6080518)]
    features, labels = zip(*[([[vals] for attribute_id in cleaned_attributes for vals in (arm.get(attribute_id),) if
                               attribute_id not in (6451791, 6080518)], arm.get(6451791)) for doc_id, arms in
                             cleaned.items() for arm_id, arm in arms.items() if arm.get(6451791)])
    features = np.squeeze(np.array(features), axis=-1)
    return rename, features, labels

def get_extended_fuzzy_data(rename, features, labels):
    inflated_data = dict()
    names = []
    for dim in range(features.shape[1]):
        values = features.iloc[:, dim]
        fltr = values.notnull()
        filtered = values[fltr].astype(float)
        if not np.max(filtered) <= 1:
            fuzzy_sets = list(FUZZY_SETS[rename[dim]].items())
            #n_def apply_fuzzy_setscentroids = 5
            #exfl = np.expand_dims(filtered, axis=-1).T
            #cntr, u, u0, d, jm, p, fpc = fc.cmeans(exfl, n_centroids, 2, error=0.005, maxiter=1000, init=None)
            #u = u[np.squeeze(cntr, axis=-1).argsort()]
            new_values = np.zeros((values.shape[0], len(fuzzy_sets)))
            new_values[fltr] = np.array([[fs(v) for _, fs in fuzzy_sets] for v in filtered])
            new_names = [f"{rename[dim]} ({x})" for x in (l for l,_ in fuzzy_sets)]
        else:
            new_values = np.expand_dims(features.iloc[:, dim], axis=-1)
            new_names = [rename[dim]]
        for n, v in zip(new_names, new_values.T):
            inflated_data[n] = v
        names += new_names
    # assert 0 <= np.min(inflated_data) and np.max(inflated_data) <= 1, (np.min(inflated_data), np.max(inflated_data))
    new_features = pd.DataFrame(inflated_data, index=features.index)
    return names, new_features, labels


def write_fuzzy(rename, features, labels):
    with open("/tmp/hbcp_gen.pkl", "wb") as fout:
        pickle.dump((features, np.array(labels), rename), fout)


def load_deleted():
    d = dict()
    arm_name_map = dict()
    with open(
            "data/cleaned_dataset_13Feb2022_notes_removed_control-2.csv") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        doc_attrs = dict()
        for line in reader:
            row = dict(zip(header, line))

            # if row.pop("remove paper") != "1":
            # del row["Will be fixable in EPPI - will need to update the JSON file"]
            # del row["will need to be fixed via supplementary file merge"]
            manual_follow_up_unit = row.pop("Manually added follow-up duration units", None)
            manual_follow_up_value = row.pop("Manually added follow-up duration value", None)
            if manual_follow_up_value == "na":
                manual_follow_up_value = None
            if manual_follow_up_unit == "days":
                follow_up_factor = 1 / 7
            elif manual_follow_up_unit == "weeks":
                follow_up_factor = 1
            elif manual_follow_up_unit == "months":
                follow_up_factor = 4.35
            elif manual_follow_up_unit in ["year", "years"]:
                follow_up_factor = 4.35 * 12
            else:
                if manual_follow_up_value:
                    raise Exception(f"{manual_follow_up_value} {manual_follow_up_unit}")

            row["Outcome value"] = row.pop("NEW Outcome value")
            cf = float(manual_follow_up_value) * float(follow_up_factor) if manual_follow_up_value else row.get(
                "Combined follow up")

            try:
                float(cf)
            except ValueError:
                row["Combined follow up"] = None
            else:
                row["Combined follow up"] = cf
            arm_name = row.pop("arm")
            # print(1 if is_control(arm_name.lower()) else 0)
            arm_id = row.pop("arm_id")
            arm_name_map[arm_id] = arm_name
            document_name = row.pop("document")
            document_id = row.pop("document_id")
            del row["Abstinence type"]
            del row["Country of intervention"]
            if not (is_placebo(arm_name.lower()) == (row["placebo"] == '1')):
                print(f"inconsistent placebo (arm name:{arm_name}, placebo: {row['placebo']}")
                if is_placebo(arm_name.lower()):
                    row["placebo"] = is_placebo(arm_name.lower())


            doc_key = f"{document_id}"
            arm_dict = doc_attrs.get(doc_key, dict())
            arm_dict[f"{arm_id}"] = {k: ((float(v) if is_number(v) else v) if v is not "-" else None) for k, v in
                                     row.items()}
            doc_attrs[doc_key] = arm_dict
        return doc_attrs, arm_name_map


import seaborn as sb
from matplotlib import pyplot as plt


def is_control(v):
    return max(fuzz.partial_ratio(v, "control"),
               fuzz.partial_ratio(v, "standard treatment"),
               fuzz.partial_ratio(v, "usual care"),
               fuzz.partial_ratio(v, "standard care"),
               fuzz.partial_ratio(v, "normal care"),
               fuzz.partial_ratio(v, "comparison")) > 80

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

            if arm_attributes["control"] == 1 or is_placebo(arm_name_map[arm_id].lower()):
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


def write_csv(doc_attrs, attributes):
    with open("features.csv", "w") as fout:
        writer = csv.writer(fout)
        columns = list({k for _, arms in doc_attrs.items() for _, sattr in arms.items() for k in sattr.keys() if
                        k != 6080518 and k != 6451791})
        writer.writerow(columns)
        for doc_id, arms in doc_attrs.items():
            for arm_id, sattr in arms.items():
                writer.writerow([*(sattr[k] for k in columns)])


def plot_control(doc_attrs, arm_name_map):
    diffs = []
    pairs = []
    for doc_id, arm_id, study, control in get_control(doc_attrs, arm_name_map):
        if control:
            assert len(control) == 1
            c = control[0].get(6451791, None)
            if c is not None:
                for sattr in study:
                    s = sattr.get(6451791, None)
                    diffs.append({"label": "diff", "value": s - c})
                    diffs.append({"label": "therapy", "value": np.average(s)})
                    diffs.append({"label": "all", "value": np.average(s)})
                    pairs.append(dict(therapy=s, control=c))
                diffs.append({"label": "control",
                              "value": c})
                diffs.append({"label": "all",
                              "value": c})
    ax = sb.boxplot(x="label", y="value", data=pd.DataFrame(diffs))
    plt.show()
    ax = sb.scatterplot(data=pd.DataFrame(pairs), x="therapy", y="control")
    plt.savefig("/tmp/scatter.png")


def sub_control(doc_attrs, arm_name_map):
    features = []
    labels = []
    index = []
    columns = list({k for _, arms in doc_attrs.items() for _, sattr in arms.items() for k in sattr.keys() if
                    k != "Outcome value" and k != "country"})
    for doc_id, study, control in get_control(doc_attrs, arm_name_map):
        if control:
            assert len(control) == 1
            c = control[0].get("Outcome value", None)
            if c is not None:
                for arm_id, sattr in study:
                    if sattr["Outcome value"]:
                        if sattr["Outcome value"]:
                            features.append([*(sattr[k] for k in columns)])
                            labels.append((sattr["Outcome value"]+1)/(c+1))
                            index.append((doc_id, arm_id))
    idx = pd.MultiIndex.from_tuples(index, names=("doc", "arm"))
    return pandas.DataFrame(features, index=idx), pd.DataFrame(labels, index=idx), columns


def default_feature_extraction(doc_attrs, arm_name_map):
    features = []
    labels = []
    index = []
    # columns = [k for k in cleaned_attributes if k != 6080518 and k != 6451791]
    columns = list({k for _, arms in doc_attrs.items() for _, sattr in arms.items() for k in sattr.keys() if
                    k != "Outcome value" and k != "country"})
    column_names = columns # [printable_attribute_names.get(k, k).replace(" ", "_") for k in columns]
    for doc_id, arms in doc_attrs.items():
        for arm_id, sattr in arms.items():
            if sattr["Outcome value"]:
                features.append([*(sattr[k] for k in columns)])
                labels.append(sattr["Outcome value"])
                index.append((doc_id, arm_id))

    idx = pd.MultiIndex.from_tuples(index, names=("doc", "arm"))
    return pandas.DataFrame(features, index=idx, columns=column_names), pd.DataFrame(labels, index=idx), column_names


FUZZY_SETS = {"Mean age": {"child": FuzzySet(0, 0, 14, 18),
                           "adolecent": FuzzySet(14, 18, 23, 27),
                           "adult": FuzzySet(23, 27, 60, 65),
                           "elderly": FuzzySet(60, 65, 100, 100)},
              "Individual-level analysed": {"less than 50": FuzzySet(0, 0, 50, 60),
                                            "less than 100": FuzzySet(50, 60, 100, 120),
                                            "less than 500": FuzzySet(100, 110, 500, 600),
                                            "less than 1000": FuzzySet(500, 600, 1000, 1200),
                                            "less than 5000": FuzzySet(1000, 1200, 5000, 6000),
                                            "less than 10000": FuzzySet(5000, 6000, 10000, 12000),
                                            "more than 10000": FuzzySet(10000, 12000, 100000, 100000)},
              "Combined follow up": {"one week": FuzzySet(0, 0, 1, 2),
                                     "less than a month": FuzzySet(1, 2, 3, 4),
                                     "1-5 months": FuzzySet(3, 4, 5 * 4.35, 6 * 4.35),
                                     "6-11 month": FuzzySet(5 * 4.35, 6 * 4.35, 11 * 4.35, 12 * 4.35),
                                     "12-18 months": FuzzySet(11 * 4.35, 12 * 4.35, 18 * 4.35, 19 * 4.35),
                                     "19-23 months": FuzzySet(18 * 4.35, 19 * 4.35, 23 * 4.35, 24 * 4.35),
                                     "2 years+": FuzzySet(23 * 4.35, 24 * 4.35, 120 * 4.35, 120 * 4.35)},
              "Mean number of times tobacco used": {
                    "less than 5": FuzzySet(0, 0, 5, 6),
                    "less than 10": FuzzySet(5, 6, 10, 11),
                    "less than 15": FuzzySet(10, 11, 15, 16),
                    "less than 20": FuzzySet(15, 16, 20, 21),
                    "less than 25": FuzzySet(20, 21, 25, 26),
                    "more than 25": FuzzySet(25, 26, 100, 100),
              },
              "Proportion identifying as female gender":{
                "no": FuzzySet(0, 0, 5, 10),
                "very few": FuzzySet(5, 10, 20, 25),
                "few": FuzzySet(20, 25, 40, 45),
                "equal": FuzzySet(40, 45, 55, 60),
                "more": FuzzySet(55, 60, 75, 80),
                "many more": FuzzySet(75, 80, 90, 95),
                "all": FuzzySet(90, 95, 100, 100),
              }
          }


def main():
    #id_map = load_id_map(init={COMBINED_TIME_POINT_ID: COMBINED_TIME_POINT_ID})
    #id_map_reverse = dict(map(reversed, id_map.items()))
    #prio, prio_names = load_prio()
    print("Load attributes")
    #attributes, doc_attrs = load_attributes(prio_names, id_map)

    attributes_to_clean = (
        6451791, 0, 6451788, 6823485, 6823487, "brief advise", 6452745, 6452746, 6452748, 6452756, 6452757, 6452763,
        6452831, 6452836, 6452838, 6452840, 6452948, 6452949, 6080701, 6080686, "phone", 6080692, 6080694, 6080693,
        "gum",
        "e_cigarette", "inhaler", "lozenge", "nasal_spray", "placebo", "nrt", 6080695, "bupropion", "varenicline",
        6080688,
        6080691, "text messaging", 6080704, "doctor", "nurse", "pychologist", "aggregate patient role", 6080481,
        6080485,
        6080512, 6080518, "healthcare facility", 6830264, 6830268, 6080719)
    #print("Clean attributes")
    #cleaned, doc_name_map, arm_name_map = clean_attributes(doc_attrs, attributes_to_clean)
    print("Build fuzzy dataset")
    cleaned_removed, arm_name_map = load_deleted()
    #write_csv(cleaned_removed, attributes)
    #features, labels, rename = sub_control(cleaned_removed, arm_name_map)
    features, labels, rename = default_feature_extraction(cleaned_removed, arm_name_map)

    rename, features, labels = get_extended_fuzzy_data(rename, features, labels)
    write_fuzzy(rename, features, labels)


if __name__ == "__main__":
    main()
