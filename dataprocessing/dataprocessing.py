import json
import csv
import pickle

import pandas
import pandas as pd
import xlsxwriter
from thefuzz import fuzz
from skfuzzy import cluster as fc
import numpy as np
import re
from fuzzysets import FUZZY_SETS

from cleaner import is_number, clean_attributes, COMBINED_TIME_POINT_ID, ATTRIBUTES_TO_CLEAN
from reader import JSONReader

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
    with open("data/trainAVs.txt", "r") as fin:
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
    attributes = {0: "Combined follow up"}
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


def write_bct_contexts(attributes, cleaned, doc_attrs, doc_name_map, arm_name_map):
    bcts = (
        6452745,
        6452746,
        6452748,
        6452756,
        6452757,
        6452838,
        6452840,
        6452948,
        6452949,
        6452763,
        6452831,
        6452836,
    )
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
                    bct_contexts[attributes[aid]][
                        (doc_name, doc_id, arm_name, arm_id)
                    ] = [context for (_, context) in values]

    with open("/tmp/bct_context.csv", "wt") as fout:
        w = csv.writer(fout)
        for attr, doc_con in bct_contexts.items():
            w.writerow([attr])
            for (doc, doc_id, arm_name, arm_id), contexts in doc_con.items():
                w.writerow(["", doc, doc_id, arm_name, arm_id] + contexts)

    with open("/tmp/outcome_value_contexts.csv", "w") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            ["document", "document_id", "arm", "arm_id"]
            + [b for a in (6451791,) for b in [attributes.get(a, a)]]
            + ["context"]
        )
        # writer.writerow(["", ""] + [b for a in cleaned_attributes for b in [a]])
        for doc_id, arms in cleaned.items():
            for arm_id, arm in arms.items():
                writer.writerow(
                    [doc_name_map[doc_id], doc_id, arm_name_map[arm_id], arm_id]
                    + [
                        x
                        for attribute_id in (6451791,)
                        for vals in (arm.get(attribute_id, "-"),)
                        for x in [vals]
                    ]
                    + [
                        list(
                            doc_attrs[doc_name_map[doc_id] + "___" + doc_id][
                                arm_name_map[arm_id] + "___" + arm_id
                            ].get(6451791, {(None, "-- no node in graph --")})
                        )[0][1]
                    ]
                )


def get_fuzzy_data(attributes, cleaned_attributes, cleaned):
    rename = [
        re.sub("[^A-z]", "_", attributes.get(attribute_id, attribute_id))
        for attribute_id in cleaned_attributes
        if attribute_id not in (6451791, 6080518)
    ]
    features, labels = zip(
        *[
            (
                [
                    [vals]
                    for attribute_id in cleaned_attributes
                    for vals in (arm.get(attribute_id),)
                    if attribute_id not in (6451791, 6080518)
                ],
                arm.get(6451791),
            )
            for doc_id, arms in cleaned.items()
            for arm_id, arm in arms.items()
            if arm.get(6451791)
        ]
    )
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
            fs = FUZZY_SETS.get(rename[dim])
            if fs is not None:
                fuzzy_sets = list(fs.items())
                new_values = np.zeros((values.shape[0], len(fuzzy_sets)))
                new_values[fltr] = np.array(
                    [[fs(v) for _, fs in fuzzy_sets] for v in filtered]
                )
                new_names = [f"{rename[dim]} ({x})" for x in (l for l, _ in fuzzy_sets)]
            else:
                n_centroids = 5
                exfl = np.expand_dims(filtered, axis=-1).T
                cntr, u, u0, d, jm, p, fpc = fc.cmeans(exfl, n_centroids, 2, error=0.005, maxiter=1000, init=None)
                u = u[np.squeeze(cntr, axis=-1).argsort()]
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
    with open("data/hbcp_gen.pkl", "wb") as fout:
        pickle.dump((features, np.array(labels), rename), fout)


def load_deleted():
    d = dict()
    arm_name_map = dict()
    with open("data/cleaned_dataset_13Feb2022_notes_removed_control-2.csv") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        doc_attrs = dict()
        for line in reader:
            row = dict(zip(header, line))

            # if row.pop("remove paper") != "1":
            # del row["Will be fixable in EPPI - will need to update the JSON file"]
            # del row["will need to be fixed via supplementary file merge"]
            manual_follow_up_unit = row.pop(
                "Manually added follow-up duration units", None
            )
            manual_follow_up_value = row.pop(
                "Manually added follow-up duration value", None
            )
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
            cf = (
                float(manual_follow_up_value) * float(follow_up_factor)
                if manual_follow_up_value
                else row.get("Combined follow up")
            )

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
            if not (is_placebo(arm_name.lower()) == (row["placebo"] == "1")):
                print(
                    f"inconsistent placebo (arm name:{arm_name}, placebo: {row['placebo']}"
                )
                if is_placebo(arm_name.lower()):
                    row["placebo"] = is_placebo(arm_name.lower())

            doc_key = f"{document_id}"
            arm_dict = doc_attrs.get(doc_key, dict())
            arm_dict[f"{arm_id}"] = {
                k: ((float(v) if is_number(v) else v) if v is not "-" else None)
                for k, v in row.items()
            }
            doc_attrs[doc_key] = arm_dict
        return doc_attrs, arm_name_map


import seaborn as sb
from matplotlib import pyplot as plt


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


def write_csv(doc_attrs, attribute_name_map):
    with open("features.csv", "w") as fout:
        writer = csv.writer(fout)
        columns = list(
            {
                k
                for _, arms in doc_attrs.items()
                for _, sattr in arms.items()
                for k in sattr.keys()
                if k != 6080518 and k != 6451791
            }
        )
        writer.writerow([attribute_name_map.get(c, c) for c in columns])
        for doc_id, arms in doc_attrs.items():
            for arm_id, sattr in arms.items():
                writer.writerow([*(sattr.get(k,"-") for k in columns)])


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
                diffs.append({"label": "control", "value": c})
                diffs.append({"label": "all", "value": c})
    ax = sb.boxplot(x="label", y="value", data=pd.DataFrame(diffs))
    plt.show()
    ax = sb.scatterplot(data=pd.DataFrame(pairs), x="therapy", y="control")
    plt.savefig("/tmp/scatter.png")


def sub_control(doc_attrs, arm_name_map):
    features = []
    labels = []
    index = []
    columns = list(
        {
            k
            for _, arms in doc_attrs.items()
            for _, sattr in arms.items()
            for k in sattr.keys()
            if k != "Outcome value" and k != "country"
        }
    )
    for doc_id, study, control in get_control(doc_attrs, arm_name_map):
        if control:
            assert len(control) == 1
            c = control[0].get("Outcome value", None)
            if c is not None:
                for arm_id, sattr in study:
                    if sattr["Outcome value"]:
                        if sattr["Outcome value"]:
                            features.append([*(sattr[k] for k in columns)])
                            labels.append((sattr["Outcome value"] + 1) / (c + 1))
                            index.append((doc_id, arm_id))
    idx = pd.MultiIndex.from_tuples(index, names=("doc", "arm"))
    return (
        pandas.DataFrame(features, index=idx),
        pd.DataFrame(labels, index=idx),
        columns,
    )


def default_feature_extraction(doc_attrs, arm_name_map):
    features = []
    labels = []
    index = []
    # columns = [k for k in cleaned_attributes if k != 6080518 and k != 6451791]
    columns = list(
        {
            k
            for _, arms in doc_attrs.items()
            for _, sattr in arms.items()
            for k in sattr.keys()
            if k != "Outcome value" and k != "country"
        }
    )
    column_names = columns  # [printable_attribute_names.get(k, k).replace(" ", "_") for k in columns]
    for doc_id, arms in doc_attrs.items():
        for arm_id, sattr in arms.items():
            if sattr["Outcome value"]:
                features.append([*(sattr[k] for k in columns)])
                labels.append(sattr["Outcome value"])
                index.append((doc_id, arm_id))

    idx = pd.MultiIndex.from_tuples(index, names=("doc", "arm"))
    return (
        pandas.DataFrame(features, index=idx, columns=column_names),
        pd.DataFrame(labels, index=idx),
        column_names,
    )


def load_attributes_from_json():
    json_reader = JSONReader()
    id_map = json_reader.load_id_map(init={COMBINED_TIME_POINT_ID: COMBINED_TIME_POINT_ID})
    prio, prio_names = json_reader.load_prio()
    print("Load attributes")
    attribute_name_map, doc_attrs, prio_names = load_attributes(prio_names, id_map)
    # print("Clean attributes")
    cleaned, doc_name_map, arm_name_map = clean_attributes(doc_attrs, ATTRIBUTES_TO_CLEAN)
    return cleaned, id_map, prio_names, attribute_name_map


def main():
    ds, id_map, prio_names, attribute_name_map = load_attributes_from_json()
    print("Build fuzzy dataset")
    cleaned_removed, arm_name_map = load_deleted()

    # Filter those documents that were manually selected by experts
    # write_csv(ds, attribute_name_map)
    ds = {doc_id: {arm_id: ds[doc_id][arm_id] for arm_id in arms} for doc_id, arms in
     cleaned_removed.items()}
    # features, labels, rename = sub_control(cleaned_removed, arm_name_map)
    features, labels, rename = default_feature_extraction(ds, arm_name_map)

    rename, features, labels = get_extended_fuzzy_data(rename, features, labels)
    write_fuzzy(rename, features, labels)


if __name__ == "__main__":
    main()
