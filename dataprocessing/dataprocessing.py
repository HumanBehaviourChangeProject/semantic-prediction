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
from fuzzysets import FUZZY_SETS

COMBINED_TIME_POINT_ID = 0

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


def load_countries_and_cities():
    with open("data/worldcities.csv", "r") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        city_dict = {line[1]: line[4] for line in reader}

    return set(city_dict.values()).union({"USA", "UK", "England"}), city_dict


def rec_attr(cs):
    for attr in cs["Attributes"]["AttributesList"]:
        yield attr["AttributeId"]
        if "Attributes" in attr:
            for r in rec_attr(attr):
                yield r


def load_prio():
    with open("data/prio.txt", "r") as fin:
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
            attributes_raw[at] = av["attribute"].split("(")[1].split(")")[0]
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
            combined_time_point = arm_attributes.get(6451782) or arm_attributes.get(
                6451773
            )
            if combined_time_point:
                arm_attributes[COMBINED_TIME_POINT_ID] = combined_time_point

            for i, attribute_id in enumerate(fixed_attributes):
                worksheet.write(
                    row_counter,
                    4 + i,
                    ";".join(
                        set(
                            x[0]
                            for x in arm_attributes.get(attribute_id, {("-", None)})
                        )
                    ),
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
            values = {
                "bupropion": 0,
                "varenicline": 0,
                "pychologist": 0,
                "doctor": 0,
                "nurse": 0,
            }
            combined_time_point = arm_attributes.get(6451782) or arm_attributes.get(
                6451773
            )
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

    actual_cleaned_attributes = list(
        sorted(
            {
                attribute_id
                for doc_id, arms in cleaned.items()
                for arm_id, attributes in arms.items()
                for attribute_id in attributes
            },
            key=str,
        )
    )
    diff = (
        set(actual_cleaned_attributes)
        .difference(attributes_to_clean)
        .union(set(attributes_to_clean).difference(actual_cleaned_attributes))
    )
    assert not diff, diff

    return cleaned, doc_name_map, arm_name_map


def write_cleaned_attributes(cleaned, attributes_to_clean, doc_name_map, arm_name_map):
    with open("/tmp/cleaned_attributes.csv", "w") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            ["document", "document_id", "arm", "arm_id"]
            + [b for a in attributes_to_clean for b in [attributes_to_clean.get(a, a)]]
        )
        # writer.writerow(["", ""] + [b for a in cleaned_attributes for b in [a]])
        for doc_id, arms in cleaned.items():
            for arm_id, arm in arms.items():
                writer.writerow(
                    [doc_name_map[doc_id], doc_id, arm_name_map[arm_id], arm_id]
                    + [
                        x
                        for attribute_id in attributes_to_clean
                        for vals in (arm.get(attribute_id, "-"),)
                        for x in [vals]
                    ]
                )

    with open("/tmp/cleaned_attributes.json", "w") as fout:
        json.dump(cleaned, fout, indent=2)


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
            fuzzy_sets = list(FUZZY_SETS[rename[dim]].items())
            # n_def apply_fuzzy_setscentroids = 5
            # exfl = np.expand_dims(filtered, axis=-1).T
            # cntr, u, u0, d, jm, p, fpc = fc.cmeans(exfl, n_centroids, 2, error=0.005, maxiter=1000, init=None)
            # u = u[np.squeeze(cntr, axis=-1).argsort()]
            new_values = np.zeros((values.shape[0], len(fuzzy_sets)))
            new_values[fltr] = np.array(
                [[fs(v) for _, fs in fuzzy_sets] for v in filtered]
            )
            new_names = [f"{rename[dim]} ({x})" for x in (l for l, _ in fuzzy_sets)]
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


def write_csv(doc_attrs, attributes):
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


def main():
    # id_map = load_id_map(init={COMBINED_TIME_POINT_ID: COMBINED_TIME_POINT_ID})
    # id_map_reverse = dict(map(reversed, id_map.items()))
    # prio, prio_names = load_prio()
    print("Load attributes")
    # attributes, doc_attrs = load_attributes(prio_names, id_map)

    attributes_to_clean = (
        6451791,
        0,
        6451788,
        6823485,
        6823487,
        "brief advise",
        6452745,
        6452746,
        6452748,
        6452756,
        6452757,
        6452763,
        6452831,
        6452836,
        6452838,
        6452840,
        6452948,
        6452949,
        6080701,
        6080686,
        "phone",
        6080692,
        6080694,
        6080693,
        "gum",
        "e_cigarette",
        "inhaler",
        "lozenge",
        "nasal_spray",
        "placebo",
        "nrt",
        6080695,
        "bupropion",
        "varenicline",
        6080688,
        6080691,
        "text messaging",
        6080704,
        "doctor",
        "nurse",
        "pychologist",
        "aggregate patient role",
        6080481,
        6080485,
        6080512,
        6080518,
        "healthcare facility",
        6830264,
        6830268,
        6080719,
    )
    # print("Clean attributes")
    # cleaned, doc_name_map, arm_name_map = clean_attributes(doc_attrs, attributes_to_clean)
    print("Build fuzzy dataset")
    cleaned_removed, arm_name_map = load_deleted()
    # write_csv(cleaned_removed, attributes)
    # features, labels, rename = sub_control(cleaned_removed, arm_name_map)
    features, labels, rename = default_feature_extraction(cleaned_removed, arm_name_map)

    rename, features, labels = get_extended_fuzzy_data(rename, features, labels)
    write_fuzzy(rename, features, labels)


if __name__ == "__main__":
    main()
