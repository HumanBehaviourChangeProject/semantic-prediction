import csv
import pickle
import numpy as np


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

def write_fuzzy(rename, features, labels):
    with open("data/hbcp_gen.pkl", "wb") as fout:
        pickle.dump((features, np.array(labels), rename), fout)

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
