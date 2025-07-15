import csv
import pickle
import numpy as np
import pandas as pd

def write_csv(doc_attrs:pd.DataFrame):
    header = []
    for a,b in doc_attrs.columns:
        if a == b:
            header.append(a)
        else:
            header.append(f"{b}({a})")
    doc_attrs.to_csv("data/model_input_data.csv", header=header, index_label=("document_id", "arm_id", "document", "arm"), na_rep="-")


def write_fuzzy(features, labels):
    with open("data/hbcp_gen.pkl", "wb") as fout:
        pickle.dump((features, np.array(labels)), fout)
    with open("data/hbcp_gen.csv", "w", encoding="utf8") as fout:
        full_frame = pd.concat((features, pd.DataFrame(np.array(labels), columns=(("Outcome","Outcome"),), index=features.index)), axis=1).reset_index(names=["doc_id", "arm_id", "doc", "arm"])
        full_frame.to_csv(fout)


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

    with open("/tmp/bct_context.csv", "wt", encoding="utf8") as fout:
        w = csv.writer(fout)
        for attr, doc_con in bct_contexts.items():
            w.writerow([attr])
            for (doc, doc_id, arm_name, arm_id), contexts in doc_con.items():
                w.writerow(["", doc, doc_id, arm_name, arm_id] + contexts)

    with open("/tmp/outcome_value_contexts.csv", "w", encoding="utf8") as fout:
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
