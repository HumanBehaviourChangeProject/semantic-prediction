from fpgrowth_py import fpgrowth

with open("results/rules.txt", "r") as fin:

    rulessets = []
    for line in fin:
        rules=[]
        for rule in line.split(";"):
            *conjunctions, weight = rule.split("&")
            conjunctions = [float(c) > 0.5 for c in conjunctions]
            rules.append((conjunctions, float(weight)))
        rulessets.append(rules)

with open("results/frequents.txt", "w") as fout:

    items = list(range(len(rulessets[0][0])))
    transactions = [{i for i, v in enumerate(con) if v} for rule in rulessets for con, _ in rule]
    initial_item_sets = [frozenset({x}) for x in items]

    tres = 100
    current = [i for i,l in ((i, len([t for t in transactions if i.issubset(t)] )) for i in initial_item_sets) if l > tres]
    while current:
        nxt = [(i,l) for i,l in ((i, len([t for t in transactions if i.issubset(t)])) for i in current) if l > tres]
        for itms, frequency in nxt:
            fout.write("&".join(map(str, itms)) + ";" + str(frequency) + "\n")
        current = [frozenset({n}.union(i)) for i,l in nxt for n in items if n > max(i)]
        print(current)
        #fout.flush()


