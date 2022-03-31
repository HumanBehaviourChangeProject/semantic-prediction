from fpgrowth_py import fpgrowth

with open("results/rules.txt", "r") as fin:
    with open("results/frequents.txt", "w") as fout:
        rulessets = []
        for line in fin:
            rules=[]
            for rule in line.split(";"):
                *conjunctions, weight = rule.split("&")
                conjunctions = [float(c) > 0.5 for c in conjunctions]
                rules.append((conjunctions, float(weight)))
            rulessets.append(rules)



        items = list(range(len(rulessets[0][0][0])))
        all_existing = set()
        transactions = [{i for i, v in enumerate(con) if v} for rule in rulessets for con, _ in rule]
        initial_item_sets = [frozenset({x}) for x in items]

        base_tres = 2000

        layer = 0
        tres = max(base_tres / (10 ** layer), 1)
        current = [i for i,l in ((i, len([t for t in transactions if i.issubset(t)] )) for i in initial_item_sets) if l > tres]
        while current:
            tres = 100
            nxt = [(i,l) for i,l in ((i, len([t for t in transactions if i.issubset(t)])) for i in current) if l > tres]
            for items, frequency in nxt:
                fout.write("&".join(map(str, items)) + ";" + str(frequency) + "\n")
            all_existing = all_existing.union(nxt)
            current = [frozenset({n}.union(i)) for i,l in nxt for n in items if n > max(i)]
            layer += 1
            fout.flush()


