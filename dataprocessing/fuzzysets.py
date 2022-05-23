import seaborn as sns
from matplotlib import pyplot as plt
from skfuzzy.cluster import cmeans_predict
import numpy as np
class FuzzySet:
    def __call__(self, x):
        raise NotImplementedError

class TrapezoidFuzzySet(FuzzySet):
    def __init__(self, l0, l, r, r0):
        self.l0 = l0
        self.l = l
        self.r = r
        self.r0 = r0

    def __call__(self, v):
        if v < self.l0:
            return 0
        elif v < self.l:
            return (v - self.l0) / (self.l - self.l0)
        elif v < self.r:
            return 1
        elif v < self.r0:
            return (v - self.r0) / (self.r - self.r0)
        else:
            return 0

class OpenTrapezoidFuzzySet(FuzzySet):
    def __init__(self, r, r0):
        self.r0 = r0
        self.r = r

    def __call__(self, v):
        if v > self.r0:
            return 0
        elif v > self.r:
            return (v - self.r0) / (self.r - self.r0)
        else:
            return 1

class RadialFuzzySet(FuzzySet):
        def __init__(self, centers, index):
            self.c = centers
            self.index = index

        def __call__(self, v):
            p = cmeans_predict(np.array([[v]]), self.c, 2, error=0.005, maxiter=10)[0]
            return p[self.index,0]

female_centers = np.array([[ 8.64393539], [26.83006829], [45.48100431], [56.32242189], [67.86115455]])
individual_analysed = np.array([[104.48384197], [  537.95792884], [ 1233.74699616], [ 3393.28183401], [14055.32421329]])

FUZZY_SETS = {
    "Mean age": {
        "child": OpenTrapezoidFuzzySet(14, 18),
        "adolecent": OpenTrapezoidFuzzySet(23, 27),
        "adult": OpenTrapezoidFuzzySet(60, 65),
        "elderly": OpenTrapezoidFuzzySet(100, 100),
    },
    "Combined follow up": {
        "one week": OpenTrapezoidFuzzySet(1, 2),
        "less than a month": OpenTrapezoidFuzzySet(3, 4),
        "1-8 months": OpenTrapezoidFuzzySet(8 * 4.35, 9 * 4.35),
        "8-15 month": OpenTrapezoidFuzzySet(15 * 4.35, 16 * 4.35),
        "16-20 months": OpenTrapezoidFuzzySet(20 * 4.35, 21 * 4.35),
        "21-28 months": OpenTrapezoidFuzzySet(28 * 4.35, 29 * 4.35),
        "29 month+": OpenTrapezoidFuzzySet(120 * 4.35, 120 * 4.35),
    },
    "Mean number of times tobacco used": {
        "<5": OpenTrapezoidFuzzySet(5, 6),
        "6-10": OpenTrapezoidFuzzySet(10, 11),
        "11-15": OpenTrapezoidFuzzySet(15, 16),
        "16-20": OpenTrapezoidFuzzySet(20, 21),
        "21-25": OpenTrapezoidFuzzySet(25, 26),
        ">26": OpenTrapezoidFuzzySet(100, 100),
    },
    "Proportion identifying as female gender": {
        "None": OpenTrapezoidFuzzySet(1, 1),
        "~10": RadialFuzzySet(female_centers, 0),
        "~25": RadialFuzzySet(female_centers, 1),
        "~45": RadialFuzzySet(female_centers, 2),
        "~55": RadialFuzzySet(female_centers, 3),
        "~70": RadialFuzzySet(female_centers, 4),
        "All": OpenTrapezoidFuzzySet(99, 99),
    },
"Individual-level analysed": {
        "~100": RadialFuzzySet(individual_analysed, 0),
        "~500": RadialFuzzySet(individual_analysed, 1),
        "~1000": RadialFuzzySet(individual_analysed, 2),
        "~3000": RadialFuzzySet(individual_analysed, 3),
        "~15000": RadialFuzzySet(individual_analysed, 4),
    },
}

if __name__ == "__main__":
    sns.set(rc={"figure.figsize": (20, 4)})
    for key, sets in FUZZY_SETS.items():
        for label, fs in sets.items():
            if fs.l0 == fs.l:
                sns.lineplot(x=[fs.l, fs.r, fs.r0], y=[1, 1, 0], label=label)
            elif fs.r0 == fs.r:
                sns.lineplot(x=[fs.l0, fs.l, fs.r], y=[0, 1, 1], label=label)
            else:
                sns.lineplot(x=[fs.l0, fs.l, fs.r, fs.r0], y=[0, 1, 1, 0], label=label)
        plt.savefig(f"/tmp/plots/{key}.png")
        plt.close()
