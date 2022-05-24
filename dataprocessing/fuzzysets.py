import seaborn as sns
from matplotlib import pyplot as plt
from skfuzzy.cluster import cmeans_predict
import numpy as np


class FuzzySet:
    def __call__(self, x):
        raise NotImplementedError

    def plot(self, canvas, label):
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

    def plot(self, canvas, label):
        if self.l0 == self.l:
            if fs.r0 == fs.r:
                sns.lineplot(x=[self.l, self.r], y=[1, 1], label=label)
            else:
                sns.lineplot(x=[self.l, self.r, self.r0], y=[1, 1, 0],
                             label=label)
        elif fs.r0 == fs.r:
            sns.lineplot(x=[self.l0, self.l, self.r], y=[0, 1, 1], label=label)
        else:
            sns.lineplot(x=[self.l0, self.l, self.r, self.r0], y=[0, 1, 1, 0],
                         label=label)


class OpenTrapezoidFuzzySet(FuzzySet):
    def __init__(self, r, r0):
        self.r0 = r0
        self.r = r

    def __call__(self, v):

        r = self.r
        r0 = self.r0

        if v > r0:
            return 0
        elif v > r:
            return (v - r0) / (r - r0)
        else:
            return 1

    def plot(self, canvas, label):
        if self.r0 == self.r:
            sns.lineplot(x=[0, self.r], y=[1, 1], label=label)
        else:
            sns.lineplot(x=[0, self.r, self.r0], y=[1, 1, 0],
                         label=label)

class Verum(TrapezoidFuzzySet):

    def __init__(self, l,r):
        super().__init__(l,l,r,r)

class RadialFuzzySet(FuzzySet):
    def __init__(self, centers, index):
        self.c = centers
        self.index = index

    def __call__(self, v):
        self._call_wrapped([v])

    def _call_wrapped(self, v):
        p = cmeans_predict(np.array([v]), self.c, 2, error=0.005, maxiter=10)[0]
        return p[self.index]

    def plot(self, canvas, label):
        x = np.arange(0,max(self.c)*1.5)
        y=self._call_wrapped(x)
        sns.lineplot(x=x, y=y, label=label)



female_centers = np.array(
    [[8.64393539], [26.83006829], [45.48100431], [56.32242189], [67.86115455]]
)
individual_analysed = np.array(
    [[104.48384197], [537.95792884], [1233.74699616], [3393.28183401], [14055.32421329]]
)

FUZZY_SETS = {
    "Mean age": {
        "child": OpenTrapezoidFuzzySet(14, 18),
        "adolecent": OpenTrapezoidFuzzySet(23, 27),
        "young adult": OpenTrapezoidFuzzySet(35, 40),
        "adult": OpenTrapezoidFuzzySet(60, 65),
        "elderly": OpenTrapezoidFuzzySet(100, 100),
    },
    "Combined follow up": {
        "one week": OpenTrapezoidFuzzySet(1, 2),
        "<= 1 month": OpenTrapezoidFuzzySet(3, 4),
        "<= 8 months": OpenTrapezoidFuzzySet(8 * 4.35, 9 * 4.35),
        "<= 15 month": OpenTrapezoidFuzzySet(15 * 4.35, 16 * 4.35),
        "<= 20 months": OpenTrapezoidFuzzySet(20 * 4.35, 21 * 4.35),
        "<= 28 months": OpenTrapezoidFuzzySet(28 * 4.35, 29 * 4.35),
        "> 29 months": Verum(0, 36 * 4.35),
    },
    "Mean number of times tobacco used": {
        "<= 5": OpenTrapezoidFuzzySet(5, 6),
        "<= 10": OpenTrapezoidFuzzySet(10, 11),
        "<= 15": OpenTrapezoidFuzzySet(15, 16),
        "<= 20": OpenTrapezoidFuzzySet(20, 21),
        "<= 25": OpenTrapezoidFuzzySet(25, 26),
        "> 26": Verum(0,26),
    },
    "Proportion identifying as female gender": {
        "None": OpenTrapezoidFuzzySet(0, 1),
        "<= 15": OpenTrapezoidFuzzySet(15, 20),
        "<= 35": OpenTrapezoidFuzzySet(35, 40),
        "<= 45": OpenTrapezoidFuzzySet(45, 50),
        "<= 60>": OpenTrapezoidFuzzySet(60, 65),
        "<= 99>": OpenTrapezoidFuzzySet(99, 100),
        "All": OpenTrapezoidFuzzySet(100, 100),
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
            fs.plot(sns, label)
        plt.savefig(f"/tmp/plots/{key}.png")
        plt.close()
