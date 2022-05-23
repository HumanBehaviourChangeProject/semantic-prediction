import seaborn as sns
from matplotlib import pyplot as plt


class FuzzySet:
    def __init__(self, l0, l, r, r0):
        self.l0 = l0
        self.l = l
        self.r = r
        self.r0 = r0



    def contains(self, v):
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

    def at_least_in(self, v):
        if v < self.l0:
            return 0
        elif v < self.l:
            return (v - self.l0) / (self.l - self.l0)
        else:
            return 1

FUZZY_SETS = {
    "Mean age": {
        "child": FuzzySet(0, 0, 14, 18),
        "adolecent": FuzzySet(14, 18, 23, 27),
        "adult": FuzzySet(23, 27, 60, 65),
        "elderly": FuzzySet(60, 65, 100, 100),
    },
    "Individual-level analysed": {
        "<= 50": FuzzySet(0, 0, 50, 60),
        "60-100": FuzzySet(50, 60, 100, 120),
        "110-500": FuzzySet(100, 110, 500, 600),
        "600-1000": FuzzySet(500, 600, 1000, 1200),
        "1200-5000": FuzzySet(1000, 1200, 5000, 6000),
        "6k-10k": FuzzySet(5000, 6000, 10000, 12000),
        "12k-100k": FuzzySet(10000, 12000, 100000, 100000),
    },
    "Combined follow up": {
        "one week": FuzzySet(0, 0, 1, 2),
        "less than a month": FuzzySet(1, 2, 3, 4),
        "1-8 months": FuzzySet(3, 4, 8 * 4.35, 9 * 4.35),
        "8-15 month": FuzzySet(8 * 4.35, 9 * 4.35, 15 * 4.35, 16 * 4.35),
        "16-20 months": FuzzySet(15 * 4.35, 16 * 4.35, 20 * 4.35, 21 * 4.35),
        "21-28 months": FuzzySet(20 * 4.35, 21 * 4.35, 28 * 4.35, 29 * 4.35),
        "29 month+": FuzzySet(28 * 4.35, 29 * 4.35, 120 * 4.35, 120 * 4.35),
    },
    "Mean number of times tobacco used": {
        "<5": FuzzySet(0, 0, 5, 6),
        "6-10": FuzzySet(5, 6, 10, 11),
        "11-15": FuzzySet(10, 11, 15, 16),
        "16-20": FuzzySet(15, 16, 20, 21),
        "21-25": FuzzySet(20, 21, 25, 26),
        ">26": FuzzySet(25, 26, 100, 100),
    },
    "Proportion identifying as female gender": {
        "<10%": FuzzySet(0, 0, 5, 10),
        "10-20%": FuzzySet(5, 10, 20, 25),
        "25-40%": FuzzySet(20, 25, 40, 45),
        "45-55%": FuzzySet(40, 45, 55, 60),
        "60-75%": FuzzySet(55, 60, 75, 80),
        "80-90%": FuzzySet(75, 80, 90, 95),
        "95-100%": FuzzySet(90, 95, 100, 100),
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
