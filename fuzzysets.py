import seaborn as sns
from matplotlib import pyplot as plt

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
          }Dear

if __name__ == "__main__":
    sns.set(rc={'figure.figsize': (20, 4)})
    for key, sets in FUZZY_SETS.items():
        for label, fs in sets.items():
            if fs.l0 == fs.l:
                sns.lineplot(x=[fs.l, fs.r, fs.r0], y=[1,1,0], label=label)
            elif fs.r0 == fs.r:
                sns.lineplot(x=[fs.l0, fs.l, fs.r], y=[0, 1, 1], label=label)
            else:
                sns.lineplot(x=[fs.l0, fs.l, fs.r, fs.r0], y=[0, 1, 1, 0], label=label)
        plt.savefig(f"/tmp/plots/{key}.png")
        plt.close()