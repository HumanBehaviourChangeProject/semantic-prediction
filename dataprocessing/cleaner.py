import csv
from thefuzz import process as fw_process, fuzz
import json
from abc import ABC, abstractmethod

_MAPPINGS = set()


def apply_diff(dataset, attr_map):
    with open("data/diff.json", "r") as fin:
        diff = json.load(fin)

    with open("data/deleted.txt", "r") as fin:
        for doc_id in fin:
            del dataset[doc_id.strip()]

    for doc_id, doc_changes in diff.items():
        for arm_id, arm_changes in doc_changes.items():
            ds = dataset[doc_id][arm_id]
            for key in arm_changes:
                num_key = attr_map.get(key, key)
                if arm_changes[key][0] == ds.get(num_key):
                    ds[num_key] = arm_changes[key][1]
                else:
                    print(
                        "Original value in diff does not match value in "
                        "dataset. Diff was not applied for this instance."
                        f"({arm_changes[key][0]} != {ds.get(num_key)}"
                    )

    return dataset


def load_countries_and_cities():
    with open("data/worldcities.csv", "r") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        city_dict = {line[1]: line[4] for line in reader}

    return set(city_dict.values()).union({"USA", "UK", "England"}), city_dict


class AttributeCleaner(ABC):
    def __init_subclass__(cls, **kwargs):
        _MAPPINGS.add(cls())

    @property
    def linked_attributes(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, ident, data):
        raise NotImplementedError


class RoundingCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6451791, 6080481, 6080485, 6080486, 6080512, 6080719)

    def __call__(self, ident, data):
        v = None
        if ident in data:
            for i in reversed(range(0, 3)):
                s = set(
                    round(float(x), i)
                    for x, _ in data[ident]
                    if _clean(x) not in ("no", "none") and is_number(x)
                )
                if len(s) == 1:
                    return {ident: s.pop()}
        if v:
            return {ident: v}
        else:
            return dict()


class PresenceCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (
            6451788,
            6823485,
            6452744,
            6452745,
            6452746,
            6452747,
            6452748,
            6452749,
            6452750,
            6452751,
            6452752,
            6452753,
            6452755,
            6452756,
            6452757,
            6452758,
            6452759,
            6452760,
            6452761,
            6452762,
            6452763,
            6452764,
            6452765,
            6452830,
            6452831,
            6452832,
            6452833,
            6452834,
            6452836,
            6452837,
            6452838,
            6452839,
            6452840,
            6452843,
            6452844,
            6452845,
            6452846,
            6452847,
            6452848,
            6452849,
            6452850,
            6452851,
            6452852,
            6452853,
            6452854,
            6452855,
            6452856,
            6452857,
            6452858,
            6452859,
            6452860,
            6452861,
            6452862,
            6452863,
            6452864,
            6452865,
            6452930,
            6452931,
            6452932,
            6452933,
            6452934,
            6452935,
            6452936,
            6452937,
            6452938,
            6452939,
            6452940,
            6452941,
            6452942,
            6452943,
            6452944,
            6452945,
            6452946,
            6452947,
            6452948,
            6452949,
            6452950,
            6452952,
            6452953,
            6452954,
            6452955,
            6452956,
            6452957,
            6452959,
            6452960,
            6452961,
            6452962,
            6452963,
            6452964,
            6452965,
            6452966,
            6452967,
            6452968,
            6452969,
            6452970,
            6452973,
            6452974,
            6452975,
            6452976,
            6452977,
            6452978,
            6452979,
            6452980,
            6452981,
            6452982,
            6452983,
            6452984,
            6452985,
            6452986,
            6452987,
            6080701,
            6080686,
            6080692,
            6080694,
            6080688,
            6830264,
        )

    def __call__(self, ident, data):
        x = data.get(ident, False)
        if x:
            v = 1
        else:
            v = 0
        return {ident: v}


def _clean(x):
    return x.replace(",", "").replace(";", "").replace("-", "").replace(" ", "").lower()


def _process_with_key_list(
    ident, keys, data, initial_dictionary=None, threshold=90, negative=False
):
    if initial_dictionary:
        d = dict(initial_dictionary)
    else:
        d = dict()
    for x, _ in data.get(ident, tuple()):
        for key, patterns in keys.items():
            match = fw_process.extract(x, patterns)
            if match[0][1] >= threshold:
                d[key] = 1 if not negative else 0
    return d


class MotivationalIntervewingCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6823487,)

    def __call__(self, ident, data):
        v = 0
        brief = 0
        for x, _ in data.get(ident, tuple()):
            match = fw_process.extract(x.lower(), ("brief advice", "ba"))
            if match[0][1] >= 80:
                brief = 1
            else:
                v = 1

        return {ident: v, "brief advise": brief}


class DigitalContentCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6080691,)

    def __call__(self, ident, data):
        v = 0
        text_message = 0
        for x, _ in data.get(ident, tuple()):
            match = fw_process.extract(x.lower(), ("text", "text message"))
            if match[0][1] >= 80:
                text_message = 1
            else:
                v = 1

        return {ident: v, "text messaging": text_message}


class DistanceCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6080687,)

    def __call__(self, ident, data):
        keys = ("phone", "call", "telephone", "quitline", "hotline")
        phone = 0
        for x, _ in data.get(ident, tuple()):
            match = fw_process.extract(x, keys)
            if match[0][1] >= 90:
                phone = 1
        return {"phone": phone}


class SomaticCleaner(AttributeCleaner):
    def __init__(self):
        self.any_as_presence = PresenceCleaner()

    @property
    def linked_attributes(self):
        return (6080693,)

    def __call__(self, ident, data):
        keys = {
            "gum": ["gum", "polacrilex"],
            "lozenge": ["lozenge"],
            "e_cigarette": ["ecig", "ecigarette"],
            "inhaler": ["inhaler", "inhal"],
            "placebo": ["placebo"],
            "varenicline": ["varenicline", "varen", "chantix", "champix"],
            "nasal_spray": ["nasal"],
            "rimonabant": ["rimonab"],
            "nrt": ["nicotine", "nrt"],
            "bupropion": ["bupropion"],
        }
        d = dict(
            gum=0, lozenge=0, e_cigarette=0, inhaler=0, placebo=0, nasal_spray=0, nrt=0
        )
        d.update(self.any_as_presence(ident, data))
        d = _process_with_key_list(ident, keys, data, initial_dictionary=d)
        patch = self.any_as_presence(6080694, data)[6080694]
        if d["gum"] or d["lozenge"] or d["e_cigarette"] or patch or d["inhaler"]:
            d["nrt"] = 1
        return d


class PillCleaner(AttributeCleaner):
    def __init__(self):
        self.any_as_presence = PresenceCleaner()

    @property
    def linked_attributes(self):
        return (6080695,)

    def __call__(self, ident, data):
        keys = {
            "bupropion": (
                ["bupropion"],
                True,
                any(
                    True
                    for x, _ in data.get(6080693, tuple())
                    if _clean(x) in ["bupropion"]
                ),
            ),
            "nortriptyline": (["nortript"], True, False),
            "varenicline": (
                ["varenicline", "varen", "chantix", "champix"],
                True,
                False,
            ),
        }
        d = self.any_as_presence(ident, data)
        for x, _ in data.get(ident, tuple()):
            for key, (patterns, ands, ors) in keys.items():
                match = fw_process.extract(x, patterns)
                if (match[0][1] > 90 or ors) and ands:
                    d[key] = 1
        return d


class HealthProfessionalCleaner(AttributeCleaner):
    def __init__(self):
        self.any_as_presence = PresenceCleaner()

    @property
    def linked_attributes(self):
        return (6080704,)

    def __call__(self, ident, data):
        keys = {
            "nurse": ["nurse"],
            "doctor": [
                "physician",
                "doctor",
                "physician",
                "cardiologist",
                "pediatrician",
                "general pract",
                "GP",
                "resident",
                "internal medicine",
            ],
        }
        d = self.any_as_presence(ident, data)
        d = _process_with_key_list(ident, keys, data, initial_dictionary=d)
        return d


class PsychologistCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6080706,)

    def __call__(self, ident, data):
        keys = {"pychologist": ["pychologist", "psychol"]}
        d = _process_with_key_list(ident, keys, data)
        return d


class PatientRoleCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6080508,)

    def __call__(self, ident, data):
        keys = {"aggregate patient role": ["patient"]}
        d = {"aggregate patient role": 0}
        d = _process_with_key_list(
            ident, keys, data, initial_dictionary=d, threshold=80
        )
        return d


class HealthCareFacilityCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6080629,)

    def __call__(self, ident, data):
        keys = {"healthcare facility": ["smok"]}
        d = _process_with_key_list(
            ident,
            keys,
            data,
            initial_dictionary={
                "healthcare facility": 1 if data.get(ident, tuple()) else 0
            },
            negative=True,
        )
        return d


def is_number(x):
    if x is None:
        return False
    try:
        float(x)
        return True
    except ValueError:
        return False


class PharmacologicalInterestCleaner(AttributeCleaner):
    def __init__(self):
        self.any_as_presence = PresenceCleaner()

    @property
    def linked_attributes(self):
        return (6830268,)

    def __call__(self, ident, data):
        return {
            ident: self.any_as_presence(ident, data).get(ident, 0)
            or self.any_as_presence(6830264, data).get(6830264, 0)
        }


class TimePointCleaner(AttributeCleaner):
    def __init__(self):
        self.use_rounded = RoundingCleaner()

    @property
    def linked_attributes(self):
        return (6451782,)

    def __call__(self, ident, data):
        v = data.get(6451782) or data.get(6451773)
        k = "combined followup"
        if v:
            return self.use_rounded(k, {k: v})
        else:
            return {k: v}


class CountryCleaner(AttributeCleaner):
    def __init__(self):
        self.countries, self.city_dict = load_countries_and_cities()

    @property
    def linked_attributes(self):
        return (6080518,)

    def __call__(self, ident, data):
        countries_values = data.get(ident, set())
        if countries_values:
            countries_values = [x for x, _ in countries_values]
            if len(countries_values) > 1:
                value = "multinational"
            else:
                match, quality = fw_process.extract(
                    countries_values[0], self.countries, scorer=fuzz.ratio
                )[0]
                if quality > 80:
                    value = match
                else:
                    value = countries_values[0]

        else:
            regions = data.get(6080519, set())
            value = [self.city_dict[x] for x, _ in regions if x in self.city_dict]
            if value:
                value = value[0]
            else:
                return dict()
        return {ident: value}


def clean_row(row):
    values = {
        "bupropion": 0,
        "varenicline": 0,
        "pychologist": 0,
        "doctor": 0,
        "nurse": 0,
    }

    for cleaner in _MAPPINGS:
        for attribute_id in cleaner.linked_attributes:
            mapped = cleaner(attribute_id, row)
            mapped_with_context = {k: mapped.get(k, 0) for k in mapped}
            values.update(mapped_with_context)
    return values
